"""
Expert review of drive-video segments, the review log, and "model vs expert" metrics.

The expert UI (expert_ui/index.html) reviews two kinds of rows:

    citizen   RoadSide uploads. Rows live in Supabase; the page writes them there
              itself (anon key + RLS, migration 003) and then logs the review here.
    video     drive-video segments (app/video_store.py). Rows live in the local
              PostgreSQL, which the browser cannot reach, so the page lists and
              saves them through app/expert_routes.py, which calls this module.

Every save, from either source, appends one row to research.expert_review_log
(migration 006): the model's answer read server-side at save time, the expert's
answer, and compare() of the two. Nothing in the log is ever updated or
deleted; a later "Update review" is a new row, and metrics use the latest row
per assessment. scripts/expert_review_metrics.py and GET /expert-api/metrics
both call metrics() below, so the paper and the UI show the same numbers.

Definitions (also printed with the metrics):
  The model says "distress" when stage1_label == "Distress"; the expert says
  "distress" unless their types are exactly ["Normal"].

  verdict   right   both Normal, or both distress with the same set of types
            close   both distress, types overlap but differ
            wrong   one says Normal and the other distress, or no type in common
  severity  (only when both say distress) exact; close = one step apart on
            Low < Medium < High; wrong = two steps apart
  first     the model's first-listed type is among the expert's types

Sampling matters: experts mostly open the "needs review" queue, which the gate
filled with the model's uncertain answers, so accuracy there is not the model's
accuracy. The "random" queue is a fixed shuffle of all answered segments and is
the unbiased estimate. Every log row records its queue, and metrics() reports
each queue separately.

Synchronous psycopg (called from worker threads by the routes).
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.gis_core import DISTRESS_LABELS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MIGRATION = PROJECT_ROOT / "migrations" / "006_expert_review_log.sql"
MIGRATION_EVENTS = PROJECT_ROOT / "migrations" / "008_review_idempotency_events.sql"
_OP_RE = re.compile(r"^[A-Za-z0-9_-]{16,64}$")


def clean_op_id(v) -> Optional[str]:
    """The page's id for one save (None from pages older than migration 008)."""
    if v is None or v == "":
        return None
    s = str(v).strip()
    if not _OP_RE.match(s):
        raise ReviewError("op_id must be 16-64 letters, digits, '-' or '_'")
    return s

NORMAL = "Normal"
IRC_TYPES: tuple[str, ...] = DISTRESS_LABELS            # the 18 vision-only IRC:82 types
EXPERT_TYPES: tuple[str, ...] = IRC_TYPES + ("Patching",)  # + the IRC condition indicator
SEVERITY_RANK = {"None": 0, "Low": 1, "Medium": 2, "High": 3}
SOURCES = ("video", "citizen")

# Video queues: what each list shows (all: answered segments only, i.e. with an assessment).
VIDEO_QUEUES = {
    "needs_review": "a.status = 'expert_review' AND NOT COALESCE(a.expert_reviewed, false)",
    "auto": "a.status = 'classified' AND NOT COALESCE(a.expert_reviewed, false)",
    "random": "NOT COALESCE(a.expert_reviewed, false)",
    "reviewed": "COALESCE(a.expert_reviewed, false)",
}
CITIZEN_QUEUES = ("citizen_pending", "citizen_reviewed")
ALL_QUEUES = tuple(VIDEO_QUEUES) + CITIZEN_QUEUES
HEATMAP_RATINGS = ("on", "partly", "off")

# Older names in the citizen tab's type list (custom_distress_types, TYPES_DEFAULT) -> IRC:82.
# Block cracking is the block form of Transverse Cracking in IRC:82 §7.3.5.1; a patch is the
# Patching condition indicator. A name with no IRC equivalent is kept as is (counts as a mismatch).
LEGACY_TYPES = {
    "longitudinal crack (d00)": "Longitudinal Cracking", "longitudinal crack": "Longitudinal Cracking",
    "transverse crack (d10)": "Transverse Cracking", "transverse crack": "Transverse Cracking",
    "alligator crack (d20)": "Alligator Cracking", "alligator crack": "Alligator Cracking",
    "pothole (d40)": "Potholes", "pothole": "Potholes",
    "block crack (d43)": "Transverse Cracking", "block crack": "Transverse Cracking",
    "inlaid patch (d44)": "Patching", "patch": "Patching",
    "raveling": "Ravelling", "edge crack": "Edge Cracking", "depression": "Shallow Depression",
}
_CANON = {t.lower(): t for t in EXPERT_TYPES + (NORMAL,)}

MAX_NOTES = 2000
MAX_NAME = 60
MAX_TYPES = 12
_schema_lock = threading.Lock()
_schema_ready = False


class ReviewError(ValueError):
    """Bad input from the page (the API answers 400 with this message)."""


class NotFound(LookupError):
    """No such row (404)."""


# ============================================================
# Labels and comparison (pure)
# ============================================================

def canonical(name: str) -> str:
    """IRC:82 spelling of a type, mapping the citizen tab's older names; unknown names unchanged."""
    s = str(name).strip()
    return _CANON.get(s.lower()) or LEGACY_TYPES.get(s.lower()) or s


def type_set(types: Iterable[str]) -> frozenset:
    """Canonical distress types, without Normal / Unknown / blanks."""
    out = set()
    for t in types or []:
        c = canonical(t)
        if c and c not in (NORMAL, "Unknown"):
            out.add(c)
    return frozenset(out)


def vlm_says_distress(vlm: Mapping) -> bool:
    if vlm.get("stage1_label") in ("Distress", "Normal"):
        return vlm["stage1_label"] == "Distress"
    return bool(type_set(vlm.get("distress_types") or []))


def expert_says_distress(types: Iterable[str]) -> bool:
    return bool(type_set(types))


def compare(vlm: Mapping, expert_types: Iterable[str], expert_severity: str) -> dict:
    """The verdicts for one review (see the module docstring)."""
    m_d, e_d = vlm_says_distress(vlm), expert_says_distress(expert_types)
    m_t = type_set(vlm.get("distress_types") or []) if m_d else frozenset()
    e_t = type_set(expert_types)
    out: dict[str, Any] = {"vlm_distress": m_d, "expert_distress": e_d,
                           "stage1": "agree" if m_d == e_d else "disagree"}
    if not m_d and not e_d:
        out.update(verdict="right", jaccard=1.0)
    elif m_d != e_d:
        out.update(verdict="wrong", jaccard=0.0)
    else:
        inter, union = m_t & e_t, m_t | e_t
        jac = len(inter) / len(union) if union else 1.0
        out.update(verdict="right" if m_t == e_t else "close" if inter else "wrong", jaccard=round(jac, 4))
    if m_d and e_d:
        first = (vlm.get("distress_types") or [None])[0]
        out["first_label"] = "right" if first is not None and canonical(first) in e_t else "wrong"
        ms, es = SEVERITY_RANK.get(str(vlm.get("severity"))), SEVERITY_RANK.get(expert_severity)
        if ms and es:
            d = abs(ms - es)
            out["severity"] = "exact" if d == 0 else "close" if d == 1 else "wrong"
            out["severity_diff"] = ms - es              # >0: the model said more severe
    out["missed_types"] = sorted(e_t - m_t)
    out["extra_types"] = sorted(m_t - e_t)
    return out


def vlm_snapshot(row: Mapping) -> dict:
    """The model's answer as stored on the assessment (what the expert is judging)."""
    def f(x):
        return None if x is None else round(float(x), 6)
    return {"status": row.get("status"), "stage1_label": row.get("stage1_label"),
            "stage1_confidence": f(row.get("stage1_confidence")),
            "distress_types": list(row.get("distress_types") or []),
            "severity": row.get("severity"), "stage2_confidence": f(row.get("stage2_confidence")),
            "needs_expert_review": row.get("needs_expert_review"),
            "model_version": row.get("model_version"),
            "processed_at": str(row["processed_at"]) if row.get("processed_at") else None}


# ============================================================
# Input validation
# ============================================================

def _text(v, name: str, lo: int, hi: int) -> str:
    s = str(v or "").strip()
    if not lo <= len(s) <= hi:
        raise ReviewError(f"{name} must be {lo}-{hi} characters")
    return s


def clean_review(p: Mapping, *, source: str, strict_types: bool) -> dict:
    """Validated review fields from the page. strict_types: only IRC types / Normal (video)."""
    if not isinstance(p, Mapping):
        raise ReviewError("body must be a JSON object")
    reviewer = _text(p.get("reviewer_name"), "reviewer_name", 2, MAX_NAME)
    raw = p.get("types")
    if not isinstance(raw, list) or not raw or len(raw) > MAX_TYPES:
        raise ReviewError(f"types must be a list of 1-{MAX_TYPES} names")
    types = []
    for t in raw:
        t = _text(t, "each type", 1, MAX_NAME)
        c = canonical(t)
        if strict_types and c not in EXPERT_TYPES and c != NORMAL:
            raise ReviewError(f"unknown type {t!r}: use an IRC:82 type or Normal")
        if c not in types:
            types.append(c if strict_types else t)
    is_normal = [canonical(t) for t in types] == [NORMAL]
    if NORMAL in (canonical(t) for t in types) and not is_normal:
        raise ReviewError("Normal cannot be combined with distress types")
    sev = str(p.get("severity") or ("None" if is_normal else "")).strip().capitalize()
    if sev not in SEVERITY_RANK:
        raise ReviewError("severity must be None, Low, Medium or High")
    if is_normal:
        sev = "None"
    elif sev == "None":
        raise ReviewError("a distress needs a severity (Low, Medium or High)")
    notes = str(p.get("notes") or "")[:MAX_NOTES]
    queue = str(p.get("queue") or "")
    allowed = VIDEO_QUEUES if source == "video" else CITIZEN_QUEUES
    if queue not in allowed:
        raise ReviewError(f"queue must be one of {sorted(allowed)}")
    hm = p.get("heatmap") or {}
    if not isinstance(hm, Mapping):
        raise ReviewError("heatmap must be an object")
    hm_types = []
    for t in (hm.get("types") or [])[:MAX_TYPES]:
        if isinstance(t, str) and canonical(t) not in hm_types:
            hm_types.append(canonical(t))
    raw_r = hm.get("ratings") or {}
    if not isinstance(raw_r, Mapping):
        raise ReviewError("heatmap.ratings must be an object {type: rating}")
    ratings = {}
    for t, r in list(raw_r.items())[:MAX_TYPES]:
        if r not in HEATMAP_RATINGS:
            raise ReviewError(f"heatmap ratings must be one of {HEATMAP_RATINGS}")
        if canonical(t) not in hm_types:
            raise ReviewError("a heatmap rating needs that type's overlay to have been viewed")
        ratings[canonical(t)] = r
    heatmap = {"viewed": bool(hm_types), "types": hm_types, "ratings": ratings}
    ui = p.get("ui") or {}
    ms = ui.get("ms_since_first_interaction") if isinstance(ui, Mapping) else None
    ui_out = {}
    if isinstance(ms, (int, float)) and math.isfinite(ms) and 0 <= ms <= 7 * 24 * 3600 * 1000:
        ui_out["ms_since_first_interaction"] = int(ms)
    return {"reviewer_name": reviewer, "types": types, "severity": sev, "notes": notes, "queue": queue,
            "heatmap": heatmap, "ui": ui_out}


# ============================================================
# Schema
# ============================================================

def ensure_schema(conn) -> None:
    """Create research.expert_review_log if missing (migration 006, idempotent)."""
    global _schema_ready
    if _schema_ready:
        return
    with _schema_lock:
        if _schema_ready:
            return
        from scripts.setup_local_db import statements_without_outer_transaction
        if conn.execute("SELECT to_regclass('research.expert_review_log')").fetchone()[0] is None:
            body = statements_without_outer_transaction(MIGRATION.read_text(encoding="utf-8-sig"))
            with conn.transaction():
                conn.execute(body)
        # migration 008 (op_id + research.expert_events); its statements are all IF NOT EXISTS
        if conn.execute("SELECT to_regclass('research.expert_events')").fetchone()[0] is None:
            body = statements_without_outer_transaction(MIGRATION_EVENTS.read_text(encoding="utf-8-sig"))
            with conn.transaction():
                conn.execute(body)
        _schema_ready = True


# ============================================================
# Video: listing and saving
# ============================================================

_LIST_COLS = """
    v.segment_id, v.session_id, v.k, v.status AS segment_status, v.centre_lat, v.centre_lon,
    v.chainage_start_km, v.s0_m, v.utc,
    (a.raw_response -> 'vlm_response' -> 'stage1_safety_net' ->> 'flagged') = 'true' AS safety_net_flagged,
    a.raw_response -> 'vlm_response' -> 'stage1_safety_net' ->> 'strongest_type' AS safety_net_type,
    a.id AS assessment_id, a.status, a.stage1_label, a.stage1_confidence, a.distress_types, a.severity,
    a.description, a.stage2_confidence, a.needs_expert_review, a.expert_reviewed,
    a.expert_corrected_types, a.expert_corrected_severity, a.expert_notes, a.reviewer_name, a.reviewed_at,
    a.model_version, a.processed_at
"""


def _random_key(segment_id: str) -> str:
    return hashlib.sha256(("expert-review-random-v1:" + segment_id).encode()).hexdigest()


def list_video(conn, queue: str, *, session: Optional[str] = None, label: Optional[str] = None,
               limit: int = 24, offset: int = 0) -> dict:
    """One page of answered segments in a queue, plus the size of every queue."""
    if queue not in VIDEO_QUEUES:
        raise ReviewError(f"queue must be one of {list(VIDEO_QUEUES)}")
    limit, offset = max(1, min(100, int(limit))), max(0, int(offset))
    where, params = ["v.assessment_id IS NOT NULL", VIDEO_QUEUES[queue]], []
    if session:
        where.append("v.session_id = %s")
        params.append(session)
    if label:
        if canonical(label) == NORMAL:
            where.append("a.stage1_label = 'Normal'")
        else:
            where.append("a.distress_types ? %s")
            params.append(canonical(label))
    w = " AND ".join(where)
    order = {"needs_review": "v.session_id DESC, v.k", "auto": "v.session_id DESC, v.k",
             "random": "md5('expert-review-random-v1:' || v.segment_id)",
             "reviewed": "a.reviewed_at DESC NULLS LAST, v.segment_id"}[queue]
    cur = conn.cursor(row_factory=dict_row)
    rows = cur.execute(f"SELECT {_LIST_COLS} FROM video_segments v JOIN assessments a ON a.id = v.assessment_id "
                       f"WHERE {w} ORDER BY {order} LIMIT %s OFFSET %s", (*params, limit, offset)).fetchall()
    total = cur.execute(f"SELECT COUNT(*) AS n FROM video_segments v JOIN assessments a ON a.id = v.assessment_id "
                        f"WHERE {w}", params).fetchone()["n"]
    base = ["v.assessment_id IS NOT NULL"] + (["v.session_id = %s"] if session else [])
    bparams = [session] if session else []
    counts = {}
    for q, cond in VIDEO_QUEUES.items():
        counts[q] = cur.execute(f"SELECT COUNT(*) AS n FROM video_segments v JOIN assessments a "
                                f"ON a.id = v.assessment_id WHERE {' AND '.join(base + [cond])}",
                                bparams).fetchone()["n"]
    items = []
    for r in rows:
        r = dict(r)
        r["assessment_id"] = str(r["assessment_id"])
        for k in ("utc", "reviewed_at", "processed_at"):
            r[k] = r[k].isoformat() if r.get(k) is not None else None
        r["image"] = f"/gis/image/{r['segment_id']}"
        items.append(r)
    return {"queue": queue, "total": total, "limit": limit, "offset": offset, "counts": counts, "items": items}


def video_sessions(conn) -> list[dict]:
    """Drives with answered segments (for the page's drive filter)."""
    cur = conn.cursor(row_factory=dict_row)
    rows = cur.execute(
        "SELECT v.session_id, COUNT(*) AS answered, "
        "COUNT(*) FILTER (WHERE a.status = 'expert_review' AND NOT COALESCE(a.expert_reviewed, false)) AS needs_review, "
        "COUNT(*) FILTER (WHERE COALESCE(a.expert_reviewed, false)) AS reviewed, MIN(v.utc) AS start_utc "
        "FROM video_segments v JOIN assessments a ON a.id = v.assessment_id "
        "GROUP BY v.session_id ORDER BY v.session_id DESC").fetchall()
    return [{**r, "start_utc": r["start_utc"].isoformat() if r["start_utc"] else None} for r in rows]


def _logged(conn, op_id: Optional[str]) -> Optional[dict]:
    """The log row an earlier arrival of the same save wrote, or None."""
    if not op_id:
        return None
    return conn.cursor(row_factory=dict_row).execute(
        "SELECT id, created_at, source, assessment_id, segment_id, action, expert_types, expert_severity, "
        "comparison FROM research.expert_review_log WHERE op_id = %s", (op_id,)).fetchone()


def _replay(prev: Mapping) -> dict:
    return {"log_id": prev["id"], "logged_at": prev["created_at"].isoformat(), "comparison": prev["comparison"],
            "replayed": True}


def _insert_log(conn, *, source, assessment_id, segment_id, session_id, review, vlm, action,
                op_id: Optional[str] = None) -> dict:
    """Append one log row. With an op_id the insert is idempotent: a second arrival of the same
    save returns the first one's row (replayed=True) instead of logging the review twice."""
    cmp = compare(vlm, review["types"], review["severity"])
    row = conn.cursor(row_factory=dict_row).execute(
        "INSERT INTO research.expert_review_log (source, assessment_id, segment_id, session_id, queue, action, "
        "reviewer_name, vlm, expert_types, expert_severity, expert_notes, comparison, heatmap, ui, op_id) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
        "ON CONFLICT (op_id) WHERE op_id IS NOT NULL DO NOTHING RETURNING id, created_at",
        (source, assessment_id, segment_id, session_id, review["queue"], action, review["reviewer_name"],
         Jsonb(vlm), Jsonb(review["types"]), review["severity"], review["notes"], Jsonb(cmp),
         Jsonb(review["heatmap"]), Jsonb(review["ui"]), op_id)).fetchone()
    if row is None:                                   # same op_id already logged
        return _replay(_logged(conn, op_id))
    return {"log_id": row["id"], "logged_at": row["created_at"].isoformat(), "comparison": cmp}


def save_video_review(conn, segment_id: str, payload: Mapping) -> dict:
    """Write the expert's answer onto the segment's assessment and log it, in one transaction.

    needs_expert_review and status are left as the pipeline set them: they are the gate's
    decision, which the metrics compare against. Returns the comparison and log id."""
    ensure_schema(conn)
    review = clean_review(payload, source="video", strict_types=True)
    with conn.transaction():
        cur = conn.cursor(row_factory=dict_row)
        row = cur.execute(
            "SELECT v.segment_id, v.session_id, a.* FROM video_segments v JOIN assessments a "
            "ON a.id = v.assessment_id WHERE v.segment_id = %s FOR UPDATE OF a", (segment_id,)).fetchone()
        if row is None:
            raise NotFound(f"no answered segment {segment_id!r}")
        vlm = vlm_snapshot(row)
        action = "update" if row.get("expert_reviewed") else "confirm"
        cur.execute(
            "UPDATE assessments SET expert_reviewed = true, expert_corrected_types = %s, "
            "expert_corrected_severity = %s, expert_notes = %s, reviewer_name = %s, reviewed_at = NOW() "
            "WHERE id = %s", (Jsonb(review["types"]), review["severity"], review["notes"],
                               review["reviewer_name"], row["id"]))
        out = _insert_log(conn, source="video", assessment_id=str(row["id"]), segment_id=row["segment_id"],
                          session_id=row["session_id"], review=review, vlm=vlm, action=action)
    return {"segment_id": segment_id, "action": action, "expert_types": review["types"],
            "expert_severity": review["severity"], **out}


def log_citizen_review(conn, assessment_id: str, payload: Mapping, row: Mapping) -> dict:
    """Log a citizen review the page already saved to Supabase. row: the assessment as the
    server read it from Supabase after the save (its status still holds the gate's decision;
    the model's fields are never changed by a review)."""
    ensure_schema(conn)
    review = clean_review(payload, source="citizen", strict_types=False)
    op_id = clean_op_id(payload.get("op_id"))
    vlm = vlm_snapshot(row)
    vlm["needs_expert_review"] = row.get("status") == "expert_review"   # the page clears the flag on save
    with conn.transaction():
        prev = _logged(conn, op_id)
        if prev is not None:
            return {"assessment_id": assessment_id, "action": prev["action"], **_replay(prev)}
        prior = conn.execute("SELECT 1 FROM research.expert_review_log WHERE source = 'citizen' "
                             "AND assessment_id = %s LIMIT 1", (assessment_id,)).fetchone()
        action = "update" if prior else "confirm"
        out = _insert_log(conn, source="citizen", assessment_id=assessment_id, segment_id=None, session_id=None,
                          review=review, vlm=vlm, action=action, op_id=op_id)
    return {"assessment_id": assessment_id, "action": action, **out}


# ============================================================
# Metrics
# ============================================================

def wilson(k: int, n: int, z: float = 1.959964) -> Optional[list]:
    """95% Wilson score interval for k of n, or None when n == 0."""
    if n <= 0:
        return None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def _share(k: int, n: int) -> dict:
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "ci95": wilson(k, n)}


def latest_reviews(conn, source: Optional[str] = None) -> list[dict]:
    """The latest log row per (source, assessment), oldest first, with sample_queue = the queue
    of that item's FIRST review (how it was sampled; a later edit from the Reviewed list must
    not move it out of the random sample) and n_reviews."""
    ensure_schema(conn)
    where, params = "", ()
    if source:
        where, params = "WHERE source = %s", (source,)
    rows = conn.cursor(row_factory=dict_row).execute(
        f"SELECT * FROM research.expert_review_log {where} ORDER BY created_at, id", params).fetchall()
    first: dict = {}
    last: dict = {}
    count: Counter = Counter()
    for r in rows:
        k = (r["source"], r["assessment_id"])
        first.setdefault(k, r["queue"])
        last[k] = r
        count[k] += 1
    out = []
    for k, r in last.items():
        out.append({**r, "sample_queue": first[k], "n_reviews": count[k]})
    out.sort(key=lambda r: (r["created_at"], r["id"]))
    return out


def summarize(rows: list[Mapping]) -> dict:
    """Model-vs-expert numbers for a set of latest reviews (pure; rows as from latest_reviews)."""
    n = len(rows)
    verdict = Counter(r["comparison"]["verdict"] for r in rows)
    s1 = Counter(r["comparison"]["stage1"] for r in rows)
    # Stage 1 as a detector, the expert as truth
    tp = sum(1 for r in rows if r["comparison"]["vlm_distress"] and r["comparison"]["expert_distress"])
    fp = sum(1 for r in rows if r["comparison"]["vlm_distress"] and not r["comparison"]["expert_distress"])
    fn = sum(1 for r in rows if not r["comparison"]["vlm_distress"] and r["comparison"]["expert_distress"])
    tn = n - tp - fp - fn
    both = [r for r in rows if r["comparison"]["vlm_distress"] and r["comparison"]["expert_distress"]]
    sev = Counter(r["comparison"].get("severity") for r in both if r["comparison"].get("severity"))
    sev_n = sum(sev.values())
    first = Counter(r["comparison"].get("first_label") for r in both)
    # Per type (multi-label, over every review; the expert's set is the truth)
    per_type: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        m = type_set(r["vlm"].get("distress_types") or []) if r["comparison"]["vlm_distress"] else frozenset()
        e = type_set(r["expert_types"])
        for t in m | e:
            per_type[t]["tp" if t in m and t in e else "fp" if t in m else "fn"] += 1
    types_out = {}
    for t, c in sorted(per_type.items(), key=lambda kv: -(kv[1]["tp"] + kv[1]["fn"])):
        p_n, r_n = c["tp"] + c["fp"], c["tp"] + c["fn"]
        types_out[t] = {"tp": c["tp"], "fp": c["fp"], "fn": c["fn"],
                        "precision": _share(c["tp"], p_n), "recall": _share(c["tp"], r_n)}
    # Gate: is the model wrong more often on rows it sent to review?
    gate = {}
    for name, flagged in (("flagged", True), ("auto_classified", False)):
        sub = [r for r in rows if bool(r["vlm"].get("needs_expert_review")) == flagged]
        gate[name] = {"n": len(sub), "wrong": _share(sum(1 for r in sub if r["comparison"]["verdict"] == "wrong"),
                                                     len(sub)),
                      "right": _share(sum(1 for r in sub if r["comparison"]["verdict"] == "right"), len(sub))}
    # The "where the model looked" overlay
    # Ratings are per (review, type): "is the ring on the damage?" for each overlay the expert opened
    viewed = [r for r in rows if (r.get("heatmap") or {}).get("viewed")]
    ratings: Counter = Counter()
    hm_by_type: dict[str, Counter] = defaultdict(Counter)
    for r in viewed:
        for t, v in ((r["heatmap"] or {}).get("ratings") or {}).items():
            ratings[v] += 1
            hm_by_type[t][v] += 1
    rated = sum(ratings[k] for k in HEATMAP_RATINGS)
    times = sorted(r["ui"]["ms_since_first_interaction"] for r in rows
                   if isinstance((r.get("ui") or {}).get("ms_since_first_interaction"), int))

    def median(xs):
        return None if not xs else (xs[len(xs) // 2] if len(xs) % 2 else (xs[len(xs) // 2 - 1] + xs[len(xs) // 2]) / 2)

    t_view = sorted(r["ui"]["ms_since_first_interaction"] for r in viewed
                    if isinstance((r.get("ui") or {}).get("ms_since_first_interaction"), int))
    t_noview = sorted(r["ui"]["ms_since_first_interaction"] for r in rows
                      if not (r.get("heatmap") or {}).get("viewed")
                      and isinstance((r.get("ui") or {}).get("ms_since_first_interaction"), int))
    return {
        "n": n,
        "verdict": {k: _share(verdict[k], n) for k in ("right", "close", "wrong")},
        "stage1": {"agree": _share(s1["agree"], n), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                   "recall": _share(tp, tp + fn), "precision": _share(tp, tp + fp),
                   "specificity": _share(tn, tn + fp)},
        "both_distress": {"n": len(both),
                          "first_label_right": _share(first["right"], len(both)),
                          "types_exact": _share(sum(1 for r in both if r["comparison"]["verdict"] == "right"), len(both)),
                          "mean_jaccard": round(sum(r["comparison"]["jaccard"] for r in both) / len(both), 4)
                          if both else None,
                          "severity": {k: _share(sev[k], sev_n) for k in ("exact", "close", "wrong")}},
        "per_type": types_out,
        "gate": gate,
        "heatmap": {"viewed": _share(len(viewed), n), "rated_overlays": rated,
                    "rating": {k: _share(ratings[k], rated) for k in HEATMAP_RATINGS},
                    "by_type": {t: {**{k: c[k] for k in HEATMAP_RATINGS},
                                    "on_rate": _share(c["on"], sum(c[k] for k in HEATMAP_RATINGS))}
                                for t, c in sorted(hm_by_type.items())}},
        "time_ms": {"median_all": median(times), "n": len(times),
                    "median_with_heatmap": median(t_view), "median_without_heatmap": median(t_noview)},
    }


def metrics(conn) -> dict:
    """Everything the Metrics tab and scripts/expert_review_metrics.py show."""
    rows = latest_reviews(conn)
    log_rows = conn.execute("SELECT COUNT(*) FROM research.expert_review_log").fetchone()[0]
    reviewers = Counter(r["reviewer_name"] for r in rows)
    out = {"definitions": {
        "verdict": "right: both Normal, or both distress with the same set of types; close: both distress, types "
                   "overlap but differ; wrong: Normal vs distress, or no type in common",
        "severity": "only when both say distress: exact, close (one step on Low<Medium<High), wrong (two steps)",
        "first_label": "the model's first-listed type is among the expert's types",
        "sampling": "the random queue is the unbiased estimate; needs_review holds the gate's uncertain answers",
        "truth": "the expert's latest review per item; earlier reviews stay in the log"},
        "log_rows": log_rows, "reviewed_items": len(rows), "reviewers": dict(reviewers),
        "by_source": {}, "by_queue": {}}
    for s in SOURCES:
        sub = [r for r in rows if r["source"] == s]
        if sub:
            out["by_source"][s] = summarize(sub)
    for q in ALL_QUEUES:
        sub = [r for r in rows if r.get("sample_queue", r["queue"]) == q]
        if sub:
            out["by_queue"][q] = summarize(sub)
    out["all"] = summarize(rows) if rows else summarize([])
    return out


# ============================================================
# Claims: several experts at once (migration 007)
# ============================================================
# A claim is a lease on one item. While it is live only the browser session holding it is shown
# the item in a review list and may save it. The page renews its leases every minute (heartbeat)
# and releases them on save, skip and unload; an expired lease is free.
#
# Handing out is one INSERT ... ON CONFLICT (source, item_id) DO UPDATE ... WHERE the old lease
# expired or is this session's, RETURNING the rows it got: whatever two sessions race for, each
# item ends up with exactly one live holder. A transaction-scoped advisory lock per source also
# serialises the hand-out queries, so they never deadlock or hand a session fewer items than
# exist. Saves lock the assessment row and the claim row, and refuse (Conflict, HTTP 409) when
# another live session holds the item or the review changed since the expert loaded it.

MIGRATION_CLAIMS = PROJECT_ROOT / "migrations" / "007_review_claims.sql"
CLAIM_TTL_S = 600                     # a lease lasts 10 minutes; the page renews it every minute
MAX_BATCH = 30                        # items per hand-out
MAX_CLAIMS_PER_SESSION = 60           # live leases one browser may hold
MAX_CANDIDATES = 200
CLAIM_QUEUES_VIDEO = ("needs_review", "auto", "random")
REVIEW_QUEUES = CLAIM_QUEUES_VIDEO + ("citizen_pending",)    # skipped items are not handed back
_ADVISORY = {"video": 0x6578_7065_7274_0001, "citizen": 0x6578_7065_7274_0002}
_SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{16,64}$")
_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._-]{1,200}$")
_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_VIDEO_ORDER = {"needs_review": "v.session_id DESC, v.k", "auto": "v.session_id DESC, v.k",
                "random": "md5('expert-review-random-v1:' || v.segment_id)",
                "reviewed": "a.reviewed_at DESC NULLS LAST, v.segment_id"}
_claims_schema_ready = False


class Conflict(RuntimeError):
    """Another expert holds the item, or the review changed since it was loaded (HTTP 409)."""

    def __init__(self, message: str, **info):
        super().__init__(message)
        self.info = {"message": message, **{k: v for k, v in info.items() if v is not None}}


def ensure_claims_schema(conn) -> None:
    """Create research.review_claims / review_skips if missing (migration 007, idempotent)."""
    global _claims_schema_ready
    ensure_schema(conn)
    if _claims_schema_ready:
        return
    with _schema_lock:
        if _claims_schema_ready:
            return
        if conn.execute("SELECT to_regclass('research.review_claims')").fetchone()[0] is None:
            from scripts.setup_local_db import statements_without_outer_transaction
            body = statements_without_outer_transaction(MIGRATION_CLAIMS.read_text(encoding="utf-8-sig"))
            with conn.transaction():
                conn.execute(body)
        _claims_schema_ready = True


def clean_session(v) -> str:
    """The browser's random reviewer-session id (it is the bearer of its leases)."""
    s = str(v or "").strip()
    if not _SESSION_RE.match(s):
        raise ReviewError("session must be 16-64 letters, digits, '-' or '_'")
    return s


def _clean_source(source) -> str:
    if source not in SOURCES:
        raise ReviewError(f"source must be one of {SOURCES}")
    return source


def _clean_item(source: str, item_id) -> str:
    s = str(item_id or "")
    ok = _UUID_RE.match(s) if source == "citizen" else _SEGMENT_RE.match(s)
    if not ok:
        raise ReviewError(f"bad {source} item id")
    return s


def _clean_queue(source: str, queue) -> str:
    allowed = VIDEO_QUEUES if source == "video" else CITIZEN_QUEUES
    if queue not in allowed:
        raise ReviewError(f"queue must be one of {sorted(allowed)}")
    return queue


def _video_where(queue: str, drive: Optional[str], label: Optional[str]) -> tuple[list, list]:
    if queue not in VIDEO_QUEUES:
        raise ReviewError(f"queue must be one of {list(VIDEO_QUEUES)}")
    where, params = ["v.assessment_id IS NOT NULL", VIDEO_QUEUES[queue]], []
    if drive:
        if not _SEGMENT_RE.match(drive):
            raise ReviewError("bad drive id")
        where.append("v.session_id = %s")
        params.append(drive)
    if label:
        if canonical(label) == NORMAL:
            where.append("a.stage1_label = 'Normal'")
        else:
            where.append("a.distress_types ? %s")
            params.append(canonical(label))
    return where, params


def _iso_rows(rows) -> list[dict]:
    items = []
    for r in rows:
        r = dict(r)
        r["assessment_id"] = str(r["assessment_id"])
        for k in ("utc", "reviewed_at", "processed_at", "claim_expires_at"):
            if k in r:
                r[k] = r[k].isoformat() if r.get(k) is not None else None
        r["image"] = f"/gis/image/{r['segment_id']}"
        sha = r.pop("image_sha256", None)
        r["image_v"] = sha[:16] if isinstance(sha, str) and len(sha) >= 16 else None
        items.append(r)
    return items


def _video_counts(cur, drive: Optional[str]) -> dict:
    base = ["v.assessment_id IS NOT NULL"] + (["v.session_id = %s"] if drive else [])
    bparams = [drive] if drive else []
    counts = {}
    for q, cond in VIDEO_QUEUES.items():
        counts[q] = cur.execute(f"SELECT COUNT(*) AS n FROM video_segments v JOIN assessments a "
                                f"ON a.id = v.assessment_id WHERE {' AND '.join(base + [cond])}",
                                bparams).fetchone()["n"]
    return counts


def _live_held_total(conn, session: str) -> int:
    return conn.execute("SELECT COUNT(*) FROM research.review_claims WHERE reviewer_session = %s "
                        "AND expires_at > now()", (session,)).fetchone()[0]


_UPSERT_TAIL = """
    ON CONFLICT (source, item_id) DO UPDATE SET
        reviewer_session = EXCLUDED.reviewer_session, reviewer_name = EXCLUDED.reviewer_name,
        queue = EXCLUDED.queue, claimed_at = EXCLUDED.claimed_at, last_seen_at = EXCLUDED.last_seen_at,
        expires_at = EXCLUDED.expires_at
    WHERE research.review_claims.expires_at < now()
       OR research.review_claims.reviewer_session = EXCLUDED.reviewer_session
    RETURNING item_id
"""


def claim_next_video(conn, *, session: str, reviewer_name: str, queue: str, n: int = 12,
                     drive: Optional[str] = None, label: Optional[str] = None) -> dict:
    """Up to n segments of a review list for this browser session: the ones it already holds that
    still belong in the list (renewed), topped up with free ones in the list's order. Free = no
    live lease by another session, not reviewed, not skipped by this session. The random list
    keeps its fixed order, so skipping over items other experts hold does not bias the sample."""
    ensure_claims_schema(conn)
    session = clean_session(session)
    name = _text(reviewer_name, "reviewer_name", 2, MAX_NAME)
    if queue not in CLAIM_QUEUES_VIDEO:
        raise ReviewError(f"queue must be one of {list(CLAIM_QUEUES_VIDEO)}")
    n = max(1, min(MAX_BATCH, int(n)))
    where, params = _video_where(queue, drive, label)
    w, order = " AND ".join(where), _VIDEO_ORDER[queue]
    new: list = []
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (_ADVISORY["video"],))
        mine = [r[0] for r in conn.execute(
            f"SELECT v.segment_id FROM research.review_claims c JOIN video_segments v ON v.segment_id = c.item_id "
            f"JOIN assessments a ON a.id = v.assessment_id "
            f"WHERE c.source = 'video' AND c.reviewer_session = %s AND {w} ORDER BY {order}",
            (session, *params)).fetchall()]
        if mine:
            conn.execute("UPDATE research.review_claims SET expires_at = now() + make_interval(secs => %s), "
                         "last_seen_at = now(), queue = %s, reviewer_name = %s WHERE source = 'video' "
                         "AND reviewer_session = %s AND item_id = ANY(%s)", (CLAIM_TTL_S, queue, name, session, mine))
        want = min(n - len(mine), MAX_CLAIMS_PER_SESSION - _live_held_total(conn, session))
        if want > 0:
            new = [r[0] for r in conn.execute(
                f"INSERT INTO research.review_claims (source, item_id, reviewer_session, reviewer_name, queue, "
                f"claimed_at, last_seen_at, expires_at) "
                f"SELECT 'video', v.segment_id, %s, %s, %s, now(), now(), now() + make_interval(secs => %s) "
                f"FROM video_segments v JOIN assessments a ON a.id = v.assessment_id "
                f"LEFT JOIN research.review_claims c ON c.source = 'video' AND c.item_id = v.segment_id "
                f"WHERE {w} AND (c.item_id IS NULL OR (c.expires_at < now() AND c.reviewer_session <> %s)) "
                f"AND NOT EXISTS (SELECT 1 FROM research.review_skips k WHERE k.source = 'video' "
                f"AND k.item_id = v.segment_id AND k.reviewer_session = %s) "
                f"ORDER BY {order} LIMIT %s" + _UPSERT_TAIL,
                (session, name, queue, CLAIM_TTL_S, *params, session, session, want)).fetchall()]
    ids = mine + [i for i in new if i not in mine]
    cur = conn.cursor(row_factory=dict_row)
    rows = cur.execute(
        f"SELECT {_LIST_COLS}, v.image_sha256, c.expires_at AS claim_expires_at FROM video_segments v "
        f"JOIN assessments a ON a.id = v.assessment_id "
        f"JOIN research.review_claims c ON c.source = 'video' AND c.item_id = v.segment_id "
        f"AND c.reviewer_session = %s WHERE v.segment_id = ANY(%s) ORDER BY {order}",
        (session, ids)).fetchall() if ids else []
    available = cur.execute(
        f"SELECT COUNT(*) AS n FROM video_segments v JOIN assessments a ON a.id = v.assessment_id "
        f"WHERE {w} AND NOT EXISTS (SELECT 1 FROM research.review_claims c WHERE c.source = 'video' "
        f"AND c.item_id = v.segment_id AND c.expires_at > now() AND c.reviewer_session <> %s)",
        (*params, session)).fetchone()["n"]
    total = cur.execute(f"SELECT COUNT(*) AS n FROM video_segments v JOIN assessments a ON a.id = v.assessment_id "
                        f"WHERE {w}", params).fetchone()["n"]
    return {"queue": queue, "items": _iso_rows(rows), "new": len(new), "held": len(rows),
            "available": available, "total": total, "in_progress_elsewhere": total - available,
            "counts": _video_counts(cur, drive), "ttl_s": CLAIM_TTL_S}


def claims_grant(conn, *, source: str, session: str, reviewer_name: str, queue: str, items: list,
                 n: Optional[int] = None) -> dict:
    """Lease specific items (in the given order) to this session, up to n of them: the ones it
    already holds are renewed, free ones are taken, ones another live session holds are reported
    in held_by_others. Used for citizen uploads (the page lists them from Supabase) and for editing
    an already-reviewed item."""
    ensure_claims_schema(conn)
    source = _clean_source(source)
    session = clean_session(session)
    name = _text(reviewer_name, "reviewer_name", 2, MAX_NAME)
    queue = _clean_queue(source, queue)
    if not isinstance(items, list) or not 1 <= len(items) <= MAX_CANDIDATES:
        raise ReviewError(f"items must be a list of 1-{MAX_CANDIDATES} ids")
    ids: list[str] = []
    for i in items:
        i = _clean_item(source, i)
        if i not in ids:
            ids.append(i)
    n = min(len(ids), MAX_BATCH) if n is None else max(1, min(MAX_BATCH, int(n)))
    skip_cond = ("AND NOT EXISTS (SELECT 1 FROM research.review_skips k WHERE k.source = %s "
                 "AND k.item_id = x.id AND k.reviewer_session = %s) ") if queue in REVIEW_QUEUES else ""
    new: list = []
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (_ADVISORY[source],))
        mine = [r[0] for r in conn.execute(
            "UPDATE research.review_claims SET expires_at = now() + make_interval(secs => %s), last_seen_at = now(), "
            "queue = %s, reviewer_name = %s WHERE source = %s AND reviewer_session = %s AND item_id = ANY(%s) "
            "RETURNING item_id", (CLAIM_TTL_S, queue, name, source, session, ids)).fetchall()]
        want = min(n - len(mine), MAX_CLAIMS_PER_SESSION - _live_held_total(conn, session))
        if want > 0:
            rest = [i for i in ids if i not in mine]
            p = [source, session, name, queue, CLAIM_TTL_S, rest, source, session]
            if skip_cond:
                p += [source, session]
            p.append(want)
            new = [r[0] for r in conn.execute(
                "INSERT INTO research.review_claims (source, item_id, reviewer_session, reviewer_name, queue, "
                "claimed_at, last_seen_at, expires_at) "
                "SELECT %s, x.id, %s, %s, %s, now(), now(), now() + make_interval(secs => %s) "
                "FROM unnest(%s::text[]) WITH ORDINALITY AS x(id, ord) "
                "LEFT JOIN research.review_claims c ON c.source = %s AND c.item_id = x.id "
                "WHERE (c.item_id IS NULL OR (c.expires_at < now() AND c.reviewer_session <> %s)) "
                f"{skip_cond}ORDER BY x.ord LIMIT %s" + _UPSERT_TAIL, p).fetchall()]
        others = {r[0]: r[1] for r in conn.execute(
            "SELECT item_id, reviewer_name FROM research.review_claims WHERE source = %s AND item_id = ANY(%s) "
            "AND reviewer_session <> %s AND expires_at > now()", (source, ids, session)).fetchall()}
    got = set(mine) | set(new)
    return {"source": source, "queue": queue, "granted": [i for i in ids if i in got], "new": len(new),
            "held_by_others": [{"item_id": i, "reviewer_name": others[i]} for i in ids if i in others],
            "ttl_s": CLAIM_TTL_S}


def _item_keys(items) -> list[tuple[str, str]]:
    if items is None:
        return []
    if not isinstance(items, list) or len(items) > MAX_CANDIDATES:
        raise ReviewError(f"items must be a list of at most {MAX_CANDIDATES} {{source, item_id}} objects")
    out = []
    for it in items:
        if not isinstance(it, Mapping):
            raise ReviewError("each item must be {source, item_id}")
        s = _clean_source(it.get("source"))
        out.append((s, _clean_item(s, it.get("item_id"))))
    return out


def claims_heartbeat(conn, *, session: str, items=None) -> dict:
    """Renew this session's leases (all of them, or the listed ones). A lease that expired but
    nobody took is renewed too; one another session took is reported as lost."""
    ensure_claims_schema(conn)
    session = clean_session(session)
    keys = _item_keys(items)
    with conn.transaction():
        if items is None:
            rows = conn.execute("UPDATE research.review_claims SET expires_at = now() + make_interval(secs => %s), "
                                "last_seen_at = now() WHERE reviewer_session = %s RETURNING source, item_id",
                                (CLAIM_TTL_S, session)).fetchall()
        else:
            rows = []
            for s in SOURCES:
                ids = [i for (src, i) in keys if src == s]
                if ids:
                    rows += conn.execute(
                        "UPDATE research.review_claims SET expires_at = now() + make_interval(secs => %s), "
                        "last_seen_at = now() WHERE reviewer_session = %s AND source = %s AND item_id = ANY(%s) "
                        "RETURNING source, item_id", (CLAIM_TTL_S, session, s, ids)).fetchall()
    held = [{"source": s, "item_id": i} for s, i in rows]
    have = {(s, i) for s, i in rows}
    lost = [{"source": s, "item_id": i} for s, i in keys if (s, i) not in have]
    return {"held": held, "lost": lost, "ttl_s": CLAIM_TTL_S}


def claims_release(conn, *, session: str, items=None, source: Optional[str] = None, skip: bool = False,
                   reviewer_name: Optional[str] = None) -> dict:
    """Drop this session's leases: the listed items, or all of one source, or all. skip=True also
    records the items as skipped by this session, so it is not handed them again."""
    ensure_claims_schema(conn)
    session = clean_session(session)
    keys = _item_keys(items)
    if source is not None:
        _clean_source(source)
    name = _text(reviewer_name, "reviewer_name", 2, MAX_NAME) if skip and keys else None
    released = []
    with conn.transaction():
        if items is None:
            q, p = "DELETE FROM research.review_claims WHERE reviewer_session = %s", [session]
            if source:
                q += " AND source = %s"
                p.append(source)
            released = conn.execute(q + " RETURNING source, item_id, queue", p).fetchall()
        else:
            for s in SOURCES:
                ids = [i for (src, i) in keys if src == s]
                if ids:
                    released += conn.execute(
                        "DELETE FROM research.review_claims WHERE reviewer_session = %s AND source = %s "
                        "AND item_id = ANY(%s) RETURNING source, item_id, queue", (session, s, ids)).fetchall()
        if name:
            queues = {(s, i): q for s, i, q in released}
            for s, i in keys:
                conn.execute("INSERT INTO research.review_skips (source, item_id, reviewer_session, reviewer_name, "
                             "queue) VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                             (s, i, session, name, queues.get((s, i))))
    return {"released": [{"source": s, "item_id": i} for s, i, _q in released],
            "skipped": len(keys) if name else 0}


def claims_active(conn, *, me: Optional[str] = None, source: Optional[str] = None,
                  with_items: bool = False) -> dict:
    """Who is reviewing now: live leases per expert name (never the session ids of others), per
    source/queue, and optionally the items held by other sessions (for "being edited by X")."""
    ensure_claims_schema(conn)
    if me is not None:
        me = clean_session(me)
    if source is not None:
        _clean_source(source)
    cur = conn.cursor(row_factory=dict_row)
    rows = cur.execute("SELECT source, item_id, reviewer_session, reviewer_name, queue, expires_at "
                       "FROM research.review_claims WHERE expires_at > now()"
                       + (" AND source = %s" if source else ""), (source,) if source else ()).fetchall()
    per_session: dict = {}
    by_queue: Counter = Counter()
    for r in rows:
        k = r["reviewer_session"]
        e = per_session.setdefault(k, {"reviewer_name": r["reviewer_name"], "video": 0, "citizen": 0,
                                       "mine": k == me})
        e[r["source"]] += 1
        by_queue[f"{r['source']}:{r['queue']}"] += 1
    out = {"experts": sorted(per_session.values(), key=lambda e: (not e["mine"], e["reviewer_name"])),
           "experts_online": len(per_session), "by_queue": dict(by_queue), "ttl_s": CLAIM_TTL_S}
    if with_items:
        out["items"] = [{"source": r["source"], "item_id": r["item_id"], "reviewer_name": r["reviewer_name"],
                         "expires_at": r["expires_at"].isoformat()}
                        for r in rows if r["reviewer_session"] != me]
    return out


def take_claim(conn, *, source: str, item_id: str, session: str, reviewer_name: str, queue: str) -> None:
    """Hold (or renew) the lease on one item for this session, or raise Conflict naming the
    expert who holds it. Used right before a save."""
    ensure_claims_schema(conn)
    source = _clean_source(source)
    item_id = _clean_item(source, item_id)
    session = clean_session(session)
    name = _text(reviewer_name, "reviewer_name", 2, MAX_NAME)
    with conn.transaction():
        got = conn.execute(
            "INSERT INTO research.review_claims (source, item_id, reviewer_session, reviewer_name, queue, claimed_at, "
            "last_seen_at, expires_at) VALUES (%s, %s, %s, %s, %s, now(), now(), now() + make_interval(secs => %s))"
            + _UPSERT_TAIL, (source, item_id, session, name, queue, CLAIM_TTL_S)).fetchone()
        if got is None:
            holder = conn.execute("SELECT reviewer_name FROM research.review_claims WHERE source = %s AND item_id = %s",
                                  (source, item_id)).fetchone()
            raise Conflict(f"{holder[0] if holder else 'Another expert'} is reviewing this item right now",
                           held_by=holder[0] if holder else None)


def _parse_instant(v):
    from datetime import datetime
    if v is None or not isinstance(v, str):
        return v
    try:
        return datetime.fromisoformat(v.replace("Z", "+00:00"))
    except ValueError:
        raise ReviewError("seen_reviewed_at must be an ISO timestamp or null") from None


def same_instant(a, b) -> bool:
    """reviewed_at as stored vs the value the page saw (datetime / ISO text / None)."""
    a, b = _parse_instant(a), _parse_instant(b)
    if a is None or b is None:
        return a is None and b is None
    return a == b


def save_video_review_claimed(conn, segment_id: str, payload: Mapping) -> dict:
    """save_video_review with the multi-expert checks, in one transaction:
      - another session's live lease on the item -> Conflict (nothing is saved);
      - the item was reviewed since the page loaded it (payload seen_reviewed_at; null or absent
        when it was unreviewed) -> Conflict "X saved a review at T";
      - otherwise write the expert_* columns, append the log row, and release the lease.
    payload.session is the page's reviewer session; without one (an old page) the save is allowed
    only when nobody holds a live lease on the item."""
    ensure_claims_schema(conn)
    review = clean_review(payload, source="video", strict_types=True)
    session = clean_session(payload["session"]) if payload.get("session") else None
    op_id = clean_op_id(payload.get("op_id"))
    seen = payload.get("seen_reviewed_at")
    if seen is not None and not isinstance(seen, str):
        raise ReviewError("seen_reviewed_at must be an ISO timestamp or null")
    _parse_instant(seen)
    with conn.transaction():
        cur = conn.cursor(row_factory=dict_row)
        row = cur.execute(
            "SELECT v.segment_id, v.session_id, a.* FROM video_segments v JOIN assessments a "
            "ON a.id = v.assessment_id WHERE v.segment_id = %s FOR UPDATE OF a", (segment_id,)).fetchone()
        if row is None:
            raise NotFound(f"no answered segment {segment_id!r}")
        # The same save arriving again (its answer was lost, or two tabs flushed the queue): the
        # row lock above orders it after the first arrival, so the first one's log row is visible.
        prev = _logged(conn, op_id)
        if prev is not None:
            if prev["segment_id"] != segment_id:
                raise ReviewError("op_id was already used for another item")
            if session:
                conn.execute("DELETE FROM research.review_claims WHERE source = 'video' AND item_id = %s "
                             "AND reviewer_session = %s", (segment_id, session))
            return {"segment_id": segment_id, "action": prev["action"], "expert_types": prev["expert_types"],
                    "expert_severity": prev["expert_severity"],
                    "reviewed_at": row["reviewed_at"].isoformat() if row.get("reviewed_at") else None,
                    **_replay(prev)}
        claim = cur.execute(
            "SELECT reviewer_session, reviewer_name, expires_at > now() AS live FROM research.review_claims "
            "WHERE source = 'video' AND item_id = %s FOR UPDATE", (segment_id,)).fetchone()
        if claim and claim["live"] and claim["reviewer_session"] != session:
            raise Conflict(f"{claim['reviewer_name']} is reviewing this item right now",
                           held_by=claim["reviewer_name"])
        if not same_instant(row.get("reviewed_at"), seen):
            when = row["reviewed_at"].isoformat() if row.get("reviewed_at") else None
            raise Conflict(f"{row.get('reviewer_name') or 'Another expert'} saved a review of this item "
                           f"({when}); reload to see it before changing it",
                           reviewed_by=row.get("reviewer_name"), reviewed_at=when)
        vlm = vlm_snapshot(row)
        action = "update" if row.get("expert_reviewed") else "confirm"
        done = cur.execute(
            "UPDATE assessments SET expert_reviewed = true, expert_corrected_types = %s, "
            "expert_corrected_severity = %s, expert_notes = %s, reviewer_name = %s, reviewed_at = NOW() "
            "WHERE id = %s RETURNING reviewed_at", (Jsonb(review["types"]), review["severity"], review["notes"],
                                                    review["reviewer_name"], row["id"])).fetchone()
        out = _insert_log(conn, source="video", assessment_id=str(row["id"]), segment_id=row["segment_id"],
                          session_id=row["session_id"], review=review, vlm=vlm, action=action, op_id=op_id)
        conn.execute("DELETE FROM research.review_claims WHERE source = 'video' AND item_id = %s", (segment_id,))
    return {"segment_id": segment_id, "action": action, "expert_types": review["types"],
            "expert_severity": review["severity"], "reviewed_at": done["reviewed_at"].isoformat(), **out}


def finish_citizen_review(conn, assessment_id: str, payload: Mapping, row: Mapping) -> dict:
    """After the server's conditional Supabase save: log it and release the lease."""
    out = log_citizen_review(conn, assessment_id, payload, row)
    ensure_claims_schema(conn)
    conn.execute("DELETE FROM research.review_claims WHERE source = 'citizen' AND item_id = %s", (assessment_id,))
    return out


def list_video_marked(conn, queue: str, *, me: Optional[str] = None, drive: Optional[str] = None,
                      label: Optional[str] = None, limit: int = 24, offset: int = 0) -> dict:
    """list_video plus, per item, who holds a live lease on it (claim_name, claim_mine) and the image
    version (image_v) for cacheable image URLs."""
    ensure_claims_schema(conn)
    me = clean_session(me) if me else None
    where, params = _video_where(queue, drive, label)
    w = " AND ".join(where)
    limit, offset = max(1, min(100, int(limit))), max(0, int(offset))
    cur = conn.cursor(row_factory=dict_row)
    rows = cur.execute(
        f"SELECT {_LIST_COLS}, v.image_sha256, c.reviewer_name AS claim_name, "
        f"(c.reviewer_session IS NOT NULL AND c.reviewer_session = %s) AS claim_mine, "
        f"c.expires_at AS claim_expires_at "
        f"FROM video_segments v JOIN assessments a ON a.id = v.assessment_id "
        f"LEFT JOIN research.review_claims c ON c.source = 'video' AND c.item_id = v.segment_id "
        f"AND c.expires_at > now() WHERE {w} ORDER BY {_VIDEO_ORDER[queue]} LIMIT %s OFFSET %s",
        (me or "", *params, limit, offset)).fetchall()
    total = cur.execute(f"SELECT COUNT(*) AS n FROM video_segments v JOIN assessments a ON a.id = v.assessment_id "
                        f"WHERE {w}", params).fetchone()["n"]
    return {"queue": queue, "total": total, "limit": limit, "offset": offset, "counts": _video_counts(cur, drive),
            "items": _iso_rows(rows)}


# ============================================================
# Save events (migration 008): what did not go to plan, from the page and the server
# ============================================================
# Every event is also appended to eval_results/expert_events.log (git-ignored), so a problem with
# the database itself is still recorded. Recording an event never raises.

EVENT_KINDS = {
    # page (client)
    "save_retry", "save_recovered", "save_conflict", "save_rejected", "save_dropped", "save_queued_on_load",
    "client_error",
    # server
    "server_save_error", "supabase_error", "db_unavailable", "db_retry",
}
EVENT_LEVELS = ("info", "warn", "error")
EVENTS_LOG = PROJECT_ROOT / "eval_results" / "expert_events.log"
_events_file_lock = threading.Lock()
MAX_EVENT_MESSAGE = 500


def _event_row(origin: str, e: Mapping) -> dict:
    kind = str(e.get("kind") or "")
    if kind not in EVENT_KINDS:
        raise ReviewError(f"unknown event kind {kind!r}")
    level = str(e.get("level") or "warn")
    if level not in EVENT_LEVELS:
        level = "warn"

    def short(v, n):
        return None if v is None else str(v)[:n]

    status = e.get("http_status")
    attempts = e.get("attempts")
    detail = e.get("detail")
    if detail is not None and (not isinstance(detail, Mapping) or len(str(detail)) > 2000):
        detail = {"note": "detail dropped (not an object or over 2000 characters)"}
    return {"origin": origin, "kind": kind, "level": level, "op_id": short(e.get("op_id"), 64),
            "source": short(e.get("source"), 10), "item_id": short(e.get("item_id"), 200),
            "reviewer_name": short(e.get("reviewer_name"), MAX_NAME),
            "reviewer_session": short(e.get("reviewer_session") or e.get("session"), 64),
            "http_status": int(status) if isinstance(status, (int, float)) and 0 <= status < 1000 else None,
            "attempts": int(attempts) if isinstance(attempts, (int, float)) and 0 <= attempts < 10**6 else None,
            "message": short(e.get("message"), MAX_EVENT_MESSAGE), "detail": detail}


def append_event_file(row: Mapping) -> None:
    """One JSON line per event in eval_results/expert_events.log. Never raises."""
    import json
    from datetime import datetime, timezone
    try:
        line = json.dumps({"at": datetime.now(timezone.utc).isoformat(), **row}, default=str)
        with _events_file_lock:
            EVENTS_LOG.parent.mkdir(parents=True, exist_ok=True)
            with open(EVENTS_LOG, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:                                    # noqa: BLE001 - logging must never fail a request
        pass


def record_events(conn, origin: str, events: Iterable[Mapping]) -> int:
    """Store events (validated, truncated) in the file and the database; returns how many."""
    rows = [_event_row(origin, e) for e in events]
    for r in rows:
        append_event_file(r)
    return insert_event_rows(conn, rows)


def insert_event_rows(conn, rows: list) -> int:
    """Insert already-validated event rows (from _event_row) into research.expert_events."""
    if not rows:
        return 0
    ensure_schema(conn)
    with conn.transaction():
        for r in rows:
            conn.execute(
                "INSERT INTO research.expert_events (origin, kind, level, op_id, source, item_id, reviewer_name, "
                "reviewer_session, http_status, attempts, message, detail) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (r["origin"], r["kind"], r["level"], r["op_id"], r["source"], r["item_id"], r["reviewer_name"],
                 r["reviewer_session"], r["http_status"], r["attempts"], r["message"],
                 Jsonb(r["detail"]) if r["detail"] is not None else None))
    return len(rows)


def events_summary(conn, hours: int = 24, recent: int = 10) -> dict:
    """Counts per kind over the last `hours`, and the latest events (no review content)."""
    ensure_schema(conn)
    hours = max(1, min(24 * 30, int(hours)))
    cur = conn.cursor(row_factory=dict_row)
    counts = {r["kind"]: r["n"] for r in cur.execute(
        "SELECT kind, COUNT(*) AS n FROM research.expert_events "
        "WHERE created_at > now() - make_interval(hours => %s) GROUP BY kind", (hours,)).fetchall()}
    last = cur.execute(
        "SELECT created_at, origin, kind, level, source, item_id, reviewer_name, http_status, attempts, message "
        "FROM research.expert_events ORDER BY id DESC LIMIT %s", (max(0, min(100, int(recent))),)).fetchall()
    unresolved = cur.execute(
        "SELECT COUNT(DISTINCT e.op_id) AS n FROM research.expert_events e WHERE e.kind = 'save_retry' "
        "AND e.created_at > now() - make_interval(hours => %s) AND e.op_id IS NOT NULL "
        "AND NOT EXISTS (SELECT 1 FROM research.expert_review_log l WHERE l.op_id = e.op_id) "
        "AND NOT EXISTS (SELECT 1 FROM research.expert_events f WHERE f.op_id = e.op_id "
        "AND f.kind IN ('save_conflict', 'save_rejected', 'save_dropped'))", (hours,)).fetchone()["n"]
    return {"hours": hours, "counts": counts, "retrying_not_yet_saved": unresolved,
            "problems": sum(n for k, n in counts.items() if k not in ("save_recovered",)),
            "recent": [{**r, "created_at": r["created_at"].isoformat()} for r in last]}


# ============================================================
# Leaderboard: items each expert has reviewed (from the review log)
# ============================================================

def leaderboard(conn, tz: str = "Asia/Kolkata") -> dict:
    """Per expert (names compared ignoring case and repeated spaces): distinct items reviewed, in all
    and today (by the day of their first review of the item), split by source. An item two experts
    reviewed counts for both; an update of one's own review does not count again."""
    ensure_schema(conn)
    cur = conn.cursor(row_factory=dict_row)
    rows = cur.execute(
        "WITH items AS (SELECT lower(regexp_replace(btrim(reviewer_name), '[[:space:]]+', ' ', 'g')) AS who, source, "
        "COALESCE(segment_id, assessment_id::text) AS item, MIN(created_at) AS first_at, MAX(created_at) AS last_at "
        "FROM research.expert_review_log GROUP BY 1, 2, 3), "
        "names AS (SELECT DISTINCT ON (lower(regexp_replace(btrim(reviewer_name), '[[:space:]]+', ' ', 'g'))) lower(regexp_replace(btrim(reviewer_name), '[[:space:]]+', ' ', 'g')) AS who, "
        "regexp_replace(btrim(reviewer_name), '[[:space:]]+', ' ', 'g') AS name FROM research.expert_review_log ORDER BY lower(regexp_replace(btrim(reviewer_name), '[[:space:]]+', ' ', 'g')), created_at DESC) "
        "SELECT n.name, i.who, COUNT(*) AS total, COUNT(*) FILTER (WHERE i.source = 'video') AS video, "
        "COUNT(*) FILTER (WHERE i.source = 'citizen') AS citizen, "
        "COUNT(*) FILTER (WHERE (i.first_at AT TIME ZONE %s)::date = (now() AT TIME ZONE %s)::date) AS today, "
        "MAX(i.last_at) AS last_at FROM items i JOIN names n USING (who) GROUP BY n.name, i.who "
        "ORDER BY total DESC, last_at DESC", (tz, tz)).fetchall()
    experts = [{"name": r["name"], "key": r["who"], "total": r["total"], "today": r["today"],
                "video": r["video"], "citizen": r["citizen"], "last_at": r["last_at"].isoformat()} for r in rows]
    return {"experts": experts, "team_total": sum(e["total"] for e in experts),
            "team_today": sum(e["today"] for e in experts), "timezone": tz}
