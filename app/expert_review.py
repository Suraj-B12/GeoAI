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
import threading
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.gis_core import DISTRESS_LABELS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MIGRATION = PROJECT_ROOT / "migrations" / "006_expert_review_log.sql"

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
        if conn.execute("SELECT to_regclass('research.expert_review_log')").fetchone()[0] is None:
            from scripts.setup_local_db import statements_without_outer_transaction
            body = statements_without_outer_transaction(MIGRATION.read_text(encoding="utf-8-sig"))
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


def _insert_log(conn, *, source, assessment_id, segment_id, session_id, review, vlm, action) -> dict:
    cmp = compare(vlm, review["types"], review["severity"])
    row = conn.cursor(row_factory=dict_row).execute(
        "INSERT INTO research.expert_review_log (source, assessment_id, segment_id, session_id, queue, action, "
        "reviewer_name, vlm, expert_types, expert_severity, expert_notes, comparison, heatmap, ui) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id, created_at",
        (source, assessment_id, segment_id, session_id, review["queue"], action, review["reviewer_name"],
         Jsonb(vlm), Jsonb(review["types"]), review["severity"], review["notes"], Jsonb(cmp),
         Jsonb(review["heatmap"]), Jsonb(review["ui"]))).fetchone()
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
    vlm = vlm_snapshot(row)
    vlm["needs_expert_review"] = row.get("status") == "expert_review"   # the page clears the flag on save
    prior = conn.execute("SELECT 1 FROM research.expert_review_log WHERE source = 'citizen' AND assessment_id = %s "
                         "LIMIT 1", (assessment_id,)).fetchone()
    with conn.transaction():
        out = _insert_log(conn, source="citizen", assessment_id=assessment_id, segment_id=None, session_id=None,
                          review=review, vlm=vlm, action="update" if prior else "confirm")
    return {"assessment_id": assessment_id, **out}


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
