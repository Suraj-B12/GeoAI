"""
Local image store + database storage for the video pipeline (GeoAI_Video segment mode).

GeoAI_Video turns a drive video into one record per 5 m road segment (schema
`geoai.segment.v1`, GeoAI_Video docs/pipeline/13-segment-frames.md section 5)
and the best full frame of each `send` / `audit` segment, which the VLM
classifies. This module stores a finished run in the local PostgreSQL
(DB_BACKEND=postgres, schema: db_schema.sql + migrations 000-005) and the
frames on the local disk in place of Cloudinary. The column-by-column mapping
is GeoAI_Video docs/DB_MAPPING.md.

    run_dir/sidecars/<id>.json   every record, all statuses   -> video_segments (one row each)
    run_dir/frames/<id>.png      send / audit frames          -> <IMAGE_ROOT>/<session>/<id>.png
    answers[<id>]                raw /classify JSON           -> photos + assessments (+ video_segments.vlm_response)

A classified segment becomes an ordinary photos + assessments pair, so the
operator dashboard, the metrics views and the expert UI show it unchanged:
the photos insert fires trg_sync_photo_to_assessment (migration 002), and the
assessment it creates is filled with the VLM result in the SAME transaction.
It therefore leaves the transaction already 'classified' / 'expert_review'
and the PipelineWorker, whose claim only sees committed 'pending' rows, can
never pick it up and classify it a second time.

Idempotent, keyed by the segment id: a re-run updates the rows it made before
(video_segments.photo_id / assessment_id, else assessments.road_segment_id,
which migration 005 makes unique) and reuses images whose SHA-256 already
matches. It never duplicates a row and never deletes one.

Images are written atomically (temporary file in the target folder, fsync,
rename) and only after their SHA-256 matches the record's frame.sha256, so the
store never holds a partial or wrong file under a final name. Files are
written before the database transaction: if the transaction fails, the files
are simply reused by the next run.

Synchronous psycopg: the video pipeline calls this from a worker thread.
Interface (coded against by app/video_pipeline.py):
    IMAGE_ROOT, connect(), store_session(), verify_session(), set_session_state(), list_sessions()
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping, Optional

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.local_images import image_root  # noqa: E402
from scripts.utils import normalize_severity  # noqa: E402  (the worker's own severity rule)

IMAGE_ROOT: Path = image_root()

SCHEMA = "geoai.segment.v1"
STATUSES = ("send", "audit", "skip", "not_covered")
SENT_STATUSES = ("send", "audit")
SESSION_STATES = ("queued", "extracting", "classifying", "storing", "stored", "cleaned", "failed")
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_CHUNK = 1 << 20


# ============================================================
# Connection / transactions
# ============================================================

def connect(dsn: Optional[str] = None, *, connect_timeout: Optional[int] = None) -> psycopg.Connection:
    """A psycopg connection to DATABASE_URL (autocommit; UTC session).

    Autocommit, so each function below runs exactly one explicit transaction
    and commits it, whatever the caller did before on the connection.
    connect_timeout (s): give up sooner than psycopg's default (~130 s) when the
    server is down; the read-only GIS API uses it so a stopped database is a quick 503.
    """
    dsn = dsn or os.environ.get("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL is not set (see SETUP.md, 'Local PostgreSQL').")
    extra = {"connect_timeout": int(connect_timeout)} if connect_timeout else {}
    return psycopg.connect(dsn, autocommit=True, options="-c TimeZone=UTC", **extra)


@contextmanager
def _transaction(conn: psycopg.Connection) -> Iterator[None]:
    """One committed transaction, also on a non-autocommit connection: there,
    conn.transaction() may only open a savepoint inside an implicit transaction,
    so the outer one is committed explicitly."""
    with conn.transaction():
        yield
    if not conn.autocommit:
        conn.commit()


# ============================================================
# Records, answers, images
# ============================================================

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _num(x) -> Optional[float]:
    """A finite float, or None (records use null for unknown values)."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _utc(ms) -> Optional[datetime]:
    v = _num(ms)
    return datetime.fromtimestamp(v / 1000.0, tz=timezone.utc) if v is not None else None


def segment_id(session_id: str, k: int) -> str:
    """`<session_id>_seg<k:05d>`, as GeoAI_Video segment_records.segment_id."""
    return f"{session_id}_seg{int(k):05d}"


def _record_problems(rec: Any, path: Path) -> list[str]:
    """What stops a sidecar from being stored (the full schema is validated by GeoAI_Video;
    this checks only what the database mapping relies on)."""
    if not isinstance(rec, dict):
        return [f"{path.name}: not a JSON object"]
    p = []
    if rec.get("schema") != SCHEMA:
        p.append(f"schema is {rec.get('schema')!r}, not {SCHEMA!r}")
    if rec.get("status") not in STATUSES:
        p.append(f"status {rec.get('status')!r} is not one of {STATUSES}")
    sid = rec.get("session_id")
    seg = rec.get("segment") if isinstance(rec.get("segment"), dict) else {}
    k = seg.get("k")
    if not isinstance(sid, str) or not sid:
        p.append("session_id missing")
    if not isinstance(k, int) or isinstance(k, bool) or k < 0:
        p.append("segment.k must be an integer >= 0")
    if not p and rec.get("id") != segment_id(sid, k):
        p.append(f"id {rec.get('id')!r} is not {segment_id(sid, k)!r}")
    if not p and path.stem != rec["id"]:
        p.append(f"file name {path.name} does not match id {rec['id']}")
    if rec.get("status") in SENT_STATUSES:
        frame = rec.get("frame") if isinstance(rec.get("frame"), dict) else {}
        if not isinstance(frame.get("file"), str) or not frame.get("file"):
            p.append("frame.file missing for a sent segment")
        if not (isinstance(frame.get("sha256"), str) and _HEX64.match(frame["sha256"])):
            p.append("frame.sha256 (64 lowercase hex) missing for a sent segment")
    return [f"{path.name}: {x}" for x in p]


def load_records(run_dir: Path) -> list[dict]:
    """Every sidecar of a run, in route order; ValueError listing every problem.

    All records must belong to one session (a run folder is one session).
    """
    folder = Path(run_dir) / "sidecars"
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} does not exist")
    recs, problems = [], []
    for path in sorted(folder.glob("*.json")):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            problems.append(f"{path.name}: unreadable ({e})")
            continue
        found = _record_problems(rec, path)
        problems.extend(found)
        if not found:
            recs.append(rec)
    sessions = sorted({r["session_id"] for r in recs})
    if len(sessions) > 1:
        problems.append(f"records of more than one session: {sessions}")
    if problems:
        raise ValueError(f"{len(problems)} problem(s) in {folder}: " + "; ".join(problems[:20]))
    if not recs:
        raise ValueError(f"no records in {folder}")
    return sorted(recs, key=lambda r: r["segment"]["k"])


def _answer_problems(seg: str, ans: Any) -> list[str]:
    """Only a successful /classify answer may become an assessment (the CHECKs would reject the rest)."""
    if not isinstance(ans, dict):
        return [f"{seg}: answer is not a JSON object"]
    p = []
    if ans.get("stage1_label") not in ("Normal", "Distress"):
        p.append(f"stage1_label {ans.get('stage1_label')!r} is not Normal/Distress")
    c1 = _num(ans.get("stage1_confidence"))
    if c1 is None or not 0.0 <= c1 <= 1.0:
        p.append("stage1_confidence must be a number in [0, 1]")
    c2 = ans.get("stage2_confidence", 0.0)
    if c2 is not None and (_num(c2) is None or not 0.0 <= float(c2) <= 1.0):
        p.append("stage2_confidence must be a number in [0, 1]")
    if not isinstance(ans.get("distress_types", []) or [], list):
        p.append("distress_types must be a list")
    return [f"{seg}: {x}" for x in p]


def _frame_source(run_dir: Path, rec: Mapping) -> Path:
    """run_dir/<frame.file>, refusing paths that leave the run folder."""
    run = Path(run_dir).resolve()
    src = run.joinpath(*PurePosixPath(rec["frame"]["file"]).parts).resolve()
    if run not in src.parents:
        raise ValueError(f"{rec['id']}: frame.file {rec['frame']['file']!r} points outside the run folder")
    return src


def _atomic_copy(src: Path, dst: Path, want_sha: str) -> None:
    """Copy src to dst via a temporary file, fsync and rename; ValueError (and no
    file) if the bytes do not hash to want_sha."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
    h = hashlib.sha256()
    try:
        with open(src, "rb") as fin, open(tmp, "wb") as fout:
            for chunk in iter(lambda: fin.read(_CHUNK), b""):
                h.update(chunk)
                fout.write(chunk)
            fout.flush()
            os.fsync(fout.fileno())
        if h.hexdigest() != want_sha:
            raise ValueError(f"{src} has SHA-256 {h.hexdigest()}, the record says {want_sha}")
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)


def store_image(run_dir: Path, rec: Mapping, root: Path) -> tuple[Path, bool]:
    """(absolute stored path, written?) for a sent segment's frame.

    Reused when the store already holds a file with the record's SHA-256 (also
    after the run's frames were cleaned up); otherwise copied from the run.
    """
    want = rec["frame"]["sha256"]
    dst = (Path(root) / rec["session_id"] / f"{rec['id']}.png").resolve()
    if dst.is_file() and _sha256_file(dst) == want:
        return dst, False
    src = _frame_source(run_dir, rec)
    if not src.is_file():
        raise FileNotFoundError(f"{rec['id']}: {src} is missing and the store has no copy with SHA-256 {want}")
    _atomic_copy(src, dst, want)
    return dst, True


def _address(rec: Mapping) -> str:
    """"<road_name> km <chainage>" when both are known, else the road name, else the session id."""
    road = ((rec.get("survey") or {}).get("road_name") or "").strip()
    ch = _num((rec.get("segment") or {}).get("chainage_start_km"))
    if road and ch is not None:
        return f"{road} km {ch:.3f}"
    return road or rec["session_id"]


def _model_version(answer: Mapping) -> str:
    """The answer's own model id if it carries one; else the classifier this server runs
    (/classify answers carry no model field; the batch runner records /health separately)."""
    for key in ("model_version", "model_name", "model_id"):
        if isinstance(answer.get(key), str) and answer[key]:
            return answer[key]
    model = os.environ.get("MODEL_PATH") or DEFAULT_MODEL
    return f"{model} prompts={os.environ.get('PROMPTS_VERSION') or 'v2'}"


def assessment_fields(rec: Mapping, answer: Mapping, image_path: str, image_sha256: str) -> dict:
    """assessments columns for a classified segment (mirrors PipelineWorker._do_inference)."""
    s1_label = answer["stage1_label"]
    distressed = s1_label == "Distress"
    needs = bool(answer.get("needs_expert_review", False))
    frame = rec.get("frame") or {}
    return {
        "status": "expert_review" if needs else "classified",
        "stage1_label": s1_label,
        "stage1_confidence": float(answer["stage1_confidence"]),
        "is_distressed": distressed,
        # Stage 2 runs only on Distress; the worker writes [] / 'None' for Normal.
        "distress_types": [str(t) for t in (answer.get("distress_types") or [])] if distressed else [],
        "severity": normalize_severity(answer.get("severity")) if distressed else "None",
        "description": answer.get("description") or "",
        "stage2_confidence": float(answer.get("stage2_confidence") or 0.0),
        "needs_expert_review": needs,
        "road_segment_id": rec["id"],
        "image_width": frame.get("width"),
        "image_height": frame.get("height"),
        "processing_time_ms": _num(answer.get("processing_time_ms")),
        "model_version": _model_version(answer),
        "raw_response": {
            "source": "video_pipeline",
            "segment_record": rec,
            "vlm_response": answer,
            "image_sha256": image_sha256,
            # Copies of keys the dashboard reads from raw_response (stage timings, indicators)
            "stage1_time_ms": answer.get("stage1_time_ms"),
            "stage2_time_ms": answer.get("stage2_time_ms"),
            "condition_indicators": answer.get("condition_indicators") or [],
        },
        "error_message": None,
        "claimed_at": None,
        "claimed_by": None,
    }


# ============================================================
# store_session
# ============================================================

_ASSESSMENT_TYPES = {"distress_types": "jsonb", "raw_response": "jsonb", "stage1_confidence": "real",
                     "stage2_confidence": "real", "processing_time_ms": "real", "image_width": "integer",
                     "image_height": "integer", "is_distressed": "boolean", "needs_expert_review": "boolean",
                     "claimed_at": "timestamptz"}


def _photo_and_assessment(cur, rec: Mapping, answer: Mapping, image_path: str, sha: str,
                          photo_id: Optional[int], assessment_id: Optional[str]) -> tuple[int, str, bool, bool]:
    """Create or update the photos + assessments pair of one classified segment.

    Returns (photo_id, assessment_id, photo_created, assessment_created).
    Lookup order: the ids stored on video_segments, then assessments.road_segment_id.
    """
    seg = rec["id"]
    centre = (rec.get("segment") or {}).get("centre") or {}
    photo = {"image_url": image_path, "address": _address(rec), "latitude": _num(centre.get("lat")),
             "longitude": _num(centre.get("lon")),
             "captured_at": _utc((rec.get("frame") or {}).get("utc_ms"))}
    if assessment_id is None:
        row = cur.execute("SELECT id, photo_id FROM assessments WHERE road_segment_id = %s", (seg,)).fetchone()
        if row:
            assessment_id, photo_id = str(row["id"]), row["photo_id"] if row["photo_id"] is not None else photo_id
    if assessment_id is not None and not cur.execute("SELECT 1 FROM assessments WHERE id = %s",
                                                     (assessment_id,)).fetchone():
        assessment_id = None
    if photo_id is not None and not cur.execute("SELECT 1 FROM photos WHERE id = %s", (photo_id,)).fetchone():
        photo_id = None

    photo_created = assessment_created = False
    if photo_id is None:
        photo_id = cur.execute(
            "INSERT INTO photos (image_url, address, latitude, longitude, captured_at) "
            "VALUES (%(image_url)s, %(address)s, %(latitude)s, %(longitude)s, %(captured_at)s) RETURNING id",
            photo).fetchone()["id"]
        photo_created = True
        made = cur.execute("SELECT id FROM assessments WHERE photo_id = %s", (photo_id,)).fetchone()
        if assessment_id is None:
            assessment_id, assessment_created = (str(made["id"]), True) if made else (None, False)
        elif made:
            # The segment already had an assessment (its photo was deleted): keep it,
            # drop the pending one the trigger just made.
            cur.execute("DELETE FROM assessments WHERE id = %s", (made["id"],))
    else:
        cur.execute("UPDATE photos SET image_url = %(image_url)s, address = %(address)s, latitude = %(latitude)s, "
                    "longitude = %(longitude)s, captured_at = %(captured_at)s WHERE id = %(id)s",
                    {**photo, "id": photo_id})
        if assessment_id is None:
            made = cur.execute("SELECT id FROM assessments WHERE photo_id = %s", (photo_id,)).fetchone()
            assessment_id = str(made["id"]) if made else None
    if assessment_id is None:
        # No trigger row (e.g. trigger missing): create the row directly, never as 'pending'.
        assessment_id = str(cur.execute(
            "INSERT INTO assessments (photo_id, image_url, status, model_version) "
            "VALUES (%s, %s, 'processing', 'video_pipeline') RETURNING id", (photo_id, image_path)).fetchone()["id"])
        assessment_created = True

    fields = {**assessment_fields(rec, answer, image_path, sha), "photo_id": photo_id, "image_url": image_path,
              "address": photo["address"], "latitude": photo["latitude"], "longitude": photo["longitude"]}
    sets = sql.SQL(", ").join(
        sql.SQL("{} = %s::{}").format(sql.Identifier(k), sql.SQL(_ASSESSMENT_TYPES.get(k, "text")))
        if k in _ASSESSMENT_TYPES else sql.SQL("{} = %s").format(sql.Identifier(k)) for k in fields)
    params = [Jsonb(v) if isinstance(v, (dict, list)) else v for v in fields.values()]
    cur.execute(sql.SQL("UPDATE assessments SET {}, processed_at = NOW() WHERE id = %s").format(sets),
                (*params, assessment_id))
    return int(photo_id), assessment_id, photo_created, assessment_created


_SEGMENT_COLS = ("segment_id", "session_id", "k", "status", "length_m", "s0_m", "s1_m", "centre_lat", "centre_lon",
                 "chainage_start_km", "chainage_end_km", "utc", "image_path", "image_sha256", "photo_id",
                 "assessment_id", "record", "vlm_response")
# On a re-run, links and answers from an earlier run are kept when this run has none.
_KEEP_IF_NULL = ("image_path", "image_sha256", "photo_id", "assessment_id", "vlm_response")


def _segment_row(rec: Mapping, image_path: Optional[str], sha: Optional[str], photo_id, assessment_id,
                 answer) -> dict:
    seg = rec.get("segment") or {}
    centre = seg.get("centre") or {}
    return {"segment_id": rec["id"], "session_id": rec["session_id"], "k": seg["k"], "status": rec["status"],
            "length_m": _num(seg.get("length_m")), "s0_m": _num(seg.get("s0_m")), "s1_m": _num(seg.get("s1_m")),
            "centre_lat": _num(centre.get("lat")), "centre_lon": _num(centre.get("lon")),
            "chainage_start_km": _num(seg.get("chainage_start_km")),
            "chainage_end_km": _num(seg.get("chainage_end_km")),
            "utc": _utc((rec.get("frame") or {}).get("utc_ms")),
            "image_path": image_path, "image_sha256": sha, "photo_id": photo_id, "assessment_id": assessment_id,
            "record": Jsonb(rec), "vlm_response": Jsonb(answer) if answer is not None else None}


_UPSERT_SEGMENT = sql.SQL("INSERT INTO video_segments ({cols}) VALUES ({vals}) ON CONFLICT (segment_id) "
                          "DO UPDATE SET {sets}").format(
    cols=sql.SQL(", ").join(sql.Identifier(c) for c in _SEGMENT_COLS),
    vals=sql.SQL(", ").join(sql.Placeholder(c) for c in _SEGMENT_COLS),
    sets=sql.SQL(", ").join(
        sql.SQL("{c} = COALESCE(EXCLUDED.{c}, video_segments.{c})").format(c=sql.Identifier(c))
        if c in _KEEP_IF_NULL else sql.SQL("{c} = EXCLUDED.{c}").format(c=sql.Identifier(c))
        for c in _SEGMENT_COLS if c != "segment_id"))


def store_session(conn: psycopg.Connection, run_dir: Path, answers: dict[str, dict], *,
                  image_root: Path | None = None) -> dict:
    """Store one segment-mode run: images into the store, every record into video_segments,
    every answered sent segment as a photos + assessments pair. ONE transaction.

    answers: {segment_id: raw /classify JSON} for the segments the VLM answered (may be
    partial; unanswered sent segments get their image and record, no photo yet).
    Raises ValueError / FileNotFoundError before touching the database when a record,
    answer or image is not usable.

    Returns {records, sent, answered, photos, assessments, images_written, images_reused,
             photos_created, assessments_created}:
      photos / assessments  rows linked to answered segments (created or updated by this call)
      *_created             of those, rows that did not exist before (0 on an identical re-run)
    """
    run_dir = Path(run_dir)
    root = Path(image_root) if image_root is not None else IMAGE_ROOT
    records = load_records(run_dir)
    session = records[0]["session_id"]
    by_id = {r["id"]: r for r in records}
    answers = dict(answers or {})

    problems = []
    for seg, ans in answers.items():
        rec = by_id.get(seg)
        if rec is None:
            problems.append(f"{seg}: answer for a segment that has no record in this run")
        elif rec["status"] not in SENT_STATUSES:
            problems.append(f"{seg}: answer for a {rec['status']} segment (only send/audit are classified)")
        else:
            problems.extend(_answer_problems(seg, ans))
    if problems:
        raise ValueError(f"{len(problems)} unusable answer(s): " + "; ".join(problems[:20]))

    images: dict[str, tuple[str, str]] = {}
    written = reused = 0
    for rec in records:
        if rec["status"] in SENT_STATUSES:
            path, was_written = store_image(run_dir, rec, root)
            images[rec["id"]] = (str(path), rec["frame"]["sha256"])
            written += was_written
            reused += not was_written

    counts = {"records": len(records), "sent": len(images), "answered": 0, "photos": 0, "assessments": 0,
              "images_written": written, "images_reused": reused, "photos_created": 0, "assessments_created": 0}
    with _transaction(conn):
        cur = conn.cursor(row_factory=dict_row)
        # One store per session at a time; a concurrent call waits, then sees these rows.
        cur.execute("SELECT pg_advisory_xact_lock(hashtext('video_store:' || %s))", (session,))
        cur.execute("INSERT INTO video_sessions (session_id, state, storing_at) VALUES (%s, 'storing', NOW()) "
                    "ON CONFLICT (session_id) DO NOTHING", (session,))
        existing = {r["segment_id"]: r for r in cur.execute(
            "SELECT segment_id, photo_id, assessment_id FROM video_segments WHERE session_id = %s FOR UPDATE",
            (session,)).fetchall()}
        for rec in records:
            seg = rec["id"]
            old = existing.get(seg) or {}
            photo_id = old.get("photo_id")
            assessment_id = str(old["assessment_id"]) if old.get("assessment_id") is not None else None
            image_path, sha = images.get(seg, (None, None))
            answer = answers.get(seg)
            if answer is not None:
                photo_id, assessment_id, p_new, a_new = _photo_and_assessment(
                    cur, rec, answer, image_path, sha, photo_id, assessment_id)
                counts["answered"] += 1
                counts["photos"] += 1
                counts["assessments"] += 1
                counts["photos_created"] += p_new
                counts["assessments_created"] += a_new
            cur.execute(_UPSERT_SEGMENT, _segment_row(rec, image_path, sha, photo_id, assessment_id, answer))
    return counts


# ============================================================
# verify_session
# ============================================================

def verify_session(conn: psycopg.Connection, session_id: str, run_dir: Path) -> dict:
    """Is the session completely and correctly stored? (the precondition for cleanup)

    ok = one row per sidecar record (same ids), every sent segment has an image path,
    every stored image exists and hashes to its recorded SHA-256, and every answered
    row's assessment exists. Returns {ok, records, rows, answered_rows, missing_images[],
    bad_checksums[], problems[]} (problems: everything else, in words).
    """
    problems: list[str] = []
    try:
        record_ids = {r["id"] for r in load_records(Path(run_dir)) if r["session_id"] == session_id}
    except (FileNotFoundError, ValueError) as e:
        record_ids = set()
        problems.append(f"records: {e}")
    with conn.cursor(row_factory=dict_row) as cur:
        rows = cur.execute(
            "SELECT g.segment_id, g.status, g.image_path, g.image_sha256, g.assessment_id, "
            "       (a.id IS NOT NULL) AS assessment_exists "
            "FROM video_segments g LEFT JOIN assessments a ON a.id = g.assessment_id "
            "WHERE g.session_id = %s ORDER BY g.k", (session_id,)).fetchall()
    if not conn.autocommit:
        conn.commit()
    row_ids = {r["segment_id"] for r in rows}
    if record_ids - row_ids:
        problems.append(f"{len(record_ids - row_ids)} record(s) without a row, e.g. {sorted(record_ids - row_ids)[0]}")
    if row_ids - record_ids and record_ids:
        problems.append(f"{len(row_ids - record_ids)} row(s) without a record, e.g. {sorted(row_ids - record_ids)[0]}")

    missing, bad = [], []
    answered = 0
    for r in rows:
        if r["assessment_id"] is not None:
            if r["assessment_exists"]:
                answered += 1
            else:
                problems.append(f"{r['segment_id']}: assessment {r['assessment_id']} no longer exists")
        if r["status"] not in SENT_STATUSES and r["image_path"] is None:
            continue
        path = r["image_path"]
        if not path or not Path(path).is_file():
            missing.append({"segment_id": r["segment_id"], "image_path": path})
        elif _sha256_file(Path(path)) != r["image_sha256"]:
            bad.append({"segment_id": r["segment_id"], "image_path": path})
    ok = (bool(record_ids) and len(rows) == len(record_ids) and row_ids == record_ids
          and not missing and not bad and not problems)
    return {"ok": ok, "records": len(record_ids), "rows": len(rows), "answered_rows": answered,
            "missing_images": missing, "bad_checksums": bad, "problems": problems}


# ============================================================
# Session state (operator UI)
# ============================================================

def set_session_state(conn: psycopg.Connection, session_id: str, state: str, *, error: str | None = None,
                      metrics: dict | None = None, report: dict | None = None) -> None:
    """Record that a session entered `state` (creates the row on first use).

    Sets <state>_at to now. `error` replaces the stored one (None clears it, so a
    retried session does not keep showing an old failure); metrics / report replace
    the stored ones only when given.
    """
    if state not in SESSION_STATES:
        raise ValueError(f"state {state!r} is not one of {SESSION_STATES}")
    if not session_id:
        raise ValueError("session_id is empty")
    stamp = sql.Identifier(f"{state}_at")
    query = sql.SQL(
        "INSERT INTO video_sessions (session_id, state, error, metrics, report, {stamp}) "
        "VALUES (%s, %s, %s, %s, %s, NOW()) ON CONFLICT (session_id) DO UPDATE SET "
        "state = EXCLUDED.state, error = EXCLUDED.error, "
        "metrics = COALESCE(EXCLUDED.metrics, video_sessions.metrics), "
        "report = COALESCE(EXCLUDED.report, video_sessions.report), {stamp} = NOW()").format(stamp=stamp)
    with _transaction(conn):
        conn.execute(query, (session_id, state, error,
                             Jsonb(metrics) if metrics is not None else None,
                             Jsonb(report) if report is not None else None))


def list_sessions(conn: psycopg.Connection, limit: int = 50) -> list[dict]:
    """Most recently updated sessions first, JSON-ready (timestamps as ISO strings), each with
    its segment counts: segments, sent (send + audit), answered (linked to an assessment)."""
    limit = max(1, min(1000, int(limit)))
    row = conn.execute(
        "SELECT COALESCE(json_agg(t ORDER BY t.updated_at DESC), '[]'::json) FROM ("
        "  SELECT s.*, COUNT(g.segment_id) AS segments,"
        "         COUNT(*) FILTER (WHERE g.status IN ('send', 'audit')) AS sent,"
        "         COUNT(g.assessment_id) AS answered"
        "  FROM video_sessions s LEFT JOIN video_segments g ON g.session_id = s.session_id"
        "  GROUP BY s.session_id ORDER BY s.updated_at DESC LIMIT %s) t", (limit,)).fetchone()
    if not conn.autocommit:
        conn.commit()
    return row[0] if row else []
