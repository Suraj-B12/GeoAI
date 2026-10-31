"""
Tests for the local PostgreSQL backend and the video-pipeline store, against a REAL
local PostgreSQL (GeoAI_Video docs/HANDOFF_PLAN.md section 7).

What must hold, and how it is checked:
  1. Same schema as Supabase. scripts/setup_local_db.py builds a scratch database
     from db_schema.sql + migrations 000-006; every live table and view
     (scripts/tests/supabase_schema_2026-10-27.json, read from the live PostgREST
     OpenAPI) must exist locally with the same columns, types and defaults, and
     the only extra relations are migration 005's two tables (006 lives in schema
     `research`, outside `public`). Setup is idempotent.
  2. store_session stores a run: one video_segments row per record (all four
     statuses), images on disk under <root>/<session>/<id>.png with the record's
     SHA-256, a photos + assessments pair per answered segment with the VLM result.
     A re-run creates nothing new. The PipelineWorker's claim RPC never returns
     these rows, also not while the storing transaction is still open.
  3. verify_session catches a deleted image and a modified image.
  4. LocalDb returns what the Supabase client returns: the worker/dashboard
     functions of app/supabase_client.py, called with a LocalDb, give rows whose
     keys and JSON types are those of the live OpenAPI definitions. The real
     PipelineWorker processes a local row end to end with a fake classifier.
  5. The operator delete removes a local image and both rows, and refuses paths
     outside LOCAL_IMAGE_ROOT without deleting anything.

The fake sidecars follow geoai.segment.v1 (GeoAI_Video
docs/pipeline/13-segment-frames.md section 5) field for field; the PNGs are tiny.

Skips cleanly when DATABASE_URL is not set. Uses a scratch database
`geoai_test` on the same server (created and dropped here; never `geoai`).

Usage:
    venv/Scripts/python.exe scripts/tests/test_local_db.py
    venv/Scripts/python.exe -m pytest scripts/tests/test_local_db.py -q
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import math
import os
import shutil
import sys
import tempfile
import threading
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import pytest
except ImportError:          # standalone run (the folder's usual style)
    pytest = None

DSN = os.environ.get("DATABASE_URL")
LIVE_SCHEMA = Path(__file__).resolve().parent / "supabase_schema_2026-10-27.json"
TEST_DB = "geoai_test"
SESSION = "20261026_101500_s21"
NEW_TABLES = {"video_sessions", "video_segments"}

if not DSN and pytest is not None and __name__ != "__main__":
    pytest.skip("DATABASE_URL is not set: local PostgreSQL tests skipped", allow_module_level=True)

import httpx  # noqa: E402
from PIL import Image  # noqa: E402

if DSN:
    import psycopg  # noqa: E402
    from psycopg.rows import dict_row  # noqa: E402

    from app import video_store  # noqa: E402
    from app import supabase_client as sb  # noqa: E402
    from app.deletion import DeleteError, delete_everywhere  # noqa: E402
    from app.local_db import LocalDb  # noqa: E402
    from app.local_images import inside_root, local_image_path  # noqa: E402
    from scripts import setup_local_db  # noqa: E402


# ============================================================
# Fake geoai.segment.v1 run
# ============================================================

W, H = 48, 27   # tiny 16:9 frames


def png_bytes(k: int) -> bytes:
    """A distinct tiny PNG per segment (distinct SHA-256)."""
    img = Image.new("RGB", (W, H), ((40 * k) % 256, (90 + 17 * k) % 256, (200 - 13 * k) % 256))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def fake_record(session: str, k: int, status: str, sha: str | None) -> dict:
    """One record with every block of spec section 5 (values plausible, not computed)."""
    rid = f"{session}_seg{k:05d}"
    s0, s1 = 5.0 * k, 5.0 * k + 5.0
    lat0, lon0, m_per_deg = 12.9716, 77.5946, 111_320.0

    def ll(s):
        return {"lat": lat0 + s / m_per_deg, "lon": lon0}

    covered = status != "not_covered"
    sent = status in ("send", "audit")
    utc_ms = 1_792_995_300_000 + 1000.0 * k
    rec = {
        "id": rid, "kind": "segment", "schema": "geoai.segment.v1", "status": status, "session_id": session,
        "pipeline_version": "0.9.0", "config_sha256": "ab" * 32,
        "segment": {"k": k, "length_m": s1 - s0, "s0_m": s0, "s1_m": s1, "partial": False,
                    "chainage_start_km": 12.0 + s0 / 1000.0, "chainage_end_km": 12.0 + s1 / 1000.0,
                    "start": ll(s0), "end": ll(s1), "centre": ll(s0 + 2.5),
                    "footprint_lonlat": [[lon0 - 2e-5, lat0 + s0 / m_per_deg], [lon0 + 2e-5, lat0 + s0 / m_per_deg],
                                         [lon0 + 2e-5, lat0 + s1 / m_per_deg], [lon0 - 2e-5, lat0 + s1 / m_per_deg],
                                         [lon0 - 2e-5, lat0 + s0 / m_per_deg]]},
        "frame": None, "camera": None,
        "target": {**ll(s0 + 2.5), "bearing_deg": None, "distance_m": None, "sigma_m": 2.5},
        "capture": None,
        "quality": {"sharpness": None, "sharpness_rank": None, "n_candidates": 0, "blur_motion_px": None,
                    "blur_rotation_px": None, "blur_px": None, "gyro_rad_s": None, "reason": "gnss_gap"},
        "gate": None,
        "survey": {"road_id": "BLR-ORR-01", "road_name": "Outer Ring Road", "direction": "increasing_chainage",
                   "lane": "left", "start_chainage_km": 12.0, "operator": "test", "vehicle": "car",
                   "weather": "dry"},
        "device": {"manufacturer": "samsung", "model": "SM-G991B", "device_id": "test-device"},
        "geometry": {"source": "calibrated", "pitch_deg": -23.0, "roll_deg": 0.2, "height_m": 1.35},
        "flags": [],
    }
    if covered:
        rec["frame"] = {"file": f"frames/{rid}.png" if sent else None, "sha256": sha if sent else None,
                        "width": W, "height": H, "segment": 0, "frame_in_segment": 30 * k, "frame_global": 30 * k,
                        "t_ns": 5_000_000_000 + 33_333_333 * 30 * k, "utc_ms": utc_ms,
                        "utc_iso": "2026-10-26T04:55:00.000Z", "local_iso": "2026-10-26T10:25:00.000+05:30",
                        "rows_of_segment": [12.0, 26.0], "near_edge_ahead_m": 1.6, "far_edge_ahead_m": 6.6}
        rec["camera"] = {**ll(s0 - 2.0), "alt_ellipsoid_m": 860.0, "alt_msl_m": None, "heading_deg": 0.0,
                         "view_direction_deg": 0.0, "speed_mps": 8.3, "h_sigma_m": 2.5, "utc_ms": utc_ms}
        rec["target"].update({"bearing_deg": 0.0, "distance_m": 4.5})
        rec["capture"] = {"exposure_ns": 1_000_000, "iso": 100, "frame_duration_ns": 33_333_333,
                          "rs_skew_ns": 16_600_000, "focus_diopters": 0.337, "focus_mode": "manual",
                          "ois_mode": 0, "low_light": False}
        rec["quality"] = {"sharpness": 0.8 + 0.01 * k, "sharpness_rank": 1, "n_candidates": 9,
                          "blur_motion_px": 0.6, "blur_rotation_px": 0.3, "blur_px": 0.67, "gyro_rad_s": 0.01,
                          "reason": "sharpest"}
        rec["gate"] = {"mode": "gate", "decision": status, "decision_if_gated": status,
                       "reasons": ["probe_frac"] if status == "send" else [],
                       "scores": {"probe_frac": 0.2 if status == "send" else 0.0, "probe_max_logit": 1.1,
                                  "evidence": 1, "crack_mm": 0.0, "z_max": 3.2, "anomalous_patch_frac": 0.05,
                                  "n_patches": 420},
                       "thresholds": {"probe_frac": 0.1, "evidence": 2, "crack_mm": 300.0},
                       "audit_draw": 0.031 if status == "audit" else 0.6}
    return rec


# k -> status; k5 is a second send that the VLM answers with low confidence
PLAN = {0: "send", 1: "audit", 2: "skip", 3: "not_covered", 4: "send", 5: "send"}


def answer(stage1="Distress", types=("Potholes", "Alligator Cracking"), severity="High.", s1c=0.97, s2c=0.91,
           review=False) -> dict:
    """A /classify response (ClassificationResponse fields)."""
    distressed = stage1 == "Distress"
    return {"is_distressed": distressed, "stage1_label": stage1, "stage1_confidence": s1c,
            "distress_types": list(types) if distressed else [],
            "primary_distress_type": types[0] if distressed and types else None,
            "secondary_distress_types": list(types[1:]) if distressed else [],
            "severity": severity if distressed else "None", "description": "test answer",
            "stage2_confidence": s2c if distressed else 0.0, "stage2_confidence_primary": 0.88,
            "needs_expert_review": review, "processing_time_ms": 11234.5, "stage1_time_ms": 1200.0,
            "stage2_time_ms": 9800.0, "stage1_raw": stage1, "stage2_raw": "DISTRESS_TYPES: ...",
            "stage2_mode": "shadow", "condition_indicators": []}


def build_run(root: Path, session: str = SESSION) -> tuple[Path, dict]:
    """RUNS/<session>/ with sidecars for every status and frames for send/audit."""
    run = root / session
    (run / "sidecars").mkdir(parents=True)
    (run / "frames").mkdir()
    for k, status in PLAN.items():
        sha = None
        if status in ("send", "audit"):
            data = png_bytes(k)
            sha = hashlib.sha256(data).hexdigest()
            (run / "frames" / f"{session}_seg{k:05d}.png").write_bytes(data)
        rec = fake_record(session, k, status, sha)
        (run / "sidecars" / f"{rec['id']}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    answers = {f"{session}_seg00000": answer(),
               f"{session}_seg00001": answer(stage1="Normal", s1c=0.93),
               f"{session}_seg00005": answer(types=("Ravelling",), severity="medium", s2c=0.55, review=True)}
    return run, answers


# ============================================================
# Environment (scratch database + temporary folders)
# ============================================================

class Env:
    def __init__(self):
        self.dsn = setup_local_db.with_database(DSN, TEST_DB)
        self.tmp = Path(tempfile.mkdtemp(prefix="geoai_localdb_test_"))
        self.image_root = self.tmp / "images"
        self.runs = self.tmp / "RUNS"
        self.prev_root = os.environ.get("LOCAL_IMAGE_ROOT")
        os.environ["LOCAL_IMAGE_ROOT"] = str(self.image_root)
        setup_local_db.drop_database(self.dsn)
        self.first_setup = setup_local_db.setup(self.dsn)
        self.second_setup = setup_local_db.setup(self.dsn)
        self.conn = video_store.connect(self.dsn)
        self.run, self.answers = build_run(self.runs)

    def close(self):
        try:
            self.conn.close()
        finally:
            if self.prev_root is None:
                os.environ.pop("LOCAL_IMAGE_ROOT", None)
            else:
                os.environ["LOCAL_IMAGE_ROOT"] = self.prev_root
            setup_local_db.drop_database(self.dsn)
            shutil.rmtree(self.tmp, ignore_errors=True)

    def q(self, query, params=()):
        return self.conn.execute(query, params).fetchall()

    def count(self, table):
        return self.q(f"SELECT COUNT(*) FROM {table}")[0][0]


if pytest is not None and DSN:
    @pytest.fixture(scope="module")
    def env():
        e = Env()
        yield e
        e.close()


# ============================================================
# 1. Schema
# ============================================================

def _norm_default(v):
    """Column default as PostgREST shows it: 'pending'::text -> pending, false -> False."""
    if v is None:
        return None
    s = str(v)
    while "::" in s and not s.endswith(")"):
        s = s.rsplit("::", 1)[0]
    if s.startswith("'") and s.endswith("'"):
        return s[1:-1]
    if s in ("true", "false"):
        return s == "true"
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def test_setup_is_idempotent(env):
    n = len(setup_local_db.SCHEMA_FILES)          # 000, db_schema, 001-006
    assert n == 8
    assert [f["action"] for f in env.first_setup["files"]] == ["applied"] * n
    assert [f["action"] for f in env.second_setup["files"]] == ["already_applied"] * n
    # 005 is also safe to run twice as a whole file (the owner runs it in the Supabase SQL editor)
    body = (PROJECT_ROOT / "migrations" / "005_video_pipeline.sql").read_text(encoding="utf-8")
    with psycopg.connect(env.dsn, autocommit=True) as c:
        c.execute(body)
        # 006 (research.expert_review_log, outside `public`) too
        c.execute((PROJECT_ROOT / "migrations" / "006_expert_review_log.sql").read_text(encoding="utf-8"))
    assert env.count("video_sessions") == 0
    assert env.count("research.expert_review_log") == 0


def test_schema_matches_live_supabase(env):
    live = json.loads(LIVE_SCHEMA.read_text(encoding="utf-8"))
    defs = live["definitions"]
    rows = env.q("SELECT table_name, column_name, data_type, column_default FROM information_schema.columns "
                 "WHERE table_schema = 'public' ORDER BY table_name, ordinal_position")
    local: dict[str, dict] = {}
    for t, c, typ, default in rows:
        local.setdefault(t, {})[c] = (typ, default)
    assert set(local) == set(defs) | NEW_TABLES, \
        f"relations differ: missing {set(defs) - set(local)}, extra {set(local) - set(defs) - NEW_TABLES}"
    problems = []
    for table, d in defs.items():
        live_cols = d["properties"]
        if list(live_cols) != list(local[table]):
            problems.append(f"{table}: columns {list(local[table])} != live {list(live_cols)}")
            continue
        for col, p in live_cols.items():
            typ, default = local[table][col]
            if typ != p.get("format"):
                problems.append(f"{table}.{col}: type {typ} != live {p.get('format')}")
            if "default" in p and _norm_default(default) != p["default"]:
                problems.append(f"{table}.{col}: default {default!r} != live {p['default']!r}")
    assert not problems, "; ".join(problems)
    # The worker's two RPCs, with the live argument names
    for fn in ("claim_pending_assessments", "reset_stale_processing_assessments"):
        args = env.q("SELECT parameter_name FROM information_schema.parameters p JOIN information_schema.routines r "
                     "ON p.specific_name = r.specific_name WHERE r.routine_schema = 'public' AND r.routine_name = %s "
                     "AND p.parameter_mode = 'IN' ORDER BY p.ordinal_position", (fn,))
        assert [a[0] for a in args] == list(live["rpc"][f"/rpc/{fn}"]["properties"]), fn
    # The photos trigger that turns a photo into a pending assessment (migration 002)
    assert env.q("SELECT 1 FROM information_schema.triggers WHERE trigger_name = 'trg_sync_photo_to_assessment'")


# ============================================================
# 2. store_session
# ============================================================

def _claimable_now(dsn) -> list[str]:
    """What the PipelineWorker would claim, from a separate connection, rolled back."""
    with psycopg.connect(dsn) as c:
        ids = [str(r[0]) for r in c.execute("SELECT id FROM claim_pending_assessments(1000, 'test-probe')")]
        c.rollback()
    return ids


def test_store_session_insert_read_back(env):
    counts = video_store.store_session(env.conn, env.run, env.answers, image_root=env.image_root)
    assert counts == {"records": 6, "sent": 4, "answered": 3, "photos": 3, "assessments": 3,
                      "images_written": 4, "images_reused": 0, "photos_created": 3, "assessments_created": 3}, counts
    segs = {r[0]: r for r in env.q("SELECT segment_id, status, image_path, image_sha256, photo_id, assessment_id, "
                                   "record, vlm_response, utc, chainage_start_km FROM video_segments ORDER BY k")}
    assert len(segs) == 6
    for k, status in PLAN.items():
        sid, row = f"{SESSION}_seg{k:05d}", segs[f"{SESSION}_seg{k:05d}"]
        assert row[1] == status
        assert row[6]["id"] == sid and row[6]["schema"] == "geoai.segment.v1"
        if status in ("send", "audit"):
            p = Path(row[2])
            assert p.is_absolute() and p == (env.image_root / SESSION / f"{sid}.png").resolve()
            assert hashlib.sha256(p.read_bytes()).hexdigest() == row[3] == hashlib.sha256(png_bytes(k)).hexdigest()
        else:
            assert row[2] is None and row[3] is None
    assert segs[f"{SESSION}_seg00003"][8] is None                      # not covered: no UTC
    assert abs(segs[f"{SESSION}_seg00000"][8].timestamp() * 1000 - 1_792_995_300_000) < 1
    assert segs[f"{SESSION}_seg00004"][4] is None and segs[f"{SESSION}_seg00004"][7] is None  # sent, unanswered

    a = {r["road_segment_id"]: r for r in env.conn.cursor(row_factory=dict_row).execute(
        "SELECT a.*, p.image_url AS photo_url, p.address AS photo_address, p.captured_at, "
        "       p.latitude AS p_lat FROM assessments a JOIN photos p ON p.id = a.photo_id")}
    assert set(a) == {f"{SESSION}_seg00000", f"{SESSION}_seg00001", f"{SESSION}_seg00005"}
    d = a[f"{SESSION}_seg00000"]
    assert d["status"] == "classified" and d["stage1_label"] == "Distress" and d["is_distressed"] is True
    assert d["distress_types"] == ["Potholes", "Alligator Cracking"] and d["severity"] == "High"
    assert math.isclose(d["stage1_confidence"], 0.97, rel_tol=1e-6) and not d["needs_expert_review"]
    assert d["image_url"] == d["photo_url"] == segs[f"{SESSION}_seg00000"][2]
    assert d["photo_address"] == "Outer Ring Road km 12.000" and d["address"] == d["photo_address"]
    assert d["image_width"] == W and d["image_height"] == H and d["processed_at"] is not None
    assert d["raw_response"]["source"] == "video_pipeline"
    assert d["raw_response"]["segment_record"]["id"] == f"{SESSION}_seg00000"
    assert d["raw_response"]["image_sha256"] == segs[f"{SESSION}_seg00000"][3]
    assert d["raw_response"]["vlm_response"]["stage2_raw"] == "DISTRESS_TYPES: ..."
    assert "Qwen" in d["model_version"]
    assert abs(d["captured_at"].timestamp() * 1000 - 1_792_995_300_000) < 1
    assert math.isclose(d["p_lat"], 12.9716 + 2.5 / 111_320.0, rel_tol=0, abs_tol=1e-12)
    n = a[f"{SESSION}_seg00001"]                                   # Stage 1 Normal
    assert n["status"] == "classified" and n["distress_types"] == [] and n["severity"] == "None"
    assert n["is_distressed"] is False
    r = a[f"{SESSION}_seg00005"]                                   # low confidence
    assert r["status"] == "expert_review" and r["needs_expert_review"] is True and r["severity"] == "Medium"
    assert {x["status"] for x in a.values()} <= {"classified", "expert_review"}
    assert not set(_claimable_now(env.dsn)) & {str(x["id"]) for x in a.values()}


def test_store_session_rerun_is_idempotent(env):
    before = {t: env.count(t) for t in ("photos", "assessments", "video_segments", "video_sessions")}
    counts = video_store.store_session(env.conn, env.run, env.answers, image_root=env.image_root)
    assert counts["images_written"] == 0 and counts["images_reused"] == 4
    assert counts["photos_created"] == 0 and counts["assessments_created"] == 0 and counts["answered"] == 3
    assert {t: env.count(t) for t in before} == before
    # A later answer for the unanswered send segment adds exactly one pair
    more = dict(env.answers, **{f"{SESSION}_seg00004": answer(types=("Longitudinal Cracking",), severity="Low")})
    counts = video_store.store_session(env.conn, env.run, more, image_root=env.image_root)
    assert counts["photos_created"] == 1 and counts["assessments_created"] == 1 and counts["answered"] == 4
    assert env.count("photos") == before["photos"] + 1 and env.count("assessments") == before["assessments"] + 1
    # Rerun without that answer keeps it (never deletes)
    video_store.store_session(env.conn, env.run, env.answers, image_root=env.image_root)
    assert env.q("SELECT vlm_response IS NOT NULL, assessment_id IS NOT NULL FROM video_segments "
                 "WHERE segment_id = %s", (f"{SESSION}_seg00004",))[0] == (True, True)
    assert env.count("assessments") == before["assessments"] + 1
    assert not set(_claimable_now(env.dsn)) & {str(r[0]) for r in env.q(
        "SELECT assessment_id FROM video_segments WHERE assessment_id IS NOT NULL")}


def test_open_store_transaction_is_never_claimable(env):
    """The trigger makes a 'pending' assessment inside the store's transaction; the worker's
    claim, from another connection, must not see it before it is filled and committed."""
    with psycopg.connect(env.dsn) as c:
        pid = c.execute("INSERT INTO photos (image_url) VALUES ('C:/x/in_flight.png') RETURNING id").fetchone()[0]
        aid = str(c.execute("SELECT id FROM assessments WHERE photo_id = %s", (pid,)).fetchone()[0])
        claimed: list = []
        t = threading.Thread(target=lambda: claimed.extend(_claimable_now(env.dsn)))
        t.start()
        t.join(10)
        c.rollback()
    assert not t.is_alive() and aid not in claimed


def test_store_session_rejects_bad_input(env):
    bad = dict(env.answers, **{f"{SESSION}_seg00002": answer()})        # skip segment answered
    try:
        video_store.store_session(env.conn, env.run, bad, image_root=env.image_root)
        raise AssertionError("an answer for a skip segment was accepted")
    except ValueError as e:
        assert "skip" in str(e)
    bad = {f"{SESSION}_seg00000": {"detail": "rate limited"}}           # an error body, not an answer
    try:
        video_store.store_session(env.conn, env.run, bad, image_root=env.image_root)
        raise AssertionError("an error response was accepted as an answer")
    except ValueError as e:
        assert "stage1_label" in str(e)


# ============================================================
# 3. verify_session
# ============================================================

def test_verify_session_catches_missing_and_modified_images(env):
    v = video_store.verify_session(env.conn, SESSION, env.run)
    assert v["ok"] and v["records"] == 6 and v["rows"] == 6 and v["answered_rows"] == 4, v
    img = env.image_root / SESSION / f"{SESSION}_seg00001.png"
    data = img.read_bytes()
    img.unlink()
    v = video_store.verify_session(env.conn, SESSION, env.run)
    assert not v["ok"] and [m["segment_id"] for m in v["missing_images"]] == [f"{SESSION}_seg00001"]
    img.write_bytes(data[:-1] + bytes([data[-1] ^ 0xFF]))
    v = video_store.verify_session(env.conn, SESSION, env.run)
    assert not v["ok"] and [b["segment_id"] for b in v["bad_checksums"]] == [f"{SESSION}_seg00001"]
    assert not v["missing_images"]
    # A re-run repairs it from the run's frame (atomic rewrite, checksum-verified)
    counts = video_store.store_session(env.conn, env.run, env.answers, image_root=env.image_root)
    assert counts["images_written"] == 1 and counts["images_reused"] == 3
    assert video_store.verify_session(env.conn, SESSION, env.run)["ok"]
    # A frame whose bytes differ from its record is refused, and nothing is replaced
    frame = env.run / "frames" / f"{SESSION}_seg00001.png"
    img.unlink()
    frame.write_bytes(data + b"junk")
    try:
        video_store.store_session(env.conn, env.run, env.answers, image_root=env.image_root)
        raise AssertionError("a frame with the wrong checksum was stored")
    except ValueError as e:
        assert "SHA-256" in str(e)
    assert not img.exists() and not list(img.parent.glob("*.tmp"))
    frame.write_bytes(data)
    video_store.store_session(env.conn, env.run, env.answers, image_root=env.image_root)
    assert video_store.verify_session(env.conn, SESSION, env.run)["ok"]


def test_session_state_and_list(env):
    video_store.set_session_state(env.conn, SESSION, "classifying")
    video_store.set_session_state(env.conn, SESSION, "failed", error="VLM timeout at seg00004")
    s = video_store.list_sessions(env.conn)[0]
    assert s["state"] == "failed" and s["error"] == "VLM timeout at seg00004" and s["failed_at"]
    video_store.set_session_state(env.conn, SESSION, "stored", metrics={"route_m": 30.0},
                                  report={"mode": "segments"})
    video_store.set_session_state(env.conn, SESSION, "cleaned")
    s = video_store.list_sessions(env.conn, limit=5)[0]
    assert s["state"] == "cleaned" and s["error"] is None
    assert s["metrics"] == {"route_m": 30.0} and s["report"] == {"mode": "segments"}
    assert s["segments"] == 6 and s["sent"] == 4 and s["answered"] == 4
    assert s["classifying_at"] and s["stored_at"] and s["cleaned_at"] and isinstance(s["updated_at"], str)
    try:
        video_store.set_session_state(env.conn, SESSION, "done")
        raise AssertionError("unknown state accepted")
    except ValueError:
        pass


# ============================================================
# 4. LocalDb = Supabase client shapes; worker end to end
# ============================================================

_JSON_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool}


def _shape_problems(row: dict, definition: dict, name: str) -> list[str]:
    """Keys equal the live definition's; each non-null value has the JSON type PostgREST gives it."""
    props = definition["properties"]
    p = []
    if set(row) != set(props):
        p.append(f"{name}: keys differ, missing {set(props) - set(row)}, extra {set(row) - set(props)}")
    for k, v in row.items():
        typ = props.get(k, {}).get("type")
        if v is None or typ is None:
            continue
        if not isinstance(v, _JSON_TYPES[typ]) or (typ != "boolean" and isinstance(v, bool)):
            p.append(f"{name}.{k}: {type(v).__name__} is not JSON {typ}")
    return p


def _seed_pending(env, k: int = 9) -> tuple[int, str, Path]:
    """A RoadSide-style upload whose image is in the local store: photos insert -> pending assessment."""
    path = env.image_root / "uploads" / f"upload_{k}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png_bytes(k))
    pid = env.q("INSERT INTO photos (image_url, address, latitude, longitude) VALUES (%s, 'test', 12.9, 77.6) "
                "RETURNING id", (str(path.resolve()),))[0][0]
    aid = str(env.q("SELECT id FROM assessments WHERE photo_id = %s", (pid,))[0][0])
    return pid, aid, path


def test_local_db_matches_supabase_shapes(env):
    defs = json.loads(LIVE_SCHEMA.read_text(encoding="utf-8"))["definitions"]
    pid, aid, path = _seed_pending(env)

    async def run():
        db = LocalDb(env.dsn)
        await db.open()
        try:
            out = {}
            out["claimed"] = await sb.claim_pending_batch(db, 5, "test-worker")
            out["updated"] = await sb.update_assessment(db, aid, {
                "status": "classified", "stage1_label": "Distress", "stage1_confidence": 0.9,
                "is_distressed": True, "distress_types": ["Potholes"], "severity": "High",
                "stage2_confidence": 0.85, "raw_response": {"stage1_time_ms": 1000.0, "stage2_time_ms": 5000.0},
                "processed_at": "2026-10-27T08:00:00+00:00", "error_message": None})
            try:
                await sb.update_assessment(db, aid, {"no_such_column": 1})
                out["bad_column"] = None
            except sb.SupabaseError as e:
                out["bad_column"] = e.status_code
            try:
                await sb.update_assessment(db, "00000000-0000-0000-0000-000000000000", {"status": "done"})
                out["missing_row"] = None
            except sb.SupabaseError as e:
                out["missing_row"] = e.status_code
            out["stale"] = await sb.reset_stale_processing(db, 300)
            out["ws"] = await sb.upsert_worker_state(db, "test-worker", {
                "is_running": True, "last_heartbeat": "2026-10-27T08:00:00+00:00", "images_processed": 3,
                "avg_stage1_time_ms": 1200.5, "current_image_id": aid, "last_error": None})
            out["ws2"] = await sb.upsert_worker_state(db, "test-worker", {"images_processed": 4})
            out["ws_get"] = await sb.get_worker_state(db, "test-worker")
            out["ws_none"] = await sb.get_worker_state(db, "nobody")
            sb.invalidate_dashboard_cache()
            out["metrics"] = await sb.get_pipeline_metrics(db)
            out["dist"] = await sb.get_class_distribution(db)
            out["recent"] = await sb.get_recent_processed(db, 5)
            out["summary"] = await sb.dashboard_summary(db)
            out["list"] = await sb.list_assessments_for_dashboard(db, status="classified,expert_review", limit=3)
            out["list_all"] = await sb.list_assessments_for_dashboard(db, status="all", date_from="2020-01-01")
            out["by_id"] = await sb.get_assessment_by_id(db, aid)
            out["image"] = await sb.download_image_bytes(None, str(path.resolve()))
            out["reclassify"] = await sb.reset_for_reclassify(db, aid)
            out["reclaimed"] = await sb.claim_pending_batch(db, 5, "test-worker")
            sb.invalidate_dashboard_cache()
            return out
        finally:
            await db.aclose()

    out = asyncio.run(run())
    p = []
    assert len(out["claimed"]) == 1 and out["claimed"][0]["id"] == aid
    assert out["claimed"][0]["status"] == "processing" and out["claimed"][0]["claimed_by"] == "test-worker"
    p += _shape_problems(out["claimed"][0], defs["assessments"], "claim")
    p += _shape_problems(out["updated"], defs["assessments"], "update")
    assert out["updated"]["distress_types"] == ["Potholes"] and out["updated"]["stage1_confidence"] == 0.9
    assert out["updated"]["processed_at"] == "2026-10-27T08:00:00+00:00"   # PostgREST's rendering
    assert out["bad_column"] == 400 and out["missing_row"] == 404 and out["stale"] == 0
    p += _shape_problems(out["ws"], defs["worker_state"], "worker_state")
    assert out["ws2"]["images_processed"] == 4 and out["ws2"]["is_running"] is True   # merge, not replace
    assert out["ws_get"] == out["ws2"] and out["ws_none"] is None
    m = out["metrics"]
    missing = set(defs["pipeline_metrics"]["properties"]) - set(m)
    assert not missing, missing
    assert {"rejected_non_pavement_count", "confidence_window", "stage_times_ms"} <= set(m)
    assert set(m["stage_times_ms"]) == {"stage0", "stage1", "stage2", "n"}
    assert m["classified_count"] >= 4 and isinstance(m["total_count"], int)
    assert out["dist"] and set(out["dist"][0]) == {"distress_type", "count", "window"}
    assert out["recent"] and set(out["recent"][0]) == {"id", "status", "stage1_label", "stage1_confidence",
                                                        "distress_types", "severity", "stage2_confidence",
                                                        "processed_at", "error_message"}
    assert set(out["summary"]) == {"total", "by_status", "by_date"} and out["summary"]["total"] == env.count(
        "assessments")
    assert set(out["list"]) == {"items", "total", "limit", "offset"} and len(out["list"]["items"]) == 3
    assert out["list"]["total"] >= 4 and out["list_all"]["total"] == env.count("assessments")
    assert set(out["list"]["items"][0]) == {"id", "image_url", "address", "latitude", "longitude", "status",
                                            "stage1_label", "stage1_confidence", "distress_types", "severity",
                                            "stage2_confidence", "needs_expert_review", "created_at",
                                            "processed_at", "condition_indicators", "pavement_filter_decision"}
    assert out["by_id"]["id"] == aid and out["by_id"]["image_url"] == str(path.resolve())
    assert out["image"] == png_bytes(9)
    assert out["reclassify"] == {"reset": True, "id": aid}
    assert [r["id"] for r in out["reclaimed"]] == [aid]
    assert not p, "; ".join(p)


class FakeClassifier:
    """The four PavementClassifier methods the worker calls, with fixed answers."""

    def __init__(self):
        self.sizes = []

    def predict_is_pavement(self, img):
        self.sizes.append(img.size)
        return {"is_pavement": True, "decision": "yes", "raw": "Yes", "time_ms": 5.0}

    def predict_stage1(self, img):
        return {"stage1_label": "Distress", "stage1_confidence": 0.95, "is_distressed": True,
                "stage1_time_ms": 1100.0, "stage1_raw": "Distress"}

    def predict_stage2(self, img):
        return {"distress_types": ["Potholes"], "severity": "High.", "description": "fake",
                "stage2_confidence": 0.9, "stage2_time_ms": 4200.0, "stage2_raw": "DISTRESS_TYPES: Potholes"}

    def stage1_safety_check(self, img):
        return None


def test_worker_processes_local_row_end_to_end(env):
    try:
        from app.worker import PipelineWorker
    except Exception as e:  # the worker imports torch via app.model
        if pytest is not None:
            pytest.skip(f"app.worker not importable here: {e}")
        print(f"  [SKIP] worker import failed: {e}")
        return
    pid, aid, path = _seed_pending(env, k=11)

    async def run():
        db = LocalDb(env.dsn)
        await db.open()
        try:
            async with httpx.AsyncClient() as http:
                fake = FakeClassifier()
                w = PipelineWorker(supabase_client=db, image_client=http, classifier=fake, worker_id="test-e2e")
                rows = await sb.claim_pending_batch(db, 10, w.worker_id)
                for row in rows:
                    await w._process_one(row)
                return rows, fake, await sb.get_assessment_by_id(db, aid)
        finally:
            await db.aclose()

    rows, fake, row = asyncio.run(run())
    assert aid in [r["id"] for r in rows]
    assert (W, H) in fake.sizes                                 # the image was read from the local store
    assert row["status"] == "classified" and row["distress_types"] == ["Potholes"] and row["severity"] == "High"
    assert row["processed_at"] is not None and row["error_message"] is None


# ============================================================
# 5. Operator delete with local images
# ============================================================

def test_delete_everywhere_local(env):
    seg = f"{SESSION}_seg00000"
    aid, pid, img = env.q("SELECT assessment_id, photo_id, image_path FROM video_segments WHERE segment_id = %s",
                          (seg,))[0]
    aid = str(aid)
    outside_dir = env.tmp / "outside"
    outside_dir.mkdir()
    outside = outside_dir / "keep.png"
    outside.write_bytes(png_bytes(1))
    opid = env.q("INSERT INTO photos (image_url) VALUES (%s) RETURNING id", (str(outside),))[0][0]
    oaid = str(env.q("SELECT id FROM assessments WHERE photo_id = %s", (opid,))[0][0])

    async def run():
        db = LocalDb(env.dsn)
        await db.open()
        try:
            async with httpx.AsyncClient() as http:
                res = await delete_everywhere(db, http, aid)
                try:
                    await delete_everywhere(db, http, oaid)
                    refused = None
                except DeleteError as e:
                    refused = e.status_code
                return res, refused
        finally:
            await db.aclose()

    res, refused = asyncio.run(run())
    assert res["deleted"] and res["photo_deleted"] and res["cloudinary"] == []
    assert res["local_images"] == [{"url": img, "path": str(Path(img).resolve()), "result": "deleted"}]
    assert not Path(img).exists()
    assert not env.q("SELECT 1 FROM assessments WHERE id = %s", (aid,))
    assert not env.q("SELECT 1 FROM photos WHERE id = %s", (pid,))
    assert env.q("SELECT photo_id, assessment_id FROM video_segments WHERE segment_id = %s", (seg,))[0] == (None, None)
    # Outside LOCAL_IMAGE_ROOT: refused, nothing deleted
    assert refused == 409 and outside.exists() and env.q("SELECT 1 FROM assessments WHERE id = %s", (oaid,))
    # verify_session now reports the deleted image honestly
    v = video_store.verify_session(env.conn, SESSION, env.run)
    assert not v["ok"] and [m["segment_id"] for m in v["missing_images"]] == [seg]


def test_local_image_path_guards(env):
    assert local_image_path("https://res.cloudinary.com/x/image/upload/v1/a.jpg") is None
    assert local_image_path("relative/a.png") is None and local_image_path("") is None
    assert local_image_path(r"C:\GeoAI\images\s\a.png") is not None
    root = env.image_root
    assert inside_root(root / "s" / "a.png", root) == (root / "s" / "a.png").resolve()
    for bad in (root / ".." / "secret.png", root, Path(str(root) + "_evil") / "a.png"):
        try:
            inside_root(bad, root)
            raise AssertionError(f"{bad} accepted")
        except PermissionError:
            pass


# ============================================================
# Standalone runner (folder style)
# ============================================================

TESTS = [test_setup_is_idempotent, test_schema_matches_live_supabase, test_store_session_insert_read_back,
         test_store_session_rerun_is_idempotent, test_open_store_transaction_is_never_claimable,
         test_store_session_rejects_bad_input, test_verify_session_catches_missing_and_modified_images,
         test_session_state_and_list, test_local_db_matches_supabase_shapes,
         test_worker_processes_local_row_end_to_end, test_delete_everywhere_local, test_local_image_path_guards]


def main() -> int:
    if not DSN:
        print("SKIP: DATABASE_URL is not set (local PostgreSQL tests need it)")
        return 0
    env = Env()
    failures = []
    try:
        for t in TESTS:
            try:
                t(env)
                print(f"  [OK]   {t.__name__}")
            except Exception as e:
                failures.append(t.__name__)
                print(f"  [FAIL] {t.__name__}: {type(e).__name__}: {e}")
                traceback.print_exc()
    finally:
        env.close()
    print(f"\n{len(TESTS) - len(failures)}/{len(TESTS)} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
