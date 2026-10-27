"""
Tests for the read-only GIS API (app/gis_routes.py + app/gis_core.py), the offline mock
(scripts/gis_mock_server.py) and the sample export (scripts/export_gis_sample.py), against a
REAL local PostgreSQL.

Uses a scratch database `geoai_gis_test` on the DATABASE_URL server (built by
scripts/setup_local_db.py from db_schema.sql + migrations 000-005, dropped at the end; never
`geoai`), a temporary LOCAL_IMAGE_ROOT and preview cache. Two fake drive sessions are stored with
the real writer (app/video_store.store_session), covering every segment status and VLM class,
an expert correction, a clockwise and an unclosed footprint and a missing footprint; citizen
uploads are inserted through photos (the trigger makes their assessments).

What must hold:
  1. GeoJSON is valid RFC 7946: FeatureCollection/Feature, lon-lat order, closed counter-clockwise
     rings with >= 4 positions, bbox = the coordinates' extent, strict JSON (no NaN).
  2. Every filter returns exactly the segments an independent SQL query selects; paging is
     consistent; bad values are 400 with a reason; unknown parameters are 400.
  3. /gis/image serves only video_segments.image_path inside LOCAL_IMAGE_ROOT: unknown ids,
     traversal, paths outside the root, non-images and missing files are 404; previews have the
     snapped width.
  4. CSV: UTF-8 BOM, the documented header, one row per matching segment.
  5. /gis/video/summary and /gis/video/sessions equal independent SQL sums and counts; per-km rows
     add up to the session totals; the track has one point per segment centre.
  6. Citizen layer: local backend excludes the video assessments and expert_review unless asked;
     the Supabase path sends the documented PostgREST filters, pages by 1000 and caches.
  7. Database unavailable -> 503 (not 500); no request changes any table.
  8. The mock over an exported sample answers exactly as the API (same bodies but generated_at),
     and install() serves webgis/ at /gis/app/.

Expected values come from SQL written here, not from the code under test.

Usage:
    venv/Scripts/python.exe scripts/tests/test_gis_routes.py
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import traceback
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass

DSN = os.environ.get("DATABASE_URL")
TEST_DB = "geoai_gis_test"
SA, SB = "20261026T050000Z_SM-G991B-TEST_aaaa", "20261027T050000Z_SM-G991B-TEST_bbbb"

FAILURES: list[str] = []
CHECKS = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if cond:
        print(f"  [OK]   {name}")
    else:
        print(f"  [FAIL] {name}" + (f" - {detail}" if detail else ""))
        FAILURES.append(name)


# ============================================================
# Fake data
# ============================================================

W, H = 960, 540
LAT0, LON0, M_PER_DEG = 12.9716, 77.5946, 111_320.0


def png_bytes(k: int, session: str) -> bytes:
    from PIL import Image
    seed = sum(map(ord, session)) + 31 * k
    img = Image.new("RGB", (W, H), (seed % 256, (seed * 7) % 256, (seed * 13) % 256))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def ll(s: float, east: float = 0.0) -> dict:
    return {"lat": LAT0 + s / M_PER_DEG, "lon": LON0 + east}


def record(session: str, k: int, status: str, sha, *, utc_base: float, footprint="ccw", length=5.0,
           lon_shift=0.0) -> dict:
    rid = f"{session}_seg{k:05d}"
    s0, s1 = 5.0 * k, 5.0 * k + length
    e = 2e-5
    w0, w1 = LON0 + lon_shift - e, LON0 + lon_shift + e
    a, b = LAT0 + s0 / M_PER_DEG, LAT0 + s1 / M_PER_DEG
    ring = {"ccw": [[w0, a], [w1, a], [w1, b], [w0, b], [w0, a]],
            "cw": [[w0, a], [w0, b], [w1, b], [w1, a], [w0, a]],
            "open": [[w0, a], [w1, a], [w1, b], [w0, b]],
            None: None}[footprint]
    sent = status in ("send", "audit")
    covered = status != "not_covered"
    utc_ms = utc_base + 1000.0 * k
    rec = {
        "id": rid, "kind": "segment", "schema": "geoai.segment.v1", "status": status, "session_id": session,
        "pipeline_version": "0.1.0", "config_sha256": "ab" * 32,
        "segment": {"k": k, "length_m": length, "s0_m": s0, "s1_m": s1, "partial": length < 5.0,
                    "chainage_start_km": None, "chainage_end_km": None,
                    "start": ll(s0, lon_shift), "end": ll(s1, lon_shift), "centre": ll((s0 + s1) / 2, lon_shift),
                    "footprint_lonlat": ring},
        "frame": None, "camera": None,
        "target": {**ll((s0 + s1) / 2, lon_shift), "bearing_deg": 0.0, "distance_m": 4.5, "sigma_m": 6.0},
        "capture": None,
        "quality": {"sharpness": None, "sharpness_rank": None, "n_candidates": 0, "blur_motion_px": None,
                    "blur_rotation_px": None, "blur_px": None, "gyro_rad_s": None, "reason": "gnss_gap"},
        "gate": None,
        "survey": {"road_id": "TST-9", "road_name": " Test Road ", "direction": "increasing_chainage",
                   "lane": "L1", "start_chainage_km": None, "operator": "", "vehicle": "", "weather": "dry"},
        "device": {"manufacturer": "samsung", "model": "SM-G991B", "device_id": "SM-G991B-TEST"},
        "geometry": {"source": "nominal", "pitch_deg": 23.0, "roll_deg": 0.0, "height_m": 1.35},
        "flags": [],
    }
    if covered:
        rec["frame"] = {"file": f"frames/{rid}.png" if sent else None, "sha256": sha if sent else None,
                        "width": W, "height": H, "segment": 1, "frame_in_segment": 30 * k,
                        "frame_global": 30 * k, "t_ns": 10 ** 9 * k, "utc_ms": utc_ms,
                        "utc_iso": "x", "local_iso": "2026-10-26T10:30:00.000+05:30",
                        "rows_of_segment": [500.0, 1500.0], "near_edge_ahead_m": 2.0, "far_edge_ahead_m": 7.0}
        rec["camera"] = {**ll(s0 - 4.0, lon_shift), "alt_ellipsoid_m": 800.0, "alt_msl_m": None,
                         "heading_deg": 0.0, "view_direction_deg": 0.0, "speed_mps": 5.0, "h_sigma_m": 6.0,
                         "utc_ms": utc_ms}
        rec["capture"] = {"exposure_ns": 1_000_000, "iso": 100, "frame_duration_ns": 33_333_000,
                          "rs_skew_ns": 16_566_040, "focus_diopters": 0.46, "focus_mode": "LOCKED_MANUAL",
                          "ois_mode": 1, "low_light": False}
        rec["quality"] = {"sharpness": 0.05, "sharpness_rank": 1, "n_candidates": 9, "blur_motion_px": 0.3,
                          "blur_rotation_px": 1.0, "blur_px": 1.04, "gyro_rad_s": 0.2, "reason": "sharpest"}
        rec["gate"] = {"mode": "gate", "decision": status, "decision_if_gated": status,
                       "reasons": ["evidence"] if status == "send" else [],
                       "scores": {"probe_frac": 0.1, "probe_max_logit": 3.0, "evidence": 40.0, "crack_mm": 0.0,
                                  "z_max": 1.0, "anomalous_patch_frac": 0.01, "n_patches": 1000},
                       "thresholds": {"source": "config", "probe_frac": 0.05, "evidence": 30.0,
                                      "crack_mm": 300.0, "audit_frac": 0.1, "probe_seed_logit": 2.0},
                       "audit_draw": 0.5}
    return rec


def answer(stage1="Distress", types=(), severity="High", s1c=0.97, s2c=0.91, review=False) -> dict:
    d = stage1 == "Distress"
    return {"is_distressed": d, "stage1_label": stage1, "stage1_confidence": s1c,
            "distress_types": list(types) if d else [], "severity": severity if d else "None",
            "description": "test", "stage2_confidence": s2c if d else 0.0, "needs_expert_review": review,
            "processing_time_ms": 1000.0, "stage1_time_ms": 300.0, "stage2_time_ms": 700.0,
            "condition_indicators": []}


# session A: every status and class; B: another day, a partial last segment, shifted east
PLAN_A = {0: ("send", "ccw", answer(types=("Potholes", "Alligator Cracking"), severity="High", s2c=0.91)),
          1: ("audit", "ccw", answer(stage1="Normal", s1c=0.93)),
          2: ("skip", "ccw", None),
          3: ("not_covered", "ccw", None),
          4: ("send", "ccw", answer(types=("Ravelling",), severity="medium", s2c=0.55, review=True)),
          5: ("send", "ccw", None),
          6: ("send", "ccw", answer(types=(), severity="None", s1c=0.61, s2c=0.0, review=True)),
          7: ("send", "cw", answer(types=("Longitudinal Cracking",), severity="Low", s2c=0.85)),
          8: ("skip", None, None),
          9: ("send", "open", answer(types=("Bleeding", "Potholes"), severity="Low", s2c=0.72, review=True))}
PLAN_B = {0: ("send", "ccw", answer(types=("Potholes",), severity="Medium", s2c=0.88)),
          1: ("send", "ccw", answer(stage1="Normal", s1c=0.99)),
          2: ("not_covered", "ccw", None)}


def build_run(root: Path, session: str, plan: dict, utc_base: float, lon_shift: float) -> tuple[Path, dict]:
    run = root / session
    (run / "sidecars").mkdir(parents=True)
    (run / "frames").mkdir()
    answers = {}
    last = max(plan)
    for k, (status, fp, ans) in plan.items():
        sha = None
        if status in ("send", "audit"):
            data = png_bytes(k, session)
            sha = hashlib.sha256(data).hexdigest()
            (run / "frames" / f"{session}_seg{k:05d}.png").write_bytes(data)
        length = 2.6 if (session == SB and k == last) else 5.0
        rec = record(session, k, status, sha, utc_base=utc_base, footprint=fp, length=length, lon_shift=lon_shift)
        (run / "sidecars" / f"{rec['id']}.json").write_text(json.dumps(rec), encoding="utf-8")
        if ans is not None:
            answers[rec["id"]] = ans
    return run, answers


# ============================================================
# Environment
# ============================================================

class Env:
    def __init__(self):
        from scripts import setup_local_db
        from app import video_store
        self.setup_local_db = setup_local_db
        self.dsn = setup_local_db.with_database(DSN, TEST_DB)
        self.tmp = Path(tempfile.mkdtemp(prefix="geoai_gis_test_"))
        self.root = self.tmp / "images"
        self.saved_env = {k: os.environ.get(k) for k in ("DATABASE_URL", "LOCAL_IMAGE_ROOT", "GIS_PREVIEW_CACHE",
                                                         "GIS_APP_DIR")}
        os.environ["DATABASE_URL"] = self.dsn
        os.environ["LOCAL_IMAGE_ROOT"] = str(self.root)
        os.environ["GIS_PREVIEW_CACHE"] = str(self.tmp / "gis_cache")
        setup_local_db.drop_database(self.dsn)
        setup_local_db.setup(self.dsn)
        self.conn = video_store.connect(self.dsn)
        runs = self.tmp / "RUNS"
        for session, plan, base, shift in ((SA, PLAN_A, 1_792_990_800_000.0, 0.0),
                                           (SB, PLAN_B, 1_793_077_200_000.0, 0.001)):
            run, answers = build_run(runs, session, plan, base, shift)
            video_store.store_session(self.conn, run, answers, image_root=self.root)
            video_store.set_session_state(self.conn, session, "stored")
        # expert correction of A k4 (Ravelling -> Edge Breaking, High)
        self.q("UPDATE assessments SET expert_reviewed = true, expert_corrected_types = %s::jsonb, "
               "expert_corrected_severity = 'High', reviewer_name = 'test' WHERE road_segment_id = %s",
               ('["Edge Breaking"]', f"{SA}_seg00004"))
        # citizen uploads (trigger makes pending assessments), then classified / expert_review / pending
        self.citizen = {}
        for name, status, s1, types, sev, url in (
                ("c1", "classified", "Distress", '["Potholes"]', "High", "https://res.cloudinary.com/demo/image/upload/v1/c1.jpg"),
                ("c2", "expert_review", "Distress", '["Alligator Cracking"]', "Medium", "https://example.org/c2.jpg"),
                ("c3", "done", "Normal", "[]", "None", str(self.root / "citizen" / "c3.png")),
                ("c4", "pending", None, None, None, "https://example.org/c4.jpg")):
            pid = self.q("INSERT INTO photos (image_url, address, latitude, longitude) VALUES (%s, %s, %s, %s) "
                         "RETURNING id", (url, f"addr {name}", 12.97, 77.6))[0][0]
            if status != "pending":
                self.q("UPDATE assessments SET status = %s, stage1_label = %s, stage1_confidence = 0.9, "
                       "is_distressed = %s, distress_types = %s::jsonb, severity = %s, stage2_confidence = 0.82, "
                       "needs_expert_review = %s, processed_at = NOW() WHERE photo_id = %s",
                       (status, s1, s1 == "Distress", types, sev, status == "expert_review", pid))
            self.citizen[name] = str(self.q("SELECT id FROM assessments WHERE photo_id = %s", (pid,))[0][0])

    def q(self, query, params=()):
        cur = self.conn.execute(query, params)
        return cur.fetchall() if cur.description else []

    def ids(self, where: str, params=()) -> set:
        return {r[0] for r in self.q(FINAL_SQL + f" SELECT segment_id FROM f WHERE {where}", params)}

    def close(self):
        try:
            self.conn.close()
        finally:
            for k, v in self.saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            try:
                self.setup_local_db.drop_database(self.dsn)
            finally:
                shutil.rmtree(self.tmp, ignore_errors=True)


DISTRESS = ["Bleeding", "Streaking", "Hungry Surface", "Hairline Cracks", "Alligator Cracking",
            "Longitudinal Cracking", "Transverse Cracking", "Edge Cracking", "Slippage", "Rutting", "Corrugation",
            "Shoving", "Shallow Depression", "Settlement", "Stripping", "Ravelling", "Potholes", "Edge Breaking"]

# The independent definition of every derived field, in SQL.
FINAL_SQL = """
WITH b AS (
  SELECT g.*, a.id IS NOT NULL AS answered, a.status AS astatus, a.needs_expert_review AS nr,
         COALESCE(a.expert_reviewed, false) AS er, a.stage1_label, a.stage1_confidence, a.stage2_confidence,
         CASE WHEN a.id IS NULL THEN '[]'::jsonb
              WHEN jsonb_typeof(a.expert_corrected_types) = 'array' AND jsonb_array_length(a.expert_corrected_types) > 0
                   THEN a.expert_corrected_types
              WHEN a.stage1_label = 'Normal' THEN '["Normal"]'::jsonb
              ELSE a.distress_types END AS fl,
         COALESCE(NULLIF(a.expert_corrected_severity, ''), a.severity) AS fs
  FROM video_segments g LEFT JOIN assessments a ON a.id = g.assessment_id),
f AS (
  SELECT b.*, CASE WHEN stage1_label = 'Distress' AND stage2_confidence > 0 THEN stage2_confidence
                   ELSE stage1_confidence END AS conf,
         CASE WHEN status = 'not_covered' THEN 'not_covered' WHEN status = 'skip' THEN 'not_sent'
              WHEN NOT answered THEN 'pending'
              WHEN fl ?| ARRAY[""" + ",".join(f"'{x}'" for x in DISTRESS) + """] THEN 'distress'
              WHEN fl = '["Normal"]'::jsonb THEN 'normal' ELSE 'unknown' END AS cls
  FROM b)
"""


# ============================================================
# Helpers
# ============================================================

def make_app(client=None, with_install=False):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app import gis_routes
    app = FastAPI()
    if with_install:
        gis_routes.install(app)
    else:
        app.include_router(gis_routes.router)
    app.state.supabase_client = client
    gis_routes.reset_citizen_cache()
    return app, TestClient(app)


class LocalStub:
    IS_LOCAL_DB = True


def strict_json(text: str):
    def bad(c):
        raise ValueError(f"non-standard JSON constant {c}")
    return json.loads(text, parse_constant=bad)


def seg_ids(body) -> set:
    return {f["id"] for f in body["features"]}


def signed_area(ring) -> float:
    return sum(ring[i][0] * ring[i + 1][1] - ring[i + 1][0] * ring[i][1] for i in range(len(ring) - 1))


def table_fingerprint(env) -> dict:
    out = {}
    for t in ("video_segments", "video_sessions", "assessments", "photos"):
        # updated_at excluded: test_images edits and restores one row (the trigger bumps its updated_at)
        out[t] = env.q(f"SELECT md5(COALESCE(string_agg((to_jsonb(x) - 'updated_at')::text, '|' "
                       f"ORDER BY (to_jsonb(x) - 'updated_at')::text), '')) FROM {t} x")[0][0]
    return out


# ============================================================
# Tests
# ============================================================

def test_geojson_validity(env, tc):
    r = tc.get("/gis/video/segments.geojson")
    check("segments.geojson 200 + geo+json", r.status_code == 200 and r.headers["content-type"].startswith(
        "application/geo+json"), f"{r.status_code} {r.headers.get('content-type')}")
    body = strict_json(r.text)
    n = env.q("SELECT COUNT(*) FROM video_segments")[0][0]
    check("FeatureCollection with every segment", body["type"] == "FeatureCollection" and body["numberMatched"] == n
          and len(body["features"]) == n == 13, f"{body.get('numberMatched')} vs {n}")
    check("schema version", body["schema"] == "geoai.gis.v1")
    problems = []
    xs, ys = [], []
    for f in body["features"]:
        p, g = f["properties"], f["geometry"]
        if f["type"] != "Feature" or f["id"] != p["segment_id"]:
            problems.append(f"{f['id']}: type/id")
        if g["type"] == "Polygon":
            ring = g["coordinates"][0]
            if len(ring) < 4 or ring[0] != ring[-1]:
                problems.append(f"{f['id']}: ring not closed / too short")
            if signed_area(ring) <= 0:
                problems.append(f"{f['id']}: ring not counter-clockwise")
            pts = ring
        else:
            pts = [g["coordinates"]]
        for x, y in pts:
            xs.append(x)
            ys.append(y)
            if not (77.5 < x < 77.7 and 12.9 < y < 13.0):    # lon first: Bengaluru lon ~77.6, lat ~12.97
                problems.append(f"{f['id']}: [{x}, {y}] not lon,lat")
    check("Features: id, closed CCW rings, lon-lat order", not problems, "; ".join(problems[:5]))
    check("bbox member = extent of coordinates", body["bbox"] == [min(xs), min(ys), max(xs), max(ys)],
          f"{body['bbox']}")
    by = {f["id"]: f for f in body["features"]}
    cw = by[f"{SA}_seg00007"]["geometry"]["coordinates"][0]
    check("clockwise footprint re-oriented", signed_area(cw) > 0 and len(cw) == 5)
    op = by[f"{SA}_seg00009"]["geometry"]["coordinates"][0]
    check("unclosed footprint closed", op[0] == op[-1] and len(op) == 5)
    miss = by[f"{SA}_seg00008"]
    check("missing footprint -> Point at the centre",
          miss["geometry"]["type"] == "Point" and miss["geometry"]["coordinates"] ==
          [miss["properties"]["centre_lon"], miss["properties"]["centre_lat"]])
    pts = strict_json(tc.get("/gis/video/segments.geojson?geom=point").text)
    check("geom=point: every geometry a Point at the centre", all(
        f["geometry"]["type"] == "Point" and f["geometry"]["coordinates"] ==
        [f["properties"]["centre_lon"], f["properties"]["centre_lat"]] for f in pts["features"]))
    lines = strict_json(tc.get("/gis/video/segments.geojson?geom=line").text)
    check("geom=line: LineStrings start -> end", all(f["geometry"]["type"] == "LineString" and
                                                     len(f["geometry"]["coordinates"]) == 2
                                                     for f in lines["features"]))
    check("geom=bogus -> 400", tc.get("/gis/video/segments.geojson?geom=bogus").status_code == 400)

    # derived fields against the SQL definitions
    rows = {r[0]: r for r in env.q(FINAL_SQL + " SELECT segment_id, cls, fl, fs, answered, conf FROM f")}
    bad = []
    for f in body["features"]:
        p = f["properties"]
        sid, cls, fl, fs, answered, conf = rows[f["id"]]
        if p["vlm_class"] != cls or p["final_labels"] != list(fl) or p["answered"] != answered:
            bad.append(f"{sid}: {p['vlm_class']}/{cls} {p['final_labels']}/{fl}")
        if answered and (p["final_severity"] != fs or abs((p["confidence"] or 0) - (conf or 0)) > 1e-3):
            bad.append(f"{sid}: severity/confidence {p['final_severity']}/{fs} {p['confidence']}/{conf}")
        if p["has_image"] != (p["status"] in ("send", "audit")):
            bad.append(f"{sid}: has_image")
        if p["has_image"] and (p["thumb_url"] != f"/gis/image/{sid}?w=480" or p["image_url"] != f"/gis/image/{sid}"):
            bad.append(f"{sid}: urls")
    check("derived fields (vlm_class, final labels/severity, confidence, urls) = SQL definitions", not bad,
          "; ".join(bad[:4]))
    k4 = by[f"{SA}_seg00004"]["properties"]
    check("expert correction wins: final_labels / marker / severity",
          k4["labels"] == ["Ravelling"] and k4["final_labels"] == ["Edge Breaking"]
          and k4["marker_label"] == "Edge Breaking" and k4["final_severity"] == "High" and k4["expert_reviewed"])
    k9 = by[f"{SA}_seg00009"]["properties"]
    check("marker_label = priority, primary_label = model order",
          k9["primary_label"] == "Bleeding" and k9["marker_label"] == "Potholes")
    check("road name trimmed, partial flag", by[f"{SB}_seg00002"]["properties"]["road_name"] == "Test Road"
          and by[f"{SB}_seg00002"]["properties"]["partial"] is True)


def test_filters(env, tc):
    def api(q):
        r = tc.get(f"/gis/video/segments.geojson?{q}")
        return r.status_code, (seg_ids(r.json()) if r.status_code == 200 else r.json().get("detail"))

    cases = [
        ("session=" + SB, "session_id = %s", (SB,)),
        (f"session={SA},{SB}", "true", ()),
        ("label=Potholes", "fl ? 'Potholes'", ()),
        ("label=potholes,NORMAL", "fl ?| ARRAY['Potholes','Normal']", ()),
        ("label=Ravelling", "fl ? 'Ravelling'", ()),            # corrected away: none
        ("label=Edge Breaking", "fl ? 'Edge Breaking'", ()),
        ("severity=High", "fs = 'High'", ()),
        ("severity=Low,Medium", "fs IN ('Low','Medium')", ()),
        ("status=skip,not_covered", "status IN ('skip','not_covered')", ()),
        ("vlm_class=pending", "cls = 'pending'", ()),
        ("vlm_class=unknown,normal", "cls IN ('unknown','normal')", ()),
        ("answered=true", "answered", ()),
        ("answered=false", "NOT answered", ()),
        ("reviewed=true", "er", ()),
        ("needs_review=true", "COALESCE(nr, false)", ()),
        ("needs_review=false", "NOT COALESCE(nr, false)", ()),
        ("min_confidence=0.8", "conf >= 0.8", ()),
        ("date_from=2026-10-27", "utc >= '2026-10-27T00:00:00Z'", ()),
        ("date_to=2026-10-26", "utc < '2026-10-27T00:00:00Z'", ()),
        ("date_from=2026-10-26T05:00:03Z&date_to=2026-10-26T05:00:07Z",
         "utc >= '2026-10-26T05:00:03Z' AND utc < '2026-10-26T05:00:07Z'", ()),
        ("bbox=77.5945,12.9716,77.5947,12.97175", "centre_lon BETWEEN 77.5945 AND 77.5947 AND "
                                                  "centre_lat BETWEEN 12.9716 AND 12.97175", ()),
        ("label=Potholes&severity=Low&session=" + SA, "fl ? 'Potholes' AND fs = 'Low' AND session_id = %s", (SA,)),
    ]
    for q, where, params in cases:
        status, got = api(q)
        want = env.ids(where, params)
        check(f"filter {q}", status == 200 and got == want, f"{status} got {sorted(got) if status == 200 else got} "
                                                             f"want {sorted(want)}")
    # paging
    all_ids = [f["id"] for f in tc.get("/gis/video/segments.geojson?geom=point").json()["features"]]
    p1 = tc.get("/gis/video/segments.geojson?geom=point&limit=5").json()
    p2 = tc.get("/gis/video/segments.geojson?geom=point&limit=5&offset=5").json()
    p3 = tc.get("/gis/video/segments.geojson?geom=point&limit=5&offset=10").json()
    check("paging: pages concatenate to the full ordered list",
          [f["id"] for f in p1["features"] + p2["features"] + p3["features"]] == all_ids
          and p1["numberMatched"] == len(all_ids) and p3["numberReturned"] == len(all_ids) - 10)
    check("order = session, k", all_ids == [r[0] for r in env.q(
        "SELECT segment_id FROM video_segments ORDER BY session_id, k")])
    check("filters echoed", p1["filters"] == {"limit": 5, "offset": 0})
    for q, frag in (("label=Pothole", "label='Pothole'"), ("severity=severe", "severity"), ("status=sent", "status"),
                    ("bbox=1,2,3", "four numbers"), ("bbox=77.6,12.9,77.5,13.0", "minLon<=maxLon"),
                    ("min_confidence=1.5", "0..1"), ("date_from=yesterday", "ISO"), ("limit=0", "limit"),
                    ("limit=20001", "limit"), ("reviewed=maybe", "true/false"), ("frobnicate=1", "unknown parameter"),
                    ("date_from=2026-10-27&date_to=2026-10-26", "before"), ("include_review=true", "unknown")):
        r = tc.get(f"/gis/video/segments.geojson?{q}")
        check(f"400 for {q}", r.status_code == 400 and frag in r.json().get("detail", ""), f"{r.status_code} {r.text}")
    check("'_' parameters ignored (cache busters)", tc.get("/gis/video/segments.geojson?_=123").status_code == 200)


def test_images(env, tc):
    from PIL import Image
    sid = f"{SA}_seg00000"
    path = env.q("SELECT image_path FROM video_segments WHERE segment_id = %s", (sid,))[0][0]
    r = tc.get(f"/gis/image/{sid}")
    check("full image = the stored PNG bytes", r.status_code == 200 and r.content == Path(path).read_bytes()
          and r.headers["content-type"] == "image/png")
    for w, want in ((480, 480), (500, 640), (5000, 1920), (100, 160)):
        r = tc.get(f"/gis/image/{sid}?w={w}")
        ok = r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
        width = Image.open(io.BytesIO(r.content)).width if ok else None
        check(f"preview w={w} -> JPEG {min(want, W)} px wide", ok and width == min(want, W), f"{r.status_code} {width}")
    check("preview cached on disk by sha + width",
          any(p.name.endswith("_w480.jpg") for p in (env.tmp / "gis_cache").glob("*.jpg")))
    check("w=0 -> 400", tc.get(f"/gis/image/{sid}?w=0").status_code == 400)
    check("unknown segment -> 404", tc.get("/gis/image/nope_seg00001").status_code == 404)
    check("skip segment (no image) -> 404", tc.get(f"/gis/image/{SA}_seg00002").status_code == 404)
    check("traversal id -> 404", tc.get("/gis/image/..%5C..%5Cwindows%5Cwin.ini").status_code == 404
          and tc.get("/gis/image/%2E%2E").status_code == 404)
    # rows pointing outside the root / at a non-image / at a missing file are refused
    victim = f"{SA}_seg00001"
    old = env.q("SELECT image_path FROM video_segments WHERE segment_id = %s", (victim,))[0][0]
    outside = env.tmp / "outside.png"
    outside.write_bytes(png_bytes(1, "outside"))
    notimg = env.root / SA / "notes.txt"
    notimg.write_text("x")
    try:
        for p, name in ((str(outside), "outside LOCAL_IMAGE_ROOT"), (str(env.root / ".." / "outside.png"), "'..' out"),
                        (str(notimg), "non-image inside root"), (str(env.root / SA / "gone.png"), "missing file"),
                        ("https://example.org/x.png", "a URL")):
            env.q("UPDATE video_segments SET image_path = %s WHERE segment_id = %s", (p, victim))
            check(f"image_path {name} -> 404", tc.get(f"/gis/image/{victim}").status_code == 404
                  and tc.get(f"/gis/image/{victim}?w=480").status_code == 404)
    finally:
        env.q("UPDATE video_segments SET image_path = %s WHERE segment_id = %s", (old, victim))
    check("restored row serves again", tc.get(f"/gis/image/{victim}?w=160").status_code == 200)


def test_csv(env, tc):
    from app.gis_core import CSV_COLUMNS
    r = tc.get("/gis/video/segments.csv?label=Potholes")
    raw = r.content
    check("CSV 200, text/csv, attachment", r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
          and "attachment" in r.headers.get("content-disposition", ""))
    check("CSV starts with the UTF-8 BOM", raw.startswith(b"\xef\xbb\xbf"))
    import csv
    rows = list(csv.reader(io.StringIO(raw.decode("utf-8-sig"))))
    check("CSV header = documented columns", tuple(rows[0]) == CSV_COLUMNS)
    want = env.ids("fl ? 'Potholes'")
    check("one CSV row per matching segment", {r_[0] for r_ in rows[1:]} == want and len(rows) - 1 == len(want))
    row = dict(zip(rows[0], rows[1]))
    feat = next(f for f in tc.get("/gis/video/segments.geojson?label=Potholes").json()["features"]
                if f["id"] == row["segment_id"])["properties"]
    check("CSV values = GeoJSON properties (lists '; ', booleans true/false)",
          row["final_labels"] == "; ".join(feat["final_labels"]) and row["answered"] == "true"
          and row["centre_lat"] == str(feat["centre_lat"]) and row["image_url"] == feat["image_url"])


def test_summary_and_sessions(env, tc):
    s = tc.get("/gis/video/summary").json()
    q = lambda sql_, p=(): env.q(FINAL_SQL + sql_, p)[0][0]   # noqa: E731
    check("summary.segments", s["segments"] == q(" SELECT COUNT(*) FROM f"))
    for st in ("send", "audit", "skip", "not_covered"):
        check(f"summary.segments_by_status.{st}", s["segments_by_status"][st] ==
              q(" SELECT COUNT(*) FROM f WHERE status = %s", (st,)))
    check("summary.answered", s["answered"] == q(" SELECT COUNT(*) FROM f WHERE answered"))
    check("summary.needs_expert_review", s["needs_expert_review"] == q(" SELECT COUNT(*) FROM f WHERE nr"))
    check("summary.expert_reviewed", s["expert_reviewed"] == q(" SELECT COUNT(*) FROM f WHERE er"))
    for key, where in (("route_m", "true"), ("surveyed_m", "status <> 'not_covered'"),
                       ("not_covered_m", "status = 'not_covered'"),
                       ("classified_m", "cls IN ('distress','normal')"), ("distress_m", "cls = 'distress'"),
                       ("pending_m", "cls = 'pending'"), ("not_sent_m", "cls = 'not_sent'")):
        want = float(q(f" SELECT COALESCE(SUM(length_m), 0) FROM f WHERE {where}"))
        check(f"summary.{key} = SQL sum ({want})", abs(s[key] - want) < 0.01, f"{s[key]}")
    labs = {x["label"]: x for x in s["labels"]}
    for lab in ("Potholes", "Normal", "Edge Breaking", "Ravelling", "Bleeding"):
        n = q(" SELECT COUNT(*) FROM f WHERE fl ? %s AND cls IN ('distress','normal')", (lab,))
        m = float(q(" SELECT COALESCE(SUM(length_m),0) FROM f WHERE fl ? %s AND cls IN ('distress','normal')", (lab,)))
        sv = float(q(" SELECT SUM(length_m) FROM f WHERE status <> 'not_covered'"))
        check(f"summary label {lab}: segments {n}, {m} m, % of surveyed",
              labs[lab]["segments"] == n and abs(labs[lab]["length_m"] - m) < 0.01
              and abs((labs[lab]["pct_of_surveyed"] or 0) - 100 * m / sv) < 0.001, json.dumps(labs[lab]))
    sp = tc.get("/gis/video/summary?label=Potholes&limit=1").json()
    check("summary honours filters and ignores limit", sp["segments"] == len(env.ids("fl ? 'Potholes'")))

    body = tc.get("/gis/video/sessions?track=true").json()
    check("sessions: both, ordered", [x["session_id"] for x in body["sessions"]] == [SA, SB] and body["count"] == 2)
    for x in body["sessions"]:
        sid = x["session_id"]
        n = env.q("SELECT COUNT(*), SUM(length_m) FROM video_segments WHERE session_id = %s", (sid,))[0]
        check(f"{sid[-4:]}: segments + route_m = SQL", x["segments"] == n[0] and abs(x["route_m"] - n[1]) < 0.01)
        km = x["per_km"]
        check(f"{sid[-4:]}: per_km sums = session totals", abs(sum(r["surveyed_m"] for r in km) - x["surveyed_m"]) < 0.01
              and abs(sum(r["distress_m"] for r in km) - x["distress_m"]) < 0.01
              and abs(sum(r["not_covered_m"] for r in km) - x["not_covered_m"]) < 0.01)
        cnt = env.q("SELECT COUNT(*) FROM video_segments WHERE session_id = %s AND centre_lat IS NOT NULL", (sid,))[0][0]
        check(f"{sid[-4:]}: track LineString through every centre",
              x["track"]["type"] == "LineString" and len(x["track"]["coordinates"]) == cnt)
        check(f"{sid[-4:]}: state, road, times", x["state"] == "stored" and x["road_name"] == "Test Road"
              and x["start_utc"] < x["end_utc"])
    one = tc.get(f"/gis/video/sessions?session={SB}&per_km=false").json()
    check("sessions?session=&per_km=false", one["count"] == 1 and "per_km" not in one["sessions"][0]
          and "track" not in one["sessions"][0])
    check("sessions?bogus -> 400", tc.get("/gis/video/sessions?bogus=1").status_code == 400)

    meta = tc.get("/gis/meta").json()
    names = [x["name"] for x in meta["labels"]]
    check("meta: 18 IRC labels + Normal, each with colour and category", len(names) == 19 and names[-1] == "Normal"
          and all(x["color"].startswith("#") and x["category"] for x in meta["labels"]))
    check("meta: video counts", meta["video"]["segments"] == 13 and meta["video"]["sessions"] == 2
          and meta["video"]["with_image"] == env.q("SELECT COUNT(*) FROM video_segments WHERE image_path IS NOT NULL")[0][0])


def test_citizen(env):
    from app import gis_routes
    _, tc = make_app(None)
    check("citizen without a backend -> 503", tc.get("/gis/citizen/assessments.geojson").status_code == 503)

    _, tc = make_app(LocalStub())
    body = tc.get("/gis/citizen/assessments.geojson").json()
    got = {f["id"] for f in body["features"]}
    check("local: classified + done only, never video rows", got == {env.citizen["c1"], env.citizen["c3"]},
          f"{got}")
    body = tc.get("/gis/citizen/assessments.geojson?include_review=true").json()
    got = {f["id"] for f in body["features"]}
    check("local: include_review adds expert_review (pending never)",
          got == {env.citizen["c1"], env.citizen["c2"], env.citizen["c3"]}, f"{got}")
    by = {f["id"]: f["properties"] for f in body["features"]}
    c1, c3 = by[env.citizen["c1"]], by[env.citizen["c3"]]
    check("citizen properties share the video semantics", c1["final_labels"] == ["Potholes"]
          and c1["marker_label"] == "Potholes" and c1["final_severity"] == "High" and c1["vlm_class"] == "distress"
          and c3["final_labels"] == ["Normal"] and c3["vlm_class"] == "normal")
    check("Cloudinary thumb/preview are transformations", "/upload/w_480,c_limit" in c1["thumb_url"]
          and "/upload/w_1280,c_limit" in c1["preview_url"])
    check("local citizen image -> /local-images URL", c3["image_url"].startswith("/local-images?path="))
    check("citizen filters (label, severity)", {f["id"] for f in tc.get(
        "/gis/citizen/assessments.geojson?include_review=true&label=Alligator Cracking&severity=Medium").json()[
        "features"]} == {env.citizen["c2"]})
    check("citizen rejects video-only filters", tc.get("/gis/citizen/assessments.geojson?status=send").status_code == 400)

    # Supabase path: documented PostgREST filters, 1000-row pages, cached
    import httpx
    calls = []
    rows = [{"id": f"00000000-0000-0000-0000-{i:012d}", "photo_id": i, "image_url": f"https://example.org/{i}.jpg",
             "latitude": 12.97, "longitude": 77.6, "address": "a", "status": "classified" if i % 2 else "expert_review",
             "stage1_label": "Distress", "stage1_confidence": 0.9, "distress_types": ["Potholes"], "severity": "Low",
             "description": "", "stage2_confidence": 0.8, "needs_expert_review": False, "expert_reviewed": False,
             "expert_corrected_types": None, "expert_corrected_severity": None, "reviewed_at": None,
             "created_at": "2026-09-23T13:31:41.969151+00:00", "processed_at": None} for i in range(1005)]

    def handler(request):
        calls.append(request.url)
        off, lim = int(request.url.params["offset"]), int(request.url.params["limit"])
        return httpx.Response(200, json=rows[off:off + lim])
    client = httpx.AsyncClient(base_url="https://x.supabase.co", transport=httpx.MockTransport(handler))
    _, tc = make_app(client)
    body = tc.get("/gis/citizen/assessments.geojson?include_review=true&limit=20000").json()
    p = calls[0].params if calls else {}
    check("supabase: status / road_segment_id / coordinates filters sent",
          p.get("status") == "in.(classified,done,expert_review)" and p.get("road_segment_id") == "is.null"
          and p.get("latitude") == "not.is.null" and "address" in p.get("select", ""))
    check("supabase: paged by 1000", [c.params["offset"] for c in calls] == ["0", "1000"]
          and body["numberMatched"] == 1005 and body["backend"] == "supabase")
    check("supabase: created_at normalised to Z", body["features"][0]["properties"]["created_at"] ==
          "2026-09-23T13:31:41.969Z")
    tc.get("/gis/citizen/assessments.geojson")
    check("supabase: rows cached (no second fetch within the TTL)", len(calls) == 2)
    body = tc.get("/gis/citizen/assessments.geojson").json()
    check("default excludes expert_review", body["numberMatched"] == 502)

    def failing(request):
        return httpx.Response(500, json={"message": "boom"})
    gis_routes.reset_citizen_cache()
    _, tc = make_app(httpx.AsyncClient(base_url="https://x.supabase.co", transport=httpx.MockTransport(failing)))
    check("supabase error -> 502", tc.get("/gis/citizen/assessments.geojson").status_code == 502)
    gis_routes.reset_citizen_cache()


def test_unavailable_and_read_only(env, before):
    after = table_fingerprint(env)
    check("no request changed any table (content fingerprints equal)", before == after,
          f"{[t for t in before if before[t] != after[t]]}")
    _, tc = make_app(None)
    saved = os.environ["DATABASE_URL"]
    try:
        from psycopg.conninfo import conninfo_to_dict, make_conninfo
        p = conninfo_to_dict(saved)
        p["port"] = "1"
        os.environ["DATABASE_URL"] = make_conninfo(**p)
        for u in ("/gis/video/segments.geojson", "/gis/video/summary", "/gis/video/sessions", "/gis/meta",
                  f"/gis/image/{SA}_seg00000", "/gis/video/segments.csv"):
            r = tc.get(u)
            check(f"database down -> 503 {u}", r.status_code == 503 and "unavailable" in r.json()["detail"],
                  f"{r.status_code} {r.text[:120]}")
        os.environ.pop("DATABASE_URL")
        r = tc.get("/gis/video/segments.geojson")
        check("DATABASE_URL unset -> 503", r.status_code == 503)
        check("503 bodies never contain the DSN", saved not in r.text and "password" not in r.text.lower())
    finally:
        os.environ["DATABASE_URL"] = saved


def test_mock_parity_and_install(env, tc):
    import export_gis_sample as exp
    import gis_mock_server as mock
    out = env.tmp / "sample"
    res = exp.export(out, env.tmp / "sample.zip", full_pngs=3, budget_mb=50, client=LocalStub(),
                     citizen_addresses=True, log=lambda *_: None)
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    check("export: bundle with manifest, schema.sql, mock, core",
          all((out / n).is_file() for n in ("manifest.json", "schema.sql", "mock_server.py", "gis_core.py", "README.md"))
          and res["zip"]["entries"] == len(m["files"]) + 1)
    check("export: every stored image has w480 and w1280 previews, 3 full PNGs",
          len(list((out / "images" / "w480").glob("*.jpg"))) == res["images"] == 9
          and len(list((out / "images" / "w1280").glob("*.jpg"))) == 9
          and len(list((out / "images" / "full").glob("*.png"))) == 3)
    check("export: manifest hashes match the files", all(
        hashlib.sha256((out / f["path"]).read_bytes()).hexdigest() == f["sha256"] for f in m["files"][:50]))
    check("export: schema.sql has the video tables and no password",
          "CREATE TABLE public.video_segments" in (out / "schema.sql").read_text(encoding="utf-8")
          and "PASSWORD" not in (out / "schema.sql").read_text(encoding="utf-8").upper())
    seg = json.loads((out / "api" / "gis" / "video" / "segments.geojson").read_text(encoding="utf-8"))
    check("export: saved segments = the API's full response",
          seg["features"] == tc.get("/gis/video/segments.geojson?limit=20000").json()["features"])
    res2 = exp.export(out, None, full_pngs=3, budget_mb=50, client=LocalStub(), citizen_addresses=True,
                      log=lambda *_: None)
    m2 = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    same = {f["path"]: f["sha256"] for f in m["files"] if f["path"].startswith("images/")} == \
        {f["path"]: f["sha256"] for f in m2["files"] if f["path"].startswith("images/")}
    check("export is idempotent (same images, no staging left)", same and res2["files"] == res["files"]
          and not (env.tmp / "sample.staging").exists())

    httpd = mock.serve(out, "127.0.0.1", 0)
    port = httpd.server_address[1]
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    _, ctc = make_app(LocalStub())

    def mget(u):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}{u}") as r:
                return r.status, r.read(), r.headers
        except urllib.error.HTTPError as e:
            return e.code, e.read(), e.headers

    def strip(b):
        d = json.loads(b)
        d.pop("generated_at", None)
        return d

    try:
        for u in ("/gis/video/segments.geojson", "/gis/video/segments.geojson?geom=point&label=Potholes",
                  "/gis/video/segments.geojson?geom=line&severity=Low,High&limit=2&offset=1",
                  "/gis/video/segments.geojson?vlm_class=pending,not_sent&bbox=77.59,12.97,77.6,12.98",
                  f"/gis/video/segments.geojson?session={SB}&date_from=2026-10-27",
                  "/gis/video/summary", "/gis/video/summary?label=Potholes&needs_review=true",
                  f"/gis/video/sessions?session={SA}", "/gis/video/sessions?track=true",
                  "/gis/citizen/assessments.geojson", "/gis/citizen/assessments.geojson?include_review=true&label=Potholes",
                  "/gis/video/segments.geojson?label=Pothole", "/gis/video/segments.geojson?nope=1"):
            st, body, _ = mget(u)
            r = ctc.get(u)
            check(f"mock = API: {u}", st == r.status_code and (strip(body) == strip(r.content) if st == 200
                                                               else json.loads(body) == r.json()),
                  f"mock {st} api {r.status_code}")
        st, body, hdr = mget("/gis/video/segments.csv?label=Potholes")
        check("mock = API: CSV bytes", st == 200 and body == ctc.get("/gis/video/segments.csv?label=Potholes").content)
        sid = f"{SA}_seg00000"
        st, body, _ = mget(f"/gis/image/{sid}?w=480")
        check("mock = API: w=480 preview bytes", st == 200 and body == ctc.get(f"/gis/image/{sid}?w=480").content)
        st, body, _ = mget(f"/gis/image/{sid}?w=1280")
        check("mock = API: w=1280 preview bytes", st == 200 and body == ctc.get(f"/gis/image/{sid}?w=1280").content)
        full = m["full_pngs"][0]
        st, body, hdr = mget(f"/gis/image/{full}")
        check("mock: full PNG for the sampled ones", st == 200 and body == ctc.get(f"/gis/image/{full}").content)
        st, _, hdr = mget(f"/gis/image/{SA}_seg00002?w=480")
        check("mock: no image -> 404", st == 404)
        st, _, hdr = mget("/gis/meta")
        check("mock: CORS header", hdr.get("Access-Control-Allow-Origin") == "*")
    finally:
        httpd.shutdown()
        httpd.server_close()

    app_dir = env.tmp / "webgis"
    app_dir.mkdir()
    (app_dir / "index.html").write_text("<!doctype html><title>gis</title>", encoding="utf-8")
    os.environ["GIS_APP_DIR"] = str(app_dir)
    try:
        _, itc = make_app(None, with_install=True)
        r = itc.get("/gis/app/")
        check("install(): webgis/ served at /gis/app/", r.status_code == 200 and "<title>gis</title>" in r.text)
        check("install(): API still reachable next to the mount", itc.get("/gis/meta").status_code == 200)
    finally:
        os.environ.pop("GIS_APP_DIR", None)


def test_main_installs_router():
    src = (PROJECT_ROOT / "app" / "main.py").read_text(encoding="utf-8")
    i, j = src.find("\ninstall_gis(app)"), src.find("\ninstall_video_pipeline(app)")
    check("app/main.py installs the GIS router just before the video pipeline (which wraps the lifespan last)",
          0 < i < j)


# ============================================================
# Runner
# ============================================================

def main() -> int:
    if not DSN:
        print("SKIP: DATABASE_URL is not set (the GIS tests need the local PostgreSQL)")
        return 0
    env = None
    try:
        env = Env()
        before = table_fingerprint(env)
        _, tc = make_app(None)
        for name, fn in (("GeoJSON validity", lambda: test_geojson_validity(env, tc)),
                         ("filters", lambda: test_filters(env, tc)),
                         ("images", lambda: test_images(env, tc)),
                         ("CSV", lambda: test_csv(env, tc)),
                         ("summary / sessions / meta", lambda: test_summary_and_sessions(env, tc)),
                         ("citizen layer", lambda: test_citizen(env)),
                         ("read-only + database down", lambda: test_unavailable_and_read_only(env, before)),
                         ("sample export + mock parity + install", lambda: test_mock_parity_and_install(env, tc)),
                         ("main.py", test_main_installs_router)):
            print(f"\n== {name}")
            try:
                fn()
            except Exception as e:
                check(f"{name}: no exception", False, f"{type(e).__name__}: {e}")
                traceback.print_exc()
    finally:
        if env is not None:
            env.close()
    print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} checks passed")
    if FAILURES:
        print("FAILED: " + "; ".join(FAILURES))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
