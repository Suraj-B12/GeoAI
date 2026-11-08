"""
Roughness features from every drive's imu.csv for every drive-video segment -> research.segment_roughness
(+ one research.session_roughness row per drive). Method: app/pci_iri.py; why: eval_results/pci/iri/RESEARCH.md.

CPU only, no GPU lock needed. Idempotent and resumable: one transaction per drive; a drive that already has a
session_roughness row for the current method_fp is skipped (use --redo to recompute). Applies
migrations/011_segment_roughness.sql itself (CREATE ... IF NOT EXISTS).

Where the IMU is looked for (first hit): $VIDEO_INBOX (default C:\\GeoAI\\inbox)/<session>/imu.csv, then
<PROJECT_ROOT>/../GeoAI_Video/field/sessions/<session>/imu.csv. GNSS from gnss_fixes.csv beside it.

Usage:
    venv/Scripts/python.exe scripts/iri_from_imu.py --database geoai          # all drives
    venv/Scripts/python.exe scripts/iri_from_imu.py --database geoai_test --session <id> --redo
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import psycopg  # noqa: E402
from psycopg.types.json import Jsonb  # noqa: E402

from app import pci_iri as P  # noqa: E402
from scripts import setup_local_db  # noqa: E402

MIGRATION = PROJECT_ROOT / "migrations" / "011_segment_roughness.sql"
COLS = ("segment_id", "method_fp", "session_id", "k", "imu_sha256", "t0_ns", "t1_ns", "duration_s", "n_samples",
        "coverage", "speed_mps", "speed_min_mps", "rms_av", "rms_body", "rms_wheel", "rms_rot", "peak_av",
        "rms_av_norm", "iri_gc_uncal", "flags", "error")


def imu_dirs(session_id: str) -> list[Path]:
    inbox = Path(os.environ.get("VIDEO_INBOX", r"C:\GeoAI\inbox"))
    return [inbox / session_id, PROJECT_ROOT.parent / "GeoAI_Video" / "field" / "sessions" / session_id]


def find_imu(session_id: str) -> Path | None:
    for d in imu_dirs(session_id):
        if (d / "imu.csv").exists():
            return d
    return None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_gnss(path: Path) -> dict:
    import pandas as pd
    g = pd.read_csv(path, usecols=["t_ns", "lat_deg", "lon_deg", "speed_mps", "is_mock"])
    g = g[g["lat_deg"].notna() & g["lon_deg"].notna() & (g["is_mock"].fillna(0) == 0)]
    return {"t_ns": g["t_ns"].to_numpy(np.int64), "speed_mps": pd.to_numeric(g["speed_mps"], errors="coerce").to_numpy(float),
            "lat": g["lat_deg"].to_numpy(float), "lon": g["lon_deg"].to_numpy(float)}


def load_segments(conn, session_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT segment_id, k, status, s0_m, s1_m, (record->'frame'->>'t_ns')::bigint, "
        "(record->'frame'->>'near_edge_ahead_m')::float8 FROM video_segments WHERE session_id = %s ORDER BY k",
        (session_id,)).fetchall()
    return [{"segment_id": r[0], "k": r[1], "status": r[2], "s0_m": r[3], "s1_m": r[4], "frame_t_ns": r[5],
             "near_edge_ahead_m": r[6], "covered": r[2] != "not_covered"} for r in rows]


def _row(est: dict, fp: str, sid: str, imu_sha: str | None) -> dict:
    r = {c: est.get(c) for c in COLS}
    r.update(method_fp=fp, session_id=sid, imu_sha256=imu_sha, flags=est.get("flags") or [])
    for c in ("speed_mps", "speed_min_mps", "rms_av", "rms_body", "rms_wheel", "rms_rot", "peak_av", "rms_av_norm",
              "iri_gc_uncal", "duration_s", "coverage"):
        if r[c] is not None and not np.isfinite(r[c]):
            r[c] = None
    return r


def process(conn, session_id: str, fp: str, p: P.Params) -> dict:
    segs = load_segments(conn, session_id)
    d = find_imu(session_id)
    summary: dict = {"n_segments": len(segs)}
    rows, err, imu_path, imu_sha = [], None, None, None
    try:
        if d is None:
            raise FileNotFoundError("no imu.csv for this session (looked in " + ", ".join(map(str, imu_dirs(session_id))) + ")")
        imu_path = d / "imu.csv"
        imu_sha = sha256(imu_path)
        t = time.time()
        imu = P.read_imu_csv(imu_path)
        gnss = read_gnss(d / "gnss_fixes.csv")
        res = P.run_session(imu, gnss, segs, p)
        summary.update(idle=res["idle"], acc_one_g=res["acc_one_g"], gravity_source=res["gravity_source"],
                       n_anchors=res["n_anchors"], time_s=round(time.time() - t, 2),
                       n_features=sum(1 for e in res["segments"] if e.get("rms_av") is not None),
                       n_iri_gc=sum(1 for e in res["segments"] if e.get("iri_gc_uncal") is not None))
        rows = [_row(e, fp, session_id, imu_sha) for e in res["segments"]]
    except Exception as e:  # the whole drive failed: one error row per segment, so the failure is visible
        err = f"{type(e).__name__}: {e}"
        rows = [_row({"segment_id": s["segment_id"], "k": s["k"], "error": err}, fp, session_id, imu_sha) for s in segs]
    with conn.transaction():
        conn.execute("DELETE FROM research.segment_roughness WHERE session_id = %s AND method_fp = %s", (session_id, fp))
        if rows:
            with conn.cursor() as cur:
                cur.executemany(
                    f"INSERT INTO research.segment_roughness ({', '.join(COLS)}) VALUES ({', '.join('%(' + c + ')s' for c in COLS)})",
                    rows)
        conn.execute(
            "INSERT INTO research.session_roughness (session_id, method_fp, imu_path, imu_sha256, summary, error) "
            "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (session_id, method_fp) DO UPDATE SET imu_path = EXCLUDED.imu_path, "
            "imu_sha256 = EXCLUDED.imu_sha256, summary = EXCLUDED.summary, error = EXCLUDED.error, created_at = NOW()",
            (session_id, fp, str(imu_path) if imu_path else None, imu_sha, Jsonb(summary), err))
    return {"session_id": session_id, "error": err, **summary}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--database", default=None, help="database name (default: the one in DATABASE_URL)")
    ap.add_argument("--session", action="append", default=[], help="only this session (repeatable)")
    ap.add_argument("--redo", action="store_true", help="recompute drives already stored for this method")
    a = ap.parse_args(argv)
    dsn = setup_local_db.database_url()
    if a.database:
        dsn = setup_local_db.with_database(dsn, a.database)
    p = P.DEFAULT
    fp = P.method_fp(p)
    print(f"method_fp {fp}  db {setup_local_db.describe(dsn)}", flush=True)
    failed = 0
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(MIGRATION.read_text(encoding="utf-8"))
        sessions = [r[0] for r in conn.execute(
            "SELECT s.session_id FROM video_sessions s WHERE EXISTS (SELECT 1 FROM video_segments g "
            "WHERE g.session_id = s.session_id) ORDER BY 1").fetchall()]
        if a.session:
            sessions = [s for s in sessions if s in set(a.session)]
        done = {r[0] for r in conn.execute(
            "SELECT session_id FROM research.session_roughness WHERE method_fp = %s AND error IS NULL", (fp,))}
        for sid in sessions:
            if sid in done and not a.redo:
                print(f"skip {sid} (stored)", flush=True)
                continue
            try:
                r = process(conn, sid, fp, p)
            except Exception:
                failed += 1
                traceback.print_exc()
                continue
            if r["error"]:
                failed += 1
            print(json.dumps(r, default=str), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
