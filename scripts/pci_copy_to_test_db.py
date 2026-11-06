"""
Copy the drive-video survey (video_sessions, video_segments and the segments' assessments) from the
production local PostgreSQL `geoai` (READ-ONLY: one read-only transaction, SELECT only) into a scratch
database (default `geoai_test`), so the PCI tests and the WebGIS harness work on real data without
touching production.

- The scratch database is created and given the full schema first (scripts/setup_local_db.setup,
  which also applies migration 009 for research.pci_reports).
- Idempotent: the copied sessions' rows (and the assessments they point at) are deleted and inserted
  again in ONE transaction; rows of other sessions in the scratch DB are left alone.
- photo_id is set to NULL on the copies: the RoadSide `photos` table is not copied (its trigger would
  create pending assessments), and nothing in the GIS / PCI code reads photo_id.
- image_path is copied unchanged, so frames resolve to the real files under LOCAL_IMAGE_ROOT (read only).
- Refuses a target database named `geoai` (or postgres/template*).

NOTE: scripts/tests/test_local_db.py and test_expert_review.py DROP and recreate `geoai_test`. Re-run this
script after them (about 10-20 s).

Usage:
    venv/Scripts/python.exe scripts/pci_copy_to_test_db.py                 # -> geoai_test
    venv/Scripts/python.exe scripts/pci_copy_to_test_db.py --database geoai_test --check
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import psycopg  # noqa: E402
from psycopg.conninfo import conninfo_to_dict  # noqa: E402
from psycopg.types.json import Jsonb  # noqa: E402

from scripts import setup_local_db  # noqa: E402

SOURCE_DB = "geoai"
DEFAULT_TARGET = "geoai_test"
_FORBIDDEN = {"geoai", "postgres", "template0", "template1"}

SESSION_COLS = ("session_id", "state", "error", "metrics", "report", "queued_at", "extracting_at",
                "classifying_at", "storing_at", "stored_at", "cleaned_at", "failed_at", "created_at", "updated_at")
SEGMENT_COLS = ("segment_id", "session_id", "k", "status", "length_m", "s0_m", "s1_m", "centre_lat", "centre_lon",
                "chainage_start_km", "chainage_end_km", "utc", "image_path", "image_sha256", "photo_id",
                "assessment_id", "record", "vlm_response", "created_at", "updated_at")
JSON_COLS = {"metrics", "report", "record", "vlm_response", "distress_types", "expert_corrected_types",
             "raw_response"}


def _cols(conn, table: str) -> list[str]:
    return [r[0] for r in conn.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name=%s "
        "ORDER BY ordinal_position", (table,)).fetchall()]


def _wrap(cols, row):
    return [Jsonb(v) if (c in JSON_COLS and v is not None) else v for c, v in zip(cols, row)]


def read_source(src_dsn: str) -> dict:
    """Everything to copy, read in ONE read-only transaction from production."""
    with psycopg.connect(src_dsn, autocommit=False, options="-c TimeZone=UTC") as c:
        c.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        a_cols = _cols(c, "assessments")
        sessions = c.execute(f"SELECT {', '.join(SESSION_COLS)} FROM video_sessions ORDER BY session_id").fetchall()
        segments = c.execute(f"SELECT {', '.join(SEGMENT_COLS)} FROM video_segments ORDER BY session_id, k").fetchall()
        assessments = c.execute(
            f"SELECT {', '.join('a.' + x for x in a_cols)} FROM assessments a "
            f"JOIN video_segments g ON g.assessment_id = a.id").fetchall()
        c.rollback()
    return {"sessions": sessions, "segments": segments, "assessments": assessments, "a_cols": a_cols}


def write_target(dst_dsn: str, data: dict) -> dict:
    a_cols = data["a_cols"]
    pid = SEGMENT_COLS.index("photo_id")
    a_pid = a_cols.index("photo_id") if "photo_id" in a_cols else None
    sids = [r[0] for r in data["sessions"]]
    with psycopg.connect(dst_dsn, autocommit=True, options="-c TimeZone=UTC") as c:
        tgt_a_cols = set(_cols(c, "assessments"))
        use_a = [x for x in a_cols if x in tgt_a_cols]
        with c.transaction():
            old = [r[0] for r in c.execute(
                "SELECT assessment_id FROM video_segments WHERE session_id = ANY(%s) AND assessment_id IS NOT NULL",
                (sids,)).fetchall()]
            c.execute("DELETE FROM video_sessions WHERE session_id = ANY(%s)", (sids,))   # cascades segments
            new_ids = [r[a_cols.index("id")] for r in data["assessments"]]
            c.execute("DELETE FROM assessments WHERE id = ANY(%s)", (list(set(old) | set(new_ids)),))
            with c.cursor() as cur:
                with cur.copy(f"COPY assessments ({', '.join(use_a)}) FROM STDIN") as cp:
                    for r in data["assessments"]:
                        r = list(r)
                        if a_pid is not None:
                            r[a_pid] = None
                        d = dict(zip(a_cols, r))
                        cp.write_row(_wrap(use_a, [d[x] for x in use_a]))
                with cur.copy(f"COPY video_sessions ({', '.join(SESSION_COLS)}) FROM STDIN") as cp:
                    for r in data["sessions"]:
                        cp.write_row(_wrap(SESSION_COLS, r))
                with cur.copy(f"COPY video_segments ({', '.join(SEGMENT_COLS)}) FROM STDIN") as cp:
                    for r in data["segments"]:
                        r = list(r)
                        r[pid] = None
                        cp.write_row(_wrap(SEGMENT_COLS, r))
        n = c.execute("SELECT count(*), count(assessment_id) FROM video_segments WHERE session_id = ANY(%s)",
                      (sids,)).fetchone()
    return {"sessions": len(sids), "segments": n[0], "with_assessment": n[1], "assessments": len(data["assessments"])}


def copy(target_db: str = DEFAULT_TARGET, *, base_dsn: str | None = None) -> dict:
    if target_db in _FORBIDDEN:
        raise ValueError(f"refusing to write to database {target_db!r}")
    base = base_dsn or setup_local_db.database_url()
    src = setup_local_db.with_database(base, SOURCE_DB)
    dst = setup_local_db.with_database(base, target_db)
    if conninfo_to_dict(dst).get("dbname") in _FORBIDDEN:
        raise ValueError("refusing to write to production")
    t0 = time.time()
    setup_local_db.setup(dst)
    data = read_source(src)
    out = write_target(dst, data)
    out["target"] = setup_local_db.describe(dst)
    out["seconds"] = round(time.time() - t0, 1)
    return out


def is_copied(dst_dsn: str, min_segments: int = 1000) -> bool:
    """True when the scratch DB already holds a copy (used by the tests to decide whether to copy)."""
    try:
        with psycopg.connect(dst_dsn, autocommit=True, connect_timeout=3) as c:
            n = c.execute("SELECT count(*) FROM video_segments WHERE session_id ~ '^2026102[0-9]T'").fetchone()[0]
            return n >= min_segments
    except psycopg.Error:
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--database", default=DEFAULT_TARGET)
    ap.add_argument("--check", action="store_true", help="only report whether the copy is present")
    a = ap.parse_args(argv)
    if a.check:
        dst = setup_local_db.with_database(setup_local_db.database_url(), a.database)
        print("present" if is_copied(dst) else "absent")
        return 0
    try:
        out = copy(a.database)
    except (ValueError, RuntimeError, psycopg.Error) as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
