"""
Create the local PostgreSQL database and apply the schema (DB_BACKEND=postgres).

The local database must have the SAME schema as the Supabase project (GeoAI_Video
docs/HANDOFF_PLAN.md section 7). There is no Supabase DB password, so no
pg_dump: the schema is rebuilt from the repo's own SQL, unchanged, in the order
Supabase received it, after a shim for what Supabase provides and plain
PostgreSQL does not:

    migrations/000_local_postgres_shim.sql   auth schema/functions, API roles, photos  (local only)
    db_schema.sql                            assessments, custom_distress_types, retrain_jobs
    migrations/001 ... 004                   worker queue, photos trigger, expert UI, pre-filter
    migrations/005_video_pipeline.sql        video_sessions, video_segments (also for Supabase)
    migrations/006_expert_review_log.sql     research.expert_review_log (local only, outside public)

scripts/tests/test_local_db.py proves the result matches the live Supabase
columns (PostgREST OpenAPI) table by table.

Idempotent. db_schema.sql is not re-runnable on its own (plain CREATE INDEX /
CREATE POLICY), so each file is applied once, inside ONE transaction together
with its row in the ledger geoai_local.schema_migrations: a file is either fully
applied and recorded, or neither. The files' own top-level BEGIN; / COMMIT;
lines are dropped for that (plpgsql bodies use BEGIN without a semicolon and
are untouched). A file whose content changed after it was applied is reported,
never re-applied: write a new numbered migration instead.

Reads DATABASE_URL from the environment or from .env. Never prints it.

Usage:
    venv/Scripts/python.exe scripts/setup_local_db.py
    venv/Scripts/python.exe scripts/setup_local_db.py --database geoai_test   # another database, same server
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS_DIR = PROJECT_ROOT / "migrations"

# Order matters: each file depends on the ones before it.
SCHEMA_FILES: tuple[Path, ...] = (
    MIGRATIONS_DIR / "000_local_postgres_shim.sql",
    PROJECT_ROOT / "db_schema.sql",
    MIGRATIONS_DIR / "001_operator_pipeline.sql",
    MIGRATIONS_DIR / "002_photos_integration.sql",
    MIGRATIONS_DIR / "003_expert_ui_anon_access.sql",
    MIGRATIONS_DIR / "004_pavement_filter_and_dashboard.sql",
    MIGRATIONS_DIR / "005_video_pipeline.sql",
    MIGRATIONS_DIR / "006_expert_review_log.sql",
)

LEDGER_SCHEMA = "geoai_local"   # outside `public`, so it is not part of the compared schema
_TX_LINE = re.compile(r"^\s*(BEGIN|COMMIT)\s*;\s*(--.*)?$", re.IGNORECASE)
_SETUP_LOCK = 0x6765_6F61_6900  # pg_advisory_lock key: one setup at a time per server


def database_url(explicit: str | None = None) -> str:
    """DATABASE_URL from the argument, the environment, or the project's .env."""
    if explicit:
        return explicit
    if os.environ.get("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    env = PROJECT_ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("DATABASE_URL=") and "=" in line:
                v = line.split("=", 1)[1].strip().strip('"').strip("'")
                if v:
                    return v
    raise RuntimeError("DATABASE_URL is not set (environment or .env). "
                       "See SETUP.md, 'Local PostgreSQL'.")


def with_database(dsn: str, dbname: str) -> str:
    """The same connection string pointing at another database on the same server."""
    params = conninfo_to_dict(dsn)
    params["dbname"] = dbname
    return make_conninfo(**params)


def describe(dsn: str) -> str:
    """host:port/dbname of a DSN, for messages (never the password)."""
    p = conninfo_to_dict(dsn)
    return f"{p.get('host', 'localhost')}:{p.get('port', 5432)}/{p.get('dbname', '?')}"


def ensure_database(dsn: str) -> bool:
    """Create the DSN's database if it does not exist. Returns True when it was created."""
    dbname = conninfo_to_dict(dsn).get("dbname")
    if not dbname:
        raise ValueError("DATABASE_URL names no database")
    with psycopg.connect(with_database(dsn, "postgres"), autocommit=True) as conn:
        if conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,)).fetchone():
            return False
        conn.execute(sql.SQL("CREATE DATABASE {} ENCODING 'UTF8' TEMPLATE template0").format(
            sql.Identifier(dbname)))
        return True


def drop_database(dsn: str) -> None:
    """Drop the DSN's database (tests only). Refuses the production name `geoai`."""
    dbname = conninfo_to_dict(dsn).get("dbname")
    if not dbname or dbname in ("geoai", "postgres", "template0", "template1"):
        raise ValueError(f"refusing to drop database {dbname!r}")
    with psycopg.connect(with_database(dsn, "postgres"), autocommit=True) as conn:
        conn.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(dbname)))


def statements_without_outer_transaction(text: str) -> str:
    """The file's SQL without its own top-level BEGIN; / COMMIT; lines."""
    return "\n".join(line for line in text.splitlines() if not _TX_LINE.match(line))


def apply_schema(dsn: str, files: tuple[Path, ...] = SCHEMA_FILES) -> list[dict]:
    """Apply every schema file not yet in the ledger. Returns one {file, action} per file.

    action: "applied" | "already_applied" | "changed_since_applied" (not re-applied).
    Raises on the first file that fails; that file leaves nothing behind.
    """
    out = []
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute("SELECT pg_advisory_lock(%s)", (_SETUP_LOCK,))
        try:
            conn.execute(sql.SQL(
                "CREATE SCHEMA IF NOT EXISTS {s};"
                "CREATE TABLE IF NOT EXISTS {s}.schema_migrations ("
                " filename TEXT PRIMARY KEY, sha256 TEXT NOT NULL,"
                " applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW())").format(s=sql.Identifier(LEDGER_SCHEMA)))
            ledger = sql.SQL("{}.schema_migrations").format(sql.Identifier(LEDGER_SCHEMA))
            for path in files:
                raw = path.read_bytes()
                digest = hashlib.sha256(raw).hexdigest()
                name = path.relative_to(PROJECT_ROOT).as_posix()
                row = conn.execute(sql.SQL("SELECT sha256 FROM {} WHERE filename = %s").format(ledger),
                                   (name,)).fetchone()
                if row:
                    out.append({"file": name, "action": "already_applied" if row[0] == digest
                                else "changed_since_applied"})
                    continue
                body = statements_without_outer_transaction(raw.decode("utf-8-sig"))
                try:
                    with conn.transaction():
                        conn.execute(body)
                        conn.execute(sql.SQL("INSERT INTO {} (filename, sha256) VALUES (%s, %s)").format(ledger),
                                     (name, digest))
                except psycopg.Error as e:
                    raise RuntimeError(f"{name} failed, nothing from it was applied: "
                                       f"{type(e).__name__}: {e}") from e
                out.append({"file": name, "action": "applied"})
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s)", (_SETUP_LOCK,))
    return out


def setup(dsn: str) -> dict:
    """ensure_database + apply_schema. Returns {database, created, files}."""
    created = ensure_database(dsn)
    return {"database": describe(dsn), "created": created, "files": apply_schema(dsn)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--database", help="database name to use instead of the one in DATABASE_URL")
    a = ap.parse_args(argv)
    try:
        dsn = database_url()
        if a.database:
            dsn = with_database(dsn, a.database)
        result = setup(dsn)
    except (RuntimeError, ValueError, psycopg.Error) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"Database {result['database']}: {'created' if result['created'] else 'exists'}")
    for f in result["files"]:
        print(f"  {f['action']:<22} {f['file']}")
    changed = [f["file"] for f in result["files"] if f["action"] == "changed_since_applied"]
    if changed:
        print("WARNING: these files changed after they were applied and were NOT re-applied: "
              + ", ".join(changed) + ". Put schema changes in a new numbered migration.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
