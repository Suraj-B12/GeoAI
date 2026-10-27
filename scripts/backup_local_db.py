"""
Back up the local PostgreSQL (DB_BACKEND=postgres / the video survey) and check the image store.

    venv\\Scripts\\python.exe scripts\\backup_local_db.py [--keep 10] [--out C:\\GeoAI\\backups]

Writes, for one UTC stamp (e.g. 20261028T101500Z), into --out:

    geoai_<stamp>.dump           pg_dump -Fc of the database in DATABASE_URL (restore: pg_restore -d <db> file)
    geoai_<stamp>_images.json    manifest of LOCAL_IMAGE_ROOT: every file's relative path, bytes, SHA-256
    geoai_<stamp>_report.json    what was checked and found (also printed)

Checks (the backup is "ok" only if all hold):
  - pg_dump exits 0 and `pg_restore --list` reads the archive back, with table data for
    video_sessions, video_segments, photos and assessments;
  - every video_segments row that has an image_path (and every send/audit row) points to a
    file that exists inside LOCAL_IMAGE_ROOT and whose SHA-256 equals the record's
    frame.sha256 and the row's image_sha256.

The images themselves are NOT copied (they are 5-9 MB each); the manifest is what lets you
verify a copy of LOCAL_IMAGE_ROOT made by other means. Only after an ok backup are older
backups removed, keeping the newest --keep sets (files matching geoai_<stamp>* only).

The password is passed to pg_dump through PGPASSWORD in the child's environment, never on a
command line, and is never printed. pg_dump takes a consistent snapshot, so a session being
stored at the same time is either in the dump completely or not at all.

Exit codes: 0 ok, 1 verification failed (the dump is kept), 2 the dump failed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUT = Path(r"C:\GeoAI\backups")
DEFAULT_PG_BIN = Path(r"C:\GeoAI\pgsql\bin")
DEFAULT_KEEP = 10
_STAMP = re.compile(r"^geoai_(\d{8}T\d{6}Z)(\.dump|_images\.json|_report\.json)$")
_REQUIRED_DATA = ("video_sessions", "video_segments", "photos", "assessments")
_CHUNK = 1 << 20


def load_env() -> None:
    """DATABASE_URL etc. from the project's .env (python-dotenv; existing variables win)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def pg_tool(name: str) -> Path:
    exe = Path(os.environ.get("PG_BIN") or DEFAULT_PG_BIN) / f"{name}.exe"
    if not exe.is_file():
        exe = exe.with_suffix("")
    if not exe.is_file():
        raise FileNotFoundError(f"{name} not found in PG_BIN ({exe.parent})")
    return exe


def pg_args_env(dsn: str) -> tuple[list[str], dict]:
    """(-h/-p/-U/-d arguments, child environment with PGPASSWORD) from a DSN; the password never
    appears in the arguments."""
    from psycopg.conninfo import conninfo_to_dict

    p = conninfo_to_dict(dsn)
    args = []
    for opt, key in (("-h", "host"), ("-p", "port"), ("-U", "user"), ("-d", "dbname")):
        if p.get(key):
            args += [opt, str(p[key])]
    env = dict(os.environ)
    if p.get("password"):
        env["PGPASSWORD"] = str(p["password"])
    env.pop("DATABASE_URL", None)
    return args, env


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================
# Steps
# ============================================================

def dump_database(dsn: str, target: Path) -> dict:
    """pg_dump -Fc into target (via a .partial file), then read it back with pg_restore --list."""
    args, env = pg_args_env(dsn)
    partial = target.with_name(target.name + ".partial")
    partial.unlink(missing_ok=True)
    t0 = time.monotonic()
    proc = subprocess.run([str(pg_tool("pg_dump")), "-Fc", "-Z", "6", "--no-password", *args, "-f", str(partial)],
                          env=env, capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"pg_dump exited {proc.returncode}: {proc.stderr.strip()[-2000:]}")
    os.replace(partial, target)
    seconds = round(time.monotonic() - t0, 1)
    listing = subprocess.run([str(pg_tool("pg_restore")), "--list", str(target)], capture_output=True, text=True,
                             timeout=600)
    entries = [ln for ln in listing.stdout.splitlines() if ln and not ln.startswith(";")]
    data_tables = sorted({m.group(1) for ln in entries
                          for m in [re.search(r"TABLE DATA public (\S+) ", ln)] if m})
    version = subprocess.run([str(pg_tool("pg_dump")), "--version"], capture_output=True, text=True).stdout.strip()
    return {"file": str(target), "bytes": target.stat().st_size, "seconds": seconds, "pg_dump": version,
            "restore_list_ok": listing.returncode == 0, "toc_entries": len(entries),
            "table_data": data_tables, "missing_table_data": [t for t in _REQUIRED_DATA if t not in data_tables]}


def table_counts(dsn: str) -> dict:
    import psycopg
    from psycopg import sql

    with psycopg.connect(dsn, autocommit=True) as c:
        tables = [r[0] for r in c.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename").fetchall()]
        counts = {t: c.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(t))).fetchone()[0]
                  for t in tables}
        seg = c.execute("SELECT status, COUNT(*) FROM video_segments GROUP BY status ORDER BY status").fetchall() \
            if "video_segments" in tables else []
    return {"tables": counts, "video_segments_by_status": dict(seg)}


def image_manifest(root: Path, workers: int = 4) -> dict:
    """Every file under root (temporary dot-files excluded): relative path, bytes, SHA-256."""
    files = sorted(p for p in root.rglob("*") if p.is_file() and not p.name.startswith(".")) if root.is_dir() else []
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        hashes = list(pool.map(sha256_file, files))
    entries = [{"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size, "sha256": h}
               for p, h in zip(files, hashes)]
    return {"root": str(root), "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "files": len(entries), "bytes": sum(e["bytes"] for e in entries),
            "seconds": round(time.monotonic() - t0, 1), "entries": entries}


def verify_segments(dsn: str, root: Path, manifest: dict) -> dict:
    """Every stored segment image exists inside root with the record's SHA-256."""
    import psycopg

    from app.local_images import inside_root, local_image_path

    by_rel = {e["path"].lower(): e["sha256"] for e in manifest["entries"]}
    root_r = root.resolve()
    with psycopg.connect(dsn, autocommit=True) as c:
        rows = c.execute(
            "SELECT segment_id, status, image_path, image_sha256, record->'frame'->>'sha256' "
            "FROM video_segments WHERE image_path IS NOT NULL OR status IN ('send', 'audit') "
            "ORDER BY session_id, k").fetchall()
    problems = {"no_image_path": [], "outside_root": [], "missing": [], "sha_mismatch": [],
                "column_vs_record": []}
    referenced = set()
    for seg, status, path, col_sha, rec_sha in rows:
        p = local_image_path(path)
        if p is None:
            problems["no_image_path"].append(seg)
            continue
        try:
            resolved = inside_root(p, root)
        except PermissionError:
            problems["outside_root"].append(seg)
            continue
        rel = resolved.relative_to(root_r).as_posix().lower()
        referenced.add(rel)
        if col_sha != rec_sha:
            problems["column_vs_record"].append(seg)
        actual = by_rel.get(rel)
        if actual is None:
            actual = sha256_file(resolved) if resolved.is_file() else None   # written after the manifest
        if actual is None:
            problems["missing"].append(seg)
        elif actual != rec_sha:
            problems["sha_mismatch"].append(seg)
    unreferenced = sorted(set(by_rel) - referenced)
    return {"checked": len(rows), "ok": len(rows) - sum(len(v) for v in problems.values()),
            "problems": {k: {"count": len(v), "examples": v[:10]} for k, v in problems.items()},
            "unreferenced_files": {"count": len(unreferenced), "examples": unreferenced[:10]},
            "passed": not any(problems.values())}


def rotate(out: Path, keep: int, current: str) -> list[str]:
    """Delete the files of all but the newest `keep` stamps (never the current one)."""
    stamps = _stamps(out)
    ordered = sorted(stamps)
    removed = []
    for stamp in ordered[:-keep] if keep > 0 else []:
        if stamp == current:
            continue
        for f in stamps[stamp]:
            f.unlink(missing_ok=True)
            removed.append(f.name)
    return removed


def _stamps(out: Path) -> dict:
    stamps: dict[str, list[Path]] = {}
    for f in out.iterdir() if out.is_dir() else []:
        m = _STAMP.match(f.name)
        if m and f.is_file():
            stamps.setdefault(m.group(1), []).append(f)
    return stamps


# ============================================================
# Main
# ============================================================

def backup(out: Path, keep: int, dsn: str, root: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {"stamp": stamp, "database": None, "dump": None, "counts": None, "images": None,
              "verification": None, "ok": False, "removed_old": []}
    from scripts.setup_local_db import describe
    report["database"] = describe(dsn)

    report["dump"] = dump_database(dsn, out / f"geoai_{stamp}.dump")
    report["counts"] = table_counts(dsn)
    manifest = image_manifest(root)
    (out / f"geoai_{stamp}_images.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    report["images"] = {k: manifest[k] for k in ("root", "files", "bytes", "seconds")}
    report["images"]["manifest"] = str(out / f"geoai_{stamp}_images.json")
    report["verification"] = verify_segments(dsn, root, manifest)
    d = report["dump"]
    report["ok"] = bool(d["restore_list_ok"] and not d["missing_table_data"] and report["verification"]["passed"])
    if report["ok"]:
        report["removed_old"] = rotate(out, keep, stamp)
    (out / f"geoai_{stamp}_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"backup folder (default {DEFAULT_OUT})")
    ap.add_argument("--keep", type=int, default=DEFAULT_KEEP, help="backups to keep (default %(default)s)")
    a = ap.parse_args(argv)
    load_env()
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        print("DATABASE_URL is not set (.env) - nothing to back up.")
        return 2
    from app.local_images import image_root
    root = image_root()
    try:
        r = backup(a.out, max(1, a.keep), dsn, root)
    except Exception as e:
        print(f"BACKUP FAILED: {type(e).__name__}: {e}")
        return 2
    d, v, im = r["dump"], r["verification"], r["images"]
    print(f"database        {r['database']}")
    print(f"dump            {d['file']}  {d['bytes'] / 1e6:.2f} MB in {d['seconds']} s  ({d['pg_dump']})")
    print(f"restore --list  {'ok' if d['restore_list_ok'] else 'FAILED'}, {d['toc_entries']} entries, "
          f"table data for {len(d['table_data'])} tables"
          + (f", MISSING {d['missing_table_data']}" if d['missing_table_data'] else ""))
    print("rows            " + ", ".join(f"{t}={n}" for t, n in r["counts"]["tables"].items()))
    print(f"images          {im['files']} files, {im['bytes'] / 1e9:.3f} GB under {im['root']} "
          f"(hashed in {im['seconds']} s) -> {im['manifest']}")
    print(f"verification    {v['ok']}/{v['checked']} segment images ok"
          + "".join(f", {k}={p['count']}" for k, p in v["problems"].items() if p["count"])
          + f"; {v['unreferenced_files']['count']} unreferenced file(s)")
    print(f"removed old     {len(r['removed_old'])} file(s)")
    print("RESULT          " + ("OK" if r["ok"] else "VERIFICATION FAILED (dump kept, old backups not rotated)"))
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
