"""
Export the portable GIS sample bundle: everything the WebGIS teammate needs to develop offline.

    venv\\Scripts\\python.exe scripts\\export_gis_sample.py
        [--out C:\\GeoAI\\gis_sample] [--zip C:\\GeoAI\\gis_sample.zip] [--full-pngs 10]
        [--budget-mb 300] [--citizen-addresses] [--no-citizen]

The bundle (and a zip of it) holds, for the data stored NOW:

    api/gis/...                  the exact bodies of every /gis endpoint (app/gis_routes.py), called
                                 in-process through the real router: meta, segments (footprint / point /
                                 line, all rows), sessions (with tracks and per-km tables), summary, CSV,
                                 citizen uploads
    examples/                    pretty-printed single features and a few filtered queries with their answers
    images/w480, images/w1280    JPEG previews of every stored segment frame (the API's own ?w= output)
    images/full                  --full-pngs full-resolution PNGs (the detail view's "full image")
    schema.sql                   pg_dump --schema-only of the local PostgreSQL
    mock_server.py, gis_core.py  a stdlib mock of the API over these files (same filter code as the server)
    README.md, manifest.json     how to use it; every file with bytes and SHA-256

Idempotent: re-run it after more sessions are stored. The bundle is built in <out>.staging and swapped
in at the end, so a failed run leaves the previous bundle intact; previews come from the API's disk
cache, so a re-run only decodes new frames.

Privacy: citizen addresses are removed from the bundle unless --citizen-addresses (the API itself
serves them, as the citizen WebGIS brief does). No secret is written: the API responses contain none,
and pg_dump gets the password through PGPASSWORD in its environment only.

Reads only. Never touches the GPU or the running server (the router runs in this process).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUT = Path(r"C:\GeoAI\gis_sample")
DEFAULT_ZIP = Path(r"C:\GeoAI\gis_sample.zip")
_MAX = 20000      # = gis_core.MAX_LIMIT: the saved files hold every row; the mock pages them


class _LocalBackend:
    """Stands in for app.local_db.LocalDb (DB_BACKEND=postgres): gis_routes only checks IS_LOCAL_DB
    and then reads the local PostgreSQL through app.video_store.connect()."""
    IS_LOCAL_DB = True

    async def aclose(self):
        pass


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def citizen_client(enabled: bool):
    """The client the server's lifespan would build (app/main.py rule), or None."""
    if not enabled:
        return None
    backend = (os.environ.get("DB_BACKEND") or "supabase").strip().lower()
    if backend == "postgres":
        return _LocalBackend() if os.environ.get("DATABASE_URL") else None
    if os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_KEY"):
        from app.supabase_client import build_supabase_client
        return build_supabase_client()
    return None


def dump_schema(target: Path) -> dict:
    """pg_dump --schema-only of DATABASE_URL (no owners / grants: portable to any PostgreSQL 15+)."""
    from scripts.backup_local_db import pg_args_env, pg_tool

    args, env = pg_args_env(os.environ["DATABASE_URL"])
    proc = subprocess.run([str(pg_tool("pg_dump")), "--schema-only", "--no-owner", "--no-privileges",
                           "--no-password", *args, "-f", str(target)], env=env, capture_output=True, text=True,
                          timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"pg_dump --schema-only exited {proc.returncode}: {proc.stderr.strip()[-1500:]}")
    return {"bytes": target.stat().st_size}


def _pick_full(features: list[dict], n: int) -> list[str]:
    """n segment ids for full PNGs: distinct marker labels first (the detail view's variety), then
    normal, pending and the rest, in route order."""
    chosen, seen = [], set()
    imaged = [f["properties"] for f in features if f["properties"]["has_image"]]
    for p in imaged:
        if p["vlm_class"] == "distress" and p["marker_label"] not in seen and len(chosen) < n:
            chosen.append(p["segment_id"])
            seen.add(p["marker_label"])
    for cls in ("normal", "pending", "distress", "unknown"):
        for p in imaged:
            if len(chosen) >= n:
                break
            if p["vlm_class"] == cls and p["segment_id"] not in chosen:
                chosen.append(p["segment_id"])
                break
    for p in imaged:
        if len(chosen) >= n:
            break
        if p["segment_id"] not in chosen:
            chosen.append(p["segment_id"])
    return chosen[:n]


def _write(path: Path, data: bytes) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return len(data)


def _pretty(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


EXAMPLE_QUERIES = [
    ("/gis/video/segments.geojson", "label=Potholes&geom=point&limit=3"),
    ("/gis/video/segments.geojson", "severity=Medium,High&vlm_class=distress&geom=point&limit=3"),
    ("/gis/video/segments.geojson", "status=send,audit&answered=false&geom=point&limit=2"),
    ("/gis/video/segments.geojson", "min_confidence=0.8&vlm_class=distress&limit=2"),
    ("/gis/video/segments.geojson", "needs_review=true&geom=line&limit=2"),
    ("/gis/video/summary", "label=Potholes,Alligator Cracking"),
    ("/gis/video/sessions", "per_km=false"),
    ("/gis/citizen/assessments.geojson", "label=Potholes&severity=High&include_review=true&limit=2"),
    ("/gis/video/segments.geojson", "label=Pothole"),          # a 400: the exact error the API gives
]


def export(out: Path, zip_path: Path | None, *, full_pngs: int = 10, budget_mb: float = 300.0,
           citizen: bool = True, citizen_addresses: bool = False, client=None, log=print) -> dict:
    """Build the bundle at `out` (swapped in atomically) and zip it. Returns the manifest summary.

    client: the citizen backend client to use (tests pass one); default: citizen_client(citizen).
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app import gis_routes

    t0 = time.monotonic()
    out = Path(out)
    staging = out.with_name(out.name + ".staging")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    budget = int(budget_mb * 1_000_000)

    app = FastAPI()
    app.include_router(gis_routes.router)
    app.state.supabase_client = client if client is not None else citizen_client(citizen)
    gis_routes.reset_citizen_cache()
    notes = []
    with TestClient(app) as tc:
        def get(path: str, query: str = "", ok=(200,)):
            r = tc.get(path + (f"?{query}" if query else ""))
            if r.status_code not in ok:
                raise RuntimeError(f"GET {path}?{query} -> HTTP {r.status_code}: {r.text[:300]}")
            return r

        api = staging / "api" / "gis"
        _write(api / "meta.json", get("/gis/meta").content)
        seg = {}
        for geom, name in (("footprint", "segments.geojson"), ("point", "segments_point.geojson"),
                           ("line", "segments_line.geojson")):
            body = get("/gis/video/segments.geojson", f"geom={geom}&limit={_MAX}").content
            _write(api / "video" / name, body)
            seg[geom] = json.loads(body)
        if seg["point"]["numberMatched"] > _MAX:
            raise RuntimeError(f"{seg['point']['numberMatched']} segments exceed one {_MAX}-row file; "
                               f"split the export by session")
        _write(api / "video" / "sessions.json", get("/gis/video/sessions", "track=true&per_km=true").content)
        _write(api / "video" / "summary.json", get("/gis/video/summary").content)
        _write(api / "video" / "segments.csv", get("/gis/video/segments.csv", f"limit={_MAX}").content)

        # Citizen layer: every eligible row (classified, done and expert_review) so the mock can filter.
        if app.state.supabase_client is not None:
            cit = json.loads(get("/gis/citizen/assessments.geojson", f"include_review=true&limit={_MAX}").content)
            if not citizen_addresses:
                for f in cit["features"]:
                    f["properties"]["address"] = None
                cit["sample_note"] = "address removed from the sample (privacy); the live API returns it"
                notes.append("citizen addresses removed (run with --citizen-addresses to keep them)")
        else:
            cit = {"type": "FeatureCollection", "schema": seg["point"]["schema"], "layer": "citizen",
                   "generated_at": seg["point"]["generated_at"], "numberMatched": 0, "numberReturned": 0,
                   "filters": {"include_review": True, "limit": _MAX, "offset": 0}, "backend": None,
                   "sample_note": "no citizen backend configured when this sample was exported",
                   "bbox": None, "features": []}
            notes.append("citizen layer empty: no backend configured")
        _write(api / "citizen" / "assessments.geojson",
               json.dumps(cit, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))

        # Examples (pretty) for reading
        ex = staging / "examples"
        feats = seg["footprint"]["features"]
        by_class = {}
        for f in feats:
            by_class.setdefault(f["properties"]["vlm_class"], f)
        for cls, f in by_class.items():
            _pretty(ex / f"segment_feature_{cls}.json", f)
        if cit["features"]:
            _pretty(ex / "citizen_feature.json", cit["features"][0])
        sessions = json.loads((api / "video" / "sessions.json").read_text(encoding="utf-8"))
        if sessions["sessions"]:
            s = dict(sessions["sessions"][0])
            if s.get("track") and s["track"]["type"] == "LineString":
                s["track"] = {"type": "LineString", "coordinates": s["track"]["coordinates"][:3],
                              "_example_note": "truncated to 3 points here; the API returns the whole line"}
            _pretty(ex / "session.json", s)
        _pretty(ex / "meta.json", json.loads((api / "meta.json").read_text(encoding="utf-8")))
        _pretty(ex / "summary.json", json.loads((api / "video" / "summary.json").read_text(encoding="utf-8")))
        queries = []
        for path, q in EXAMPLE_QUERIES:
            if path.startswith("/gis/citizen") and app.state.supabase_client is None:
                continue
            r = get(path, q, ok=(200, 400))
            body = r.json()
            if path.startswith("/gis/citizen") and not citizen_addresses:
                for f in body.get("features", []):
                    f["properties"]["address"] = None
            name = f"query_{len(queries) + 1:02d}.json"
            _pretty(ex / name, body)
            queries.append({"request": f"GET {path}?{q}", "status": r.status_code, "file": f"examples/{name}",
                            "numberMatched": body.get("numberMatched")})
        _pretty(ex / "queries.json", queries)

        # Images: warm the API's preview cache in parallel (same function, same bytes), then fetch
        # through the API so the files are exactly what /gis/image returns.
        ids = [f["properties"]["segment_id"] for f in seg["point"]["features"] if f["properties"]["has_image"]]
        log(f"[sample] {len(ids)} stored images: building previews (cached after the first run)...")

        def warm(sid):
            found = gis_routes.resolve_segment_image(sid)
            if found:
                for w in (480, 1280):
                    gis_routes.make_preview(found[0], w, found[2])
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(warm, ids))
        size = sum(p.stat().st_size for p in staging.rglob("*") if p.is_file())
        for sid in ids:
            size += _write(staging / "images" / "w480" / f"{sid}.jpg", get(f"/gis/image/{sid}", "w=480").content)
        full_ids = _pick_full(seg["point"]["features"], full_pngs)
        full_done = []
        reserve_1280 = 120_000 * len(ids)                 # ~70-110 KB each measured; keep room for them
        for sid in full_ids:
            r = get(f"/gis/image/{sid}")
            if size + len(r.content) + reserve_1280 > budget:
                notes.append(f"full PNGs limited to {len(full_done)} by the {budget_mb:.0f} MB budget")
                break
            size += _write(staging / "images" / "full" / f"{sid}.png", r.content)
            full_done.append(sid)
        w1280_missing = 0
        for sid in ids:
            body = get(f"/gis/image/{sid}", "w=1280").content
            if size + len(body) > budget:
                w1280_missing += 1
                continue
            size += _write(staging / "images" / "w1280" / f"{sid}.jpg", body)
        if w1280_missing:
            notes.append(f"{w1280_missing} w1280 previews left out by the budget (the mock falls back to w480)")

    # schema, mock, core, README, manifest
    schema = dump_schema(staging / "schema.sql")
    shutil.copy2(PROJECT_ROOT / "scripts" / "gis_mock_server.py", staging / "mock_server.py")
    shutil.copy2(PROJECT_ROOT / "app" / "gis_core.py", staging / "gis_core.py")
    summary = json.loads((staging / "api" / "gis" / "video" / "summary.json").read_text(encoding="utf-8"))
    info = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "schema": summary["schema"], "sessions": summary["sessions"], "segments": summary["segments"],
        "segments_by_status": summary["segments_by_status"], "answered": summary["answered"],
        "images": len(ids), "full_pngs": full_done, "w1280_missing": w1280_missing,
        "citizen_features": len(cit["features"]), "citizen_backend": cit.get("backend"),
        "citizen_addresses": bool(citizen_addresses and cit["features"]), "schema_sql_bytes": schema["bytes"],
        "notes": notes,
    }
    (staging / "README.md").write_text(readme(info), encoding="utf-8")
    files = sorted(p for p in staging.rglob("*") if p.is_file())
    manifest = {**info, "files": [{"path": p.relative_to(staging).as_posix(), "bytes": p.stat().st_size,
                                   "sha256": _sha256(p)} for p in files]}
    manifest["total_bytes"] = sum(f["bytes"] for f in manifest["files"])
    _pretty(staging / "manifest.json", manifest)

    # swap in
    old = out.with_name(out.name + ".old")
    shutil.rmtree(old, ignore_errors=True)
    if out.exists():
        os.replace(out, old)
    os.replace(staging, out)
    shutil.rmtree(old, ignore_errors=True)

    zinfo = None
    if zip_path is not None:
        zinfo = write_zip(out, Path(zip_path))
        if zinfo["bytes"] > budget:
            raise RuntimeError(f"zip is {zinfo['bytes'] / 1e6:.1f} MB, over the {budget_mb:.0f} MB budget")
    return {**info, "out": str(out), "total_bytes": manifest["total_bytes"], "files": len(manifest["files"]),
            "zip": zinfo, "seconds": round(time.monotonic() - t0, 1)}


def write_zip(folder: Path, zip_path: Path) -> dict:
    """folder -> zip_path (via a temporary file); images stored, text deflated; paths under gis_sample/."""
    tmp = zip_path.with_name(zip_path.name + ".tmp")
    tmp.unlink(missing_ok=True)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(tmp, "w") as z:
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                method = zipfile.ZIP_STORED if p.suffix.lower() in (".jpg", ".png") else zipfile.ZIP_DEFLATED
                z.write(p, f"gis_sample/{p.relative_to(folder).as_posix()}", compress_type=method)
    with zipfile.ZipFile(tmp) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError(f"zip check failed at {bad}")
        n = len(z.namelist())
    os.replace(tmp, zip_path)
    return {"path": str(zip_path), "bytes": zip_path.stat().st_size, "entries": n, "sha256": _sha256(zip_path)}


def readme(i: dict) -> str:
    full = "\n".join(f"  - `images/full/{s}.png`" for s in i["full_pngs"]) or "  - (none)"
    notes = "\n".join(f"- {n}" for n in i["notes"]) or "- none"
    return f"""# GIS sample bundle (drive-video survey + citizen uploads)

Exported {i['generated_at']} from the VLM PC by `scripts/export_gis_sample.py`. Contract: `GIS_VIDEO_HANDOFF.md`
(sent with this bundle). Schema version `{i['schema']}`.

Data in this snapshot: {i['sessions']} drive session(s), {i['segments']} segments
({', '.join(f'{k} {v}' for k, v in i['segments_by_status'].items())}), {i['answered']} classified by the VLM,
{i['images']} stored frames; {i['citizen_features']} citizen uploads (backend: {i['citizen_backend']}).
The survey was still being classified when some sessions were exported: segments with `vlm_class = "pending"`
have a frame but no VLM answer yet. That is a real state the WebGIS must show.

## Run the mock API (Python 3.9+, standard library only)

```
python mock_server.py                      # API at http://127.0.0.1:8765/gis/meta
python mock_server.py --app ..\\webgis      # and your app at http://127.0.0.1:8765/gis/app/
```

The mock answers the same URLs, query parameters, errors and JSON shapes as the real server, using the
same filter code (`gis_core.py` is a copy of the server's `app/gis_core.py`). Serve your app through
`--app` and use relative URLs (`/gis/...`): that is exactly how it will run on the VLM PC
(the server mounts `webgis/` at `/gis/app/`). `python -m http.server` also works for the JSON files
under `api/`, but cannot filter, page, or map `?w=` image URLs: use `mock_server.py`.

## What is here

| Path | What |
|---|---|
| `api/gis/meta.json` | `GET /gis/meta`: labels, colours, marker sizes, statuses, extents |
| `api/gis/video/segments.geojson` | `GET /gis/video/segments.geojson?limit=20000` (footprint polygons, every segment) |
| `api/gis/video/segments_point.geojson`, `segments_line.geojson` | same with `geom=point` / `geom=line` |
| `api/gis/video/sessions.json` | `GET /gis/video/sessions?track=true&per_km=true` |
| `api/gis/video/summary.json` | `GET /gis/video/summary` |
| `api/gis/video/segments.csv` | `GET /gis/video/segments.csv?limit=20000` |
| `api/gis/citizen/assessments.geojson` | `GET /gis/citizen/assessments.geojson?include_review=true&limit=20000` |
| `examples/` | pretty-printed: one feature per `vlm_class`, a citizen feature, a session, and `queries.json` (filtered requests with their answers, including a 400) |
| `images/w480/<segment_id>.jpg` | `GET /gis/image/<id>?w=480` for every stored frame (list / popup thumbnails) |
| `images/w1280/<segment_id>.jpg` | `GET /gis/image/<id>?w=1280` (detail panel) |
| `images/full/` | `GET /gis/image/<id>` full 3840x2160 PNG, only these: |
{full}
| `schema.sql` | `pg_dump --schema-only` of the database (reference; you do not need a database) |
| `mock_server.py`, `gis_core.py` | the mock |
| `manifest.json` | every file with bytes and SHA-256 |

## Mock vs real server

- `generated_at` is the export time.
- Previews: the real server snaps `?w=` up to 160/320/480/640/960/1280/1600/1920 px; the mock has only
  480 and 1280 (`w` <= 480 gets the 480 file, anything larger the 1280 file).
- Full PNG without `w`: the mock returns the PNG for the files listed above, otherwise the 1280 JPEG.
- Citizen images are Cloudinary URLs (internet needed).

Notes for this export:
{notes}
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Export the portable GIS sample bundle.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--full-pngs", type=int, default=10)
    ap.add_argument("--budget-mb", type=float, default=300.0)
    ap.add_argument("--no-citizen", action="store_true", help="leave the citizen layer empty")
    ap.add_argument("--citizen-addresses", action="store_true", help="keep citizen addresses in the bundle")
    a = ap.parse_args(argv)
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env", override=False)
    except ImportError:
        pass
    if not os.environ.get("DATABASE_URL"):
        print("DATABASE_URL is not set (.env): nothing to export.")
        return 2
    try:
        r = export(a.out, None if a.no_zip else a.zip, full_pngs=a.full_pngs, budget_mb=a.budget_mb,
                   citizen=not a.no_citizen, citizen_addresses=a.citizen_addresses)
    except Exception as e:
        print(f"EXPORT FAILED: {type(e).__name__}: {e}")
        return 1
    print(f"bundle   {r['out']}: {r['files']} files, {r['total_bytes'] / 1e6:.1f} MB ({r['seconds']} s)")
    print(f"data     {r['sessions']} session(s), {r['segments']} segments, {r['answered']} answered, "
          f"{r['images']} images, {len(r['full_pngs'])} full PNGs, {r['citizen_features']} citizen features")
    if r["zip"]:
        print(f"zip      {r['zip']['path']}: {r['zip']['bytes'] / 1e6:.1f} MB, {r['zip']['entries']} entries, "
              f"sha256 {r['zip']['sha256']}")
    for n in r["notes"]:
        print(f"note     {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
