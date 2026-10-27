"""
Offline mock of the read-only GIS API (/gis/*), for developing the WebGIS without our database or GPU.

Shipped in the GIS sample bundle as mock_server.py, next to gis_core.py (a copy of
app/gis_core.py: the SAME filter / aggregate code the real API runs) and the saved API
responses under api/. Standard library only (Python 3.9+).

    python mock_server.py                          # http://127.0.0.1:8765/gis/meta
    python mock_server.py --app ..\\webgis          # also serves your app at http://127.0.0.1:8765/gis/app/
    python mock_server.py --port 9000 --host 0.0.0.0

Same URLs, query parameters, errors (400 with {"detail": ...}) and response shapes as the
real server, so the WebGIS uses one base URL ("" when served by the same server) in both places:

    /gis/meta                              api/gis/meta.json
    /gis/video/segments.geojson?...        api/gis/video/segments{,_point,_line}.geojson, filtered + paged here
    /gis/video/segments.csv?...            built here from the point file (gis_core.segments_csv)
    /gis/video/sessions?session=&track=&per_km=   api/gis/video/sessions.json (track / per_km dropped unless asked)
    /gis/video/summary?...                 computed here (gis_core.summary_document)
    /gis/citizen/assessments.geojson?...   api/gis/citizen/assessments.geojson, filtered + paged here
    /gis/image/<segment_id>?w=N            images/w480/<id>.jpg when N snaps to <= 480, else images/w1280/<id>.jpg
    /gis/image/<segment_id>                images/full/<id>.png when the sample has it (10 do), else the w1280 JPEG

Differences from the real server (by design, documented in README.md): `generated_at` is the
time the sample was exported; previews exist only at 480 and 1280 px; most full PNGs are absent.
Every response carries Access-Control-Allow-Origin: * so a dev server on another port works too.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import posixpath
import re
import sys
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:
    import gis_core as core                      # the bundle: gis_core.py next to this file
except ImportError:                              # the Capstone repository: scripts/ -> app/
    sys.path.insert(0, str(HERE.parent))
    from app import gis_core as core

_SEGMENT_ID = re.compile(r"^[A-Za-z0-9._-]{1,200}$")


class ApiError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


class MockData:
    """The saved responses, loaded once."""

    def __init__(self, root: Path):
        self.root = root
        api = root / "api" / "gis"
        self.meta = (api / "meta.json").read_bytes()
        self.segments = {g: json.loads((api / "video" / f"segments{'' if g == 'footprint' else '_' + g}.geojson")
                                       .read_text(encoding="utf-8")) for g in ("footprint", "point", "line")}
        self.sessions = json.loads((api / "video" / "sessions.json").read_text(encoding="utf-8"))
        self.citizen = json.loads((api / "citizen" / "assessments.geojson").read_text(encoding="utf-8"))

    def image(self, segment_id: str, w):
        if not _SEGMENT_ID.match(segment_id):
            return None
        img = self.root / "images"
        if w is None:
            full = img / "full" / f"{segment_id}.png"
            if full.is_file():
                return full, "image/png"
            w = 1280
        folder = "w480" if core.snap_width(w) <= 480 else "w1280"
        p = img / folder / f"{segment_id}.jpg"
        if not p.is_file() and folder == "w1280":
            p = img / "w480" / f"{segment_id}.jpg"
        return (p, "image/jpeg") if p.is_file() else None


def _query(qs: str) -> dict:
    """Last value wins (as FastAPI's request.query_params.get)."""
    return {k: v[-1] for k, v in urllib.parse.parse_qs(qs, keep_blank_values=True).items()}


def _bool_param(q: dict, name: str, default: bool) -> bool:
    v = (q.get(name) or "").strip().lower()
    if not v:
        return default
    if v in ("true", "1", "yes", "false", "0", "no"):
        return v in ("true", "1", "yes")
    raise ApiError(400, f"{name} must be true or false")


def _filters(q: dict, layer: str, allow=()) -> "core.Filters":
    try:
        return core.parse_filters(q, layer, allow=allow)
    except core.FilterError as e:
        raise ApiError(400, str(e)) from None


def route(data: MockData, path: str, q: dict):
    """(status, content type, body bytes) for one API request."""
    def js(obj, media="application/json"):
        return 200, media, json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    if path == "/gis/meta":
        return 200, "application/json", data.meta
    if path == "/gis/video/segments.geojson":
        geom = (q.get("geom") or "footprint").strip().lower()
        if geom not in ("footprint", "point", "line"):
            raise ApiError(400, "geom must be footprint, point or line")
        f = _filters(q, "video", allow=("geom",))
        saved = data.segments[geom]
        page, total = core.apply(saved["features"], f)
        return js(core.feature_collection(page, total, f, "video", generated_at=saved["generated_at"], geom=geom),
                  "application/geo+json")
    if path == "/gis/video/segments.csv":
        f = _filters(q, "video")
        page, _ = core.apply(data.segments["point"]["features"], f)
        return 200, "text/csv; charset=utf-8", core.segments_csv(page).encode("utf-8")
    if path == "/gis/video/summary":
        f = _filters(q, "video")
        saved = data.segments["point"]
        return js(core.summary_document(saved["features"], f, generated_at=saved["generated_at"]))
    if path == "/gis/video/sessions":
        unknown = sorted(k for k in q if k not in ("session", "track", "per_km") and not k.startswith("_"))
        if unknown:
            raise ApiError(400, f"unknown parameter(s): {', '.join(unknown)}; allowed: session, track, per_km")
        wanted = {s.strip() for s in (q.get("session") or "").split(",") if s.strip()}
        track, with_km = _bool_param(q, "track", False), _bool_param(q, "per_km", True)
        out = []
        for s in data.sessions["sessions"]:
            if wanted and s["session_id"] not in wanted:
                continue
            s = dict(s)
            if not track:
                s.pop("track", None)
            if not with_km:
                s.pop("per_km", None)
            out.append(s)
        return js({"schema": data.sessions["schema"], "generated_at": data.sessions["generated_at"],
                   "count": len(out), "sessions": out})
    if path == "/gis/citizen/assessments.geojson":
        f = _filters(q, "citizen")
        saved = data.citizen
        page, total = core.apply(saved["features"], f)
        extra = {k: saved[k] for k in ("backend", "sample_note") if k in saved}
        return js(core.feature_collection(page, total, f, "citizen", generated_at=saved["generated_at"], **extra),
                  "application/geo+json")
    if path.startswith("/gis/image/"):
        seg = urllib.parse.unquote(path[len("/gis/image/"):])
        w = q.get("w")
        if w is not None:
            try:
                w = int(w)
            except ValueError:
                raise ApiError(422, "w must be an integer") from None
            if w < 1:
                raise ApiError(400, "w must be a positive width in pixels")
        found = data.image(seg, w)
        if found is None:
            raise ApiError(404, "no stored image for this segment")
        p, media = found
        return 200, media, p.read_bytes()
    raise ApiError(404, "Not Found")


def make_handler(data: MockData, app_dir: Path | None):
    class Handler(BaseHTTPRequestHandler):
        server_version = "GisMock/1"

        def _send(self, status: int, media: str, body: bytes, extra: dict | None = None):
            self.send_response(status)
            self.send_header("Content-Type", media)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "public, max-age=86400" if media.startswith("image/") else "no-cache")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.end_headers()

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            url = urllib.parse.urlsplit(self.path)
            path = url.path
            if path in ("/", "/gis", "/gis/") and app_dir is not None:
                return self._send(302, "text/plain", b"", {"Location": "/gis/app/"})
            if path == "/gis/app" and app_dir is not None:
                return self._send(301, "text/plain", b"", {"Location": "/gis/app/"})
            if path.startswith("/gis/app/") and app_dir is not None:
                return self._static(path[len("/gis/app/"):])
            try:
                status, media, body = route(data, path, _query(url.query))
            except ApiError as e:
                status, media = e.status, "application/json"
                body = json.dumps({"detail": e.detail}).encode("utf-8")
            except Exception as e:                      # never kill the mock on a bad request
                status, media = 500, "application/json"
                body = json.dumps({"detail": f"mock error: {type(e).__name__}: {e}"}).encode("utf-8")
            extra = {}
            if path == "/gis/video/segments.csv" and status == 200:
                extra["Content-Disposition"] = 'attachment; filename="gis_video_segments_sample.csv"'
            self._send(status, media, body, extra)

        def _static(self, rel: str):
            rel = posixpath.normpath(urllib.parse.unquote(rel or "index.html")).lstrip("/")
            if rel in (".", ""):
                rel = "index.html"
            p = (app_dir / rel).resolve()
            if app_dir not in p.parents and p != app_dir:
                return self._send(404, "application/json", b'{"detail":"Not Found"}')
            if p.is_dir():
                p = p / "index.html"
            if not p.is_file():
                return self._send(404, "application/json", b'{"detail":"Not Found"}')
            media = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
            if p.suffix == ".js":
                media = "text/javascript"
            self._send(200, media, p.read_bytes())

        def log_message(self, fmt, *args):
            sys.stderr.write("%s %s\n" % (self.log_date_time_string(), fmt % args))

    return Handler


def serve(data_dir: Path, host: str, port: int, app_dir: Path | None = None) -> ThreadingHTTPServer:
    """A started-but-not-serving server (call .serve_forever()); used by the tests too."""
    data = MockData(data_dir)
    return ThreadingHTTPServer((host, port), make_handler(data, app_dir.resolve() if app_dir else None))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Offline mock of the GIS API (/gis/*) over the sample bundle.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--data", type=Path, default=HERE, help="the sample bundle folder (default: this folder)")
    ap.add_argument("--app", type=Path, default=None, help="your WebGIS folder, served at /gis/app/")
    a = ap.parse_args(argv)
    if a.app is not None and not (a.app / "index.html").is_file():
        print(f"--app {a.app}: no index.html there")
        return 2
    httpd = serve(a.data, a.host, a.port, a.app)
    base = f"http://{a.host}:{a.port}"
    print(f"GIS mock API on {base}/gis/meta" + (f"  |  WebGIS on {base}/gis/app/" if a.app else ""))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
