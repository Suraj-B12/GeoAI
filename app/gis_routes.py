"""
Read-only GIS API for the drive-video survey and the citizen uploads (the WebGIS's only data source).

    GET /gis/meta                              labels (IRC:82-2015 + Normal) with categories, colours, marker
                                               sizes; statuses; data extents and counts; schema version
    GET /gis/video/segments.geojson            one Feature per 5 m segment, filters (app/gis_filters.py),
                                               geom=footprint (default) | point | line
    GET /gis/video/segments.csv                the same rows and filters as CSV (UTF-8 with BOM)
    GET /gis/video/sessions                    per drive: road, times, lengths, counts, % of surveyed
                                               length per label, per-km table; track=true adds the drive line
    GET /gis/video/summary                     totals for the stats bar (same filters as segments)
    GET /gis/image/{segment_id}[?w=480]        the stored segment frame: full PNG, or a JPEG preview
    GET /gis/citizen/assessments.geojson       RoadSide uploads from the backend this server runs on
                                               (Supabase by default, or the local PostgreSQL)

Contract: GIS_VIDEO_HANDOFF (paper/GIS_VIDEO_HANDOFF.md). Every response carries
`schema: "geoai.gis.v1"`; fields are only ever added under that version.

Read-only by construction: SELECT statements only, every value a bound parameter
(psycopg placeholders, PostgREST query values URL-encoded by httpx). The video
side reads the local PostgreSQL through app.video_store.connect() (DATABASE_URL)
in a worker thread; if it is unreachable the endpoints answer 503 and nothing
else on the server is affected. The citizen side uses the client the server
already built (app.state.supabase_client: the Supabase httpx client, or a
LocalDb with DB_BACKEND=postgres) and caches its rows for CITIZEN_TTL_S, because
a map that polls must not pull every row from Supabase each time (T7.3).

Open, like /local-images: the browser loads these from <img> tags and from a
static page that holds no secret, so there is no X-API-Key. Nothing here writes
to a database; the only file writes are JPEG previews in a disk cache
(GIS_PREVIEW_CACHE, default <LOCAL_IMAGE_ROOT>\\..\\gis_cache), keyed by the
image's SHA-256 and width.

Images resolve ONLY through video_segments.image_path, and only to .png/.jpg files
inside LOCAL_IMAGE_ROOT (app.local_images.inside_root); anything else is 404.

install(app) includes the router and, when webgis/index.html exists (or
GIS_APP_DIR points at a folder with one), serves that folder at /gis/app.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import re
import threading
import time
import urllib.parse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response

from app.gis_core import (DEFAULT_LIMIT, DISTRESS_LABELS, LABELS, MARKER_PRIORITY, MAX_LIMIT, PREVIEW_WIDTHS,
                          SCHEMA_VERSION, SEVERITIES, STATUSES, FilterError, Filters, aggregate, apply, bbox_of,
                          canonical_label, feature_collection, iso_utc, lonlat, marker_label, now_utc, num,
                          parse_filters, parse_time, per_km, segments_csv, snap_width, summary_document,
                          track_geometry)
from app.local_images import image_root, inside_root, local_image_path
from app.supabase_client import is_local_db

PROJECT_ROOT = Path(__file__).resolve().parent.parent

router = APIRouter(prefix="/gis", tags=["gis"])

CITIZEN_TTL_S = 60.0                # citizen rows are re-read at most once a minute
CITIZEN_MAX_ROWS = 20000
CONNECT_TIMEOUT_S = 3              # a stopped database is a quick 503, not a 2-minute hang
PREVIEW_QUALITY = 80
_IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
_SEGMENT_ID = re.compile(r"^[A-Za-z0-9._-]{1,200}$")
_PREVIEW_SLOTS = threading.BoundedSemaphore(2)   # 4K PNG decodes at once (CPU; the GPU work is unaffected)
_JSON = "application/json"
_GEOJSON = "application/geo+json"
_NO_CACHE = {"Cache-Control": "no-cache"}


# ============================================================
# Display conventions (served by /gis/meta; the WebGIS must not hard-code them)
# ============================================================

LABEL_COLORS = {
    # Surface defects (IRC:82 7.2) - violets
    "Bleeding": "#6d28d9", "Streaking": "#a855f7", "Hungry Surface": "#d946ef",
    # Cracks (7.3) - yellow to red, alligator the strongest
    "Hairline Cracks": "#eab308", "Longitudinal Cracking": "#f59e0b", "Transverse Cracking": "#f97316",
    "Edge Cracking": "#c2410c", "Alligator Cracking": "#dc2626",
    # Deformation (7.4) - blues / teals
    "Slippage": "#0ea5e9", "Rutting": "#2563eb", "Corrugation": "#0891b2", "Shoving": "#1e40af",
    "Shallow Depression": "#14b8a6", "Settlement": "#0f766e",
    # Disintegration (7.5) - browns / crimson, potholes darkest
    "Stripping": "#a16207", "Ravelling": "#92400e", "Potholes": "#7f1d1d", "Edge Breaking": "#be123c",
    "Normal": "#16a34a",
}
OTHER_COLOR = "#6b7280"            # a label outside the list, or answered without a valid label
SEVERITY_RADIUS_PX = {"None": 4, "Low": 6, "Medium": 9, "High": 13, "Unknown": 4}
CLASS_STYLE = {
    "distress": {"color": None, "meaning": "answered; at least one IRC:82 distress label (colour by marker_label)"},
    "normal": {"color": LABEL_COLORS["Normal"], "meaning": "answered; Normal"},
    "unknown": {"color": OTHER_COLOR, "meaning": "answered, but without a valid label"},
    "pending": {"color": "#9ca3af", "meaning": "sent to the VLM, answer not stored yet"},
    "not_sent": {"color": "#d1d5db", "meaning": "covered, the frame-level gate said skip (not classified)"},
    "not_covered": {"color": "#111827", "meaning": "never imaged (GNSS gap, stopped, black frames): no data"},
}
STATUS_MEANING = {
    "send": "the gate found possible distress; the best frame went to the VLM",
    "audit": "the gate found none, but a random 10 % sample goes to the VLM to measure the gate",
    "skip": "the gate found none; not sent to the VLM (no image stored)",
    "not_covered": "no usable frame for this 5 m (no image, no result)",
}


def _taxonomy() -> dict:
    """{label: {category, irc_section, description}} from scripts/irc82_taxonomy.py (single source of truth)."""
    try:
        from scripts.irc82_taxonomy import IRC82_DISTRESS_TAXONOMY as tax
    except Exception:   # the API still works without the descriptions
        tax = {}
    out = {}
    for lab in DISTRESS_LABELS:
        e = tax.get(lab, {})
        desc = str(e.get("description") or "")
        first = desc.split(". ")[0].strip()
        out[lab] = {"category": e.get("category"), "irc_section": e.get("irc_section"),
                    "description": (first + ".") if first and not first.endswith(".") else first}
    out["Normal"] = {"category": "Normal", "irc_section": None,
                     "description": "No distress visible (Stage 1 said Normal)."}
    return out


# ============================================================
# Database plumbing
# ============================================================

class _Unavailable(Exception):
    pass


def _query(fn: Callable) -> Any:
    """Run fn(cursor) on a fresh local-PostgreSQL connection (dict rows); _Unavailable if it can't."""
    import psycopg
    from psycopg.rows import dict_row

    from app import video_store
    try:
        conn = video_store.connect(connect_timeout=CONNECT_TIMEOUT_S)
    except RuntimeError as e:                       # DATABASE_URL not set
        raise _Unavailable(str(e)) from None
    except psycopg.OperationalError as e:
        print(f"[GIS] local PostgreSQL unreachable: {type(e).__name__}")
        raise _Unavailable("local PostgreSQL is unreachable") from None
    try:
        with conn.cursor(row_factory=dict_row) as cur:
            return fn(cur)
    except psycopg.errors.UndefinedTable:
        raise _Unavailable("the video tables are missing (scripts/setup_local_db.py applies migration 005)") from None
    except psycopg.OperationalError as e:
        print(f"[GIS] local PostgreSQL error: {type(e).__name__}")
        raise _Unavailable("local PostgreSQL connection failed") from None
    finally:
        conn.close()


def _filters(request: Request, layer: str, allow: Iterable[str] = ()) -> Filters:
    try:
        return parse_filters(request.query_params, layer, allow=allow)
    except FilterError as e:
        raise HTTPException(status_code=400, detail=str(e))


def _bool_param(q, name: str, default: bool) -> bool:
    v = (q.get(name) or "").strip().lower()
    if not v:
        return default
    if v in ("true", "1", "yes", "false", "0", "no"):
        return v in ("true", "1", "yes")
    raise HTTPException(status_code=400, detail=f"{name} must be true or false")


def _json(obj: Any, media: str = _JSON) -> Response:
    return Response(content=json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
                    media_type=media, headers=_NO_CACHE)


# ============================================================
# Small value helpers
# ============================================================

def _obj(x) -> dict:
    return x if isinstance(x, dict) else {}


def _str_list(x) -> list[str]:
    return [str(v) for v in x if isinstance(v, str) and v.strip()] if isinstance(x, list) else []


def _bool(x) -> bool:
    if isinstance(x, str):
        return x.strip().lower() == "true"
    return bool(x)


def _ring(points) -> Optional[list]:
    """A valid RFC 7946 exterior ring from [[lon, lat], ...]: closed, >= 4 positions, counter-clockwise."""
    if not isinstance(points, list):
        return None
    ring = []
    for p in points:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            return None
        q = lonlat(p[0], p[1])
        if q is None:
            return None
        if not ring or q != ring[-1]:
            ring.append(q)
    if len(ring) >= 2 and ring[0] == ring[-1]:
        ring.pop()
    if len({tuple(p) for p in ring}) < 3:
        return None
    area2 = sum(ring[i][0] * ring[(i + 1) % len(ring)][1] - ring[(i + 1) % len(ring)][0] * ring[i][1]
                for i in range(len(ring)))
    if area2 == 0:
        return None
    if area2 < 0:                       # clockwise -> right-hand rule (RFC 7946 section 3.1.6)
        ring.reverse()
    return ring + [ring[0]]


# ============================================================
# Shared classification fields (both layers: same marker semantics)
# ============================================================

def _classification(stage1: Optional[str], types, severity, s1c, s2c, needs, reviewed, corr_types,
                    corr_severity, description) -> dict:
    """The fields both layers share, from an answer (stage1 None = not answered)."""
    if stage1 is None:
        return {"stage1_label": None, "stage1_confidence": None, "stage2_confidence": None, "confidence": None,
                "labels": [], "primary_label": None, "severity": None, "description": None,
                "needs_expert_review": False, "expert_reviewed": False, "expert_corrected_types": None,
                "expert_corrected_severity": None, "final_labels": [], "final_primary_label": None,
                "marker_label": None, "final_severity": None}
    labels = ["Normal"] if stage1 == "Normal" else _str_list(types)
    severity = severity if severity in SEVERITIES else ("None" if stage1 == "Normal" else "Unknown")
    corrected = _str_list(corr_types)
    final = corrected or labels
    corr_sev = corr_severity if isinstance(corr_severity, str) and corr_severity.strip() else None
    s1c, s2c = num(s1c, 4), num(s2c, 4)
    return {"stage1_label": stage1, "stage1_confidence": s1c, "stage2_confidence": s2c,
            # the brief's rule: Stage 2's confidence when Stage 2 ran, else Stage 1's
            "confidence": s2c if stage1 == "Distress" and s2c else s1c,
            "labels": labels, "primary_label": labels[0] if labels else None,
            "severity": severity, "description": description or "",
            "needs_expert_review": _bool(needs), "expert_reviewed": _bool(reviewed),
            "expert_corrected_types": corrected or None, "expert_corrected_severity": corr_sev,
            "final_labels": final, "final_primary_label": final[0] if final else None,
            "marker_label": marker_label(final), "final_severity": corr_sev or severity}


def _answer_class(c: dict) -> str:
    valid = {canonical_label(x) for x in c["final_labels"]} - {None}
    if valid - {"Normal"}:
        return "distress"
    if valid == {"Normal"}:
        return "normal"
    return "unknown"


# ============================================================
# Video segments
# ============================================================

_SEGMENTS_SQL = """
SELECT g.segment_id, g.session_id, g.k, g.status, g.length_m, g.s0_m, g.s1_m, g.centre_lat, g.centre_lon,
       g.chainage_start_km, g.chainage_end_km, g.utc, g.image_path IS NOT NULL AS has_image,
       g.record->'segment' AS seg, g.record->'frame' AS frame, g.record->'camera' AS camera,
       g.record->'target' AS target, g.record->'quality' AS quality, g.record->'gate' AS gate,
       g.record->'survey' AS survey, g.record->'capture' AS capture, g.record->'device' AS device,
       g.vlm_response IS NOT NULL AS has_answer,
       g.vlm_response->>'stage1_label' AS v_stage1_label, g.vlm_response->>'stage1_confidence' AS v_s1c,
       g.vlm_response->'distress_types' AS v_types, g.vlm_response->>'severity' AS v_severity,
       g.vlm_response->>'description' AS v_description, g.vlm_response->>'stage2_confidence' AS v_s2c,
       g.vlm_response->>'needs_expert_review' AS v_needs, g.vlm_response->'condition_indicators' AS v_ci,
       a.id::text AS assessment_id, a.status AS assessment_status, a.stage1_label, a.stage1_confidence,
       a.distress_types, a.severity, a.description, a.stage2_confidence, a.needs_expert_review,
       a.expert_reviewed, a.expert_corrected_types, a.expert_corrected_severity, a.reviewed_at,
       a.processed_at, a.raw_response->'condition_indicators' AS a_ci
FROM video_segments g
LEFT JOIN assessments a ON a.id = g.assessment_id
{where}
ORDER BY g.session_id, g.k
"""


def _fetch_segment_rows(sessions: Optional[Iterable[str]] = None) -> list[dict]:
    where, params = "", ()
    if sessions:
        where, params = "WHERE g.session_id = ANY(%s)", (sorted(sessions),)
    return _query(lambda cur: cur.execute(_SEGMENTS_SQL.format(where=where), params).fetchall())


def _segment_feature(r: dict, geom: str = "footprint") -> dict:
    """One video_segments row (+ its assessment) as a GeoJSON Feature with flat properties."""
    seg, frame, cam = _obj(r["seg"]), _obj(r["frame"]), _obj(r["camera"])
    target, quality, gate = _obj(r["target"]), _obj(r["quality"]), _obj(r["gate"])
    survey, capture, device = _obj(r["survey"]), _obj(r["capture"]), _obj(r["device"])
    scores = _obj(gate.get("scores"))

    if r["assessment_id"] is not None:
        c = _classification(r["stage1_label"], r["distress_types"], r["severity"], r["stage1_confidence"],
                            r["stage2_confidence"], r["needs_expert_review"], r["expert_reviewed"],
                            r["expert_corrected_types"], r["expert_corrected_severity"], r["description"])
        ci = r["a_ci"]
    elif r["has_answer"] and r["v_stage1_label"] in ("Normal", "Distress"):
        # The answer is stored but its assessment was deleted: show the answer as the store would have.
        from scripts.utils import normalize_severity
        distressed = r["v_stage1_label"] == "Distress"
        c = _classification(r["v_stage1_label"], r["v_types"] if distressed else [],
                            normalize_severity(r["v_severity"]) if distressed else "None", r["v_s1c"], r["v_s2c"],
                            r["v_needs"], False, None, None, r["v_description"])
        ci = r["v_ci"]
    else:
        c = _classification(None, None, None, None, None, None, None, None, None, None)
        ci = None
    answered = c["stage1_label"] is not None

    status = r["status"]
    if status == "not_covered":
        vlm_class = "not_covered"
    elif status == "skip":
        vlm_class = "not_sent"
    elif not answered:
        vlm_class = "pending"
    else:
        vlm_class = _answer_class(c)

    centre = lonlat(r["centre_lon"], r["centre_lat"])
    if geom == "point":
        geometry = {"type": "Point", "coordinates": centre} if centre else None
    elif geom == "line":
        a, b = lonlat(_obj(seg.get("start")).get("lon"), _obj(seg.get("start")).get("lat")), \
            lonlat(_obj(seg.get("end")).get("lon"), _obj(seg.get("end")).get("lat"))
        geometry = ({"type": "LineString", "coordinates": [a, b]} if a and b and a != b
                    else ({"type": "Point", "coordinates": centre} if centre else None))
    else:
        ring = _ring(seg.get("footprint_lonlat"))
        geometry = ({"type": "Polygon", "coordinates": [ring]} if ring
                    else ({"type": "Point", "coordinates": centre} if centre else None))

    sid = r["segment_id"]
    img = f"/gis/image/{urllib.parse.quote(sid)}" if r["has_image"] else None
    speed = num(cam.get("speed_mps"), 2)
    props = {
        "layer": "video",
        "segment_id": sid, "session_id": r["session_id"], "k": r["k"], "status": status,
        "gate_mode": gate.get("mode"), "gate_decision": gate.get("decision"),
        "gate_decision_if_gated": gate.get("decision_if_gated"), "gate_reasons": _str_list(gate.get("reasons")),
        "gate_probe_frac": num(scores.get("probe_frac"), 4), "gate_probe_max_logit": num(scores.get("probe_max_logit"), 3),
        "gate_evidence": num(scores.get("evidence"), 1), "gate_crack_mm": num(scores.get("crack_mm"), 1),
        "s0_m": num(r["s0_m"], 2), "s1_m": num(r["s1_m"], 2), "length_m": num(r["length_m"], 3),
        "partial": bool(seg.get("partial")),
        "chainage_start_km": num(r["chainage_start_km"], 4), "chainage_end_km": num(r["chainage_end_km"], 4),
        "utc": iso_utc(r["utc"]), "local_time": frame.get("local_iso"),
        "centre_lat": centre[1] if centre else None, "centre_lon": centre[0] if centre else None,
        "position_sigma_m": num(target.get("sigma_m"), 2),
        "camera_lat": num(cam.get("lat"), 7), "camera_lon": num(cam.get("lon"), 7),
        "camera_heading_deg": num(cam.get("heading_deg"), 1), "camera_speed_mps": speed,
        "camera_speed_kmh": round(speed * 3.6, 1) if speed is not None else None,
        "camera_h_sigma_m": num(cam.get("h_sigma_m"), 2),
        "road_id": (survey.get("road_id") or "").strip() or None,
        "road_name": (survey.get("road_name") or "").strip() or None,
        "direction": survey.get("direction") or None, "lane": survey.get("lane") or None,
        "device_model": device.get("model"),
        "answered": answered, "assessment_id": r["assessment_id"], "assessment_status": r["assessment_status"],
        "vlm_class": vlm_class,
        **c,
        "condition_indicators": _str_list(ci),
        "reviewed_at": iso_utc(r["reviewed_at"]) if r["reviewed_at"] else None,
        "processed_at": iso_utc(r["processed_at"]) if r["processed_at"] else None,
        "quality_sharpness": num(quality.get("sharpness"), 5), "quality_sharpness_rank": quality.get("sharpness_rank"),
        "quality_n_candidates": quality.get("n_candidates"), "quality_blur_px": num(quality.get("blur_px"), 2),
        "quality_reason": quality.get("reason"), "low_light": bool(capture.get("low_light")),
        "image_width": frame.get("width") if r["has_image"] else None,
        "image_height": frame.get("height") if r["has_image"] else None,
        "has_image": bool(r["has_image"]),
        "image_url": img, "preview_url": f"{img}?w=1280" if img else None, "thumb_url": f"{img}?w=480" if img else None,
    }
    return {"type": "Feature", "id": sid, "geometry": geometry, "properties": props}


async def _video_features(sessions=None, geom: str = "footprint") -> list[dict]:
    try:
        rows = await asyncio.to_thread(_fetch_segment_rows, sessions)
    except _Unavailable as e:
        raise HTTPException(status_code=503, detail=f"GIS data unavailable: {e}")
    return [_segment_feature(r, geom) for r in rows]


def _geom_param(request: Request) -> str:
    g = (request.query_params.get("geom") or "footprint").strip().lower()
    if g not in ("footprint", "point", "line"):
        raise HTTPException(status_code=400, detail="geom must be footprint, point or line")
    return g


@router.get("/video/segments.geojson")
async def video_segments_geojson(request: Request):
    """Video segments as GeoJSON (footprint polygons by default; geom=point|line)."""
    geom = _geom_param(request)
    f = _filters(request, "video", allow=("geom",))
    feats = await _video_features(f.sessions, geom)
    page, total = apply(feats, f)
    return _json(feature_collection(page, total, f, "video", geom=geom), _GEOJSON)


@router.get("/video/segments.csv")
async def video_segments_csv(request: Request):
    """The filtered segments as CSV for spreadsheets (UTF-8 with BOM so Excel reads it as UTF-8)."""
    f = _filters(request, "video")
    page, _ = apply(await _video_features(f.sessions, "point"), f)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Response(content=segments_csv(page).encode("utf-8"), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="gis_video_segments_{stamp}.csv"',
                             **_NO_CACHE})


# ============================================================
# Aggregates: sessions, summary (analyze.py definitions)
# ============================================================

_SESSIONS_SQL = """
SELECT session_id, state, error IS NOT NULL AS has_error, created_at, updated_at, stored_at, cleaned_at
FROM video_sessions {where} ORDER BY session_id
"""


@router.get("/video/sessions")
async def video_sessions(request: Request):
    """Per drive session: road, times, lengths, counts by status and label, % of surveyed length per label,
    per-km table; track=true adds the drive line (GeoJSON geometry)."""
    q = request.query_params
    unknown = sorted(k for k in q.keys() if k not in ("session", "track", "per_km") and not k.startswith("_"))
    if unknown:
        raise HTTPException(status_code=400, detail=f"unknown parameter(s): {', '.join(unknown)}; "
                                                    f"allowed: session, track, per_km")
    sessions = [s.strip() for s in (q.get("session") or "").split(",") if s.strip()] or None
    track, with_km = _bool_param(q, "track", False), _bool_param(q, "per_km", True)
    where, params = ("WHERE session_id = ANY(%s)", (sessions,)) if sessions else ("", ())
    try:
        srows, rows = await asyncio.to_thread(
            lambda: (_query(lambda cur: cur.execute(_SESSIONS_SQL.format(where=where), params).fetchall()),
                     _fetch_segment_rows(sessions)))
    except _Unavailable as e:
        raise HTTPException(status_code=503, detail=f"GIS data unavailable: {e}")
    feats = [_segment_feature(r, "point") for r in rows]
    by_session = defaultdict(list)
    for ft in feats:
        by_session[ft["properties"]["session_id"]].append(ft["properties"])
    out = []
    for s in srows:
        props = by_session.get(s["session_id"], [])
        first = props[0] if props else {}
        times = [p["utc"] for p in props if p["utc"]]
        item = {
            "session_id": s["session_id"], "state": s["state"], "has_error": s["has_error"],
            "stored_at": iso_utc(s["stored_at"]) if s["stored_at"] else None,
            "updated_at": iso_utc(s["updated_at"]),
            "road_id": first.get("road_id"), "road_name": first.get("road_name"),
            "direction": first.get("direction"), "lane": first.get("lane"), "device_model": first.get("device_model"),
            "start_utc": min(times) if times else None, "end_utc": max(times) if times else None,
            "local_date": next((p["local_time"][:10] for p in props if p.get("local_time")), None),
            **aggregate(props),
            "bbox": bbox_of([{"geometry": {"type": "Point", "coordinates": [p["centre_lon"], p["centre_lat"]]}}
                           for p in props if p["centre_lon"] is not None]),
        }
        if with_km:
            item["per_km"] = per_km(props)
        if track:
            item["track"] = track_geometry(props)
        out.append(item)
    return _json({"schema": SCHEMA_VERSION, "generated_at": now_utc(), "count": len(out), "sessions": out})


@router.get("/video/summary")
async def video_summary(request: Request):
    """Totals for the stats bar, over the segments matching the same filters as segments.geojson
    (limit/offset ignored: a summary covers every match)."""
    f = _filters(request, "video")
    return _json(summary_document(await _video_features(f.sessions, "point"), f))


# ============================================================
# Images
# ============================================================

def preview_cache_dir() -> Path:
    env = os.environ.get("GIS_PREVIEW_CACHE")
    return Path(env) if env else image_root().parent / "gis_cache"


def make_preview(path: Path, width: int, sha: Optional[str]) -> bytes:
    """JPEG preview of a stored frame, cached on disk by (SHA-256, width); display only."""
    from PIL import Image

    if not (isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha)):
        st = path.stat()
        sha = hashlib.sha256(f"{path}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()
    cache = preview_cache_dir() / f"{sha}_w{width}.jpg"
    try:
        return cache.read_bytes()
    except OSError:
        pass
    with _PREVIEW_SLOTS:
        with Image.open(path) as im:
            im = im.convert("RGB")
            if im.width > width:
                im.thumbnail((width, max(1, round(im.height * width / im.width))), Image.LANCZOS, reducing_gap=3.0)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=PREVIEW_QUALITY, optimize=True, progressive=True)
    data = buf.getvalue()
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_name(f".{cache.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, cache)
    except OSError:
        pass                         # an unwritable cache only costs speed
    return data


def resolve_segment_image(segment_id: str) -> Optional[tuple[Path, str, Optional[str]]]:
    """(resolved path, media type, sha256) of a segment's stored image, or None. Only
    video_segments.image_path, only image files inside LOCAL_IMAGE_ROOT."""
    if not _SEGMENT_ID.match(segment_id or ""):
        return None
    row = _query(lambda cur: cur.execute("SELECT image_path, image_sha256 FROM video_segments WHERE segment_id = %s",
                                         (segment_id,)).fetchone())
    if not row:
        return None
    p = local_image_path(row["image_path"])
    media = _IMAGE_TYPES.get(p.suffix.lower()) if p is not None else None
    if media is None:
        return None
    try:
        resolved = inside_root(p)
    except PermissionError:
        return None
    if not resolved.is_file():
        return None
    return resolved, media, row["image_sha256"]


@router.get("/image/{segment_id}")
async def segment_image(segment_id: str, w: Optional[int] = None):
    """A segment's stored frame. No w: the full-resolution PNG (5-9 MB, for the detail view's
    "open full image" only). w: a JPEG preview, width snapped up to PREVIEW_WIDTHS."""
    try:
        found = await asyncio.to_thread(resolve_segment_image, segment_id)
    except _Unavailable as e:
        raise HTTPException(status_code=503, detail=f"GIS data unavailable: {e}")
    if found is None:
        raise HTTPException(status_code=404, detail="no stored image for this segment")
    path, media, sha = found
    cache = {"Cache-Control": "public, max-age=86400"}
    if w is None:
        if sha:
            cache["ETag"] = f'"{sha}"'
        return FileResponse(path, media_type=media, headers=cache)
    if w < 1:
        raise HTTPException(status_code=400, detail="w must be a positive width in pixels")
    width = snap_width(int(w))
    try:
        data = await asyncio.to_thread(make_preview, path, width, sha)
    except (OSError, ValueError) as e:
        raise HTTPException(status_code=404, detail=f"image not readable: {type(e).__name__}")
    if sha:
        cache["ETag"] = f'"{sha}-w{width}"'
    return Response(content=data, media_type="image/jpeg", headers=cache)


# ============================================================
# Citizen uploads (RoadSide app)
# ============================================================

_CITIZEN_COLS = ("id, photo_id, image_url, latitude, longitude, address, status, stage1_label, stage1_confidence, "
                 "distress_types, severity, description, stage2_confidence, needs_expert_review, expert_reviewed, "
                 "expert_corrected_types, expert_corrected_severity, reviewed_at, created_at, processed_at")
CITIZEN_STATUSES = ("classified", "done", "expert_review")
_citizen_cache: dict = {"at": 0.0, "key": None, "rows": None}
_citizen_lock = asyncio.Lock()


def _cloudinary_variant(url: str, transform: str) -> str:
    """A Cloudinary delivery URL with an on-the-fly transformation (w_480 etc.); other URLs unchanged."""
    if "res.cloudinary.com/" in url and "/upload/" in url:
        return url.replace("/upload/", f"/upload/{transform}/", 1)
    return url


def _citizen_image_urls(url: Optional[str]) -> dict:
    if not url:
        return {"image_url": None, "preview_url": None, "thumb_url": None}
    if local_image_path(url) is not None:      # DB_BACKEND=postgres: a stored file, served by /local-images
        u = "/local-images?path=" + urllib.parse.quote(url, safe="")
        return {"image_url": u, "preview_url": u, "thumb_url": u}
    return {"image_url": url, "preview_url": _cloudinary_variant(url, "w_1280,c_limit,q_auto,f_jpg"),
            "thumb_url": _cloudinary_variant(url, "w_480,c_limit,q_auto,f_jpg")}


def citizen_feature(r: dict) -> dict:
    """One assessments row (PostgREST JSON or local SQL) as a GeoJSON Point Feature."""
    stage1 = r.get("stage1_label") if r.get("stage1_label") in ("Normal", "Distress") else None
    c = _classification(stage1, r.get("distress_types") if stage1 == "Distress" else [],
                        r.get("severity"), r.get("stage1_confidence"), r.get("stage2_confidence"),
                        r.get("needs_expert_review"), r.get("expert_reviewed"), r.get("expert_corrected_types"),
                        r.get("expert_corrected_severity"), r.get("description"))
    pt = lonlat(r.get("longitude"), r.get("latitude"))

    def t(v):
        return iso_utc(v) if isinstance(v, datetime) else iso_utc(parse_time(v))

    props = {"layer": "citizen", "id": str(r["id"]), "photo_id": r.get("photo_id"),
             "assessment_status": r.get("status"), "answered": stage1 is not None,
             "vlm_class": _answer_class(c) if stage1 else "unknown",
             "latitude": pt[1] if pt else None, "longitude": pt[0] if pt else None, "address": r.get("address"),
             "created_at": t(r.get("created_at")), "processed_at": t(r.get("processed_at")),
             "reviewed_at": t(r.get("reviewed_at")), **c, **_citizen_image_urls(r.get("image_url"))}
    return {"type": "Feature", "id": props["id"], "geometry": {"type": "Point", "coordinates": pt} if pt else None,
            "properties": props}


def _local_citizen_rows() -> list[dict]:
    return _query(lambda cur: cur.execute(
        f"SELECT {_CITIZEN_COLS} FROM assessments WHERE status = ANY(%s) AND road_segment_id IS NULL "
        f"AND latitude IS NOT NULL AND longitude IS NOT NULL ORDER BY created_at DESC, id LIMIT %s",
        (list(CITIZEN_STATUSES), CITIZEN_MAX_ROWS)).fetchall())


async def _supabase_citizen_rows(client) -> list[dict]:
    """Every eligible row via PostgREST, 1000 per page (PostgREST's usual max-rows)."""
    import httpx
    rows, page = [], 1000
    while len(rows) < CITIZEN_MAX_ROWS:
        params = {"select": _CITIZEN_COLS.replace(" ", ""), "status": f"in.({','.join(CITIZEN_STATUSES)})",
                  "road_segment_id": "is.null", "latitude": "not.is.null", "longitude": "not.is.null",
                  "order": "created_at.desc,id.asc", "limit": str(page), "offset": str(len(rows))}
        try:
            resp = await client.get("/rest/v1/assessments", params=params)
        except httpx.HTTPError as e:
            raise HTTPException(status_code=503, detail=f"citizen data unavailable: {type(e).__name__}")
        if resp.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"citizen data: Supabase answered HTTP {resp.status_code}")
        batch = resp.json()
        rows.extend(batch)
        if len(batch) < page:
            break
    return rows


async def citizen_rows(app) -> tuple[list[dict], str]:
    """(rows, backend) from the server's own backend, cached for CITIZEN_TTL_S."""
    client = getattr(app.state, "supabase_client", None)
    if client is None:
        raise HTTPException(status_code=503, detail="citizen uploads are not configured on this server "
                                                    "(no Supabase / local database client)")
    backend = "postgres" if is_local_db(client) else "supabase"
    async with _citizen_lock:
        c = _citizen_cache
        if c["rows"] is not None and c["key"] == (backend, id(client)) and time.monotonic() - c["at"] < CITIZEN_TTL_S:
            return c["rows"], backend
        if backend == "postgres":
            try:
                rows = await asyncio.to_thread(_local_citizen_rows)
            except _Unavailable as e:
                raise HTTPException(status_code=503, detail=f"citizen data unavailable: {e}")
        else:
            rows = await _supabase_citizen_rows(client)
        c.update(at=time.monotonic(), key=(backend, id(client)), rows=rows)
        return rows, backend


@router.get("/citizen/assessments.geojson")
async def citizen_geojson(request: Request):
    """RoadSide uploads as GeoJSON points: classified/done (include_review=true adds expert_review),
    never the video segments' own assessments. Same property names as the video layer."""
    f = _filters(request, "citizen")
    rows, backend = await citizen_rows(request.app)
    feats = [citizen_feature(r) for r in rows]
    page, total = apply(feats, f)
    return _json(feature_collection(page, total, f, "citizen", backend=backend), _GEOJSON)


def reset_citizen_cache() -> None:
    _citizen_cache.update(at=0.0, key=None, rows=None)


# ============================================================
# Meta
# ============================================================

def meta_document(feats: list[dict]) -> dict:
    tax = _taxonomy()
    props = [ft["properties"] for ft in feats]
    times = sorted(p["utc"] for p in props if p["utc"])
    labels = [{"name": lab, "is_distress": lab != "Normal", **tax[lab], "color": LABEL_COLORS[lab],
               "marker_priority": MARKER_PRIORITY.index(lab) + 1} for lab in LABELS]
    cats = []
    for lab in labels:
        if lab["category"] and lab["category"] not in [c["name"] for c in cats]:
            cats.append({"name": lab["category"], "labels": []})
        if lab["category"]:
            next(c for c in cats if c["name"] == lab["category"])["labels"].append(lab["name"])
    try:
        from scripts.utils import CONFIDENCE_THRESHOLD
    except Exception:
        CONFIDENCE_THRESHOLD = 0.80
    return {
        "schema": SCHEMA_VERSION, "generated_at": now_utc(),
        "labels": labels, "categories": cats, "other_label_color": OTHER_COLOR,
        "severities": [{"name": s, "radius_px": SEVERITY_RADIUS_PX[s]} for s in SEVERITIES],
        "statuses": [{"name": s, "meaning": STATUS_MEANING[s]} for s in STATUSES],
        "vlm_classes": [{"name": k, **v} for k, v in CLASS_STYLE.items()],
        "marker_rule": "colour = marker_label (first of marker_priority among final_labels); "
                       "radius = final_severity; final_* = expert correction when present, else the VLM",
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "preview_widths": list(PREVIEW_WIDTHS),
        "limits": {"default_limit": DEFAULT_LIMIT, "max_limit": MAX_LIMIT},
        "video": {
            "sessions": len({p["session_id"] for p in props}), "segments": len(props),
            "with_image": sum(1 for p in props if p["has_image"]),
            "answered": sum(1 for p in props if p["answered"]),
            "segments_by_status": {s: sum(1 for p in props if p["status"] == s) for s in STATUSES},
            "route_m": round(sum(num(p.get("length_m")) or 0.0 for p in props), 2),
            "bbox": bbox_of(feats), "time_range": {"from": times[0] if times else None,
                                                 "to": times[-1] if times else None},
        },
        "endpoints": {
            "segments": "/gis/video/segments.geojson", "segments_csv": "/gis/video/segments.csv",
            "sessions": "/gis/video/sessions", "summary": "/gis/video/summary",
            "image": "/gis/image/{segment_id}", "citizen": "/gis/citizen/assessments.geojson",
        },
    }


@router.get("/meta")
async def gis_meta():
    return _json(meta_document(await _video_features(None, "footprint")))


# ============================================================
# Install
# ============================================================

def webgis_dir() -> Path:
    return Path(os.environ.get("GIS_APP_DIR") or (PROJECT_ROOT / "webgis"))


def install(app: FastAPI) -> None:
    """Include the /gis router; serve the WebGIS folder at /gis/app/ when it exists."""
    app.include_router(router)
    folder = webgis_dir()
    if (folder / "index.html").is_file():
        from fastapi.staticfiles import StaticFiles
        app.mount("/gis/app", StaticFiles(directory=str(folder), html=True), name="gis_app")
        print(f"[API] WebGIS served at /gis/app from {folder}")
