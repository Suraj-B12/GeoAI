"""
API for the expert UI's Video and Metrics tabs and its review log (/expert-api/*).

    GET  /expert-api/meta                         type list, severities, queues, the measured
                                                  reliability of the overlay (paper §6.10)
    GET  /expert-api/video/sessions               drives with answered segments
    GET  /expert-api/video/list                   ?queue=needs_review|auto|random|reviewed
                                                  &session=&label=&limit=&offset=
    POST /expert-api/video/review/{segment_id}    save + log an expert review (one transaction)
    POST /expert-api/citizen/log/{assessment_id}  log a citizen review the page saved to Supabase
    GET  /expert-api/explain/video/{segment_id}?type=Potholes
    GET  /expert-api/explain/citizen/{assessment_id}?type=Potholes
                                                  "where the model looked": Grad-CAM of the
                                                  per-type probe (app/model.explain_type)
    GET  /expert-api/metrics                      model vs expert (app/expert_review.metrics)

Open like the rest of the expert page (which writes citizen reviews to Supabase with the anon
key): no X-API-Key, because the page is static and holds no secret. Writes are therefore
narrow: a video save can only set the expert_* columns of an assessment that belongs to a
video segment, with validated values; a citizen log row is accepted only when Supabase already
holds that same review (the page saved it first), so the log cannot be filled with reviews that
were never made. Explanations cost GPU time, so they are cached on disk by image SHA-256 + type
+ variant, limited per client, and at most two wait for the model at once.

The local PostgreSQL is reached through app.video_store.connect() in a worker thread; if it is
down these endpoints answer 503 and nothing else on the server is affected.
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
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from PIL import Image

from app import expert_review as ER

PROJECT_ROOT = Path(__file__).resolve().parent.parent
router = APIRouter(prefix="/expert-api", tags=["expert"])

CONNECT_TIMEOUT_S = 3
EXPLAIN_PER_MINUTE = 30               # per client address
EXPLAIN_MAX_WAITING = 2               # explanations waiting for the model at once
MAX_IMAGE_BYTES = 25 * 1024 * 1024
_ID = re.compile(r"^[A-Za-z0-9._-]{1,200}$")
_UUID = re.compile(r"^[0-9a-fA-F-]{36}$")
_explain_waiting = 0
_explain_guard = threading.Lock()
_hits: dict[str, deque] = defaultdict(deque)
_GRADCAM_SHA = hashlib.sha256((PROJECT_ROOT / "scripts" / "stage2_gradcam.py").read_bytes()).hexdigest()[:12] \
    if (PROJECT_ROOT / "scripts" / "stage2_gradcam.py").exists() else "missing"


class _Unavailable(Exception):
    pass


def _db(fn: Callable) -> Any:
    """fn(conn) on a fresh local-PostgreSQL connection; _Unavailable when it can't connect."""
    import psycopg

    from app import video_store
    try:
        conn = video_store.connect(connect_timeout=CONNECT_TIMEOUT_S)
    except RuntimeError as e:
        raise _Unavailable(str(e)) from None
    except psycopg.OperationalError:
        raise _Unavailable("local PostgreSQL is unreachable") from None
    try:
        return fn(conn)
    except psycopg.errors.UndefinedTable:
        raise _Unavailable("tables missing: run scripts/setup_local_db.py") from None
    except psycopg.OperationalError:
        raise _Unavailable("local PostgreSQL connection failed") from None
    finally:
        conn.close()


async def _run(fn: Callable) -> Any:
    try:
        return await asyncio.to_thread(_db, fn)
    except _Unavailable as e:
        raise HTTPException(status_code=503, detail=f"review database unavailable: {e}")
    except ER.NotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ER.ReviewError as e:
        raise HTTPException(status_code=400, detail=str(e))


async def _json_body(request: Request) -> dict:
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(status_code=400, detail="body must be JSON")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")
    return body


# ============================================================
# Meta
# ============================================================

def _overlay_reliability() -> Optional[dict]:
    """Per-class test hit rates of the overlay vs the location prior (paper §6.10), if present."""
    p = PROJECT_ROOT / "eval_results" / "gradcam_localization_report.json"
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    to_irc = {"Linear crack": ["Longitudinal Cracking", "Transverse Cracking"],
              "Alligator crack": ["Alligator Cracking"], "Pothole": ["Potholes"], "Raveling": ["Ravelling"],
              "Weathering": ["Hungry Surface"], "Patch and utility cut": ["Patching"],
              "Block crack": ["Transverse Cracking"]}
    out = {}
    for cls, row in (d.get("per_class_test") or {}).items():
        for t in to_irc.get(cls, []):
            if t in out:        # Transverse: keep the larger sample (linear cracks)
                continue
            out[t] = {"attain_class": cls, "hit": row["gradcam"]["hit"], "prior": row["prior"]["hit"],
                      "n": row["n_test"]}
    return {"variant": d.get("chosen_variant"), "macro_hit": d["test"]["gradcam"]["point"]
            if isinstance(d.get("test", {}).get("gradcam"), dict) else None, "per_type": out,
            "source": "eval_results/gradcam_localization_report.json (paper §6.10)"}


@router.get("/meta")
async def meta():
    return {"types": list(ER.EXPERT_TYPES), "normal": ER.NORMAL, "severities": list(ER.SEVERITY_RANK),
            "video_queues": list(ER.VIDEO_QUEUES), "heatmap_ratings": list(ER.HEATMAP_RATINGS),
            "overlay_reliability": _overlay_reliability()}


# ============================================================
# Video
# ============================================================

@router.get("/video/sessions")
async def video_sessions():
    return {"items": await _run(ER.video_sessions)}


@router.get("/video/list")
async def video_list(queue: str = "needs_review", session: Optional[str] = None, label: Optional[str] = None,
                     limit: int = 24, offset: int = 0):
    if session is not None and not _ID.match(session):
        raise HTTPException(status_code=400, detail="bad session id")
    if label is not None and len(label) > 60:
        raise HTTPException(status_code=400, detail="bad label")
    return await _run(lambda c: ER.list_video(c, queue, session=session or None, label=label or None,
                                              limit=limit, offset=offset))


@router.post("/video/review/{segment_id}")
async def video_review(segment_id: str, request: Request):
    if not _ID.match(segment_id):
        raise HTTPException(status_code=400, detail="bad segment id")
    body = await _json_body(request)
    return await _run(lambda c: ER.save_video_review(c, segment_id, body))


# ============================================================
# Citizen
# ============================================================

_CITIZEN_COLS = ("id,image_url,status,stage1_label,stage1_confidence,distress_types,severity,stage2_confidence,"
                 "needs_expert_review,model_version,processed_at,expert_reviewed,expert_corrected_types,"
                 "expert_corrected_severity,reviewer_name")


async def _citizen_row(request: Request, assessment_id: str) -> dict:
    if not _UUID.match(assessment_id):
        raise HTTPException(status_code=400, detail="bad assessment id")
    client = getattr(request.app.state, "supabase_client", None)
    if client is None:
        raise HTTPException(status_code=503, detail="Supabase client not configured on this server")
    from app.supabase_client import is_local_db
    if is_local_db(client):
        row = await client.get_assessment_by_id(assessment_id)
    else:
        resp = await client.get(f"/rest/v1/assessments?id=eq.{assessment_id}&select={_CITIZEN_COLS}&limit=1")
        if not resp.is_success:
            raise HTTPException(status_code=502, detail=f"Supabase answered {resp.status_code}")
        rows = resp.json() or []
        row = rows[0] if rows else None
    if not row:
        raise HTTPException(status_code=404, detail="no such assessment")
    return row


@router.post("/citizen/log/{assessment_id}")
async def citizen_log(assessment_id: str, request: Request):
    body = await _json_body(request)
    row = await _citizen_row(request, assessment_id)
    saved = [ER.canonical(t) for t in (row.get("expert_corrected_types") or [])]
    sent = [ER.canonical(t) for t in (body.get("types") or []) if isinstance(t, str)]
    if not row.get("expert_reviewed") or sorted(saved) != sorted(sent):
        raise HTTPException(status_code=409, detail="Supabase does not hold this review; save it there first")
    return await _run(lambda c: ER.log_citizen_review(c, assessment_id, body, row))


# ============================================================
# "Where the model looked" (Grad-CAM, reference only)
# ============================================================

def _cache_dir() -> Path:
    from app.local_images import image_root
    d = Path(os.environ.get("EXPERT_GRADCAM_CACHE") or (image_root().parent / "gradcam_cache"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _allow(addr: str) -> bool:
    now = time.time()
    q = _hits[addr]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= EXPLAIN_PER_MINUTE:
        return False
    q.append(now)
    return True


def _explain_cached(sha: str, type_key: str, load_image: Callable[[], Image.Image]) -> dict:
    """Disk-cached explain_type (the cache key covers the image, type, variant and code)."""
    from app.model import PavementClassifier, get_classifier
    variant = f"{PavementClassifier.GRADCAM_METHOD}@{PavementClassifier.GRADCAM_LAYER}"
    key = hashlib.sha256(f"{sha}|{type_key}|{variant}|{_GRADCAM_SHA}".encode()).hexdigest()[:32]
    path = _cache_dir() / f"{key}.json"
    try:
        out = json.loads(path.read_text(encoding="utf-8"))
        out["cached"] = True
        return out
    except (OSError, ValueError):
        pass
    img = load_image()
    size = list(img.size)
    try:
        out = get_classifier().explain_type(img, type_key)
    finally:
        img.close()
    out["image_size"] = size
    out["cached"] = False
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out), encoding="utf-8")
    os.replace(tmp, path)
    return out


async def _explain(request: Request, type_key: str, sha: str, load_image: Callable[[], Image.Image]) -> dict:
    global _explain_waiting
    from scripts.stage2_probe import all_probe_types
    if type_key not in {t.key for t in all_probe_types()}:
        raise HTTPException(status_code=400, detail="type must be an IRC:82 type, Patching or Block Cracking")
    if not _allow(request.client.host if request.client else "?"):
        raise HTTPException(status_code=429, detail="too many explanations this minute")
    with _explain_guard:
        if _explain_waiting >= EXPLAIN_MAX_WAITING:
            raise HTTPException(status_code=429, detail="the model is busy with other explanations; try again")
        _explain_waiting += 1
    try:
        out = await asyncio.to_thread(_explain_cached, sha, type_key, load_image)
    except KeyError:
        raise HTTPException(status_code=400, detail="unknown type")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"explanation unavailable: {e}")
    finally:
        with _explain_guard:
            _explain_waiting -= 1
    out["reference_only"] = ("Where the model's evidence for this type is strongest when asked about it "
                             "directly. Not a measured damage box (paper §6.10).")
    return out


@router.get("/explain/video/{segment_id}")
async def explain_video(segment_id: str, request: Request, type_key: str = Query(..., alias="type")):
    from app.gis_routes import _Unavailable as GisUnavailable, resolve_segment_image
    if not _ID.match(segment_id):
        raise HTTPException(status_code=400, detail="bad segment id")
    try:
        found = await asyncio.to_thread(resolve_segment_image, segment_id)
    except GisUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))
    if found is None:
        raise HTTPException(status_code=404, detail="no stored image for this segment")
    path, _media, sha = found
    if not sha:
        sha = await asyncio.to_thread(lambda: hashlib.sha256(path.read_bytes()).hexdigest())
    return await _explain(request, type_key, sha, lambda: Image.open(path).convert("RGB"))


@router.get("/explain/citizen/{assessment_id}")
async def explain_citizen(assessment_id: str, request: Request, type_key: str = Query(..., alias="type")):
    row = await _citizen_row(request, assessment_id)
    url = row.get("image_url") or ""
    from app.local_images import local_image_path, read_local_image
    if local_image_path(url) is not None:
        data = await asyncio.to_thread(read_local_image, url)
    else:
        u = urllib.parse.urlparse(url)
        if u.scheme != "https" or u.hostname != "res.cloudinary.com":
            raise HTTPException(status_code=400, detail="image is not a Cloudinary or local image")
        client = getattr(request.app.state, "image_client", None)
        import httpx
        try:
            if client is not None:
                resp = await client.get(url)
            else:
                async with httpx.AsyncClient(timeout=30) as c:
                    resp = await c.get(url)
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"image download failed: {type(e).__name__}")
        if not resp.is_success or len(resp.content) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=502, detail="image download failed")
        data = resp.content
    sha = hashlib.sha256(data).hexdigest()
    return await _explain(request, type_key, sha, lambda: Image.open(io.BytesIO(data)).convert("RGB"))


# ============================================================
# Metrics
# ============================================================

@router.get("/metrics")
async def metrics():
    return await _run(ER.metrics)


def install(app: FastAPI) -> None:
    app.include_router(router)
