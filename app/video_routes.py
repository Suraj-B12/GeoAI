"""
HTTP endpoints of the video pipeline (app/video_pipeline.py) for the operator dashboard.

    GET  /operator/video/status               toggle, current session + progress, worker hold, paths, checks
    POST /operator/video/enable               turn the "video_pipeline" toggle on (persisted)
    POST /operator/video/disable              off: stops after the current item; resumable
    GET  /operator/video/sessions             sessions with state, error and metrics (newest first)
    GET  /operator/video/sessions/{id}/log    the session's JSON-lines log events
    POST /operator/video/sessions/{id}/retry  put a failed session back in the queue
    GET  /local-images?path=<image_url>       a stored image, for <img> tags (see below)

All but /local-images require the same X-API-Key as the other /operator endpoints. An <img> tag
can't send that header, and the images it replaces were public Cloudinary URLs, so /local-images
is open like they were, but serves only .png/.jpg/.jpeg files inside LOCAL_IMAGE_ROOT
(app/local_images.inside_root): with DB_BACKEND=postgres, image_url holds that absolute path.

install(app) includes the router and wraps the app's lifespan so the service starts after the
model and the PipelineWorker are ready, and stops (killing a running extraction or runner, the
session keeping its stage) before the worker and the HTTP clients are shut down. It leaves the
lifespan in app/main.py untouched.
"""

from __future__ import annotations

import asyncio
import contextlib
import traceback

from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse

from app.local_images import inside_root, local_image_path
from app.security import verify_api_key
from app.video_pipeline import VideoPipelineService, get_service, set_service

router = APIRouter(prefix="/operator/video", tags=["video_pipeline"], dependencies=[Depends(verify_api_key)])
image_router = APIRouter(tags=["video_pipeline"])
_IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


@image_router.get("/local-images")
async def local_image(path: str):
    """A stored image by its image_url. 404 for anything that isn't an image file inside the root."""
    p = local_image_path(path)
    media = _IMAGE_TYPES.get(p.suffix.lower()) if p is not None else None
    try:
        resolved = inside_root(p) if media else None
    except PermissionError:
        resolved = None
    if resolved is None or not resolved.is_file():
        raise HTTPException(status_code=404, detail="no such stored image")
    return FileResponse(resolved, media_type=media)


def _require_service() -> VideoPipelineService:
    svc = get_service()
    if svc is None:
        raise HTTPException(status_code=503, detail="video pipeline service is not running (see the server log)")
    return svc


@router.get("/status")
async def video_status():
    return _require_service().status()


@router.post("/enable")
async def video_enable():
    return _require_service().enable()


@router.post("/disable")
async def video_disable():
    return _require_service().disable()


@router.get("/sessions")
async def video_sessions(limit: int = 100):
    svc = _require_service()
    return {"items": svc.sessions(limit=max(1, min(500, limit)))}


@router.get("/sessions/{session_id}/log")
async def video_session_log(session_id: str, limit: int = 500):
    svc = _require_service()
    out = await asyncio.to_thread(svc.session_log, session_id, max(1, min(5000, limit)))
    if not out["known"] and not out["events"]:
        raise HTTPException(status_code=404, detail=f"no session {session_id!r}")
    return out


@router.post("/sessions/{session_id}/retry")
async def video_session_retry(session_id: str):
    svc = _require_service()
    try:
        return svc.retry(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no session {session_id!r}")
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


async def start_service() -> None:
    """Build and start the singleton. A failure here must never take the API down."""
    try:
        svc = VideoPipelineService()
        await svc.start()
        set_service(svc)
    except Exception as e:
        print(f"[API] WARNING: video pipeline service not started: {e}")
        traceback.print_exc()


async def stop_service() -> None:
    svc = get_service()
    if svc is None:
        return
    try:
        await svc.stop()
    except Exception as e:
        print(f"[API] Error stopping the video pipeline service: {e}")
    finally:
        set_service(None)


def install(app: FastAPI) -> None:
    """Include the router and run the service inside the app's existing lifespan."""
    app.include_router(router)
    app.include_router(image_router)
    inner = app.router.lifespan_context

    @contextlib.asynccontextmanager
    async def lifespan_with_video_pipeline(a):
        async with inner(a) as state:
            await start_service()
            try:
                yield state
            finally:
                await stop_service()

    app.router.lifespan_context = lifespan_with_video_pipeline
