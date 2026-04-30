"""
FastAPI server for the Pavement Distress Classification API.

Endpoints:
  POST /classify        - Upload an image for two-stage classification
  POST /classify/stream - Upload an image, receive SSE progress events per stage
  POST /classify/base64 - Classify a base64-encoded image
  GET  /health          - Server and model status
  POST /retrain/start   - Trigger incremental retraining
  GET  /retrain/status  - Check retraining progress

Operator pipeline (autonomous batch processor for the RoadSide app queue):
  GET  /operator                 - HTML dashboard (Start/Stop + live metrics)
  POST /operator/start           - Begin polling Supabase for pending images
  POST /operator/stop            - Graceful drain + stop
  GET  /operator/status          - JSON snapshot of worker state
  GET  /operator/metrics         - Aggregate Supabase metrics + worker counters
  GET  /operator/metrics/stream  - SSE stream of metrics (auto-refresh)
  GET  /operator/recent          - Recent processed images for activity feed

Security:
  - API Key auth via X-API-Key header (disabled if API_KEYS env var not set)
  - Rate limiting: 10 req/min on classify, 60 req/min on health
  - Upload size limit: 10MB
  - Image validation: max 4096x4096, common formats only
  - CORS restricted to CORS_ORIGINS env var

Usage:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1

    # With security:
    API_KEYS="mykey123" CORS_ORIGINS="https://myapp.com" uvicorn app.main:app --host 0.0.0.0 --port 8000

    # With operator pipeline (set Supabase env vars before starting):
    SUPABASE_URL=https://xxx.supabase.co SUPABASE_SERVICE_KEY=sb_secret_xxx \\
      uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
"""

import asyncio
import io
import json
import os
import sys
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sse_starlette.sse import EventSourceResponse

from app.model import get_classifier
from app.schemas import (
    ClassificationResponse,
    ErrorResponse,
    HealthResponse,
    RetrainStatusResponse,
)
from app.security import (
    LimitUploadSizeMiddleware,
    validate_image,
    verify_api_key,
)
from app.supabase_client import (
    build_image_client,
    build_supabase_client,
    get_assessment_by_id,
    get_class_distribution,
    get_pipeline_metrics,
    get_recent_processed,
)
from app.worker import (
    PipelineWorker,
    clear_worker,
    get_worker,
    set_worker,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils import CONFIDENCE_THRESHOLD


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OPERATOR_UI_HTML = PROJECT_ROOT / "operator_ui" / "index.html"
TEST_FIXTURES_DIR = PROJECT_ROOT / "test_fixtures"


def _has_supabase_env() -> bool:
    """Operator pipeline only activates if both env vars are set."""
    return bool(os.environ.get("SUPABASE_URL")) and bool(os.environ.get("SUPABASE_SERVICE_KEY"))


def _test_fixtures_enabled() -> bool:
    """Static test images mounted only when explicitly opted-in (smoke testing)."""
    return os.environ.get("ENABLE_TEST_FIXTURES", "").lower() in ("1", "true", "yes")

# ============================================================
# Rate Limiter
# ============================================================
limiter = Limiter(key_func=get_remote_address)


# ============================================================
# App Lifecycle
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: load model, build Supabase clients (if env vars present),
             create PipelineWorker singleton (does NOT auto-start).
    Shutdown: gracefully stop worker, close HTTP clients.

    The worker is built but not started — operator must explicitly POST
    /operator/start. This prevents accidental autonomous batch processing
    in dev environments.
    """
    print("[API] Loading model on startup...")
    classifier = get_classifier()
    print("[API] Model ready.")

    if _has_supabase_env():
        print("[API] Supabase env vars detected — initializing operator pipeline...")
        try:
            supabase_client = build_supabase_client()
            image_client = build_image_client()
            worker = PipelineWorker(
                supabase_client=supabase_client,
                image_client=image_client,
                classifier=classifier,
            )
            set_worker(worker)
            # Stash clients on app.state so we can close them on shutdown
            app.state.supabase_client = supabase_client
            app.state.image_client = image_client
            print(f"[API] Operator pipeline ready. Worker ID: {worker.worker_id}")
            print("[API] POST /operator/start to begin processing the queue.")
        except Exception as e:
            print(f"[API] WARNING: Failed to initialize operator pipeline: {e}")
            traceback.print_exc()
            app.state.supabase_client = None
            app.state.image_client = None
    else:
        print("[API] Supabase env vars not set — operator pipeline disabled.")
        print("[API] Set SUPABASE_URL and SUPABASE_SERVICE_KEY to enable.")
        app.state.supabase_client = None
        app.state.image_client = None

    print("[API] Server accepting requests.")
    yield

    print("[API] Shutting down...")
    worker = get_worker()
    if worker and worker.is_running:
        print("[API] Stopping pipeline worker (graceful drain)...")
        try:
            await worker.stop(drain_timeout_seconds=60.0)
        except Exception as e:
            print(f"[API] Error stopping worker: {e}")
    clear_worker()

    if getattr(app.state, "supabase_client", None):
        await app.state.supabase_client.aclose()
    if getattr(app.state, "image_client", None):
        await app.state.image_client.aclose()
    print("[API] Shutdown complete.")


# ============================================================
# FastAPI App
# ============================================================
app = FastAPI(
    title="Pavement Distress Classification API",
    description=(
        "Two-stage AI pipeline for pavement distress detection and classification. "
        "Stage 1 detects if distress is present. Stage 2 classifies the distress type. "
        "Powered by Qwen2.5-VL fine-tuned on GAPs V2 and RDD datasets."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# Attach rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Upload size limit middleware
app.add_middleware(LimitUploadSizeMiddleware)

# CORS — restricted by CORS_ORIGINS env var, defaults to "*" for dev
ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Test fixtures static mount — used by smoke tests so the worker can download
# real RDD images from localhost without external dependencies.
# Only mounted when ENABLE_TEST_FIXTURES=1 — never expose in production.
if _test_fixtures_enabled() and TEST_FIXTURES_DIR.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount(
        "/test_fixtures",
        StaticFiles(directory=str(TEST_FIXTURES_DIR)),
        name="test_fixtures",
    )
    print(f"[API] TEST FIXTURES enabled — serving {TEST_FIXTURES_DIR} at /test_fixtures")


# ============================================================
# Endpoints
# ============================================================

@app.get("/health", response_model=HealthResponse)
@limiter.limit("60/minute")
async def health(request: Request):
    """Check server and model status."""
    try:
        clf = get_classifier()
        return HealthResponse(
            status="ok",
            model_loaded=clf.is_loaded,
            model_name=clf.model_path,
            device=clf.device,
            adapter_loaded=clf.has_adapter,
        )
    except Exception:
        return HealthResponse(
            status="error",
            model_loaded=False,
            model_name="unknown",
            device="unknown",
            adapter_loaded=False,
        )


@app.post(
    "/classify",
    response_model=ClassificationResponse,
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit("10/minute")
async def classify(
    request: Request,
    file: UploadFile = File(..., description="Pavement image to classify"),
):
    """
    Classify a pavement image for distress.

    Returns detection result, distress types, severity, confidence scores,
    and whether the result needs expert review (confidence < 80%).
    """
    # Validate content type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"File must be an image. Got: {file.content_type}",
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Validate image dimensions and format
        validate_image(image)

        # Run classification in a thread to keep the async event loop responsive.
        # GPU inference releases the GIL during CUDA kernel execution,
        # so other requests can be accepted while inference runs.
        clf = get_classifier()
        result = await asyncio.to_thread(clf.predict, image)
        return ClassificationResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@app.post(
    "/classify/base64",
    response_model=ClassificationResponse,
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit("10/minute")
async def classify_base64(request: Request, payload: dict):
    """
    Classify a base64-encoded image.
    Payload: {"image": "<base64-string>"}
    """
    import base64

    if "image" not in payload:
        raise HTTPException(status_code=400, detail="Missing 'image' field in payload")

    try:
        image_data = base64.b64decode(payload["image"])
        image = Image.open(io.BytesIO(image_data))

        validate_image(image)

        clf = get_classifier()
        result = await asyncio.to_thread(clf.predict, image)
        return ClassificationResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


# ============================================================
# SSE Streaming Classification
# ============================================================

@app.post(
    "/classify/stream",
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit("10/minute")
async def classify_stream(
    request: Request,
    file: UploadFile = File(..., description="Pavement image to classify"),
):
    """
    Classify an image with live SSE progress events per stage.

    Streams events:
      event: progress  — stage start/complete updates
      event: result    — final ClassificationResponse JSON
      event: error     — error details if something fails
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"File must be an image. Got: {file.content_type}",
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        validate_image(image)
        image.load()  # Force full decode so BytesIO buffer can be released
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

    clf = get_classifier()

    async def event_generator():
        import time as _time

        total_start = _time.time()

        try:
            # --- Stage 1: Binary Detection ---
            yield {
                "event": "progress",
                "data": json.dumps({
                    "stage": "stage1",
                    "status": "running",
                    "message": "Detecting if pavement shows distress...",
                }),
            }

            s1 = await asyncio.to_thread(clf.predict_stage1, image)

            yield {
                "event": "progress",
                "data": json.dumps({
                    "stage": "stage1",
                    "status": "complete",
                    "result": s1,
                }),
            }

            # Build full result starting from Stage 1
            result = {
                "is_distressed": s1["is_distressed"],
                "stage1_label": s1["stage1_label"],
                "stage1_confidence": s1["stage1_confidence"],
                "distress_types": [],
                "severity": "None",
                "description": "",
                "stage2_confidence": 0.0,
                "needs_expert_review": s1["needs_expert_review"],
                "stage1_time_ms": s1["stage1_time_ms"],
                "stage2_time_ms": 0.0,
                "stage1_raw": s1["stage1_raw"],
                "stage2_raw": "",
            }

            # --- Stage 2: Type Classification (only if distressed) ---
            if s1["is_distressed"]:
                # Skip stage 2 if client disconnected during stage 1
                if await request.is_disconnected():
                    return

                yield {
                    "event": "progress",
                    "data": json.dumps({
                        "stage": "stage2",
                        "status": "running",
                        "message": "Classifying distress type and severity...",
                    }),
                }

                s2 = await asyncio.to_thread(clf.predict_stage2, image)

                yield {
                    "event": "progress",
                    "data": json.dumps({
                        "stage": "stage2",
                        "status": "complete",
                        "result": s2,
                    }),
                }

                result["distress_types"] = s2["distress_types"]
                result["severity"] = s2["severity"]
                result["description"] = s2["description"]
                result["stage2_confidence"] = s2["stage2_confidence"]
                result["stage2_time_ms"] = s2["stage2_time_ms"]
                result["stage2_raw"] = s2["stage2_raw"]

                result["needs_expert_review"] = (
                    s1["stage1_confidence"] < CONFIDENCE_THRESHOLD
                    or s2["stage2_confidence"] < CONFIDENCE_THRESHOLD
                )

            total_time = (_time.time() - total_start) * 1000
            result["processing_time_ms"] = round(total_time, 1)

            yield {
                "event": "result",
                "data": json.dumps(result),
            }

        except Exception as e:
            traceback.print_exc()
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)}),
            }

    return EventSourceResponse(event_generator())


# ============================================================
# Retrain Endpoints
# ============================================================
# Simple in-memory state for tracking retrain jobs.
# In production, this should be backed by the retrain_jobs DB table.

_retrain_state = {
    "is_running": False,
    "progress_pct": 0.0,
    "current_step": 0,
    "total_steps": 0,
    "eta_seconds": 0.0,
    "pending_corrections": 0,
    "last_retrain_at": "",
}


@app.post(
    "/retrain/start",
    response_model=RetrainStatusResponse,
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit("2/minute")
async def retrain_start(request: Request):
    """
    Trigger incremental retraining on expert corrections.

    This is a placeholder that sets the retrain state to 'running'.
    Actual retraining is done via scripts/06_incremental_retrain.py
    on the GPU machine. In production, this would queue a job
    (e.g., via Celery or a subprocess).
    """
    if _retrain_state["is_running"]:
        raise HTTPException(status_code=409, detail="Retraining already in progress")

    _retrain_state["is_running"] = True
    _retrain_state["progress_pct"] = 0.0
    _retrain_state["current_step"] = 0
    _retrain_state["total_steps"] = 100  # Placeholder
    _retrain_state["eta_seconds"] = 0.0

    return RetrainStatusResponse(**_retrain_state)


@app.get(
    "/retrain/status",
    response_model=RetrainStatusResponse,
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit("30/minute")
async def retrain_status(request: Request):
    """Check the status of the current or last retraining job."""
    return RetrainStatusResponse(**_retrain_state)


# ============================================================
# Operator Pipeline Endpoints
# ============================================================
# These endpoints control the autonomous batch processor that pulls
# pending images from Supabase and writes back classification results.
# All require Supabase env vars to be set at server startup.


def _require_worker() -> PipelineWorker:
    """Return the worker singleton, or 503 if pipeline isn't initialized."""
    worker = get_worker()
    if worker is None:
        raise HTTPException(
            status_code=503,
            detail="Operator pipeline not initialized. "
                   "Set SUPABASE_URL and SUPABASE_SERVICE_KEY env vars and restart the server.",
        )
    return worker


@app.get("/operator", include_in_schema=False)
async def operator_dashboard():
    """Serve the operator dashboard HTML."""
    if not OPERATOR_UI_HTML.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Operator UI not found at {OPERATOR_UI_HTML}",
        )
    return FileResponse(str(OPERATOR_UI_HTML), media_type="text/html")


@app.post("/operator/start", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def operator_start(request: Request):
    """Begin the worker loop. Idempotent — returns current state if already running."""
    worker = _require_worker()
    result = await worker.start()
    return JSONResponse(result)


@app.post("/operator/stop", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def operator_stop(request: Request):
    """
    Signal the worker to stop. Current image finishes, then loop exits.
    Drain timeout is 60s; after that, the task is force-cancelled.
    """
    worker = _require_worker()
    result = await worker.stop(drain_timeout_seconds=60.0)
    return JSONResponse(result)


@app.get("/operator/status", dependencies=[Depends(verify_api_key)])
@limiter.limit("60/minute")
async def operator_status(request: Request):
    """In-memory worker state snapshot (counters, current image, last error)."""
    worker = _require_worker()
    return JSONResponse(worker.metrics_snapshot())


@app.get("/operator/metrics", dependencies=[Depends(verify_api_key)])
@limiter.limit("60/minute")
async def operator_metrics(request: Request):
    """
    Combined metrics for the dashboard:
      - worker      : in-memory counters
      - queue       : Supabase queue depth (pending/processing/done/failed)
      - distribution: per-class counts (last 24h)
    """
    worker = _require_worker()
    supabase = request.app.state.supabase_client

    out = {"worker": worker.metrics_snapshot()}

    # Queue + throughput from the pipeline_metrics view (best-effort)
    try:
        out["queue"] = await get_pipeline_metrics(supabase)
    except Exception as e:
        out["queue"] = {"error": str(e)}

    # Per-class distribution
    try:
        out["distribution"] = await get_class_distribution(supabase)
    except Exception as e:
        out["distribution"] = {"error": str(e)}

    return JSONResponse(out)


@app.get("/operator/recent", dependencies=[Depends(verify_api_key)])
@limiter.limit("60/minute")
async def operator_recent(request: Request, limit: int = 20):
    """Recent processed images for the dashboard activity feed."""
    _require_worker()
    supabase = request.app.state.supabase_client
    limit = max(1, min(100, limit))
    try:
        rows = await get_recent_processed(supabase, limit=limit)
        return JSONResponse({"items": rows})
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Supabase fetch failed: {e}")


@app.get("/operator/inflight", dependencies=[Depends(verify_api_key)])
@limiter.limit("120/minute")
async def operator_inflight(request: Request, id: str):
    """
    Fetch metadata for the currently-processing image (live preview card).

    The dashboard calls this when worker.current_image_id changes — gives it
    the image_url for the thumbnail, GPS coords, and any partial results.

    Returns 404 if the row no longer exists (e.g., already moved to done).
    """
    _require_worker()
    supabase = request.app.state.supabase_client
    try:
        row = await get_assessment_by_id(supabase, id)
        if not row:
            raise HTTPException(status_code=404, detail="assessment not found")
        return JSONResponse(row)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Supabase fetch failed: {e}")


@app.get("/operator/metrics/stream", dependencies=[Depends(verify_api_key)])
async def operator_metrics_stream(request: Request, interval_seconds: float = 2.0):
    """
    SSE stream that pushes a metrics snapshot every `interval_seconds`.
    Dashboard subscribes once and updates the UI as events arrive.
    """
    worker = _require_worker()
    supabase = request.app.state.supabase_client
    interval = max(1.0, min(30.0, float(interval_seconds)))

    async def event_generator():
        while True:
            if await request.is_disconnected():
                return

            try:
                payload = {"worker": worker.metrics_snapshot()}
                try:
                    payload["queue"] = await get_pipeline_metrics(supabase)
                except Exception as e:
                    payload["queue"] = {"error": str(e)}
                try:
                    payload["distribution"] = await get_class_distribution(supabase)
                except Exception as e:
                    payload["distribution"] = {"error": str(e)}

                yield {"event": "metrics", "data": json.dumps(payload)}
            except Exception as e:
                yield {"event": "error", "data": json.dumps({"error": str(e)})}

            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                return

    return EventSourceResponse(event_generator())
