"""
FastAPI server for the Pavement Distress Classification API.

Endpoints:
  POST /classify        - Upload an image for two-stage classification
  POST /classify/stream - Upload an image, receive SSE progress events per stage
  POST /classify/base64 - Classify a base64-encoded image
  GET  /health          - Server and model status
  POST /retrain/start   - Trigger incremental retraining
  GET  /retrain/status  - Check retraining progress

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils import CONFIDENCE_THRESHOLD

# ============================================================
# Rate Limiter
# ============================================================
limiter = Limiter(key_func=get_remote_address)


# ============================================================
# App Lifecycle
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    print("[API] Loading model on startup...")
    get_classifier()
    print("[API] Model ready. Server accepting requests.")
    yield
    print("[API] Shutting down.")


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
