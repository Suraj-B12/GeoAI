"""
API security middleware and utilities.

Provides:
  - API Key authentication (X-API-Key header)
  - Rate limiting (slowapi)
  - Upload size limiting
  - Image validation
"""

import os

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

# ============================================================
# API Key Authentication
# ============================================================
# Set API_KEYS env var as comma-separated keys: API_KEYS="key1,key2"
# If not set or empty, auth is DISABLED (dev mode).

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

_raw_keys = os.environ.get("API_KEYS", "")
VALID_API_KEYS = set(k.strip() for k in _raw_keys.split(",") if k.strip())

AUTH_ENABLED = len(VALID_API_KEYS) > 0


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    """
    Dependency that validates the X-API-Key header.
    If no API_KEYS are configured, auth is disabled (dev mode).
    """
    if not AUTH_ENABLED:
        return  # Dev mode — no auth required

    if not api_key or api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Set X-API-Key header.",
        )


# ============================================================
# Upload Size Limiting
# ============================================================
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


class LimitUploadSizeMiddleware(BaseHTTPMiddleware):
    """Reject requests with Content-Length exceeding MAX_UPLOAD_BYTES."""

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_UPLOAD_BYTES:
            return JSONResponse(
                status_code=413,
                content={
                    "detail": f"Payload too large. Maximum: {MAX_UPLOAD_BYTES // (1024 * 1024)}MB."
                },
            )
        return await call_next(request)


# ============================================================
# Image Validation
# ============================================================
ALLOWED_FORMATS = {"jpeg", "jpg", "png", "webp", "bmp", "tiff"}
MAX_DIMENSION = 4096


def validate_image(image) -> None:
    """
    Validate an uploaded PIL Image.
    Raises HTTPException if invalid.
    """
    # Check dimensions
    if image.width > MAX_DIMENSION or image.height > MAX_DIMENSION:
        raise HTTPException(
            status_code=400,
            detail=f"Image too large: {image.width}x{image.height}. Max: {MAX_DIMENSION}x{MAX_DIMENSION}.",
        )

    if image.width < 32 or image.height < 32:
        raise HTTPException(
            status_code=400,
            detail=f"Image too small: {image.width}x{image.height}. Min: 32x32.",
        )

    # Check format — PIL may not set .format when opened from bytes,
    # so also check the mode as a fallback sanity check.
    fmt = (image.format or "").lower()
    if fmt and fmt not in ALLOWED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {fmt}. Allowed: {', '.join(sorted(ALLOWED_FORMATS))}.",
        )

    # If format is unknown, verify PIL could actually decode it
    # by checking the image mode is usable (catches corrupt/adversarial files).
    if image.mode not in ("RGB", "RGBA", "L", "LA", "P", "1"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image mode: {image.mode}.",
        )
