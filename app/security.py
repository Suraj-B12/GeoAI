"""
API security middleware and utilities.

Provides:
  - API Key authentication (X-API-Key header)
  - Rate limiting (slowapi)
  - Upload size limiting
  - Image validation
"""

import io
import os

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from PIL import Image, UnidentifiedImageError
from starlette.datastructures import Headers
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


class LimitUploadSizeMiddleware:
    """Bound the actual body, including chunked uploads without a length.

    Validate before FastAPI parses multipart/JSON or runs an endpoint. At
    most MAX_UPLOAD_BYTES are retained; bodyless requests stay unbuffered.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        length = Headers(scope=scope).get("content-length")
        if length is not None and (not length.isascii() or not length.isdecimal()):
            return await JSONResponse(status_code=400, content={
                "detail": "Invalid Content-Length header."
            })(scope, receive, send)

        too_large = JSONResponse(status_code=413, content={
            "detail": f"Payload too large. Maximum: {MAX_UPLOAD_BYTES // (1024 * 1024)}MB."
        })
        # Comparing digit counts first also handles absurdly large headers
        # without hitting Python's integer-string conversion limit.
        digits = length.lstrip("0") if length is not None else ""
        if (len(digits) > len(str(MAX_UPLOAD_BYTES))
                or (digits and int(digits) > MAX_UPLOAD_BYTES)):
            return await too_large(scope, receive, send)
        if scope["method"] not in ("POST", "PUT", "PATCH", "DELETE"):
            return await self.app(scope, receive, send)

        chunks = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                return await too_large(scope, receive, send)
            chunks.append(chunk)
            if not message.get("more_body", False):
                break

        body = b"".join(chunks)
        chunks.clear()
        delivered = False

        async def replay():
            nonlocal delivered, body
            if not delivered:
                delivered = True
                message = {"type": "http.request", "body": body, "more_body": False}
                body = b""
                return message
            return await receive()

        await self.app(scope, replay, send)


# ============================================================
# Image Validation
# ============================================================
ALLOWED_FORMATS = {"jpeg", "jpg", "png", "webp", "bmp", "tiff"}
MAX_DIMENSION = 4096


def decode_image(contents: bytes):
    """Fully validate/decode an upload before invoking the model."""
    try:
        with Image.open(io.BytesIO(contents)) as image:
            validate_image(image)
            image.load()
            return image.copy()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=400, detail="Invalid or corrupt image.") from exc


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
