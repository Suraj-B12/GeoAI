"""
Async Supabase REST client for the operator pipeline.

Wraps PostgREST + RPC calls used by the pipeline worker:
  - claim_pending_assessments(batch_size)  -> atomic claim via RPC
  - update_assessment(id, fields)          -> PATCH single row
  - reset_stale_processing(seconds)        -> RPC for crash recovery
  - upsert_worker_state(...)               -> heartbeat
  - get_pipeline_metrics()                 -> dashboard view
  - download_image(url)                    -> Cloudinary download

Uses ONE shared httpx.AsyncClient — created in FastAPI lifespan and
passed in. Don't construct this class per-request.

Env vars required:
    SUPABASE_URL        = https://<project-ref>.supabase.co
    SUPABASE_SERVICE_KEY = sb_secret_xxx  (server-side only — bypasses RLS)
"""

from __future__ import annotations

import os
from typing import Any, Optional

import httpx


# ============================================================
# Configuration
# ============================================================

DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0)
DEFAULT_LIMITS = httpx.Limits(
    max_connections=20,
    max_keepalive_connections=10,
    keepalive_expiry=30.0,
)


def _env(name: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """Read env var with required-flag guard."""
    val = os.environ.get(name, default)
    if required and not val:
        raise RuntimeError(
            f"Required environment variable '{name}' is not set. "
            f"Set it in .env or your shell before starting the server."
        )
    return val


def build_supabase_client(
    url: Optional[str] = None,
    service_key: Optional[str] = None,
) -> httpx.AsyncClient:
    """
    Build the shared httpx.AsyncClient for Supabase calls.

    Caller is responsible for closing it (use lifespan).
    """
    url = url or _env("SUPABASE_URL", required=True)
    service_key = service_key or _env("SUPABASE_SERVICE_KEY", required=True)

    return httpx.AsyncClient(
        base_url=url.rstrip("/"),
        timeout=DEFAULT_TIMEOUT,
        limits=DEFAULT_LIMITS,
        headers={
            "apikey": service_key,
            "Authorization": f"Bearer {service_key}",
            "Content-Type": "application/json",
        },
    )


def build_image_client() -> httpx.AsyncClient:
    """
    Separate httpx client for downloading images from Cloudinary / external URLs.
    No auth headers, longer read timeout (large images on slow networks).
    """
    return httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=5.0),
        limits=DEFAULT_LIMITS,
        follow_redirects=True,
    )


# ============================================================
# Errors
# ============================================================

class SupabaseError(Exception):
    """Raised for any non-2xx Supabase response."""
    def __init__(self, status_code: int, message: str, body: Any = None):
        self.status_code = status_code
        self.body = body
        super().__init__(f"Supabase error {status_code}: {message}")


def _check(resp: httpx.Response) -> None:
    """Raise SupabaseError on non-2xx with the response body for debugging."""
    if resp.is_success:
        return
    try:
        body = resp.json()
        msg = body.get("message") or body.get("error") or resp.text
    except Exception:
        body = resp.text
        msg = resp.text
    raise SupabaseError(resp.status_code, msg, body)


# ============================================================
# Pipeline operations
# ============================================================

async def claim_pending_batch(
    client: httpx.AsyncClient,
    batch_size: int,
    worker_id: str,
) -> list[dict]:
    """
    Atomically claim a batch of pending assessments.

    Calls the claim_pending_assessments(batch_size, worker_id) RPC defined in
    migration 001. Uses FOR UPDATE SKIP LOCKED, safe for concurrent workers.

    Returns the claimed rows (list of dicts). Empty list if queue is empty.
    """
    resp = await client.post(
        "/rest/v1/rpc/claim_pending_assessments",
        json={"batch_size": batch_size, "worker_id": worker_id},
    )
    _check(resp)
    return resp.json() or []


async def update_assessment(
    client: httpx.AsyncClient,
    assessment_id: str,
    fields: dict,
) -> dict:
    """
    PATCH a single assessment row by id. Returns the updated row.
    """
    resp = await client.patch(
        f"/rest/v1/assessments?id=eq.{assessment_id}",
        json=fields,
        headers={"Prefer": "return=representation"},
    )
    _check(resp)
    rows = resp.json()
    if not rows:
        raise SupabaseError(404, f"No assessment with id={assessment_id}")
    return rows[0]


async def reset_stale_processing(
    client: httpx.AsyncClient,
    stale_threshold_seconds: int = 300,
) -> int:
    """
    Reset rows stuck in 'processing' for too long back to 'pending'.

    Increments their retry_count. Call on worker startup AND periodically
    inside the worker loop so a crashed worker's claims get recovered.

    Returns the number of rows reset.
    """
    resp = await client.post(
        "/rest/v1/rpc/reset_stale_processing_assessments",
        json={"stale_threshold_seconds": stale_threshold_seconds},
    )
    _check(resp)
    # Function returns INTEGER; PostgREST wraps single scalar in array
    result = resp.json()
    if isinstance(result, list):
        return result[0] if result else 0
    return int(result or 0)


async def upsert_worker_state(
    client: httpx.AsyncClient,
    worker_id: str,
    fields: dict,
) -> dict:
    """
    Upsert the worker_state row for this worker. Used for heartbeat
    and cumulative metrics.
    """
    payload = {"worker_id": worker_id, **fields}
    resp = await client.post(
        "/rest/v1/worker_state",
        json=payload,
        headers={
            "Prefer": "resolution=merge-duplicates,return=representation",
        },
    )
    _check(resp)
    rows = resp.json()
    return rows[0] if rows else {}


async def get_worker_state(
    client: httpx.AsyncClient,
    worker_id: str,
) -> Optional[dict]:
    """Fetch the worker_state row for this worker, or None if it doesn't exist."""
    resp = await client.get(
        f"/rest/v1/worker_state?worker_id=eq.{worker_id}&select=*",
    )
    _check(resp)
    rows = resp.json()
    return rows[0] if rows else None


async def get_pipeline_metrics(client: httpx.AsyncClient) -> dict:
    """
    Read the pipeline_metrics view. Returns a single dict with all
    aggregate counts used by the operator dashboard.
    """
    resp = await client.get("/rest/v1/pipeline_metrics?select=*")
    _check(resp)
    rows = resp.json()
    return rows[0] if rows else {}


async def get_class_distribution(client: httpx.AsyncClient) -> list[dict]:
    """Read pipeline_class_distribution view (list of {distress_type, count})."""
    resp = await client.get("/rest/v1/pipeline_class_distribution?select=*")
    _check(resp)
    return resp.json() or []


async def get_assessment_by_id(
    client: httpx.AsyncClient,
    assessment_id: str,
) -> Optional[dict]:
    """
    Fetch a single assessment row by ID. Used by the dashboard to render
    the live preview of the in-flight image (URL, GPS, address).

    Returns None if the row doesn't exist (e.g., already moved to done
    by the time the dashboard polled).
    """
    # Validate UUID format minimally to avoid odd characters in the URL.
    # PostgREST will reject malformed UUIDs anyway, but cheap to gate here.
    if not assessment_id or len(assessment_id) > 64:
        return None

    resp = await client.get(
        "/rest/v1/assessments"
        f"?id=eq.{assessment_id}"
        "&select=id,image_url,latitude,longitude,address,status,"
        "stage1_label,stage1_confidence,distress_types,severity,"
        "stage2_confidence,error_message,claimed_at,processed_at"
        "&limit=1"
    )
    _check(resp)
    rows = resp.json() or []
    return rows[0] if rows else None


async def get_recent_processed(
    client: httpx.AsyncClient,
    limit: int = 20,
) -> list[dict]:
    """
    Fetch the most recently processed assessments for the dashboard
    activity feed. Includes status, confidences, and class.
    """
    resp = await client.get(
        "/rest/v1/assessments"
        "?select=id,status,stage1_label,stage1_confidence,distress_types,"
        "severity,stage2_confidence,processed_at,error_message"
        f"&processed_at=not.is.null&order=processed_at.desc&limit={limit}"
    )
    _check(resp)
    return resp.json() or []


# ============================================================
# Image download
# ============================================================

async def download_image_bytes(
    image_client: httpx.AsyncClient,
    url: str,
) -> bytes:
    """
    Download an image from a public URL (typically Cloudinary).

    Raises httpx.HTTPError on network failure. Caller is responsible
    for retry logic.
    """
    resp = await image_client.get(url)
    resp.raise_for_status()
    return resp.content
