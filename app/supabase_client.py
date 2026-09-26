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


async def _exact_count(client: httpx.AsyncClient, query: str) -> Optional[int]:
    """Row count for a PostgREST filter, via the Content-Range header.

    Returns None rather than raising: every caller here is decorating a
    dashboard panel, and a missing number should degrade to a dash, not take
    the whole metrics response down.
    """
    try:
        resp = await client.get(
            f"/rest/v1/assessments?select=id&{query}",
            headers={"Prefer": "count=exact", "Range": "0-0"},
        )
        if resp.status_code >= 400:
            return None
        rng = resp.headers.get("content-range", "")
        return int(rng.split("/")[-1]) if "/" in rng else None
    except Exception:
        return None


async def _fetch_get_pipeline_metrics(client: httpx.AsyncClient) -> dict:
    """
    Read the pipeline_metrics view, then fill the gaps it cannot cover.

    Two gaps, both of which showed as wrong numbers on the dashboard:

    1. The view predates migration 004, so it has no count for
       `rejected_non_pavement`. The dashboard was reading that figure from the
       worker's in-memory counter instead, which resets to zero on every
       restart - so a database holding 9 rejected images displayed 0.

    2. The averaged confidences are windowed to the last hour. On a queue that
       is caught up, that window is empty and both averages come back NULL, so
       the performance panel reads as dead even though there is plenty of
       history. We fall back to an all-time average and flag which one is being
       shown, so the UI can label it honestly rather than imply it is recent.
    """
    resp = await client.get("/rest/v1/pipeline_metrics?select=*")
    _check(resp)
    rows = resp.json()
    out = rows[0] if rows else {}

    out["rejected_non_pavement_count"] = await _exact_count(
        client, "status=eq.rejected_non_pavement")

    out["confidence_window"] = "last_hour"
    if out.get("avg_stage1_confidence_last_hour") is None:
        alltime = await _average_confidences(client)
        if alltime:
            out["avg_stage1_confidence_last_hour"] = alltime.get("stage1")
            out["avg_stage2_confidence_last_hour"] = alltime.get("stage2")
            out["confidence_window"] = "all_time"

    # Per-stage latency. The dashboard previously took these from the worker
    # rolling averages, which are in-process and start empty - so a fresh
    # server showed a dash for all three even with a table full of timings.
    out["stage_times_ms"] = await _average_stage_times(client)
    return out


async def _average_stage_times(client: httpx.AsyncClient) -> dict:
    """All-time mean Stage 0/1/2 latency, read from raw_response.

    The worker writes stage0_time_ms / stage1_time_ms / stage2_time_ms into
    raw_response on every processed row. Stage 2 is averaged only over rows
    where it actually ran: it is skipped for Normal images and for
    non-pavement rejections, and counting those as zero would halve the
    reported figure.
    """
    empty = {"stage0": None, "stage1": None, "stage2": None, "n": 0}
    try:
        # Only the three numbers, extracted by Postgres. This used to select
        # the whole raw_response of every processed row on every dashboard
        # poll (every 3 s): ~0.7 MB per call once the Stage 2 probe started
        # storing per-type and per-tile scores, tens of GB over a few days,
        # which is the most likely reason the Supabase database stopped
        # answering on 2026-09-26 (503 "could not query the database for the
        # schema cache").
        resp = await client.get(
            "/rest/v1/assessments"
            "?select=s0:raw_response->stage0_time_ms,s1:raw_response->stage1_time_ms,"
            "s2:raw_response->stage2_time_ms&processed_at=not.is.null&limit=5000"
        )
        if resp.status_code >= 400:
            return empty
        buckets: dict[str, list] = {"stage0": [], "stage1": [], "stage2": []}
        rows = resp.json() or []
        for row in rows:
            for stage, key in (("stage0", "s0"), ("stage1", "s1"), ("stage2", "s2")):
                v = row.get(key)
                if isinstance(v, (int, float)) and v > 0:
                    buckets[stage].append(v)
        out = {k: (sum(v) / len(v) if v else None) for k, v in buckets.items()}
        out["n"] = len(rows)
        return out
    except Exception:
        return empty


async def _average_confidences(client: httpx.AsyncClient) -> Optional[dict]:
    """All-time mean Stage 1 / Stage 2 confidence over processed rows.

    PostgREST has no AVG aggregate without an RPC, so this pulls the two
    columns and averages client-side. The table is small (hundreds of rows);
    revisit with a view if it ever reaches five figures.
    """
    try:
        resp = await client.get(
            "/rest/v1/assessments"
            "?select=stage1_confidence,stage2_confidence"
            "&processed_at=not.is.null&limit=5000"
        )
        if resp.status_code >= 400:
            return None
        rows = resp.json() or []
        s1 = [r["stage1_confidence"] for r in rows if r.get("stage1_confidence") is not None]
        # Stage 2 only runs on distressed images; zeros are "did not run", not
        # "scored zero", and averaging them in would understate the metric.
        s2 = [r["stage2_confidence"] for r in rows if (r.get("stage2_confidence") or 0) > 0]
        if not s1:
            return None
        return {
            "stage1": sum(s1) / len(s1),
            "stage2": (sum(s2) / len(s2)) if s2 else None,
        }
    except Exception:
        return None


async def _fetch_get_class_distribution(client: httpx.AsyncClient) -> list[dict]:
    """Per-class counts for the distribution panel.

    `pipeline_class_distribution` windows to the last 24 hours. Whenever the
    queue is caught up - which is the normal state - that view returns an empty
    list and the panel reads "No classifications yet" despite a table full of
    results. So: use the view when it has data, otherwise aggregate all-time.

    Each row carries `window` so the UI can label which one it is showing.
    """
    resp = await client.get("/rest/v1/pipeline_class_distribution?select=*")
    _check(resp)
    rows = resp.json() or []
    if rows:
        for r in rows:
            r["window"] = "24h"
        return rows

    # All-time fallback, aggregated here because PostgREST cannot GROUP BY
    # over a jsonb array without a dedicated view.
    try:
        resp = await client.get(
            "/rest/v1/assessments"
            "?select=distress_types"
            "&status=in.(classified,expert_review,done)"
            "&limit=5000"
        )
        if resp.status_code >= 400:
            return []
        counts: dict[str, int] = {}
        for row in resp.json() or []:
            for t in (row.get("distress_types") or []):
                if t:
                    counts[t] = counts.get(t, 0) + 1
        return [
            {"distress_type": k, "count": v, "window": "all_time"}
            for k, v in sorted(counts.items(), key=lambda kv: -kv[1])
        ]
    except Exception:
        return []


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


async def _fetch_get_recent_processed(
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
# Dashboard browse API (paginated, status-filtered)
# ============================================================
_PAVEMENT_FILTER_COL = "pavement_filter_decision"
# Probe + cache whether migration 004 added the pavement_filter_* columns.
# Avoids re-querying on every dashboard request. We set the cache lazily
# on first dashboard call from main.py.
_migration_004_status: Optional[bool] = None

def _migration_004_applied() -> bool:
    """Return cached migration-004 status. Probe via probe_migration_004()."""
    return bool(_migration_004_status)


async def probe_migration_004(client: httpx.AsyncClient) -> bool:
    """One-time check: does the pavement_filter_decision column exist?

    Caches the result module-globally. Re-probes on next call ONLY when the
    cached value is None (e.g. first request after a fresh process start).
    """
    global _migration_004_status
    if _migration_004_status is not None:
        return _migration_004_status
    try:
        resp = await client.get(
            f"/rest/v1/assessments?select={_PAVEMENT_FILTER_COL}&limit=1"
        )
        _migration_004_status = resp.status_code < 400
    except Exception:
        _migration_004_status = False
    return _migration_004_status


async def list_assessments_for_dashboard(
    client: httpx.AsyncClient,
    status: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    # Lazy probe: first dashboard request determines if migration 004 ran
    await probe_migration_004(client)
    """
    Paginated list for the operator dashboard.

    Returns:
      {
        items: [{...row metadata, NO heavy fields...}],
        total: int,  # total matching rows (for pagination)
      }

    Image URLs are returned but the client decides when to actually load
    them (the dashboard uses loading='lazy' + tab-based reveal to control
    bandwidth).
    """
    # NOTE: pavement_filter_decision is added at SELECT time only if the
    # column exists (i.e. migration 004 has been applied). The probe avoids
    # failing on un-migrated databases — the dashboard still works, just
    # without per-row pre-filter info.
    # condition_indicators lives inside raw_response (Stage 2 probe, e.g.
    # "Patching"); the JSON-path alias fetches only that key, not the whole
    # raw_response blob. Rows from before the probe simply return null.
    base_cols = (
        "id,image_url,address,latitude,longitude,status,"
        "stage1_label,stage1_confidence,distress_types,severity,"
        "stage2_confidence,needs_expert_review,created_at,processed_at,"
        "condition_indicators:raw_response->condition_indicators"
    )
    select = base_cols + ("," + _PAVEMENT_FILTER_COL if _migration_004_applied() else "")
    filters = []
    if status and status != "all":
        # Comma-separated whitelist support: "classified,expert_review"
        statuses = [s.strip() for s in status.split(",") if s.strip()]
        if len(statuses) == 1:
            filters.append(f"status=eq.{statuses[0]}")
        else:
            filters.append(f"status=in.({','.join(statuses)})")
    if date_from:
        filters.append(f"created_at=gte.{date_from}")
    if date_to:
        filters.append(f"created_at=lte.{date_to}")

    limit = max(1, min(200, int(limit)))
    offset = max(0, int(offset))

    # Count via HEAD with Prefer: count=exact
    count_url = "/rest/v1/assessments?select=id"
    if filters:
        count_url += "&" + "&".join(filters)
    count_resp = await client.head(count_url, headers={"Prefer": "count=exact"})
    total = 0
    if "content-range" in count_resp.headers:
        # Format: "0-49/250"
        try:
            total = int(count_resp.headers["content-range"].split("/")[-1])
        except (ValueError, IndexError):
            total = 0

    list_url = f"/rest/v1/assessments?select={select}&order=created_at.desc"
    list_url += f"&limit={limit}&offset={offset}"
    if filters:
        list_url += "&" + "&".join(filters)
    resp = await client.get(list_url)
    _check(resp)
    items = resp.json() or []

    return {"items": items, "total": total, "limit": limit, "offset": offset}


async def reset_for_reclassify(
    client: httpx.AsyncClient,
    assessment_id: str,
) -> dict:
    """
    Reset a single assessment row back to 'pending' so the worker re-runs it.
    Used by the dashboard's "Send back to AI pipeline" button.
    """
    if not assessment_id or len(assessment_id) > 64:
        raise ValueError(f"invalid assessment_id: {assessment_id!r}")
    payload = {
        "status": "pending",
        "processed_at": None,
        "stage1_label": None,
        "stage1_confidence": None,
        "is_distressed": None,
        "distress_types": None,
        "severity": None,
        "description": None,
        "stage2_confidence": None,
        "needs_expert_review": False,
        "processing_time_ms": None,
        "raw_response": None,
        "pavement_filter_decision": None,
        "pavement_filter_raw": None,
        "pavement_filter_at": None,
        "error_message": None,
        "retry_count": 0,
        "claimed_by": None,
        "claimed_at": None,
    }
    resp = await client.patch(
        f"/rest/v1/assessments?id=eq.{assessment_id}",
        json=payload,
        headers={"Prefer": "return=representation"},
    )
    _check(resp)
    rows = resp.json() or []
    return {"reset": bool(rows), "id": assessment_id}


async def _fetch_dashboard_summary(client: httpx.AsyncClient) -> dict:
    """
    Aggregate counts by status, for the dashboard's tab badges.
    """
    resp = await client.get("/rest/v1/assessments?select=status,created_at")
    _check(resp)
    rows = resp.json() or []
    from collections import Counter
    statuses = Counter(r.get("status", "unknown") for r in rows)
    # Date histogram for filter dropdown
    dates: Counter = Counter()
    for r in rows:
        ca = (r.get("created_at") or "")[:10]
        if ca:
            dates[ca] += 1
    return {
        "total": len(rows),
        "by_status": dict(statuses.most_common()),
        "by_date": sorted(dates.items(), reverse=True)[:60],  # last 60 days
    }


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



# ============================================================
# Dashboard read cache + backoff
# ============================================================
# The operator dashboard polls /operator/metrics every 3 s and the activity
# feed every 5 s, from every open browser tab. Each poll used to run several
# PostgREST queries directly - one of them downloading every row's
# raw_response. Over 2.7 days that was ~38,000 metrics calls. These wrappers
# make the database cost independent of how many dashboards are open:
#
#   * a result is reused for `ttl` seconds (all callers share it);
#   * concurrent misses for the same key share ONE fetch;
#   * if Supabase errors, the last good value is served (marked stale by the
#     caller's own error handling path not being hit) and the query is not
#     retried until an exponential backoff (up to 2 min) has passed - a
#     recovering database is not hammered every 3 s.
#
# Only dashboard READS are cached. The worker's claim / update / heartbeat
# calls never go through here.

import asyncio as _asyncio
import time as _time


class _ReadCache:
    def __init__(self, max_backoff: float = 120.0):
        self._entries: dict = {}
        self._locks: dict = {}
        self.max_backoff = max_backoff

    async def get(self, key, ttl: float, fetch):
        ent = self._entries.get(key)
        now = _time.monotonic()
        if ent and "value" in ent and now < ent["fresh_until"]:
            return ent["value"]
        if ent and now < ent.get("retry_after", 0.0):
            if "value" in ent:
                ent["stale"] = True
                return ent["value"]
            raise ent["error"]
        lock = self._locks.setdefault(key, _asyncio.Lock())
        async with lock:
            ent = self._entries.get(key)          # another caller may have refreshed it
            now = _time.monotonic()
            if ent and "value" in ent and now < ent["fresh_until"]:
                return ent["value"]
            try:
                value = await fetch()
            except Exception as e:
                prev = ent or {}
                backoff = min(self.max_backoff, max(ttl, prev.get("backoff", ttl / 2) * 2))
                self._entries[key] = {**prev, "error": e, "backoff": backoff,
                                      "retry_after": now + backoff, "fresh_until": 0.0,
                                      "stale": "value" in prev}
                if "value" in prev:
                    return prev["value"]
                raise
            self._entries[key] = {"value": value, "fresh_until": now + ttl, "backoff": ttl / 2}
            return value

    def staleness(self, key) -> Optional[str]:
        """The error behind a value served stale, or None if it is fresh."""
        ent = self._entries.get(key) or {}
        return f"{type(ent['error']).__name__}: {ent['error']}"[:200] if ent.get("stale") else None

    def clear(self):
        self._entries.clear()


_read_cache = _ReadCache()

METRICS_TTL_S = float(os.environ.get("DASHBOARD_METRICS_TTL_S", "15"))
DISTRIBUTION_TTL_S = float(os.environ.get("DASHBOARD_DISTRIBUTION_TTL_S", "30"))
RECENT_TTL_S = float(os.environ.get("DASHBOARD_RECENT_TTL_S", "10"))
SUMMARY_TTL_S = float(os.environ.get("DASHBOARD_SUMMARY_TTL_S", "15"))


def _copy(v, key=None):
    """Shallow copy for the caller. A dict served while Supabase is failing
    carries `stale: true` and the error, so the dashboard can say so instead
    of presenting old numbers as live."""
    if isinstance(v, dict):
        out = dict(v)
        err = _read_cache.staleness(key) if key is not None else None
        if err:
            out["stale"] = True
            out["stale_reason"] = err
        return out
    return list(v) if isinstance(v, list) else v


async def get_pipeline_metrics(client: httpx.AsyncClient) -> dict:
    """Cached (METRICS_TTL_S) - see _fetch_get_pipeline_metrics."""
    return _copy(await _read_cache.get("metrics", METRICS_TTL_S,
                                       lambda: _fetch_get_pipeline_metrics(client)), "metrics")


async def get_class_distribution(client: httpx.AsyncClient) -> list[dict]:
    """Cached (DISTRIBUTION_TTL_S) - see _fetch_get_class_distribution."""
    return _copy(await _read_cache.get("distribution", DISTRIBUTION_TTL_S,
                                       lambda: _fetch_get_class_distribution(client)))


async def get_recent_processed(client: httpx.AsyncClient, limit: int = 20) -> list[dict]:
    """Cached (RECENT_TTL_S) per limit - see _fetch_get_recent_processed."""
    return _copy(await _read_cache.get(("recent", limit), RECENT_TTL_S,
                                       lambda: _fetch_get_recent_processed(client, limit)))


async def dashboard_summary(client: httpx.AsyncClient) -> dict:
    """Cached (SUMMARY_TTL_S) - see _fetch_dashboard_summary."""
    return _copy(await _read_cache.get("summary", SUMMARY_TTL_S,
                                       lambda: _fetch_dashboard_summary(client)), "summary")


def invalidate_dashboard_cache() -> None:
    """Drop cached dashboard reads after an operator write (delete,
    re-classify) so badges and feeds reflect it on the next poll."""
    _read_cache.clear()
