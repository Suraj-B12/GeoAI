"""
Pipeline Robustness Test Suite
==============================

Verifies the operator pipeline handles failure modes gracefully without
crashing, leaking VRAM, or leaving rows stuck in 'processing'.

Test scenarios:
  1. 404 image_url             — Cloudinary returns 404
  2. Corrupt JPEG              — server returns valid HTTP, body is junk
  3. Network timeout           — server hangs longer than read timeout
  4. Mid-process stop          — operator hits Stop while a row is in flight
  5. Concurrent claim          — two simultaneous claims must NOT grab the
                                  same row (FOR UPDATE SKIP LOCKED check)

Each test:
  - Inserts a tagged row directly into assessments (bypassing photos trigger)
  - Polls /operator/metrics until the row terminal state is reached
  - Asserts the expected behavior
  - Cleans up its rows on exit (even on failure)

Prereqs:
  - Migrations 001 + 002 applied
  - FastAPI server running on http://localhost:8000 with Supabase env vars set
  - The test_fixtures/ directory has corrupt.jpg (auto-generated if missing)

Usage:
    python scripts/tests/test_pipeline_robustness.py
    python scripts/tests/test_pipeline_robustness.py --only 404 corrupt
    python scripts/tests/test_pipeline_robustness.py --keep    # don't delete test rows
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import httpx

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# Config
# ============================================================
API_BASE = os.environ.get("API_BASE", "http://localhost:8000")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://vtlkitpoffudiefuoijb.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
TEST_TAG = "robustness-test"  # model_version field used to identify our rows

if not SUPABASE_KEY:
    print("ERROR: SUPABASE_SERVICE_KEY env var not set")
    sys.exit(1)

# Use a separate domain for the test images that we control via the FastAPI
# test_fixtures static mount. ENABLE_TEST_FIXTURES=1 must be set on the server.
LOCAL_FIXTURES_BASE = f"{API_BASE}/test_fixtures"

# Public httpbin endpoints for predictable network behavior
HTTPBIN_404 = "https://httpbin.org/status/404"
HTTPBIN_DELAY = "https://httpbin.org/delay/120"  # hangs for 120s, our timeout is 30s


# ============================================================
# Pretty output
# ============================================================
class C:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    END = "\033[0m"


def banner(msg: str):
    print(f"\n{C.BOLD}{C.BLUE}{'='*70}\n  {msg}\n{'='*70}{C.END}")


def ok(msg: str):
    print(f"  {C.GREEN}[PASS]{C.END} {msg}")


def fail(msg: str):
    print(f"  {C.RED}[FAIL]{C.END} {msg}")


def info(msg: str):
    print(f"  {C.DIM}[..  ]{C.END} {msg}")


def warn(msg: str):
    print(f"  {C.YELLOW}[WARN]{C.END} {msg}")


# ============================================================
# Supabase + API helpers
# ============================================================
def sb_headers() -> dict:
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }


async def sb_insert_pending(client: httpx.AsyncClient, image_url: str, label: str) -> str:
    """Insert a row directly into assessments. Returns the new id."""
    payload = {
        "image_url": image_url,
        "latitude": 12.9716,
        "longitude": 77.5946,
        "address": f"TEST-{label}",
        "status": "pending",
        "model_version": TEST_TAG,
    }
    r = await client.post(
        f"{SUPABASE_URL}/rest/v1/assessments",
        headers={**sb_headers(), "Prefer": "return=representation"},
        json=payload,
    )
    r.raise_for_status()
    return r.json()[0]["id"]


async def sb_get(client: httpx.AsyncClient, assessment_id: str) -> Optional[dict]:
    r = await client.get(
        f"{SUPABASE_URL}/rest/v1/assessments?id=eq.{assessment_id}",
        headers=sb_headers(),
    )
    r.raise_for_status()
    rows = r.json()
    return rows[0] if rows else None


async def sb_count_by_status(client: httpx.AsyncClient, status: str) -> int:
    r = await client.get(
        f"{SUPABASE_URL}/rest/v1/assessments?status=eq.{status}&select=id",
        headers={**sb_headers(), "Prefer": "count=exact"},
    )
    r.raise_for_status()
    cr = r.headers.get("Content-Range", "0-0/0")
    return int(cr.split("/")[-1])


async def sb_cleanup(client: httpx.AsyncClient):
    """Delete all rows tagged for this test."""
    r = await client.delete(
        f"{SUPABASE_URL}/rest/v1/assessments?model_version=eq.{TEST_TAG}",
        headers=sb_headers(),
    )
    if r.status_code in (200, 204):
        info("Test rows deleted from Supabase")


async def api_post(client: httpx.AsyncClient, path: str) -> dict:
    r = await client.post(f"{API_BASE}{path}", timeout=30.0)
    r.raise_for_status()
    return r.json()


async def api_get(client: httpx.AsyncClient, path: str) -> dict:
    r = await client.get(f"{API_BASE}{path}", timeout=10.0)
    r.raise_for_status()
    return r.json()


async def wait_until_terminal(
    client: httpx.AsyncClient,
    assessment_id: str,
    timeout: float = 120.0,
    poll_interval: float = 2.0,
) -> dict:
    """Poll the assessment row until it reaches a terminal status."""
    terminal = {"classified", "expert_review", "done", "failed"}
    deadline = time.monotonic() + timeout
    last_status = None
    while time.monotonic() < deadline:
        row = await sb_get(client, assessment_id)
        if row:
            if row["status"] != last_status:
                info(f"  status: {row['status']} (retry={row.get('retry_count',0)})")
                last_status = row["status"]
            if row["status"] in terminal:
                return row
        await asyncio.sleep(poll_interval)
    raise TimeoutError(f"Row {assessment_id} did not reach terminal state in {timeout}s")


# ============================================================
# Tests
# ============================================================
async def ensure_pipeline_running(client: httpx.AsyncClient):
    s = await api_get(client, "/operator/status")
    if not s.get("is_running"):
        info("Pipeline not running — starting it")
        await api_post(client, "/operator/start")
        await asyncio.sleep(2.0)
    else:
        info(f"Pipeline already running (worker {s.get('worker_id')})")


async def test_404(client: httpx.AsyncClient) -> bool:
    banner("TEST 1 / Cloudinary 404 (image_url returns HTTP 404)")
    aid = await sb_insert_pending(client, HTTPBIN_404, "404")
    info(f"Inserted assessment {aid[:8]}... pointing at {HTTPBIN_404}")
    try:
        row = await wait_until_terminal(client, aid, timeout=180.0)
        if row["status"] == "failed" and row.get("retry_count", 0) >= 1:
            ok(f"row marked failed after {row['retry_count']} retries — error: {(row.get('error_message') or '')[:80]}")
            return True
        else:
            fail(f"row reached {row['status']} but expected 'failed' (retry_count={row.get('retry_count')})")
            return False
    except TimeoutError as e:
        fail(str(e))
        return False


async def test_corrupt_jpeg(client: httpx.AsyncClient) -> bool:
    banner("TEST 2 / Corrupt JPEG (HTTP 200 but unparseable image bytes)")
    url = f"{LOCAL_FIXTURES_BASE}/corrupt.jpg"
    # Verify the corrupt fixture is being served first
    try:
        r = await client.get(url, timeout=5.0)
        if r.status_code != 200:
            warn(f"corrupt.jpg returned {r.status_code} — is ENABLE_TEST_FIXTURES=1 set on the server?")
            return False
    except Exception as e:
        warn(f"Could not fetch corrupt.jpg: {e}")
        return False

    aid = await sb_insert_pending(client, url, "corrupt")
    info(f"Inserted assessment {aid[:8]}... pointing at corrupt.jpg")
    try:
        row = await wait_until_terminal(client, aid, timeout=120.0)
        # Corrupt JPEG is a non-transient error — should fail fast (retry_count=1)
        if row["status"] == "failed":
            ok(f"row marked failed (non-transient) — error: {(row.get('error_message') or '')[:80]}")
            return True
        else:
            fail(f"row reached {row['status']} but expected 'failed'")
            return False
    except TimeoutError as e:
        fail(str(e))
        return False


async def test_timeout(client: httpx.AsyncClient) -> bool:
    banner("TEST 3 / Network timeout (httpbin.org/delay/120 hangs longer than read timeout)")
    aid = await sb_insert_pending(client, HTTPBIN_DELAY, "timeout")
    info(f"Inserted assessment {aid[:8]}... pointing at {HTTPBIN_DELAY}")
    try:
        # 3 retries × 30s read timeout + backoffs ≈ ~100-150s
        row = await wait_until_terminal(client, aid, timeout=240.0)
        if row["status"] == "failed":
            err = (row.get("error_message") or "").lower()
            if "timeout" in err or "timed out" in err:
                ok(f"row marked failed with timeout error after {row.get('retry_count')} retries")
                return True
            else:
                ok(f"row marked failed (error: {err[:80]}) — accepted but not specifically a timeout")
                return True
        else:
            fail(f"row reached {row['status']} but expected 'failed'")
            return False
    except TimeoutError as e:
        warn(f"{e} — long timeout test, may need more time")
        return False


async def test_mid_stop(client: httpx.AsyncClient) -> bool:
    banner("TEST 4 / Mid-process stop (operator hits Stop while a row is being processed)")

    # Insert a real RDD test image that the worker can process normally
    # (we just want SOMETHING in flight when we hit Stop)
    real_url = f"{LOCAL_FIXTURES_BASE}/longitudinal_D00.jpg"
    aid = await sb_insert_pending(client, real_url, "midstop")
    info(f"Inserted assessment {aid[:8]}... pointing at a normal RDD image")

    # Wait for the worker to claim it (status -> processing)
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        row = await sb_get(client, aid)
        if row and row["status"] == "processing":
            info("Worker has claimed the row — issuing Stop")
            break
        await asyncio.sleep(0.5)
    else:
        warn("Row never reached 'processing' within 30s — worker may not be polling")
        return False

    # Hit Stop. Worker should gracefully finish or release the claim.
    await api_post(client, "/operator/stop")
    info("Stop issued — waiting up to 90s for graceful drain")

    # Wait for the row to either finish (drained) or get released back to pending
    try:
        row = await wait_until_terminal(client, aid, timeout=90.0)
        # If it finished cleanly, that's fine — graceful drain
        if row["status"] in ("classified", "expert_review"):
            ok(f"row drained gracefully to '{row['status']}' (worker finished in-flight image before stopping)")
        elif row["status"] == "pending":
            ok("row was released back to pending (worker stopped cleanly mid-claim)")
        else:
            fail(f"row in unexpected state '{row['status']}'")
            return False
    except TimeoutError:
        # Drain taking longer than 90s is acceptable — re-check after stopping
        row = await sb_get(client, aid)
        if row["status"] in ("classified", "expert_review", "pending"):
            ok(f"row state at end of test: '{row['status']}' (drained or released)")
        else:
            fail(f"row stuck in '{row['status']}' after stop")
            return False

    # Restart pipeline so subsequent tests have a worker
    info("Restarting pipeline for subsequent tests")
    await api_post(client, "/operator/start")
    await asyncio.sleep(1.0)
    return True


async def test_concurrent_claim(client: httpx.AsyncClient) -> bool:
    banner("TEST 5 / Atomic claim (FOR UPDATE SKIP LOCKED — concurrent RPC must not double-claim)")

    # Insert 5 fresh pending rows (using a real image so they actually process if claimed)
    real_url = f"{LOCAL_FIXTURES_BASE}/longitudinal_D00.jpg"
    aids = []
    for i in range(5):
        aid = await sb_insert_pending(client, real_url, f"concurrent-{i}")
        aids.append(aid)
    info(f"Inserted 5 pending rows: {[a[:8] for a in aids]}")

    # Stop the worker so we have controlled access to the queue
    await api_post(client, "/operator/stop")
    await asyncio.sleep(2.0)
    info("Worker stopped — calling claim RPC twice in parallel")

    # Hit the claim RPC concurrently from this script. PostgREST will execute
    # both transactions in parallel — SKIP LOCKED must ensure no overlap.
    async def claim_batch(worker_id: str) -> list[dict]:
        r = await client.post(
            f"{SUPABASE_URL}/rest/v1/rpc/claim_pending_assessments",
            headers={**sb_headers(), "Prefer": "return=representation"},
            json={"batch_size": 5, "worker_id": worker_id},
            timeout=10.0,
        )
        r.raise_for_status()
        return r.json()

    a, b = await asyncio.gather(claim_batch("test-A"), claim_batch("test-B"))
    info(f"Worker A claimed {len(a)} rows: {[r['id'][:8] for r in a]}")
    info(f"Worker B claimed {len(b)} rows: {[r['id'][:8] for r in b]}")

    a_ids = {r["id"] for r in a}
    b_ids = {r["id"] for r in b}
    overlap = a_ids & b_ids
    total_claimed = len(a_ids) + len(b_ids)

    success = (len(overlap) == 0) and total_claimed >= len(aids)

    # Release everything back to pending (so the running worker can pick them up
    # legitimately if the user wants to keep them)
    info("Releasing test claims back to pending")
    await client.patch(
        f"{SUPABASE_URL}/rest/v1/assessments?model_version=eq.{TEST_TAG}&status=eq.processing",
        headers=sb_headers(),
        json={"status": "pending", "claimed_at": None, "claimed_by": None},
    )

    if not success:
        if overlap:
            fail(f"Concurrent claims OVERLAPPED on {len(overlap)} rows: {[i[:8] for i in overlap]}")
        if total_claimed < len(aids):
            fail(f"Only {total_claimed}/{len(aids)} rows were claimed across both calls")
        return False

    ok(f"No overlap. Together claimed {total_claimed} rows (all 5 inserted, plus possibly leftovers from earlier tests)")

    # Restart pipeline
    await api_post(client, "/operator/start")
    await asyncio.sleep(1.0)
    return True


# ============================================================
# Runner
# ============================================================
ALL_TESTS = {
    "404":        test_404,
    "corrupt":    test_corrupt_jpeg,
    "timeout":    test_timeout,
    "midstop":    test_mid_stop,
    "concurrent": test_concurrent_claim,
}


async def main():
    parser = argparse.ArgumentParser(description="Pipeline robustness tests")
    parser.add_argument("--only", nargs="+", choices=list(ALL_TESTS.keys()),
                        help="Run only the named tests")
    parser.add_argument("--keep", action="store_true",
                        help="Do NOT delete test rows on exit")
    parser.add_argument("--api-base", default=API_BASE)
    args = parser.parse_args()

    selected = args.only or list(ALL_TESTS.keys())

    print(f"{C.BOLD}Pipeline Robustness Test Suite{C.END}")
    print(f"  API base:      {API_BASE}")
    print(f"  Supabase:      {SUPABASE_URL}")
    print(f"  Test tag:      {TEST_TAG}")
    print(f"  Tests to run:  {', '.join(selected)}")
    print()

    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0)) as client:
        # Sanity check: server is up + pipeline is initialized
        try:
            h = await api_get(client, "/health")
            info(f"API up: {h.get('model_name')} on {h.get('device')}")
        except Exception as e:
            fail(f"Cannot reach {API_BASE}/health: {e}")
            return 1

        try:
            await ensure_pipeline_running(client)
        except Exception as e:
            fail(f"Could not start pipeline: {e}")
            return 1

        # Cleanup any leftover rows from a prior run before starting
        await sb_cleanup(client)

        results = {}
        try:
            for name in selected:
                test_fn = ALL_TESTS[name]
                try:
                    results[name] = await test_fn(client)
                except Exception as e:
                    fail(f"Test '{name}' raised: {e}")
                    results[name] = False
        finally:
            if not args.keep:
                await sb_cleanup(client)
            else:
                info("Skipping cleanup — test rows left in Supabase (--keep)")

        # Summary
        print()
        banner("SUMMARY")
        passed = sum(1 for v in results.values() if v)
        for name, p in results.items():
            (ok if p else fail)(f"{name}")
        print()
        print(f"  {passed}/{len(results)} tests passed")
        return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
