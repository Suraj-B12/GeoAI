"""
Dashboard reads must not hammer Supabase (network-free).

On 2026-09-26 the Supabase database stopped answering (503 "could not query
the database for the schema cache"). The operator dashboard had been polling
/operator/metrics every 3 s for days, and every poll downloaded the whole
raw_response of every processed row. These checks pin the fix:

  1. the stage-timing average selects three JSON numbers, never raw_response;
  2. repeated polls within the TTL cost ONE database round;
  3. concurrent polls on a cold cache share ONE fetch;
  4. when Supabase errors, the last good value is served and the database is
     not queried again until the backoff has passed;
  5. with no good value yet, the error surfaces, and is not retried at once;
  6. an operator write invalidates the cache.

Supabase is replaced by an httpx.MockTransport that records every request.

Usage:
    venv/Scripts/python.exe scripts/tests/test_dashboard_cache.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import httpx  # noqa: E402

from app import supabase_client as sc  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {name}" + ("" if cond else f" - {detail}"))
    if not cond:
        FAILS.append(name)


class FakeSupabase:
    def __init__(self):
        self.requests = []
        self.down = False

    def handler(self, request: httpx.Request) -> httpx.Response:
        from urllib.parse import unquote
        url = unquote(str(request.url))      # PostgREST sees the decoded query
        self.requests.append(url)
        if self.down:
            return httpx.Response(503, json={"message": "Could not query the database for the schema cache"})
        if "pipeline_metrics" in url:
            return httpx.Response(200, json=[{"pending_count": 0, "avg_stage1_confidence_last_hour": 0.9}])
        if "pipeline_class_distribution" in url:
            return httpx.Response(200, json=[{"distress_type": "Potholes", "count": 3}])
        if "select=id" in url and "status=eq.rejected_non_pavement" in url:
            return httpx.Response(200, json=[], headers={"content-range": "0-0/4"})
        if "raw_response->stage0_time_ms" in url:
            return httpx.Response(200, json=[{"s0": 1000, "s1": 1500, "s2": 9000},
                                             {"s0": 1200, "s1": 1700, "s2": None}])
        return httpx.Response(200, json=[])


def client_for(fake):
    return httpx.AsyncClient(base_url="https://example.supabase.co",
                             transport=httpx.MockTransport(fake.handler))


async def main_async():
    fake = FakeSupabase()
    sc.METRICS_TTL_S = 0.5
    sc._read_cache.clear()
    async with client_for(fake) as c:
        m = await sc.get_pipeline_metrics(c)
        n1 = len(fake.requests)
        check("1. timing average selects JSON numbers only",
              any("raw_response->stage0_time_ms" in u for u in fake.requests)
              and not any("select=raw_response&" in u or "select=raw_response" == u.split("?")[-1]
                          for u in fake.requests),
              str(fake.requests))
        check("1. averages computed from the extracted numbers",
              m["stage_times_ms"]["stage2"] == 9000 and abs(m["stage_times_ms"]["stage0"] - 1100) < 1e-9,
              str(m.get("stage_times_ms")))

        for _ in range(20):
            await sc.get_pipeline_metrics(c)
        check("2. 20 polls inside the TTL cost no extra queries", len(fake.requests) == n1,
              f"{len(fake.requests) - n1} extra")

        sc._read_cache.clear()
        before = len(fake.requests)
        await asyncio.gather(*[sc.get_pipeline_metrics(c) for _ in range(10)])
        check("3. 10 concurrent cold polls share one fetch", len(fake.requests) - before == n1,
              f"{len(fake.requests) - before} requests vs {n1} for one fetch")

        time.sleep(0.6)                        # let the entry go stale
        fake.down = True
        before = len(fake.requests)
        m2 = await sc.get_pipeline_metrics(c)
        after_first_error = len(fake.requests)
        for _ in range(10):
            m3 = await sc.get_pipeline_metrics(c)
        check("4. during an outage the last good value is served",
              m2.get("pending_count") == 0 and m3.get("pending_count") == 0)
        check("4. and it is marked stale with the reason",
              m3.get("stale") is True and "503" in (m3.get("stale_reason") or "")
              or "Supabase" in (m3.get("stale_reason") or ""), str({k: m3.get(k) for k in ("stale", "stale_reason")}))
        check("4. and the database is not re-queried inside the backoff",
              len(fake.requests) == after_first_error and after_first_error > before)

        sc._read_cache.clear()
        before = len(fake.requests)
        try:
            await sc.get_class_distribution(c)
            raised = False
        except Exception:
            raised = True
        try:
            await sc.get_class_distribution(c)
            raised2 = False
        except Exception:
            raised2 = True
        check("5. no cached value: the error surfaces, and is not retried at once",
              raised and raised2 and len(fake.requests) - before == 1,
              f"{len(fake.requests) - before} requests")

        fake.down = False
        sc.invalidate_dashboard_cache()
        before = len(fake.requests)
        d = await sc.get_class_distribution(c)
        check("6. invalidation lets the next poll refetch", len(fake.requests) > before
              and d and d[0]["distress_type"] == "Potholes")


def main() -> int:
    asyncio.run(main_async())
    print("ALL PASSED" if not FAILS else "FAILED: " + ", ".join(FAILS))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
