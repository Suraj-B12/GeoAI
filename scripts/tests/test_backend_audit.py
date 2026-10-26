"""Follow-up backend audit regressions, with all external services replaced."""

import asyncio
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI

from app import main, supabase_client as sc


def client_for(handler):
    return httpx.AsyncClient(base_url="https://database.example", transport=httpx.MockTransport(handler))


@pytest.mark.parametrize("operation", ["summary", "timings", "confidence", "distribution"])
def test_aggregates_include_rows_beyond_server_page_cap(operation):
    rows = [
        {"status": "classified", "created_at": "2026-10-25T00:00:00Z", "s0": 10,
         "s1": 20, "s2": 30, "stage1_confidence": 0.6, "stage2_confidence": 0.4,
         "distress_types": ["Potholes"]},
        {"status": "pending", "created_at": "2026-10-26T00:00:00Z", "s0": 20,
         "s1": 40, "s2": 60, "stage1_confidence": 0.8, "stage2_confidence": 0.6,
         "distress_types": ["Potholes", "Ravelling"]},
        {"status": "expert_review", "created_at": "2026-10-27T00:00:00Z", "s0": 30,
         "s1": 60, "s2": 90, "stage1_confidence": 1.0, "stage2_confidence": 0.8,
         "distress_types": ["Ravelling"]},
    ]
    offsets = []

    def handler(request):
        if request.url.path.endswith("pipeline_class_distribution"):
            return httpx.Response(200, json=[])
        offset = int(request.url.params.get("offset", 0))
        offsets.append(offset)
        # A server may cap results below the client's requested page size.
        return httpx.Response(200, json=rows[offset:offset + 2],
                              headers={"content-range": f"{offset}-{min(offset + 1, 2)}/3"})

    async def scenario():
        async with client_for(handler) as client:
            if operation == "summary":
                result = await sc._fetch_dashboard_summary(client)
                assert result["total"] == 3
                assert result["by_status"]["expert_review"] == 1
                assert result["by_date"][0] == ("2026-10-27", 1)
            elif operation == "timings":
                assert await sc._average_stage_times(client) == {
                    "stage0": 20, "stage1": 40, "stage2": 60, "n": 3}
            elif operation == "confidence":
                result = await sc._average_confidences(client)
                assert result["stage1"] == pytest.approx(0.8)
                assert result["stage2"] == pytest.approx(0.6)
            else:
                result = await sc._fetch_get_class_distribution(client)
                assert {row["distress_type"]: row["count"] for row in result} == {
                    "Potholes": 2, "Ravelling": 2}
        assert offsets == [0, 2]

    asyncio.run(scenario())


@pytest.mark.parametrize("header", [None, "0-1/*", "not-a-range"])
def test_pagination_without_count_finishes_at_empty_page(header):
    offsets = []

    def handler(request):
        offset = int(request.url.params["offset"])
        offsets.append(offset)
        assert request.url.params["order"] == "id.asc"
        assert request.url.params["limit"] == "1000"
        assert request.url.params["select"] == "status,created_at"
        rows = [{"status": "pending"}] if offset < 2 else []
        return httpx.Response(200, json=rows, headers={"content-range": header} if header else {})

    async def scenario():
        async with client_for(handler) as client:
            result = await sc._fetch_dashboard_summary(client)
        assert result == {"total": 2, "by_status": {"pending": 2}, "by_date": []}
        assert offsets == [0, 1, 2]

    asyncio.run(scenario())


def test_summary_page_failure_keeps_cached_complete_result():
    def handler(request):
        if request.url.params["offset"] == "0":
            return httpx.Response(200, json=[{"status": "pending"}],
                                  headers={"content-range": "0-0/2"})
        return httpx.Response(503, json={"message": "temporary outage"})

    async def scenario():
        cache = sc._ReadCache()
        previous = {"total": 2, "by_status": {"pending": 2}, "by_date": []}
        await cache.get("summary", 10, AsyncMock(return_value=previous))
        cache._entries["summary"]["fresh_until"] = 0
        async with client_for(handler) as client:
            assert await cache.get("summary", 10, lambda: sc._fetch_dashboard_summary(client)) == previous
        assert "503" in cache.staleness("summary")

    asyncio.run(scenario())


def test_empty_aggregates_preserve_response_shapes():
    async def scenario():
        async with client_for(lambda request: httpx.Response(200, json=[])) as client:
            assert await sc._fetch_dashboard_summary(client) == {
                "total": 0, "by_status": {}, "by_date": []}
            assert await sc._average_stage_times(client) == {
                "stage0": None, "stage1": None, "stage2": None, "n": 0}
            assert await sc._average_confidences(client) is None
            assert await sc._fetch_get_class_distribution(client) == []

    asyncio.run(scenario())


def test_summary_exceeds_old_5000_row_limit_in_bounded_pages():
    requests = []

    def handler(request):
        requests.append(request)
        offset = int(request.url.params["offset"])
        count = min(250, 5251 - offset)
        assert 0 < count <= 250
        assert request.url.params["select"] == "status,created_at"
        return httpx.Response(200, json=[{"status": "classified"}] * count,
                              headers={"content-range": f"{offset}-{offset + count - 1}/5251"})

    async def scenario():
        async with client_for(handler) as client:
            result = await sc._fetch_dashboard_summary(client)
        assert result["total"] == 5251
        assert result["by_status"] == {"classified": 5251}
        assert len(requests) == 22
        assert requests[0].headers["Prefer"] == "count=exact"
        assert all("Prefer" not in request.headers for request in requests[1:])

    asyncio.run(scenario())


def test_skipped_stage_values_still_do_not_lower_averages():
    rows = [{"s0": 10, "s1": 20, "s2": 30, "stage1_confidence": 0.8,
             "stage2_confidence": 0.6},
            {"s0": 0, "s1": None, "s2": 0, "stage1_confidence": 0.0,
             "stage2_confidence": 0.0}]

    async def scenario():
        async with client_for(lambda request: httpx.Response(
                200, json=rows, headers={"content-range": "0-1/2"})) as client:
            assert await sc._average_stage_times(client) == {
                "stage0": 10, "stage1": 20, "stage2": 30, "n": 2}
            assert await sc._average_confidences(client) == {"stage1": 0.4, "stage2": 0.6}

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", [False, True])
def test_cache_invalidation_survives_inflight_refresh(failure):
    async def scenario():
        cache = sc._ReadCache()
        await cache.get("key", 10, AsyncMock(return_value="old"))
        cache._entries["key"]["fresh_until"] = 0
        started, finish = asyncio.Event(), asyncio.Event()

        async def fetch_old():
            started.set()
            await finish.wait()
            if failure:
                raise RuntimeError("transient failure")
            return "old"

        request = asyncio.create_task(cache.get("key", 10, fetch_old))
        await started.wait()
        cache.clear()
        finish.set()
        await request
        fetch_new = AsyncMock(return_value="new")
        assert await cache.get("key", 10, fetch_new) == "new"
        fetch_new.assert_awaited_once()

    asyncio.run(scenario())


@pytest.mark.parametrize("first_result", [503, "timeout"])
def test_schema_probe_retries_transient_failure(monkeypatch, first_result):
    monkeypatch.setattr(sc, "_migration_004_status", None)
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            if first_result == "timeout":
                raise httpx.ReadTimeout("offline fixture")
            return httpx.Response(first_result, json={"message": "temporarily unavailable"})
        return httpx.Response(200, json=[])

    async def scenario():
        async with client_for(handler) as client:
            assert await sc.probe_migration_004(client) is False
            assert await sc.probe_migration_004(client) is True
            assert await sc.probe_migration_004(client) is True
        assert calls == 2

    asyncio.run(scenario())


def test_schema_probe_caches_confirmed_missing_column(monkeypatch):
    monkeypatch.setattr(sc, "_migration_004_status", None)
    handler = Mock(return_value=httpx.Response(400, json={"code": "42703"}))

    async def scenario():
        async with client_for(handler) as client:
            assert await sc.probe_migration_004(client) is False
            assert await sc.probe_migration_004(client) is False
        handler.assert_called_once()

    asyncio.run(scenario())


def test_failed_dashboard_count_is_not_reported_as_zero(monkeypatch):
    monkeypatch.setattr(sc, "_migration_004_status", True)

    def handler(request):
        if request.method == "HEAD":
            return httpx.Response(503, json={"message": "unavailable"})
        return httpx.Response(200, json=[{"id": "existing"}])

    async def scenario():
        async with client_for(handler) as client:
            with pytest.raises(sc.SupabaseError):
                await sc.list_assessments_for_dashboard(client)

    asyncio.run(scenario())


@pytest.mark.parametrize("field", ["status", "date_from", "date_to"])
def test_dashboard_filters_cannot_add_query_parameters(monkeypatch, field):
    monkeypatch.setattr(sc, "_migration_004_status", True)
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=[], headers={"content-range": "*/0"})

    async def scenario():
        value = "classified&limit=99999&select=raw_response"
        async with client_for(handler) as client:
            await sc.list_assessments_for_dashboard(client, **{field: value}, limit=5)
        for request in requests:
            assert "raw_response" not in request.url.params.get_list("select")
            expected_column = "status" if field == "status" else "created_at"
            assert request.url.params[expected_column].endswith(value)
        assert requests[-1].url.params.get_list("limit") == ["5"]

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["read", "reset", "update"])
def test_assessment_id_remains_one_filter_value(operation):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=[{"id": "fixture"}])

    async def scenario():
        value = "fixture&select=raw_response"
        async with client_for(handler) as client:
            if operation == "read":
                await sc.get_assessment_by_id(client, value)
            elif operation == "reset":
                await sc.reset_for_reclassify(client, value)
            else:
                await sc.update_assessment(client, value, {"status": "pending"})
        assert requests[0].url.params["id"] == "eq." + value
        assert "raw_response" not in requests[0].url.params.get_list("select")

    asyncio.run(scenario())


def test_dashboard_keeps_both_date_bounds_and_limit_clamps(monkeypatch):
    monkeypatch.setattr(sc, "_migration_004_status", True)
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=[], headers={"content-range": "*/0"})

    async def scenario():
        async with client_for(handler) as client:
            result = await sc.list_assessments_for_dashboard(
                client, status="classified,expert_review", date_from="2026-10-01",
                date_to="2026-10-27", limit=99999, offset=-10)
        assert result == {"items": [], "total": 0, "limit": 200, "offset": 0}
        for request in requests:
            assert request.url.params.get_list("created_at") == ["gte.2026-10-01", "lte.2026-10-27"]
            assert request.url.params["status"] == "in.(classified,expert_review)"
        assert requests[-1].url.params["order"] == "created_at.desc,id.desc"

    asyncio.run(scenario())


@pytest.mark.parametrize("download_failure", [False, True])
def test_download_response_stream_closes_on_success_or_failure(download_failure):
    class ImageStream(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            yield b"image"
            if download_failure:
                raise httpx.ReadTimeout("fixture stalled download")
            yield b" bytes"

        async def aclose(self):
            self.closed = True

    async def scenario():
        stream = ImageStream()
        async with client_for(lambda request: httpx.Response(200, stream=stream)) as client:
            if download_failure:
                with pytest.raises(httpx.ReadTimeout):
                    await sc.download_image_bytes(client, "https://images.example/road.jpg")
            else:
                assert await sc.download_image_bytes(client, "https://images.example/road.jpg") == b"image bytes"
        assert stream.closed

    asyncio.run(scenario())


@pytest.fixture
def lifecycle(monkeypatch):
    database = Mock(aclose=AsyncMock())
    images = Mock(aclose=AsyncMock())
    worker = Mock(worker_id="test-worker", is_running=True, stop=AsyncMock())
    monkeypatch.setattr(main, "get_classifier", Mock())
    monkeypatch.setattr(main, "_has_supabase_env", lambda: True)
    monkeypatch.setattr(main, "build_supabase_client", lambda: database)
    monkeypatch.setattr(main, "build_image_client", lambda: images)
    monkeypatch.setattr(main, "PipelineWorker", lambda **kwargs: worker)
    main.clear_worker()
    yield database, images, worker
    main.clear_worker()


@pytest.mark.parametrize("failure_point", ["images", "worker"])
def test_partial_pipeline_startup_closes_created_clients(monkeypatch, lifecycle, failure_point):
    database, images, _ = lifecycle
    target = "build_image_client" if failure_point == "images" else "PipelineWorker"
    monkeypatch.setattr(main, target, Mock(side_effect=RuntimeError("fixture startup failure")))

    async def scenario():
        app = FastAPI()
        async with main.lifespan(app):
            database.aclose.assert_awaited_once()
            if failure_point == "worker":
                images.aclose.assert_awaited_once()
            assert app.state.supabase_client is None
            assert main.get_worker() is None
        database.aclose.assert_awaited_once()

    asyncio.run(scenario())


@pytest.mark.parametrize("exit_error", [RuntimeError, asyncio.CancelledError])
def test_exceptional_lifespan_exit_stops_worker_and_closes_clients(lifecycle, exit_error):
    database, images, worker = lifecycle

    async def scenario():
        with pytest.raises(exit_error):
            async with main.lifespan(FastAPI()):
                raise exit_error("fixture shutdown")
        worker.stop.assert_awaited_once()
        database.aclose.assert_awaited_once()
        images.aclose.assert_awaited_once()
        assert main.get_worker() is None

    asyncio.run(scenario())


def test_client_close_failure_does_not_leak_other_client(lifecycle):
    database, images, worker = lifecycle
    database.aclose.side_effect = RuntimeError("fixture close failure")

    async def scenario():
        async with main.lifespan(FastAPI()):
            pass
        worker.stop.assert_awaited_once()
        database.aclose.assert_awaited_once()
        images.aclose.assert_awaited_once()
        assert main.get_worker() is None

    asyncio.run(scenario())


def test_cancellation_during_client_close_still_closes_other_client(lifecycle):
    database, images, _ = lifecycle
    database.aclose.side_effect = asyncio.CancelledError()

    async def scenario():
        with pytest.raises(asyncio.CancelledError):
            async with main.lifespan(FastAPI()):
                pass
        images.aclose.assert_awaited_once()

    asyncio.run(scenario())
