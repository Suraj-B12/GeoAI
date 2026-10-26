"""Offline regressions: no model downloads, GPU, or production service calls.

Run: python -m pytest -q scripts/tests/test_reliability_regressions.py
"""

import asyncio
import base64
import importlib.util
import io
import json
import re
import shutil
import subprocess
import threading
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import deletion, main, security, supabase_client, worker
from scripts.tests.test_delete_everywhere import FakeWorld, run


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setattr(main.limiter, "enabled", False)
    monkeypatch.setattr(security, "AUTH_ENABLED", False)
    # No lifespan is entered: model startup and Supabase are never invoked.
    return TestClient(main.app, raise_server_exceptions=False)


def png_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (40, 40), "gray").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.parametrize("endpoint", ["/classify", "/classify/stream"])
def test_corrupt_upload_is_a_client_error(api, monkeypatch, endpoint):
    classifier = Mock()
    monkeypatch.setattr(main, "get_classifier", classifier)
    response = api.post(endpoint, files={"file": ("bad.jpg", b"broken", "image/jpeg")})
    assert response.status_code == 400
    classifier.assert_not_called()


@pytest.mark.parametrize("value", [None, 1, [], "", "%%%%", "YQ==!!"])
def test_bad_base64_is_a_client_error(api, monkeypatch, value):
    classifier = Mock()
    monkeypatch.setattr(main, "get_classifier", classifier)
    assert api.post("/classify/base64", json={"image": value}).status_code == 400
    classifier.assert_not_called()


def test_valid_base64_preserves_classification(api, monkeypatch):
    expected = {
        "is_distressed": False, "stage1_label": "Normal", "stage1_confidence": 0.95,
        "processing_time_ms": 1.0, "stage1_time_ms": 1.0,
    }
    classifier = Mock()
    classifier.predict.return_value = expected
    monkeypatch.setattr(main, "get_classifier", lambda: classifier)
    response = api.post("/classify/base64", json={
        "image": base64.b64encode(png_bytes()).decode("ascii"),
    })
    assert response.status_code == 200
    assert response.json()["stage1_label"] == "Normal"
    assert classifier.predict.call_args.args[0].getpixel((0, 0)) == (128, 128, 128)


def test_valid_upload_still_streams_the_result(api, monkeypatch):
    classifier = Mock()
    classifier.predict_stage1.return_value = {
        "is_distressed": False, "stage1_label": "Normal", "stage1_confidence": 0.95,
        "stage1_time_ms": 1.0, "stage1_raw": "Normal", "needs_expert_review": False,
    }
    classifier.stage1_safety_check.return_value = None
    monkeypatch.setattr(main, "get_classifier", lambda: classifier)
    response = api.post("/classify/stream", files={
        "file": ("road.png", png_bytes(), "image/png"),
    })
    assert response.status_code == 200
    assert "event: result" in response.text
    assert '"stage1_label": "Normal"' in response.text


@pytest.mark.parametrize("header", ["bad", "-1", "1.5", "+12"])
def test_malformed_content_length(api, header):
    assert api.post("/classify/base64", content=b"{}", headers={
        "Content-Length": header,
    }).status_code == 400


def test_streamed_upload_limit_counts_actual_bytes(monkeypatch):
    monkeypatch.setattr(security, "MAX_UPLOAD_BYTES", 4)

    async def scenario(chunks, length=None):
        messages = iter({"type": "http.request", "body": c,
                         "more_body": i < len(chunks) - 1}
                        for i, c in enumerate(chunks))
        sent = []
        app = AsyncMock()
        headers = [(b"content-length", length)] if length is not None else []

        async def receive():
            return next(messages)

        async def send(message):
            sent.append(message)

        await security.LimitUploadSizeMiddleware(app)(
            {"type": "http", "method": "POST", "headers": headers}, receive, send)
        assert sent[0]["status"] == 413
        app.assert_not_awaited()

    asyncio.run(scenario([b"abc", b"de"]))
    asyncio.run(scenario([b"abc", b"de"], b"1"))


def test_huge_content_length_is_rejected_without_integer_overflow(api):
    assert api.post("/classify/base64", content=b"{}", headers={
        "Content-Length": "9" * 5000,
    }).status_code == 413


@pytest.mark.parametrize("payload", [[], None, 1, "body", {"quantization_bits": False},
                                     {"quantization_bits": 4.0}, {"quantization_bits": "²"},
                                     {"quantization_bits": "--4"}])
def test_bad_runtime_body_does_not_reload(api, monkeypatch, payload):
    classifier = Mock()
    monkeypatch.setattr(main, "get_classifier", classifier)
    assert api.post("/operator/runtime/quantization", json=payload).status_code == 400
    classifier.assert_not_called()


@pytest.mark.parametrize("payload", [{}, [], {"adapter_path": []}, {"adapter_path": True}])
def test_invalid_adapter_request_cannot_unload_model(api, monkeypatch, payload):
    classifier = Mock()
    monkeypatch.setattr(main, "get_classifier", classifier)
    assert api.post("/operator/adapters/switch", json=payload).status_code == 400
    classifier.assert_not_called()


def test_dashboard_delete_cors_preflight(api):
    response = api.options("/dashboard/delete/example", headers={
        "Origin": "https://example.com",
        "Access-Control-Request-Method": "DELETE",
    })
    assert response.status_code == 200
    assert "DELETE" in response.headers["access-control-allow-methods"]


@pytest.mark.parametrize("status", [403, 429, 500, 503, "timeout"])
def test_uncertain_cloudinary_removal_preserves_database(monkeypatch, status):
    monkeypatch.setattr(deletion, "cloudinary_credentials", lambda: ("testcloud", "key", "secret"))
    world = FakeWorld(destroy=("not found", 200))
    original = world.http

    def cloudinary(request):
        if request.method == "HEAD":
            if status == "timeout":
                raise httpx.ReadTimeout("offline test")
            return httpx.Response(status)
        return original(request)

    world.http = cloudinary
    result, error = run(world)
    assert result is None
    assert error.status_code == 502
    assert world.db_deletes() == 0
    assert world.assessment and world.photo


def test_cloudinary_hostname_must_have_domain_boundary():
    assert deletion.parse_cloudinary_url(
        "https://notcloudinary.com/testcloud/image/upload/v1/test.jpg") is None
    assert deletion.parse_cloudinary_url(
        "https://res.cloudinary.com:443/testcloud/image/upload/v1/test.jpg").public_id == "test"


@pytest.mark.parametrize("has_cached_value", [False, True])
def test_concurrent_cache_failures_share_backoff(has_cached_value):
    async def scenario():
        cache = supabase_client._ReadCache()
        if has_cached_value:
            await cache.get("key", 10, AsyncMock(return_value={"total": 2}))
            cache._entries["key"]["fresh_until"] = 0
        calls = 0

        async def fail():
            nonlocal calls
            calls += 1
            # Let every caller queue behind the first database fetch.
            await asyncio.sleep(0.01)
            raise RuntimeError("temporary outage")

        results = await asyncio.gather(*[cache.get("key", 10, fail) for _ in range(10)],
                                       return_exceptions=True)
        assert calls == 1
        if has_cached_value:
            assert all(value == {"total": 2} for value in results)
            assert cache.staleness("key") is not None
        else:
            assert all(isinstance(value, RuntimeError) for value in results)

    asyncio.run(scenario())


def make_worker(monkeypatch):
    instance = worker.PipelineWorker(None, None, None, worker_id="offline-test")
    monkeypatch.setattr(instance, "_write_heartbeat", AsyncMock())
    monkeypatch.setattr(instance, "_release_claim", AsyncMock())
    monkeypatch.setattr(worker, "reset_stale_processing", AsyncMock(return_value=0))
    return instance


def test_concurrent_start_creates_only_one_worker_loop(monkeypatch):
    async def scenario():
        instance = make_worker(monkeypatch)

        async def reset(*args, **kwargs):
            await asyncio.sleep(0.01)
            return 0

        monkeypatch.setattr(worker, "reset_stale_processing", reset)
        loops = 0

        async def loop():
            nonlocal loops
            loops += 1
            await instance._stop_event.wait()

        monkeypatch.setattr(instance, "_run_loop", loop)
        results = await asyncio.gather(instance.start(), instance.start(), instance.start())
        await instance.stop()
        assert loops == 1
        assert sorted(result["status"] for result in results) == [
            "already_running", "already_running", "started"]

    asyncio.run(scenario())


def test_mid_batch_stop_releases_every_remaining_claim(monkeypatch):
    async def scenario():
        instance = make_worker(monkeypatch)
        rows = [{"id": str(i)} for i in range(3)]
        monkeypatch.setattr(worker, "claim_pending_batch", AsyncMock(return_value=rows))

        async def process(row):
            instance._stop_event.set()

        monkeypatch.setattr(instance, "_process_one", AsyncMock(side_effect=process))
        await instance.start()
        await instance._task
        assert instance._process_one.await_count == 1
        assert {call.args[0] for call in instance._release_claim.await_args_list} == {"1", "2"}
        assert not instance.is_running

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel", [False, True])
def test_worker_failure_releases_claims_and_clears_running_state(monkeypatch, cancel):
    async def scenario():
        instance = make_worker(monkeypatch)
        rows = [{"id": str(i)} for i in range(3)]
        monkeypatch.setattr(worker, "claim_pending_batch", AsyncMock(return_value=rows))
        entered = asyncio.Event()

        async def process(row):
            entered.set()
            if cancel:
                await asyncio.Event().wait()
            raise RuntimeError("unexpected processing failure")

        monkeypatch.setattr(instance, "_process_one", process)
        await instance.start()
        await entered.wait()
        if cancel:
            await instance.stop(drain_timeout_seconds=0.01)
        else:
            await instance._task
        assert {call.args[0] for call in instance._release_claim.await_args_list} == {"0", "1", "2"}
        assert not instance.is_running

    asyncio.run(scenario())


def test_adapter_switch_drains_worker_and_resolves_project_path(api, monkeypatch):
    from scripts.tests.test_operator_runtime import FakeWorker

    instance = FakeWorker()
    instance._running = True
    classifier = Mock()

    def reload(path):
        assert not instance.is_running
        assert Path(path).is_absolute()
        return {"reloaded": True}

    classifier.reload_adapter.side_effect = reload
    monkeypatch.setattr(main, "get_classifier", lambda: classifier)
    monkeypatch.setattr(main, "get_worker", lambda: instance)
    response = api.post("/operator/adapters/switch", json={"adapter_path": "adapters/test"})
    assert response.status_code == 200
    assert instance.events == ["stop", "start"]
    assert response.json()["worker_restarted"] is True


def test_failed_worker_restart_is_reported_accurately(api, monkeypatch):
    from scripts.tests.test_operator_runtime import FakeClassifier, FakeWorker

    instance = FakeWorker()
    instance._running = True
    instance.fail_start = True
    monkeypatch.setattr(main, "get_classifier", FakeClassifier)
    monkeypatch.setattr(main, "get_worker", lambda: instance)
    response = api.post("/operator/runtime/quantization", json={"quantization_bits": 0})
    assert response.status_code == 200
    assert response.json()["worker_restarted"] is False
    assert "did NOT restart" in response.json()["worker_note"]


def test_cancelled_reload_keeps_worker_paused_until_thread_finishes(monkeypatch):
    from scripts.tests.test_operator_runtime import FakeWorker

    async def scenario():
        instance = FakeWorker()
        instance._running = True
        started, finish = threading.Event(), threading.Event()
        monkeypatch.setattr(main, "get_worker", lambda: instance)
        monkeypatch.setattr(main, "_runtime_reload_lock", asyncio.Lock())

        def reload():
            started.set()
            assert finish.wait(5)
            return {"reloaded": True}

        task = asyncio.create_task(main._reload_classifier(reload))
        try:
            assert await asyncio.to_thread(started.wait, 5)
            task.cancel()
            await asyncio.sleep(0)
            assert not instance.is_running
            assert main._runtime_reload_lock.locked()
        finally:
            finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert instance.events == ["stop", "start"]
        assert not main._runtime_reload_lock.locked()

    asyncio.run(scenario())


@pytest.mark.parametrize("types,severity", [
    (["Pothole (D40)", "Ravelling"], "High"),
    (["Pothole (D40)"], "Low"),
    ([], "None"),
])
def test_expert_training_keeps_all_labels_and_reviewed_severity(types, severity):
    path = Path(__file__).resolve().parents[1] / "06_incremental_retrain.py"
    spec = importlib.util.spec_from_file_location("incremental_retrain", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.build_correction_entry({
        "distress_types": types, "severity": severity, "image_path": "test.jpg",
    })
    text = result["conversations"][1]["value"]
    assert f"SEVERITY: {severity}\n" in text
    for label in types:
        assert label in text


class Tags(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.tags = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js required for UI rendering checks")
@pytest.mark.parametrize("page", ["expert_ui/index.html", "operator_ui/dashboard.html"])
def test_untrusted_ui_values_stay_in_data_attributes(page):
    root = Path(__file__).resolve().parents[2]
    source = (root / page).read_text()
    functions = (["escapeHtml", "escapeAttr", "buildCardHTML"] if page.startswith("expert")
                 else ["esc", "renderCard"])
    code = "\n".join(re.search(r"function " + name + r"\b[\s\S]*?^}", source, re.M).group()
                     for name in functions)
    label = "Expert's crack');globalThis.injected=true;//"
    url = 'https://example.com/photo.jpg\" onerror=\"globalThis.injected=true'
    row = {"id": "test-id", "image_url": url, "distress_types": [label], "severity": "Low"}
    runner = """
        const fs = require('node:fs');
        const vm = require('node:vm');
        const input = JSON.parse(fs.readFileSync(0, 'utf8'));
        const context = vm.createContext({
            row: input.row, distressTypes: input.row.distress_types,
            optimizedImageUrl: value => value, badgeClass: () => 'low', pct: () => '0%',
            fmtDate: () => '', fmtTime: () => '', statusLabel: value => value,
            state: { loadedImageIds: new Set() },
        });
        vm.runInContext(input.code, context);
        process.stdout.write(vm.runInContext(input.call, context));
    """
    call = "buildCardHTML(row, 0, 'pending')" if page.startswith("expert") else "renderCard(row)"
    result = subprocess.run([shutil.which("node"), "-e", runner], input=json.dumps({
        "code": code, "row": row, "call": call,
    }), capture_output=True, text=True, check=True)
    tags = Tags(result.stdout).tags
    image = next(attrs for tag, attrs in tags if tag == "img")
    assert image.get("src") == url or image.get("data-src") == url
    assert "globalThis.injected" not in image.get("onerror", "")
    assert all("globalThis.injected" not in attrs.get("onclick", "") for _, attrs in tags)
    if page.startswith("expert"):
        assert any(attrs.get("data-type") == label for _, attrs in tags)
        assert image["data-image-url"] == url
