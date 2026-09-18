"""
Tests for the operator runtime-precision control (GPU-free).

The endpoint under test orchestrates a sequence that is genuinely easy to get
wrong and expensive to discover in production:

    stop the worker -> reload the model -> restart the worker

with the requirement that the worker is restarted on EVERY path, including the
failure paths, so a rejected request never silently leaves the pipeline
stopped. A real reload needs a free GPU for ~60s, so this substitutes a fake
classifier and asserts the orchestration instead.

What is NOT covered here: that a real CUDA reload at a different precision
actually succeeds. That needs the GPU and is verified by loading the server.

Usage:
    venv/Scripts/python.exe scripts/tests/test_operator_runtime.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import app.main WITHOUT loading a real model onto the GPU.
os.environ.setdefault("SKIP_MODEL_SELFTEST", "1")

FAILURES: list[str] = []
CHECKS = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if cond:
        print(f"  [OK]   {name}")
    else:
        print(f"  [FAIL] {name}" + (f" - {detail}" if detail else ""))
        FAILURES.append(name)


# ============================================================
# Fakes
# ============================================================

class FakeClassifier:
    """Stands in for PavementClassifier. Records reload calls."""

    def __init__(self):
        self.model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
        self.adapter_path = None
        self.quantization_bits = 4
        self.reload_calls: list[int] = []
        self.fail_next_reload = False
        self._device = "cuda"

    # --- surface used by /operator/runtime and /health ---
    def runtime_info(self) -> dict:
        return {
            "model_path": self.model_path,
            "adapter_path": self.adapter_path,
            "quantization_bits": self.quantization_bits,
            "quantization_label": {0: "bf16 (no quantization)", 4: "4-bit NF4",
                                   8: "8-bit int8"}[self.quantization_bits],
            "quantization_effective": self.quantization_bits,
            "oom_fallback_used": False,
            "model_family": "qwen2_5_vl",
            "model_class": "Qwen2_5_VLForConditionalGeneration",
            "max_image_pixels": 4840000,
            "stage2_confidence_mode": "field",
            "prompts_version": "v2_improved_baseline",
            "device": self._device,
            "is_loaded": True,
            "vram": {"total_gb": 24.0, "free_gb": 7.4, "used_gb": 16.6,
                     "gpu_name": "NVIDIA RTX A5000"},
            "options": [
                {"bits": 0, "label": "bf16 (no quantization)", "weights_gb": 16.6, "note": "n"},
                {"bits": 4, "label": "4-bit NF4", "weights_gb": 4.3, "note": "n"},
                {"bits": 8, "label": "8-bit int8", "weights_gb": 8.6, "note": "n"},
            ],
        }

    def reload(self, quantization_bits=None, adapter_path=None):
        bits = self.quantization_bits if quantization_bits is None else quantization_bits
        self.reload_calls.append(bits)
        if self.fail_next_reload:
            self.fail_next_reload = False
            raise RuntimeError("simulated CUDA OOM during reload")
        if bits == self.quantization_bits:
            info = self.runtime_info()
            info["reloaded"] = False
            info["reason"] = "already running this configuration"
            return info
        self.quantization_bits = bits
        info = self.runtime_info()
        info["reloaded"] = True
        return info

    # --- surface used by /health ---
    @property
    def is_loaded(self):
        return True

    @property
    def device(self):
        return self._device

    @property
    def has_adapter(self):
        return self.adapter_path is not None

    @property
    def load_info(self):
        return {"model_type": "qwen2_5_vl", "model_class": "Fake",
                "quantization_bits": self.quantization_bits,
                "oom_fallback_used": False, "max_pixels": 4840000,
                "vram_used_gb": 16.6}

    @property
    def confidence_mode(self):
        return "field"


class FakeWorker:
    """Stands in for PipelineWorker. Records start/stop ordering."""

    def __init__(self):
        self._running = False
        self.events: list[str] = []
        self.fail_start = False

    @property
    def is_running(self):
        return self._running

    async def start(self):
        if self.fail_start:
            self.events.append("start_failed")
            raise RuntimeError("simulated worker start failure")
        self._running = True
        self.events.append("start")
        return {"status": "started"}

    async def stop(self, drain_timeout_seconds: float = 60.0):
        self._running = False
        self.events.append("stop")
        return {"status": "stopped"}

    def metrics_snapshot(self):
        return {"worker_id": "fake", "is_running": self._running}


# ============================================================
# Harness
# ============================================================

def build_client(fake_clf: FakeClassifier, fake_worker: FakeWorker):
    from fastapi.testclient import TestClient
    import app.main as main

    main.get_classifier = lambda *a, **k: fake_clf
    main.get_worker = lambda: fake_worker

    # Neutralise the lifespan: it would load a real model and build real
    # Supabase clients. Ordering assertions do not need either.
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def noop_lifespan(app):
        app.state.supabase_client = None
        app.state.image_client = None
        yield

    main.app.router.lifespan_context = noop_lifespan

    # The endpoint is rate-limited to 4/minute in production because a reload
    # costs ~60s of GPU time. That limit is asserted separately in step 7; the
    # orchestration tests below would otherwise exhaust it and 429.
    main.limiter.enabled = False

    return TestClient(main.app), main


def main() -> int:
    clf = FakeClassifier()
    worker = FakeWorker()
    client, _main = build_client(clf, worker)

    print("\n[1/7] GET /operator/runtime")
    r = client.get("/operator/runtime")
    check("returns 200", r.status_code == 200, f"got {r.status_code}: {r.text[:200]}")
    body = r.json() if r.status_code == 200 else {}
    check("reports current precision", body.get("quantization_bits") == 4, str(body)[:200])
    check("offers all three options", len(body.get("options", [])) == 3)
    check("reports live VRAM", "vram" in body and body["vram"].get("total_gb") == 24.0)
    check("reports worker state", body.get("worker_running") is False)
    check("reports the confidence mode", body.get("stage2_confidence_mode") == "field")

    print("\n[2/7] Invalid precision is rejected")
    for bad in (3, 16, "abc", None, -1):
        r = client.post("/operator/runtime/quantization", json={"quantization_bits": bad})
        check(f"{bad!r} -> 400", r.status_code == 400, f"got {r.status_code}")
    check("no reload was attempted for bad input", clf.reload_calls == [],
          f"reload_calls={clf.reload_calls}")

    print("\n[3/7] Switch while the worker is STOPPED")
    r = client.post("/operator/runtime/quantization", json={"quantization_bits": 0})
    check("returns 200", r.status_code == 200, r.text[:200])
    body = r.json() if r.status_code == 200 else {}
    check("reload happened", body.get("reloaded") is True)
    check("now bf16", clf.quantization_bits == 0)
    check("worker was not touched", worker.events == [], f"events={worker.events}")
    check("response says worker not restarted", body.get("worker_restarted") is False)

    print("\n[4/7] Switch while the worker is RUNNING")
    worker._running = True
    worker.events.clear()
    r = client.post("/operator/runtime/quantization", json={"quantization_bits": 4})
    check("returns 200", r.status_code == 200, r.text[:200])
    body = r.json() if r.status_code == 200 else {}
    check("worker stopped then restarted, in that order",
          worker.events == ["stop", "start"], f"events={worker.events}")
    check("worker is running again", worker.is_running is True)
    check("response says worker restarted", body.get("worker_restarted") is True)
    check("precision applied", clf.quantization_bits == 4)

    print("\n[5/7] No-op switch does not reload")
    worker.events.clear()
    before = len(clf.reload_calls)
    r = client.post("/operator/runtime/quantization", json={"quantization_bits": 4})
    body = r.json() if r.status_code == 200 else {}
    check("returns 200", r.status_code == 200)
    check("reported as no-op", body.get("reloaded") is False, str(body)[:200])
    check("classifier was asked exactly once", len(clf.reload_calls) == before + 1)

    print("\n[6/7] Failed reload still restarts the worker")
    worker._running = True
    worker.events.clear()
    clf.fail_next_reload = True
    r = client.post("/operator/runtime/quantization", json={"quantization_bits": 0})
    check("returns 500", r.status_code == 500, f"got {r.status_code}")
    check("error explains the previous config was restored",
          "restored" in r.text.lower(), r.text[:200])
    check("worker was restarted despite the failure",
          worker.events == ["stop", "start"], f"events={worker.events}")
    check("pipeline is running again", worker.is_running is True)

    print("\n[7/7] The endpoint is rate-limited")
    _main.limiter.enabled = True
    try:
        codes = []
        for _ in range(7):
            codes.append(client.post("/operator/runtime/quantization",
                                     json={"quantization_bits": 4}).status_code)
        check("a burst eventually gets 429", 429 in codes, f"codes={codes}")
    finally:
        _main.limiter.enabled = False

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} of {CHECKS} checks FAILED:")
        for f in FAILURES:
            print(f"   - {f}")
        return 1
    print(f"ALL {CHECKS} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
