"""
Weight-space programming (WSP) forward-pass server (paper §6.11).

A plain Python HTTP server on 127.0.0.1:8011 (NOT uvicorn app.main) that loads
the production model ONCE - PavementClassifier, 4-bit NF4, 1024x1024 cap, v2
prompts - and serves forward "passes" to the editor. It is the editor's only
way to run the model. Every pass is appended to eval_results/finetune/
wsp_ledger.jsonl and counted against the pre-registered budget (wsp_core.BUDGET).

Pass kinds (each request = one pass):
  stage1  up to 256 images -> Stage 1 first-token logits (production prompt;
          P(Distress) = softmax over the Normal/Distress first sub-tokens, as
          PavementClassifier._compute_stage1_confidence), optional captures of
          the residual stream / MLP output / attention output at the decision
          position for requested layers, saved to C:\\GeoAI\\wsp\\captures\\<pass>.npz
  stage2  up to 64 images -> production free-form Stage 2 generations
  probe   up to 256 images -> per-type probe P(yes) (production variant min:name)
Each runs with or without named edits (wsp_core hooks on the untouched NF4
modules). Not passes (free): /status, /weight (dequantized weight of one
editable module to CPU, for reading), the start-up hook self-test.

Guards: inference_mode everywhere; no optimizer; test frames and Bengaluru
frames are refused unless the request is a FINAL evaluation (allowed only after
the final edit is frozen in eval_results/finetune/wsp_final_edit/manifest.json),
and every final request is logged with its run id.

Takes C:\\GeoAI\\gpu.lock only while it loads/self-tests and while each pass or
weight read computes, then releases it (shared-card rule); the model stays resident.

Start detached:
    venv/Scripts/python.exe scripts/wsp_server.py
"""

from __future__ import annotations

import gc
import json
import os
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["PROMPTS_VERSION"] = "v2"
os.environ["DISABLE_ADAPTER"] = "true"
os.environ["STAGE2_CONFIDENCE_MODE"] = "field"
os.environ["STAGE2_MODE"] = "generate"
os.environ["STAGE1_SAFETY_NET"] = "false"
os.environ["MAX_IMAGE_PIXELS"] = "1048576"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512,garbage_collection_threshold:0.9")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from scripts import wsp_core as W  # noqa: E402
from scripts.ft_common import GpuLock, atomic_write_json, read_json, utc_now  # noqa: E402

torch.set_grad_enabled(False)


class Engine:
    def __init__(self):
        from app.model import PavementClassifier
        from scripts.stage2_probe import TypeProber, all_probe_types
        from scripts.utils_v2_prompts import STAGE1_SYSTEM_PROMPT_V2, STAGE1_USER_PROMPT_V2
        from app import model as appmodel
        self.clf = PavementClassifier(model_path="Qwen/Qwen2.5-VL-7B-Instruct", adapter_path=None,
                                      quantization_bits=4)
        self.model = self.clf._model
        self.s1_sys = appmodel.STAGE1_SYSTEM_PROMPT
        self.s1_user = appmodel.STAGE1_USER_PROMPT
        assert self.s1_sys == STAGE1_SYSTEM_PROMPT_V2 and self.s1_user == STAGE1_USER_PROMPT_V2
        self.n_id = self.clf._normal_token_ids[0]
        self.d_id = self.clf._distress_token_ids[0]
        self.layers = self.model.model.language_model.layers
        self.prober = TypeProber(self.model, self.clf._processor,
                                 all_probe_types(include_indicators=True), "min", "name")
        self.pools = W.pools()
        self.test_keys = {r["key"] for r in self.pools["test"]}
        self.by_key = {r["key"]: r for p in self.pools.values() for r in p}
        self.selftest = self.hook_selftest()
        print(f"[wsp] hook self-test: {self.selftest}", flush=True)
        if not self.selftest["ok"]:
            raise RuntimeError("edit hooks do not reproduce the dense edited weight")

    # ---------------- weights ----------------

    def module(self, name: str):
        W.check_module_name(name)
        mods = dict(self.model.named_modules())
        return mods[name]

    def dequant(self, name: str) -> torch.Tensor:
        import bitsandbytes.functional as bnbf
        m = self.module(name)
        w = m.weight
        if hasattr(w, "quant_state") and w.quant_state is not None:
            return bnbf.dequantize_4bit(w.data, w.quant_state).float()
        return w.float()

    def hook_selftest(self) -> dict:
        """Hooked NF4 module == x @ W'^T with W' the dense edited (dequantized) weight."""
        out = {}
        g = torch.Generator().manual_seed(0)
        for name, kind in (("model.language_model.layers.19.mlp.down_proj", "rank1_out"),
                           ("model.language_model.layers.10.self_attn.o_proj", "columns")):
            Wd = self.dequant(name)
            mod = self.module(name)
            x = torch.randn(4, Wd.shape[1], generator=g).to(torch.bfloat16).to(Wd.device)
            if kind == "rank1_out":
                e = {"name": "selftest", "kind": kind, "module": name, "explanation": "self-test",
                     "u": torch.randn(Wd.shape[0], generator=g), "c": 1.3}
            else:
                S = torch.arange(128, 256)
                e = {"name": "selftest", "kind": kind, "module": name, "explanation": "self-test",
                     "cols": S, "D": Wd[:, S].cpu() * 0.7}
            plain = mod(x).float()
            h = W.apply_edits(self.model, [e])
            try:
                hooked = mod(x).float()
            finally:
                for hh in h:
                    hh.remove()
            dense_plain = x.float() @ Wd.T
            dense_edit = x.float() @ W.dense_edited_weight(Wd.cpu(), e).to(Wd.device).T
            scale = dense_edit.abs().mean().item()
            out[name] = {"kind": kind,
                         "rel_err_hooked_vs_dense_edit": round((hooked - dense_edit).abs().mean().item() / scale, 5),
                         "rel_err_plain_vs_dense_plain": round((plain - dense_plain).abs().mean().item() / scale, 5)}
        ok = all(v["rel_err_hooked_vs_dense_edit"] <= v["rel_err_plain_vs_dense_plain"] + 2e-3
                 for v in out.values())
        return {"ok": ok, "modules": out}

    # ---------------- passes ----------------

    def _items(self, keys, final: bool):
        items = []
        for k in keys:
            r = self.by_key.get(k)
            if r is None:
                raise W.RuleViolation(f"unknown item {k!r}")
            if r["pool"] in ("test", "bengaluru") and not final:
                raise W.RuleViolation(f"{k} is in the {r['pool']} pool: only a FINAL evaluation may use it")
            items.append(r)
        return items

    def _captures(self, spec):
        """Hooks that keep the last-position output of requested sites."""
        if not spec:
            return {}, []
        store = {}
        hs = []
        sites = spec.get("sites", ["resid"])
        layers = spec.get("layers", list(range(len(self.layers))))
        for li in layers:
            L = self.layers[li]
            for s in sites:
                key = f"{s}_{li}"
                if s in ("attn_in", "mlp_in"):
                    # INPUT of o_proj (per-head outputs, heads x 128) or of down_proj
                    # (MLP neuron activations): what a column edit acts on
                    mod = L.self_attn.o_proj if s == "attn_in" else L.mlp.down_proj

                    def hook(m, i, o, key=key):
                        store.setdefault(key, []).append(i[0][0, -1].float().cpu().numpy().astype(np.float16))
                else:
                    mod = {"resid": L, "mlp": L.mlp, "attn": L.self_attn}[s]

                    def hook(m, i, o, key=key):
                        t = o[0] if isinstance(o, tuple) else o
                        store.setdefault(key, []).append(t[0, -1].float().cpu().numpy().astype(np.float16))
                hs.append(mod.register_forward_hook(hook))
        return store, hs

    def stage1(self, items, capture):
        store, hs = self._captures(capture)
        res = []
        try:
            for it in items:
                img = Image.open(it["path"]).convert("RGB")
                r = W.stage1_read(self.clf, self.model, img, self.n_id, self.d_id, self.s1_sys, self.s1_user)
                res.append({"key": it["key"], **r})
                img.close()
                gc.collect()
                torch.cuda.empty_cache()
        finally:
            for h in hs:
                h.remove()
        return res, store

    def stage2(self, items):
        res = []
        for it in items:
            img = Image.open(it["path"]).convert("RGB")
            s2 = self.clf.predict_stage2(img, record_views=False)
            raw = s2.get("stage2_raw") or ""
            res.append({"key": it["key"], "distress_types": s2.get("distress_types"),
                        "severity": s2.get("severity"), "description": (s2.get("description") or "")[:300],
                        "stage2_confidence": s2.get("stage2_confidence"), "raw": raw[:1500]})
            img.close()
            gc.collect()
            torch.cuda.empty_cache()
        return res

    def probe(self, items):
        res = []
        for it in items:
            img = Image.open(it["path"]).convert("RGB")
            with self.clf._lock:
                r = self.prober.probe(img)
            res.append({"key": it["key"], "p": {x["key"]: round(x["p_yes"], 6) for x in r},
                        "min_mass": round(min(x["yes_no_mass"] for x in r), 5)})
            img.close()
            gc.collect()
            torch.cuda.empty_cache()
        return res

    def run_pass(self, req: dict) -> dict:
        kind = req.get("kind")
        final = bool(req.get("final"))
        purpose = (req.get("purpose") or "").strip()
        if len(purpose) < 10:
            raise W.RuleViolation("every pass needs a purpose (>= 10 characters)")
        if final:
            if not (W.FINAL_DIR / "manifest.json").exists():
                raise W.RuleViolation("final evaluation refused: the final edit is not frozen")
            if not req.get("run_id"):
                raise W.RuleViolation("final evaluation needs a run_id")
        else:
            if W.passes_used() >= W.BUDGET:
                raise W.RuleViolation(f"budget exhausted ({W.BUDGET} passes)")
        keys = req.get("items") or []
        cap = W.MAX_GENERATIONS_PER_PASS if kind == "stage2" else W.MAX_IMAGES_PER_PASS
        if not keys or len(keys) > cap:
            raise W.RuleViolation(f"a {kind} pass takes 1..{cap} images, got {len(keys)}")
        items = self._items(keys, final)
        edit_names = req.get("edits") or []
        edits = W.load_edit_group(edit_names)
        if kind not in ("stage1", "stage2", "probe"):
            raise W.RuleViolation(f"unknown pass kind {kind!r}")
        pass_id = f"{time.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}"
        # The GPU lock is held only while this pass computes (shared-card rule).
        with GpuLock(f"wsp server pass {pass_id} ({kind}, n={len(items)}): {purpose[:120]}"):
            t0 = time.time()
            handles = W.apply_edits(self.model, edits)
            try:
                if kind == "stage1":
                    res, store = self.stage1(items, req.get("capture"))
                elif kind == "stage2":
                    res, store = self.stage2(items), {}
                else:
                    res, store = self.probe(items), {}
            finally:
                for h in handles:
                    h.remove()
                gc.collect()
                torch.cuda.empty_cache()
        cap_path = None
        if store:
            W.CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
            cap_path = W.CAPTURE_DIR / f"{pass_id}.npz"
            tmp = W.CAPTURE_DIR / f"{pass_id}.tmp.npz"
            np.savez(tmp, keys=np.array([it["key"] for it in items]),
                     **{k: np.stack(v) for k, v in store.items()})
            os.replace(tmp, cap_path)
        pools = sorted({it["pool"] for it in items})
        rec = {"pass_id": pass_id, "at": utc_now(), "kind": kind, "n": len(items), "pools": pools,
               "edits": edit_names, "capture": req.get("capture"), "purpose": purpose,
               "final": final, "run_id": req.get("run_id"), "counted": not final,
               "seconds": round(time.time() - t0, 1)}
        W.append_ledger(rec)
        rec["budget_used"] = W.passes_used()
        print(f"[pass] {pass_id} {kind} n={len(items)} edits={edit_names} final={final} "
              f"{rec['seconds']}s used={rec['budget_used']}/{W.BUDGET} :: {purpose[:80]}", flush=True)
        return {"ledger": rec, "results": res, "capture_file": str(cap_path) if cap_path else None}


ENGINE: Engine | None = None
LOCK = threading.Lock()
STOP = threading.Event()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/status":
            self._send(200, {"budget": W.BUDGET, "used": W.passes_used(), "pid": os.getpid(),
                             "selftest": ENGINE.selftest if ENGINE else None})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._send(400, {"error": "bad json"})
        try:
            with LOCK:
                if self.path == "/pass":
                    return self._send(200, ENGINE.run_pass(req))
                if self.path == "/weight":
                    name = req["module"]
                    W.check_module_name(name)
                    dst = W.WSP_DIR / "weights" / f"{name}.pt"
                    if not dst.exists():
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        with GpuLock(f"wsp server weight read {name}", quiet=True):
                            t = ENGINE.dequant(name).cpu().to(torch.float16)
                            torch.cuda.empty_cache()
                        torch.save(t, dst.with_suffix(".pt.tmp"))
                        os.replace(dst.with_suffix(".pt.tmp"), dst)
                    return self._send(200, {"path": str(dst)})
                if self.path == "/shutdown":
                    STOP.set()
                    return self._send(200, {"ok": True})
            self._send(404, {"error": "not found"})
        except W.RuleViolation as e:
            self._send(403, {"error": f"rule: {e}"})
        except Exception as e:
            traceback.print_exc()
            self._send(500, {"error": f"{type(e).__name__}: {e}"})


def main() -> int:
    global ENGINE
    # Loading + self-tests use the GPU, so they run under the lock; afterwards the
    # model stays resident (one NF4 copy) and the lock is taken per pass only.
    with GpuLock("wsp server: model load + self-tests"):
        ENGINE = Engine()
        gc.collect()
        torch.cuda.empty_cache()
    srv = HTTPServer(("127.0.0.1", W.PORT), Handler)
    srv.timeout = 1.0
    atomic_write_json(W.WSP_DIR / "server.json", {"pid": os.getpid(), "port": W.PORT,
                                                  "started": utc_now(), "selftest": ENGINE.selftest,
                                                  "lock": "taken per pass, released after"})
    print(f"[wsp] serving on 127.0.0.1:{W.PORT} (pid {os.getpid()}); budget used "
          f"{W.passes_used()}/{W.BUDGET}; gpu.lock taken per pass", flush=True)
    try:
        while not STOP.is_set():
            srv.handle_request()
    finally:
        srv.server_close()
        print("[wsp] shutdown", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
