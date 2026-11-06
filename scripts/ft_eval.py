"""
Evaluation harness of the post-training study (paper §6.11). Records model
outputs only; scoring lives in scripts/ft_report.py (like the probe study's
experiment/report split), so no number is computed while a job runs.

The model is PavementClassifier with production settings (4-bit, 1024x1024 cap,
v2 prompts, field confidence; STAGE2_MODE=generate and no safety net so
predict_stage2 returns only the free-form answer). Adapters are attached with
peft as named adapters; the base model is the SAME object with the adapter
disabled (PeftModel.disable_adapter), so base and tuned differ only in the LoRA
weights. The probe is the unmodified production TypeProber (variant min:name).

Per image and model it records:
  probe   P(yes) for all 20 questions (+ min yes/no mass, time)
  stage1  production Stage 1 label/confidence/raw
  stage2  free-form Stage 2 (types, severity, description, raw) for the format check

Crash-safe: holds C:\\GeoAI\\gpu.lock, writes its results atomically every
--save-every images, refuses to resume a file made under another configuration,
and resumes when re-run with the same arguments.

Usage:
    venv/Scripts/python.exe scripts/ft_eval.py --name base_dev --models base \
        --items attain_dev --what probe
    venv/Scripts/python.exe scripts/ft_eval.py --name A1_dev \
        --models A1e1=C:/GeoAI/finetune/runs/A1/epoch_1,A1e2=C:/GeoAI/finetune/runs/A1/epoch_2 \
        --items attain_dev --what probe
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Production settings, assigned before app.model is imported (it reads them at import).
os.environ["PROMPTS_VERSION"] = "v2"
os.environ["DISABLE_ADAPTER"] = "true"
os.environ["STAGE2_CONFIDENCE_MODE"] = "field"
os.environ["STAGE2_MODE"] = "generate"
os.environ["STAGE1_SAFETY_NET"] = "false"
os.environ["MAX_IMAGE_PIXELS"] = "1048576"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512,garbage_collection_threshold:0.9")

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from scripts.ft_common import (ART_DIR, FT_DIR, GpuLock, atomic_write_json,  # noqa: E402
                               load_v2_rows, read_json, sha256_file, utc_now)
from scripts.stage2_probe import TypeProber, all_probe_types  # noqa: E402

VARIANT = ("min", "name")
RAW_DIR = ART_DIR / "raw"
FIELDS_RE = {f: re.compile(rf"^\s*\**\s*{f}\s*\**\s*:", re.I | re.M)
             for f in ("DISTRESS_TYPES", "SEVERITY", "DESCRIPTION")}


def load_items(which: str, stage2_limit: int) -> list[dict]:
    if which.startswith("attain"):
        rows = load_v2_rows()
        part = which.split("_", 1)[1]
        pick = [r for r in rows if part == "all" or r["split"] == part]
        items = [{"key": r["image"], "path": r["image_path"], "gt": r["gt"], "split": r["split"],
                  "block": r["block"]} for r in pick]
    elif which == "bengaluru":
        b = read_json(FT_DIR / "bengaluru_items.json")
        if not b:
            sys.exit("eval_results/finetune/bengaluru_items.json missing - run ft_prepare.py first")
        items = [{"key": x["key"], "path": x["path"], "kind": x["kind"]} for x in b["items"]]
    else:
        sys.exit(f"unknown --items {which}")
    # deterministic Stage 2 format sample (every image gets the flag, sample by seed)
    rng = random.Random(20261106)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    s2 = set(idx[:stage2_limit]) if stage2_limit >= 0 else set(idx)
    for i, it in enumerate(items):
        it["stage2_sample"] = i in s2
    return items


def worker_idle() -> bool:
    """True unless the production worker is classifying right now."""
    try:
        import httpx
        kf = Path(r"C:\GeoAI\operator_api_key.txt")
        hdr = {"X-API-Key": kf.read_text(encoding="utf-8").strip()} if kf.exists() else {}
        r = httpx.get("http://127.0.0.1:8000/operator/status", headers=hdr, timeout=5)
        if r.status_code != 200:
            return True
        return not r.json().get("current_image_id")
    except Exception:
        return True


def adapter_sha(d: str) -> str:
    p = Path(d)
    f = p / "adapter_model.safetensors"
    return sha256_file(f)[:16] if f.exists() else "missing"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--models", required=True, help="'base' and/or name=adapter_dir, comma-separated")
    ap.add_argument("--items", required=True, help="attain_dev | attain_test | attain_all | bengaluru")
    ap.add_argument("--what", default="probe", help="comma list of probe,stage1,stage2")
    ap.add_argument("--stage2-limit", type=int, default=40, help="images in the Stage 2 format sample (-1 = all)")
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    what = [w for w in a.what.split(",") if w]
    models = []
    for m in a.models.split(","):
        if m == "base":
            models.append(("base", None))
        else:
            n, d = m.split("=", 1)
            models.append((n, d))
    items = load_items(a.items, a.stage2_limit)
    if a.limit:
        items = items[: a.limit]
    cfg = {"models": [[n, d, adapter_sha(d) if d else None] for n, d in models],
           "items": a.items, "n_items": len(items), "what": what, "stage2_limit": a.stage2_limit,
           "variant": list(VARIANT), "max_pixels": 1048576,
           "code_sha256": {f: sha256_file(PROJECT_ROOT / f)[:16] for f in
                           ("scripts/ft_eval.py", "scripts/stage2_probe.py", "app/model.py")}}
    fp = hashlib.sha256(json.dumps({k: cfg[k] for k in cfg if k != "code_sha256"},
                                   sort_keys=True).encode()).hexdigest()[:16]
    out = RAW_DIR / f"{a.name}.json"
    blob = read_json(out)
    if blob and blob.get("fingerprint") != fp:
        sys.exit(f"ERROR: {out} was made under another configuration ({blob.get('fingerprint')} != {fp})")
    rows = blob["rows"] if blob else {}
    if blob and blob.get("complete"):
        print(f"[done] {out} already complete")
        return 0

    with GpuLock(f"eval {a.name}"):
        while not worker_idle():
            print("[wait] production worker is classifying; waiting 30 s", flush=True)
            time.sleep(30)
        from app.model import PavementClassifier
        clf = PavementClassifier(model_path="Qwen/Qwen2.5-VL-7B-Instruct", adapter_path=None,
                                 quantization_bits=4)
        named = [(n, d) for n, d in models if d]
        if named:
            from peft import PeftModel
            pm = PeftModel.from_pretrained(clf._model, named[0][1], adapter_name=named[0][0])
            for n, d in named[1:]:
                pm.load_adapter(d, adapter_name=n)
            pm.eval()
            clf._model = pm          # adapter_path stays None: no Stage 2 adapter cascade
        prober = TypeProber(clf._model, clf._processor, all_probe_types(include_indicators=True), *VARIANT)

        from contextlib import nullcontext

        def ctx(name):
            if not named:
                return nullcontext()
            if name == "base":
                return clf._model.disable_adapter()
            clf._model.set_adapter(name)
            return nullcontext()

        # parity of the fast probe path, per model, on the first image
        parity = blob.get("parity", {}) if blob else {}
        img0 = Image.open(items[0]["path"]).convert("RGB")
        for n, _d in models:
            if n not in parity:
                with ctx(n), clf._lock:
                    parity[n] = prober.parity_check(img0)
                print(f"[parity] {n}: {parity[n]}", flush=True)
                if not parity[n]["ok"]:
                    sys.exit(f"ERROR: fast/slow probe paths disagree for {n}")
        img0.close()

        def save(complete=False):
            atomic_write_json(out, {"fingerprint": fp, "config": cfg, "parity": parity,
                                    "updated_at": utc_now(), "complete": complete, "rows": rows}, indent=None)

        t0 = time.time()
        n_new = 0
        for i, it in enumerate(items, 1):
            rec = rows.get(it["key"]) or {k: it[k] for k in it if k != "path"}
            rec.setdefault("models", {})
            todo = [(n, d) for n, d in models if not all(
                w in rec["models"].get(n, {}) for w in what
                if not (w == "stage2" and not it["stage2_sample"]))]
            if not todo:
                continue
            try:
                img = Image.open(it["path"]).convert("RGB")
                rec["size"] = list(img.size)
                for n, _d in todo:
                    r = rec["models"].setdefault(n, {})
                    with ctx(n):
                        if "probe" in what and "probe" not in r:
                            ts = time.time()
                            with clf._lock:
                                res = prober.probe(img)
                            r["probe"] = {"p": {x["key"]: round(x["p_yes"], 6) for x in res},
                                          "min_mass": round(min(x["yes_no_mass"] for x in res), 5),
                                          "ms": round((time.time() - ts) * 1000)}
                        if "stage1" in what and "stage1" not in r:
                            s1 = clf.predict_stage1(img)
                            r["stage1"] = {k: s1.get(k) for k in
                                           ("stage1_label", "stage1_confidence", "stage1_raw")}
                        if "stage2" in what and it["stage2_sample"] and "stage2" not in r:
                            s2 = clf.predict_stage2(img, record_views=False)
                            raw = s2.get("stage2_raw") or ""
                            r["stage2"] = {"distress_types": s2.get("distress_types"),
                                           "severity": s2.get("severity"),
                                           "description": (s2.get("description") or "")[:300],
                                           "stage2_confidence": s2.get("stage2_confidence"),
                                           "fields_present": {f: bool(rx.search(raw))
                                                              for f, rx in FIELDS_RE.items()},
                                           "raw": raw[:1500]}
                    gc.collect()
                    torch.cuda.empty_cache()
                img.close()
            except Exception as ex:
                rec["error"] = f"{type(ex).__name__}: {ex}"
                print(f"[{i}/{len(items)}] ERROR {it['key']}: {rec['error']}", flush=True)
            rows[it["key"]] = rec
            n_new += 1
            if n_new % a.save_every == 0:
                save()
                el = time.time() - t0
                print(f"[{i}/{len(items)}] {el / n_new:.1f} s/img ({len(todo)} models)", flush=True)
        save(complete=True)
        print(f"[done] {len(rows)} rows -> {out} ({time.time() - t0:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
