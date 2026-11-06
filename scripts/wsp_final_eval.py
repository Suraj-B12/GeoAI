"""
The ONE held-out evaluation of the weight-space programming study (paper §6.11).

Refuses to run unless the final edit is frozen in
eval_results/finetune/wsp_final_edit/manifest.json (written by the editor when
editing is finished), and refuses a second run unless --rerun-reason is given
(any rerun is logged and reported). Runs in its own process with the production
model (PavementClassifier -> scripts/model_loader.py, 4-bit NF4); the forward-pass
server must be stopped (both need C:\\GeoAI\\gpu.lock).

Steps (all results checkpointed atomically to C:\\GeoAI\\wsp\\final\\final_raw.json;
re-running the same command resumes):
  1. bake check: Stage 1 logits on 20 practice frames, hook version of the final
     edit vs the baked bf16 weights -> max abs difference; time per image and
     extra VRAM of the baked layers
  2. TEST split (547 frames), Stage 1: base, baked final edit, baked random control
  3. Bengaluru (300 video segments), Stage 1: base, final, random
  4. DEV regression: Stage 2 free-form on 30 dev frames and per-type probe on all
     300 dev frames, base vs final
Scoring: scripts/wsp_report.py (no inference).
"""

from __future__ import annotations

import argparse
import gc
import os
import random
import sys
import time
import uuid
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

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from scripts import wsp_core as W  # noqa: E402
from scripts.ft_common import GpuLock, atomic_write_json, read_json, utc_now  # noqa: E402
from scripts.wsp_apply_edit import bake_into, restore  # noqa: E402

torch.set_grad_enabled(False)
OUT = W.WSP_DIR / "final" / "final_raw.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun-reason", default=None)
    a = ap.parse_args()
    man = read_json(W.FINAL_DIR / "manifest.json")
    if not man:
        sys.exit("final edit not frozen (eval_results/finetune/wsp_final_edit/manifest.json missing)")
    raw = read_json(OUT) or {}
    if raw.get("complete") and not a.rerun_reason:
        sys.exit(f"{OUT} is complete; a second test run needs --rerun-reason (it will be reported)")
    if raw.get("complete") and a.rerun_reason:
        raw = {"previous_runs": raw.get("previous_runs", []) + [{"run_id": raw["run_id"]}],
               "rerun_reason": a.rerun_reason}
    raw.setdefault("run_id", f"final_{time.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:4]}")
    raw.setdefault("started", utc_now())
    raw["manifest"] = man
    for k in ("configs", "bengaluru", "stage2_dev", "probe_dev"):
        raw.setdefault(k, {})

    P = W.pools()
    test = sorted(P["test"], key=lambda r: r["key"])
    beng = P["bengaluru"]
    dev = sorted(P["dev"], key=lambda r: r["key"])
    rng = random.Random(20261106)
    dev30 = sorted(rng.sample(dev, 30), key=lambda r: r["key"])
    prac20 = sorted(P["practice"], key=lambda r: r["key"])[:20]
    final_dir = Path(man["baked"]["final"])
    random_dir = Path(man["baked"]["random"])

    def save(complete=False):
        raw["complete"] = complete
        raw["updated"] = utc_now()
        atomic_write_json(OUT, raw, indent=None)

    with GpuLock("wsp FINAL held-out evaluation"):
        from app.model import PavementClassifier
        from app import model as appmodel
        from scripts.stage2_probe import TypeProber, all_probe_types
        clf = PavementClassifier(model_path="Qwen/Qwen2.5-VL-7B-Instruct", adapter_path=None, quantization_bits=4)
        model = clf._model
        n_id, d_id = clf._normal_token_ids[0], clf._distress_token_ids[0]
        s1s, s1u = appmodel.STAGE1_SYSTEM_PROMPT, appmodel.STAGE1_USER_PROMPT
        prober = TypeProber(model, clf._processor, all_probe_types(include_indicators=True), "min", "name")

        def read(items):
            out = {}
            for it in items:
                img = Image.open(it["path"]).convert("RGB")
                out[it["key"]] = W.stage1_read(clf, model, img, n_id, d_id, s1s, s1u)
                img.close()
                gc.collect()
                torch.cuda.empty_cache()
            return out

        def ledger(kind, items, cfg, pools):
            W.append_ledger({"pass_id": f"{raw['run_id']}_{kind}_{cfg}", "at": utc_now(), "kind": kind,
                             "n": len(items), "pools": pools, "edits": cfg, "final": True,
                             "run_id": raw["run_id"], "counted": False,
                             "purpose": f"FINAL evaluation ({cfg}) - {kind}"})

        # 1. bake check + cost
        if "bake_check" not in raw:
            edits = W.load_edit_group(man["final_edits"])
            h = W.apply_edits(model, edits)
            try:
                t0 = time.time()
                hook = read(prac20)
                t_hook = (time.time() - t0) / len(prac20)
            finally:
                for x in h:
                    x.remove()
            t0 = time.time()
            base20 = read(prac20)
            t_base = (time.time() - t0) / len(prac20)
            torch.cuda.synchronize()
            m0 = torch.cuda.memory_allocated()
            orig = bake_into(model, final_dir)
            torch.cuda.synchronize()
            m1 = torch.cuda.memory_allocated()
            try:
                t0 = time.time()
                baked = read(prac20)
                t_baked = (time.time() - t0) / len(prac20)
            finally:
                restore(model, orig)
            diffs = [abs(hook[k][f] - baked[k][f]) for k in hook for f in ("logit_normal", "logit_distress")]
            raw["bake_check"] = {"n": len(prac20), "max_abs_diff": round(max(diffs), 5),
                                 "mean_abs_diff": round(sum(diffs) / len(diffs), 5),
                                 "max_abs_p_distress_diff": round(max(abs(hook[k]["p_distress"] - baked[k]["p_distress"])
                                                                      for k in hook), 6),
                                 "same_argmax": sum((hook[k]["logit_distress"] > hook[k]["logit_normal"]) ==
                                                    (baked[k]["logit_distress"] > baked[k]["logit_normal"]) for k in hook)}
            raw["cost"] = {"extra_vram_mb_after_bake": round((m1 - m0) / 2 ** 20, 1),
                           "baked_modules": list(orig.keys()),
                           "s_per_image_stage1_base_nf4": round(t_base, 3),
                           "s_per_image_stage1_hooks": round(t_hook, 3),
                           "s_per_image_stage1_baked": round(t_baked, 3),
                           "note": "20 practice frames each, one process, cache cleared between images"}
            print(f"[bake] {raw['bake_check']} {raw['cost']}", flush=True)
            save()

        # 2-3. test + Bengaluru, Stage 1
        for cfg, bdir in (("base", None), ("final", final_dir), ("random", random_dir)):
            orig = bake_into(model, bdir) if bdir else {}
            try:
                for kind, items, store in (("stage1_test", test, raw["configs"]),
                                           ("stage1_bengaluru", beng, raw["bengaluru"])):
                    if cfg in store and len(store[cfg]) == len(items):
                        continue
                    t0 = time.time()
                    store[cfg] = read(items)
                    ledger(kind, items, cfg, ["test"] if kind == "stage1_test" else ["bengaluru"])
                    print(f"[{kind}] {cfg}: {len(items)} frames in {time.time() - t0:.0f} s", flush=True)
                    save()
            finally:
                restore(model, orig)

        # 4. dev regression: Stage 2 parse (30 frames) and probe (300 frames), base vs final
        for cfg, bdir in (("base", None), ("final", final_dir)):
            orig = bake_into(model, bdir) if bdir else {}
            try:
                if cfg not in raw["stage2_dev"]:
                    res = {}
                    for it in dev30:
                        img = Image.open(it["path"]).convert("RGB")
                        s2 = clf.predict_stage2(img, record_views=False)
                        res[it["key"]] = {"distress_types": s2.get("distress_types"), "severity": s2.get("severity"),
                                          "raw": (s2.get("stage2_raw") or "")[:1500]}
                        img.close()
                        gc.collect()
                        torch.cuda.empty_cache()
                    raw["stage2_dev"][cfg] = res
                    ledger("stage2_dev", dev30, cfg, ["dev"])
                    save()
                if cfg not in raw["probe_dev"]:
                    res = {}
                    for it in dev:
                        img = Image.open(it["path"]).convert("RGB")
                        with clf._lock:
                            r = prober.probe(img)
                        res[it["key"]] = {x["key"]: round(x["p_yes"], 6) for x in r}
                        img.close()
                        gc.collect()
                        torch.cuda.empty_cache()
                    raw["probe_dev"][cfg] = res
                    ledger("probe_dev", dev, cfg, ["dev"])
                    save()
            finally:
                restore(model, orig)
        raw["finished"] = utc_now()
        save(complete=True)
        print(f"[done] {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
