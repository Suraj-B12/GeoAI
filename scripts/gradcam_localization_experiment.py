"""
GPU sandbox: Grad-CAM maps of the per-type probe on every Attain WS_V2.0 frame.

For each frame and each distress class annotated in it, the probe question(s)
for that class are explained with scripts/stage2_gradcam.ProbeGradCAM (Linear
crack: both the Longitudinal and the Transverse question, as in
stage2_probe_report.PROBE_KEYS). Maps for all candidate (method, layer)
variants are stored; nothing is scored here. Scoring, the dev-only variant
choice and the test comparison against a location prior live in
scripts/gradcam_localization_report.py, following
eval_results/gradcam_localization_preregistration.json.

Production settings: Qwen2.5-VL-7B base, 4-bit NF4, 1024^2 cap, probe variant
from configs/stage2_probe.json - the same model and question production asks.

Sharing the GPU with the live server: before every frame the script asks the
server whether it is classifying (citizen worker or video pipeline) and waits
while it is, so the two never run inference at the same time for longer than
one frame. Unreachable server = not running = go ahead.

Crash-safe: per-frame maps go to eval_results/gradcam_attain/<frame>.npz, the
checkpoint (meta per frame) is rewritten after every frame and refuses to
resume under a different configuration.

Usage:
    venv/Scripts/python.exe scripts/gradcam_localization_experiment.py --limit 3
    venv/Scripts/python.exe scripts/gradcam_localization_experiment.py
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["MAX_IMAGE_PIXELS"] = "1048576"

import importlib.util  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from scripts import stage2_probe_report as R  # noqa: E402
from scripts import stage2_probe_rules as rules  # noqa: E402
from scripts.model_loader import load_vlm  # noqa: E402
from scripts.stage2_gradcam import DEFAULT_LAYERS, ProbeGradCAM  # noqa: E402
from scripts.stage2_probe import all_probe_types  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "attain_eval", PROJECT_ROOT / "scripts" / "07_cross_dataset_eval.py")
_ae = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ae)

EVAL_DIR = PROJECT_ROOT / "eval_results"
MAP_DIR = EVAL_DIR / "gradcam_attain"
CKPT = EVAL_DIR / "gradcam_attain_checkpoint.json"
OUT = EVAL_DIR / "gradcam_attain_raw.json"
KEY_FILE = Path(r"C:\GeoAI\operator_api_key.txt")


def release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def gt_boxes_by_class(image_path: str) -> dict:
    """{Attain class: [[x1, y1, x2, y2], ...]} for distress classes only."""
    lab = Path(image_path.replace("Images", "Labels")).with_suffix(".xml")
    out: dict = {}
    if not lab.exists():
        return out
    for o in ET.parse(lab).findall(".//object"):
        name = o.findtext("name") or ""
        cls, _sev = _ae.parse_attain_class(name)
        m = _ae.ATTAIN_CLASS_MAP.get(cls)
        if m is None or m.get("label") == "EXCLUDE" or cls not in R.PROBE_KEYS:
            continue
        b = o.find("bndbox")
        out.setdefault(cls, []).append([float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")])
    return out


def server_busy(url: str) -> str | None:
    """Why the live server is using the GPU right now, or None."""
    try:
        key = KEY_FILE.read_text(encoding="utf-8").strip() if KEY_FILE.exists() else ""
        hdr = {"X-API-Key": key} if key else {}

        def get(path):
            req = urllib.request.Request(url + path, headers=hdr)
            with urllib.request.urlopen(req, timeout=5) as r:
                return json.loads(r.read().decode())
        w = get("/operator/status")
        if w.get("current_image_id"):
            return "citizen worker is classifying"
        v = get("/operator/video/status")
        c = v.get("counts") or {}
        if v.get("busy") or c.get("extracting") or c.get("classifying"):
            return "video pipeline is processing"
    except Exception:
        return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--no-yield", action="store_true", help="do not wait for the live server")
    args = ap.parse_args()

    cfg = rules.load_config(PROJECT_ROOT / "configs" / "stage2_probe.json")
    variant = cfg["variant"]
    code_sha = hashlib.sha256((PROJECT_ROOT / "scripts" / "stage2_gradcam.py").read_bytes()).hexdigest()[:12]
    setup = {"model": "Qwen/Qwen2.5-VL-7B-Instruct", "quantization_bits": 4,
             "max_pixels": 1048576, "variant": variant, "layers": list(DEFAULT_LAYERS),
             "target": "log-odds yes vs no at the answer position", "gradcam_code_sha": code_sha}
    fp = hashlib.sha256(json.dumps(setup, sort_keys=True).encode()).hexdigest()[:16]

    d = json.loads((EVAL_DIR / "stage2_probe_raw_attain.json").read_text(encoding="utf-8"))
    rows = [r for r in d["rows"] if "error" not in r]
    R.split(rows)
    items = []
    for r in rows:
        boxes = gt_boxes_by_class(r["image_path"])
        if boxes:
            items.append({"image": r["image"], "image_path": r["image_path"], "index": r["index"],
                          "split": r["split"], "boxes": boxes,
                          "probe_p": r["probe"].get(f"{variant['system_style']}:{variant['question_style']}", {}).get("p", {})})
    if args.limit:
        items = items[: args.limit]
    n_pass = sum(len(R.PROBE_KEYS[c]) for it in items for c in it["boxes"])
    print(f"[data] {len(items)} frames with distress boxes, {n_pass} explanations; config {fp}", flush=True)

    done = {}
    if CKPT.exists():
        blob = json.loads(CKPT.read_text(encoding="utf-8"))
        if blob.get("_fp") != fp:
            sys.exit(f"{CKPT.name} was made under another configuration; delete it to start over")
        done = blob["rows"]
        print(f"[resume] {len(done)} frames done", flush=True)
    MAP_DIR.mkdir(parents=True, exist_ok=True)

    model, proc, info = load_vlm(setup["model"], quantization_bits=4, min_pixels=256 * 28,
                                 max_pixels=setup["max_pixels"], run_self_test=True)
    cam = ProbeGradCAM(model, proc, all_probe_types(), variant["system_style"],
                       variant["question_style"], DEFAULT_LAYERS)

    t_run, n_new = time.time(), 0
    for i, it in enumerate(items, 1):
        if it["image"] in done:
            continue
        if not args.no_yield:
            waited = 0
            while (why := server_busy(args.server)):
                if waited == 0:
                    print(f"[wait] {why}; pausing", flush=True)
                time.sleep(20)
                waited += 20
            if waited:
                print(f"[wait] resumed after {waited} s", flush=True)
        rec = {k: it[k] for k in ("image", "image_path", "index", "split", "boxes")}
        try:
            img = Image.open(it["image_path"]).convert("RGB")
            rec["size"] = list(img.size)
            arrays, keys = {}, {}
            t0 = time.time()
            torch.cuda.reset_peak_memory_stats()
            for cls in it["boxes"]:
                for k in R.PROBE_KEYS[cls]:
                    if k in keys:
                        continue
                    e = cam.explain(img, k)
                    keys[k] = {"p_yes": round(e["p_yes"], 6), "logodds": round(e["logodds"], 4),
                               "probe_p_stored": it["probe_p"].get(k)}
                    rec["grid"], rec["resized"] = list(e["grid"]), list(e["resized"])
                    for name, m in e["maps"].items():
                        arrays[f"{k}|{name}"] = m.astype(np.float16)
                    release()
            rec["keys"] = keys
            rec["time_s"] = round(time.time() - t0, 2)
            rec["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
            np.savez_compressed(MAP_DIR / (Path(it["image"]).stem + ".npz"), **arrays)
            img.close()
        except Exception as ex:
            rec["error"] = f"{type(ex).__name__}: {ex}"
            release()
        done[it["image"]] = rec
        tmp = CKPT.with_suffix(".tmp")
        tmp.write_text(json.dumps({"_fp": fp, "_setup": setup, "rows": done}), encoding="utf-8")
        os.replace(tmp, CKPT)
        n_new += 1
        el = (time.time() - t_run) / n_new
        left = sum(1 for x in items if x["image"] not in done)
        if "error" in rec:
            print(f"[{i}/{len(items)}] {it['image'][-10:]} ERROR {rec['error']}", flush=True)
        else:
            ps = " ".join(f"{k[:5]}={v['p_yes']:.2f}/{(v['probe_p_stored'] or 0):.2f}" for k, v in rec["keys"].items())
            print(f"[{i}/{len(items)}] {it['image'][-10:]} {rec['size']} grid={rec['grid']} "
                  f"t={rec['time_s']}s peak={rec['peak_gb']}GB {ps} | eta {left * el / 3600:.1f} h", flush=True)

    out = {"setup": setup, "fingerprint": fp, "loader": {k: info.get(k) for k in ("model_class", "quantization_effective", "max_pixels") if isinstance(info, dict)},
           "rows": [done[it["image"]] for it in items if it["image"] in done]}
    OUT.write_text(json.dumps(out), encoding="utf-8")
    print(f"[done] {len(out['rows'])} frames -> {OUT.name}; maps in {MAP_DIR.name}/", flush=True)
    if not args.limit:
        CKPT.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
