"""
Sandbox experiment: production Stage 2 (free-form list) vs per-type probing.

Runs every Attain SMP WS_V2.0 image through, in ONE process and on the SAME
decoded image:

  stage1   production Stage 1 (PavementClassifier.predict_stage1)
  v0       production Stage 2 (PavementClassifier.predict_stage2) - run on
           every image, not only the ones Stage 1 passes, so Stage 2 can be
           scored as a component; the end-to-end score applies Stage 1 later
  probe:*  scripts/stage2_probe.TypeProber, one entry per variant
           (system prompt style : question style)

Nothing is decided here. This script only records model outputs; the split,
threshold tuning and scoring live in scripts/stage2_probe_report.py, so the
decision rules can be changed and re-scored without re-running the GPU.

Design notes
------------
* Production code path. The model is loaded by PavementClassifier with the
  production settings (4-bit, 1024x1024 input cap, v2 prompts, no adapter),
  and the probes run on that same model object, so a difference between v0
  and a probe cannot be a loader or precision difference.
* No-distress images are kept. 78 of 847 images carry only non-distress
  annotations (faded marking, drop-off) or none; they are the only negatives
  for every class at once, and the earlier evaluations dropped them.
* Crash-safe. Rows are checkpointed after every image with the previous save
  kept as .bak (Windows rename semantics, see calibrate_confidence.py). The
  checkpoint records a hash of everything that determines an output - model,
  precision, pixel cap, prompts, probe questions - and refuses to resume from
  a checkpoint made under a different configuration, so two configurations
  can never be pooled into one results file.
* No allocator carry-over. gc + empty_cache after every image: Attain mixes
  three frame shapes, and without the release the cache grows to the full
  card and Windows pages the excess to system RAM (outputs unchanged, timing
  inflated up to 10x).

Usage:
    venv/Scripts/python.exe scripts/stage2_probe_experiment.py
    venv/Scripts/python.exe scripts/stage2_probe_experiment.py --limit 20 --out probe_smoke.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Production settings, fixed BEFORE app.model is imported (it reads them at
# import time). setdefault would let a stray shell variable change the
# experiment silently, so they are assigned.
os.environ["PROMPTS_VERSION"] = "v2"
os.environ["DISABLE_ADAPTER"] = "true"
os.environ["STAGE2_CONFIDENCE_MODE"] = "field"
os.environ.setdefault("MAX_IMAGE_PIXELS", "1048576")

import torch  # noqa: E402
from PIL import Image  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "attain_eval", PROJECT_ROOT / "scripts" / "07_cross_dataset_eval.py")
_ae = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ae)

from scripts.stage2_probe import (  # noqa: E402
    SYSTEM_PROMPTS, TypeProber, all_probe_types, question_for,
)

EVAL_DIR = PROJECT_ROOT / "eval_results"
SEV_RANK = {"Unknown": 0, "Low": 1, "Medium": 2, "High": 3}


# ============================================================
# Ground truth - all images, distress or not
# ============================================================

def load_all_attain(subset: str) -> list[dict]:
    """Every image in the subset with its distress classes (possibly none).

    Unlike 07_cross_dataset_eval.load_ground_truth this keeps images whose
    only annotations are non-distress classes, because they are the
    negatives. Per class it records the highest annotated severity and the
    share of annotated box area (a proxy for how dominant the class is).
    """
    info = _ae.get_subset_info(subset)
    rows = []
    for img_path in sorted(info["images_dir"].glob("*.jpg")):
        label = info["labels_dir"] / (img_path.stem + ".xml")
        sev: dict = {}
        area: dict = {}
        excluded_present = []
        if label.exists():
            try:
                for obj in ET.parse(label).findall(".//object"):
                    name = obj.findtext("name")
                    if not name:
                        continue
                    cls, s = _ae.parse_attain_class(name)
                    m = _ae.ATTAIN_CLASS_MAP.get(cls)
                    if m is None:
                        raise ValueError(f"unmapped Attain class {cls!r} in {label.name}")
                    if m.get("label") == "EXCLUDE":
                        excluded_present.append(cls)
                        continue
                    if SEV_RANK.get(s, 0) >= SEV_RANK.get(sev.get(cls, "Unknown"), 0):
                        sev[cls] = s
                    box = obj.find("bndbox")
                    if box is not None:
                        w = float(box.findtext("xmax")) - float(box.findtext("xmin"))
                        h = float(box.findtext("ymax")) - float(box.findtext("ymin"))
                        if w > 0 and h > 0:
                            area[cls] = area.get(cls, 0.0) + w * h
            except ET.ParseError as e:
                raise ValueError(f"unreadable label file {label}: {e}")
        total = sum(area.values())
        rows.append({
            "image": img_path.name,
            "image_path": str(img_path),
            "index": int(img_path.stem.rsplit("_", 1)[-1]),
            "gt": sorted(sev),
            "gt_severity": sev,
            "gt_area_share": {k: round(v / total, 4) for k, v in area.items()} if total else {},
            "excluded_present": sorted(set(excluded_present)),
        })
    return rows


# ============================================================
# Checkpointing
# ============================================================

def save_checkpoint(path: Path, data: dict) -> None:
    bak = path.with_suffix(".json.bak")
    try:
        if path.exists():
            try:
                if bak.exists():
                    bak.unlink()
                path.rename(bak)
            except OSError:
                pass
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except OSError as e:
        print(f"  [warn] checkpoint save failed: {e} - continuing in memory")


def load_checkpoint(path: Path):
    for cand in (path, path.with_suffix(".json.bak")):
        if cand.exists():
            try:
                return json.loads(cand.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"  [warn] {cand.name} unreadable ({type(e).__name__}), trying fallback")
    return None


def release_gpu() -> None:
    """Return cached allocator blocks to the driver between steps. On a card
    shared with the production server, a reserve left over from one step's
    peak is what pushes the next step into WDDM paging."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def config_fingerprint(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


# ============================================================
# Main
# ============================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="WS_V2.0")
    ap.add_argument("--variants", default="min:def,min:name,tax:def",
                    help="comma-separated system:question probe styles")
    ap.add_argument("--limit", type=int, default=0, help="first N images only (smoke test)")
    ap.add_argument("--no-v0", action="store_true",
                    help="skip the production Stage 2 generation (probe-only runs)")
    ap.add_argument("--out", default="stage2_probe_raw_attain.json")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    variants = [tuple(v.split(":")) for v in args.variants.split(",") if v]
    for s, q in variants:
        if s not in SYSTEM_PROMPTS:
            sys.exit(f"unknown system style {s!r}")
        question_for(all_probe_types()[0], q)  # raises on unknown style

    rows_in = load_all_attain(args.subset)
    if args.limit:
        rows_in = rows_in[: args.limit]
    n_neg = sum(1 for r in rows_in if not r["gt"])
    print(f"[data] {len(rows_in)} images ({len(rows_in) - n_neg} with distress, {n_neg} without)")

    from app.model import PavementClassifier, _ACTIVE_PROMPTS, STAGE2_CONFIDENCE_MODE
    from scripts.utils_v2_prompts import STAGE2_SYSTEM_PROMPT_V2, STAGE2_USER_PROMPT_V2

    probe_types = all_probe_types(include_indicators=True)
    cfg = {
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "quantization_bits": 4,
        "max_pixels": int(os.environ["MAX_IMAGE_PIXELS"]),
        "prompts": _ACTIVE_PROMPTS,
        "stage2_confidence_mode": STAGE2_CONFIDENCE_MODE,
        "stage2_prompt_sha": hashlib.sha256(
            (STAGE2_SYSTEM_PROMPT_V2 + STAGE2_USER_PROMPT_V2).encode()).hexdigest()[:16],
        "run_v0": not args.no_v0,
        "variants": [f"{s}:{q}" for s, q in variants],
        "probe_questions_sha": hashlib.sha256(json.dumps(
            {f"{s}:{q}": [SYSTEM_PROMPTS[s]] + [question_for(p, q) for p in probe_types]
             for s, q in variants}, sort_keys=True).encode()).hexdigest()[:16],
    }
    fp = config_fingerprint(cfg)
    print(f"[config] {fp} {json.dumps(cfg)}")

    ckpt = EVAL_DIR / (Path(args.out).stem + "_checkpoint.json")
    done: dict = {}
    if not args.fresh:
        blob = load_checkpoint(ckpt)
        if blob:
            if blob.get("_fingerprint") != fp:
                sys.exit(f"ERROR: {ckpt.name} was made under config "
                         f"{blob.get('_fingerprint')} != {fp}. Use --fresh or another --out.")
            done = blob.get("rows", {})
            print(f"[resume] {len(done)} images already done")

    clf = PavementClassifier(model_path=cfg["model"], adapter_path=None, quantization_bits=4)
    probers = {f"{s}:{q}": TypeProber(clf._model, clf._processor, probe_types, s, q)
               for s, q in variants}

    # Fast path must agree with the slow path before any number is trusted.
    first_img = Image.open(rows_in[0]["image_path"]).convert("RGB")
    parity = {}
    for name, pr in probers.items():
        parity[name] = pr.parity_check(first_img)
        print(f"[parity] {name}: {parity[name]}")
        if not parity[name]["ok"]:
            sys.exit(f"ERROR: fast/slow probe paths disagree for {name}; not running")
    first_img.close()
    release_gpu()

    # Content-free baseline (Zhao et al. 2021, "Calibrate Before Use"): the
    # P(yes) each question gets for an image with nothing in it. It measures
    # each question's built-in yes-bias, which is what per-type thresholds
    # have to absorb.
    blank = Image.new("RGB", (640, 640), (128, 128, 128))
    content_free = {}
    with clf._lock:
        for name, pr in probers.items():
            content_free[name] = {r["key"]: round(r["p_yes"], 6) for r in pr.probe(blank)}
            release_gpu()
    print(f"[content-free] {json.dumps(content_free)[:400]}...")

    def save():
        save_checkpoint(ckpt, {"_fingerprint": fp, "_config": cfg, "rows": done})

    t_start = time.time()
    n_new = 0
    for i, e in enumerate(rows_in, 1):
        if e["image"] in done:
            continue
        rec = dict(e)
        try:
            img = Image.open(e["image_path"]).convert("RGB")
            rec["size"] = list(img.size)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            s1 = clf.predict_stage1(img)
            rec["stage1"] = {k: s1[k] for k in
                             ("stage1_label", "stage1_confidence", "stage1_raw", "stage1_time_ms")}
            release_gpu()
            if not args.no_v0:
                s2 = clf.predict_stage2(img)
                rec["v0"] = {k: s2.get(k) for k in (
                    "distress_types", "severity", "description", "stage2_confidence",
                    "stage2_confidence_field", "stage2_confidence_sequence",
                    "stage2_confidence_primary", "stage2_field_span_found",
                    "stage2_time_ms", "stage2_raw")}
                release_gpu()
            rec["probe"] = {}
            with clf._lock:  # same serialisation production uses
                for name, pr in probers.items():
                    t0 = time.time()
                    res = pr.probe(img)
                    rec["probe"][name] = {
                        "p": {r["key"]: round(r["p_yes"], 6) for r in res},
                        "min_yes_no_mass": round(min(r["yes_no_mass"] for r in res), 5),
                        "time_ms": round((time.time() - t0) * 1000, 1),
                        "chunk": pr.last_chunk_size,
                    }
                    release_gpu()
            if torch.cuda.is_available():
                rec["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2)
            img.close()
        except Exception as ex:
            rec["error"] = f"{type(ex).__name__}: {ex}"
            print(f"[{i}/{len(rows_in)}] ERROR {e['image']}: {rec['error']}")
        done[e["image"]] = rec
        n_new += 1
        save()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if "error" not in rec:
            el = time.time() - t_start
            v0 = rec.get("v0", {}).get("distress_types")
            pm = rec["probe"][next(iter(probers))]["time_ms"]
            print(f"[{i}/{len(rows_in)}] {e['image'][-10:]} gt={e['gt']} v0={v0} "
                  f"probe={pm:.0f}ms  ({el / n_new:.1f}s/img)", flush=True)

    rows = [done[r["image"]] for r in rows_in if r["image"] in done]
    out = {
        "config": cfg,
        "fingerprint": fp,
        "parity": parity,
        "content_free_p_yes": content_free,
        "n_images": len(rows),
        "n_errors": sum(1 for r in rows if "error" in r),
        "seconds": round(time.time() - t_start, 1),
        "rows": rows,
    }
    (EVAL_DIR / args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[done] {len(rows)} rows, {out['n_errors']} errors -> {EVAL_DIR / args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
