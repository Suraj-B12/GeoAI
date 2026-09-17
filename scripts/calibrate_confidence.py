"""
Does the field-restricted Stage 2 confidence discriminate better than the
whole-sequence one?

A confidence score is only useful if it SEPARATES correct predictions from
wrong ones. "Higher numbers" is not an improvement — a metric that returns 0.99
for everything is worse than useless, because the expert-review gate would
auto-accept every mistake.

This runs the PRODUCTION path (PavementClassifier, the same code the worker
uses) over N Attain images with ground truth, records both confidence metrics
for every image, and reports:

  - mean confidence on correct vs incorrect predictions (separation)
  - AUC: probability that a randomly chosen correct prediction scores higher
    than a randomly chosen incorrect one. 0.5 = the metric carries no
    information. Higher is better.
  - at the live 0.80 threshold: how many images each metric auto-accepts, and
    what fraction of those auto-accepts are actually wrong (the real cost)

Usage:
    venv/Scripts/python.exe scripts/calibrate_confidence.py --n 40
    venv/Scripts/python.exe scripts/calibrate_confidence.py --n 40 \
        --model Qwen/Qwen3-VL-8B-Instruct --out conf_calib_qwen3vl.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "attain_eval", PROJECT_ROOT / "scripts" / "07_cross_dataset_eval.py"
)
_ae = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ae)


def auc(pos: list[float], neg: list[float]) -> float:
    """Probability a random positive outranks a random negative (ties = 0.5)."""
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


def summarize(rows: list[dict], key: str, threshold: float) -> dict:
    correct = [r[key] for r in rows if r["correct"]]
    wrong = [r[key] for r in rows if not r["correct"]]
    accepted = [r for r in rows if r[key] >= threshold]
    accepted_wrong = [r for r in accepted if not r["correct"]]
    return {
        "mean_correct": round(sum(correct) / len(correct), 4) if correct else None,
        "mean_wrong": round(sum(wrong) / len(wrong), 4) if wrong else None,
        "separation": (round(sum(correct) / len(correct) - sum(wrong) / len(wrong), 4)
                       if correct and wrong else None),
        "auc": round(auc(correct, wrong), 4),
        "auto_accepted": len(accepted),
        "auto_accepted_pct": round(100.0 * len(accepted) / max(len(rows), 1), 1),
        "auto_accepted_wrong": len(accepted_wrong),
        "auto_accept_error_rate_pct": (
            round(100.0 * len(accepted_wrong) / len(accepted), 1) if accepted else None),
        "sent_to_expert": len(rows) - len(accepted),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--subset", default="WS_V2.0")
    ap.add_argument("--quantization-bits", type=int, default=0, choices=[0, 4, 8])
    ap.add_argument("--out", default="confidence_calibration.json")
    args = ap.parse_args()

    os.environ.setdefault("PROMPTS_VERSION", "v2")
    os.environ["QUANTIZATION_BITS"] = str(args.quantization_bits)
    os.environ["DISABLE_ADAPTER"] = "true"

    from app.model import PavementClassifier
    from scripts.utils import CONFIDENCE_THRESHOLD, EVAL_DIR

    info = _ae.get_subset_info(args.subset)
    gt = _ae.load_ground_truth(info)[: args.n]
    print(f"{len(gt)} images from Attain {args.subset}")

    clf = PavementClassifier(model_path=args.model, adapter_path=None,
                             quantization_bits=args.quantization_bits)

    rows = []
    t_start = time.time()
    for i, entry in enumerate(gt, 1):
        img = Image.open(entry["image_path"]).convert("RGB")
        s1 = clf.predict_stage1(img)
        if not s1["is_distressed"]:
            # Stage 2 never runs; no Stage 2 confidence to calibrate.
            print(f"  [{i}/{len(gt)}] Stage 1 said Normal — skipped")
            continue
        s2 = clf.predict_stage2(img)
        pred_attain = _ae.map_pipeline_to_attain_classes(s2["distress_types"])
        gt_attain = set(entry["gt_attain_classes"])
        hit = bool(pred_attain & gt_attain)
        rows.append({
            "image": Path(entry["image_path"]).name,
            "gt": sorted(gt_attain),
            "pred_pipeline": s2["distress_types"],
            "pred_attain": sorted(pred_attain),
            "correct": hit,
            "field": s2["stage2_confidence_field"],
            "sequence": s2["stage2_confidence_sequence"],
            "field_span_found": s2["stage2_field_span_found"],
            "stage1_confidence": s1["stage1_confidence"],
        })
        print(f"  [{i}/{len(gt)}] {'HIT ' if hit else 'MISS'} "
              f"field={s2['stage2_confidence_field']:.3f} "
              f"seq={s2['stage2_confidence_sequence']:.3f} "
              f"pred={s2['distress_types']}")

    elapsed = time.time() - t_start
    if not rows:
        print("No Stage 2 predictions collected — nothing to calibrate.")
        return 1

    report = {
        "model": args.model,
        "n_images_run": len(gt),
        "n_with_stage2": len(rows),
        "n_correct": sum(r["correct"] for r in rows),
        "threshold": CONFIDENCE_THRESHOLD,
        "seconds_total": round(elapsed, 1),
        "seconds_per_image": round(elapsed / max(len(gt), 1), 1),
        "field_span_found_rate": round(
            sum(r["field_span_found"] for r in rows) / len(rows), 4),
        "metrics": {
            "field": summarize(rows, "field", CONFIDENCE_THRESHOLD),
            "sequence": summarize(rows, "sequence", CONFIDENCE_THRESHOLD),
        },
        "per_image": rows,
    }

    out = EVAL_DIR / args.out
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    f, s = report["metrics"]["field"], report["metrics"]["sequence"]
    print("\n" + "=" * 68)
    print(f"CONFIDENCE CALIBRATION — {args.model}")
    print(f"{len(rows)} Stage 2 predictions, {report['n_correct']} correct "
          f"({100*report['n_correct']/len(rows):.0f}%), threshold {CONFIDENCE_THRESHOLD}")
    print("=" * 68)
    print(f"{'':<34}{'field':>15}{'sequence':>15}")
    for label, k in [("mean conf when CORRECT", "mean_correct"),
                     ("mean conf when WRONG", "mean_wrong"),
                     ("separation (correct-wrong)", "separation"),
                     ("AUC (0.5 = no signal)", "auc"),
                     ("auto-accepted at 0.80", "auto_accepted"),
                     ("  of those, wrong", "auto_accepted_wrong"),
                     ("  auto-accept error rate %", "auto_accept_error_rate_pct"),
                     ("sent to expert review", "sent_to_expert")]:
        print(f"{label:<34}{str(f[k]):>15}{str(s[k]):>15}")
    print("=" * 68)
    print(f"Written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
