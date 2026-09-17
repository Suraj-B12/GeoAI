"""
Re-score an existing Attain result JSON without re-running inference.

Why this exists
---------------
`PIPELINE_TO_ATTAIN` in 07_cross_dataset_eval.py was keyed on the PRE-IRC
pipeline labels. After the IRC:82 migration (commit 73e5863) the pipeline began
emitting canonical IRC names, and two of them fell through the fuzzy matcher and
were silently dropped during scoring:

    "Ravelling"      -> IRC spells it with two Ls; the table had "Raveling"
    "Hungry Surface" -> IRC's name for what Attain calls "Weathering"

A dropped label is scored as neither a true positive nor a false positive, so a
model could name the correct distress and receive no credit for it. Every
IRC-era Tier 2 (zero-shot) number is therefore understated.

Because each result file stores `pred_pipeline` (the raw emitted labels) per
image, the fix can be applied retroactively: re-derive `pred_attain` with the
corrected mapping and recompute every metric offline, in seconds, with zero GPU
time and no risk of a different model version changing the predictions.

Usage:
    venv/Scripts/python.exe scripts/rescore_attain.py \
        --in attain_ab_qwen3vl8b_100.json --out attain_ab_qwen3vl8b_100_rescored.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "attain_eval", PROJECT_ROOT / "scripts" / "07_cross_dataset_eval.py"
)
_ae = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ae)

from scripts.utils import EVAL_DIR  # noqa: E402


def rescore(data: dict) -> dict:
    """Recompute all metrics from stored per-image predictions."""
    rows = data["per_image_results"]
    tracked = {c for c, m in _ae.ATTAIN_CLASS_MAP.items() if m.get("label") != "EXCLUDE"}

    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    in_dist_correct = in_dist_total = 0
    zero_correct = zero_total = 0
    sev_correct = sev_total = 0
    dropped_before: dict[str, int] = defaultdict(int)

    for r in rows:
        pred_pipeline = r.get("pred_pipeline") or []
        pred_attain = _ae.map_pipeline_to_attain_classes(pred_pipeline)
        old_attain = set(r.get("pred_attain") or [])

        # Track which labels the OLD mapping dropped, for the report.
        for lbl in pred_pipeline:
            mapped_now = _ae.map_pipeline_to_attain_classes([lbl])
            if mapped_now and not (mapped_now & old_attain):
                dropped_before[lbl] += 1

        r["pred_attain_rescored"] = sorted(pred_attain)
        gt = set(r.get("gt_attain") or [])

        for cls in tracked:
            in_p, in_g = cls in pred_attain, cls in gt
            if in_p and in_g:
                tp[cls] += 1
            elif in_p:
                fp[cls] += 1
            elif in_g:
                fn[cls] += 1

        for cls in (r.get("gt_attain") or []):
            in_dist = _ae.ATTAIN_CLASS_MAP.get(cls, {}).get("in_dist")
            if in_dist is True:
                in_dist_total += 1
                in_dist_correct += cls in pred_attain
            elif in_dist is False:
                zero_total += 1
                zero_correct += cls in pred_attain

        gt_sev = r.get("gt_severity_top") or "Unknown"
        pred_sev = r.get("pred_severity")
        pred_sev = pred_sev if pred_sev in ("Low", "Medium", "High") else "Unknown"
        if gt_sev != "Unknown":
            sev_total += 1
            sev_correct += pred_sev == gt_sev

    per_class = {}
    for cls in sorted(tracked):
        t, f, n = tp[cls], fp[cls], fn[cls]
        prec = t / (t + f) if (t + f) else 0.0
        rec = t / (t + n) if (t + n) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[cls] = {
            "tier": ("in-dist" if _ae.ATTAIN_CLASS_MAP[cls].get("in_dist") else "zero-shot"),
            "tp": t, "fp": f, "fn": n,
            "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
        }

    s = data["summary"]
    s_old = {
        "tier1_accuracy": s["tier1_in_distribution"]["accuracy"],
        "tier2_accuracy": s["tier2_zero_shot"]["accuracy"],
        "severity_accuracy": s["severity"]["accuracy"],
        "per_class": s["per_class"],
    }
    s["tier1_in_distribution"] = {
        "instances": in_dist_total, "correct": in_dist_correct,
        "accuracy": round(in_dist_correct / max(in_dist_total, 1), 4),
    }
    s["tier2_zero_shot"] = {
        "instances": zero_total, "correct": zero_correct,
        "accuracy": round(zero_correct / max(zero_total, 1), 4),
    }
    s["severity"] = {
        "instances": sev_total, "correct": sev_correct,
        "accuracy": round(sev_correct / max(sev_total, 1), 4),
    }
    s["per_class"] = per_class
    s["rescored"] = {
        "reason": ("PIPELINE_TO_ATTAIN was keyed on pre-IRC labels; canonical IRC "
                   "names 'Ravelling' and 'Hungry Surface' were silently dropped "
                   "during scoring (neither TP nor FP)."),
        "labels_recovered": dict(sorted(dropped_before.items(), key=lambda x: -x[1])),
        "before": s_old,
    }
    # Keep pred_attain in sync so downstream comparison uses corrected values.
    for r in rows:
        r["pred_attain"] = r.pop("pred_attain_rescored")
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    args = ap.parse_args()

    src = Path(args.inp)
    if not src.is_absolute():
        src = EVAL_DIR / src
    data = json.loads(src.read_text(encoding="utf-8"))

    before_t1 = data["summary"]["tier1_in_distribution"]["accuracy"]
    before_t2 = data["summary"]["tier2_zero_shot"]["accuracy"]
    data = rescore(data)
    after_t1 = data["summary"]["tier1_in_distribution"]["accuracy"]
    after_t2 = data["summary"]["tier2_zero_shot"]["accuracy"]

    dst = Path(args.out)
    if not dst.is_absolute():
        dst = EVAL_DIR / dst
    dst.write_text(json.dumps(data, indent=2), encoding="utf-8")

    rec = data["summary"]["rescored"]["labels_recovered"]
    print(f"{src.name} -> {dst.name}")
    print(f"  Tier 1 in-distribution: {before_t1:.4f} -> {after_t1:.4f}")
    print(f"  Tier 2 zero-shot:       {before_t2:.4f} -> {after_t2:.4f}")
    print(f"  labels recovered by the corrected mapping: {rec or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
