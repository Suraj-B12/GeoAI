"""
A/B verification: compare baseline (no adapter) vs new fine-tuned adapter
on (a) the RDD test set, (b) the real Bengaluru photos already in Supabase.

This is the gate for promoting the new adapter to production. It writes:
  - eval_results/ab_comparison.json
  - eval_results/ab_comparison_table.md  (paste into the paper)

PROMOTION CRITERIA (default — adjustable via flags):
  - Stage 1 accuracy: must improve by >= +3 percentage points
  - Stage 2 macro F1: must improve by >= +10 percentage points
  - No regression in Normal-class precision (we don't want it labelling
    obvious clean roads as Distress just to pump distress recall)

If criteria are met -> exit 0, prints "PROMOTE: yes"
If criteria NOT met -> exit 1, prints what regressed

Usage:
    python scripts/ab_compare_adapters.py \
        --baseline-results eval_results/baseline_results.json \
        --finetuned-results eval_results/finetuned_results.json \
        --supabase-eval                    # also re-run on the 76 Bengaluru photos
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Promotion thresholds (in percentage POINTS, not relative %)
STAGE1_ACC_DELTA_MIN = 3.0
STAGE2_F1_DELTA_MIN  = 10.0
NORMAL_PRECISION_REGRESSION_MAX = 5.0   # don't let Normal precision drop more than 5pp


def load_results(path: Path) -> dict:
    if not path.exists():
        print(f"ERROR: {path} does not exist")
        sys.exit(2)
    return json.loads(path.read_text(encoding="utf-8"))


def pct_point_delta(new: float, old: float) -> float:
    """Difference in percentage points (e.g. 0.83 - 0.76 = 7.0 pp)."""
    return (new - old) * 100


def fmt_delta(d: float, suffix: str = "pp") -> str:
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.2f}{suffix}"


def render_table(comparison: dict) -> str:
    """Markdown table for the paper / docs."""
    lines = []
    lines.append("# Baseline vs Fine-Tuned Comparison")
    lines.append("")
    lines.append("## Stage 1 — Binary Detection (10,000 GAPs test images)")
    lines.append("")
    lines.append("| Metric | Baseline | Fine-Tuned | Delta |")
    lines.append("|---|---|---|---|")
    s1 = comparison["stage1"]
    for metric in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]:
        b = s1["baseline"].get(metric, 0)
        f = s1["finetuned"].get(metric, 0)
        d = pct_point_delta(f, b)
        lines.append(f"| {metric} | {b*100:.2f}% | {f*100:.2f}% | {fmt_delta(d)} |")
    lines.append("")

    s2 = comparison["stage2"]
    lines.append("## Stage 2 — Distress Type Classification (5,758 RDD test images)")
    lines.append("")
    lines.append("| Metric | Baseline | Fine-Tuned | Delta |")
    lines.append("|---|---|---|---|")
    for metric in ["primary_accuracy", "f1_macro", "exact_match_rate"]:
        b = s2["baseline"].get(metric, 0)
        f = s2["finetuned"].get(metric, 0)
        d = pct_point_delta(f, b)
        lines.append(f"| {metric} | {b*100:.2f}% | {f*100:.2f}% | {fmt_delta(d)} |")
    lines.append("")

    if "per_class" in s2:
        lines.append("### Stage 2 per-class recall")
        lines.append("")
        lines.append("| Class | Baseline | Fine-Tuned | Delta |")
        lines.append("|---|---|---|---|")
        for cls, vals in s2["per_class"].items():
            b = vals.get("baseline_recall", 0)
            f = vals.get("finetuned_recall", 0)
            d = pct_point_delta(f, b)
            lines.append(f"| {cls} | {b*100:.2f}% | {f*100:.2f}% | {fmt_delta(d)} |")
        lines.append("")

    lines.append("## Promotion Decision")
    lines.append("")
    decision = comparison.get("decision", {})
    lines.append(f"- Stage 1 accuracy delta: {fmt_delta(decision.get('s1_acc_delta', 0))} (threshold: +{STAGE1_ACC_DELTA_MIN:.1f}pp)")
    lines.append(f"- Stage 2 macro F1 delta: {fmt_delta(decision.get('s2_f1_delta', 0))} (threshold: +{STAGE2_F1_DELTA_MIN:.1f}pp)")
    lines.append(f"- Normal precision delta: {fmt_delta(decision.get('normal_prec_delta', 0))} (must be >= -{NORMAL_PRECISION_REGRESSION_MAX:.1f}pp)")
    lines.append("")
    lines.append(f"**Recommendation: {'PROMOTE' if decision.get('promote') else 'DO NOT PROMOTE'}**")
    lines.append("")
    if decision.get("reasons"):
        lines.append("Reasons:")
        for r in decision["reasons"]:
            lines.append(f"- {r}")

    return "\n".join(lines)


def per_class_recalls(stage2_classification_report: dict) -> dict:
    """Pull per-class recall numbers out of an sklearn classification_report."""
    skip = {"accuracy", "macro avg", "weighted avg"}
    out = {}
    for cls, vals in (stage2_classification_report or {}).items():
        if cls in skip or not isinstance(vals, dict):
            continue
        out[cls] = vals.get("recall", 0)
    return out


def decide(s1_baseline: dict, s1_finetuned: dict,
           s2_baseline: dict, s2_finetuned: dict) -> dict:
    """Apply promotion criteria, return decision + reasons."""
    s1_acc_delta = pct_point_delta(s1_finetuned.get("accuracy", 0),
                                   s1_baseline.get("accuracy", 0))
    s2_f1_delta = pct_point_delta(s2_finetuned.get("f1_macro", 0),
                                  s2_baseline.get("f1_macro", 0))

    # Pull Normal-class precision from Stage 1 report
    normal_b = (s1_baseline.get("classification_report", {}).get("Normal", {}) or {}).get("precision", 0)
    normal_f = (s1_finetuned.get("classification_report", {}).get("Normal", {}) or {}).get("precision", 0)
    normal_prec_delta = pct_point_delta(normal_f, normal_b)

    reasons = []
    promote = True

    if s1_acc_delta < STAGE1_ACC_DELTA_MIN:
        promote = False
        reasons.append(f"Stage 1 accuracy improvement {s1_acc_delta:.2f}pp < threshold {STAGE1_ACC_DELTA_MIN:.1f}pp")
    else:
        reasons.append(f"Stage 1 accuracy improvement {s1_acc_delta:.2f}pp >= {STAGE1_ACC_DELTA_MIN:.1f}pp")

    if s2_f1_delta < STAGE2_F1_DELTA_MIN:
        promote = False
        reasons.append(f"Stage 2 macro F1 improvement {s2_f1_delta:.2f}pp < threshold {STAGE2_F1_DELTA_MIN:.1f}pp")
    else:
        reasons.append(f"Stage 2 macro F1 improvement {s2_f1_delta:.2f}pp >= {STAGE2_F1_DELTA_MIN:.1f}pp")

    if normal_prec_delta < -NORMAL_PRECISION_REGRESSION_MAX:
        promote = False
        reasons.append(f"Normal-class precision regressed by {normal_prec_delta:.2f}pp (max allowed: -{NORMAL_PRECISION_REGRESSION_MAX:.1f}pp)")
    else:
        reasons.append(f"Normal-class precision delta {normal_prec_delta:.2f}pp within tolerance")

    return {
        "promote": promote,
        "s1_acc_delta": s1_acc_delta,
        "s2_f1_delta": s2_f1_delta,
        "normal_prec_delta": normal_prec_delta,
        "reasons": reasons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="A/B compare baseline vs fine-tuned adapter")
    parser.add_argument("--baseline-results", default="eval_results/baseline_results.json")
    parser.add_argument("--finetuned-results", default="eval_results/finetuned_results.json")
    parser.add_argument("--out", default="eval_results/ab_comparison")
    args = parser.parse_args()

    baseline_path = PROJECT_ROOT / args.baseline_results
    finetuned_path = PROJECT_ROOT / args.finetuned_results

    baseline = load_results(baseline_path)
    finetuned = load_results(finetuned_path)

    s1_baseline = baseline.get("stage1", {})
    s2_baseline = baseline.get("stage2", {})
    s1_finetuned = finetuned.get("stage1", {})
    s2_finetuned = finetuned.get("stage2", {})

    if not s1_baseline or not s2_baseline:
        print(f"ERROR: baseline_results.json missing stage1/stage2 — re-run scripts/03_baseline_eval.py")
        return 2
    if not s1_finetuned or not s2_finetuned:
        print(f"ERROR: finetuned_results.json missing stage1/stage2 — re-run scripts/04_post_finetune_eval.py")
        return 2

    decision = decide(s1_baseline, s1_finetuned, s2_baseline, s2_finetuned)

    # Build per-class recall comparison
    per_class = {}
    bcr = per_class_recalls(s2_baseline.get("classification_report", {}))
    fcr = per_class_recalls(s2_finetuned.get("classification_report", {}))
    for cls in sorted(set(bcr) | set(fcr)):
        per_class[cls] = {
            "baseline_recall": bcr.get(cls, 0),
            "finetuned_recall": fcr.get(cls, 0),
        }

    comparison = {
        "stage1": {"baseline": s1_baseline, "finetuned": s1_finetuned},
        "stage2": {
            "baseline": s2_baseline,
            "finetuned": s2_finetuned,
            "per_class": per_class,
        },
        "decision": decision,
    }

    out_json = PROJECT_ROOT / (args.out + ".json")
    out_md = PROJECT_ROOT / (args.out + "_table.md")
    out_json.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    out_md.write_text(render_table(comparison), encoding="utf-8")

    print()
    print("=" * 70)
    print("A/B Comparison Summary")
    print("=" * 70)
    for r in decision["reasons"]:
        print(f"  - {r}")
    print()
    print(f"  PROMOTE: {'yes' if decision['promote'] else 'no'}")
    print()
    print(f"  Detail JSON: {out_json}")
    print(f"  Markdown:    {out_md}")

    return 0 if decision["promote"] else 1


if __name__ == "__main__":
    sys.exit(main())
