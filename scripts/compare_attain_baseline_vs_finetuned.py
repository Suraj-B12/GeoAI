"""
Compare baseline (no adapter) vs fine-tuned (LoRA adapter) on Attain WS_V2.0.

Inputs:
  eval_results/attain_baseline_results.json   (run with no --adapter-path)
  eval_results/attain_finetuned_results.json  (run with --adapter-path adapters/v2-...)

Outputs:
  eval_results/attain_baseline_vs_finetuned.json
  eval_results/attain_baseline_vs_finetuned.md   (paper-ready)

Answers the questions:
  1. Is the QLoRA adapter actually helping on cross-dataset (Attain) data?
  2. Is the adapter biased toward predicting Pothole everywhere?
  3. Did fine-tuning improve known classes at the cost of zero-shot ones?
  4. Where exactly does the adapter win, and where does it lose?
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "eval_results"


def load(name: str) -> dict:
    p = EVAL_DIR / name
    if not p.exists():
        sys.exit(f"ERROR: {p} not found")
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def index_by_image(rows: list[dict]) -> dict[str, dict]:
    return {r["image"]: r for r in rows}


def prediction_frequencies(rows: list[dict]) -> Counter:
    """How often did the model predict each pipeline label?"""
    c: Counter = Counter()
    for r in rows:
        for p in r.get("pred_pipeline") or []:
            c[p] += 1
    return c


def attain_prediction_frequencies(rows: list[dict]) -> Counter:
    """How often did the mapped Attain class appear in predictions?"""
    c: Counter = Counter()
    for r in rows:
        for p in r.get("pred_attain") or []:
            c[p] += 1
    return c


def stage1_detection_rate(rows: list[dict]) -> dict:
    n = len(rows)
    distressed = sum(1 for r in rows if r.get("is_distressed_pred"))
    return {
        "total": n,
        "predicted_distress": distressed,
        "predicted_normal": n - distressed,
        "distress_rate": round(distressed / max(n, 1), 4),
    }


def severity_distribution(rows: list[dict]) -> Counter:
    c: Counter = Counter()
    for r in rows:
        c[r.get("pred_severity") or "None"] += 1
    return c


def per_image_correctness(rows: list[dict], from_class_map: dict) -> dict[str, set]:
    """
    For each image, which GT classes did the model correctly identify?
    Returns {image_name: set_of_correctly_predicted_attain_classes}.
    """
    out: dict[str, set] = {}
    for r in rows:
        gt = set(r.get("gt_attain") or [])
        pred = set(r.get("pred_attain") or [])
        out[r["image"]] = gt & pred
    return out


def head_to_head(base_rows: list[dict], ft_rows: list[dict]) -> dict:
    """
    Per-image win/loss/tie analysis.
    'Better' = correctly identified MORE GT classes (recall++).
    'Worse' = correctly identified FEWER.
    """
    base_idx = index_by_image(base_rows)
    ft_idx = index_by_image(ft_rows)
    common = set(base_idx) & set(ft_idx)

    base_better = []
    ft_better = []
    tied_correct = 0
    tied_wrong = 0

    for img in common:
        b = base_idx[img]
        f = ft_idx[img]
        gt = set(b.get("gt_attain") or [])
        b_correct = gt & set(b.get("pred_attain") or [])
        f_correct = gt & set(f.get("pred_attain") or [])
        if len(f_correct) > len(b_correct):
            ft_better.append({
                "image": img,
                "gt": list(gt),
                "baseline_pred": b.get("pred_pipeline"),
                "finetuned_pred": f.get("pred_pipeline"),
                "baseline_correct": list(b_correct),
                "finetuned_correct": list(f_correct),
            })
        elif len(b_correct) > len(f_correct):
            base_better.append({
                "image": img,
                "gt": list(gt),
                "baseline_pred": b.get("pred_pipeline"),
                "finetuned_pred": f.get("pred_pipeline"),
                "baseline_correct": list(b_correct),
                "finetuned_correct": list(f_correct),
            })
        else:
            if b_correct:
                tied_correct += 1
            else:
                tied_wrong += 1

    return {
        "n_compared": len(common),
        "finetuned_strictly_better": len(ft_better),
        "baseline_strictly_better": len(base_better),
        "tied_both_correct": tied_correct,
        "tied_both_wrong_or_partial": tied_wrong,
        "ft_wins": ft_better[:25],   # cap example list
        "base_wins": base_better[:25],
    }


def main():
    base = load("attain_baseline_results.json")
    ft = load("attain_finetuned_results.json")

    base_rows = base.get("per_image_results") or []
    ft_rows = ft.get("per_image_results") or []
    base_summary = base["summary"]
    ft_summary = ft["summary"]

    # ===== Aggregate accuracy =====
    accuracy_table = {
        "tier1_in_distribution": {
            "baseline":  base_summary["tier1_in_distribution"],
            "finetuned": ft_summary["tier1_in_distribution"],
            "delta_accuracy": round(
                ft_summary["tier1_in_distribution"]["accuracy"]
                - base_summary["tier1_in_distribution"]["accuracy"], 4),
        },
        "tier2_zero_shot": {
            "baseline":  base_summary["tier2_zero_shot"],
            "finetuned": ft_summary["tier2_zero_shot"],
            "delta_accuracy": round(
                ft_summary["tier2_zero_shot"]["accuracy"]
                - base_summary["tier2_zero_shot"]["accuracy"], 4),
        },
        "severity": {
            "baseline":  base_summary["severity"],
            "finetuned": ft_summary["severity"],
            "delta_accuracy": round(
                ft_summary["severity"]["accuracy"]
                - base_summary["severity"]["accuracy"], 4),
        },
    }

    # ===== Per-class P/R/F1 deltas =====
    classes = set(base_summary["per_class"]) | set(ft_summary["per_class"])
    per_class = {}
    for c in sorted(classes):
        b = base_summary["per_class"].get(c) or {"tp": 0, "fp": 0, "fn": 0, "precision": 0, "recall": 0, "f1": 0, "tier": "?"}
        f = ft_summary["per_class"].get(c) or {"tp": 0, "fp": 0, "fn": 0, "precision": 0, "recall": 0, "f1": 0, "tier": b.get("tier", "?")}
        per_class[c] = {
            "tier": f.get("tier") or b.get("tier"),
            "baseline":  {k: b[k] for k in ("tp", "fp", "fn", "precision", "recall", "f1")},
            "finetuned": {k: f[k] for k in ("tp", "fp", "fn", "precision", "recall", "f1")},
            "delta": {
                "tp": f["tp"] - b["tp"],
                "fp": f["fp"] - b["fp"],
                "fn": f["fn"] - b["fn"],
                "precision": round(f["precision"] - b["precision"], 4),
                "recall": round(f["recall"] - b["recall"], 4),
                "f1": round(f["f1"] - b["f1"], 4),
            },
        }

    # ===== Bias analysis: how often each model PREDICTS each class =====
    base_pred_freq = prediction_frequencies(base_rows)
    ft_pred_freq = prediction_frequencies(ft_rows)
    base_attain_freq = attain_prediction_frequencies(base_rows)
    ft_attain_freq = attain_prediction_frequencies(ft_rows)

    all_pipeline_labels = set(base_pred_freq) | set(ft_pred_freq)
    pipeline_freq_table = {}
    for lbl in sorted(all_pipeline_labels):
        b_n = base_pred_freq.get(lbl, 0)
        f_n = ft_pred_freq.get(lbl, 0)
        pipeline_freq_table[lbl] = {
            "baseline_count": b_n,
            "finetuned_count": f_n,
            "delta": f_n - b_n,
        }

    # ===== Stage 1 detection =====
    s1_table = {
        "baseline":  stage1_detection_rate(base_rows),
        "finetuned": stage1_detection_rate(ft_rows),
    }

    # ===== Severity distribution =====
    base_sev = severity_distribution(base_rows)
    ft_sev = severity_distribution(ft_rows)
    sev_keys = sorted(set(base_sev) | set(ft_sev))
    sev_table = {
        k: {"baseline": base_sev.get(k, 0), "finetuned": ft_sev.get(k, 0)}
        for k in sev_keys
    }

    # ===== Head-to-head per-image =====
    h2h = head_to_head(base_rows, ft_rows)

    # ===== Pothole bias-specific check =====
    pothole_check = {
        "baseline_predictions_with_pothole": sum(
            1 for r in base_rows if "Pothole (D40)" in (r.get("pred_pipeline") or [])
        ),
        "finetuned_predictions_with_pothole": sum(
            1 for r in ft_rows if "Pothole (D40)" in (r.get("pred_pipeline") or [])
        ),
        "actual_pothole_instances_in_gt": sum(
            1 for r in ft_rows if "Pothole" in (r.get("gt_attain") or [])
        ),
    }

    out = {
        "subset": ft_summary["subset"],
        "n_images": ft_summary["n_images"],
        "accuracy_table": accuracy_table,
        "per_class": per_class,
        "stage1_detection": s1_table,
        "severity_distribution": sev_table,
        "prediction_frequencies_pipeline": pipeline_freq_table,
        "prediction_frequencies_attain": {
            c: {"baseline": base_attain_freq.get(c, 0), "finetuned": ft_attain_freq.get(c, 0)}
            for c in sorted(set(base_attain_freq) | set(ft_attain_freq))
        },
        "pothole_bias_check": pothole_check,
        "head_to_head": h2h,
    }

    json_out = EVAL_DIR / "attain_baseline_vs_finetuned.json"
    md_out = EVAL_DIR / "attain_baseline_vs_finetuned.md"
    json_out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    md_out.write_text(to_markdown(out), encoding="utf-8")

    # ===== Print summary =====
    print()
    print("=" * 78)
    print("ATTAIN  WS_V2.0  —  BASELINE  vs  FINE-TUNED  COMPARISON")
    print("=" * 78)
    print()
    a = accuracy_table
    print("Accuracy:")
    print(f"  Tier 1 (in-distribution): {a['tier1_in_distribution']['baseline']['accuracy']:.2%}  ->  "
          f"{a['tier1_in_distribution']['finetuned']['accuracy']:.2%}   "
          f"(Δ {a['tier1_in_distribution']['delta_accuracy']:+.2%})")
    print(f"  Tier 2 (zero-shot):       {a['tier2_zero_shot']['baseline']['accuracy']:.2%}  ->  "
          f"{a['tier2_zero_shot']['finetuned']['accuracy']:.2%}   "
          f"(Δ {a['tier2_zero_shot']['delta_accuracy']:+.2%})")
    print(f"  Severity:                 {a['severity']['baseline']['accuracy']:.2%}  ->  "
          f"{a['severity']['finetuned']['accuracy']:.2%}   "
          f"(Δ {a['severity']['delta_accuracy']:+.2%})")
    print()
    print("Per-class F1:")
    print(f"  {'Class':24s} {'tier':10s} {'base F1':>8s} {'ft F1':>8s} {'ΔF1':>8s} {'base TP':>8s} {'ft TP':>8s}")
    for c, m in per_class.items():
        print(f"  {c:24s} {m['tier']:10s} "
              f"{m['baseline']['f1']:>8.3f} {m['finetuned']['f1']:>8.3f} "
              f"{m['delta']['f1']:>+8.3f} {m['baseline']['tp']:>8d} {m['finetuned']['tp']:>8d}")
    print()
    print("Pothole-bias check:")
    pc = pothole_check
    print(f"  GT pothole instances: {pc['actual_pothole_instances_in_gt']}")
    print(f"  Baseline predictions containing Pothole (D40): {pc['baseline_predictions_with_pothole']}")
    print(f"  Fine-tuned predictions containing Pothole (D40): {pc['finetuned_predictions_with_pothole']}")
    print()
    print("Head-to-head per-image:")
    h = h2h
    print(f"  Compared: {h['n_compared']} images")
    print(f"  Fine-tuned strictly better: {h['finetuned_strictly_better']}")
    print(f"  Baseline strictly better:   {h['baseline_strictly_better']}")
    print(f"  Tied (both correct):        {h['tied_both_correct']}")
    print(f"  Tied (both wrong/partial):  {h['tied_both_wrong_or_partial']}")
    print()
    print("Pipeline-label prediction counts (top of bias check):")
    for lbl, c in sorted(pipeline_freq_table.items(), key=lambda kv: -max(kv[1]['baseline_count'], kv[1]['finetuned_count']))[:12]:
        print(f"  {lbl:32s} base={c['baseline_count']:>4}  ft={c['finetuned_count']:>4}  Δ={c['delta']:>+5}")
    print()
    print(f"JSON: {json_out}")
    print(f"MD:   {md_out}")


def to_markdown(d: dict) -> str:
    lines = []
    lines.append("# Attain WS_V2.0 — Baseline vs Fine-tuned Comparison")
    lines.append("")
    lines.append(f"Subset: {d['subset']}, {d['n_images']} images")
    lines.append("")
    lines.append("## Headline accuracy")
    lines.append("")
    a = d["accuracy_table"]
    lines.append("| Metric | Baseline | Fine-tuned | Δ |")
    lines.append("|---|---|---|---|")
    for k, label in [
        ("tier1_in_distribution", "Tier 1 (RDD-trained classes)"),
        ("tier2_zero_shot",       "Tier 2 (zero-shot, taxonomy-only)"),
        ("severity",              "Severity classification"),
    ]:
        b = a[k]["baseline"]["accuracy"]
        f = a[k]["finetuned"]["accuracy"]
        delta = a[k]["delta_accuracy"]
        lines.append(f"| {label} | {b:.2%} | {f:.2%} | **{delta:+.2%}** |")
    lines.append("")

    lines.append("## Per-class precision / recall / F1")
    lines.append("")
    lines.append("| Class | Tier | Base P | FT P | Base R | FT R | Base F1 | FT F1 | ΔF1 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for c, m in d["per_class"].items():
        b, f = m["baseline"], m["finetuned"]
        lines.append(f"| {c} | {m['tier']} | {b['precision']:.3f} | {f['precision']:.3f} "
                     f"| {b['recall']:.3f} | {f['recall']:.3f} | {b['f1']:.3f} | {f['f1']:.3f} "
                     f"| **{m['delta']['f1']:+.3f}** |")
    lines.append("")

    lines.append("## Per-class TP / FP / FN")
    lines.append("")
    lines.append("| Class | Base TP | FT TP | Base FP | FT FP | Base FN | FT FN |")
    lines.append("|---|---|---|---|---|---|---|")
    for c, m in d["per_class"].items():
        b, f = m["baseline"], m["finetuned"]
        lines.append(f"| {c} | {b['tp']} | {f['tp']} | {b['fp']} | {f['fp']} | {b['fn']} | {f['fn']} |")
    lines.append("")

    lines.append("## Stage 1 detection rate")
    lines.append("")
    s1 = d["stage1_detection"]
    lines.append("| Model | Total | Predicted Distress | Predicted Normal | Distress rate |")
    lines.append("|---|---|---|---|---|")
    for k in ("baseline", "finetuned"):
        x = s1[k]
        lines.append(f"| {k} | {x['total']} | {x['predicted_distress']} | {x['predicted_normal']} | {x['distress_rate']:.2%} |")
    lines.append("")

    lines.append("## Pothole bias check")
    lines.append("")
    pc = d["pothole_bias_check"]
    lines.append(f"- Ground-truth pothole instances in dataset: **{pc['actual_pothole_instances_in_gt']}**")
    lines.append(f"- Baseline predictions containing 'Pothole (D40)': **{pc['baseline_predictions_with_pothole']}**")
    lines.append(f"- Fine-tuned predictions containing 'Pothole (D40)': **{pc['finetuned_predictions_with_pothole']}**")
    lines.append("")

    lines.append("## Pipeline-label prediction frequency")
    lines.append("")
    lines.append("How often each model emitted each pipeline label across all 769 images:")
    lines.append("")
    lines.append("| Pipeline label | Baseline | Fine-tuned | Δ |")
    lines.append("|---|---|---|---|")
    sorted_labels = sorted(d["prediction_frequencies_pipeline"].items(),
                           key=lambda kv: -max(kv[1]["baseline_count"], kv[1]["finetuned_count"]))
    for lbl, c in sorted_labels[:25]:
        lines.append(f"| {lbl} | {c['baseline_count']} | {c['finetuned_count']} | {c['delta']:+} |")
    lines.append("")

    lines.append("## Severity prediction distribution")
    lines.append("")
    lines.append("| Severity | Baseline | Fine-tuned |")
    lines.append("|---|---|---|")
    for k, c in d["severity_distribution"].items():
        lines.append(f"| {k} | {c['baseline']} | {c['finetuned']} |")
    lines.append("")

    lines.append("## Head-to-head per-image (whose predictions cover MORE of GT)")
    lines.append("")
    h = d["head_to_head"]
    lines.append(f"- Images compared: {h['n_compared']}")
    lines.append(f"- **Fine-tuned strictly better**: {h['finetuned_strictly_better']}")
    lines.append(f"- **Baseline strictly better**:   {h['baseline_strictly_better']}")
    lines.append(f"- Tied, both correct on at least one type: {h['tied_both_correct']}")
    lines.append(f"- Tied, both wrong / partial:             {h['tied_both_wrong_or_partial']}")
    lines.append("")
    if h["base_wins"]:
        lines.append("### Examples where baseline beat fine-tuned (first 10)")
        lines.append("")
        lines.append("| Image | Ground truth | Baseline preds | Fine-tuned preds |")
        lines.append("|---|---|---|---|")
        for ex in h["base_wins"][:10]:
            lines.append(f"| {ex['image']} | {', '.join(ex['gt'])} "
                         f"| {', '.join(ex['baseline_pred'] or [])} "
                         f"| {', '.join(ex['finetuned_pred'] or [])} |")
        lines.append("")
    if h["ft_wins"]:
        lines.append("### Examples where fine-tuned beat baseline (first 10)")
        lines.append("")
        lines.append("| Image | Ground truth | Baseline preds | Fine-tuned preds |")
        lines.append("|---|---|---|---|")
        for ex in h["ft_wins"][:10]:
            lines.append(f"| {ex['image']} | {', '.join(ex['gt'])} "
                         f"| {', '.join(ex['baseline_pred'] or [])} "
                         f"| {', '.join(ex['finetuned_pred'] or [])} |")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
