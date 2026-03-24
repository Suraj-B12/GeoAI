"""
Step 4: Evaluate the fine-tuned model and compare with baseline.

This is a convenience wrapper around 03_baseline_eval.py that:
  1. Runs evaluation with the LoRA adapter loaded
  2. Loads baseline results for side-by-side comparison
  3. Prints a formatted comparison table

Usage:
    python scripts/04_post_finetune_eval.py --adapter-path outputs/qwen25vl-qlora-gaps-rdd/
    python scripts/04_post_finetune_eval.py --adapter-path outputs/qwen25vl-qlora-gaps-rdd/ --max-samples 500
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils import EVAL_DIR, OUTPUTS_DIR

# Python can't import modules starting with digits via normal import syntax.
# Use importlib to dynamically import 03_baseline_eval.
eval_module = importlib.import_module("scripts.03_baseline_eval")


def print_comparison(baseline: dict, finetuned: dict) -> None:
    """Print a side-by-side comparison table."""

    print("\n" + "=" * 70)
    print("COMPARISON: Baseline vs Fine-Tuned")
    print("=" * 70)

    # Stage 1
    if "stage1" in baseline and "stage1" in finetuned:
        b1 = baseline["stage1"]
        f1 = finetuned["stage1"]
        print("\n--- Stage 1: Binary Detection (GAPs) ---")
        print(f"{'Metric':<25} {'Baseline':>12} {'Fine-Tuned':>12} {'Delta':>12}")
        print("-" * 61)
        for metric in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]:
            bv = b1.get(metric, 0)
            fv = f1.get(metric, 0)
            delta = fv - bv
            arrow = "+" if delta > 0 else ""
            print(f"{metric:<25} {bv:>12.4f} {fv:>12.4f} {arrow}{delta:>11.4f}")
        print(f"{'unparseable':<25} {b1.get('unparseable_responses', 0):>12} {f1.get('unparseable_responses', 0):>12}")

    # Stage 2
    if "stage2" in baseline and "stage2" in finetuned:
        b2 = baseline["stage2"]
        f2 = finetuned["stage2"]
        print("\n--- Stage 2: Distress Type Classification (RDD) ---")
        print(f"{'Metric':<25} {'Baseline':>12} {'Fine-Tuned':>12} {'Delta':>12}")
        print("-" * 61)
        for metric in ["primary_accuracy", "f1_macro", "exact_match_rate"]:
            bv = b2.get(metric, 0)
            fv = f2.get(metric, 0)
            delta = fv - bv
            arrow = "+" if delta > 0 else ""
            print(f"{metric:<25} {bv:>12.4f} {fv:>12.4f} {arrow}{delta:>11.4f}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Post-finetune evaluation + comparison")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="Base model name or path")
    parser.add_argument("--adapter-path", type=str, required=True,
                        help="Path to the LoRA adapter directory")
    parser.add_argument("--stage", type=str, default="both", choices=["1", "2", "both"])
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Run fine-tuned evaluation
    print("Running fine-tuned model evaluation...")
    model, processor = eval_module.load_model(args.model, args.adapter_path)

    all_results = {}
    if args.stage in ("1", "both"):
        r1 = eval_module.evaluate_stage1(model, processor, args.max_samples, tag="finetuned")
        if r1:
            all_results["stage1"] = r1

    if args.stage in ("2", "both"):
        r2 = eval_module.evaluate_stage2(model, processor, args.max_samples, tag="finetuned")
        if r2:
            all_results["stage2"] = r2

    # Save fine-tuned results
    ft_path = EVAL_DIR / "finetuned_results.json"
    with open(ft_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Fine-tuned results saved to: {ft_path}")

    # Load baseline for comparison
    baseline_path = EVAL_DIR / "baseline_results.json"
    if baseline_path.exists():
        with open(baseline_path, "r", encoding="utf-8") as f:
            baseline = json.load(f)
        print_comparison(baseline, all_results)
    else:
        print("\nWARNING: No baseline results found. Run 03_baseline_eval.py first.")
        print("Showing fine-tuned results only:")
        print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
