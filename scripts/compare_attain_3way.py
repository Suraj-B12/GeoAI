"""
3-way Attain WS_V2.0 comparison — definitive test of the hypothesis:

  "Baseline (no adapter) + sufficiently engineered prompts can match or exceed
   the LoRA-fine-tuned adapter on cross-dataset evaluation."

Loads:
  eval_results/attain_baseline_results.json      (Plain baseline: v1 prompts, no adapter)
  eval_results/attain_baseline_v2_results.json   (Improved Baseline: v2 prompts, no adapter)
  eval_results/attain_finetuned_results.json     (Fine-tuned:     v1 prompts + LoRA adapter)

Each represents one of three configurations on the same 769-image set:
  CONFIG A — Plain Baseline:    base model + v1 prompts (current production prompts)
  CONFIG B — Improved Baseline: base model + v2 prompts (deep persona + stakes + protocol)
  CONFIG C — Fine-tuned:        LoRA adapter + v1 prompts (current production)

Decomposes the total improvement A→C into:
  • Prompt-engineering contribution: A → B   (what improved prompts add over plain prompts)
  • Adapter contribution: B → C  (what the adapter adds on top of improved prompts; sign
    is meaningful — negative means adapter actively hurts vs improved baseline)

Outputs:
  eval_results/attain_3way_comparison.json
  eval_results/attain_3way_comparison.md       (paper-ready, all metrics, all per-class)
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "eval_results"


def load(name: str) -> dict:
    p = EVAL_DIR / name
    if not p.exists():
        sys.exit(f"ERROR: {p} not found")
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def _pred_freq(rows: list[dict], key="pred_pipeline") -> Counter:
    c: Counter = Counter()
    for r in rows:
        for t in r.get(key) or []:
            c[t] += 1
    return c


def _multi_pred_rate(rows: list[dict]) -> dict:
    sizes = Counter(len(r.get("pred_pipeline") or []) for r in rows)
    n = max(len(rows), 1)
    return {
        "0_preds": sizes.get(0, 0),
        "1_pred": sizes.get(1, 0),
        "2_preds": sizes.get(2, 0),
        "3_or_more": sum(v for k, v in sizes.items() if k >= 3),
        "multi_rate": sum(v for k, v in sizes.items() if k >= 2) / n,
        "avg_preds_per_distress_image": (
            sum(len(r.get("pred_pipeline") or []) for r in rows if r.get("is_distressed_pred"))
            / max(sum(1 for r in rows if r.get("is_distressed_pred")), 1)
        ),
    }


def _head_to_head(A: list[dict], B: list[dict]) -> dict:
    """Per-image win/loss: who recovered more of GT?"""
    Ax = {r["image"]: r for r in A}
    Bx = {r["image"]: r for r in B}
    common = set(Ax) & set(Bx)
    A_better = B_better = tie_some = tie_none = 0
    for img in common:
        gt = set(Ax[img].get("gt_attain") or [])
        a_correct = gt & set(Ax[img].get("pred_attain") or [])
        b_correct = gt & set(Bx[img].get("pred_attain") or [])
        if len(a_correct) > len(b_correct):
            A_better += 1
        elif len(b_correct) > len(a_correct):
            B_better += 1
        elif a_correct:
            tie_some += 1
        else:
            tie_none += 1
    return {
        "n_compared": len(common),
        "A_strictly_better": A_better,
        "B_strictly_better": B_better,
        "tied_both_correct_on_>=1": tie_some,
        "tied_both_wrong_or_partial": tie_none,
    }


def main():
    a = load("attain_baseline_results.json")       # v1 prompts, baseline
    b = load("attain_baseline_v2_results.json")    # v2 prompts, baseline
    c = load("attain_finetuned_results.json")      # v1 prompts, adapter

    # The 3 configurations
    configs = {
        "A_plain_baseline": a,
        "B_improved_baseline": b,
        "C_finetuned_adapter": c,
    }

    # =====================================================
    # Aggregate metrics
    # =====================================================
    agg = {}
    for name, d in configs.items():
        s = d["summary"]
        rows = d.get("per_image_results") or []
        agg[name] = {
            "tier1_in_dist_acc": s["tier1_in_distribution"]["accuracy"],
            "tier1_correct": s["tier1_in_distribution"]["correct"],
            "tier1_total": s["tier1_in_distribution"]["instances"],
            "tier2_zero_shot_acc": s["tier2_zero_shot"]["accuracy"],
            "tier2_correct": s["tier2_zero_shot"]["correct"],
            "tier2_total": s["tier2_zero_shot"]["instances"],
            "severity_acc": s["severity"]["accuracy"],
            "severity_correct": s["severity"]["correct"],
            "severity_total": s["severity"]["instances"],
            "stage1_distress_rate": sum(1 for r in rows if r.get("is_distressed_pred")) / max(len(rows), 1),
            **_multi_pred_rate(rows),
        }

    # Deltas
    A = agg["A_plain_baseline"]
    B = agg["B_improved_baseline"]
    C = agg["C_finetuned_adapter"]
    deltas = {}
    for k in ("tier1_in_dist_acc", "tier2_zero_shot_acc", "severity_acc"):
        deltas[k] = {
            "prompt_only_AtoB": round(B[k] - A[k], 4),
            "adapter_only_BtoC": round(C[k] - B[k], 4),
            "total_AtoC": round(C[k] - A[k], 4),
        }

    # =====================================================
    # Per-class TP/FP/FN/P/R/F1 across all three
    # =====================================================
    per_class_keys = set()
    for d in configs.values():
        per_class_keys.update(d["summary"]["per_class"].keys())
    per_class_keys = sorted(per_class_keys)
    per_class = {}
    for cls in per_class_keys:
        row = {"tier": None}
        for name, d in configs.items():
            pc = d["summary"]["per_class"].get(cls)
            if pc:
                row["tier"] = row["tier"] or pc.get("tier")
                row[name] = {
                    "tp": pc.get("tp", 0),
                    "fp": pc.get("fp", 0),
                    "fn": pc.get("fn", 0),
                    "precision": pc.get("precision", 0),
                    "recall": pc.get("recall", 0),
                    "f1": pc.get("f1", 0),
                }
            else:
                row[name] = {"tp": 0, "fp": 0, "fn": 0, "precision": 0, "recall": 0, "f1": 0}
        # F1 deltas
        row["f1_delta_prompt"] = round(row["B_improved_baseline"]["f1"]
                                       - row["A_plain_baseline"]["f1"], 4)
        row["f1_delta_adapter"] = round(row["C_finetuned_adapter"]["f1"]
                                        - row["B_improved_baseline"]["f1"], 4)
        row["f1_delta_total"] = round(row["C_finetuned_adapter"]["f1"]
                                      - row["A_plain_baseline"]["f1"], 4)
        per_class[cls] = row

    # =====================================================
    # Prediction frequency analysis (which labels does each emit?)
    # =====================================================
    freq_analysis = {}
    all_pipeline_labels = set()
    for d in configs.values():
        all_pipeline_labels.update(_pred_freq(d.get("per_image_results", [])).keys())
    for lbl in sorted(all_pipeline_labels):
        freq_analysis[lbl] = {}
        for name, d in configs.items():
            freq_analysis[lbl][name] = _pred_freq(d.get("per_image_results", [])).get(lbl, 0)

    # =====================================================
    # Head-to-head per-image (3 pairwise comparisons)
    # =====================================================
    h2h = {
        "A_vs_B_prompts_effect": _head_to_head(a["per_image_results"], b["per_image_results"]),
        "B_vs_C_adapter_effect": _head_to_head(b["per_image_results"], c["per_image_results"]),
        "A_vs_C_total_effect":   _head_to_head(a["per_image_results"], c["per_image_results"]),
    }

    # =====================================================
    # Verdict logic — does Improved Baseline >= Fine-tuned?
    # =====================================================
    improved_beats_ft_tier1 = B["tier1_in_dist_acc"] >= C["tier1_in_dist_acc"]
    improved_beats_ft_tier2 = B["tier2_zero_shot_acc"] >= C["tier2_zero_shot_acc"]
    improved_beats_ft_sev   = B["severity_acc"] >= C["severity_acc"]
    n_metrics_improved_wins = sum([improved_beats_ft_tier1, improved_beats_ft_tier2, improved_beats_ft_sev])

    verdict = {
        "hypothesis": "Improved Baseline (no adapter) >= Fine-tuned (with adapter)",
        "n_aggregate_metrics": 3,
        "n_metrics_improved_baseline_wins": n_metrics_improved_wins,
        "tier1_in_dist_improved_wins": improved_beats_ft_tier1,
        "tier2_zero_shot_improved_wins": improved_beats_ft_tier2,
        "severity_improved_wins": improved_beats_ft_sev,
        "h2h_improved_baseline_wins_per_image": h2h["B_vs_C_adapter_effect"]["A_strictly_better"],
        "h2h_finetuned_wins_per_image":         h2h["B_vs_C_adapter_effect"]["B_strictly_better"],
        "supported": n_metrics_improved_wins >= 2 and (
            h2h["B_vs_C_adapter_effect"]["A_strictly_better"]
            >= h2h["B_vs_C_adapter_effect"]["B_strictly_better"]
        ),
    }

    result = {
        "subset": "WS_V2.0",
        "n_images": 769,
        "aggregate": agg,
        "deltas": deltas,
        "per_class": per_class,
        "prediction_frequency": freq_analysis,
        "head_to_head": h2h,
        "verdict": verdict,
    }

    # =====================================================
    # Save
    # =====================================================
    out_json = EVAL_DIR / "attain_3way_comparison.json"
    out_md = EVAL_DIR / "attain_3way_comparison.md"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    out_md.write_text(_render_markdown(result), encoding="utf-8")

    # =====================================================
    # Print summary
    # =====================================================
    print()
    print("=" * 80)
    print("ATTAIN  WS_V2.0  —  3-way comparison")
    print("=" * 80)
    print()
    print("Configurations:")
    print(f"  A = Plain Baseline    (base model + v1 prompts, no adapter)")
    print(f"  B = Improved Baseline (base model + v2 prompts, no adapter)")
    print(f"  C = Fine-tuned        (LoRA adapter + v1 prompts)")
    print()
    print(f"{'Metric':30s} {'A':>10s} {'B':>10s} {'C':>10s} {'B-A':>10s} {'C-B':>10s}")
    print("-" * 80)
    for k, label in [
        ("tier1_in_dist_acc", "Tier 1 in-distribution"),
        ("tier2_zero_shot_acc", "Tier 2 zero-shot"),
        ("severity_acc", "Severity classification"),
        ("stage1_distress_rate", "Stage 1 distress rate"),
        ("multi_rate", "Multi-class pred rate"),
    ]:
        print(f"{label:30s} {A[k]:>9.2%} {B[k]:>9.2%} {C[k]:>9.2%} "
              f"{B[k]-A[k]:>+9.2%} {C[k]-B[k]:>+9.2%}")
    print()
    print(f"{'Per-class F1':30s} {'A_F1':>8s} {'B_F1':>8s} {'C_F1':>8s} "
          f"{'prompt':>9s} {'adapter':>9s} {'tier':>10s}")
    print("-" * 90)
    for cls in per_class_keys:
        row = per_class[cls]
        a_f1 = row["A_plain_baseline"]["f1"]
        b_f1 = row["B_improved_baseline"]["f1"]
        c_f1 = row["C_finetuned_adapter"]["f1"]
        print(f"{cls:30s} {a_f1:>8.3f} {b_f1:>8.3f} {c_f1:>8.3f} "
              f"{row['f1_delta_prompt']:>+9.3f} {row['f1_delta_adapter']:>+9.3f} "
              f"{row['tier']:>10s}")
    print()
    print("Head-to-head per-image:")
    print(f"  B vs C — Improved Baseline vs Fine-tuned (the prompt-vs-adapter test):")
    bc = h2h["B_vs_C_adapter_effect"]
    print(f"    Improved Baseline strictly better: {bc['A_strictly_better']}/769 ({bc['A_strictly_better']/7.69:.1f}%)")
    print(f"    Fine-tuned strictly better:        {bc['B_strictly_better']}/769 ({bc['B_strictly_better']/7.69:.1f}%)")
    print(f"    tied (both right on >=1):          {bc['tied_both_correct_on_>=1']}/769")
    print(f"    tied (both wrong/partial):         {bc['tied_both_wrong_or_partial']}/769")
    print()
    print("VERDICT:")
    print(f"  Hypothesis: {verdict['hypothesis']}")
    print(f"  Aggregate metrics where Improved Baseline >= Fine-tuned: "
          f"{verdict['n_metrics_improved_baseline_wins']}/3")
    print(f"    - Tier 1 in-dist:   Improved Baseline wins = {verdict['tier1_in_dist_improved_wins']}")
    print(f"    - Tier 2 zero-shot: Improved Baseline wins = {verdict['tier2_zero_shot_improved_wins']}")
    print(f"    - Severity:         Improved Baseline wins = {verdict['severity_improved_wins']}")
    print(f"  Head-to-head: Improved Baseline wins {bc['A_strictly_better']} vs Fine-tuned wins {bc['B_strictly_better']}")
    print(f"  SUPPORTED: {verdict['supported']}")
    print()
    print(f"JSON: {out_json}")
    print(f"MD:   {out_md}")


def _render_markdown(d: dict) -> str:
    L = []
    A = d["aggregate"]["A_plain_baseline"]
    B = d["aggregate"]["B_improved_baseline"]
    C = d["aggregate"]["C_finetuned_adapter"]
    v = d["verdict"]
    L.append("# Attain WS_V2.0 — 3-way comparison (Improved-Baseline vs Fine-tuned)")
    L.append("")
    L.append(f"**Subset:** WS_V2.0, {d['n_images']} images, ground-truth pavement distress only (EXCLUDE filter applied).")
    L.append("")
    L.append("## The three configurations")
    L.append("")
    L.append("| Config | Prompts | Adapter |")
    L.append("|---|---|---|")
    L.append("| **A** Plain Baseline    | v1 (current production prompts) | none |")
    L.append("| **B** Improved Baseline | v2 (deep persona + stakes + protocol + taxonomy) | none |")
    L.append("| **C** Fine-tuned        | v1 (current production prompts) | LoRA `adapters/v2-rdd-2epochs-20260507` |")
    L.append("")
    L.append("Decomposition of total improvement A→C:")
    L.append("  - Prompt-engineering contribution: **A → B** (improved prompts vs plain prompts)")
    L.append("  - Adapter contribution: **B → C** (what the adapter adds *on top of* the improved prompts — sign matters: negative means the adapter actively hurts vs improved baseline)")
    L.append("")
    L.append("## Headline accuracy")
    L.append("")
    L.append("| Metric | A: Plain Baseline | B: Improved Baseline | C: Fine-tuned | Prompt Δ (A→B) | Adapter Δ (B→C) |")
    L.append("|---|---|---|---|---|---|")
    for k, label in [
        ("tier1_in_dist_acc", "Tier 1 in-distribution (Linear, Alligator, Pothole)"),
        ("tier2_zero_shot_acc", "Tier 2 zero-shot (Block, Raveling, Weathering)"),
        ("severity_acc", "Severity classification"),
        ("stage1_distress_rate", "Stage 1 distress prediction rate"),
        ("multi_rate", "Multi-class prediction rate"),
    ]:
        L.append(f"| {label} | {A[k]:.2%} | {B[k]:.2%} | {C[k]:.2%} | "
                 f"**{B[k]-A[k]:+.2%}** | **{C[k]-B[k]:+.2%}** |")
    L.append("")

    L.append("## Per-class F1 — all three configurations")
    L.append("")
    L.append("| Class | Tier | A: Plain | B: Improved | C: Fine-tuned | Prompt Δ | Adapter Δ |")
    L.append("|---|---|---|---|---|---|---|")
    for cls in sorted(d["per_class"].keys()):
        r = d["per_class"][cls]
        L.append(f"| {cls} | {r['tier']} "
                 f"| {r['A_plain_baseline']['f1']:.3f} "
                 f"| {r['B_improved_baseline']['f1']:.3f} "
                 f"| {r['C_finetuned_adapter']['f1']:.3f} "
                 f"| {r['f1_delta_prompt']:+.3f} "
                 f"| {r['f1_delta_adapter']:+.3f} |")
    L.append("")

    L.append("## Per-class TP / FP / FN")
    L.append("")
    L.append("| Class | A: Plain TP/FP/FN | B: Improved TP/FP/FN | C: Fine-tuned TP/FP/FN |")
    L.append("|---|---|---|---|")
    for cls in sorted(d["per_class"].keys()):
        r = d["per_class"][cls]
        L.append(f"| {cls} "
                 f"| {r['A_plain_baseline']['tp']}/{r['A_plain_baseline']['fp']}/{r['A_plain_baseline']['fn']} "
                 f"| {r['B_improved_baseline']['tp']}/{r['B_improved_baseline']['fp']}/{r['B_improved_baseline']['fn']} "
                 f"| {r['C_finetuned_adapter']['tp']}/{r['C_finetuned_adapter']['fp']}/{r['C_finetuned_adapter']['fn']} |")
    L.append("")

    L.append("## Pipeline-label emission counts (out of 769 images)")
    L.append("")
    L.append("Reveals which classes each configuration prefers to emit:")
    L.append("")
    L.append("| Pipeline label | A: Plain Baseline | B: Improved Baseline | C: Fine-tuned |")
    L.append("|---|---|---|---|")
    fq = d["prediction_frequency"]
    sorted_labels = sorted(fq.keys(),
                           key=lambda k: -max(fq[k]["A_plain_baseline"],
                                              fq[k]["B_improved_baseline"],
                                              fq[k]["C_finetuned_adapter"]))
    for lbl in sorted_labels[:20]:
        L.append(f"| {lbl} "
                 f"| {fq[lbl]['A_plain_baseline']} "
                 f"| {fq[lbl]['B_improved_baseline']} "
                 f"| {fq[lbl]['C_finetuned_adapter']} |")
    L.append("")

    L.append("## Head-to-head per-image")
    L.append("")
    L.append("Each pairwise comparison: who recovered MORE of the ground-truth classes on each image.")
    L.append("")
    for k, label in [
        ("A_vs_B_prompts_effect", "**A vs B** — prompt-engineering effect (Plain Baseline vs Improved Baseline, both no adapter)"),
        ("B_vs_C_adapter_effect", "**B vs C** — the prompt-vs-adapter test (Improved Baseline vs Fine-tuned)"),
        ("A_vs_C_total_effect",   "**A vs C** — total fine-tune-plus-prompts effect (Plain Baseline vs Fine-tuned)"),
    ]:
        h = d["head_to_head"][k]
        L.append(f"### {label}")
        L.append("")
        L.append(f"- A strictly better: **{h['A_strictly_better']}** ({h['A_strictly_better']/769:.1%})")
        L.append(f"- B strictly better: **{h['B_strictly_better']}** ({h['B_strictly_better']/769:.1%})")
        L.append(f"- Tied (both correct on ≥1 class): {h['tied_both_correct_on_>=1']}")
        L.append(f"- Tied (both wrong/partial): {h['tied_both_wrong_or_partial']}")
        L.append("")

    L.append("## Hypothesis — final verdict")
    L.append("")
    L.append(f"**Hypothesis being tested:** {v['hypothesis']}")
    L.append("")
    L.append(f"- Aggregate metrics where Improved Baseline ≥ Fine-tuned: **{v['n_metrics_improved_baseline_wins']} / 3**")
    L.append(f"  - Tier 1 in-distribution: Improved Baseline wins = `{v['tier1_in_dist_improved_wins']}`")
    L.append(f"  - Tier 2 zero-shot: Improved Baseline wins = `{v['tier2_zero_shot_improved_wins']}`")
    L.append(f"  - Severity: Improved Baseline wins = `{v['severity_improved_wins']}`")
    L.append(f"- Head-to-head per-image: Improved Baseline wins **{v['h2h_improved_baseline_wins_per_image']}** "
             f"vs Fine-tuned wins **{v['h2h_finetuned_wins_per_image']}**")
    L.append("")
    L.append(f"**Hypothesis supported overall:** `{v['supported']}`")
    L.append("")

    L.append("## How to read this")
    L.append("")
    L.append("- **Prompt Δ positive** → improved prompts help on this metric, regardless of adapter.")
    L.append("- **Adapter Δ positive** → adapter still adds value *even with the strongest prompts*. "
             "The improvement is real and not just confounded by prompts.")
    L.append("- **Adapter Δ negative** → adapter actively hurts when improved prompts are already in play. "
             "On these classes, the Improved Baseline pathway is strictly better.")
    L.append("- **Class-level signal trumps aggregate signal for routing decisions.** Even if the adapter "
             "wins on aggregate, it might be the wrong choice for specific class subsets — which is the "
             "argument for a hybrid router architecture.")
    L.append("")

    return "\n".join(L)


if __name__ == "__main__":
    main()
