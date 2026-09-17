"""
A/B comparison of two VLMs evaluated on the same Attain images.

Takes two result JSONs produced by scripts/07_cross_dataset_eval.py and emits a
paper-ready comparison (markdown + JSON): tier accuracies, per-class F1, label
emission frequencies, Stage 1 detection rate, multi-class rate, and per-image
head-to-head.

A/B VALIDITY GUARD
------------------
The script refuses to compare two runs that differ in anything other than the
model — prompts version, subset, sample count, or adapter. A comparison across
different prompts would attribute a prompt effect to the model, which is
exactly the confound this whole exercise is meant to avoid. Override with
--force only if you know why the mismatch is acceptable and say so in the
writeup.

Usage:
    venv/Scripts/python.exe scripts/compare_model_ab.py \
        --a attain_ab_qwen25vl7b_100.json \
        --b attain_ab_qwen3vl8b_100.json \
        --label-a "Qwen2.5-VL-7B" --label-b "Qwen3-VL-8B" \
        --out attain_model_ab
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils import EVAL_DIR  # noqa: E402


def load(name: str) -> dict:
    p = Path(name)
    if not p.is_absolute():
        p = EVAL_DIR / name
    if not p.exists():
        sys.exit(f"ERROR: {p} not found")
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def _pred_freq(rows: list[dict], key: str = "pred_pipeline") -> Counter:
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
        "multi_rate": round(sum(v for k, v in sizes.items() if k >= 2) / n, 4),
    }


def _stage1_rate(rows: list[dict]) -> float:
    n = max(len(rows), 1)
    return round(sum(1 for r in rows if r.get("is_distressed_pred")) / n, 4)


def _head_to_head(A: list[dict], B: list[dict]) -> dict:
    """Per-image win/loss: which model recovered more ground-truth classes."""
    Ax = {r["image"]: r for r in A}
    Bx = {r["image"]: r for r in B}
    common = set(Ax) & set(Bx)
    a_better = b_better = tie_some = tie_none = 0
    examples: list[dict] = []
    for img in sorted(common):
        gt = set(Ax[img].get("gt_attain") or [])
        a_correct = gt & set(Ax[img].get("pred_attain") or [])
        b_correct = gt & set(Bx[img].get("pred_attain") or [])
        if len(a_correct) > len(b_correct):
            a_better += 1
            verdict = "A"
        elif len(b_correct) > len(a_correct):
            b_better += 1
            verdict = "B"
        elif a_correct:
            tie_some += 1
            verdict = "tie(both)"
        else:
            tie_none += 1
            verdict = "tie(neither)"
        if verdict in ("A", "B") and len(examples) < 12:
            examples.append({
                "image": img,
                "winner": verdict,
                "gt": sorted(gt),
                "A_pred": sorted(set(Ax[img].get("pred_attain") or [])),
                "B_pred": sorted(set(Bx[img].get("pred_attain") or [])),
            })
    return {
        "n_compared": len(common),
        "A_strictly_better": a_better,
        "B_strictly_better": b_better,
        "tied_both_correct_on_at_least_1": tie_some,
        "tied_both_wrong_or_partial": tie_none,
        "examples": examples,
    }


def _check_comparable(a: dict, b: dict, force: bool) -> list[str]:
    """Refuse to compare runs that differ in more than the model."""
    sa, sb = a["summary"], b["summary"]
    problems = []
    for field in ("subset", "prompts_version", "adapter_path", "n_images"):
        if sa.get(field) != sb.get(field):
            problems.append(
                f"{field}: A={sa.get(field)!r} vs B={sb.get(field)!r}")
    if sa.get("model_path") and sa.get("model_path") == sb.get("model_path"):
        problems.append(
            f"both runs used the SAME model ({sa.get('model_path')}) — "
            f"there is nothing to compare")
    if problems and not force:
        print("ERROR: these two runs are not a valid A/B pair:")
        for p in problems:
            print(f"  - {p}")
        print("\nA comparison across differing conditions would attribute the "
              "difference to the model when it may be caused by the other "
              "factor. Re-run both evals under identical settings, or pass "
              "--force if the mismatch is understood and will be disclosed.")
        sys.exit(2)
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="baseline result JSON")
    ap.add_argument("--b", required=True, help="candidate result JSON")
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    ap.add_argument("--out", default="attain_model_ab",
                    help="output basename (writes .json and .md into eval_results/)")
    ap.add_argument("--force", action="store_true",
                    help="compare even when the runs differ in more than the model")
    args = ap.parse_args()

    a, b = load(args.a), load(args.b)
    mismatches = _check_comparable(a, b, args.force)

    sa, sb = a["summary"], b["summary"]
    label_a = args.label_a or sa.get("model_path") or "A"
    label_b = args.label_b or sb.get("model_path") or "B"
    ra, rb = a["per_image_results"], b["per_image_results"]

    per_class_keys = sorted(set(sa["per_class"]) | set(sb["per_class"]))
    per_class = {}
    for cls in per_class_keys:
        pa = sa["per_class"].get(cls) or {}
        pb = sb["per_class"].get(cls) or {}
        per_class[cls] = {
            "tier": pa.get("tier") or pb.get("tier"),
            "A": {k: pa.get(k) for k in ("f1", "precision", "recall", "tp", "fp", "fn")},
            "B": {k: pb.get(k) for k in ("f1", "precision", "recall", "tp", "fp", "fn")},
            "f1_delta": round((pb.get("f1") or 0) - (pa.get("f1") or 0), 4),
        }

    out = {
        "comparison": f"{label_a} vs {label_b}",
        "validity": {
            "identical_conditions": not mismatches,
            "mismatches": mismatches,
            "subset": sa.get("subset"),
            "n_images": sa.get("n_images"),
            "prompts_version": sa.get("prompts_version"),
            "adapter_path": sa.get("adapter_path"),
        },
        "models": {
            "A": {k: sa.get(k) for k in
                  ("model_path", "model_family", "model_class", "quantization_bits",
                   "attn_implementation", "max_pixels", "oom_fallback_used")},
            "B": {k: sb.get(k) for k in
                  ("model_path", "model_family", "model_class", "quantization_bits",
                   "attn_implementation", "max_pixels", "oom_fallback_used")},
        },
        "headline": {
            "tier1_in_distribution": {
                "A": sa["tier1_in_distribution"]["accuracy"],
                "B": sb["tier1_in_distribution"]["accuracy"],
                "delta_pp": round((sb["tier1_in_distribution"]["accuracy"]
                                   - sa["tier1_in_distribution"]["accuracy"]) * 100, 2),
            },
            "tier2_zero_shot": {
                "A": sa["tier2_zero_shot"]["accuracy"],
                "B": sb["tier2_zero_shot"]["accuracy"],
                "delta_pp": round((sb["tier2_zero_shot"]["accuracy"]
                                   - sa["tier2_zero_shot"]["accuracy"]) * 100, 2),
            },
            "severity": {
                "A": sa["severity"]["accuracy"],
                "B": sb["severity"]["accuracy"],
                "delta_pp": round((sb["severity"]["accuracy"]
                                   - sa["severity"]["accuracy"]) * 100, 2),
            },
            "stage1_distress_rate": {
                "A": _stage1_rate(ra), "B": _stage1_rate(rb),
                "delta_pp": round((_stage1_rate(rb) - _stage1_rate(ra)) * 100, 2),
            },
            "multi_class_rate": {
                "A": _multi_pred_rate(ra)["multi_rate"],
                "B": _multi_pred_rate(rb)["multi_rate"],
                "delta_pp": round((_multi_pred_rate(rb)["multi_rate"]
                                   - _multi_pred_rate(ra)["multi_rate"]) * 100, 2),
            },
        },
        "per_class": per_class,
        "label_emissions": {
            "A": dict(_pred_freq(ra).most_common()),
            "B": dict(_pred_freq(rb).most_common()),
        },
        "head_to_head": _head_to_head(ra, rb),
    }

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    json_path = EVAL_DIR / f"{args.out}.json"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    md = _render_markdown(out, label_a, label_b)
    md_path = EVAL_DIR / f"{args.out}.md"
    md_path.write_text(md, encoding="utf-8")

    # The markdown contains non-cp1252 characters (Δ, ≥). Files are written as
    # UTF-8 and are fine; only the Windows console encoding is the problem, so
    # degrade the CONSOLE copy rather than failing after the work is done.
    try:
        print(md)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "ascii"
        print(md.encode(enc, errors="replace").decode(enc, errors="replace"))
        print(f"\n[note] console encoding is {enc}; some characters were replaced "
              f"above. The written .md file is full UTF-8.")
    print(f"\nWritten: {json_path}\n         {md_path}")
    return 0


def _render_markdown(d: dict, la: str, lb: str) -> str:
    v = d["validity"]
    L = []
    L.append(f"# Model A/B — {la} vs {lb}")
    L.append("")
    L.append(f"**Dataset:** Attain {v['subset']}, {v['n_images']} images · "
             f"**Prompts:** {v['prompts_version']} · "
             f"**Adapter:** {v['adapter_path'] or 'none'}")
    L.append("")
    if v["identical_conditions"]:
        L.append("Both runs used identical prompts, subset, sample count and adapter "
                 "setting — the only difference is the model, so the deltas below are "
                 "attributable to the model.")
    else:
        L.append("> **WARNING — conditions differ between runs:** "
                 + "; ".join(v["mismatches"])
                 + ". Deltas below are NOT purely model effects.")
    L.append("")
    L.append("| | A | B |")
    L.append("|---|---|---|")
    for k in ("model_path", "model_family", "quantization_bits", "attn_implementation",
              "max_pixels"):
        L.append(f"| {k} | {d['models']['A'].get(k)} | {d['models']['B'].get(k)} |")
    L.append("")
    L.append("## Headline metrics")
    L.append("")
    L.append(f"| Metric | {la} | {lb} | Δ (pp) |")
    L.append("|---|---|---|---|")
    names = {
        "tier1_in_distribution": "Tier 1 in-distribution accuracy",
        "tier2_zero_shot": "Tier 2 zero-shot accuracy",
        "severity": "Severity accuracy",
        "stage1_distress_rate": "Stage 1 distress prediction rate",
        "multi_class_rate": "Multi-class prediction rate",
    }
    for key, name in names.items():
        h = d["headline"][key]
        L.append(f"| {name} | {h['A']*100:.2f}% | {h['B']*100:.2f}% | "
                 f"**{h['delta_pp']:+.2f}** |")
    L.append("")
    L.append("## Per-class F1")
    L.append("")
    L.append(f"| Class | Tier | {la} | {lb} | Δ |")
    L.append("|---|---|---|---|---|")
    for cls in sorted(d["per_class"]):
        r = d["per_class"][cls]
        L.append(f"| {cls} | {r['tier']} | {r['A'].get('f1')} | {r['B'].get('f1')} | "
                 f"{r['f1_delta']:+.3f} |")
    L.append("")
    L.append("## Per-class TP / FP / FN")
    L.append("")
    L.append(f"| Class | {la} TP/FP/FN | {lb} TP/FP/FN |")
    L.append("|---|---|---|")
    for cls in sorted(d["per_class"]):
        r = d["per_class"][cls]
        L.append(f"| {cls} | {r['A'].get('tp')}/{r['A'].get('fp')}/{r['A'].get('fn')} "
                 f"| {r['B'].get('tp')}/{r['B'].get('fp')}/{r['B'].get('fn')} |")
    L.append("")
    L.append("## Label emissions")
    L.append("")
    keys = sorted(set(d["label_emissions"]["A"]) | set(d["label_emissions"]["B"]))
    L.append(f"| Pipeline label | {la} | {lb} |")
    L.append("|---|---|---|")
    for k in keys:
        L.append(f"| {k} | {d['label_emissions']['A'].get(k, 0)} "
                 f"| {d['label_emissions']['B'].get(k, 0)} |")
    L.append("")
    h = d["head_to_head"]
    L.append("## Head-to-head (per image)")
    L.append("")
    L.append(f"- Images compared: **{h['n_compared']}**")
    L.append(f"- {la} strictly better: **{h['A_strictly_better']}**")
    L.append(f"- {lb} strictly better: **{h['B_strictly_better']}**")
    L.append(f"- Tied, both correct on >=1 class: {h['tied_both_correct_on_at_least_1']}")
    L.append(f"- Tied, both wrong/partial: {h['tied_both_wrong_or_partial']}")
    if h["examples"]:
        L.append("")
        L.append("### Sample disagreements")
        L.append("")
        L.append("| Image | Winner | Ground truth | A predicted | B predicted |")
        L.append("|---|---|---|---|---|")
        for e in h["examples"]:
            L.append(f"| {e['image']} | {e['winner']} | {', '.join(e['gt']) or '—'} "
                     f"| {', '.join(e['A_pred']) or '—'} "
                     f"| {', '.join(e['B_pred']) or '—'} |")
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
