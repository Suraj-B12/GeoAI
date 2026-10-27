"""
Paper figures for sections 6.5-6.9, regenerated from the stored result files.

No inference: every number is read from eval_results/*.json, so the figures
always match the tables in paper/baseline_evaluation.md.

Output: eval_results/plots/paper/fig_*.png (200 dpi)

Usage:
    venv/Scripts/python.exe scripts/generate_paper_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL = PROJECT_ROOT / "eval_results"
OUT = EVAL / "plots" / "paper"

# Validated categorical slots 1-2 (dataviz reference palette, light mode).
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1c1917", "#78716c", "#e7e5e4"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
})


def load(name: str) -> dict:
    return json.loads((EVAL / name).read_text(encoding="utf-8"))


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print("  wrote", (OUT / name).relative_to(PROJECT_ROOT))


def chance(ax, horizontal=True):
    (ax.axvline if horizontal else ax.axhline)(0.5, color=MUTED, lw=1, ls="--")


def fig_confidence_auc():
    """6.5: field vs sequence AUC by correctness rule (Attain n=407)."""
    m = load("calib_attain_407.json")["metrics"]["attain"]
    rules = [("no_false_positives", "No false positives"), ("any_overlap", "Any overlap"),
             ("jaccard_50", "Jaccard ≥ 0.5"), ("exact_match", "Exact set (15 correct)")]
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    h = 0.36
    for i, (k, lab) in enumerate(rules):
        for j, (metric, col, name) in enumerate([("field", BLUE, "Field-restricted"),
                                                  ("sequence", ORANGE, "Whole sequence")]):
            a, se = m[k][metric]["auc"], m[k][metric]["auc_stderr"]
            y = i + (j - 0.5) * h
            ax.barh(y, a, height=h - 0.04, color=col, label=name if i == 0 else None)
            ax.errorbar(a, y, xerr=se, color=INK, lw=0.8, capsize=2)
            ax.text(a + se + 0.01, y, f"{a:.2f}", va="center", fontsize=7.5, color=INK)
    chance(ax)
    ax.set_yticks(range(len(rules)), [r[1] for r in rules])
    ax.invert_yaxis()
    ax.set_xlim(0.2, 0.9)
    ax.set_xlabel("AUC (± Hanley–McNeil SE); dashed line = chance")
    ax.set_title("Stage 2 confidence: does it rank right answers above wrong ones?",
                 loc="left", fontsize=9.5, color=INK)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8)
    save(fig, "fig_6_5_confidence_auc.png")


def fig_resolution():
    """6.6: worker time, peak memory, Attain accuracy by input cap."""
    t = load("resolution_timing.json")["summary"]
    caps = ["768", "1024", "1280", "native"]
    labels = ["768²", "1024²", "1280²", "Full (1736²)"]
    secs = [t[c]["median_total_s"] for c in caps]
    mem = [t[c]["median_peak_reserved_gb"] for c in caps]
    a = load("resolution_ab.json")["analysis"]["attain"]
    acc_caps = [c for c in ["native", "1280", "1024", "768", "640", "512"] if c in a]
    acc = [a[c]["no_false_positives"]["acc_level_pct"] for c in acc_caps]

    fig, axs = plt.subplots(1, 3, figsize=(8.6, 2.7))
    ax = axs[0]
    ax.bar(labels, secs, color=[BLUE if c == "1024" else "#86b6ef" for c in caps], width=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("Seconds per photo (log)")
    ax.set_title("Whole-worker time", loc="left", fontsize=9, color=INK)
    for x, s in zip(labels, secs):
        ax.text(x, s * 1.12, f"{s:.0f} s", ha="center", fontsize=7.5, color=INK)
    ax = axs[1]
    ax.bar(labels, mem, color=[BLUE if c == "1024" else "#86b6ef" for c in caps], width=0.6)
    ax.axhline(24, color=ORANGE, lw=1.2)
    ax.text(-0.45, 25.0, "24 GB card", ha="left", fontsize=7.5, color=INK)
    ax.set_ylabel("Peak reserved GPU memory (GB)")
    ax.set_title("Peak memory", loc="left", fontsize=9, color=INK)
    ax = axs[2]
    ax.plot([c if c == "native" else c + "²" for c in acc_caps], acc, color=BLUE, lw=2,
            marker="o", ms=5)
    ax.set_ylim(70, 100)
    ax.set_ylabel("Attain accuracy, no false positives (%)")
    ax.set_title("Accuracy (n = 200, paired)", loc="left", fontsize=9, color=INK)
    for ax in axs:
        ax.tick_params(axis="x", labelsize=7.5)
    save(fig, "fig_6_6_resolution.png")


def fig_first_label():
    """6.7: first-label vs whole-field confidence; production review load."""
    p = load("primary_confidence.json")["primary_auc"]
    g = load("production_gate_comparison.json")
    fig, axs = plt.subplots(1, 2, figsize=(8.2, 2.6), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axs[0]
    names = [("primary (joint)", "First label, joint probability"),
             ("primary (geomean)", "First label, geometric mean"),
             ("field (all labels)", "Whole field (current gate)")]
    for i, (k, lab) in enumerate(names):
        a, se = p[k]["auc"], p[k]["se"]
        col = BLUE if k.startswith("field") else ORANGE
        ax.barh(i, a, height=0.6, color=col)
        ax.errorbar(a, i, xerr=se, color=INK, lw=0.8, capsize=2)
        ax.text(a + se + 0.01, i, f"{a:.3f}", va="center", fontsize=7.5, color=INK)
    chance(ax)
    ax.set_yticks(range(3), [n[1] for n in names])
    ax.invert_yaxis()
    ax.set_xlim(0.3, 0.8)
    ax.set_xlabel("AUC for first-label correctness (Attain, n = 407)")
    ax.set_title("Which score predicts a right first label?", loc="left", fontsize=9, color=INK)

    ax = axs[1]
    groups = [("All", g["n"], g["review_load"]),
              ("Single label", g["single_label"]["n"], g["single_label"]),
              ("Multi-label", g["multi_label"]["n"], g["multi_label"])]
    w = 0.36
    for i, (lab, n, d) in enumerate(groups):
        for j, (k, col, name) in enumerate([("field", BLUE, "Whole-field gate"),
                                            ("primary", ORANGE, "First-label gate")]):
            v = 100 * d[k] / n
            ax.bar(i + (j - 0.5) * w, v, width=w - 0.04, color=col,
                   label=name if i == 0 else None)
            ax.text(i + (j - 0.5) * w, v + 2, f"{v:.0f}%", ha="center", fontsize=7, color=INK)
    ax.set_xticks(range(3), [f"{x[0]}\n(n = {x[1]})" for x in groups])
    ax.set_ylim(0, 110)
    ax.set_ylabel("Sent to review at 0.80 (%)")
    ax.set_title("Bengaluru uploads", loc="left", fontsize=9, color=INK)
    ax.legend(fontsize=7.5, loc="upper left")
    save(fig, "fig_6_7_first_label.png")


PROBE_CLASSES = ["Linear crack", "Alligator crack", "Pothole", "Raveling", "Weathering",
                 "Patch and utility cut"]


def _short(c: str) -> str:
    return c.replace(" and utility cut", "")


def fig_probe_mcc():
    """6.8a: per-class MCC, free-form list vs per-type probe, test split, with CI."""
    T = load("stage2_probe_report.json")["test"]
    probe = next(k for k in T if k.startswith("probe_"))
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    for off, key, col, lab in ((-0.2, "v0_production", ORANGE, "Free-form list (production)"),
                               (0.2, probe, BLUE, "Per-type probe")):
        pts = [T[key]["ci"]["per_class_mcc"][c] for c in PROBE_CLASSES]
        xs = [i + off for i in range(len(PROBE_CLASSES))]
        ax.bar(xs, [p["point"] for p in pts], width=0.38, color=col, label=lab)
        ax.errorbar(xs, [p["point"] for p in pts],
                    yerr=[[p["point"] - p["lo"] for p in pts], [p["hi"] - p["point"] for p in pts]],
                    fmt="none", ecolor=INK, capsize=2.5, lw=0.9)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(range(len(PROBE_CLASSES)), [_short(c) for c in PROBE_CLASSES])
    ax.set_ylabel("MCC (0 = constant predictor)")
    ax.set_title("Per-class MCC, Attain test split (95% block-bootstrap CI)", loc="left",
                 fontsize=9.5, color=INK)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    save(fig, "fig_6_8a_probe_mcc.png")


def fig_probe_operating_points():
    """6.8b: recall and false-positive rate per class at the operating point, list vs probe."""
    T = load("stage2_probe_report.json")["test"]
    probe = next(k for k in T if k.startswith("probe_"))
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9), sharey=True)
    for ax, (metric, title) in zip(axes, (("recall", "Recall (frames with the type)"),
                                          ("fpr", "False-positive rate (frames without)"))):
        for i, c in enumerate(PROBE_CLASSES):
            vals = []
            for key in ("v0_production", probe):
                pc = T[key]["per_class"][c]
                vals.append(pc["recall"] if metric == "recall" else 1 - pc["specificity"])
            ax.plot(vals, [i, i], color=GRID, lw=2, zorder=1)
            ax.scatter(vals[0], i, color=ORANGE, s=28, zorder=2, label="Free-form list" if i == 0 else None)
            ax.scatter(vals[1], i, color=BLUE, s=28, zorder=2, label="Per-type probe" if i == 0 else None)
        ax.set_xlim(-0.03, 1.03)
        ax.set_title(title, loc="left", fontsize=9, color=INK)
        ax.set_xlabel("Share of frames")
    auc = [T[probe]["per_class"][c]["auroc"] for c in PROBE_CLASSES]
    axes[0].set_yticks(range(len(PROBE_CLASSES)),
                       [f"{_short(c)} (AUROC {a:.2f})" for c, a in zip(PROBE_CLASSES, auc)])
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=7.5, loc="lower right")
    save(fig, "fig_6_8b_probe_operating_points.png")


def fig_views():
    """6.9: probe macro AUROC change vs the full photo, test split, with CI."""
    v = load("stage2_views_report.json")["test"]
    rows = [("tile_up+full", "Tiles (upscaled) + full"), ("zoomavgmax", "Find-then-zoom"),
            ("oracleavgmax", "Oracle crops (ceiling)")]
    fig, ax = plt.subplots(figsize=(5.6, 2.2))
    for i, (k, lab) in enumerate(rows):
        d = v[k]["delta_vs_full"]
        ax.errorbar(d["delta"], i, xerr=[[d["delta"] - d["lo"]], [d["hi"] - d["delta"]]],
                    fmt="o", color=BLUE, ms=6, lw=1.2, capsize=3)
        ax.text(d["hi"] + 0.004, i, f"{d['delta']:+.3f}  [{d['lo']:+.3f}, {d['hi']:+.3f}]",
                va="center", fontsize=7.5, color=INK)
    ax.axvline(0, color=MUTED, lw=1, ls="--")
    ax.set_yticks(range(len(rows)), [r[1] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(-0.06, 0.16)
    ax.set_xticks([-0.05, 0, 0.05, 0.10])
    ax.set_xlabel("Macro AUROC minus full photo\n(Attain test split, block-bootstrap 95% CI)")
    ax.set_title("Where the probe looks", loc="left", fontsize=9.5, color=INK)
    save(fig, "fig_6_9_views.png")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig_confidence_auc, fig_resolution, fig_first_label, fig_probe_mcc,
              fig_probe_operating_points, fig_views):
        f()
    return 0


if __name__ == "__main__":
    sys.exit(main())
