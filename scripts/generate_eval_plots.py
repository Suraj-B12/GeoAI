"""
Generate paper-quality plots for all evaluation results.

Loads everything from eval_results/ (gracefully skips missing files), produces
PNGs at 300 DPI under eval_results/plots/{rdd,bengaluru,attain}/, and writes
INDEX.md with embedded thumbnails for easy review.

Conventions:
  - Baseline color (teal):  #1b9e77
  - Fine-tuned color (red): #d62728
  - Improvement (green):    #2ca02c
  - Regression (red):       #e41a1c
  - Neutral (gray):         #7f7f7f

  - figsize: (8, 5) default, (12, 6) for long category bars, (10, 8) for heatmaps
  - Tick labels rotated 30° right-aligned for any string label > 10 chars
  - Title at top, axis labels with units, legend in 'best' position
  - Tight layout, 300 DPI, white background
  - Annotations on bars (counts/percentages) where it adds clarity

Run:
    python scripts/generate_eval_plots.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "eval_results"
PLOTS_DIR = EVAL_DIR / "plots"

# Color palette
C_BASE = "#1b9e77"
C_FT = "#d62728"
C_GAIN = "#2ca02c"
C_LOSS = "#e41a1c"
C_NEUT = "#7f7f7f"

# Matplotlib defaults
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ============================================================
# Helpers
# ============================================================

def load_json(name: str) -> Optional[dict]:
    p = EVAL_DIR / name
    if not p.exists():
        print(f"  [skip] {name} not found")
        return None
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def annotate_bars(ax, bars, fmt="{:.0f}", fontsize=9, offset=0.005):
    """Place text labels above each bar."""
    ymax = ax.get_ylim()[1]
    for b in bars:
        h = b.get_height()
        if h == 0:
            continue
        ax.text(b.get_x() + b.get_width() / 2, h + offset * ymax,
                fmt.format(h), ha="center", va="bottom", fontsize=fontsize)


def short(label: str, n: int = 22) -> str:
    return label if len(label) <= n else label[: n - 1] + "…"


# ============================================================
# RDD plots — baseline vs fine-tuned (in-distribution)
# ============================================================

def plot_rdd_stage1_accuracy(ab: dict):
    """Stage 1 accuracy + Distress recall: baseline vs fine-tuned."""
    s1b = ab["stage1"]["baseline"]
    s1f = ab["stage1"]["finetuned"]
    metrics = ["Accuracy", "Distress recall", "Distress precision", "Distress F1"]
    base = [
        s1b["accuracy"],
        s1b["classification_report"]["Distress"]["recall"],
        s1b["classification_report"]["Distress"]["precision"],
        s1b["classification_report"]["Distress"]["f1-score"],
    ]
    ft = [
        s1f["accuracy"],
        s1f["classification_report"]["Distress"]["recall"],
        s1f["classification_report"]["Distress"]["precision"],
        s1f["classification_report"]["Distress"]["f1-score"],
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(metrics))
    w = 0.38
    b1 = ax.bar(x - w/2, base, w, label="Baseline (no adapter)", color=C_BASE)
    b2 = ax.bar(x + w/2, ft, w, label="Fine-tuned (LoRA)", color=C_FT)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.set_title("RDD Stage 1 — Binary distress detection (10,000 images)")
    annotate_bars(ax, b1, fmt="{:.1%}")
    annotate_bars(ax, b2, fmt="{:.1%}")
    ax.legend(loc="upper right")
    fig.tight_layout()
    out = PLOTS_DIR / "rdd" / "stage1_accuracy.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_rdd_stage2_per_class(ab: dict):
    """Per-class recall: baseline vs fine-tuned (Stage 2 RDD types)."""
    pc = ab["stage2"]["per_class"]
    classes = [c for c in pc.keys() if c != "Other Distress" and c != "Unparseable/Normal"]
    classes.sort()
    base_r = [pc[c]["baseline_recall"] for c in classes]
    ft_r = [pc[c]["finetuned_recall"] for c in classes]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(classes))
    w = 0.38
    b1 = ax.bar(x - w/2, base_r, w, label="Baseline", color=C_BASE)
    b2 = ax.bar(x + w/2, ft_r, w, label="Fine-tuned", color=C_FT)
    ax.set_xticks(x)
    ax.set_xticklabels([short(c, 28) for c in classes], rotation=15, ha="right")
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.set_title("RDD Stage 2 — Per-class recall (5,758 images)")
    annotate_bars(ax, b1, fmt="{:.1%}", offset=0.015)
    annotate_bars(ax, b2, fmt="{:.1%}", offset=0.015)
    ax.legend(loc="upper left")
    fig.tight_layout()
    out = PLOTS_DIR / "rdd" / "stage2_per_class_recall.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_rdd_stage1_confusion(ab: dict):
    """Side-by-side confusion matrices for Stage 1."""
    cm_b = np.array(ab["stage1"]["baseline"]["confusion_matrix"])
    cm_f = np.array(ab["stage1"]["finetuned"]["confusion_matrix"])
    labels = ["Normal", "Distress"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, cm, title in zip(axes, [cm_b, cm_f], ["Baseline", "Fine-tuned"]):
        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground truth")
        ax.set_title(title)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color=color, fontsize=11)
        ax.grid(False)
    fig.suptitle("RDD Stage 1 — Confusion matrices", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = PLOTS_DIR / "rdd" / "stage1_confusion_matrices.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_rdd_promotion_summary(ab: dict):
    """Headline A/B promotion gate metrics."""
    dec = ab["decision"]
    metrics = ["Stage 1\naccuracy Δ", "Stage 2\nmacro F1 Δ", "Normal-class\nprecision Δ"]
    deltas = [dec["s1_acc_delta"], dec["s2_f1_delta"], dec["normal_prec_delta"]]
    thresholds = [3.0, 10.0, -3.0]  # third one is a tolerance band, not a min

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [C_GAIN if d >= t else C_LOSS for d, t in zip(deltas, thresholds)]
    bars = ax.bar(metrics, deltas, color=colors)
    for i, (b, t) in enumerate(zip(bars, thresholds)):
        if i < 2:
            ax.axhline(t, color="black", linestyle=":", linewidth=0.8, alpha=0.6)
            ax.text(i, t + 0.5, f"min: {t:+.1f}pp", ha="center", fontsize=8, color="black")
    annotate_bars(ax, bars, fmt="{:+.2f}pp", fontsize=10, offset=0.01)
    ax.set_ylabel("Delta (percentage points)")
    ax.set_title("A/B promotion gate — fine-tuned vs baseline (RDD test set)")
    fig.tight_layout()
    out = PLOTS_DIR / "rdd" / "ab_promotion_gate.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ============================================================
# Bengaluru plots — 76 photos before/after cascade-enabled fine-tune
# ============================================================

def plot_bengaluru_status(comp: dict):
    """Status distribution: classified vs expert_review."""
    s = comp["summary"]
    statuses = ["classified", "expert_review"]
    base = [s["baseline_status_counts"].get(k, 0) for k in statuses]
    ft = [s["current_status_counts"].get(k, 0) for k in statuses]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(statuses))
    w = 0.38
    b1 = ax.bar(x - w/2, base, w, label="Baseline (pre-cascade)", color=C_BASE)
    b2 = ax.bar(x + w/2, ft, w, label="Fine-tuned + cascade", color=C_FT)
    ax.set_xticks(x)
    ax.set_xticklabels(["Auto-classified\n(≥80% conf)", "Expert review\n(<80% conf)"])
    ax.set_ylabel("Number of photos")
    ax.set_title(f"Bengaluru re-classification — review status ({s['n_rows_compared']} photos)")
    annotate_bars(ax, b1, fmt="{:.0f}")
    annotate_bars(ax, b2, fmt="{:.0f}")
    ax.legend()
    fig.tight_layout()
    out = PLOTS_DIR / "bengaluru" / "status_distribution.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_bengaluru_stage1_transitions(comp: dict):
    """Stage 1 label transitions: count of each (baseline, fine-tuned) pair."""
    xtab = comp["summary"]["stage1_label_xtab"]
    cells = []
    for k, v in xtab.items():
        if "->" in k:
            a, b = [x.strip() for x in k.split("->")]
            cells.append((a, b, v))

    labels = sorted(set([c[0] for c in cells] + [c[1] for c in cells]))
    mat = np.zeros((len(labels), len(labels)), dtype=int)
    for a, b, v in cells:
        mat[labels.index(a), labels.index(b)] = v

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(mat, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Fine-tuned label")
    ax.set_ylabel("Baseline label")
    ax.set_title("Bengaluru — Stage 1 label transitions (baseline → fine-tuned)")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            color = "white" if mat[i, j] > mat.max() / 2 else "black"
            ax.text(j, i, f"{mat[i, j]}", ha="center", va="center", color=color, fontsize=11)
    ax.grid(False)
    fig.tight_layout()
    out = PLOTS_DIR / "bengaluru" / "stage1_transitions.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_bengaluru_review_flag(comp: dict):
    """Expert-review flag flips: resolved vs newly-flagged vs unchanged."""
    f = comp["summary"]["review_flag_flips"]
    cats = ["Resolved\n(was flagged,\nnow auto)", "Newly flagged\n(was auto,\nnow flagged)", "Unchanged"]
    vals = [f.get("resolved", 0), f.get("newly_flagged", 0), f.get("unchanged", 0)]
    colors = [C_GAIN, C_LOSS, C_NEUT]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(cats, vals, color=colors)
    annotate_bars(ax, bars)
    ax.set_ylabel("Number of photos")
    ax.set_title("Bengaluru — review-flag flips after cascade-enabled fine-tune")
    fig.tight_layout()
    out = PLOTS_DIR / "bengaluru" / "review_flag_flips.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_bengaluru_severity_changes(comp: dict):
    """Severity changes: baseline → fine-tuned (heatmap)."""
    changes = comp.get("severity_changes") or []
    if not changes:
        return None
    levels = ["None", "Low", "Medium", "High"]
    mat = np.zeros((len(levels), len(levels)), dtype=int)
    for sv in changes:
        b = sv["baseline_severity"]
        f = sv["finetuned_severity"]
        if b in levels and f in levels:
            mat[levels.index(b), levels.index(f)] += 1

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(mat, cmap="Oranges", aspect="auto")
    ax.set_xticks(range(len(levels)))
    ax.set_yticks(range(len(levels)))
    ax.set_xticklabels(levels)
    ax.set_yticklabels(levels)
    ax.set_xlabel("Fine-tuned severity")
    ax.set_ylabel("Baseline severity")
    ax.set_title(f"Bengaluru — severity transitions on {len(changes)} changed rows")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j] > 0:
                ax.text(j, i, f"{mat[i, j]}", ha="center", va="center", color="black", fontsize=11)
    ax.grid(False)
    fig.tight_layout()
    out = PLOTS_DIR / "bengaluru" / "severity_transitions.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_bengaluru_type_changes(comp: dict):
    """Frequency of distress types in baseline vs fine-tuned predictions."""
    changes = comp.get("type_changes") or []
    base_c = Counter()
    ft_c = Counter()
    for t in changes:
        for x in t.get("baseline_types") or []:
            base_c[x] += 1
        for x in t.get("finetuned_types") or []:
            ft_c[x] += 1
    if not base_c and not ft_c:
        return None
    types = sorted(set(base_c) | set(ft_c))
    base_v = [base_c.get(t, 0) for t in types]
    ft_v = [ft_c.get(t, 0) for t in types]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(types))
    w = 0.38
    b1 = ax.bar(x - w/2, base_v, w, label="Baseline", color=C_BASE)
    b2 = ax.bar(x + w/2, ft_v, w, label="Fine-tuned + cascade", color=C_FT)
    ax.set_xticks(x)
    ax.set_xticklabels([short(t, 24) for t in types], rotation=30, ha="right")
    ax.set_ylabel("Mentions across changed rows")
    ax.set_title(f"Bengaluru — distress-type mentions on {len(changes)} rows where types changed")
    annotate_bars(ax, b1)
    annotate_bars(ax, b2)
    ax.legend(loc="upper right")
    fig.tight_layout()
    out = PLOTS_DIR / "bengaluru" / "type_change_frequency.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ============================================================
# Attain plots — cross-dataset (NZ roads, 769 images)
# ============================================================

def plot_attain_tier_accuracy(ft: dict, base: Optional[dict] = None):
    """Tier 1 / Tier 2 / Severity accuracy. If base provided, side-by-side."""
    s = ft["summary"]
    tiers = ["Tier 1\n(in-distribution)", "Tier 2\n(zero-shot)", "Severity"]
    ft_acc = [
        s["tier1_in_distribution"]["accuracy"],
        s["tier2_zero_shot"]["accuracy"],
        s["severity"]["accuracy"],
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(tiers))
    if base:
        sb = base["summary"]
        base_acc = [
            sb["tier1_in_distribution"]["accuracy"],
            sb["tier2_zero_shot"]["accuracy"],
            sb["severity"]["accuracy"],
        ]
        w = 0.38
        b1 = ax.bar(x - w/2, base_acc, w, label="Baseline (no adapter)", color=C_BASE)
        b2 = ax.bar(x + w/2, ft_acc, w, label="Fine-tuned (LoRA)", color=C_FT)
        annotate_bars(ax, b1, fmt="{:.1%}")
        annotate_bars(ax, b2, fmt="{:.1%}")
        ax.legend(loc="upper right")
    else:
        bars = ax.bar(x, ft_acc, color=C_FT, label="Fine-tuned (LoRA)")
        annotate_bars(ax, bars, fmt="{:.1%}")
        ax.legend(loc="upper right")
    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    title = "Attain WS_V2.0 cross-dataset — accuracy by tier (769 images, NZ roads)"
    ax.set_title(title)
    fig.tight_layout()
    out = PLOTS_DIR / "attain" / "tier_accuracy.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_attain_per_class_f1(ft: dict, base: Optional[dict] = None):
    """Per-class F1 grouped bar. Supports baseline-vs-finetuned overlay."""
    pc = ft["summary"]["per_class"]
    classes = sorted(pc.keys(), key=lambda c: (pc[c]["tier"], c))
    ft_f1 = [pc[c]["f1"] for c in classes]
    tiers = [pc[c]["tier"] for c in classes]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(classes))
    if base:
        bpc = base["summary"]["per_class"]
        base_f1 = [bpc.get(c, {"f1": 0})["f1"] for c in classes]
        w = 0.38
        b1 = ax.bar(x - w/2, base_f1, w, label="Baseline (no adapter)", color=C_BASE)
        b2 = ax.bar(x + w/2, ft_f1, w, label="Fine-tuned (LoRA)", color=C_FT)
        annotate_bars(ax, b1, fmt="{:.2f}", offset=0.015)
        annotate_bars(ax, b2, fmt="{:.2f}", offset=0.015)
        ax.legend(loc="upper right")
    else:
        # Color by tier
        colors = [C_FT if t == "in-dist" else C_NEUT for t in tiers]
        bars = ax.bar(x, ft_f1, color=colors)
        annotate_bars(ax, bars, fmt="{:.2f}", offset=0.015)
        # Custom legend for tier color coding
        from matplotlib.patches import Patch
        legend_elems = [
            Patch(color=C_FT, label="In-distribution (RDD-trained)"),
            Patch(color=C_NEUT, label="Zero-shot (taxonomy only)"),
        ]
        ax.legend(handles=legend_elems, loc="upper right")
    ax.set_xticks(x)
    ax.set_xticklabels([short(c, 18) for c in classes], rotation=20, ha="right")
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1.0)
    ax.set_title("Attain WS_V2.0 — per-class F1")
    fig.tight_layout()
    out = PLOTS_DIR / "attain" / "per_class_f1.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_attain_per_class_tp_fn(ft: dict):
    """Per-class TP and FN — visualises the recall problem."""
    pc = ft["summary"]["per_class"]
    classes = sorted(pc.keys(), key=lambda c: (pc[c]["tier"], -pc[c]["fn"]))
    tp = [pc[c]["tp"] for c in classes]
    fn = [pc[c]["fn"] for c in classes]
    fp = [pc[c]["fp"] for c in classes]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(classes))
    w = 0.27
    b1 = ax.bar(x - w, tp, w, label="True positive", color=C_GAIN)
    b2 = ax.bar(x, fp, w, label="False positive", color="#ff7f0e")
    b3 = ax.bar(x + w, fn, w, label="False negative", color=C_LOSS)
    ax.set_xticks(x)
    ax.set_xticklabels([short(c, 18) for c in classes], rotation=20, ha="right")
    ax.set_ylabel("Number of instances")
    ax.set_title("Attain WS_V2.0 — fine-tuned per-class TP / FP / FN")
    annotate_bars(ax, b1)
    annotate_bars(ax, b2)
    annotate_bars(ax, b3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    out = PLOTS_DIR / "attain" / "per_class_tp_fp_fn.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_attain_pothole_bias(ft: dict, base: Optional[dict] = None):
    """How many predictions contain Pothole vs how many GT pothole instances exist."""
    rows = ft.get("per_image_results") or []
    gt_potholes = sum(1 for r in rows if "Pothole" in (r.get("gt_attain") or []))
    ft_pred_potholes = sum(1 for r in rows if "Pothole (D40)" in (r.get("pred_pipeline") or []))

    cats = ["GT pothole\ninstances", "Fine-tuned\npredicted pothole"]
    vals = [gt_potholes, ft_pred_potholes]
    colors = [C_NEUT, C_FT]

    if base:
        base_rows = base.get("per_image_results") or []
        base_pred_potholes = sum(1 for r in base_rows if "Pothole (D40)" in (r.get("pred_pipeline") or []))
        cats.insert(1, "Baseline\npredicted pothole")
        vals.insert(1, base_pred_potholes)
        colors.insert(1, C_BASE)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(cats, vals, color=colors)
    annotate_bars(ax, bars)
    ax.set_ylabel("Count of images")
    ax.set_title("Attain WS_V2.0 — pothole prediction frequency vs ground truth")
    fig.tight_layout()
    out = PLOTS_DIR / "attain" / "pothole_bias.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_attain_label_frequency(ft: dict, base: Optional[dict] = None, top_n: int = 12):
    """Top pipeline labels by emission count."""
    ft_rows = ft.get("per_image_results") or []
    ft_freq = Counter()
    for r in ft_rows:
        for p in r.get("pred_pipeline") or []:
            ft_freq[p] += 1

    if base:
        base_rows = base.get("per_image_results") or []
        base_freq = Counter()
        for r in base_rows:
            for p in r.get("pred_pipeline") or []:
                base_freq[p] += 1
        labels_set = set(ft_freq) | set(base_freq)
    else:
        base_freq = Counter()
        labels_set = set(ft_freq)

    # Pick top N by max(base, ft)
    ranked = sorted(labels_set, key=lambda k: -max(ft_freq.get(k, 0), base_freq.get(k, 0)))
    labels = ranked[:top_n]
    ft_v = [ft_freq.get(k, 0) for k in labels]
    base_v = [base_freq.get(k, 0) for k in labels]

    fig, ax = plt.subplots(figsize=(12, 6))
    y = np.arange(len(labels))
    h = 0.38
    if base:
        ax.barh(y - h/2, base_v, h, label="Baseline", color=C_BASE)
        ax.barh(y + h/2, ft_v, h, label="Fine-tuned", color=C_FT)
        ax.legend(loc="lower right")
    else:
        ax.barh(y, ft_v, color=C_FT, label="Fine-tuned")
        ax.legend(loc="lower right")
    ax.set_yticks(y)
    ax.set_yticklabels([short(k, 30) for k in labels])
    ax.invert_yaxis()
    ax.set_xlabel("Times this label was emitted (out of 769 images)")
    ax.set_title(f"Attain WS_V2.0 — top {len(labels)} predicted pipeline labels")
    fig.tight_layout()
    out = PLOTS_DIR / "attain" / "label_frequency.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_attain_stage1_rate(ft: dict, base: Optional[dict] = None):
    """Stage 1 distress detection rate."""
    ft_rows = ft.get("per_image_results") or []
    ft_dist = sum(1 for r in ft_rows if r.get("is_distressed_pred"))
    ft_norm = len(ft_rows) - ft_dist

    cats = ["Predicted Distress", "Predicted Normal"]
    fig, ax = plt.subplots(figsize=(8, 5))

    if base:
        base_rows = base.get("per_image_results") or []
        base_dist = sum(1 for r in base_rows if r.get("is_distressed_pred"))
        base_norm = len(base_rows) - base_dist
        x = np.arange(len(cats))
        w = 0.38
        b1 = ax.bar(x - w/2, [base_dist, base_norm], w, label="Baseline", color=C_BASE)
        b2 = ax.bar(x + w/2, [ft_dist, ft_norm], w, label="Fine-tuned", color=C_FT)
        ax.set_xticks(x)
        ax.set_xticklabels(cats)
        annotate_bars(ax, b1)
        annotate_bars(ax, b2)
        ax.legend(loc="upper right")
    else:
        bars = ax.bar(cats, [ft_dist, ft_norm], color=[C_FT, C_BASE])
        annotate_bars(ax, bars)
    ax.set_ylabel("Number of images")
    ax.set_title("Attain WS_V2.0 — Stage 1 distress detection rate")
    fig.tight_layout()
    out = PLOTS_DIR / "attain" / "stage1_detection_rate.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_attain_severity_distribution(ft: dict, base: Optional[dict] = None):
    """Predicted severity distribution."""
    levels = ["None", "Low", "Medium", "High"]
    ft_rows = ft.get("per_image_results") or []
    ft_c = Counter(r.get("pred_severity", "None") or "None" for r in ft_rows)
    ft_v = [ft_c.get(l, 0) for l in levels]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(levels))
    if base:
        base_rows = base.get("per_image_results") or []
        base_c = Counter(r.get("pred_severity", "None") or "None" for r in base_rows)
        base_v = [base_c.get(l, 0) for l in levels]
        w = 0.38
        b1 = ax.bar(x - w/2, base_v, w, label="Baseline", color=C_BASE)
        b2 = ax.bar(x + w/2, ft_v, w, label="Fine-tuned", color=C_FT)
        annotate_bars(ax, b1)
        annotate_bars(ax, b2)
        ax.legend(loc="upper right")
    else:
        bars = ax.bar(x, ft_v, color=C_FT)
        annotate_bars(ax, bars)
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_ylabel("Number of images")
    ax.set_title("Attain WS_V2.0 — predicted severity distribution")
    fig.tight_layout()
    out = PLOTS_DIR / "attain" / "severity_distribution.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_attain_head_to_head(comparison: dict):
    """Pie chart of head-to-head outcomes (only when comparison JSON exists)."""
    h = comparison.get("head_to_head") or {}
    cats = [
        "Fine-tuned\nstrictly better",
        "Baseline\nstrictly better",
        "Tied — both\ncorrect on ≥1 type",
        "Tied — both\nwrong / partial",
    ]
    vals = [
        h.get("finetuned_strictly_better", 0),
        h.get("baseline_strictly_better", 0),
        h.get("tied_both_correct", 0),
        h.get("tied_both_wrong_or_partial", 0),
    ]
    colors = [C_FT, C_BASE, C_GAIN, C_NEUT]

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, _, autotxts = ax.pie(
        vals, labels=cats, colors=colors,
        autopct=lambda p: f"{int(round(p * sum(vals) / 100))}\n({p:.1f}%)",
        startangle=90, pctdistance=0.78,
    )
    for at in autotxts:
        at.set_color("white")
        at.set_fontsize(10)
        at.set_fontweight("bold")
    ax.set_title(f"Attain WS_V2.0 — head-to-head per-image (n={h.get('n_compared', 0)})")
    fig.tight_layout()
    out = PLOTS_DIR / "attain" / "head_to_head.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_attain_f1_delta(comparison: dict):
    """F1 delta per class — does fine-tuning help or hurt each one?"""
    pc = comparison.get("per_class") or {}
    if not pc:
        return None
    classes = sorted(pc.keys(), key=lambda c: pc[c]["delta"]["f1"])
    deltas = [pc[c]["delta"]["f1"] for c in classes]
    tiers = [pc[c]["tier"] for c in classes]
    labels = [f"{c}\n({t})" for c, t in zip(classes, tiers)]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = [C_GAIN if d > 0 else (C_LOSS if d < 0 else C_NEUT) for d in deltas]
    bars = ax.barh(labels, deltas, color=colors)
    # Place delta value clearly outside each bar in the same direction
    xmax = max(abs(d) for d in deltas) if deltas else 1
    for b, d in zip(bars, deltas):
        offset = 0.01 if d >= 0 else -0.01
        ha = "left" if d >= 0 else "right"
        ax.text(b.get_width() + offset, b.get_y() + b.get_height() / 2,
                f"{d:+.3f}", va="center", ha=ha,
                fontsize=10, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("F1 change after fine-tuning (positive = improvement)")
    ax.set_title("Attain WS_V2.0 — per-class F1 delta (fine-tuned − baseline)")
    # Add headroom for the label text
    ax.set_xlim(-xmax * 1.25, xmax * 1.25)
    fig.tight_layout()
    out = PLOTS_DIR / "attain" / "per_class_f1_delta.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ============================================================
# Index
# ============================================================

def write_index(generated: dict[str, list[Path]]):
    out = PLOTS_DIR / "INDEX.md"
    lines = []
    lines.append("# Evaluation Plots Index")
    lines.append("")
    lines.append("Auto-generated by `scripts/generate_eval_plots.py`. "
                 "All plots at 300 DPI. Click thumbnails for full size.")
    lines.append("")
    for section, paths in generated.items():
        if not paths:
            continue
        lines.append(f"## {section}")
        lines.append("")
        for p in paths:
            rel = p.relative_to(PLOTS_DIR).as_posix()
            title = p.stem.replace("_", " ").title()
            lines.append(f"### {title}")
            lines.append("")
            lines.append(f"![{title}]({rel})")
            lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nIndex: {out}")


# ============================================================
# Main
# ============================================================

def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    (PLOTS_DIR / "rdd").mkdir(exist_ok=True)
    (PLOTS_DIR / "bengaluru").mkdir(exist_ok=True)
    (PLOTS_DIR / "attain").mkdir(exist_ok=True)

    generated: dict[str, list[Path]] = {"RDD (in-distribution)": [],
                                         "Bengaluru (real-world)": [],
                                         "Attain (cross-dataset)": []}

    print("Loading evaluation results...")
    ab = load_json("ab_comparison.json")
    bengaluru = load_json("bengaluru_comparison.json")
    attain_ft = load_json("attain_finetuned_results.json") or load_json("attain_cross_dataset_results.json")
    attain_base = load_json("attain_baseline_results.json")
    attain_compare = load_json("attain_baseline_vs_finetuned.json")

    print("\nGenerating RDD plots...")
    if ab:
        for fn in (plot_rdd_stage1_accuracy, plot_rdd_stage2_per_class,
                   plot_rdd_stage1_confusion, plot_rdd_promotion_summary):
            try:
                p = fn(ab)
                print(f"  [ok] {p.relative_to(PROJECT_ROOT)}")
                generated["RDD (in-distribution)"].append(p)
            except Exception as e:
                print(f"  [err] {fn.__name__}: {e}")

    print("\nGenerating Bengaluru plots...")
    if bengaluru:
        for fn in (plot_bengaluru_status, plot_bengaluru_stage1_transitions,
                   plot_bengaluru_review_flag, plot_bengaluru_severity_changes,
                   plot_bengaluru_type_changes):
            try:
                p = fn(bengaluru)
                if p:
                    print(f"  [ok] {p.relative_to(PROJECT_ROOT)}")
                    generated["Bengaluru (real-world)"].append(p)
            except Exception as e:
                print(f"  [err] {fn.__name__}: {e}")

    print("\nGenerating Attain plots...")
    if attain_ft:
        plots = [
            ("tier_accuracy", lambda: plot_attain_tier_accuracy(attain_ft, attain_base)),
            ("per_class_f1", lambda: plot_attain_per_class_f1(attain_ft, attain_base)),
            ("per_class_tp_fp_fn", lambda: plot_attain_per_class_tp_fn(attain_ft)),
            ("pothole_bias", lambda: plot_attain_pothole_bias(attain_ft, attain_base)),
            ("label_frequency", lambda: plot_attain_label_frequency(attain_ft, attain_base)),
            ("stage1_detection_rate", lambda: plot_attain_stage1_rate(attain_ft, attain_base)),
            ("severity_distribution", lambda: plot_attain_severity_distribution(attain_ft, attain_base)),
        ]
        for name, fn in plots:
            try:
                p = fn()
                print(f"  [ok] {p.relative_to(PROJECT_ROOT)}")
                generated["Attain (cross-dataset)"].append(p)
            except Exception as e:
                print(f"  [err] attain {name}: {e}")

    if attain_compare:
        for fn in (plot_attain_head_to_head, plot_attain_f1_delta):
            try:
                p = fn(attain_compare)
                if p:
                    print(f"  [ok] {p.relative_to(PROJECT_ROOT)}")
                    generated["Attain (cross-dataset)"].append(p)
            except Exception as e:
                print(f"  [err] {fn.__name__}: {e}")

    write_index(generated)
    total = sum(len(v) for v in generated.values())
    print(f"\nGenerated {total} plots across {sum(1 for v in generated.values() if v)} sections.")


if __name__ == "__main__":
    main()
