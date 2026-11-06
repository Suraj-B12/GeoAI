"""
Figures 6.11a-d of the weight-space programming study (paper §6.11).
No inference: everything comes from stored JSON.

  fig_6_11a  test score distributions + ROC: base, edited, recalibrated base, random control
  fig_6_11b  dose-response on DEV: c vs recall / specificity / MCC (eval_results/finetune/wsp_dev_runs.json)
  fig_6_11c  localisation: per-layer direct contribution to logit(Distress)-logit(Normal),
             distress vs clean frames (C:\\GeoAI\\wsp\\analysis\\dla.json, copied into wsp_dev_runs.json)
  fig_6_11d  Bengaluru: Distress share base vs edited vs random control
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts import wsp_core as W  # noqa: E402
from scripts.ft_common import FT_DIR, read_json  # noqa: E402

OUT = PROJECT_ROOT / "eval_results" / "plots" / "paper"
COL = {"base": "#57534e", "edited": "#2563eb", "recalibrated_base": "#d97706", "random_control": "#a8a29e"}
LAB = {"base": "Base (production)", "edited": "Edited weights", "recalibrated_base": "Recalibrated base",
       "random_control": "Random-direction control"}


def roc(scores, y):
    pts = sorted(zip(scores, y), key=lambda t: -t[0])
    P = sum(y)
    N = len(y) - P
    tp = fp = 0
    xs, ys = [0.0], [0.0]
    for s, l in pts:
        if l:
            tp += 1
        else:
            fp += 1
        xs.append(fp / N)
        ys.append(tp / P)
    return xs, ys


def fig_a():
    raw = read_json(W.WSP_DIR / "final" / "final_raw.json")
    rep = read_json(FT_DIR / "wsp_report.json")
    if not raw or not rep:
        print("[skip] 6.11a: no final evaluation yet")
        return
    test = sorted(W.pools()["test"], key=lambda r: r["key"])
    b = rep["recalibration"]["offset_b_star"]
    series = {"base": (raw["configs"]["base"], 0.0), "recalibrated_base": (raw["configs"]["base"], b),
              "edited": (raw["configs"]["final"], 0.0), "random_control": (raw["configs"]["random"], 0.0)}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, (res, off) in series.items():
        s = [res[r["key"]]["logit_distress"] - res[r["key"]]["logit_normal"] + off for r in test]
        y = [bool(r["gt"]) for r in test]
        xs, ys = roc(s, y)
        ax[1].plot(xs, ys, color=COL[name], label=f"{LAB[name]} (AUROC {rep['test'][name]['auroc']['point']:.3f})",
                   lw=1.6, ls="--" if name == "recalibrated_base" else "-")
        if name in ("base", "edited"):
            for lab, ls in ((True, "-"), (False, ":")):
                v = [x for x, t in zip(s, y) if t == lab]
                ax[0].hist(v, bins=40, histtype="step", color=COL[name], ls=ls, lw=1.4, density=True,
                           label=f"{LAB[name]}, {'distress' if lab else 'clean'} (n={len(v)})")
    ax[0].axvline(0, color="#dc2626", lw=1)
    ax[0].set_xlabel("logit(Distress) - logit(Normal)  (> 0 = Distress)")
    ax[0].set_ylabel("density")
    ax[0].set_title("Stage 1 decision scores, held-out test")
    ax[0].legend(fontsize=7)
    ax[1].plot([0, 1], [0, 1], color="#e7e5e4", lw=1)
    ax[1].set_xlabel("1 - specificity")
    ax[1].set_ylabel("recall")
    ax[1].set_title("ROC, held-out test")
    ax[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_6_11a_wsp_test_scores_roc.png", dpi=160)
    plt.close(fig)


def fig_b():
    runs = read_json(FT_DIR / "wsp_dev_runs.json", {})
    dose = [r for r in runs.get("runs", []) if r.get("dose_group")]
    if not dose:
        print("[skip] 6.11b: no dose-response runs")
        return
    groups = sorted({r["dose_group"] for r in dose})
    fig, axs = plt.subplots(1, len(groups), figsize=(5.2 * len(groups), 3.8), squeeze=False)
    for ax, g in zip(axs[0], groups):
        allr = [r for r in dose if r["dose_group"] == g]
        base = next((r["dev_base_same_frames"] for r in allr), None)
        rs = sorted([r for r in allr if not r.get("control")], key=lambda r: r["c"])
        c = [0.0] + [r["c"] for r in rs]
        for k, col in (("recall", "#2563eb"), ("specificity", "#16a34a"), ("mcc", "#1c1917")):
            ax.plot(c, [base[k]] + [r["dev"][k] for r in rs], marker="o", color=col, label=k)
        for r in [r for r in allr if r.get("control")]:
            ax.scatter([r["c"]], [r["dev"]["mcc"]], marker="x", s=60, color="#dc2626", zorder=5,
                       label="MCC, random direction")
        rb = runs.get("recalibrated_base_dev256")
        if rb:
            ax.axhline(rb["mcc"], color="#d97706", ls="--", lw=1, label="MCC, recalibrated base")
        ax.set_title(g, fontsize=9)
        ax.set_xlabel("c")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7)
    fig.suptitle("Dose-response on DEV (256 frames; c = 0 is the base model; x = norm-matched random direction)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_6_11b_wsp_dose_response_dev.png", dpi=160)
    plt.close(fig)


def fig_c():
    runs = read_json(FT_DIR / "wsp_dev_runs.json", {})
    dla = runs.get("dla_per_layer")
    if not dla:
        print("[skip] 6.11c: no localisation data")
        return
    L = len(dla["attn_distress"])
    x = list(range(L))
    fig, ax = plt.subplots(figsize=(10, 3.8))
    w = 0.2
    for off, key, col, lab in ((-1.5, "attn_distress", "#93c5fd", "attention, distress"),
                               (-0.5, "attn_clean", "#2563eb", "attention, clean"),
                               (0.5, "mlp_distress", "#fcd34d", "MLP, distress"),
                               (1.5, "mlp_clean", "#d97706", "MLP, clean")):
        ax.bar([i + off * w for i in x], dla[key], width=w, color=col, label=lab)
    for l in dla.get("edited_layers", []):
        ax.axvspan(l - 0.5, l + 0.5, color="#dc2626", alpha=0.08)
    ax.axhline(0, color="#1c1917", lw=0.8)
    ax.set_xlabel("decoder layer")
    ax.set_ylabel("mean direct contribution\n(+ = toward Distress)")
    ax.set_title("Where the Stage 1 decision is written (practice + dev, base model; shaded = edited)")
    ax.legend(fontsize=7, ncol=4)
    fig.tight_layout()
    fig.savefig(OUT / "fig_6_11c_wsp_localisation.png", dpi=160)
    plt.close(fig)


def fig_d():
    rep = read_json(FT_DIR / "wsp_report.json")
    if not rep:
        print("[skip] 6.11d: no report yet")
        return
    b = rep["regression"]["bengaluru"]
    names = [("base", "Base"), ("final", "Edited"), ("random", "Random control")]
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for i, (k, lab) in enumerate(names):
        if k not in b:
            continue
        s = b[k]["share"]
        lo, hi = b[k]["ci95"]
        ax.bar(i, 100 * s, color=["#57534e", "#2563eb", "#a8a29e"][i])
        ax.errorbar(i, 100 * s, yerr=[[100 * (s - lo)], [100 * (hi - s)]], color="#1c1917", capsize=4)
        ax.text(i, 100 * hi + 3, f"{100 * s:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n[1] for n in names])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Stage 1 'Distress' share (%)")
    ax.set_title(f"Bengaluru drive-video segments (n = {b['base']['n']}, unlabelled)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_6_11d_wsp_bengaluru_shift.png", dpi=160)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig_a, fig_b, fig_c, fig_d):
        f()
    print(f"[figures] -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
