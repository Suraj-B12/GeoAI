"""Plots for FINDINGS.md (matplotlib only, no torch). Run with the Capstone venv (it has matplotlib)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

D = Path(r"C:/GeoAI/analysis/discriminator")
O = D / "out"
INK, MUTED, A, B, C = "#1c1917", "#78716c", "#2563eb", "#d97706", "#16a34a"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})

# ---- fig 1: VLM seconds per image, by stage ----
J = pd.read_csv(O / "joined.csv")
rows = [json.loads(l) for l in open(r"C:/Users/PRO-LAB-3/Documents/GeoAI_Video/field/batches/batch_001/responses.jsonl")]
tiles = np.median([r["response"]["stage1_safety_net"]["views"]["recorded"]["time_ms"] for r in rows
                   if r["response"].get("stage1_safety_net")]) / 1000
sn = J.loc[J.s1_label == "Normal", "sn_ms"].median() / 1000
s1n = J.loc[J.s1_label == "Normal", "s1_ms"].median() / 1000
s1d = J.loc[J.s1_label == "Distress", "s1_ms"].median() / 1000
s2d = J.loc[J.s1_label == "Distress", "s2_ms"].median() / 1000
fig, ax = plt.subplots(figsize=(7.5, 2.6))
bars = [("Stage 1 Normal\n(79 % of answers)", [(s1n, "Stage 1", A), (sn - tiles, "probe, full photo", B), (tiles, "6 recorded tiles (unused)", "#dc2626")]),
        ("Stage 1 Distress\n(21 %)", [(s1d, "Stage 1", A), (s2d - tiles, "Stage 2 text + probe (est.)", C), (tiles, "6 recorded tiles (unused, est.)", "#dc2626")])]
for i, (lab, parts) in enumerate(bars):
    x0 = 0
    for w, name, col in parts:
        ax.barh(i, w, left=x0, color=col, height=0.55)
        if w > 2.5:
            ax.text(x0 + w / 2, i, f"{name}\n{w:.1f} s", ha="center", va="center", color="white", fontsize=8)
        x0 += w
    ax.text(x0 + 0.2, i, f"{x0:.1f} s", va="center", color=INK)
ax.set_yticks([0, 1], [b[0] for b in bars])
ax.invert_yaxis()
ax.set_xlabel("seconds per image on the A5000 (median, batch_001 complete, n = %d)" % len(J))
ax.set_xlim(0, 26)
fig.tight_layout()
fig.savefig(D / "fig1_vlm_time.png", dpi=150)

# ---- fig 2: gate AUROC per signal ----
tab = json.loads((O / "tabular.json").read_text())
items = [("record: " + k[2:], v["auroc"], v["auroc_ci95"]) for k, v in tab["auroc"].items()
         if k in ("s_probe_frac", "s_probe_max_logit", "s_evidence", "s_crack_mm", "s_z_max", "s_anomalous_patch_frac")]
gv = O / "gate_variants_auroc.json"
if gv.exists():
    for k, v in json.loads(gv.read_text()).items():
        if k.startswith(("probe_max_x", "probe_frac_x", "crack_fixlen_road", "crack_code_z10")):
            items.append(("fix: " + k, v["auroc"], v["auroc_ci95"]))
fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(items) + 0.9))
for i, (n, a, ci) in enumerate(items):
    col = C if a - 0.5 > 0.15 else (B if a > 0.6 else MUTED)
    ax.plot(ci, [i, i], color=col, lw=2)
    ax.plot(a, i, "o", color=col)
    ax.text(1.0, i, f"{a:.2f}", va="center", fontsize=8)
ax.axvline(0.5, color=MUTED, ls=":", lw=1)
ax.set_yticks(range(len(items)), [x[0] for x in items])
ax.invert_yaxis()
ax.set_xlim(0.3, 1.05)
ax.set_xlabel("AUROC, 95 %% CI; record: all 3 sessions n=%d, %d pos.; fix: f428 n=162" % (tab["n_answered"], tab["vlm_positive"]))
fig.tight_layout()
fig.savefig(D / "fig2_gate_auroc.png", dpi=150)

# ---- fig 3: best-frame position ----
fig, axs = plt.subplots(1, 2, figsize=(8, 2.8))
for ax, sid in zip(axs, ("d47c", "f428")):
    p = O / f"bestframe_summary_{sid}.csv"
    if not p.exists():
        continue
    R = pd.read_csv(p)
    bins = np.linspace(0, 1, 11)
    ax.hist(R.code_win, bins=bins, color=B, alpha=0.8, label="chosen now (sharpest, raw Tenengrad)")
    ax.hist(R.fair_win, bins=bins, color=A, alpha=0.55, label="sharpest at equal ground resolution")
    ax.set_xlabel("position in candidate window (0 = nearest frame)")
    ax.set_title(f"{sid}: {len(R)} segments", fontsize=10)
axs[0].set_ylabel("segments")
axs[0].legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(D / "fig3_bestframe_position.png", dpi=150)

# ---- fig 4: extraction time d47c, now vs proposed ----
proj = json.loads((O / "extraction_projection.json").read_text()) if (O / "extraction_projection.json").exists() else None
if proj:
    fig, ax = plt.subplots(figsize=(7.5, 2.4))
    for i, (lab, parts) in enumerate(proj["bars"]):
        x0 = 0
        for name, w, col in parts:
            ax.barh(i, w, left=x0, color=col, height=0.55)
            if w > 60:
                ax.text(x0 + w / 2, i, f"{name}\n{w:.0f}", ha="center", va="center", color="white", fontsize=7)
            x0 += w
        ax.text(x0 + 5, i, f"{x0:.0f} s", va="center")
    ax.set_yticks(range(len(proj["bars"])), [b[0] for b in proj["bars"]])
    ax.invert_yaxis()
    ax.set_xlabel("seconds, session d47c (2.64 km, 528 segments)")
    fig.tight_layout()
    fig.savefig(D / "fig4_extraction_time.png", dpi=150)

# ---- fig 5: VLM positives along the route ----
J = J[J.session == "f428"].sort_values("k")
fig, ax = plt.subplots(figsize=(8, 2.0))
ax.bar(J.k, J.vlm_distress.astype(int), width=1.0, color="#dc2626")
ax.plot(J.k, (J.s_probe_max_logit - J.s_probe_max_logit.min()) / (J.s_probe_max_logit.max() - J.s_probe_max_logit.min()),
        color=A, lw=0.8, label="gate probe_max_logit (scaled)")
ax.set_yticks([])
ax.set_xlabel("segment k (f428, 5 m each); red = VLM says distress")
ax.legend(fontsize=7, frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(D / "fig5_vlm_positive_runs.png", dpi=150)
print("plots written")
