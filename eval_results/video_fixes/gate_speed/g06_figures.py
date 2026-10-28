"""Two figures (CPU only, matplotlib): fig1_gate_heldout.png and fig2_extraction_time.png.

fig1: per held-out session, recall vs pass rate of probe_frac (the session's own curve, grey), the
      leave-one-session-out calibrated operating point and the default thresholds with 95 % block-
      bootstrap CIs; fourth panel: pooled held-out points of every pre-registered rule.
fig2: extraction wall time per km of planned road by stage, before (HEAD, default config) and after
      (this study, probe-only calibrated gate), mean of 2 repetitions; tick marks = each repetition's total.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
loso = json.loads((HERE / "out" / "loso.json").read_text(encoding="utf-8"))
cmp_ = json.loads((HERE / "out" / "compare.json").read_text(encoding="utf-8"))
BLUE, ORANGE, AQUA, YELLOW, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8986", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})
SESS = [("f428", "f428 (1.15 km)"), ("293d", "293d (0.27 km)"), ("d47c", "d47c (2.64 km)")]


def point(ax, d, color, filled, label, dx=0.012, dy=0.0):
    x, y = d["pass_rate"], d["recall"]
    xe = np.array([[x - d["pass_rate_ci95"][0]], [d["pass_rate_ci95"][1] - x]])
    ye = np.array([[y - d["recall_ci95"][0]], [d["recall_ci95"][1] - y]])
    ax.errorbar([x], [y], xerr=xe, yerr=ye, fmt="o", ms=7, color=color, mfc=color if filled else "white",
                mec=color, mew=2, elinewidth=1.5, capsize=3, zorder=3)
    ax.annotate(label, (x, y), xytext=(x + dx, y + dy), fontsize=8, color=INK, va="center", zorder=4)


fig, axs = plt.subplots(1, 4, figsize=(13, 3.9), sharey=True)
for ax, (s, title) in zip(axs, SESS):
    c = np.array(loso["curves_probe_frac"][s])
    ax.plot(c[:, 0], c[:, 1], color=GREY, lw=1.5, drawstyle="steps-post", zorder=1)
    ax.axhline(0.9, color=MUTED, lw=1, ls="--", zorder=0)
    v1, v0 = loso["sessions"]["V1_probe_frac"][s], loso["sessions"]["V0_default"][s]
    point(ax, v1, BLUE, True, f"calibrated on the\nother two: {v1['recall']:.2f} / {v1['pass_rate']:.2f}", dx=0.03, dy=-0.07)
    point(ax, v0, ORANGE, False, "default", dx=-0.16, dy=-0.04)
    n, p = v1["n"], v1["n_pos"]
    ax.set_title(f"held out: {title}\n{n} segments, {p} VLM-positive", fontsize=9, color=INK, loc="left")
    ax.set_xlabel("pass rate (share of segments sent)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0.4, 1.02)
    ax.grid(color="#e8e7e3", lw=0.6)
axs[0].set_ylabel("recall vs VLM (distress kept)")
axs[0].text(0.02, 0.43, "grey: probe_frac curve of the held-out session\n- - target recall 0.9", fontsize=7.5, color=MUTED)
ax = axs[3]
P = loso["pooled"]
names = [("V0_default", ORANGE, False, "default thresholds"), ("V1_probe_frac", BLUE, True, "probe_frac (ships)"),
         ("V2_probe_max_logit", AQUA, True, "probe_max_logit"), ("V3_probe_frac_or_max_logit", YELLOW, True, "frac OR max_logit"),
         ("V4_anyof_shipping", GREY, True, "any-of frac/evidence/crack")]
offs = {"V0_default": (-0.30, -0.05), "V1_probe_frac": (0.03, -0.10), "V2_probe_max_logit": (0.03, 0.06),
        "V3_probe_frac_or_max_logit": (-0.42, -0.02), "V4_anyof_shipping": (-0.48, -0.13)}
for k, col, filled, lab in names:
    point(ax, P[k], col, filled, lab, *offs[k])
ax.axhline(0.9, color=MUTED, lw=1, ls="--", zorder=0)
ax.set_title("pooled over the 3 held-out sessions\n813 segments, 164 VLM-positive", fontsize=9, color=INK, loc="left")
ax.set_xlabel("pass rate")
ax.set_xlim(0, 1.02)
ax.grid(color="#e8e7e3", lw=0.6)
fig.suptitle("Probe-only gate, leave-one-session-out (target recall 0.9); bars: 95 % CI, 20-segment block bootstrap with "
             "re-calibration, B = 2000. Reference = VLM labels.", fontsize=9, color=MUTED, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(HERE / "fig1_gate_heldout.png", dpi=200)

# ---- fig2 ----
rows = []
for s, title in SESS + [("all", "all three (4.06 km)")]:
    for var, lab in (("before", "before"), ("after", "after")):
        segs = [x for x, _ in SESS] if s == "all" else [s]
        km = sum(cmp_["sessions"][x]["planned_m"] for x in segs) / 1000
        stages, totals = [], []
        for rep in (1, 2):
            st = {"bank": 0.0, "decode": 0.0, "gate": 0.0, "other": 0.0}
            tot = 0.0
            for x in segs:
                t = cmp_["sessions"][x]["runs"][f"{var}_rep{rep}"]["timings_s"]
                visible = t["gate"] if var == "before" else t["gate_wait"]
                st["bank"] += t["bank_total"]
                st["gate"] += visible
                st["decode"] += t["decode_pass"] - visible
                st["other"] += t["setup"] + t["outputs"]
                tot += t["total"]
            stages.append({k: v / km for k, v in st.items()})
            totals.append(tot / km)
        rows.append((f"{title}  {lab}", {k: np.mean([d[k] for d in stages]) for k in stages[0]}, totals))
fig, ax = plt.subplots(figsize=(9.5, 4.6))
cols = {"bank": GREY, "decode": BLUE, "gate": ORANGE, "other": "#c9c8c3"}
labels = {"bank": "bank pass", "decode": "decode + select (+ hidden gate, after)", "gate": "gate (serial before; waited for, after)",
          "other": "setup + outputs"}
y = np.arange(len(rows))[::-1] + np.repeat(np.arange(4)[::-1] * 0.6, 2)
for yi, (lab, st, tots) in zip(y, rows):
    left = 0.0
    for k in ("bank", "decode", "gate", "other"):
        ax.barh(yi, st[k], left=left, height=0.8, color=cols[k], edgecolor="white", linewidth=2)
        left += st[k]
    for tt in tots:
        ax.plot([tt, tt], [yi - 0.45, yi + 0.45], color=INK, lw=1)
    ax.text(max(tots) + 4, yi, f"{np.mean(tots):.0f} s/km", va="center", fontsize=8, color=INK)
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows], fontsize=8)
ax.set_xlabel("extraction wall time per km of planned road (s/km), RTX A5000, mean of 2 runs (ticks: each run)")
ax.set_xlim(0, max(max(r[2]) for r in rows) * 1.18)
ax.grid(axis="x", color="#e8e7e3", lw=0.6)
handles = [plt.Rectangle((0, 0), 1, 1, color=cols[k]) for k in cols]
ax.legend(handles, [labels[k] for k in cols], fontsize=7.5, frameon=False, loc="lower right")
ax.set_title("Segment-mode extraction: HEAD with default gate (bank + crack filter) vs probe-only calibrated gate,\n"
             "no bank, gate on its own thread", fontsize=9, color=INK, loc="left")
fig.tight_layout()
fig.savefig(HERE / "fig2_extraction_time.png", dpi=200)
print("ok")
