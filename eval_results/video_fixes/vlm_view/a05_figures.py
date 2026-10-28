"""VLM-view A/B, step 5: paper-style figures from band_vs_frame.json, nearest_geometry.json, nearest_vs_sharpest.json."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(r"C:/GeoAI/analysis/fixes/vlm_view")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlesize": 10, "axes.titleweight": "bold",
                     "savefig.dpi": 200, "savefig.bbox": "tight"})
C_FRAME, C_BAND, C_NEAR, C_GREY = "#3b6ea8", "#d9822b", "#2f9e6e", "#9aa0a6"
bv = json.loads((OUT / "band_vs_frame.json").read_text())

# ---- Fig 1: agreement matrix and flips by stratum ----------------------------------------------------
t = bv["agreement"]["band_vs_frame_rerun"]
c = bv["agreement"]["frame_rerun_vs_frame_b001"]
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2), gridspec_kw={"width_ratios": [1, 1, 1.25]})
for a, tab, title, xl in ((ax[0], t, "Band vs frame (same day)", "band"), (ax[1], c, "Frame re-run vs 27 Oct (control)", "frame, 28 Oct")):
    m = np.array([[tab["pp"], tab["pn"]], [tab["np"], tab["nn"]]])
    a.imshow(m, cmap="Blues", vmin=0, vmax=m.max() * 1.15)
    for i in range(2):
        for j in range(2):
            a.text(j, i, str(m[i, j]), ha="center", va="center", fontsize=12,
                   color="white" if m[i, j] > m.max() * 0.6 else "black")
    a.set_xticks([0, 1], ["Distress", "Normal"])
    a.set_yticks([0, 1], ["Distress", "Normal"])
    a.set_xlabel(xl)
    a.set_ylabel("frame, 28 Oct" if tab is t else "frame, 27 Oct")
    a.set_title(f"{title}\nagreement {100 * tab['agreement']:.1f} %, kappa {tab['kappa']:.2f}", fontsize=9)
    for s in a.spines.values():
        s.set_visible(False)
st = bv["strata"]
names = [("frame_b001", "frame 27 Oct", C_GREY), ("frame_rerun", "frame 28 Oct", C_FRAME), ("band", "band 28 Oct", C_BAND)]
x = np.arange(2)
for i, (k, lab, col) in enumerate(names):
    vals = [st[k]["pos_stratum_share"], st[k]["normal_stratum_share"]]
    cis = [st[k]["pos_stratum_ci"], st[k]["normal_stratum_ci"]]
    err = np.array([[v - ci[0] for v, ci in zip(vals, cis)], [ci[1] - v for v, ci in zip(vals, cis)]])
    ax[2].bar(x + (i - 1) * 0.26, vals, 0.26, color=col, label=lab, yerr=err, capsize=2, error_kw={"lw": 0.8})
ax[2].set_xticks(x, ["27-Oct positives\n(n = 164)", "27-Oct Normals\n(n = 164)"])
ax[2].set_ylabel("share answered Distress")
ax[2].set_ylim(0, 1.05)
ax[2].legend(frameon=False, fontsize=8)
ax[2].set_title("Distress rate by stratum (95 % Wilson CI)", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "fig1_agreement_flips.png")

# ---- Fig 2: clustering before / after ---------------------------------------------------------------
w, ci = bv["weighted"], bv["weighted_ci95"]
fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.0))
for i, (k, lab, col) in enumerate(names):
    for j, (q, qc) in enumerate((("base", "base"), ("p_pos_given_prev_pos", "cond"))):
        v = w[k][q]
        lo, hi = ci[k][qc]
        ax[0].bar(j + (i - 1) * 0.26, v, 0.26, color=col, label=lab if j == 0 else None,
                  yerr=[[v - lo], [hi - v]], capsize=2, error_kw={"lw": 0.8})
ax[0].set_xticks([0, 1], ["P(distress)", "P(distress | previous\nsegment distress)"])
ax[0].set_ylim(0, 1)
ax[0].legend(frameon=False, fontsize=8)
ax[0].set_title("Clustering of VLM positives (weighted sample)", fontsize=9)
lifts = [w[k]["lift"] for k, _, _ in names]
ax[1].bar(range(3), lifts, color=[c for _, _, c in names])
ax[1].axhline(1.0, color="black", lw=0.8, ls=":")
ax[1].set_xticks(range(3), [lab for _, lab, _ in names])
ax[1].set_ylabel("lift = P(pos | prev pos) / P(pos)")
ax[1].set_title("Run-on of positives into the next segment", fontsize=9)
if (OUT / "nearest_vs_sharpest.json").exists():
    nv = json.loads((OUT / "nearest_vs_sharpest.json").read_text())["clustering_full_session"]
    ax[1].text(2.6, max(lifts) * 0.95, f"d47c full session:\nsharpest lift {nv['sharpest']['lift']:.2f}\n"
               f"nearest lift {nv['nearest']['lift']:.2f}", fontsize=7.5, va="top")
fig.tight_layout()
fig.savefig(OUT / "fig2_clustering.png")

# ---- Fig 3: resolution gain (band and nearest) ---------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.0))
tk = bv["timing"]
ax[0].bar(["frame", "band"], [tk["frame_rerun"]["tokens_median"], tk["band"]["tokens_median"]], color=[C_FRAME, C_BAND])
ax[0].set_ylabel("visual tokens (median)")
ax[0].set_title("VLM input size", fontsize=9)
ax[1].bar(["frame", "band"], [tk["frame_rerun"]["seconds_median"], tk["band"]["seconds_median"]], color=[C_FRAME, C_BAND])
ax[1].set_ylabel("seconds per image (median)")
ax[1].set_title(f"VLM time per image (profile {tk['band']['profile'][-1]})", fontsize=9)
if (OUT / "nearest_segments.csv").exists():
    import csv
    rows = [r for r in csv.DictReader(open(OUT / "nearest_segments.csv")) if r["changed"] == "True"]
    s = np.array([float(r["s_near_ahead_m"]) for r in rows])
    n = np.array([float(r["n_near_ahead_m"]) for r in rows])
    gs = np.array([float(r["s_gsd_centre_vlm_mm"]) for r in rows])
    gn = np.array([float(r["n_gsd_centre_vlm_mm"]) for r in rows])
    ax[2].scatter(gs, gn, s=6, color=C_NEAR, alpha=0.6)
    lim = [min(gs.min(), gn.min()) * 0.95, max(gs.max(), gn.max()) * 1.05]
    ax[2].plot(lim, lim, color="black", lw=0.8, ls=":")
    ax[2].set_xlabel("sharpest: mm/px at segment centre (VLM input)")
    ax[2].set_ylabel("nearest_sharp: mm/px")
    ax[2].set_title(f"d47c changed segments (n = {len(rows)}): median\n{np.median(gs):.1f} -> {np.median(gn):.1f} mm/px, "
                    f"near edge {np.median(s):.2f} -> {np.median(n):.2f} m", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "fig3_resolution_time.png")
print("figures written")
