"""
Paper figures for the drive-video survey (GeoAI_Video segment mode), from the stored run outputs.

No inference: every number is read from the GeoAI_Video run folders (report.json, sidecars), the
session packages (gnss_fixes.csv, session.json) and the answered VLM batch, so the figures always
match the tables in paper/baseline_evaluation.md §8 and GeoAI_Video docs/research/METRICS.md.
Figures that need VLM answers are skipped (with a note) until the batch is imported.

Output: eval_results/plots/paper/fig_v*.png (200 dpi), same style as generate_paper_figures.py.

Usage:
    venv/Scripts/python.exe scripts/generate_video_figures.py [--geoai-video PATH]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "eval_results" / "plots" / "paper"

# Same palette and rcParams as generate_paper_figures.py (dataviz reference palette, light mode).
BLUE, ORANGE, GREEN, RED, PURPLE = "#2a78d6", "#eb6834", "#2f9e66", "#d64545", "#7b5cc4"
INK, MUTED, GRID = "#1c1917", "#78716c", "#e7e5e4"
STATUS_COLORS = {"send": BLUE, "audit": ORANGE, "skip": "#b8b2ac", "not_covered": RED}

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
})

# Short names and what distinguishes the three drives (focus as logged in session.json).
SESSIONS = {
    "20261026T114332Z_SM-G991B-F0D1_f428": "26 Oct · drive A",
    "20261026T114907Z_SM-G991B-F0D1_293d": "26 Oct · drive B",
    "20261027T050257Z_SM-G991B-F0D1_d47c": "27 Oct · drive C",
}
WGS84_A, WGS84_E2 = 6_378_137.0, 6.694379990141e-3


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print("  wrote", (OUT / name).relative_to(PROJECT_ROOT))


def title(ax, text: str) -> None:
    ax.set_title(text, loc="left", fontsize=9.5, color=INK)


class Run:
    """One session's segment-mode outputs and its package metadata."""

    def __init__(self, gv: Path, sid: str, runs: str = "runs_segments"):
        self.sid, self.name = sid, SESSIONS.get(sid, sid[-4:])
        self.dir = gv / "field" / runs / sid
        self.session_dir = gv / "field" / "sessions" / sid
        self.report = json.loads((self.dir / "report.json").read_text(encoding="utf-8"))
        self.recs = sorted((json.loads(p.read_text(encoding="utf-8")) for p in (self.dir / "sidecars").glob("*.json")),
                           key=lambda r: r["segment"]["k"])
        meta = json.loads((self.session_dir / "session.json").read_text(encoding="utf-8"))
        focus = (meta.get("capture_settings") or {}).get("focus") or {}
        dpt = (meta.get("capture_settings") or {}).get("focus_distance_diopters")
        self.focus = f"{focus.get('method') or 'autofocus'} {dpt:.2f} dpt" if dpt is not None else "unknown"
        self.covered = [r for r in self.recs if r["status"] != "not_covered"]

    @property
    def km(self) -> float:
        return self.report["coverage"]["covered_m"] / 1000.0


def enu(lat, lon, lat0, lon0):
    """Local metres east/north around (lat0, lon0) with the WGS-84 radii of curvature there (meridional
    for north, prime vertical for east): < 1 mm error over a few km. A mean-Earth sphere would overstate
    north-south distances by ~0.5 % at Bengaluru's latitude, i.e. 5 m segments would read 5.02 m."""
    s2 = math.sin(math.radians(lat0)) ** 2
    m = WGS84_A * (1 - WGS84_E2) / (1 - WGS84_E2 * s2) ** 1.5
    n = WGS84_A / math.sqrt(1 - WGS84_E2 * s2)
    x = np.radians(np.asarray(lon) - lon0) * n * math.cos(math.radians(lat0))
    y = np.radians(np.asarray(lat) - lat0) * m
    return x, y


# ---------------------------------------------------------------- figures without VLM answers
def fig_routes(runs: list[Run]) -> None:
    """Every 5 m segment centre on each route, coloured by the distress check's decision."""
    fig, axes = plt.subplots(1, len(runs), figsize=(7.2, 2.9))
    for ax, run in zip(np.atleast_1d(axes), runs):
        lat = np.array([r["segment"]["centre"]["lat"] for r in run.recs])
        lon = np.array([r["segment"]["centre"]["lon"] for r in run.recs])
        x, y = enu(lat, lon, lat.mean(), lon.mean())
        st = [r["status"] for r in run.recs]
        ax.plot(x, y, color=GRID, lw=2.2, zorder=1)
        for s, c in STATUS_COLORS.items():
            m = np.array([v == s for v in st])
            if m.any():
                ax.scatter(x[m], y[m], s=5, color=c, zorder=2, label=f"{s} ({m.sum()})", linewidths=0)
        ax.set_aspect("equal")
        ax.set_xticks([]), ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        span = max(np.ptp(x), np.ptp(y))
        bar = 100 if span < 600 else 200
        x0, y0 = x.min(), y.min() - 0.08 * span
        ax.plot([x0, x0 + bar], [y0, y0], color=INK, lw=1.2)
        ax.text(x0 + bar / 2, y0 - 0.03 * span, f"{bar} m", ha="center", va="top", fontsize=7, color=INK)
        ax.set_title(f"{run.name}\n{run.km:.2f} km · {len(run.recs)} segments", fontsize=8.5, color=INK, loc="left")
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.08), fontsize=7, ncol=2, handletextpad=0.2,
                  columnspacing=0.8, markerscale=1.8)
    fig.suptitle("One best frame per 5 m: segment centres and the uncalibrated distress check",
                 x=0.01, ha="left", fontsize=9.5, color=INK)
    save(fig, "fig_v1_routes_gate.png")


def fig_sharpness(runs: list[Run]) -> None:
    """Chosen-frame sharpness per drive (normalised Tenengrad, exposure-invariant), log scale."""
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    data = [np.array([r["quality"]["sharpness"] for r in run.covered if r["quality"]["sharpness"]]) for run in runs]
    pos = np.arange(len(runs))
    vp = ax.violinplot([np.log10(d) for d in data], positions=pos, vert=False, widths=0.75, showextrema=False)
    for b in vp["bodies"]:
        b.set_facecolor(BLUE), b.set_alpha(0.25), b.set_edgecolor("none")
    for i, d in enumerate(data):
        q1, med, q3 = np.percentile(d, [25, 50, 75])
        ax.plot(np.log10([q1, q3]), [i, i], color=BLUE, lw=3, solid_capstyle="butt")
        ax.plot(np.log10(med), i, "o", color="white", mec=INK, ms=5)
        ax.text(np.log10(np.percentile(d, 99)) + 0.05, i, f"median {med:.3f}", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(pos, [f"{r.name}\n{r.focus}" for r in runs], fontsize=7.5)
    ax.invert_yaxis()
    ticks = [0.003, 0.01, 0.03, 0.1, 0.3, 1]
    ax.set_xticks(np.log10(ticks), [str(t) for t in ticks])
    ax.set_xlabel("Best-frame sharpness on the segment rows (normalised Tenengrad, log)")
    fig.suptitle("Focus from geometry (drive C): best frames ~3.5× sharper than with autofocus",
                 x=0.01, ha="left", fontsize=9.5, color=INK)
    save(fig, "fig_v2_sharpness_by_focus.png")


def fig_blur(runs: list[Run]) -> None:
    """Predicted blur of the chosen frame vs speed: motion and rotation parts, and the 4 px limit."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    colors = [BLUE, ORANGE, GREEN]
    for run, c in zip(runs, colors):
        sp = np.array([r["camera"]["speed_mps"] * 3.6 for r in run.covered])
        mo = np.array([r["quality"]["blur_motion_px"] or np.nan for r in run.covered])
        ro = np.array([r["quality"]["blur_rotation_px"] or np.nan for r in run.covered])
        axes[0].scatter(sp, mo, s=4, color=c, alpha=0.55, linewidths=0, label=run.name)
        axes[1].scatter(sp, ro, s=4, color=c, alpha=0.55, linewidths=0)
    for ax, t in zip(axes, ("(a) Motion smear", "(b) Camera rotation (gyro)")):
        ax.axhline(4.0, color=RED, lw=1, ls="--")
        ax.set_xlabel("Speed (km/h)")
        ax.set_yscale("log")
        title(ax, t)
    axes[0].text(1, 4.3, "selection limit 4 px", color=RED, fontsize=7)
    axes[0].set_ylabel("Predicted blur (px at 4K)")
    axes[0].legend(fontsize=7, markerscale=2.5, loc="lower right")
    fig.suptitle("Predicted blur of each chosen frame stays below the 4 px limit up to 45 km/h",
                 x=0.01, ha="left", fontsize=9.5, color=INK)
    save(fig, "fig_v3_blur_vs_speed.png")


def fig_extraction_time(runs: list[Run]) -> None:
    """Extraction time per km of road by stage (A5000, after the speed fixes of 27 Oct)."""
    stages = [("bank_decode", "Bank: decode", "#9cc3ee"), ("bank_features", "Bank: DINOv2 + cracks", BLUE),
              ("bank_finalize", "Bank: coreset", "#1b4f8f"), ("decode_only", "Decode + select", "#f4b183"),
              ("gate", "Distress check", ORANGE), ("outputs", "Records + PNG", MUTED)]
    fig, ax = plt.subplots(figsize=(6.8, 3.1))
    for i, run in enumerate(runs):
        t = dict(run.report["timings_s"])
        t["decode_only"] = t["decode_pass"] - t["gate"]
        left = 0.0
        for key, lab, col in stages:
            v = t.get(key, 0.0) / run.km
            ax.barh(i, v, left=left, color=col, height=0.6, label=lab if i == 0 else None)
            left += v
        ax.text(left + 8, i, f"{left:.0f} s/km · {t['total']:.0f} s total", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(range(len(runs)), [f"{r.name}\n{r.km:.2f} km" for r in runs], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Seconds per km of road (RTX A5000, 4K HEVC)")
    ax.set_xlim(0, max(r.report["timings_s"]["total"] / r.km for r in runs) * 1.45)
    title(ax, "Extraction takes 3.6–5.3 min per km of road on the A5000")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.26), ncol=3, fontsize=7)
    save(fig, "fig_v4_extraction_time.png")


def _fixes(session_dir: Path):
    out = []
    with open(session_dir / "gnss_fixes.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["lat_deg"] and row["t_ns"] and float(row["h_acc_m"] or 99) <= 25:
                out.append((int(row["t_ns"]), float(row["lat_deg"]), float(row["lon_deg"])))
    return sorted(out)


def fig_gps(runs: list[Run]) -> None:
    """Position check, independent of the pipeline: (a) camera position vs linear interpolation of the
    raw phone fixes at the frame time; (b) centre-to-centre distance vs how much the road turns."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    colors = [BLUE, ORANGE, GREEN]
    allerr = []
    for run, c in zip(runs, colors):
        fx = _fixes(run.session_dir)
        ft = np.array([f[0] for f in fx], dtype=np.float64)
        err = []
        for r in run.covered:
            t = r["frame"]["t_ns"] + (r["capture"]["exposure_ns"] or 0) / 2 + (r["capture"]["rs_skew_ns"] or 0) / 2
            i = int(np.searchsorted(ft, t)) - 1
            if i < 0 or i + 1 >= len(fx):
                continue
            w = (t - ft[i]) / (ft[i + 1] - ft[i])
            la = fx[i][1] + w * (fx[i + 1][1] - fx[i][1])
            lo = fx[i][2] + w * (fx[i + 1][2] - fx[i][2])
            x, y = enu([r["camera"]["lat"]], [r["camera"]["lon"]], la, lo)
            err.append(math.hypot(x[0], y[0]))
        allerr += err
        axes[0].hist(err, bins=np.linspace(0, 0.6, 31), histtype="step", color=c, lw=1.4, label=run.name)
        # (b) spacing vs turn
        cen = [(r["segment"]["centre"]["lat"], r["segment"]["centre"]["lon"]) for r in run.recs]
        d, turn = [], []
        for a, b in zip(run.recs, run.recs[1:]):
            if b["segment"]["partial"]:
                continue
            ca, cb = a["segment"]["centre"], b["segment"]["centre"]
            x, y = enu([cb["lat"]], [cb["lon"]], ca["lat"], ca["lon"])
            d.append(math.hypot(x[0], y[0]))
            ha = _seg_heading(a)
            hb = _seg_heading(b)
            turn.append(abs((hb - ha + 180) % 360 - 180))
        axes[1].scatter(turn, d, s=5, color=c, alpha=0.6, linewidths=0)
    axes[0].set_xlabel("Camera position − raw-fix interpolation (m)")
    axes[0].set_ylabel("Best frames")
    axes[0].legend(fontsize=7)
    title(axes[0], f"(a) vs the phone's raw fixes (median {np.median(allerr):.2f} m)")
    th = np.linspace(0, 90, 91)
    axes[1].plot(th, 5.0 * np.sinc(np.radians(th) / (2 * np.pi)), color=INK, lw=1, ls="--")
    axes[1].text(55, 4.75, "chord of a 5 m arc\nturning by that angle", fontsize=7, color=INK)
    axes[1].set_xlabel("Road turn between consecutive segments (°)")
    axes[1].set_ylabel("Centre-to-centre distance (m)")
    title(axes[1], "(b) Spacing of segment centres")
    fig.suptitle("Every best frame has its own position, consistent with the raw GNSS and with 5 m of road",
                 x=0.01, ha="left", fontsize=9.5, color=INK)
    save(fig, "fig_v5_gps_consistency.png")


def _seg_heading(r) -> float:
    s, e = r["segment"]["start"], r["segment"]["end"]
    x, y = enu([e["lat"]], [e["lon"]], s["lat"], s["lon"])
    return math.degrees(math.atan2(x[0], y[0]))


# ---------------------------------------------------------------- figures with VLM answers
def fig_vlm(gv: Path, batch: Path, runs: list[Run]) -> None:
    analysis = batch / "analysis"
    if not (analysis / "label_summary.csv").exists():
        print("  skip VLM figures: run `geoai vlm-import` and `geoai analyze` on", batch, "first")
        return
    _fig_label_share(analysis, runs)
    _fig_vlm_time(batch)
    cal = batch / "gate_calibration.json"
    if cal.exists():
        _fig_gate_curves(json.loads(cal.read_text(encoding="utf-8")), runs)
    else:
        print("  skip gate figure: no", cal)


def _fig_label_share(analysis: Path, runs: list[Run]) -> None:
    rows = list(csv.DictReader(open(analysis / "label_summary.csv", encoding="utf-8-sig")))
    labels = [r["label"] for r in rows if r["scope"] == "all" and r["kind"] == "label" and float(r["length_m"] or 0) > 0]
    labels = sorted(labels, key=lambda lab: -float(next(r["pct_of_surveyed"] for r in rows
                                                         if r["scope"] == "all" and r["label"] == lab)))
    scopes = [("all", f"All drives (n={sum(len(r.recs) for r in runs)} segments)")] + \
        [(r.sid, f"{r.name} (n={len(r.recs)})") for r in runs]
    fig, ax = plt.subplots(figsize=(7.0, 0.32 * len(labels) + 1.4))
    h = 0.8 / len(scopes)
    cols = [INK, BLUE, ORANGE, GREEN]
    for j, ((scope, name), c) in enumerate(zip(scopes, cols)):
        vals = [float(next((r["pct_of_surveyed"] for r in rows if r["scope"] == scope and r["label"] == lab), 0) or 0)
                for lab in labels]
        ax.barh(np.arange(len(labels)) + (j - (len(scopes) - 1) / 2) * h, vals, height=h * 0.92, color=c, label=name)
    ax.set_yticks(range(len(labels)), labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("% of surveyed road length with the label (a segment can carry several labels)")
    fig.suptitle("VLM labels by road length (every 5 m segment classified; the VLM is the reference)",
                 x=0.01, ha="left", fontsize=9.5, color=INK)
    ax.legend(fontsize=7, loc="lower right")
    save(fig, "fig_v6_label_share.png")


def _fig_vlm_time(batch: Path) -> None:
    secs = {"Stage 1 Normal": [], "Stage 1 Distress (+ Stage 2)": []}
    for line in (batch / "responses.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        resp = r.get("response") or {}
        if r.get("error") or not isinstance(r.get("seconds"), (int, float)):
            continue
        key = "Stage 1 Normal" if str(resp.get("stage1_label")).lower() == "normal" else "Stage 1 Distress (+ Stage 2)"
        secs[key].append(r["seconds"])
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    bins = np.linspace(10, 37, 55)
    for (k, v), c in zip(secs.items(), (GREEN, ORANGE)):
        if v:
            ax.hist(v, bins=bins, color=c, alpha=0.6, label=f"{k}: n={len(v)}, median {np.median(v):.1f} s")
    ax.set_xlabel("Seconds per segment frame (round trip to /classify; production profile, probe tiles recorded)")
    ax.set_ylabel("Frames")
    ax.legend(fontsize=7.5)
    title(ax, f"VLM time per 5 m segment frame, by the Stage 1 answer (n={sum(len(v) for v in secs.values())})")
    save(fig, "fig_v7_vlm_time.png")


def _fig_gate_curves(cal: dict, runs: list[Run]) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    names = {"probe_frac": ("Attain probe share", BLUE), "evidence": ("Bank + crack evidence", ORANGE),
             "crack_mm": ("Crack length", PURPLE), "z_max": ("Bank novelty peak", MUTED)}
    for key, (lab, col) in names.items():
        cur = (cal.get("signals") or {}).get(key, {}).get("curve") or []
        if cur:
            ax.plot([p["pass_rate"] for p in cur], [p["recall"] for p in cur], color=col, lw=1.4, label=lab)
    anyc = (cal.get("any_of") or {}).get("curve") or []
    if anyc:
        ax.plot([p["pass_rate"] for p in anyc if p["pass_rate"] is not None],
                [p["recall"] for p in anyc if p["recall"] is not None], "o-", color=INK, ms=3, lw=1.2,
                label="Best any-of thresholds (fitted in-sample)")
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1)
    # Pass rate of the default thresholds over all segments (weighted by segment count, not a mean of sessions).
    n_pass = sum(sum(1 for r in run.covered if (r.get("gate") or {}).get("decision_if_gated") == "send") for run in runs)
    gate_pass = n_pass / sum(len(run.covered) for run in runs)
    ax.axvline(gate_pass, color=RED, lw=1, ls=":")
    ax.text(gate_pass - 0.01, 0.22, f"default thresholds\npass {gate_pass:.1%}", color=RED, fontsize=7,
            ha="right", va="bottom")
    ax.set_xlabel("Share of segments sent to the VLM (pass rate)")
    ax.set_ylabel("Recall of VLM-distress segments")
    ax.set_xlim(0, 1), ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Distress check: recall of VLM-distress segments vs share sent to the VLM (n=813)",
                 x=0.01, ha="left", fontsize=9.5, color=INK)
    save(fig, "fig_v8_gate_curves.png")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geoai-video", type=Path, default=PROJECT_ROOT.parent / "GeoAI_Video")
    ap.add_argument("--batch", type=Path, default=None, help="answered send-all batch (default field/batches/batch_001)")
    a = ap.parse_args()
    gv = a.geoai_video
    OUT.mkdir(parents=True, exist_ok=True)
    runs = [Run(gv, sid) for sid in SESSIONS if (gv / "field" / "runs_segments" / sid / "report.json").exists()]
    if not runs:
        print("no segment runs under", gv / "field" / "runs_segments")
        return 1
    print(f"Drive-video figures from {len(runs)} sessions -> {OUT.relative_to(PROJECT_ROOT)}")
    fig_routes(runs)
    fig_sharpness(runs)
    fig_blur(runs)
    fig_extraction_time(runs)
    fig_gps(runs)
    fig_vlm(gv, a.batch or gv / "field" / "batches" / "batch_001", runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
