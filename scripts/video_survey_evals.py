"""Reliability and validity checks for the drive-video survey (paper §8.8). No inference.

Reads every stored drive: the first three (send-all batch_001) and every run the video_pipeline
toggle stored under VIDEO_RUNS_ROOT (segments_results.csv), plus the toggle's per-session timings.

1. Repeatability: where two drives passed the same spot, do they get the same answer? Mutual nearest
   segment centres from different drives within R metres; Cohen's kappa for distress yes/no and for
   Potholes, with a bootstrap over drive pairs (pairs from one drive pair are not independent).
2. Position signal: kappa as a function of the distance between the paired centres. If positions
   carry information, agreement falls as the centres move apart.
3. Image-quality confounds: distress and review rates by ISO, speed, motion blur and sharpness.
   Observational: road, light and speed vary together, so these are associations, not effects.
4. Review load: why answers are flagged (Stage 1 or Stage 2 confidence below 0.80, or other).
5. Throughput model: wall time per drive = fixed overhead + cost per km (least squares).

Writes eval_results/video_evals.json, video_evals.md, plots/paper/fig_v14..v17.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from generate_video_figures import BLUE, INK, MUTED, ORANGE, RED, title  # noqa: E402

OUT_PLOTS = PROJECT_ROOT / "eval_results" / "plots" / "paper"
OUT_JSON = PROJECT_ROOT / "eval_results" / "video_evals.json"
OUT_MD = PROJECT_ROOT / "eval_results" / "video_evals.md"
SNAPSHOT = PROJECT_ROOT / "eval_results" / "video_survey_sessions_snapshot.json"
THRESHOLD = 0.80
SEED = 20261030


def read_csv(p: Path) -> list[dict]:
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fnum(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def truthy(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes")


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def kappa(a: list[int], b: list[int]):
    n = len(a)
    if n == 0:
        return None, None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return po, (None if pe >= 1 else (po - pe) / (1 - pe))


def load_rows(runs_root: Path, baseline: Path) -> list[dict]:
    rows = []
    srcs = []
    b = baseline / "analysis" / "segments_results.csv"
    if b.exists():
        srcs.append(("send-all", b))
    for p in sorted(runs_root.glob("*/vlm/analysis/segments_results.csv")):
        srcs.append(("gated", p))
    for mode, p in srcs:
        for r in read_csv(p):
            r["_mode"] = mode
            rows.append(r)
    return rows


def answered(rows):
    return [r for r in rows if truthy(r.get("answered")) and r.get("vlm_class") in ("distress", "normal")]


# ---------------------------------------------------------------- 1-2 repeatability
def local_xy(rows):
    lat0 = np.mean([float(r["centre_lat"]) for r in rows])
    lon0 = np.mean([float(r["centre_lon"]) for r in rows])
    k = 111_320.0
    xy = np.array([((float(r["centre_lon"]) - lon0) * k * math.cos(math.radians(lat0)),
                    (float(r["centre_lat"]) - lat0) * k) for r in rows])
    return xy


def pairs_between_drives(rows, max_d: float):
    """Mutual nearest neighbours between every two drives, within max_d metres."""
    xy = local_xy(rows)
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[r["session_id"]].append(i)
    sids = sorted(by)
    out = []
    for ia, sa in enumerate(sids):
        A = np.array(by[sa])
        for sb in sids[ia + 1:]:
            B = np.array(by[sb])
            d = np.linalg.norm(xy[A][:, None, :] - xy[B][None, :, :], axis=2)
            if d.min() > max_d:
                continue
            nb = d.argmin(axis=1)
            na = d.argmin(axis=0)
            for i, j in enumerate(nb):
                if na[j] == i and d[i, j] <= max_d:
                    ra, rb = rows[A[i]], rows[B[j]]
                    ha, hb = fnum(ra.get("heading_deg")), fnum(rb.get("heading_deg"))
                    dh = None if ha is None or hb is None else abs((ha - hb + 180) % 360 - 180)
                    out.append({"a": int(A[i]), "b": int(B[j]), "d": float(d[i, j]), "dh": dh,
                                "pair": (sa, sb)})
    return out


def agreement(rows, prs, fn):
    a = [fn(rows[p["a"]]) for p in prs]
    b = [fn(rows[p["b"]]) for p in prs]
    po, k = kappa(a, b)
    n = len(a)
    pa, pb = (sum(a) / n, sum(b) / n) if n else (0, 0)
    either = sum(1 for x, y in zip(a, b) if x or y)
    return {"n_pairs": n, "agreement": None if po is None else round(po, 4),
            "chance_agreement": round(pa * pb + (1 - pa) * (1 - pb), 4) if n else None,
            "pairs_with_any_distress": either,
            "positive_agreement": round(sum(1 for x, y in zip(a, b) if x and y) / either, 4) if either else None,
            "kappa": None if k is None else round(k, 4),
            "positive_rate": round((sum(a) + sum(b)) / (2 * len(a)), 4) if a else None}


def boot_kappa(rows, prs, fn, B=2000):
    groups = defaultdict(list)
    for p in prs:
        groups[p["pair"]].append(p)
    keys = list(groups)
    if len(keys) < 3:
        return None
    rng = random.Random(SEED)
    ks = []
    for _ in range(B):
        sample = [p for k in (rng.choice(keys) for _ in keys) for p in groups[k]]
        _, k = kappa([fn(rows[p["a"]]) for p in sample], [fn(rows[p["b"]]) for p in sample])
        if k is not None:
            ks.append(k)
    ks.sort()
    return [round(ks[int(0.025 * len(ks))], 4), round(ks[int(0.975 * len(ks)) - 1], 4)] if ks else None


def is_dist(r):
    return int(r.get("vlm_class") == "distress")


def has_pothole(r):
    return int("Potholes" in (r.get("vlm_distress_labels") or ""))


def repeatability(rows):
    ans = answered(rows)
    prs = pairs_between_drives(ans, 20.0)
    res = {"definition": "mutual nearest answered segment centres of two different drives",
           "by_distance": [], "same_direction_le5m": None}
    for lo, hi in ((0, 2), (2, 5), (5, 10), (10, 20)):
        sel = [p for p in prs if lo <= p["d"] < hi]
        e = agreement(ans, sel, is_dist)
        e.update({"from_m": lo, "to_m": hi, "kappa_ci95": boot_kappa(ans, sel, is_dist)})
        res["by_distance"].append(e)
    close = [p for p in prs if p["d"] < 5]
    same = [p for p in close if p["dh"] is not None and p["dh"] <= 45]
    opp = [p for p in close if p["dh"] is not None and p["dh"] >= 135]
    for name, sel in (("le5m_all", close), ("le5m_same_direction", same), ("le5m_opposite_direction", opp)):
        e = agreement(ans, sel, is_dist)
        e["kappa_ci95"] = boot_kappa(ans, sel, is_dist)
        e["drive_pairs"] = len({p["pair"] for p in sel})
        res[name] = e
        p = agreement(ans, sel, has_pothole)
        p["kappa_ci95"] = boot_kappa(ans, sel, has_pothole)
        res[name + "_potholes"] = p
    # same-day vs cross-day revisits
    day = lambda i: ans[i]["session_id"][:8]
    for name, sel in (("le5m_same_day", [p for p in close if day(p["a"]) == day(p["b"])]),
                      ("le5m_different_day", [p for p in close if day(p["a"]) != day(p["b"])])):
        e = agreement(ans, sel, is_dist)
        e["kappa_ci95"] = boot_kappa(ans, sel, is_dist)
        res[name] = e
    # Kappa needs both answers to vary: with few revisits, nearly all Normal, it is not estimable.
    c = res["le5m_all"]
    res["sufficient"] = bool(c["n_pairs"] >= 100 and c["drive_pairs"] >= 5 and c["pairs_with_any_distress"] >= 20)
    res["note"] = ("enough revisits for a kappa estimate" if res["sufficient"] else
                   f"insufficient overlap: {c['n_pairs']} revisit pairs from {c['drive_pairs']} drive pairs, "
                   f"{c['pairs_with_any_distress']} with any distress; kappa is not interpretable. Needs repeat "
                   "drives of the same roads.")
    res["_pairs"] = prs
    return res


# ---------------------------------------------------------------- 3 confounds
BINS = {
    "iso": [0, 200, 800, 1600, 1e9],
    "speed_mps": [0, 3, 6, 9, 1e9],
    "blur_px": [0, 0.5, 1, 2, 1e9],
}


def binned(ans, key, edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in ans if (v := fnum(r.get(key))) is not None and lo <= v < hi]
        k = sum(is_dist(r) for r in sel)
        rv = sum(truthy(r.get("vlm_needs_expert_review")) for r in sel)
        out.append({"from": lo, "to": None if hi >= 1e9 else hi, "n": len(sel),
                    "distress_rate": round(k / len(sel), 4) if sel else None, "distress_ci95": wilson(k, len(sel)),
                    "review_rate": round(rv / len(sel), 4) if sel else None})
    return out


def quantile_bins(ans, key, q=4):
    vals = sorted(v for r in ans if (v := fnum(r.get(key))) is not None)
    if len(vals) < q * 10:
        return []
    edges = [vals[0]] + [vals[int(len(vals) * i / q)] for i in range(1, q)] + [vals[-1] + 1e-9]
    return binned(ans, key, edges)


def spearman(x, y):
    def rank(v):
        o = np.argsort(v)
        r = np.empty(len(v))
        r[o] = np.arange(len(v))
        return r
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.corrcoef(rank(x), rank(y))[0, 1]) if len(x) > 2 else None


def confounds(ans):
    res = {k: binned(ans, k, e) for k, e in BINS.items()}
    res["sharpness_quartiles"] = quantile_bins(ans, "sharpness")
    # Pooled bins mix roads: a slow, dark or blurry drive may simply be a worse road. Inside each
    # drive the road mix is held roughly fixed, so compare each segment with its own drive:
    # rank correlation per drive (drive-weighted mean), and the distress rate of segments below vs
    # above their drive's median value, pooled over drives.
    res["within_drive"] = {k: within_drive(ans, k) for k in ("iso", "speed_mps", "blur_px", "sharpness")}
    return res


def within_drive(ans, key):
    corr, acc = [], [0, 0, 0, 0]  # below-median distress, n, above-median distress, n
    for rs in group(ans).values():
        xs = [(fnum(r.get(key)), is_dist(r)) for r in rs if fnum(r.get(key)) is not None]
        if len(xs) < 30 or len({x for x, _ in xs}) <= 3 or not 0 < sum(d for _, d in xs) < len(xs):
            continue
        corr.append((len(xs), spearman([x for x, _ in xs], [d for _, d in xs])))
        med = float(np.median([x for x, _ in xs]))
        for x, d in xs:
            if x < med:
                acc[0] += d
                acc[1] += 1
            elif x > med:
                acc[2] += d
                acc[3] += 1
    return {"n_drives": len(corr),
            "spearman_drive_weighted": round(sum(n * r for n, r in corr) / sum(n for n, _ in corr), 4) if corr else None,
            "below_median_rate": round(acc[0] / acc[1], 4) if acc[1] else None,
            "below_median_ci95": wilson(acc[0], acc[1]),
            "above_median_rate": round(acc[2] / acc[3], 4) if acc[3] else None,
            "above_median_ci95": wilson(acc[2], acc[3])}


def group(rows):
    g = defaultdict(list)
    for r in rows:
        g[r["session_id"]].append(r)
    return g


# ---------------------------------------------------------------- 4 review load
def safety_net_flags() -> dict:
    """segment id -> (safety net flagged, top probe type), from the answers stored in local PostgreSQL."""
    try:
        from dotenv import dotenv_values
        import psycopg
        dsn = dotenv_values(PROJECT_ROOT / ".env").get("DATABASE_URL")
        if not dsn:
            return {}
        out = {}
        with psycopg.connect(dsn, connect_timeout=5) as c:
            cur = c.cursor()
            cur.execute("select v.segment_id, a.raw_response->'vlm_response'->'stage1_safety_net' "
                        "from video_segments v join assessments a on a.id = v.assessment_id")
            for sid, sn in cur.fetchall():
                sn = json.loads(sn) if isinstance(sn, str) else sn
                if isinstance(sn, dict):
                    p = sn.get("p") or {}
                    out[sid] = (bool(sn.get("flagged", sn.get("triggered"))), max(p, key=p.get) if p else None)
        return out
    except Exception as e:  # noqa: BLE001 - the attribution is optional
        print("  safety-net attribution unavailable:", e)
        return {}


def review(ans):
    flagged = [r for r in ans if truthy(r.get("vlm_needs_expert_review"))]
    reasons = Counter()
    sn = safety_net_flags()
    sn_type = Counter()
    for r in flagged:
        s1, s2 = fnum(r.get("vlm_stage1_confidence")), fnum(r.get("vlm_stage2_confidence"))
        low1 = s1 is not None and s1 < THRESHOLD
        low2 = r.get("vlm_class") == "distress" and s2 is not None and s2 < THRESHOLD
        net = sn.get(r.get("id"), (False, None))
        if not (low1 or low2) and net[0]:
            sn_type[net[1]] += 1
        reasons["stage1 and stage2 below 0.80" if low1 and low2 else
                "stage1 below 0.80" if low1 else
                "stage2 below 0.80" if low2 else
                "Stage 1 safety net (Normal, probe sees distress)" if net[0] else
                "other / not attributable"] += 1
    by_class = {}
    for c in ("normal", "distress"):
        sel = [r for r in ans if r.get("vlm_class") == c]
        k = sum(truthy(r.get("vlm_needs_expert_review")) for r in sel)
        by_class[c] = {"n": len(sel), "flagged": k, "rate": round(k / len(sel), 4) if sel else None}
    by_label = {}
    for lab in ("Potholes", "Longitudinal Cracking", "Bleeding", "Transverse Cracking", "Alligator Cracking",
                "Edge Breaking"):
        sel = [r for r in ans if lab in (r.get("vlm_distress_labels") or "")]
        k = sum(truthy(r.get("vlm_needs_expert_review")) for r in sel)
        by_label[lab] = {"n": len(sel), "rate": round(k / len(sel), 4) if sel else None}
    s1 = {c: [fnum(r.get("vlm_stage1_confidence")) for r in ans if r.get("vlm_class") == c
              and fnum(r.get("vlm_stage1_confidence")) is not None] for c in ("normal", "distress")}
    return {"n_answered": len(ans), "n_flagged": len(flagged),
            "flagged_rate": round(len(flagged) / len(ans), 4) if ans else None,
            "reasons": dict(reasons.most_common()), "safety_net_top_type": dict(sn_type.most_common(6)),
            "safety_net_attributed": bool(sn), "by_class": by_class, "by_label": by_label,
            "stage1_conf_median": {c: round(float(np.median(v)), 4) if v else None for c, v in s1.items()},
            "_s1": s1}


# ---------------------------------------------------------------- 5 throughput
def sessions_meta(url: str, key_file: Path):
    try:
        key = key_file.read_text(encoding="utf-8").strip() if key_file.exists() else ""
        req = urllib.request.Request(url.rstrip("/") + "/operator/video/sessions",
                                     headers={"X-API-Key": key} if key else {})
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.load(r)["items"]
    except OSError:
        return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


def throughput(items, rows):
    dist = Counter(r["session_id"] for r in rows if r.get("vlm_class") == "distress")
    pts = []
    stage = Counter()
    seg = 0
    for s in items:
        m = s.get("metrics") or {}
        t = m.get("timings_s") or {}
        if s.get("state") not in ("stored", "cleaned") or t.get("extract") is None or not m.get("route_m"):
            continue
        wall = sum(v for v in t.values() if isinstance(v, (int, float)))
        pts.append((m["route_m"] / 1000, wall, m.get("segments", 0), s["session_id"], dist[s["session_id"]]))
        for k, v in t.items():
            if isinstance(v, (int, float)):
                stage[k] += v
        seg += m.get("segments", 0)
    if len(pts) < 3:
        return {"n_drives": len(pts)}
    km = np.array([p[0] for p in pts])
    w = np.array([p[1] for p in pts])
    nseg = np.array([p[2] for p in pts], float)
    ndis = np.array([p[4] for p in pts], float)
    # Distress answers also run Stage 2 (~13 s vs ~4 s), so time depends on the road, not only on
    # its length: wall = fixed per drive + cost per segment + extra cost per distress answer.
    A = np.vstack([np.ones_like(km), nseg, ndis]).T
    coef, *_ = np.linalg.lstsq(A, w, rcond=None)
    pred = A @ coef
    r2 = 1 - ((w - pred) ** 2).sum() / ((w - w.mean()) ** 2).sum()
    a = float(coef[0])
    b = float(np.sum(coef[1] * nseg + coef[2] * ndis) / km.sum())  # s per km at this survey's mix
    tot = sum(stage.values())
    return {"n_drives": len(pts), "km": round(float(km.sum()), 3), "segments": seg,
            "wall_h": round(tot / 3600, 3), "s_per_segment": round(tot / seg, 3),
            "fixed_overhead_s_per_drive": round(float(a), 1), "s_per_km": round(float(b), 1),
            "min_per_km_marginal": round(float(b) / 60, 2), "r2": round(float(r2), 4),
            "s_per_segment_marginal": round(float(coef[1]), 2), "s_per_distress_answer_extra": round(float(coef[2]), 2),
            "model": "wall_s = fixed + s_per_segment * segments + extra * distress_answers",
            "stage_share": {k: round(v / tot, 4) for k, v in stage.most_common()},
            "stage_s_per_segment": {k: round(v / seg, 3) for k, v in stage.most_common()},
            "_pts": pts}


# ---------------------------------------------------------------- figures
def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT_PLOTS / name, dpi=200)
    plt.close(fig)
    print("  wrote", (OUT_PLOTS / name).relative_to(PROJECT_ROOT))


def fig_repeat(rep):
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    bd = [e for e in rep["by_distance"] if e["n_pairs"]]
    x = np.arange(len(bd))
    ax.bar(x - 0.18, [100 * e["agreement"] for e in bd], 0.36, color=BLUE, label="observed agreement")
    ax.bar(x + 0.18, [100 * e["chance_agreement"] for e in bd], 0.36, color="#b8b2ac", label="expected by chance")
    for i, e in enumerate(bd):
        ax.text(i, 102, f"n={e['n_pairs']}\n{e['pairs_with_any_distress']} w/ distress", ha="center", va="bottom",
                fontsize=6.5, color=MUTED)
    ax.set_ylim(0, 125)
    ax.set_xticks(x, [f"{e['from_m']}–{e['to_m']} m" for e in bd])
    ax.set_xlabel("distance between the two segment centres (different drives)")
    ax.set_ylabel("% of pairs with the same answer")
    ax.legend(fontsize=7, loc="lower left")
    t = "Figure 8.14  Same spot, different drive" + ("" if rep["sufficient"] else ": too few revisits to judge")
    title(ax, t)
    save(fig, "fig_v14_repeatability.png")


def fig_confounds(conf):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
    for ax, (key, lab) in zip(axes, (("iso", "ISO"), ("speed_mps", "speed (m/s)"), ("blur_px", "motion blur (px)"))):
        bs = [b for b in conf[key] if b["n"] >= 20]
        x = np.arange(len(bs))
        r = [100 * b["distress_rate"] for b in bs]
        lo = [r[i] - 100 * bs[i]["distress_ci95"][0] for i in range(len(bs))]
        hi = [100 * bs[i]["distress_ci95"][1] - r[i] for i in range(len(bs))]
        ax.bar(x, r, color=ORANGE, width=0.6)
        ax.errorbar(x, r, yerr=[lo, hi], fmt="none", ecolor=INK, capsize=2, lw=0.8)
        ax.plot(x, [100 * b["review_rate"] for b in bs], "o-", color=BLUE, ms=3, lw=1, label="review rate")
        ax.set_xticks(x, [f"{b['from']:g}–{b['to']:g}" if b["to"] else f"≥{b['from']:g}" for b in bs], fontsize=6.5, rotation=20)
        ax.set_xlabel(f"{lab} (n per bin ≥ 20)", fontsize=8)
    axes[0].set_ylabel("% of answered segments")
    axes[0].legend(fontsize=6.5, loc="upper left")
    title(axes[0], "Figure 8.15  Distress (bars) and review rate by capture condition")
    save(fig, "fig_v15_capture_conditions.png")


def fig_review(rv):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1.2, 1]})
    bins = np.linspace(0.5, 1.0, 26)
    a1.hist(rv["_s1"]["normal"], bins=bins, color=BLUE, alpha=0.85, label="answered Normal")
    a1.hist(rv["_s1"]["distress"], bins=bins, color=RED, alpha=0.7, label="answered Distress")
    a1.axvline(THRESHOLD, color=INK, ls="--", lw=0.8)
    a1.set_xlabel("Stage 1 confidence")
    a1.set_ylabel("segments")
    a1.legend(fontsize=6.5, loc="upper left")
    title(a1, "Figure 8.16a  Stage 1 confidence (gate at 0.80)")
    rs = rv["reasons"]
    a2.barh(list(rs), [100 * v / rv["n_flagged"] for v in rs.values()], color=ORANGE)
    a2.invert_yaxis()
    a2.set_xlabel("% of flagged segments")
    a2.tick_params(axis="y", labelsize=7)
    title(a2, f"8.16b  Why {rv['n_flagged']} were flagged")
    save(fig, "fig_v16_review_reasons.png")


def fig_throughput(tp):
    pts = tp["_pts"]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    km = np.array([p[0] for p in pts])
    w = np.array([p[1] for p in pts]) / 60
    dshare = np.array([p[4] / max(p[2], 1) for p in pts])
    sc = ax.scatter(km, w, s=18, c=100 * dshare, cmap="OrRd", vmin=0, edgecolors=MUTED, linewidths=0.4)
    fig.colorbar(sc, ax=ax, label="% distress answers", pad=0.01)
    fixed = tp["fixed_overhead_s_per_drive"]
    ax.plot([], [], " ", label=(f"{fixed:.0f} s per drive + " if fixed > 30 else "≈0 s per drive + ") +
            f"{tp['s_per_segment_marginal']:.1f} s per segment + {tp['s_per_distress_answer_extra']:.1f} s per "
            f"distress answer (R² {tp['r2']:.3f})")
    ax.set_xlabel("drive length (km)")
    ax.set_ylabel("inbox → map (min)")
    ax.legend(fontsize=7)
    title(ax, f"Figure 8.17  Time from inbox to GIS per drive ({tp['n_drives']} drives)")
    save(fig, "fig_v17_latency_model.png")


# ---------------------------------------------------------------- report
def pc(x):
    return "—" if x is None else f"{100 * x:.1f} %"


def kci(e):
    if e.get("kappa") is None:
        return "—"
    c = e.get("kappa_ci95")
    return f"{e['kappa']:.2f}" + (f" [{c[0]:.2f}, {c[1]:.2f}]" if c else "")


def write_md(out):
    rep, conf, rv, tp = out["repeatability"], out["confounds"], out["review"], out["throughput"]
    L = ["# Drive-video survey: reliability and validity checks", "",
         f"Generated by `scripts/video_survey_evals.py` (no inference) on {out['n_drives']} drives, "
         f"{out['n_answered']} answered segments. JSON: `video_evals.json`.", "",
         "## 1. Repeatability: same spot, different drive", "",
         "Mutual nearest answered segment centres of two different drives. κ = Cohen's kappa; "
         "CIs by bootstrap over drive pairs.", "",
         "| Pairs | n | drive pairs | agreement | κ distress [95 % CI] | κ potholes [95 % CI] |",
         "|---|---|---|---|---|---|"]
    for key, name in (("le5m_all", "≤ 5 m, any direction"), ("le5m_same_direction", "≤ 5 m, same direction"),
                      ("le5m_opposite_direction", "≤ 5 m, opposite direction")):
        e, p = rep[key], rep[key + "_potholes"]
        L.append(f"| {name} | {e['n_pairs']} | {e['drive_pairs']} | {pc(e['agreement'])} | {kci(e)} | {kci(p)} |")
    for key, name in (("le5m_same_day", "≤ 5 m, same day"), ("le5m_different_day", "≤ 5 m, different days")):
        e = rep[key]
        L.append(f"| {name} | {e['n_pairs']} | — | {pc(e['agreement'])} | {kci(e)} | |")
    L += ["", f"**Status:** {rep['note']}", "",
          "| Distance between centres | pairs | agreement | chance | pairs with any distress | both distress |",
          "|---|---|---|---|---|---|"]
    for e in rep["by_distance"]:
        L.append(f"| {e['from_m']}–{e['to_m']} m | {e['n_pairs']} | {pc(e['agreement'])} | {pc(e['chance_agreement'])} | "
                 f"{e['pairs_with_any_distress']} | {pc(e['positive_agreement'])} |")
    L += ["", "## 2. Capture conditions (associations, not effects)", ""]
    for key, name in (("iso", "ISO"), ("speed_mps", "Speed (m/s)"), ("blur_px", "Motion blur (px)"),
                      ("sharpness_quartiles", "Sharpness (quartiles)")):
        L += [f"**{name}**", "", "| bin | n | distress [95 % CI] | review |", "|---|---|---|---|"]
        for b in conf[key]:
            ci = b["distress_ci95"]
            rng = f"{b['from']:.3g}–{b['to']:.3g}" if b["to"] is not None else f"≥ {b['from']:.3g}"
            L.append(f"| {rng} | {b['n']} | {pc(b['distress_rate'])}"
                     f"{'' if not ci else f' [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]'} | {pc(b['review_rate'])} |")
        L.append("")
    L += ["**Within each drive** (removes differences between roads): distress rate of segments below vs above "
          "their own drive's median, and drive-weighted rank correlation.", "",
          "| Variable | drives | below median | above median | Spearman |", "|---|---|---|---|---|"]
    for k, v in conf["within_drive"].items():
        L.append(f"| {k} | {v['n_drives']} | {pc(v['below_median_rate'])} | {pc(v['above_median_rate'])} | "
                 f"{v['spearman_drive_weighted']} |")
    L += ["",
          "## 3. Review load", "",
          f"{rv['n_flagged']} of {rv['n_answered']} answered segments flagged ({pc(rv['flagged_rate'])}). "
          f"Normal answers flagged: {pc(rv['by_class']['normal']['rate'])}; distress answers: "
          f"{pc(rv['by_class']['distress']['rate'])}. Median Stage 1 confidence: Normal "
          f"{rv['stage1_conf_median']['normal']}, Distress {rv['stage1_conf_median']['distress']}.", "",
          "| Reason | segments | share |", "|---|---|---|"]
    for k, v in rv["reasons"].items():
        L.append(f"| {k} | {v} | {pc(v / rv['n_flagged'])} |")
    if rv["safety_net_top_type"]:
        L += ["", "Safety-net flags by the probe's highest type: " +
              ", ".join(f"{k} {v}" for k, v in rv["safety_net_top_type"].items()) + "."]
    L += ["", "| Label | segments | flagged |", "|---|---|---|"]
    for k, v in rv["by_label"].items():
        L.append(f"| {k} | {v['n']} | {pc(v['rate'])} |")
    if tp.get("n_drives", 0) >= 3:
        L += ["", "## 4. Throughput: inbox → GIS", "",
              f"{tp['n_drives']} drives, {tp['km']} km, {tp['segments']} segments, {tp['wall_h']} h: "
              f"**{tp['s_per_segment']} s per segment**. Fit: {tp['fixed_overhead_s_per_drive']:.0f} s per drive + "
              f"{tp['s_per_segment_marginal']} s per segment + {tp['s_per_distress_answer_extra']} s extra per distress "
              f"answer (R² {tp['r2']}); at this survey's mix {tp['min_per_km_marginal']} min per km.", "",
              "| Stage | s per segment | share |", "|---|---|---|"]
        for k, v in tp["stage_share"].items():
            L.append(f"| {k} | {tp['stage_s_per_segment'][k]} | {pc(v)} |")
    L += ["", "Figures: `plots/paper/fig_v14_repeatability.png`, `fig_v15_capture_conditions.png`, "
          "`fig_v16_review_reasons.png`, `fig_v17_latency_model.png`.", ""]
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print("  wrote", OUT_MD.relative_to(PROJECT_ROOT))


def strip(d):
    if isinstance(d, dict):
        return {k: strip(v) for k, v in d.items() if not k.startswith("_")}
    if isinstance(d, list):
        return [strip(v) for v in d]
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", type=Path, default=Path(r"C:\GeoAI\runs"))
    ap.add_argument("--baseline", type=Path,
                    default=PROJECT_ROOT.parent / "GeoAI_Video" / "field" / "batches" / "batch_001")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--key-file", type=Path, default=Path(r"C:\GeoAI\operator_api_key.txt"))
    a = ap.parse_args()
    OUT_PLOTS.mkdir(parents=True, exist_ok=True)
    rows = load_rows(a.runs_root, a.baseline)
    ans = [r for r in answered(rows) if r.get("centre_lat") and r.get("centre_lon")]
    rep = repeatability(ans)
    conf = confounds(ans)
    rv = review(ans)
    tp = throughput(sessions_meta(a.url, a.key_file), rows)
    out = {"generated_by": "scripts/video_survey_evals.py", "n_drives": len({r["session_id"] for r in rows}),
           "n_segments": len(rows), "n_answered": len(ans), "repeatability": rep, "confounds": conf,
           "review": rv, "throughput": tp}
    OUT_JSON.write_text(json.dumps(strip(out), indent=1), encoding="utf-8")
    print("  wrote", OUT_JSON.relative_to(PROJECT_ROOT))
    fig_repeat(rep)
    fig_confounds(conf)
    fig_review(rv)
    if tp.get("n_drives", 0) >= 3:
        fig_throughput(tp)
    write_md(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
