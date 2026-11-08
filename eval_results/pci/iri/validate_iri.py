"""
Pre-registered validation of the IMU roughness feature (criteria: VALIDATION.md Part A, written before results).
Reads research.segment_roughness / session_roughness (scripts/iri_from_imu.py) + video_segments + assessments
from the given database, runs the synthetic check, writes validation.json next to this file. CPU only.

    venv/Scripts/python.exe eval_results/pci/iri/validate_iri.py --database geoai
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import psycopg  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from app import pci_iri as P  # noqa: E402
from scripts import setup_local_db  # noqa: E402

DX = 0.025
SEDAN = {"k1": 520.0, "k2": 40.0, "c": 3.6, "mu": 0.12}
B_KW = dict(mount_hz=18.0, mount_zeta=0.1, noise=0.03, scale=1.1, tilt_deg=25.0, jitter_ns=2e5)
NBOOT = 2000


def boot_ci(groups: dict, stat, n=NBOOT, seed=0):
    """Percentile CI of stat(list of rows) resampling whole clusters."""
    keys = list(groups)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = rng.choice(len(keys), len(keys), replace=True)
        rows = [r for i in pick for r in groups[keys[i]]]
        try:
            v = stat(rows)
        except Exception:
            continue
        if v is not None and np.isfinite(v):
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if len(vals) > 100 else None


# ---------------------------------------------------------------- V1 synthetic

def _sections(y, res):
    hi = len(y) * DX - 30
    secs = [b for b in P.sections_along_session(res["segments"]) if b["s0_m"] >= 30 and b["s1_m"] <= hi]
    true = [P.iri_from_profile(y[int((b["s0_m"] - 11) / DX):int(b["s1_m"] / DX)], DX) for b in secs]
    return secs, true


def v1():
    roads = []
    i = 0
    for gd in (4e-6, 16e-6, 64e-6, 256e-6):
        for w in (1.8, 2.0, 2.3):
            roads.append({"i": i, "gd": gd, "w": w, "y": P.make_profile(600, DX, gd, w, rng=np.random.default_rng(i))})
            i += 1
    out = {"roads": []}
    a_err, b_pairs = [], []
    vrng = np.random.default_rng(99)
    for r in roads:
        y = r["y"]
        rec = {"gd": r["gd"], "w": r["w"]}
        if r["w"] == 2.0:
            s = P.synthetic_session(y, DX, np.full(len(y), 6.0), P.GOLDEN_CAR, rng=np.random.default_rng(1000 + r["i"]))
            secs, true = _sections(y, P.run_session(s["imu"], s["gnss"], s["segments"]))
            ratios = [b["iri_gc_uncal"] / t for b, t in zip(secs, true)]
            a_err += [abs(x - 1) for x in ratios]
            rec["A_ratio_median"] = float(np.median(ratios))
        v = float(vrng.uniform(3, 10))
        s = P.synthetic_session(y, DX, np.full(len(y), v), SEDAN, rng=np.random.default_rng(2000 + r["i"]), **B_KW)
        secs, true = _sections(y, P.run_session(s["imu"], s["gnss"], s["segments"]))
        for b, t in zip(secs, true):
            b_pairs.append((b["iri_gc_uncal"], t))
        rec.update(B_speed=v, B_true_mean=float(np.mean(true)), B_est_mean=float(np.mean([b["iri_gc_uncal"] for b in secs])))
        out["roads"].append(rec)
    est, tru = np.array(b_pairs).T
    out["S1_median_abs_rel_err"] = float(np.median(a_err))
    out["S1_pass"] = out["S1_median_abs_rel_err"] <= 0.10
    out["S2_spearman"] = float(stats.spearmanr(est, tru).statistic)
    out["S2_n_sections"] = len(est)
    out["S2_pass"] = out["S2_spearman"] >= 0.90
    rat = est / tru
    out["S3_ratio_median"] = float(np.median(rat))
    out["S3_ratio_iqr"] = [float(np.percentile(rat, 25)), float(np.percentile(rat, 75))]
    s4 = {}
    for w in (1.8, 2.0, 2.3):
        y = P.make_profile(600, DX, 64e-6, w, rng=np.random.default_rng(500))
        means = []
        for v in (3.0, 6.0, 10.0):
            s = P.synthetic_session(y, DX, np.full(len(y), v), SEDAN, rng=np.random.default_rng(3000), **B_KW)
            secs, _ = _sections(y, P.run_session(s["imu"], s["gnss"], s["segments"]))
            means.append(float(np.mean([b["rms_av_norm"] for b in secs])))
        s4[str(w)] = {"rms_av_norm_by_speed_3_6_10": means, "cv": float(np.std(means) / np.mean(means))}
    out["S4"] = s4
    out["S4_pass"] = s4["2.0"]["cv"] <= 0.15
    return out


# ---------------------------------------------------------------- real data

def load(dsn, fp):
    with psycopg.connect(dsn) as c:
        rows = c.execute("""
            SELECT r.segment_id, r.session_id, r.k, r.speed_mps, r.rms_av, r.rms_av_norm, r.iri_gc_uncal, r.peak_av,
                   r.duration_s, r.flags, r.error, g.s0_m, g.s1_m, g.centre_lat, g.centre_lon,
                   (g.record->'camera'->>'heading_deg')::float8, g.assessment_id IS NOT NULL,
                   COALESCE(a.distress_types ? 'Potholes', false)
            FROM research.segment_roughness r JOIN video_segments g ON g.segment_id = r.segment_id
            LEFT JOIN assessments a ON a.id = g.assessment_id
            WHERE r.method_fp = %s""", (fp,)).fetchall()
        sess = c.execute("SELECT session_id, summary, error FROM research.session_roughness WHERE method_fp = %s",
                         (fp,)).fetchall()
    keys = ("segment_id", "session_id", "k", "speed_mps", "rms_av", "rms_av_norm", "iri_gc_uncal", "peak_av",
            "duration_s", "flags", "error", "s0_m", "s1_m", "lat", "lon", "heading", "classified", "pothole")
    return [dict(zip(keys, r)) for r in rows], sess


def usable(r):
    return r["error"] is None and r["rms_av"] is not None and "low_speed" not in (r["flags"] or [])


def v2(rows, sess):
    idle = [s[1]["idle"]["rms_av"] for s in sess if s[1] and s[1].get("idle", {}).get("rms_av") is not None]
    moving = [r["rms_av"] for r in rows if usable(r)]
    out = {"n_drives_with_idle": len(idle), "idle_rms_by_drive": idle,
           "idle_s_total": float(sum(s[1]["idle"]["idle_s"] for s in sess if s[1] and s[1].get("idle"))),
           "moving_rms_median": float(np.median(moving))}
    if len(idle) < 3:
        out["pass"] = None
        out["verdict"] = "not testable (< 3 drives with an idle period)"
        return out
    out["idle_rms_median"] = float(np.median(idle))
    out["ratio"] = out["idle_rms_median"] / out["moving_rms_median"]
    out["pass"] = out["ratio"] <= 0.25
    return out


def match_pairs(rows):
    use = [r for r in rows if usable(r) and r["lat"] is not None and r["heading"] is not None]
    lat0 = np.mean([r["lat"] for r in use])
    lon0 = np.mean([r["lon"] for r in use])
    kx = 111195.0 * math.cos(math.radians(lat0))
    xy = np.array([((r["lon"] - lon0) * kx, (r["lat"] - lat0) * 111195.0) for r in use])
    tree = cKDTree(xy)
    pairs = []          # (i, j) nearest partner j of i in each other drive
    for i, r in enumerate(use):
        best = {}
        for j in tree.query_ball_point(xy[i], 6.0):
            q = use[j]
            if q["session_id"] == r["session_id"]:
                continue
            dh = abs((r["heading"] - q["heading"] + 180) % 360 - 180)
            if dh >= 30:
                continue
            d = float(np.hypot(*(xy[i] - xy[j])))
            if q["session_id"] not in best or d < best[q["session_id"]][0]:
                best[q["session_id"]] = (d, j)
        for sid, (d, j) in best.items():
            pairs.append((i, j))
    return use, pairs


def _agg(rs):
    return P.aggregate([{**r, "flags": r["flags"] or []} for r in rs])


def v3_v4(rows):
    use, pairs = match_pairs(rows)
    out = {"n_segment_pairs": len(pairs)}
    # blocks: drive A, drive B, k // 20 of A
    blocks = {}
    for i, j in pairs:
        a, b = use[i], use[j]
        blocks.setdefault((a["session_id"], b["session_id"], a["k"] // 20), []).append((a, b))
    bl = []
    for (sa, sb, kb), lst in blocks.items():
        if len(lst) < 16:
            continue
        A, B = _agg([x[0] for x in lst]), _agg([x[1] for x in lst])
        if A["rms_av"] is None or B["rms_av"] is None:
            continue
        bl.append({"pair": tuple(sorted((sa, sb))), "A": A, "B": B, "dirpair": (sa, sb)})
    # each unordered block appears twice (A->B and B->A); keep one direction: sa < sb
    bl = [x for x in bl if x["dirpair"][0] < x["dirpair"][1]]
    out["n_blocks"] = len(bl)
    out["n_drive_pairs_with_blocks"] = len({x["pair"] for x in bl})
    res = {}
    for feat in ("rms_av_norm", "iri_gc_uncal", "rms_av"):
        a = np.array([x["A"][feat] for x in bl], float)
        b = np.array([x["B"][feat] for x in bl], float)
        ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
        groups = {}
        for x, okk in zip(bl, ok):
            if okk:
                groups.setdefault(x["pair"], []).append((x["A"][feat], x["B"][feat]))
        if ok.sum() < 3:
            res[feat] = {"n": int(ok.sum())}
            continue
        rho = float(stats.spearmanr(a[ok], b[ok]).statistic)
        mad = float(np.median(np.abs(np.log(a[ok] / b[ok]))))
        res[feat] = {"n": int(ok.sum()), "spearman": rho, "median_abs_log_ratio": mad,
                     "median_ratio_pct": float((math.exp(mad) - 1) * 100),
                     "spearman_ci": boot_ci(groups, lambda rs: stats.spearmanr([r[0] for r in rs], [r[1] for r in rs]).statistic),
                     "mad_ci": boot_ci(groups, lambda rs: float(np.median(np.abs(np.log(np.array([r[0] for r in rs]) / np.array([r[1] for r in rs]))))))}
    out["blocks"] = res
    p = res.get("rms_av_norm", {})
    if out["n_blocks"] < 10:
        out["R1_pass"] = out["R2_pass"] = None
    else:
        out["R1_pass"] = p["spearman"] >= 0.70
        out["R2_pass"] = p["median_abs_log_ratio"] <= math.log(1.25)
    # R3: segment level (unordered pairs once)
    seen, sa, sb = set(), [], []
    for i, j in pairs:
        key = tuple(sorted((use[i]["segment_id"], use[j]["segment_id"])))
        if key in seen:
            continue
        seen.add(key)
        sa.append(use[i]["rms_av_norm"])
        sb.append(use[j]["rms_av_norm"])
    out["R3_segment_spearman"] = float(stats.spearmanr(sa, sb).statistic) if len(sa) > 3 else None
    out["R3_n"] = len(sa)
    # V4: within-location speed exponent
    X, Y, grp = [], [], {}
    seen = set()
    for i, j in pairs:
        a, b = use[i], use[j]
        key = tuple(sorted((a["segment_id"], b["segment_id"])))
        if key in seen:
            continue
        seen.add(key)
        lv = math.log(a["speed_mps"] / b["speed_mps"])
        if abs(lv) < math.log(1.1):
            continue
        ly = math.log(a["rms_av"] / b["rms_av"])
        X.append(lv)
        Y.append(ly)
        grp.setdefault(tuple(sorted((a["session_id"], b["session_id"]))), []).append((lv, ly))

    def slope(rs):
        x = np.array([r[0] for r in rs])
        y = np.array([r[1] for r in rs])
        return float(np.polyfit(x, y, 1)[0]) if len(x) > 5 and np.ptp(x) > 0 else None
    v4 = {"n_pairs": len(X)}
    if len(X) > 10:
        beta = slope(list(zip(X, Y)))
        ci = boot_ci(grp, slope)
        v4.update(beta=beta, beta_ci=ci, n_drive_pairs=len(grp),
                  pass_=bool(abs(beta - 0.5) <= 0.15 or (ci is not None and ci[0] <= 0.5 <= ci[1])))
    else:
        v4["pass_"] = None
    al = [r for r in rows if usable(r)]
    v4["across_spearman_rms_vs_speed"] = float(stats.spearmanr([r["rms_av"] for r in al], [r["speed_mps"] for r in al]).statistic)
    v4["across_spearman_norm_vs_speed"] = float(stats.spearmanr([r["rms_av_norm"] for r in al], [r["speed_mps"] for r in al]).statistic)
    v4["speed_mps_percentiles_5_50_95"] = [float(np.percentile([r["speed_mps"] for r in al], q)) for q in (5, 50, 95)]
    return out, v4


def auc(pos, neg):
    if not pos or not neg:
        return None
    u = stats.mannwhitneyu(pos, neg, alternative="two-sided").statistic
    return float(u / (len(pos) * len(neg)))


def v5(rows):
    by = {}
    for r in rows:
        by.setdefault(r["session_id"], {})[r["k"]] = r
    out = {}
    for name, feat, ahead in (("primary_max_norm_k_to_k6", "rms_av_norm", 6), ("peak_av_k_to_k6", "peak_av", 6),
                              ("own_segment_norm", "rms_av_norm", 0)):
        per_drive = {}
        for sid, ks in by.items():
            for k, r in ks.items():
                if not r["classified"]:
                    continue
                vals = [ks[j][feat] for j in range(k, k + ahead + 1) if j in ks and usable(ks[j]) and ks[j][feat] is not None]
                if len(vals) < (4 if ahead else 1):
                    continue
                per_drive.setdefault(sid, []).append((max(vals), r["pothole"]))
        allr = [x for v in per_drive.values() for x in v]
        pos = [s for s, p in allr if p]
        neg = [s for s, p in allr if not p]
        a = auc(pos, neg)
        ci = boot_ci(per_drive, lambda rs: auc([s for s, p in rs if p], [s for s, p in rs if not p]))
        out[name] = {"n_pos": len(pos), "n_neg": len(neg), "auc": a, "auc_ci": ci,
                     "median_pos": float(np.median(pos)) if pos else None, "median_neg": float(np.median(neg)) if neg else None}
    p = out["primary_max_norm_k_to_k6"]
    out["pass"] = (p["auc"] is not None and p["auc"] >= 0.60 and p["auc_ci"] is not None and p["auc_ci"][0] > 0.5)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--database", default="geoai")
    ap.add_argument("--skip-synthetic", action="store_true")
    a = ap.parse_args(argv)
    dsn = setup_local_db.with_database(setup_local_db.database_url(), a.database)
    fp = P.method_fp()
    rows, sess = load(dsn, fp)
    res = {"method_fp": fp, "database": a.database, "n_rows": len(rows), "n_usable": sum(usable(r) for r in rows),
           "n_drives": len(sess), "n_drives_failed": sum(1 for s in sess if s[2]),
           "flag_counts": {}}
    for r in rows:
        for f in r["flags"] or []:
            res["flag_counts"][f] = res["flag_counts"].get(f, 0) + 1
    res["error_counts"] = {}
    for r in rows:
        if r["error"]:
            res["error_counts"][r["error"][:60]] = res["error_counts"].get(r["error"][:60], 0) + 1
    res["V2"] = v2(rows, sess)
    res["V3"], res["V4"] = v3_v4(rows)
    res["V5"] = v5(rows)
    if not a.skip_synthetic:
        res["V1"] = v1()
    (HERE / "validation.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
