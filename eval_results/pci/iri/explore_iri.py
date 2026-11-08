"""EXPLORATORY analyses (not pre-registered; run after validate_iri.py results were seen). Writes explore.json."""
import json, math, os, sys
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent)); sys.path.insert(0, str(HERE))
import numpy as np
from scipy import signal, stats
from scipy.spatial import cKDTree
import validate_iri as V
from app import pci_iri as P
from scripts import setup_local_db

dsn = setup_local_db.with_database(setup_local_db.database_url(), "geoai")
rows, sess = V.load(dsn, P.method_fp())
use = [r for r in rows if V.usable(r) and r["lat"] is not None and r["heading"] is not None]
lat0 = np.mean([r["lat"] for r in use]); lon0 = np.mean([r["lon"] for r in use])
kx = 111195.0 * math.cos(math.radians(lat0))
xy = np.array([((r["lon"] - lon0) * kx, (r["lat"] - lat0) * 111195.0) for r in use])
tree = cKDTree(xy)
out = {}
for name, radius, same in (("same_dir_10m", 10.0, True), ("opposite_dir_10m", 10.0, False), ("same_dir_6m", 6.0, True)):
    best = {}
    for i, r in enumerate(use):
        for j in tree.query_ball_point(xy[i], radius):
            q = use[j]
            if q["session_id"] == r["session_id"]:
                continue
            dh = abs((r["heading"] - q["heading"] + 180) % 360 - 180)
            if (same and dh >= 30) or (not same and dh <= 150):
                continue
            d = float(np.hypot(*(xy[i] - xy[j])))
            key = (i, q["session_id"])
            if key not in best or d < best[key][0]:
                best[key] = (d, j)
    pairs = [(i, j) for (i, _), (d, j) in best.items()]
    blocks = {}
    for i, j in pairs:
        a, b = use[i], use[j]
        if a["session_id"] < b["session_id"]:
            blocks.setdefault((a["session_id"], b["session_id"], a["k"] // 20), []).append((a, b))
    res = {"n_pairs": len(pairs), "n_drive_pairs": len({(use[i]["session_id"], use[j]["session_id"]) for i, j in pairs}) // 2}
    for minm in (16, 8):
        A, B = [], []
        for lst in blocks.values():
            if len(lst) >= minm:
                A.append(V._agg([x[0] for x in lst])["rms_av_norm"]); B.append(V._agg([x[1] for x in lst])["rms_av_norm"])
        A, B = np.array(A), np.array(B)
        res[f"blocks_min{minm}"] = {"n": len(A), "spearman": float(stats.spearmanr(A, B).statistic) if len(A) > 3 else None,
                                    "median_abs_log_ratio": float(np.median(np.abs(np.log(A / B)))) if len(A) else None,
                                    "pearson_log": float(np.corrcoef(np.log(A), np.log(B))[0, 1]) if len(A) > 3 else None}
    seen = set(); sa = []; sb = []
    for i, j in pairs:
        k = tuple(sorted((use[i]["segment_id"], use[j]["segment_id"])))
        if k in seen: continue
        seen.add(k); sa.append(use[i]["rms_av_norm"]); sb.append(use[j]["rms_av_norm"])
    res["segment_spearman"] = float(stats.spearmanr(sa, sb).statistic) if len(sa) > 3 else None
    res["segment_n"] = len(sa)
    out[name] = res
# idle spectrum: which frequencies carry the idle noise
spec = []
for sid, summ, err in sess:
    if not summ or not summ.get("idle", {}).get("n_stops"):
        continue
    d = next((p for p in (Path(os.environ.get("VIDEO_INBOX", r"C:\GeoAI\inbox")) / sid,
                          HERE.parent.parent.parent.parent / "GeoAI_Video" / "field" / "sessions" / sid) if (p / "imu.csv").exists()), None)
    imu = P.read_imu_csv(d / "imu.csv")
    import scripts.iri_from_imu as R
    gn = R.read_gnss(d / "gnss_fixes.csv")
    sig = P.prepare_signals(imu)
    v = np.interp(sig.t, (gn["t_ns"] - sig.t0_ns) / 1e9, np.nan_to_num(gn["speed_mps"], nan=99))
    still = (v < 0.3) & ~sig.edge
    mov = (v > 3) & ~sig.edge
    # raw (unfiltered) vertical acc on the grid for the spectrum
    ta = (imu["acc"]["t_ns"] - sig.t0_ns) / 1e9
    gh = P._unit(P._interp_vec(ta, (imu["gravity"]["t_ns"] - sig.t0_ns) / 1e9, imu["gravity"]["xyz"]))
    x = np.interp(sig.t, ta, np.einsum("ij,ij->i", imu["acc"]["xyz"], gh))
    def bands(mask):
        idx = np.flatnonzero(mask)
        if len(idx) < 2048: return None
        seg = x[idx[:]] - x[idx].mean()
        f, Pxx = signal.welch(seg, fs=200, nperseg=1024)
        return {f"{lo}-{hi}": float(np.sqrt(np.trapezoid(Pxx[(f >= lo) & (f < hi)], f[(f >= lo) & (f < hi)])))
                for lo, hi in ((0.5, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 100))}
    spec.append({"session_id": sid, "idle_rms_av": summ["idle"]["rms_av"], "idle_bands": bands(still), "moving_bands": bands(mov)})
out["idle_spectra"] = spec
(HERE / "explore.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
