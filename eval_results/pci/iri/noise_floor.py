"""Noise-floor re-check, criteria in NOISE_FLOOR.md Part A (written before this was run). Writes noise_floor.json."""
import json, math, os, sys
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent)); sys.path.insert(0, str(HERE))
import numpy as np
import validate_iri as V
from app import pci_iri as P
from scripts import setup_local_db

dsn = setup_local_db.with_database(setup_local_db.database_url(), "geoai")
rows, sess = V.load(dsn, P.method_fp())
idle = {s[0]: s[1]["idle"]["rms_av"] for s in sess if s[1] and s[1].get("idle", {}).get("rms_av") is not None}
f_med = float(np.median(list(idle.values())))


def corrected(rows, floor_of):
    out = []
    for r in rows:
        r = dict(r)
        if V.usable(r):
            f = floor_of(r["session_id"])
            rc = math.sqrt(max(r["rms_av"] ** 2 - f * f, (0.1 * r["rms_av"]) ** 2))
            r["rms_av"] = rc
            r["rms_av_norm"] = rc / math.sqrt(r["speed_mps"])
            r["iri_gc_uncal"] = P.iri_gc_uncal(rc, r["speed_mps"])
        out.append(r)
    return out


def summary(rs):
    v3, _ = V.v3_v4(rs)
    v5 = V.v5(rs)
    by = {}
    for r in rs:
        by.setdefault(r["session_id"], []).append({**r, "flags": r["flags"] or []})
    bl = [b for v in by.values() for b in P.sections_along_session(v) if b["n_used"] >= 16 and b["iri_gc_uncal"]]
    x = np.array([b["iri_gc_uncal"] for b in bl])
    return {"pothole_auc": v5["primary_max_norm_k_to_k6"]["auc"], "pothole_auc_ci": v5["primary_max_norm_k_to_k6"]["auc_ci"],
            "R3_segment_spearman": v3["R3_segment_spearman"], "R3_n": v3["R3_n"], "n_blocks_repeat": v3["n_blocks"],
            "n_drive_pairs_same_dir": v3["n_drive_pairs_with_blocks"], "n_segment_pairs": v3["n_segment_pairs"],
            "blocks100_iri_gc_pct_5_50_95": [float(np.percentile(x, q)) for q in (5, 50, 95)], "n_blocks100": len(x)}


res = {"floor_median": f_med, "idle_by_drive": idle}
# (c) representativeness
rep = {}
for sid, f in idle.items():
    mv = [r["rms_av"] for r in rows if r["session_id"] == sid and V.usable(r)]
    rep[sid] = {"idle": f, "moving_p5": float(np.percentile(mv, 5)), "moving_median": float(np.median(mv)), "ok": f <= float(np.percentile(mv, 5))}
res["representativeness"] = rep
res["c_share_ok"] = sum(v["ok"] for v in rep.values()) / len(rep)
res["c_pass"] = res["c_share_ok"] >= 0.80
res["raw"] = summary(rows)
res["C1"] = summary(corrected(rows, lambda s: idle.get(s, f_med)))
res["C2"] = summary(corrected(rows, lambda s: f_med))
for k in ("C1", "C2"):
    d = res[k]
    d["a_pass"] = d["pothole_auc"] >= res["raw"]["pothole_auc"] - 0.02
    d["b_pass"] = d["R3_segment_spearman"] >= res["raw"]["R3_segment_spearman"] - 0.05
    d["adopt"] = bool(d["a_pass"] and d["b_pass"] and res["c_pass"])
(HERE / "noise_floor.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if k not in ("idle_by_drive",)}, indent=1))
