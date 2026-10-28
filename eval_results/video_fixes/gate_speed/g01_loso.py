"""Gate validation (CPU only): leave-one-session-out + 20-segment block bootstrap, per PREREGISTRATION.md.

Reads batch_001 (read-only) through geoai's own loaders; calibrates with the shipping exact solver
(geoai.gate._solve_any_of, used by geoai.gate.calibrate); writes out/loso.json.
Usage (CPU): set CUDA_VISIBLE_DEVICES=-1; python g01_loso.py
"""
import os
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, r"C:/Users/PRO-LAB-3/Documents/GeoAI_Video/pipeline")
import torch  # noqa: E402
assert not torch.cuda.is_available()
from geoai.analyze import load_batch  # noqa: E402
from geoai.config import Config  # noqa: E402
from geoai.gate import _need, _solve_any_of, calibrate  # noqa: E402
from geoai.vlm import answer_class  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
BATCH = Path(r"C:/Users/PRO-LAB-3/Documents/GeoAI_Video/field/batches/batch_001")
TARGET, B, BLOCK, AUDIT = 0.9, 2000, 20, 0.10
VARIANTS = {"V1_probe_frac": ["probe_frac"], "V2_probe_max_logit": ["probe_max_logit"],
            "V3_probe_frac_or_max_logit": ["probe_frac", "probe_max_logit"],
            "V4_anyof_shipping": ["probe_frac", "evidence", "crack_mm"]}
DEFAULT = {"probe_frac": 0.05, "evidence": 30.0, "crack_mm": 300.0}
SIGNALS = ["probe_frac", "probe_max_logit", "evidence", "crack_mm"]

cfg = Config()
assert (cfg.gate.probe_frac, cfg.gate.evidence, cfg.gate.crack_mm) == tuple(DEFAULT.values())
data = load_batch(BATCH, cfg)
allowed = set(cfg.vlm.labels)
rows = []
for r in data.rows:
    if not r["answered"]:
        continue
    rec = data.records[r["id"]]
    cls, _ = answer_class(data.answers.results[r["id"]], allowed)
    assert cls in ("distress", "normal"), (r["id"], cls)
    sc = rec["gate"]["scores"]
    rows.append((rec["session_id"], rec["segment"]["k"],
                 [(-np.inf if sc[s] is None else float(sc[s])) for s in SIGNALS], cls == "distress"))
rows.sort(key=lambda x: (x[0], x[1]))
sess_all = np.array([x[0] for x in rows])
X_all = np.array([x[2] for x in rows], dtype=np.float64)
y_all = np.array([x[3] for x in rows], bool)
SESS = sorted(set(sess_all))
SHORT = {s: s[-4:] for s in SESS}
print({SHORT[s]: (int((sess_all == s).sum()), int(y_all[sess_all == s].sum())) for s in SESS}, flush=True)

# Cross-check: the shipping calibrate() on all rows gives the same any-of thresholds as batch_001's file.
cal_rows = [{"scores": {s: (None if not math.isfinite(v) else v) for s, v in zip(SIGNALS, x[2])}
             | {"z_max": None, "anomalous_patch_frac": None}, "vlm_distress": x[3]} for x in rows]
full = calibrate(cal_rows, TARGET)
shipped = json.loads((BATCH / "gate_calibration.json").read_text(encoding="utf-8"))
assert full["any_of"]["thresholds"] == shipped["any_of"]["thresholds"], (full["any_of"]["thresholds"], shipped["any_of"]["thresholds"])


def fit(X, y, cols):
    """Thresholds (+inf = off) of the any-of rule over `cols` at TARGET recall, by the shipping solver."""
    idx = [SIGNALS.index(c) for c in cols]
    Xs = X[:, idx]
    P = int(y.sum())
    reach = int((y & np.isfinite(Xs).any(1)).sum())
    sol = _solve_any_of(Xs, y, [min(_need(TARGET, P), reach)])[0]
    t = np.full(len(SIGNALS), np.inf)
    t[idx] = sol[0]
    return t


def passes(X, t):
    return (X >= t[None, :]).any(1)


def metrics(p, y):
    P = int(y.sum())
    return {"recall": float((p & y).sum() / P) if P else float("nan"), "pass_rate": float(p.mean()),
            "n": int(len(y)), "n_pos": P, "n_pass": int(p.sum()), "n_pos_pass": int((p & y).sum())}


T_DEFAULT = np.array([DEFAULT.get(s, np.inf) for s in SIGNALS])


def blocks_of(n):
    return [np.arange(a, min(a + BLOCK, n)) for a in range(0, n, BLOCK)]


def resample(idx, rng):
    bl = blocks_of(len(idx))
    pick = rng.integers(0, len(bl), len(bl))
    return idx[np.concatenate([bl[i] for i in pick])]


def auroc(x, y):
    from scipy.stats import rankdata
    P, N = int(y.sum()), int((~y).sum())
    r = rankdata(x)
    return float((r[y].sum() - P * (P + 1) / 2) / (P * N))


def ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


t0 = time.time()
res = {"protocol": "PREREGISTRATION.md", "prereg_sha256": hashlib.sha256((HERE / "PREREGISTRATION.md").read_bytes()).hexdigest(),
       "target_recall": TARGET, "B": B, "block": BLOCK, "seed": 0, "sessions": {}, "pooled": {}, "full_batch_calibration": {}}
for name, cols in VARIANTS.items():
    t = fit(X_all, y_all, cols)
    res["full_batch_calibration"][name] = {"thresholds": {s: (None if not math.isfinite(v) else float(v)) for s, v in zip(SIGNALS, t)},
                                           **metrics(passes(X_all, t), y_all)}

idx_of = {s: np.flatnonzero(sess_all == s) for s in SESS}
names = ["V0_default"] + list(VARIANTS)
point = {n: {} for n in names}
boot = {n: {s: {"recall": [], "pass_rate": [], "n_pos_pass": [], "n_pass": [], "n_pos": [], "n": []} for s in SESS} for n in names}
thr_point = {}
for test in SESS:
    train = [s for s in SESS if s != test]
    tr = np.concatenate([idx_of[s] for s in train])
    te = idx_of[test]
    point["V0_default"][test] = metrics(passes(X_all[te], T_DEFAULT), y_all[te])
    thr_point[test] = {}
    for name, cols in VARIANTS.items():
        t = fit(X_all[tr], y_all[tr], cols)
        thr_point[test][name] = {s: (None if not math.isfinite(v) else float(v)) for s, v in zip(SIGNALS, t)}
        point[name][test] = metrics(passes(X_all[te], t), y_all[te])
    rng = np.random.default_rng([0, SESS.index(test)])
    for b in range(B):
        trb = np.concatenate([resample(idx_of[s], rng) for s in train])
        teb = resample(te, rng)
        Xte, yte = X_all[teb], y_all[teb]
        for name in names:
            t = T_DEFAULT if name == "V0_default" else fit(X_all[trb], y_all[trb], VARIANTS[name])
            m = metrics(passes(Xte, t), yte)
            for k in boot[name][test]:
                boot[name][test][k].append(m[k])
    print(f"held-out {SHORT[test]} done, {time.time() - t0:.0f} s", flush=True)

for name in names:
    res["sessions"][name] = {}
    for s in SESS:
        bt = boot[name][s]
        res["sessions"][name][SHORT[s]] = {**point[name][s], "recall_ci95": ci(bt["recall"]), "pass_rate_ci95": ci(bt["pass_rate"]),
                                           "thresholds": thr_point.get(s, {}).get(name, DEFAULT if name == "V0_default" else None)}
    # Pooled: sum counts over the three held-out sessions, per replicate (replicate b of each fold).
    P = sum(point[name][s]["n_pos"] for s in SESS)
    N = sum(point[name][s]["n"] for s in SESS)
    pp = sum(point[name][s]["n_pos_pass"] for s in SESS)
    pa = sum(point[name][s]["n_pass"] for s in SESS)
    bP = np.sum([boot[name][s]["n_pos"] for s in SESS], 0)
    bN = np.sum([boot[name][s]["n"] for s in SESS], 0)
    bpp = np.sum([boot[name][s]["n_pos_pass"] for s in SESS], 0)
    bpa = np.sum([boot[name][s]["n_pass"] for s in SESS], 0)
    res["pooled"][name] = {"recall": pp / P, "pass_rate": pa / N, "sent_rate_audit_0.10": pa / N + AUDIT * (1 - pa / N),
                           "n": N, "n_pos": P, "n_pass": pa, "n_pos_pass": pp,
                           "recall_ci95": ci(bpp / bP), "pass_rate_ci95": ci(bpa / bN), "_bpass": (bpa / bN), "_brec": (bpp / bP)}

# Paired differences vs V1 (pooled) and the pre-registered rule.
ref = res["pooled"]["V1_probe_frac"]
decision = {"default": "V1_probe_frac", "candidates": {}}
for name in ["V2_probe_max_logit", "V3_probe_frac_or_max_logit", "V4_anyof_shipping"]:
    v = res["pooled"][name]
    dp = v["_bpass"] - ref["_bpass"]
    dr = v["_brec"] - ref["_brec"]
    c1 = (v["pass_rate"] <= ref["pass_rate"] - 0.03) and ci(dp)[1] < 0
    c2 = v["recall"] >= ref["recall"] - 0.02
    c3 = all(res["sessions"][name][SHORT[s]]["recall"] >= 0.80 for s in SESS)
    decision["candidates"][name] = {"delta_pass": v["pass_rate"] - ref["pass_rate"], "delta_pass_ci95": ci(dp),
                                    "delta_recall": v["recall"] - ref["recall"], "delta_recall_ci95": ci(dr),
                                    "rule1_pass_lower": bool(c1), "rule2_recall_kept": bool(c2), "rule3_min_session_recall": bool(c3),
                                    "wins": bool(c1 and c2 and c3)}
winners = [n for n, d in decision["candidates"].items() if d["wins"]]
decision["winners"] = winners
decision["ships"] = "V1_probe_frac" if not winners or all(w in ("V2_probe_max_logit", "V3_probe_frac_or_max_logit") for w in winners) else winners[0]
decision["evidence_crack"] = "on" if "V4_anyof_shipping" in winners else "off"
res["decision"] = decision
for name in names:
    res["pooled"][name].pop("_bpass")
    res["pooled"][name].pop("_brec")

# Held-out AUROC per session (threshold-free, information only), block-bootstrap CI.
res["auroc"] = {}
for s in SESS:
    te = idx_of[s]
    rng = np.random.default_rng([1, SESS.index(s)])
    out = {}
    for sig in ["probe_frac", "probe_max_logit", "evidence", "crack_mm"]:
        x = X_all[te, SIGNALS.index(sig)]
        vals = []
        for _ in range(B):
            ib = resample(np.arange(len(te)), rng)
            if y_all[te][ib].all() or not y_all[te][ib].any():
                continue
            vals.append(auroc(x[ib], y_all[te][ib]))
        out[sig] = {"auroc": auroc(x, y_all[te]), "ci95": ci(vals)}
    res["auroc"][SHORT[s]] = out

# Held-out curves of probe_frac per session (for the figure): recall and pass at every threshold.
res["curves_probe_frac"] = {}
for s in SESS:
    te = idx_of[s]
    x, y = X_all[te, 0], y_all[te]
    ts = np.unique(x)[::-1]
    res["curves_probe_frac"][SHORT[s]] = [[float((x >= t).mean()), float(((x >= t) & y).sum() / y.sum())] for t in ts]
res["elapsed_s"] = time.time() - t0
(OUT / "loso.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps({"pooled": res["pooled"], "decision": res["decision"]}, indent=1))
for name in names:
    print(name, {k: (round(v["recall"], 3), v["recall_ci95"], round(v["pass_rate"], 3), v["pass_rate_ci95"])
                 for k, v in res["sessions"][name].items()})
