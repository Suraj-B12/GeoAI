"""AUROC + pass rate at 90 % recall of the recomputed gate variants (out/gate_variants.csv) vs the VLM."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

O = Path(r"C:/GeoAI/analysis/discriminator/out")
V = pd.read_csv(O / "gate_variants.csv")
J = pd.read_csv(O / "joined.csv")[["id", "k", "vlm_distress"]]
D = V.merge(J, on="id").sort_values("k").reset_index(drop=True)
y = D.vlm_distress.to_numpy(bool)
rng = np.random.default_rng(0)


def auroc(x, y):
    P, N = y.sum(), (~y).sum()
    r = pd.Series(x).rank().to_numpy()
    return float((r[y].sum() - P * (P + 1) / 2) / (P * N))


def par(x, y, t=0.9):
    xp = np.sort(x[y])[::-1]
    thr = xp[math.ceil(t * xp.size) - 1]
    return float((x >= thr).mean()), float(thr)


def boot(fn, x, y, B=2000, block=10):
    n = len(x); starts = np.arange(0, n, block); out = []
    for _ in range(B):
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in rng.choice(starts, len(starts))])
        if y[idx].all() or not y[idx].any():
            continue
        out.append(fn(x[idx], y[idx]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


res = {}
cols = [c for c in D.columns if c.startswith(("probe_", "crack_", "rec_"))]
for c in cols:
    x = D[c].to_numpy(float)
    res[c] = {"auroc": auroc(x, y), "auroc_ci95": boot(auroc, x, y), "pass_rate_at_recall90": par(x, y)[0],
              "pass_rate_ci95": boot(lambda a, b: par(a, b)[0], x, y), "threshold": par(x, y)[1]}
# paired difference road-only vs all-columns probe_max
for a, b in (("probe_max_x1.75", "probe_max_all"), ("probe_max_x2.5", "probe_max_all"), ("probe_frac_x1.75", "probe_frac_all")):
    xa, xb = D[a].to_numpy(float), D[b].to_numpy(float)
    res[f"diff_{a}_minus_{b}"] = {"delta_auroc": auroc(xa, y) - auroc(xb, y),
                                  "ci95": boot(lambda i, yy, xa=xa, xb=xb: 0, xa, y)}  # placeholder replaced below
    n = len(y); starts = np.arange(0, n, 10); ds = []
    for _ in range(2000):
        idx = np.concatenate([np.arange(s, min(s + 10, n)) for s in rng.choice(starts, len(starts))])
        if y[idx].all() or not y[idx].any():
            continue
        ds.append(auroc(xa[idx], y[idx]) - auroc(xb[idx], y[idx]))
    res[f"diff_{a}_minus_{b}"]["ci95"] = [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))]
res["_n"] = int(len(y)); res["_pos"] = int(y.sum())
(O / "gate_variants_auroc.json").write_text(json.dumps(res, indent=1))
for k, v in res.items():
    if isinstance(v, dict) and "auroc" in v:
        print(f"{k:28s} AUROC {v['auroc']:.3f} {np.round(v['auroc_ci95'],3)}  pass@90 {v['pass_rate_at_recall90']:.2f} {np.round(v['pass_rate_ci95'],2)}")
    elif isinstance(v, dict):
        print(k, v)
