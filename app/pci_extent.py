"""
IRC:82-2023 distress EXTENTS (% of road area) and POTHOLE NUMBER from per-type probe scores, with intervals.

Pure (no I/O except load_calibration). The calibration is fitted on Attain WS_V2.0 (vehicle-mounted phone frames,
Pascal-VOC boxes) by scripts/pci_attain_extent_calibration.py; rules in eval_results/pci/extent/PREREG.md, class
mapping in eval_results/pci/extent/MAPPING.md, results in eval_results/pci/extent/REPORT.md.

    feature(scores, keys)              s = max P(yes) over a parameter's probe keys
    frame_mean(spec, s, label)         E[area % | s (, label flag)] from the calibrated monotone curve
    estimate_extents(frames_scores, lengths, lane_width_m, calibration, labels, domain)
        -> {param: {"mean", "lo90", "hi90", ...}, "potholes_number": {...}}
    fit_calibration(rows, target_fn, ...)  used by the calibration script (dev rows only)

Model per parameter (PREREG.md): x = logit(s); quantile bins of x, bin means made non-decreasing (PAV),
piecewise-linear between bin centres, flat outside; or a constant; optionally one curve per Stage 2 label flag.
Interval: block-bootstrap curves x residual spread on the square-root scale, scaled to the section size with an
intra-block correlation model; for Bengaluru frames an extra declared viewpoint/domain factor.

Pothole number (IRC:82-2023 eq. 2): N = (pothole area % / 100) x imaged length x lane width / 0.1 m2.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Callable, Iterable, Mapping, Optional, Sequence

SCHEMA = 1
EDITION = "IRC:82-2023"
POTHOLE_UNIT_AREA_M2 = 0.1           # IRC:82-2023 eq. (2), §7.5.3.4
DEFAULT_LANE_WIDTH_M = 3.5
SEGMENT_M = 5.0
REF_N = 20                           # frames per Attain block = 5 m segments per 100 m section
LOW_COVERAGE = 0.50                  # same rule as app/pci_core.LOW_COVERAGE
EPS_P = 1e-4
N_Z = 41                             # normal quantiles used for the residual mixture
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "eval_results" / "pci" / "extent" / \
    "attain_extent_calibration.json"

# MAPPING.md §3
PARAM_KEYS: dict[str, tuple] = {
    "cracking": ("Alligator Cracking", "Longitudinal Cracking", "Transverse Cracking", "Block Cracking"),
    "ravelling": ("Ravelling",),
    "ravelling_incl_weathering": ("Ravelling", "Hungry Surface"),
    "patching": ("Patching",),
    "potholes": ("Potholes",),
}
PARAM_LABELS: dict[str, tuple] = {
    "cracking": ("Alligator Cracking", "Longitudinal Cracking", "Transverse Cracking", "Edge Cracking",
                 "Hairline Cracks"),
    "ravelling": ("Ravelling",),
    "ravelling_incl_weathering": ("Ravelling", "Hungry Surface"),
    "patching": (),
    "potholes": ("Potholes",),
}
PCI_PARAMS = ("cracking", "ravelling", "patching", "potholes")      # sensitivity targets are not PCI inputs
CANDIDATES = ("constant", "bins3", "bins5", "bins8", "bins3_label")
MIN_STRATUM = 30


# ------------------------------------------------------------------ small helpers
def _logit(s: float) -> float:
    s = min(max(float(s), EPS_P), 1.0 - EPS_P)
    return math.log(s / (1.0 - s))


def feature(scores: Mapping, keys: Sequence[str]) -> float:
    """Largest P(yes) over keys; missing keys count as 0. Raises if none of the keys is present."""
    vals = [scores.get(k) for k in keys]
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        raise KeyError(f"none of {list(keys)} in the probe scores")
    return max(vals)


def label_flag(labels: Optional[Iterable[str]], param: str) -> Optional[int]:
    if labels is None:
        return None
    return 1 if set(labels) & set(PARAM_LABELS.get(param, ())) else 0


def _pav(y: list, w: list) -> list:
    """Weighted pool-adjacent-violators, non-decreasing."""
    blocks = [[yi, wi, 1] for yi, wi in zip(y, w)]
    out = []
    for b in blocks:
        out.append(b)
        while len(out) > 1 and out[-2][0] > out[-1][0] + 1e-15:
            y2, w2, n2 = out.pop()
            y1, w1, n1 = out.pop()
            ww = w1 + w2
            out.append([(y1 * w1 + y2 * w2) / ww if ww > 0 else (y1 + y2) / 2, ww, n1 + n2])
    res = []
    for yv, _, n in out:
        res += [yv] * n
    return res


def _interp(x: float, xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) == 1:
        return float(ys[0])
    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    t = (x - xs[lo]) / (xs[hi] - xs[lo]) if xs[hi] > xs[lo] else 0.0
    return float(ys[lo] + t * (ys[hi] - ys[lo]))


def _quantile(v: Sequence[float], q: float) -> float:
    s = sorted(v)
    if not s:
        return float("nan")
    pos = q * (len(s) - 1)
    i = int(math.floor(pos))
    j = min(i + 1, len(s) - 1)
    return s[i] + (pos - i) * (s[j] - s[i])


# ------------------------------------------------------------------ curves
def _edges(x: Sequence[float], k: int) -> list:
    """Interior quantile edges of x for k bins (duplicates removed)."""
    e = sorted({_quantile(x, i / k) for i in range(1, k)})
    return e


def _bin_index(x: float, edges: Sequence[float]) -> int:
    i = 0
    while i < len(edges) and x > edges[i]:
        i += 1
    return i


def _fit_bins(x: Sequence[float], y: Sequence[float], edges: Sequence[float]) -> Optional[dict]:
    """Bin means (PAV) and bin centres for fixed edges; None if a bin is empty."""
    nb = len(edges) + 1
    bx: list = [[] for _ in range(nb)]
    by: list = [[] for _ in range(nb)]
    for xi, yi in zip(x, y):
        b = _bin_index(xi, edges)
        bx[b].append(xi)
        by[b].append(yi)
    keep = [i for i in range(nb) if by[i]]
    if not keep:
        return None
    means = [sum(by[i]) / len(by[i]) for i in keep]
    ws = [len(by[i]) for i in keep]
    centres = [_quantile(bx[i], 0.5) for i in keep]
    return {"x": centres, "y": _pav(means, ws), "n": ws}


def _boot_y(x, y, edges, centres, n_bins_full) -> list:
    """Bin means for fixed edges evaluated at the full-fit centres; empty bins borrow the neighbour (then PAV)."""
    nb = len(edges) + 1
    sums, cnt = [0.0] * nb, [0] * nb
    for xi, yi in zip(x, y):
        b = _bin_index(xi, edges)
        sums[b] += yi
        cnt[b] += 1
    m = [sums[i] / cnt[i] if cnt[i] else None for i in range(nb)]
    # fill empties from nearest filled bin
    for i in range(nb):
        if m[i] is None:
            for d in range(1, nb):
                for j in (i - d, i + d):
                    if 0 <= j < nb and m[j] is not None and cnt[j]:
                        m[i] = sums[j] / cnt[j]
                        break
                if m[i] is not None:
                    break
    if any(v is None for v in m):
        m = [0.0] * nb
    return _pav(m, [max(c, 1) for c in cnt])


def fit_model(name: str, x: Sequence[float], y: Sequence[float], flags: Optional[Sequence[int]] = None) -> Optional[dict]:
    """One candidate model on frames (x = logit feature, y = area %). None if not fittable."""
    if not x:
        return None
    if name == "constant":
        return {"kind": "constant", "y": [sum(y) / len(y)], "x": [0.0], "edges": []}
    if name.startswith("bins") and not name.endswith("_label"):
        k = int(name[4:])
        edges = _edges(x, k)
        f = _fit_bins(x, y, edges)
        if f is None or len(f["x"]) != len(edges) + 1:
            return None
        return {"kind": "bins", "k": k, "edges": edges, "x": f["x"], "y": f["y"], "n": f["n"]}
    if name == "bins3_label":
        if flags is None or any(fl is None for fl in flags):
            return None
        strata = {}
        for s in (0, 1):
            xs = [xi for xi, fl in zip(x, flags) if fl == s]
            ys = [yi for yi, fl in zip(y, flags) if fl == s]
            if len(xs) < MIN_STRATUM:
                return None
            m = fit_model("bins3", xs, ys)
            if m is None:
                return None
            strata[str(s)] = m
        return {"kind": "label", "strata": strata}
    raise ValueError(name)


def model_predict(m: Mapping, x: float, flag: Optional[int] = None, ys: Optional[Sequence[float]] = None) -> float:
    if m["kind"] == "label":
        if flag is None:
            raise ValueError("this calibration needs the Stage 2 labels of every frame")
        sub = m["strata"][str(int(flag))]
        return model_predict(sub, x, ys=None if ys is None else ys[str(int(flag))])
    yy = m["y"] if ys is None else ys
    if m["kind"] == "constant":
        return float(yy[0])
    return max(0.0, _interp(x, m["x"], yy))


def frame_mean(spec: Mapping, s: float, label: Optional[int] = None) -> float:
    """Expected area % of one frame with feature s (P(yes), not logit) under the point calibration."""
    return model_predict(spec["model"], _logit(s), label)


# ------------------------------------------------------------------ fitting (calibration script)
def _blocks(rows, block_key):
    b: dict = {}
    for i, r in enumerate(rows):
        b.setdefault(r[block_key], []).append(i)
    return b


def _cv_oof(names, x, y, flags, rows, block_key, folds_k, seed):
    import random
    blocks = sorted(_blocks(rows, block_key))
    rng = random.Random(seed)
    sh = list(blocks)
    rng.shuffle(sh)
    fold = {b: i % folds_k for i, b in enumerate(sh)}
    oof = {n: [None] * len(rows) for n in names}
    for k in range(folds_k):
        tr = [i for i, r in enumerate(rows) if fold[r[block_key]] != k]
        ho = [i for i, r in enumerate(rows) if fold[r[block_key]] == k]
        if not ho:
            continue
        for n in names:
            m = fit_model(n, [x[i] for i in tr], [y[i] for i in tr], None if flags is None else [flags[i] for i in tr])
            if m is None:
                oof[n] = None
                continue
            if oof[n] is None:
                continue
            for i in ho:
                oof[n][i] = model_predict(m, x[i], None if flags is None else flags[i])
    return oof


def _block_mae(pred, y, rows, block_key):
    errs = []
    for b, idx in _blocks(rows, block_key).items():
        errs.append(abs(sum(pred[i] for i in idx) / len(idx) - sum(y[i] for i in idx) / len(idx)))
    return sum(errs) / len(errs)


def fit_param(rows, param: str, target_fn: Callable, block_key="block", n_boot=400, seed=0, select_folds=5,
              select_seed=0) -> tuple[dict, dict]:
    keys = PARAM_KEYS[param]
    x = [_logit(feature(r["scores"], keys)) for r in rows]
    y = [float(target_fn(r, param)) for r in rows]
    flags = [label_flag(r.get("labels"), param) for r in rows] if PARAM_LABELS.get(param) else None
    names = [n for n in CANDIDATES if not (n.endswith("_label") and flags is None)]
    oof = _cv_oof(names, x, y, flags, rows, block_key, select_folds, select_seed)
    cv = {n: (_block_mae(oof[n], y, rows, block_key) if oof[n] is not None and all(v is not None for v in oof[n])
              else None) for n in names}
    valid = [n for n in names if cv[n] is not None]
    best = min(valid, key=lambda n: (round(cv[n], 12), names.index(n)))
    model = fit_model(best, x, y, flags)

    # residual model from the chosen candidate's out-of-fold predictions
    pr = oof[best]
    bl = _blocks(rows, block_key)
    u, bvar_terms = [], []
    res = [y[i] - pr[i] for i in range(len(rows))]
    for b, idx in bl.items():
        if len(idx) < 10:
            continue
        T = sum(y[i] for i in idx) / len(idx)
        Yh = sum(pr[i] for i in idx) / len(idx)
        u.append(math.sqrt(max(T, 0.0)) - math.sqrt(max(Yh, 0.0)))
        bvar_terms.append((sum(res[i] for i in idx) / len(idx), len(idx)))
    tau20 = math.sqrt(sum(v * v for v in u) / len(u)) if u else 0.0
    mres = sum(res) / len(res)
    sigma2 = sum((r - mres) ** 2 for r in res) / max(1, len(res) - 1)
    if sigma2 > 0 and bvar_terms:
        vb = sum((m - mres) ** 2 for m, _ in bvar_terms) / max(1, len(bvar_terms) - 1)
        nbar = sum(n for _, n in bvar_terms) / len(bvar_terms)
        rho = (nbar * vb / sigma2 - 1.0) / (nbar - 1.0) if nbar > 1 else 1.0
    else:
        rho = 1.0
    rho = min(1.0, max(0.0, rho))

    # bootstrap of DEV blocks: refit with the full-fit bin edges fixed
    import random
    rng = random.Random(seed)
    bkeys = sorted(bl)
    boot = []
    for _ in range(n_boot):
        pick = [bkeys[rng.randrange(len(bkeys))] for _ in bkeys]
        idx = [i for b in pick for i in bl[b]]
        bx, by = [x[i] for i in idx], [y[i] for i in idx]
        if model["kind"] == "constant":
            boot.append([sum(by) / len(by)])
        elif model["kind"] == "bins":
            boot.append(_boot_y(bx, by, model["edges"], model["x"], len(model["x"])))
        else:
            d = {}
            for s, sub in model["strata"].items():
                sx = [x[i] for i in idx if flags[i] == int(s)]
                sy = [y[i] for i in idx if flags[i] == int(s)]
                d[s] = _boot_y(sx, sy, sub["edges"], sub["x"], len(sub["x"])) if sx else list(sub["y"])
            boot.append(d)
    spec = {"keys": list(keys), "uses_labels": model["kind"] == "label",
            "label_set": list(PARAM_LABELS.get(param, ())), "model": model, "boot_y": boot,
            "residual": {"scale": "sqrt", "tau20": tau20, "rho": rho, "sigma2_frame": sigma2, "ref_n": REF_N,
                         "n_blocks": len(u)},
            "support": [min(feature(r["scores"], keys) for r in rows), max(feature(r["scores"], keys) for r in rows)],
            "dev_mean_pct": sum(y) / len(y), "pci_input": param in PCI_PARAMS}
    diag = {"cv_block_mae": cv, "chosen": best, "n_frames": len(rows), "n_blocks": len(bl), "tau20": tau20,
            "rho": rho, "sigma2_frame": sigma2, "oof_block_sqrt_residuals": u}
    return spec, diag


def fit_calibration(rows, target_fn: Callable, params: Sequence[str] = tuple(PARAM_KEYS), block_key="block",
                    n_boot=400, seed=0, select_folds=5, select_seed=0) -> tuple[dict, dict]:
    cal = {"schema": SCHEMA, "edition": EDITION, "kind": "attain_extent_calibration", "unit": "percent of road area",
           "params": {}, "domain_factor": {"default_90": 1.5, "shifted_90": 2.0, "ks_d_threshold": 0.20,
                                           "note": "declared assumption for frames outside Attain (PREREG.md)"},
           "transfer": {}}
    diag = {"selection": {}}
    for p in params:
        spec, d = fit_param(rows, p, target_fn, block_key, n_boot, seed, select_folds, select_seed)
        cal["params"][p] = spec
        diag["selection"][p] = {k: v for k, v in d.items() if k != "oof_block_sqrt_residuals"}
        diag[p] = d
    return cal, diag


# ------------------------------------------------------------------ calibration file
def validate_calibration(cal: Mapping) -> dict:
    if not isinstance(cal, Mapping) or cal.get("schema") != SCHEMA or cal.get("edition") != EDITION:
        raise ValueError("extent calibration: schema 1 / edition IRC:82-2023 required")
    params = cal.get("params")
    if not isinstance(params, Mapping) or not params:
        raise ValueError("extent calibration: params missing")
    for p, s in params.items():
        if p not in PARAM_KEYS:
            raise ValueError(f"extent calibration: unknown parameter {p!r}")
        m = s.get("model") or {}
        if m.get("kind") not in ("constant", "bins", "label"):
            raise ValueError(f"extent calibration: {p}: bad model kind")
        subs = list(m["strata"].values()) if m["kind"] == "label" else [m]
        for sub in subs:
            xs, ys = sub.get("x") or [], sub.get("y") or []
            if len(xs) != len(ys) or not ys or any(b < a for a, b in zip(xs, xs[1:])) \
                    or any(b < a - 1e-9 for a, b in zip(ys, ys[1:])) or any(v < 0 for v in ys):
                raise ValueError(f"extent calibration: {p}: curve must be non-decreasing with increasing knots")
        r = s.get("residual") or {}
        if not (r.get("tau20", -1) >= 0 and 0 <= r.get("rho", -1) <= 1):
            raise ValueError(f"extent calibration: {p}: residual model missing")
        if not set(s.get("keys") or []) <= set(PARAM_KEYS[p]) or not s.get("keys"):
            raise ValueError(f"extent calibration: {p}: keys must be among {PARAM_KEYS[p]}")
        if not isinstance(s.get("boot_y"), list) or not s["boot_y"]:
            raise ValueError(f"extent calibration: {p}: bootstrap curves missing")
    return dict(cal)


def load_calibration(path=None) -> dict:
    p = Path(path) if path else DEFAULT_PATH
    return validate_calibration(json.loads(p.read_text(encoding="utf-8")))


# ------------------------------------------------------------------ estimation
def _g(n: float, rho: float) -> float:
    """Variance factor of a mean over n correlated frames relative to REF_N frames (PREREG.md)."""
    n = max(1.0, float(n))
    ref = (1.0 + (REF_N - 1) * rho) / REF_N
    if n <= REF_N:
        return ((1.0 + (n - 1.0) * rho) / n) / ref
    return REF_N / n


_ZQ = [NormalDist().inv_cdf((i + 0.5) / N_Z) for i in range(N_Z)]


def _interval(curves_mean: Sequence[float], tau: float, factor90: float) -> tuple[float, float]:
    """5th/95th percentile of max(0, sqrt(Y_b) + tau z)^2 x domain factor, over bootstrap means x normal grid."""
    sd_log = math.log(factor90) / 1.6448536269514722 if factor90 > 1 else 0.0
    zq = _ZQ if sd_log > 0 else [0.0]
    vals = []
    for yb in curves_mean:
        r = math.sqrt(max(yb, 0.0))
        for z in _ZQ:
            t = max(0.0, r + tau * z) ** 2
            if sd_log > 0:
                for z2 in zq[::4]:
                    vals.append(t * math.exp(sd_log * z2))
            else:
                vals.append(t)
    vals.sort()
    return _quantile(vals, 0.05), _quantile(vals, 0.95)


def estimate_extents(frames_scores: Sequence[Optional[Mapping]], lengths: Sequence[float],
                     lane_width_m: float = DEFAULT_LANE_WIDTH_M, calibration: Optional[Mapping] = None,
                     labels: Optional[Sequence[Optional[Iterable[str]]]] = None, params: Optional[Sequence[str]] = None,
                     domain: str = "attain", total_length_m: Optional[float] = None) -> dict:
    """Section extents (% of road area) with 90 % intervals, and the IRC:82-2023 pothole number.

    frames_scores: one probe-score dict (key -> P(yes)) per frame / 5 m segment, None for an unscored one.
    lengths:       the length (m) each frame stands for (5 m segments; any positive weights for Attain frames).
    labels:        Stage 2 free-form list per frame (needed only by a calibration that uses label flags).
    domain:        "attain" (calibration domain) or "bengaluru" (adds the declared viewpoint/domain factor).
    total_length_m: section length for the pothole number (default: sum of lengths of scored frames).

    Returns {param: {mean, lo90, hi90, n_frames, scored_m, coverage, model, domain_factor90, status}, and
    "potholes_number": {mean, lo90, hi90, area_m2_mean, road_area_m2, lane_width_m}}. A parameter whose scored
    length is < 50 % of the given length returns mean/lo90/hi90 = None and status "insufficient coverage"."""
    cal = validate_calibration(calibration) if calibration is not None else load_calibration()
    if len(frames_scores) != len(lengths):
        raise ValueError("frames_scores and lengths must have the same length")
    if labels is not None and len(labels) != len(frames_scores):
        raise ValueError("labels must have one entry per frame")
    if domain not in ("attain", "bengaluru"):
        raise ValueError("domain must be 'attain' or 'bengaluru'")
    lw = float(lane_width_m)
    if not (lw > 0 and math.isfinite(lw)):
        raise ValueError("lane_width_m must be > 0")
    Ls = [float(L) for L in lengths]
    if any(not (L >= 0 and math.isfinite(L)) for L in Ls):
        raise ValueError("lengths must be finite and >= 0")
    total = sum(Ls)
    out: dict = {}
    for p in (params or list(cal["params"])):
        spec = cal["params"][p]
        m = spec["model"]
        xs, ws, fl = [], [], []
        for i, (sc, L) in enumerate(zip(frames_scores, Ls)):
            if not isinstance(sc, Mapping) or L <= 0:
                continue
            try:
                s = feature(sc, spec["keys"])
            except KeyError:
                continue
            xs.append(_logit(s))
            ws.append(L)
            fl.append(label_flag(labels[i], p) if labels is not None else None)
        scored = sum(ws)
        row = {"n_frames": len(xs), "scored_m": scored, "coverage": (scored / total) if total > 0 else 0.0,
               "model": m["kind"] if m["kind"] != "bins" else f"bins{m['k']}", "unit": "%"}
        if not xs or total <= 0 or scored / total < LOW_COVERAGE - 1e-9:
            row.update(mean=None, lo90=None, hi90=None, status="insufficient coverage")
            out[p] = row
            continue
        if m["kind"] == "label" and any(f is None for f in fl):
            raise ValueError(f"{p}: this calibration uses Stage 2 labels; pass labels for every frame")

        def sec_mean(ys=None):
            return sum(w * model_predict(m, x, f, ys) for x, w, f in zip(xs, ws, fl)) / scored
        point = sec_mean()
        boots = [sec_mean(b) for b in spec["boot_y"]]
        # effective number of frames: scored length / 5 m (a 5 m segment = one frame)
        n_eff = scored / SEGMENT_M if domain == "bengaluru" else float(len(xs))
        r = spec["residual"]
        tau = r["tau20"] * math.sqrt(_g(n_eff, r["rho"]))
        f90 = 1.0
        if domain == "bengaluru":
            df = cal.get("domain_factor") or {}
            tr = (cal.get("transfer") or {}).get(p) or {}
            f90 = float(df.get("shifted_90", 2.0)) if tr.get("ks_D", 0.0) >= float(df.get("ks_d_threshold", 0.2)) \
                else float(df.get("default_90", 1.5))
        lo, hi = _interval(boots, tau, f90)
        lo, hi = min(lo, point), max(hi, point)
        row.update(mean=point, lo90=lo, hi90=hi, n_eff=n_eff, tau=tau, domain_factor90=f90,
                   status="calibrated on Attain" + (" (applied to Bengaluru frames)" if domain == "bengaluru" else ""),
                   pci_input=bool(spec.get("pci_input", p in PCI_PARAMS)))
        out[p] = row
    if "potholes" in out:
        e = out["potholes"]
        L = float(total_length_m) if total_length_m is not None else e["scored_m"]
        area = L * lw
        if e["mean"] is None:
            out["potholes_number"] = {"mean": None, "lo90": None, "hi90": None, "road_area_m2": area,
                                      "lane_width_m": lw, "status": e["status"]}
        else:
            k = area / 100.0 / POTHOLE_UNIT_AREA_M2
            out["potholes_number"] = {"mean": e["mean"] * k, "lo90": e["lo90"] * k, "hi90": e["hi90"] * k,
                                      "area_m2_mean": e["mean"] / 100.0 * area, "road_area_m2": area,
                                      "length_m": L, "lane_width_m": lw, "unit": "number (eq. 2, 0.1 m2 units)",
                                      "status": e["status"]}
    return out
