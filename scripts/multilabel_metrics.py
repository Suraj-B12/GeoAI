"""
Multi-label evaluation metrics that stay honest under skewed class prevalence.

Why not just F1
---------------
On Attain WS_V2.0, 81% of images carry "Linear crack" and 66% carry
"Alligator crack". A classifier that answers "yes" to both on every image
scores F1 = 0.90 and 0.80 on them without looking at a single pixel. The
production pipeline's reported Linear-crack F1 of 0.95 is therefore not
evidence of skill until it is compared with that constant predictor.

So every per-class result here carries:
  * MCC (Matthews correlation) - 0 for ANY constant predictor, whatever the
    prevalence; 1 only for a perfect one. The headline metric.
  * balanced accuracy - mean of sensitivity and specificity; 0.5 for constant
    predictors.
  * AUROC / average precision when a score is available - threshold-free.
  * F1 - reported for continuity with earlier results, never alone.

Uncertainty
-----------
Attain frames are consecutive shots from a moving vehicle, so neighbouring
images are near-duplicates and NOT independent. Confidence intervals come
from a cluster (block) bootstrap: whole blocks of consecutive images are
resampled, which keeps correlated frames together. Treating 550 correlated
frames as 550 independent samples would make every interval too narrow.

Pure Python + math; no numpy dependency so it can be imported anywhere.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Optional, Sequence


# ============================================================
# Per-class binary metrics
# ============================================================

def counts(y_true: Sequence[bool], y_pred: Sequence[bool]) -> dict:
    tp = fp = fn = tn = 0
    for t, p in zip(y_true, y_pred):
        if t and p:
            tp += 1
        elif p:
            fp += 1
        elif t:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _div(a: float, b: float) -> float:
    return a / b if b else 0.0


def binary_metrics(y_true: Sequence[bool], y_pred: Sequence[bool]) -> dict:
    c = counts(y_true, y_pred)
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    prec = _div(tp, tp + fp)
    rec = _div(tp, tp + fn)
    spec = _div(tn, tn + fp)
    f1 = _div(2 * prec * rec, prec + rec)
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else 0.0
    n = tp + fp + fn + tn
    return {
        **c,
        "n": n,
        "prevalence": _div(tp + fn, n),
        "predicted_rate": _div(tp + fp, n),
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "f1": f1,
        "mcc": mcc,
        "balanced_accuracy": (rec + spec) / 2 if (tp + fn) and (tn + fp) else float("nan"),
    }


def roc_auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Mann-Whitney AUROC; ties count one half. NaN if one class is absent."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    # rank-based, O(n log n)
    allv = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    rank_sum = 0.0
    i = 0
    n = len(allv)
    while i < n:
        j = i
        while j + 1 < n and allv[j + 1][0] == allv[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        rank_sum += avg_rank * sum(1 for k in range(i, j + 1) if allv[k][1] == 1)
        i = j + 1
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def average_precision(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Area under the precision-recall curve (step interpolation)."""
    n_pos = sum(1 for y in labels if y)
    if n_pos == 0:
        return float("nan")
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    tp = fp = 0
    ap = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        new_tp = sum(1 for k in range(i, j + 1) if pairs[k][1])
        tp += new_tp
        fp += (j - i + 1) - new_tp
        ap += (new_tp / n_pos) * (tp / (tp + fp))
        i = j + 1
    return ap


def best_threshold(scores: Sequence[float], labels: Sequence[bool],
                   objective: str = "mcc") -> tuple[float, float]:
    """Threshold t (predict score >= t) maximising the objective.

    Candidate thresholds are the observed scores, so the result reproduces a
    real decision boundary. Ties in the objective go to the HIGHER threshold
    (fewer positives), the conservative choice for a system that dispatches
    repair crews.
    """
    cands = sorted(set(scores))
    best_t, best_v = 0.5, -float("inf")
    for t in cands:
        pred = [s >= t for s in scores]
        v = binary_metrics(labels, pred)[objective]
        if v > best_v or (v == best_v and t > best_t):
            best_t, best_v = t, v
    return best_t, best_v


# ============================================================
# Sample-level (per image) metrics over a label set
# ============================================================

def sample_metrics(true_sets: Sequence[set], pred_sets: Sequence[set],
                   classes: Sequence[str]) -> dict:
    """Per-image agreement, restricted to `classes`."""
    cl = set(classes)
    jac, exact, nofp, ham = [], [], [], []
    for t, p in zip(true_sets, pred_sets):
        t, p = t & cl, p & cl
        u = t | p
        jac.append(len(t & p) / len(u) if u else 1.0)
        exact.append(t == p)
        nofp.append(p <= t)
        ham.append(len(t ^ p) / len(cl))
    n = len(jac)
    return {
        "n": n,
        "mean_jaccard": sum(jac) / n if n else float("nan"),
        "exact_match": sum(exact) / n if n else float("nan"),
        "no_false_positive": sum(nofp) / n if n else float("nan"),
        "hamming_loss": sum(ham) / n if n else float("nan"),
    }


def macro(per_class: dict, key: str) -> float:
    vals = [m[key] for m in per_class.values() if m.get(key) == m.get(key)]
    return sum(vals) / len(vals) if vals else float("nan")


# ============================================================
# Cluster bootstrap
# ============================================================

def cluster_bootstrap(stat: Callable[[list], float], clusters: dict,
                      B: int = 2000, seed: int = 0) -> dict:
    """Percentile CI of stat(sample) with whole clusters resampled.

    clusters: {cluster_id: [item, ...]}. stat receives a flat list of items.
    """
    ids = sorted(clusters)
    rng = random.Random(seed)
    point = stat([x for c in ids for x in clusters[c]])
    vals = []
    for _ in range(B):
        draw = [rng.choice(ids) for _ in ids]
        v = stat([x for c in draw for x in clusters[c]])
        if v == v:
            vals.append(v)
    vals.sort()
    if not vals:
        return {"point": point, "lo": float("nan"), "hi": float("nan"), "B": 0}
    lo = vals[int(0.025 * (len(vals) - 1))]
    hi = vals[int(0.975 * (len(vals) - 1))]
    return {"point": point, "lo": lo, "hi": hi, "B": len(vals)}


def paired_cluster_bootstrap(stat_a: Callable[[list], float],
                             stat_b: Callable[[list], float],
                             clusters: dict, B: int = 2000, seed: int = 0) -> dict:
    """CI and two-sided p-value for stat_b - stat_a on the SAME resamples."""
    ids = sorted(clusters)
    rng = random.Random(seed)
    flat = [x for c in ids for x in clusters[c]]
    point = stat_b(flat) - stat_a(flat)
    deltas = []
    for _ in range(B):
        draw = [rng.choice(ids) for _ in ids]
        s = [x for c in draw for x in clusters[c]]
        d = stat_b(s) - stat_a(s)
        if d == d:
            deltas.append(d)
    deltas.sort()
    if not deltas:
        return {"delta": point, "lo": float("nan"), "hi": float("nan"), "p": float("nan")}
    lo = deltas[int(0.025 * (len(deltas) - 1))]
    hi = deltas[int(0.975 * (len(deltas) - 1))]
    # two-sided bootstrap p: how often the resampled delta falls on the other
    # side of zero, doubled
    if point >= 0:
        tail = sum(1 for d in deltas if d <= 0) / len(deltas)
    else:
        tail = sum(1 for d in deltas if d >= 0) / len(deltas)
    return {"delta": point, "lo": lo, "hi": hi, "p": min(1.0, 2 * tail), "B": len(deltas)}


# ============================================================
# Paired test on per-image correctness
# ============================================================

def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value. b = A right & B wrong, c = the reverse."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


# ============================================================
# Probability calibration (Platt scaling, 1-D logistic regression)
# ============================================================

def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    e = math.exp(z)
    return e / (1 + e)


def fit_platt(scores: Sequence[float], labels: Sequence[bool],
              iters: int = 100, l2: float = 1e-2) -> Optional[tuple[float, float]]:
    """Fit q = sigmoid(a * logit(p) + b) by Newton's method with a small L2
    penalty on the slope, which keeps the fit finite when a small sample is
    perfectly separable. None if one class is absent (nothing to calibrate
    against)."""
    y = [1.0 if v else 0.0 for v in labels]
    if not any(y) or all(y):
        return None
    x = [_logit(s) for s in scores]

    def loss(a, b):
        # penalised negative log-likelihood, computed stably
        tot = 0.5 * l2 * a * a
        for xi, yi in zip(x, y):
            z = a * xi + b
            # log(1 + e^z) - y z
            tot += (z if z > 0 else 0.0) + math.log1p(math.exp(-abs(z))) - yi * z
        return tot

    # Damped Newton with backtracking. The undamped version overshot when
    # P(yes) spans 1e-4..1 (logits -9..+9) and ran into the parameter clamp,
    # returning a step function instead of a calibration (bug found
    # 2026-09-24 in the first probe report; fixed before any decision used it).
    a, b = 1.0, 0.0
    cur = loss(a, b)
    for _ in range(iters):
        ga = gb = haa = hab = hbb = 0.0
        for xi, yi in zip(x, y):
            q = _sigmoid(a * xi + b)
            r = q - yi
            w = q * (1 - q)
            ga += r * xi
            gb += r
            haa += w * xi * xi
            hab += w * xi
            hbb += w
        ga += l2 * a
        haa += l2
        hbb += 1e-9
        det = haa * hbb - hab * hab
        if abs(det) < 1e-12:
            break
        da = (hbb * ga - hab * gb) / det
        db = (haa * gb - hab * ga) / det
        step = 1.0
        while step > 1e-6:
            na, nb = a - step * da, b - step * db
            nl = loss(na, nb)
            if nl <= cur:
                break
            step /= 2
        else:
            break
        moved = abs(na - a) + abs(nb - b)
        a, b, cur = na, nb, nl
        if moved < 1e-9:
            break
    return a, b


def apply_platt(p: float, ab: Optional[tuple[float, float]]) -> float:
    if ab is None:
        return p
    a, b = ab
    return _sigmoid(a * _logit(p) + b)


def expected_calibration_error(probs: Sequence[float], labels: Sequence[bool],
                               bins: int = 10) -> float:
    n = len(probs)
    if not n:
        return float("nan")
    ece = 0.0
    for k in range(bins):
        lo, hi = k / bins, (k + 1) / bins
        idx = [i for i, p in enumerate(probs) if (lo <= p < hi) or (k == bins - 1 and p == 1.0)]
        if not idx:
            continue
        conf = sum(probs[i] for i in idx) / len(idx)
        acc = sum(1 for i in idx if labels[i]) / len(idx)
        ece += len(idx) / n * abs(conf - acc)
    return ece
