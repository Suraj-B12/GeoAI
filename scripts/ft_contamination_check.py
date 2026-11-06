"""
Contamination check for the post-training study (paper §6.11): does any frame we
might TRAIN on (Attain WS_V1.0, OS_V1.0) appear, whole or as a crop, in the
frames we EVALUATE on (Attain WS_V2.0, dev and test)?

PostTrainBench (Rank et al., 2026) found that the main way agents "improved" a
model was by training on the evaluation data, sometimes indirectly through a
derived dataset. WS_V1.0 and WS_V2.0 come from the same lab and the same kind of
drive, so a V1 crop cut out of a V2 frame is the obvious way that could happen.

Method (keypoint geometry, not appearance):
  1. ORB keypoints + descriptors on every image at native resolution (8-level
     pyramid, scale factor 1.2, so a crop shown at 0.3x-3.6x its size still matches).
  2. Retrieval: an LSH index over all WS_V2.0 descriptors; each training frame's
     descriptors vote (Lowe ratio test) for the V2 frames they match.
  3. Verification: for the top-K voted V2 frames, matched keypoints are fitted
     with a RANSAC homography; the inlier count is the evidence. A real crop or
     copy gives a large, geometrically consistent inlier set; similar-looking
     asphalt gives a handful of scattered matches.
  4. Controls, so the threshold is measured rather than assumed:
       positive: synthetic crops of WS_V2.0 frames (native crop, rescaled crop,
                 upscaled crop of the 508-row frames, JPEG re-encoded) must be
                 found;
       negative: WS_V2.0 frames of one 20-frame block vs frames of other,
                 non-adjacent blocks (same camera, same roads, different places).
  A training frame is dropped if its best inlier count reaches the threshold
  (the larger of 20 and the 99.9th percentile of the negative control, capped so
  that every positive control is still detected).

Needs OpenCV (the GeoAI_Video pipeline venv has it; the Capstone venv does not).
Run isolated with paths as arguments:
    <geoai venv python> -I scripts/ft_contamination_check.py --attain <Attain root> \
        --cache C:/GeoAI/finetune/contam --out eval_results/finetune/contamination.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

N_FEATURES = 1500
TOP_K = 8
DETECTOR = "orb"          # set from --detector before any worker starts
RATIO = 0.8
RANSAC_PX = 5.0
SEED = 20261106

_orb = None


def _get_orb():
    global _orb
    if _orb is None:
        if DETECTOR == "sift":
            _orb = cv2.SIFT_create(nfeatures=N_FEATURES, contrastThreshold=0.02)
        else:
            _orb = cv2.ORB_create(nfeatures=N_FEATURES, scaleFactor=1.2, nlevels=8,
                                  edgeThreshold=15, patchSize=31, fastThreshold=10)
    return _orb


def _set_detector(name: str):
    global DETECTOR, _orb
    DETECTOR, _orb = name, None


def _norm():
    return cv2.NORM_L2 if DETECTOR == "sift" else cv2.NORM_HAMMING


def features(img: np.ndarray):
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = _get_orb().detectAndCompute(g, None)
    if des is None:
        return (np.zeros((0, 2), np.float32),
                np.zeros((0, 128), np.float32) if DETECTOR == "sift" else np.zeros((0, 32), np.uint8))
    return np.array([k.pt for k in kp], np.float32), des


def _feat_init(detector: str):
    _set_detector(detector)


def features_of_path(p: str):
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"unreadable image {p}")
    return features(img)


def inliers(pa, da, pb, db) -> tuple[int, int]:
    """(RANSAC homography inliers, ratio-test matches) between two images."""
    if len(da) < 8 or len(db) < 8:
        return 0, 0
    bf = cv2.BFMatcher(_norm())
    good = []
    for m in bf.knnMatch(da, db, k=2):
        if len(m) == 2 and m[0].distance < RATIO * m[1].distance:
            good.append(m[0])
    if len(good) < 8:
        return 0, len(good)
    src = pa[[m.queryIdx for m in good]].reshape(-1, 1, 2)
    dst = pb[[m.trainIdx for m in good]].reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX)
    if H is None or mask is None:
        return 0, len(good)
    return int(mask.sum()), len(good)


# ---------------- worker state (one copy per process) ----------------
_V2 = None


def _init(cache_file: str, detector: str = "orb"):
    global _V2
    _set_detector(detector)
    z = np.load(cache_file, allow_pickle=True)
    _V2 = {"pts": list(z["pts"]), "des": list(z["des"])}
    # LSH index over all V2 descriptors, with the owning image per row
    _V2["owner"] = np.concatenate([np.full(len(d), i, np.int32) for i, d in enumerate(_V2["des"])])
    allde = np.concatenate([d for d in _V2["des"] if len(d)])
    if DETECTOR == "sift":
        idx = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=64))
    else:
        idx = cv2.FlannBasedMatcher(dict(algorithm=6, table_number=8, key_size=16,
                                         multi_probe_level=1), dict(checks=64))
    idx.add([allde])
    idx.train()
    _V2["index"] = idx


def _query(pts, des):
    """Best V2 match for one image: (inliers, matches, v2_index), plus top-K list."""
    if len(des) < 8:
        return (0, 0, -1), []
    votes = np.zeros(len(_V2["des"]), np.int32)
    for m in _V2["index"].knnMatch(des, k=2):
        if len(m) == 2 and m[0].distance < RATIO * m[1].distance:
            votes[_V2["owner"][m[0].trainIdx]] += 1
    cand = np.argsort(-votes)[:TOP_K]
    res = []
    for j in cand:
        il, nm = inliers(pts, des, _V2["pts"][j], _V2["des"][j])
        res.append((il, nm, int(j), int(votes[j])))
    res.sort(key=lambda r: -r[0])
    return res[0][:3], res


def _task_path(p: str):
    pts, des = features_of_path(p)
    best, res = _query(pts, des)
    return p, best, [r[:2] + (r[2],) for r in res[:3]]


def _task_array(item):
    name, img = item
    pts, des = features(img)
    best, res = _query(pts, des)
    return name, best


def _neg_pair(args):
    a, b = args
    return inliers(_V2["pts"][a], _V2["des"][a], _V2["pts"][b], _V2["des"][b])[0]


def synthetic_crops(v2_paths, n, rng):
    """Crops of real V2 frames, transformed the ways a derived dataset could be."""
    out = []
    for k in range(n):
        p = rng.choice(v2_paths)
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        mode = k % 4
        if mode == 0:            # native 640x640 crop (or as large as fits)
            s = min(640, h, w)
        elif mode == 1:          # larger crop downscaled to 640
            s = min(int(640 / 0.5625), h, w)
        elif mode == 2:          # small crop upscaled to 640
            s = min(360, h, w)
        else:                    # medium crop, then JPEG q=70
            s = min(480, h, w)
        y = rng.randint(0, h - s)
        x = rng.randint(0, w - s)
        c = cv2.resize(img[y:y + s, x:x + s], (640, 640), interpolation=cv2.INTER_AREA)
        if mode == 3:
            ok, buf = cv2.imencode(".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, 70])
            c = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        out.append((f"synthetic_{k}_{['native', 'down', 'up', 'jpeg'][mode]}_{p.name}", c))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attain", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--n-positive", type=int, default=80)
    ap.add_argument("--n-negative", type=int, default=4000)
    ap.add_argument("--detector", choices=["orb", "sift"], default="orb")
    a = ap.parse_args()
    _set_detector(a.detector)
    t0 = time.time()
    root = Path(a.attain)
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    v2 = sorted((root / "Attain_SMP_WS_V2.0" / "Images").glob("*.jpg"))
    train = {s: sorted((root / f"Attain_SMP_{s}" / "Images").glob("*.jpg"))
             for s in ("WS_V1.0", "OS_V1.0")}

    cfile = cache / f"v2_{a.detector}.npz"
    if not cfile.exists():
        with ProcessPoolExecutor(a.workers, initializer=_feat_init, initargs=(a.detector,)) as ex:
            feats = list(ex.map(features_of_path, [str(p) for p in v2], chunksize=8))
        np.savez(cfile, pts=np.array([f[0] for f in feats], dtype=object),
                 des=np.array([f[1] for f in feats], dtype=object), names=[p.name for p in v2])
    print(f"[v2] {len(v2)} frames featurised ({time.time() - t0:.0f}s)", flush=True)

    rng = random.Random(SEED)
    # Controls
    pos = synthetic_crops(v2, a.n_positive, rng)
    blocks = [(int(p.stem.rsplit("_", 1)[-1]) - 1) // 20 for p in v2]
    pairs = []
    while len(pairs) < a.n_negative:
        i, j = rng.randrange(len(v2)), rng.randrange(len(v2))
        if abs(blocks[i] - blocks[j]) >= 2:
            pairs.append((i, j))
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(str(cfile), a.detector)) as ex:
        pos_res = list(ex.map(_task_array, pos, chunksize=2))
        neg = list(ex.map(_neg_pair, pairs, chunksize=50))
        print(f"[controls] done ({time.time() - t0:.0f}s)", flush=True)
        train_res = {}
        for s, ps in train.items():
            train_res[s] = list(ex.map(_task_path, [str(p) for p in ps], chunksize=4))
            print(f"[{s}] {len(ps)} frames checked ({time.time() - t0:.0f}s)", flush=True)

    pos_il = np.array([r[1][0] for r in pos_res])
    neg_il = np.array(neg)
    neg999 = float(np.percentile(neg_il, 99.9))
    thr = max(20.0, neg999)
    if pos_il.min() < thr:
        # never set the bar above what a known crop reaches
        thr = max(float(pos_il.min()), 12.0)
    flagged = {}
    summary = {}
    for s, res in train_res.items():
        il = np.array([r[1][0] for r in res])
        summary[s] = {
            "n": len(res),
            "best_inliers_quantiles": {q: float(np.percentile(il, q)) for q in (50, 90, 99, 100)},
            "n_flagged": int((il >= thr).sum()),
        }
        for p, best, top in res:
            if best[0] >= thr:
                flagged[Path(p).name] = {"subset": s, "inliers": best[0], "matches": best[1],
                                         "v2_frame": v2[best[2]].name, "top3": [
                                             {"inliers": t[0], "matches": t[1], "v2_frame": v2[t[2]].name}
                                             for t in top]}
    top_pairs = sorted(((r[1][0], Path(r[0]).name, v2[r[1][2]].name if r[1][2] >= 0 else None)
                        for res in train_res.values() for r in res), reverse=True)[:25]
    out = {
        "method": {"features": (f"SIFT nfeatures={N_FEATURES} contrastThreshold=0.02" if a.detector == "sift"
                                 else f"ORB nfeatures={N_FEATURES} scaleFactor=1.2 nlevels=8"),
                   "retrieval": (f"FLANN {'KD-tree' if a.detector == 'sift' else 'LSH'} over all WS_V2.0 "
                                 f"descriptors, Lowe ratio {RATIO}, top {TOP_K} by votes"),
                   "verification": f"RANSAC homography, reprojection {RANSAC_PX}px; evidence = inlier count",
                   "threshold_rule": "max(20, 99.9th pct of negative control), never above the weakest positive control",
                   "seed": SEED},
        "controls": {
            "positive_synthetic_crops": {"n": len(pos_il), "min_inliers": int(pos_il.min()),
                                         "median_inliers": float(np.median(pos_il)),
                                         "detected_at_threshold": int((pos_il >= thr).sum()),
                                         "per_mode_min": {m: int(min(r[1][0] for r in pos_res if f"_{m}_" in r[0]))
                                                          for m in ("native", "down", "up", "jpeg")}},
            "negative_v2_cross_block_pairs": {"n": len(neg_il), "max": int(neg_il.max()),
                                              "p99": float(np.percentile(neg_il, 99)), "p99_9": neg999},
        },
        "threshold_inliers": thr,
        "train_subsets": summary,
        "flagged": flagged,
        "top_25_pairs_by_inliers": [{"inliers": t[0], "train": t[1], "v2": t[2]} for t in top_pairs],
        "per_control": [{"name": r[0], "inliers": r[1][0]} for r in pos_res],
        "all_train_best": {Path(r[0]).name: {"inliers": r[1][0], "v2_frame": v2[r[1][2]].name if r[1][2] >= 0 else None}
                           for res in train_res.values() for r in res},
        "seconds": round(time.time() - t0, 1),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("controls", "threshold_inliers", "train_subsets")}, indent=1))
    print(f"[done] {len(flagged)} flagged -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
