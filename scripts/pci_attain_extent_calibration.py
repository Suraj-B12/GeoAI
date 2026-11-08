"""
Attain WS_V2.0 -> IRC:82-2023 distress EXTENT (% area) and pothole area calibration of the per-type probe.

Stages (each writes into eval_results/pci/extent/, each is re-runnable):

    gt         per-frame ground truth from the Pascal-VOC boxes: union box area of each IRC:82-2023 parameter
               inside a road region of interest (ROI), for four ROI definitions; pothole box count and
               connected-component count.                                      -> attain_gt.json
    roi-png    ROI overlays for the by-eye check                              -> roi_check/*.jpg
    fit        dev split only: candidate models compared by grouped CV inside dev (pre-registered rule),
               chosen model refitted on all dev frames, block-bootstrap curves, residual model
                                                                              -> attain_extent_calibration.json,
                                                                                 dev_fit.json
    test       ONCE: frozen calibration on the test blocks vs baselines        -> test_results.json
               (refuses to run unless PREREG.md exists and its sha256 matches PREREG.sha256)
    cv         grouped 5-fold CV over all 847 frames (secondary look)          -> cv_results.json
    transfer   Attain vs Bengaluru video probe-score distributions (local PostgreSQL), sample stretches
                                                                              -> transfer.json

Split: identical to scripts/stage2_probe_report.split (blocks of 20 consecutive frames, seed 20260923, 35 %
of blocks DEV). Rules: eval_results/pci/extent/PREREG.md. Mapping: eval_results/pci/extent/MAPPING.md.

CPU only. Usage:
    venv/Scripts/python.exe scripts/pci_attain_extent_calibration.py gt|roi-png|fit|test|cv|transfer
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import pci_extent as PE  # noqa: E402

OUT = PROJECT_ROOT / "eval_results" / "pci" / "extent"
RAW = PROJECT_ROOT / "eval_results" / "stage2_probe_raw_attain.json"
ATTAIN = PROJECT_ROOT / "Attain" / "Attain" / "Attain" / "Attain_SMP_WS_V2.0"
VARIANT = "min:name"            # production probe variant (configs/stage2_probe.json)

# ---------------------------------------------------------------- split (copy of stage2_probe_report)
BLOCK_SIZE = 20
DEV_FRAC = 0.35
SPLIT_SEED = 20260923
CV_SEED = 20260924
K = 5


def block_of(index: int) -> int:
    return (index - 1) // BLOCK_SIZE


def split_blocks(indices) -> tuple[set, set]:
    blocks = sorted({block_of(i) for i in indices})
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(blocks)
    n_dev = round(DEV_FRAC * len(blocks))
    return set(blocks[:n_dev]), set(blocks[n_dev:])


def cv_folds(blocks) -> dict:
    """Same construction as stage2_probe_cv.py: shuffled blocks, fold = position mod K."""
    b = sorted(blocks)
    rng = random.Random(CV_SEED)
    rng.shuffle(b)
    return {x: i % K for i, x in enumerate(b)}


# ---------------------------------------------------------------- Attain classes -> IRC:82-2023 parameters
# MAPPING.md. 'Patch and utility cut- Low' has no space before the dash: split on '-' and strip.
GT_PARAMS = {
    "cracking": ("Alligator crack", "Linear crack", "Block crack"),
    "ravelling": ("Raveling",),
    "ravelling_incl_weathering": ("Raveling", "Weathering"),      # sensitivity only
    "patching": ("Patch and utility cut",),
    "potholes": ("Pothole",),
}
EXCLUDED = ("Faded marking", "Lane shoulder drop-off", "Manhole")


def parse_class(name: str) -> tuple[str, str]:
    base, _, sev = name.rpartition("-")
    if not base:
        return name.strip(), ""
    return base.strip(), sev.strip()


# ---------------------------------------------------------------- road ROI (fractions of width/height)
# Trapezoid (y_top, x_top_left, x_top_right, y_bottom, x_bot_left, x_bot_right), checked by eye (roi_check/).
ROI_TRAPEZOID = {
    (1479, 508): (0.00, 0.00, 1.00, 1.00, 0.00, 1.00),      # already a road-band crop
    (1920, 1080): (0.50, 0.05, 0.92, 1.00, 0.00, 1.00),     # sky/buildings above ~0.45 H
    (640, 640): (0.00, 0.08, 0.92, 0.97, 0.00, 1.00),       # verge at top corners, vehicle hood at bottom
}
ROI_NAMES = ("trapezoid", "full_image", "box_hull", "trapezoid_near_half")
PRIMARY_ROI = "trapezoid"


def trapezoid_mask(w: int, h: int, spec, ys=None) -> np.ndarray:
    yt, xtl, xtr, yb, xbl, xbr = spec
    yy = (np.arange(h) + 0.5) / h
    xx = (np.arange(w) + 0.5) / w
    inside_y = (yy >= yt) & (yy <= yb)
    t = np.clip((yy - yt) / max(yb - yt, 1e-9), 0, 1)
    left = xtl + (xbl - xtl) * t
    right = xtr + (xbr - xtr) * t
    m = (xx[None, :] >= left[:, None]) & (xx[None, :] <= right[:, None]) & inside_y[:, None]
    if ys is not None:
        m &= ((yy >= ys[0]) & (yy <= ys[1]))[:, None]
    return m


def hull_mask(w: int, h: int, boxes) -> np.ndarray:
    """Convex hull of every annotated box corner (incl. faded marking / drop-off), rasterised."""
    pts = []
    for (x0, y0, x1, y1) in boxes:
        pts += [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
    m = np.zeros((h, w), bool)
    if len(pts) < 3:
        return m
    pts = sorted(set(pts))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    from PIL import Image, ImageDraw
    im = Image.new("1", (w, h), 0)
    ImageDraw.Draw(im).polygon(hull, fill=1)
    return np.array(im, bool)


def read_xml(path: Path):
    t = ET.parse(path).getroot()
    s = t.find("size")
    w, h = int(float(s.find("width").text)), int(float(s.find("height").text))
    objs = []
    for o in t.findall("object"):
        cls, sev = parse_class(o.find("name").text)
        b = o.find("bndbox")
        x0, x1 = (float(b.find(k).text) for k in ("xmin", "xmax"))
        y0, y1 = (float(b.find(k).text) for k in ("ymin", "ymax"))
        x0, x1 = max(0, min(x0, x1)), min(w, max(x0, x1))
        y0, y1 = max(0, min(y0, y1)), min(h, max(y0, y1))
        if x1 > x0 and y1 > y0:
            objs.append((cls, sev, (int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1)))))
    return t.find("filename").text, w, h, objs


def components(mask: np.ndarray) -> int:
    """4-connected components of a boolean mask (iterative flood fill on the box-union mask)."""
    m = mask.copy()
    n = 0
    H, W = m.shape
    ys, xs = np.nonzero(m)
    for y, x in zip(ys, xs):
        if not m[y, x]:
            continue
        n += 1
        stack = [(y, x)]
        m[y, x] = False
        while stack:
            cy, cx = stack.pop()
            for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                if 0 <= ny < H and 0 <= nx < W and m[ny, nx]:
                    m[ny, nx] = False
                    stack.append((ny, nx))
    return n


def frame_gt(path: Path, scale: int = 4) -> dict:
    """Area fractions on a 1/scale raster (boxes rounded outward), per parameter and ROI."""
    fn, W, H, objs = read_xml(path)
    w, h = math.ceil(W / scale), math.ceil(H / scale)

    def box_mask(classes, roi=None):
        m = np.zeros((h, w), bool)
        for cls, _, (x0, y0, x1, y1) in objs:
            if cls in classes:
                m[y0 // scale: math.ceil(y1 / scale), x0 // scale: math.ceil(x1 / scale)] = True
        return m
    spec = ROI_TRAPEZOID[(W, H)]
    rois = {
        "trapezoid": trapezoid_mask(w, h, spec),
        "full_image": np.ones((h, w), bool),
        "box_hull": hull_mask(w, h, [(x0 // scale, y0 // scale, math.ceil(x1 / scale), math.ceil(y1 / scale))
                                     for _, _, (x0, y0, x1, y1) in objs]),
        # nearer half of the trapezoid's rows (closer to the strip our video segments use)
        "trapezoid_near_half": trapezoid_mask(w, h, spec, ys=((spec[0] + spec[3]) / 2, spec[3])),
    }
    out = {"image": fn, "size": [W, H], "classes": sorted({c for c, _, _ in objs}),
           "area": {}, "roi_area_frac": {k: float(v.mean()) for k, v in rois.items()}}
    for p, classes in GT_PARAMS.items():
        bm = box_mask(classes)
        out["area"][p] = {}
        for rn, rm in rois.items():
            ra = rm.sum()
            out["area"][p][rn] = float((bm & rm).sum() / ra) if ra > 0 else 0.0
    pm = box_mask(("Pothole",))
    out["pothole_boxes"] = sum(1 for c, _, _ in objs if c == "Pothole")
    out["pothole_components"] = components(pm)
    return out


# ---------------------------------------------------------------- data assembly
def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_rows() -> list[dict]:
    """Probe scores + Stage 2 labels from the raw file joined with attain_gt.json; adds block and split."""
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    gt = {g["image"]: g for g in json.loads((OUT / "attain_gt.json").read_text(encoding="utf-8"))["frames"]}
    dev_b, test_b = split_blocks([r["index"] for r in raw["rows"]])
    rows = []
    for r in raw["rows"]:
        if "error" in r or not r.get("probe"):
            continue
        g = gt[r["image"]]
        rows.append({"image": r["image"], "index": r["index"], "block": block_of(r["index"]),
                     "split": "dev" if block_of(r["index"]) in dev_b else "test",
                     "scores": r["probe"][VARIANT]["p"],
                     "labels": [t for t in (r["v0"].get("distress_types") or [])],
                     "size": g["size"], "area": g["area"], "pothole_boxes": g["pothole_boxes"],
                     "pothole_components": g["pothole_components"]})
    return rows


def target(r, p, roi=PRIMARY_ROI) -> float:
    """Ground-truth area in PERCENT of the ROI."""
    return 100.0 * r["area"][p][roi]


# ---------------------------------------------------------------- stage: gt
def cmd_gt(args) -> int:
    files = sorted((ATTAIN / "Labels").glob("*.xml"))
    frames = [frame_gt(f) for f in files]
    blob = {"source": str(ATTAIN.relative_to(PROJECT_ROOT)), "n": len(frames), "raster_scale": 4,
            "roi_trapezoid": {f"{w}x{h}": v for (w, h), v in ROI_TRAPEZOID.items()},
            "params": {k: list(v) for k, v in GT_PARAMS.items()}, "frames": frames}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "attain_gt.json").write_text(json.dumps(blob, indent=0), encoding="utf-8")
    # summary
    summ = {}
    for p in GT_PARAMS:
        summ[p] = {}
        for rn in ROI_NAMES:
            v = np.array([f["area"][p][rn] for f in frames]) * 100
            summ[p][rn] = {"mean_pct": float(v.mean()), "median_pct": float(np.median(v)),
                           "p90_pct": float(np.percentile(v, 90)), "frames_gt0": int((v > 0).sum())}
    pb = np.array([f["pothole_boxes"] for f in frames])
    pc = np.array([f["pothole_components"] for f in frames])
    summ["pothole_counts"] = {"frames_with_pothole": int((pb > 0).sum()), "boxes": int(pb.sum()),
                              "components": int(pc.sum())}
    summ["roi_area_frac_mean"] = {rn: float(np.mean([f["roi_area_frac"][rn] for f in frames]))
                                  for rn in ROI_NAMES}
    (OUT / "attain_gt_summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))
    return 0


def cmd_roi_png(args) -> int:
    from PIL import Image, ImageDraw
    d = OUT / "roi_check"
    d.mkdir(parents=True, exist_ok=True)
    files = sorted((ATTAIN / "Labels").glob("*.xml"))
    by = {}
    for f in files:
        fn, W, H, objs = read_xml(f)
        by.setdefault((W, H), []).append((fn, objs))
    rng = random.Random(7)
    for (W, H), L in by.items():
        pick = rng.sample(L, 7)
        tiles = []
        for fn, objs in pick:
            im = Image.open(ATTAIN / "Images" / fn).convert("RGB")
            dr = ImageDraw.Draw(im)
            yt, xtl, xtr, yb, xbl, xbr = ROI_TRAPEZOID[(W, H)]
            dr.polygon([(xtl * W, yt * H), (xtr * W, yt * H), (xbr * W, yb * H), (xbl * W, yb * H)],
                       outline=(0, 255, 0), width=max(3, W // 300))
            for cls, _, b in objs:
                col = (255, 255, 0) if cls in EXCLUDED else (255, 0, 0)
                dr.rectangle(b, outline=col, width=2)
            im.thumbnail((640, 640))
            tiles.append(im)
        tw, th = tiles[0].size
        sheet = Image.new("RGB", (tw * 4, th * 2), "white")
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % 4) * tw, (i // 4) * th))
        sheet.save(d / f"roi_{W}x{H}.jpg", quality=80)
    print("wrote", d)
    return 0


# ---------------------------------------------------------------- stage: fit (dev only)
def cmd_fit(args) -> int:
    rows = load_rows()
    dev = [r for r in rows if r["split"] == "dev"]
    cal, diag = PE.fit_calibration(dev, target_fn=target, block_key="block", n_boot=args.boot, seed=SPLIT_SEED,
                                   select_folds=K, select_seed=CV_SEED)
    cal["fitted_on"] = {"dataset": "Attain SMP WS_V2.0", "split": "DEV",
                        "split_rule": f"blocks of {BLOCK_SIZE} consecutive frames, seed {SPLIT_SEED}, "
                                      f"{int(DEV_FRAC * 100)} % of blocks (scripts/stage2_probe_report.split)",
                        "n_frames": len(dev), "n_blocks": len({r['block'] for r in dev}),
                        "probe_variant": VARIANT, "roi": PRIMARY_ROI,
                        "raw_sha256": sha256_file(RAW), "gt_sha256": sha256_file(OUT / "attain_gt.json"),
                        "code_sha256": {"app/pci_extent.py": sha256_file(PROJECT_ROOT / "app" / "pci_extent.py"),
                                        "scripts/pci_attain_extent_calibration.py": sha256_file(Path(__file__))}}
    PE.validate_calibration(cal)
    (OUT / "attain_extent_calibration.json").write_text(json.dumps(cal, indent=1), encoding="utf-8")
    (OUT / "dev_fit.json").write_text(json.dumps(diag, indent=1), encoding="utf-8")
    print(json.dumps(diag["selection"], indent=1))
    return 0


# ---------------------------------------------------------------- evaluation helpers
def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")

    def rank(x):
        o = np.argsort(x, kind="mergesort")
        r = np.empty(len(x))
        r[o] = np.arange(len(x))
        # average ties
        for v in np.unique(x):
            idx = np.where(x == v)[0]
            if len(idx) > 1:
                r[idx] = r[idx].mean()
        return r
    ra, rb = rank(a), rank(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - s) / d, (c + s) / d)


def block_eval(rows, cal, params, flag_rule, dev_stats, n_boot=2000, seed=SPLIT_SEED):
    """Per-block predicted vs true mean area % for the model and the baselines; block-bootstrap CIs."""
    blocks = sorted({r["block"] for r in rows})
    byb = {b: [r for r in rows if r["block"] == b] for b in blocks}
    res = {}
    for p in params:
        truth, model, lo, hi, const, flag, flag_lin = [], [], [], [], [], [], []
        for b in blocks:
            fr = byb[b]
            truth.append(float(np.mean([target(r, p) for r in fr])))
            est = PE.estimate_extents([r["scores"] for r in fr], [1.0] * len(fr), calibration=cal,
                                      labels=[r["labels"] for r in fr], params=(p,))[p]
            model.append(est["mean"])
            lo.append(est["lo90"])
            hi.append(est["hi90"])
            const.append(dev_stats[p]["const"])
            f = float(np.mean([flag_rule(r, p) for r in fr]))
            flag.append(100.0 * f)
            flag_lin.append(max(0.0, dev_stats[p]["flag_a"] + dev_stats[p]["flag_b"] * f))
        T = np.array(truth)
        preds = {"model": np.array(model), "constant_dev_mean": np.array(const),
                 "flagged_pct_of_frames": np.array(flag), "flagged_linear_dev": np.array(flag_lin)}
        out = {"n_blocks": len(blocks), "true_mean_pct": float(T.mean()), "per_block": {
            "block": blocks, "true": truth, "model": model, "lo90": lo, "hi90": hi}}
        for name, P in preds.items():
            out[name] = {"mae": float(np.mean(np.abs(P - T))), "bias": float(np.mean(P - T)),
                         "spearman": spearman(P, T)}
        cov = [(l <= t <= h) for l, t, h in zip(lo, truth, hi)]
        k = int(sum(cov))
        out["coverage90"] = {"covered": k, "n": len(cov), "rate": k / len(cov), "wilson95": wilson(k, len(cov)),
                             "mean_width_pct": float(np.mean(np.array(hi) - np.array(lo))),
                             "below": int(sum(t < l for l, t in zip(lo, truth))),
                             "above": int(sum(t > h for h, t in zip(hi, truth)))}
        # block bootstrap of MAE differences and Spearman
        rng = np.random.default_rng(seed)
        n = len(blocks)
        d_const, d_flag, d_lin, rho = [], [], [], []
        Pm = preds["model"]
        for _ in range(n_boot):
            i = rng.integers(0, n, n)
            em = np.abs(Pm[i] - T[i]).mean()
            d_const.append(em - np.abs(preds["constant_dev_mean"][i] - T[i]).mean())
            d_flag.append(em - np.abs(preds["flagged_pct_of_frames"][i] - T[i]).mean())
            d_lin.append(em - np.abs(preds["flagged_linear_dev"][i] - T[i]).mean())
            rho.append(spearman(Pm[i], T[i]))
        q = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]  # noqa: E731
        out["ci95"] = {"mae_model_minus_constant": q(d_const), "mae_model_minus_flagged": q(d_flag),
                       "mae_model_minus_flagged_linear": q(d_lin), "spearman_model": q(rho)}
        res[p] = out
    return res


def flag_rule_factory(cal):
    """The current '% of frames flagged' rule. cracking / potholes: Stage 2 free-form labels (what the report
    uses today); ravelling / patching: probe P(yes) >= the Attain production threshold (Raveling 0.08878 from
    configs/stage2_probe.json; Patching has no production threshold -> 0.5, the verify threshold)."""
    probe_cfg = json.loads((PROJECT_ROOT / "configs" / "stage2_probe.json").read_text(encoding="utf-8"))
    rav_t = probe_cfg["groups"]["Raveling"]["threshold"]
    lab = {"cracking": {"Alligator Cracking", "Longitudinal Cracking", "Transverse Cracking", "Edge Cracking",
                        "Hairline Cracks"}, "potholes": {"Potholes"}}

    def f(r, p):
        if p in lab:
            return 1.0 if lab[p] & set(r["labels"]) else 0.0
        if p in ("ravelling", "ravelling_incl_weathering"):
            return 1.0 if r["scores"]["Ravelling"] >= rav_t else 0.0
        if p == "patching":
            return 1.0 if r["scores"]["Patching"] >= 0.5 else 0.0
        raise KeyError(p)
    return f


def dev_baselines(dev, params, flag_rule):
    st = {}
    for p in params:
        y = np.array([target(r, p) for r in dev])
        st[p] = {"const": float(y.mean())}
        # block-level least squares of mean area on fraction flagged (the a + b f form of pci_core), dev blocks
        blocks = sorted({r["block"] for r in dev})
        F = np.array([np.mean([flag_rule(r, p) for r in dev if r["block"] == b]) for b in blocks])
        Y = np.array([np.mean([target(r, p) for r in dev if r["block"] == b]) for b in blocks])
        if np.ptp(F) > 0:
            b_, a_ = np.polyfit(F, Y, 1)
        else:
            a_, b_ = float(Y.mean()), 0.0
        st[p].update(flag_a=float(a_), flag_b=float(b_))
    return st


def prereg_ok() -> bool:
    pr, sh = OUT / "PREREG.md", OUT / "PREREG.sha256"
    return pr.exists() and sh.exists() and sha256_file(pr) == sh.read_text().split()[0]


def cmd_test(args) -> int:
    if not prereg_ok():
        print("refused: PREREG.md missing or changed since PREREG.sha256 was written")
        return 2
    cal = PE.load_calibration(OUT / "attain_extent_calibration.json")
    if cal["fitted_on"]["gt_sha256"] != sha256_file(OUT / "attain_gt.json"):
        print("refused: attain_gt.json changed since the fit")
        return 2
    rows = load_rows()
    dev = [r for r in rows if r["split"] == "dev"]
    test = [r for r in rows if r["split"] == "test"]
    params = list(cal["params"].keys())
    fr = flag_rule_factory(cal)
    st = dev_baselines(dev, params, fr)
    res = block_eval(test, cal, params, fr, st)
    res["_dev_baselines"] = st
    res["_sensitivity_roi"] = roi_sensitivity(test, cal, params)
    (OUT / "test_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for p in params:
        r = res[p]
        print(p, "true", round(r["true_mean_pct"], 2), "| model MAE", round(r["model"]["mae"], 2), "bias",
              round(r["model"]["bias"], 2), "rho", round(r["model"]["spearman"], 3), "| const MAE",
              round(r["constant_dev_mean"]["mae"], 2), "| flag MAE", round(r["flagged_pct_of_frames"]["mae"], 2),
              "| flaglin MAE", round(r["flagged_linear_dev"]["mae"], 2), "| cov", r["coverage90"]["covered"], "/",
              r["coverage90"]["n"], "ci", [round(x, 2) for x in r["ci95"]["mae_model_minus_constant"]])
    return 0


def roi_sensitivity(rows, cal, params):
    """Block MAE / bias of the frozen model against ground truth under each ROI (the model is fitted on the
    primary ROI; this shows how much the target itself moves with the ROI choice)."""
    out = {}
    blocks = sorted({r["block"] for r in rows})
    for p in params:
        out[p] = {}
        pred = []
        for b in blocks:
            fr = [r for r in rows if r["block"] == b]
            pred.append(PE.estimate_extents([r["scores"] for r in fr], [1.0] * len(fr), calibration=cal,
                                            labels=[r["labels"] for r in fr], params=(p,))[p]["mean"])
        pred = np.array(pred)
        for rn in ROI_NAMES:
            T = np.array([np.mean([target(r, p, rn) for r in rows if r["block"] == b]) for b in blocks])
            out[p][rn] = {"true_mean_pct": float(T.mean()), "model_mae": float(np.abs(pred - T).mean()),
                          "model_bias": float((pred - T).mean()), "spearman": spearman(pred, T)}
    return out


def cmd_cv(args) -> int:
    rows = load_rows()
    folds = cv_folds({r["block"] for r in rows})
    params = None
    oof_rows = []
    cals = []
    for k in range(K):
        train = [r for r in rows if folds[r["block"]] != k]
        hold = [r for r in rows if folds[r["block"]] == k]
        cal, _ = PE.fit_calibration(train, target_fn=target, block_key="block", n_boot=args.boot, seed=SPLIT_SEED + k,
                                    select_folds=K, select_seed=CV_SEED)
        cal["fitted_on"] = {"split": f"cv fold {k}"}
        cals.append({p: v["model"] for p, v in cal["params"].items()})
        params = list(cal["params"].keys())
        fr = flag_rule_factory(cal)
        st = dev_baselines(train, params, fr)
        for b in sorted({r["block"] for r in hold}):
            fb = [r for r in hold if r["block"] == b]
            row = {"block": b, "fold": k}
            for p in params:
                est = PE.estimate_extents([r["scores"] for r in fb], [1.0] * len(fb), calibration=cal,
                                          labels=[r["labels"] for r in fb], params=(p,))[p]
                f = float(np.mean([fr(r, p) for r in fb]))
                row[p] = {"true": float(np.mean([target(r, p) for r in fb])), "model": est["mean"],
                          "lo90": est["lo90"], "hi90": est["hi90"], "const": st[p]["const"], "flag": 100 * f,
                          "flag_lin": max(0.0, st[p]["flag_a"] + st[p]["flag_b"] * f)}
            oof_rows.append(row)
    res = {"k": K, "seed": CV_SEED, "n_blocks": len(oof_rows), "models_per_fold": cals}
    for p in params:
        T = np.array([r[p]["true"] for r in oof_rows])
        o = {"true_mean_pct": float(T.mean())}
        for name in ("model", "const", "flag", "flag_lin"):
            P = np.array([r[p][name] for r in oof_rows])
            o[name] = {"mae": float(np.abs(P - T).mean()), "bias": float((P - T).mean()), "spearman": spearman(P, T)}
        cov = sum(r[p]["lo90"] <= r[p]["true"] <= r[p]["hi90"] for r in oof_rows)
        o["coverage90"] = {"covered": int(cov), "n": len(oof_rows), "rate": cov / len(oof_rows),
                           "wilson95": wilson(cov, len(oof_rows)),
                           "mean_width_pct": float(np.mean([r[p]["hi90"] - r[p]["lo90"] for r in oof_rows]))}
        res[p] = o
        print(p, {k2: (round(v["mae"], 2) if isinstance(v, dict) and "mae" in v else None) for k2, v in o.items()},
              "rho", round(o["model"]["spearman"], 3), "cov", cov, "/", len(oof_rows))
    res["blocks"] = oof_rows
    (OUT / "cv_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


# ---------------------------------------------------------------- stage: transfer (Bengaluru)
def load_env():
    env = PROJECT_ROOT / ".env"
    import os
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def ks_2samp(a, b) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov D and asymptotic p-value."""
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    allv = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, allv, side="right") / len(a)
    cdf_b = np.searchsorted(b, allv, side="right") / len(b)
    d = float(np.max(np.abs(cdf_a - cdf_b)))
    ne = len(a) * len(b) / (len(a) + len(b))
    lam = (math.sqrt(ne) + 0.12 + 0.11 / math.sqrt(ne)) * d
    p = 2 * sum((-1) ** (j - 1) * math.exp(-2 * j * j * lam * lam) for j in range(1, 101))
    return d, float(min(1.0, max(0.0, p)))


def cmd_transfer(args) -> int:
    load_env()
    from psycopg.rows import dict_row

    from app import video_store
    cal = PE.load_calibration(OUT / "attain_extent_calibration.json")
    rows = load_rows()
    with video_store.connect() as c:
        cur = c.cursor(row_factory=dict_row)
        cur.execute("SELECT probe_fp, count(*) n FROM research.video_probe_scores WHERE scores IS NOT NULL "
                    "GROUP BY 1 ORDER BY 2 DESC")
        fps = cur.fetchall()
        fp = fps[0]["probe_fp"]
        cur.execute("SELECT s.segment_id, s.scores, g.length_m, g.status, g.record->'frame' AS frame, "
                    "g.record->'geometry' AS geom FROM research.video_probe_scores s "
                    "JOIN video_segments g USING (segment_id) WHERE s.probe_fp = %s AND s.scores IS NOT NULL", (fp,))
        vid = cur.fetchall()
        cur.execute("SELECT id, title, report->'segment_ids' AS seg, input FROM research.pci_reports "
                    "WHERE id BETWEEN 33 AND 40 ORDER BY id")
        reps = cur.fetchall()
    out = {"probe_fp": fp, "probe_fps": fps, "n_video_segments": len(vid), "n_attain": len(rows), "params": {}}
    for p, spec in cal["params"].items():
        a = np.array([PE.feature(r["scores"], spec["keys"]) for r in rows])
        b = np.array([PE.feature(v["scores"], spec["keys"]) for v in vid])
        d, pv = ks_2samp(a, b)
        qs = [5, 25, 50, 75, 95]
        knots = np.array(spec["model"]["x"])
        out["params"][p] = {
            "keys": spec["keys"], "ks_D": d, "ks_p": pv,
            "attain_quantiles": dict(zip(map(str, qs), np.percentile(a, qs).tolist())),
            "video_quantiles": dict(zip(map(str, qs), np.percentile(b, qs).tolist())),
            "video_share_outside_attain_dev_range": float(np.mean((b < spec["support"][0]) | (b > spec["support"][1]))),
            "attain_mean_pred_pct": float(np.mean([PE.frame_mean(spec, s) for s in a])),
            "video_mean_pred_pct": float(np.mean([PE.frame_mean(spec, s) for s in b])),
            "knots_logit": knots.tolist()}
        print(p, "KS D", round(d, 3), "p", pv, "attain med", round(float(np.median(a)), 4), "video med",
              round(float(np.median(b)), 4), "mean pred attain/video", round(out["params"][p]["attain_mean_pred_pct"], 2),
              round(out["params"][p]["video_mean_pred_pct"], 2))
    # geometry summary of our frames (rows of the segment band, metres ahead)
    fr = [v["frame"] for v in vid if v["frame"] and v["frame"].get("rows_of_segment")]
    band = np.array([(f["rows_of_segment"][1] - f["rows_of_segment"][0]) / f["height"] for f in fr])
    top = np.array([f["rows_of_segment"][0] / f["height"] for f in fr])
    out["video_geometry"] = {"n": len(fr), "band_height_share_median": float(np.median(band)),
                             "band_top_share_median": float(np.median(top)),
                             "near_m_median": float(np.median([f["near_edge_ahead_m"] for f in fr])),
                             "far_m_median": float(np.median([f["far_edge_ahead_m"] for f in fr])),
                             "geometry_sources": sorted({json.dumps(v["geom"].get("source")) for v in vid if v["geom"]}),
                             "pitch_deg_quantiles": np.percentile([v["geom"]["pitch_deg"] for v in vid if v["geom"]],
                                                                  [5, 50, 95]).tolist()}
    # record the shift in the calibration (used by estimate_extents(domain="bengaluru"), PREREG.md)
    cal_path = OUT / "attain_extent_calibration.json"
    cal_raw = json.loads(cal_path.read_text(encoding="utf-8"))
    cal_raw["transfer"] = {p: {"ks_D": v["ks_D"], "ks_p": v["ks_p"], "probe_fp": fp, "n_video": len(vid)}
                           for p, v in out["params"].items()}
    PE.validate_calibration(cal_raw)
    cal_path.write_text(json.dumps(cal_raw, indent=1), encoding="utf-8")
    cal = PE.load_calibration(cal_path)
    # sample stretches: every segment id of the stored report; unscored segments count in the length
    vmap = {v["segment_id"]: v for v in vid}
    out["samples"] = []
    with video_store.connect() as c:
        cur = c.cursor(row_factory=dict_row)
        for rp in reps:
            segs = list(rp["seg"] or [])
            cur.execute("SELECT segment_id, length_m, status FROM video_segments WHERE segment_id = ANY(%s)", (segs,))
            meta = {r["segment_id"]: r for r in cur.fetchall()}
            ids = [s_ for s_ in segs if s_ in meta]
            L = [float(meta[s_]["length_m"] or PE.SEGMENT_M) for s_ in ids]
            sc = [vmap[s_]["scores"] if s_ in vmap else None for s_ in ids]
            lw = float((rp["input"] or {}).get("lane_width_m") or 3.5)
            est_b = PE.estimate_extents(sc, L, calibration=cal, lane_width_m=lw, domain="bengaluru",
                                        params=PE.PCI_PARAMS)
            est_a = PE.estimate_extents(sc, L, calibration=cal, lane_width_m=lw, domain="attain",
                                        params=PE.PCI_PARAMS)
            # per 100 m sections of the stretch (consecutive 20 segment ids)
            secs = []
            for i in range(0, len(ids), 20):
                e = PE.estimate_extents(sc[i:i + 20], L[i:i + 20], calibration=cal, lane_width_m=lw,
                                        domain="bengaluru", params=PE.PCI_PARAMS)
                secs.append({p: (e[p]["mean"], e[p]["lo90"], e[p]["hi90"]) for p in PE.PCI_PARAMS} |
                            {"potholes_number": (e["potholes_number"]["mean"], e["potholes_number"]["lo90"],
                                                 e["potholes_number"]["hi90"]), "length_m": sum(L[i:i + 20])})
            out["samples"].append({"id": rp["id"], "title": rp["title"], "n_segment_ids": len(segs),
                                   "n_with_length": len(ids), "n_scored": sum(x is not None for x in sc),
                                   "length_m": sum(L), "estimate_bengaluru": est_b, "estimate_attain_domain": est_a,
                                   "sections_100m": secs})
            print(rp["id"], rp["title"], len(segs), sum(x is not None for x in sc), round(sum(L)),
                  {p: (round(e["mean"], 2), round(e["lo90"], 2), round(e["hi90"], 2)) for p, e in est_b.items()
                   if isinstance(e, dict) and e.get("mean") is not None})
    (OUT / "transfer.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gt", "roi-png", "fit", "test", "cv", "transfer"])
    ap.add_argument("--boot", type=int, default=400)
    a = ap.parse_args()
    return {"gt": cmd_gt, "roi-png": cmd_roi_png, "fit": cmd_fit, "test": cmd_test, "cv": cmd_cv,
            "transfer": cmd_transfer}[a.stage](a)


if __name__ == "__main__":
    raise SystemExit(main())
