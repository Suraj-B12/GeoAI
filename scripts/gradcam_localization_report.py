"""
Scores the Grad-CAM maps from gradcam_localization_experiment.py against the
annotated Attain boxes, following eval_results/gradcam_localization_preregistration.json.

CPU only, no inference. Steps:
  1. DEV: pointing-game hit rate per (method, layer) variant; pick the best.
  2. Location prior learned on DEV (where each class's boxes usually are).
  3. TEST: chosen variant vs prior, macro over classes, block bootstrap;
     per class; secondary measures (chance, cell AUROC, energy ratio,
     suggested-box IoU, the stored 4-tile probe scores).
  4. Figures 6.10a/b and eval_results/gradcam_localization_report.{md,json}.

Usage:
    venv/Scripts/python.exe scripts/gradcam_localization_report.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from scripts import multilabel_metrics as M  # noqa: E402
from scripts import stage2_probe_report as R  # noqa: E402
from scripts.stage2_gradcam import hottest_box  # noqa: E402

EVAL_DIR = PROJECT_ROOT / "eval_results"
MAP_DIR = EVAL_DIR / "gradcam_attain"
PLOT_DIR = EVAL_DIR / "plots" / "paper"
EVAL_OUT = EVAL_DIR
B, SEED, MIN_PAIRS = 2000, 20260923, 10
CLASSES = R.ALL_CLASSES


# ============================================================
# Geometry
# ============================================================

def cell_centres(grid, size):
    rows, cols = grid
    w, h = size
    cx = (np.arange(cols) + 0.5) * w / cols
    cy = (np.arange(rows) + 0.5) * h / rows
    return np.meshgrid(cx, cy)  # each (rows, cols)


def inside_mask(grid, size, boxes) -> np.ndarray:
    """Cells whose centre lies in any box widened by half a cell."""
    rows, cols = grid
    w, h = size
    hx, hy = 0.5 * w / cols, 0.5 * h / rows
    X, Y = cell_centres(grid, size)
    m = np.zeros(grid, dtype=bool)
    for x1, y1, x2, y2 in boxes:
        m |= (X >= x1 - hx) & (X <= x2 + hx) & (Y >= y1 - hy) & (Y <= y2 + hy)
    return m


def hit(score: np.ndarray, inside: np.ndarray) -> float:
    """Pointing game with ties split evenly (expected hit under a random
    tie-break), so a flat or coarse map is not favoured by argmax order.
    An all-zero map is a miss."""
    if not np.isfinite(score).all() or score.max() <= 0:
        return 0.0
    top = score >= score.max() - 1e-12 * max(1.0, abs(score.max()))
    return float(inside[top].mean())


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# ============================================================
# Data
# ============================================================

RAW = EVAL_DIR / "gradcam_attain_raw.json"


def load_pairs():
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    rows = raw["rows"].values() if isinstance(raw["rows"], dict) else raw["rows"]  # checkpoint or final
    pairs, n_err = [], 0
    for r in rows:
        if "error" in r:
            n_err += 1
            continue
        z = np.load(MAP_DIR / (Path(r["image"]).stem + ".npz"))
        grid, size = tuple(r["grid"]), tuple(r["size"])
        for cls, boxes in r["boxes"].items():
            keys = [k for k in R.PROBE_KEYS[cls] if k in r["keys"]]
            key = max(keys, key=lambda k: r["keys"][k]["p_yes"])
            maps = {name.split("|", 1)[1]: z[name].astype(np.float32)
                    for name in z.files if name.startswith(key + "|")}
            pairs.append({"image": r["image"], "image_path": r["image_path"], "index": r["index"], "split": r["split"],
                          "block": (r["index"] - 1) // R.BLOCK_SIZE, "cls": cls, "key": key,
                          "p_yes": r["keys"][key]["p_yes"], "grid": grid, "size": size,
                          "boxes": boxes, "inside": inside_mask(grid, size, boxes), "maps": maps,
                          "time_s": r.get("time_s"), "n_keys": len(r["keys"])})
    return raw, pairs, n_err


def tile_scores():
    """{image: (tile_boxes, [per-tile probe p dict])} from the stored views study."""
    out = {}
    for f in ("stage2_views_attain_dev.json", "stage2_views_attain_test.json"):
        p = EVAL_DIR / f
        if p.exists():
            for r in json.loads(p.read_text(encoding="utf-8"))["rows"]:
                if "tile_up" in r:
                    out[r["key"]] = (r["tile_boxes"], r["tile_up"])
    return out


# ============================================================
# Scoring
# ============================================================

def macro_hit(pairs, fn, classes) -> float:
    by = defaultdict(list)
    for p in pairs:
        by[p["cls"]].append(fn(p))
    vals = [np.mean(by[c]) for c in classes if by[c]]
    return float(np.mean(vals)) if vals else float("nan")


def blocks(pairs):
    c = defaultdict(list)
    for p in pairs:
        c[p["block"]].append(p)
    return c


def main() -> int:
    global RAW, EVAL_OUT, PLOT_DIR
    if len(sys.argv) > 2 and sys.argv[1] == "--dry-run":
        # Code check on a partial checkpoint; writes to a scratch folder and
        # prints nothing about the numbers, so TEST results are first seen on
        # the full run.
        RAW, EVAL_OUT = EVAL_DIR / "gradcam_attain_checkpoint.json", Path(sys.argv[2])
        PLOT_DIR = EVAL_OUT
        import builtins
        builtins.print = lambda *a, **k: None
    pre = json.loads((EVAL_DIR / "gradcam_localization_preregistration.json").read_text(encoding="utf-8"))
    raw, pairs, n_err = load_pairs()
    dev = [p for p in pairs if p["split"] == "dev"]
    test = [p for p in pairs if p["split"] == "test"]
    n_cls = {s: {c: sum(1 for p in ps if p["cls"] == c) for c in CLASSES} for s, ps in (("dev", dev), ("test", test))}
    classes = [c for c in CLASSES if n_cls["dev"][c] >= MIN_PAIRS and n_cls["test"][c] >= MIN_PAIRS]
    variants = sorted(pairs[0]["maps"], key=lambda v: (int(v.split("@")[1]), v))
    print(f"[data] {len(pairs)} pairs ({len(dev)} dev / {len(test)} test), {n_err} frame errors; "
          f"macro classes {classes}")

    # 1) DEV variant choice
    dev_hits = {v: macro_hit(dev, lambda p, v=v: hit(p["maps"][v], p["inside"]), classes) for v in variants}
    order = sorted(variants, key=lambda v: (-round(dev_hits[v], 12), int(v.split("@")[1]), v != f"gradcam@{v.split('@')[1]}"))
    chosen = order[0]
    print("[dev] " + ", ".join(f"{v} {dev_hits[v]:.3f}" for v in order) + f" -> {chosen}")

    # 2) location prior from DEV, per (class, size), on the cell grid
    acc = defaultdict(list)
    for p in dev:
        acc[(p["cls"], p["size"])].append(p["inside"].astype(np.float32))
    prior = {k: np.mean(v, axis=0) for k, v in acc.items()}
    # fallback for a (class, size) unseen in DEV: the class prior of the most common DEV size,
    # resampled to the frame's grid by relative position
    def prior_map(p):
        k = (p["cls"], p["size"])
        if k in prior:
            return prior[k]
        cands = [kk for kk in prior if kk[0] == p["cls"]]
        if not cands:
            return np.ones(p["grid"], dtype=np.float32)
        src = prior[max(cands, key=lambda kk: len(acc[kk]))]
        rows, cols = p["grid"]
        ii = (np.arange(rows) + 0.5) / rows * src.shape[0]
        jj = (np.arange(cols) + 0.5) / cols * src.shape[1]
        return src[np.clip(ii.astype(int), 0, src.shape[0] - 1)][:, np.clip(jj.astype(int), 0, src.shape[1] - 1)]
    n_fallback = sum(1 for p in test if (p["cls"], p["size"]) not in prior)

    tiles = tile_scores()

    def tile_map(p):
        t = tiles.get(p["image"])
        if t is None:
            return None
        tb, sc = t
        X, Y = cell_centres(p["grid"], p["size"])
        m = np.zeros(p["grid"], dtype=np.float32)
        for b, s in zip(tb, sc):
            v = max(s[k] for k in R.PROBE_KEYS[p["cls"]])
            inb = (X >= b[0]) & (X <= b[2]) & (Y >= b[1]) & (Y <= b[3])
            m = np.where(inb, np.maximum(m, v), m)
        return m

    methods = {
        "gradcam": lambda p: p["maps"][chosen],
        "prior": prior_map,
    }

    def m_hit(name):
        return lambda p: hit(methods[name](p), p["inside"])

    # 3) TEST primary
    cl = blocks(test)
    prim = M.paired_cluster_bootstrap(lambda s: macro_hit(s, m_hit("prior"), classes),
                                      lambda s: macro_hit(s, m_hit("gradcam"), classes), cl, B=B, seed=SEED)
    cam_ci = M.cluster_bootstrap(lambda s: macro_hit(s, m_hit("gradcam"), classes), cl, B=B, seed=SEED)
    pri_ci = M.cluster_bootstrap(lambda s: macro_hit(s, m_hit("prior"), classes), cl, B=B, seed=SEED)
    chance = macro_hit(test, lambda p: float(p["inside"].mean()), classes)
    passed = prim["delta"] > 0 and prim["lo"] > 0
    print(f"[test] gradcam {cam_ci['point']:.3f} prior {pri_ci['point']:.3f} chance {chance:.3f} "
          f"delta {prim['delta']:+.3f} [{prim['lo']:+.3f}, {prim['hi']:+.3f}] -> {'PASS' if passed else 'FAIL'}")

    # per class + secondary
    def auroc(p, m):
        return M.roc_auc(m.ravel().tolist(), p["inside"].ravel().tolist())

    def energy(p, m):
        s = m.sum()
        share = p["inside"].mean()
        if s <= 0 or share <= 0 or share >= 1:
            return float("nan")
        return float(m[p["inside"]].sum() / s / share)

    def box_ok(p, m):
        b = hottest_box(m, p["size"])
        return float(b is not None and max(iou(b, g) for g in p["boxes"]) >= 0.5)

    def box_cover(p, m):
        b = hottest_box(m, p["size"])
        if b is None:
            return float("nan")
        return (b[2] - b[0]) * (b[3] - b[1]) / (p["size"][0] * p["size"][1])

    per_class = {}
    for c in CLASSES:
        ps = [p for p in test if p["cls"] == c]
        if not ps:
            continue
        cc = blocks(ps)
        row = {"n_test": len(ps), "n_dev": n_cls["dev"][c],
               "box_area_share_mean": round(float(np.mean([p["inside"].mean() for p in ps])), 3)}
        for name in ("gradcam", "prior"):
            ci = M.cluster_bootstrap(lambda s, name=name: float(np.mean([m_hit(name)(p) for p in s])), cc, B=B, seed=SEED)
            row[name] = {"hit": round(ci["point"], 3), "lo": round(ci["lo"], 3), "hi": round(ci["hi"], 3),
                         "cell_auroc": round(float(np.nanmean([auroc(p, methods[name](p)) for p in ps])), 3),
                         "energy_ratio": round(float(np.nanmean([energy(p, methods[name](p)) for p in ps])), 3),
                         "box_iou50": round(float(np.mean([box_ok(p, methods[name](p)) for p in ps])), 3),
                         "box_frame_share": round(float(np.nanmean([box_cover(p, methods[name](p)) for p in ps])), 3)}
        d = M.paired_cluster_bootstrap(lambda s: float(np.mean([m_hit("prior")(p) for p in s])),
                                       lambda s: float(np.mean([m_hit("gradcam")(p) for p in s])), cc, B=B, seed=SEED)
        row["delta"] = {k: round(d[k], 3) for k in ("delta", "lo", "hi", "p")}
        row["chance"] = round(float(np.mean([p["inside"].mean() for p in ps])), 3)
        tm = [(p, tile_map(p)) for p in ps]
        tm = [(p, m) for p, m in tm if m is not None]
        if tm:
            row["tiles4"] = {"n": len(tm), "hit": round(float(np.mean([hit(m, p["inside"]) for p, m in tm])), 3),
                             "cell_auroc": round(float(np.nanmean([auroc(p, m) for p, m in tm])), 3)}
        per_class[c] = row
        print(f"  {c:22s} n={len(ps):3d} cam {row['gradcam']['hit']:.2f} prior {row['prior']['hit']:.2f} "
              f"chance {row['chance']:.2f} d {row['delta']['delta']:+.2f} [{row['delta']['lo']:+.2f},{row['delta']['hi']:+.2f}] "
              f"auroc {row['gradcam']['cell_auroc']:.2f}/{row['prior']['cell_auroc']:.2f} "
              f"box {row['gradcam']['box_iou50']:.2f}/{row['prior']['box_iou50']:.2f}")

    # Does the map localise better when the model is confident the damage is there?
    conf = {}
    for lo_, hi_ in ((0, 0.5), (0.5, 1.01)):
        ps = [p for p in test if p["cls"] in classes and lo_ <= p["p_yes"] < hi_]
        conf[f"p_yes {lo_}-{min(hi_, 1)}"] = {"n": len(ps), "gradcam": round(macro_hit(ps, m_hit("gradcam"), classes), 3),
                                               "prior": round(macro_hit(ps, m_hit("prior"), classes), 3)}

    # all variants on TEST, descriptive only (the choice was made on DEV)
    test_all = {v: round(macro_hit(test, lambda p, v=v: hit(p["maps"][v], p["inside"]), classes), 3) for v in variants}

    times = [p["time_s"] / p["n_keys"] for p in pairs if p["time_s"]]
    useful = [c for c in classes if per_class[c]["delta"]["lo"] > 0]
    report = {
        "generated_by": "scripts/gradcam_localization_report.py",
        "preregistration": "eval_results/gradcam_localization_preregistration.json",
        "pass_rule": pre["pass_rule"],
        "n": {"pairs": len(pairs), "dev": len(dev), "test": len(test), "frame_errors": n_err,
              "per_class": n_cls, "macro_classes": classes, "prior_fallbacks_test": n_fallback},
        "dev_variant_hits": {v: round(dev_hits[v], 4) for v in order}, "chosen_variant": chosen,
        "test_variant_hits_descriptive": test_all,
        "test": {"gradcam_macro_hit": {k: round(cam_ci[k], 3) for k in ("point", "lo", "hi")},
                 "prior_macro_hit": {k: round(pri_ci[k], 3) for k in ("point", "lo", "hi")},
                 "chance_macro_hit": round(chance, 3),
                 "delta": {k: round(prim[k], 3) for k in ("delta", "lo", "hi", "p")},
                 "passed": passed, "classes_beating_prior": useful},
        "per_class_test": per_class, "by_probe_confidence_test": conf,
        "seconds_per_explanation": {"median": round(float(np.median(times)), 2),
                                    "p90": round(float(np.percentile(times, 90)), 2)} if times else None,
    }
    (EVAL_OUT / "gradcam_localization_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_md(report)
    figures(report, test, methods, chosen)
    return 0


# ============================================================
# Output
# ============================================================

def write_md(r):
    t = r["test"]
    L = ["# Grad-CAM localisation of the per-type probe (Attain WS_V2.0)", "",
         f"Pre-registered in `{r['preregistration']}`. Generated by `{r['generated_by']}` (no inference).", "",
         f"Pairs (frame, class): {r['n']['pairs']} ({r['n']['dev']} dev / {r['n']['test']} test). "
         f"Macro over {', '.join(r['n']['macro_classes'])}.", "",
         f"**Chosen on DEV:** `{r['chosen_variant']}`.", "",
         "| TEST | Macro pointing hit rate [95% CI] |", "|---|---|",
         f"| Grad-CAM (`{r['chosen_variant']}`) | {t['gradcam_macro_hit']['point']:.3f} [{t['gradcam_macro_hit']['lo']:.3f}, {t['gradcam_macro_hit']['hi']:.3f}] |",
         f"| Location prior (DEV) | {t['prior_macro_hit']['point']:.3f} [{t['prior_macro_hit']['lo']:.3f}, {t['prior_macro_hit']['hi']:.3f}] |",
         f"| Uniform chance | {t['chance_macro_hit']:.3f} |",
         f"| **Grad-CAM − prior** | **{t['delta']['delta']:+.3f} [{t['delta']['lo']:+.3f}, {t['delta']['hi']:+.3f}]**, p = {t['delta']['p']:.3f} |", "",
         f"**Pre-registered result: {'PASS' if t['passed'] else 'FAIL'}.** Classes whose own CI is above the prior: "
         f"{', '.join(t['classes_beating_prior']) or 'none'}.", "",
         "## Per class (TEST)", "",
         "| Class | n | Grad-CAM hit | Prior hit | Chance | Δ [95% CI] | Cell AUROC cam / prior | Box IoU≥0.5 cam / prior | 4 tiles hit |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c, x in r["per_class_test"].items():
        L.append(f"| {c} | {x['n_test']} | {x['gradcam']['hit']:.2f} | {x['prior']['hit']:.2f} | {x['chance']:.2f} | "
                 f"{x['delta']['delta']:+.2f} [{x['delta']['lo']:+.2f}, {x['delta']['hi']:+.2f}] | "
                 f"{x['gradcam']['cell_auroc']:.2f} / {x['prior']['cell_auroc']:.2f} | "
                 f"{x['gradcam']['box_iou50']:.2f} / {x['prior']['box_iou50']:.2f} | "
                 f"{x.get('tiles4', {}).get('hit', float('nan')):.2f} |")
    L += ["", "## DEV variant hit rates", "", "| Variant | DEV | TEST (descriptive) |", "|---|---:|---:|"]
    for v, h in r["dev_variant_hits"].items():
        L.append(f"| {v} | {h:.3f} | {r['test_variant_hits_descriptive'][v]:.3f} |")
    L += ["", "## By probe confidence (TEST)", ""]
    for k, x in r["by_probe_confidence_test"].items():
        L.append(f"- {k}: n = {x['n']}, Grad-CAM {x['gradcam']:.3f}, prior {x['prior']:.3f}")
    if r.get("seconds_per_explanation"):
        s = r["seconds_per_explanation"]
        L += ["", f"Time per explanation (one type, one frame, A5000, 4-bit, checkpointed): median {s['median']} s, p90 {s['p90']} s."]
    (EVAL_OUT / "gradcam_localization_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def figures(r, test, methods, chosen):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    pc = r["per_class_test"]
    cls = [c for c in pc]
    x = np.arange(len(cls))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for off, name, col in ((-0.2, "gradcam", "#2563eb"), (0.2, "prior", "#9ca3af")):
        h = [pc[c][name]["hit"] for c in cls]
        lo = [pc[c][name]["hit"] - pc[c][name]["lo"] for c in cls]
        hi = [pc[c][name]["hi"] - pc[c][name]["hit"] for c in cls]
        ax.bar(x + off, h, 0.38, yerr=[lo, hi], capsize=3, color=col,
               label=f"Grad-CAM ({chosen})" if name == "gradcam" else "Location prior (dev)")
    ax.scatter(x, [pc[c]["chance"] for c in cls], marker="_", s=400, color="#dc2626", zorder=3, label="Uniform chance")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={pc[c]['n_test']})" for c in cls], fontsize=8)
    ax.set_ylabel("Pointing-game hit rate (test)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "fig_6_10a_gradcam_pointing.png", dpi=160)
    plt.close(fig)

    # Examples: the first TEST frame of each headline class (by frame index), not picked by result.
    from scripts.stage2_gradcam import overlay
    ex = []
    for c in R.HEADLINE + [R.PATCH]:
        ps = sorted([p for p in test if p["cls"] == c], key=lambda p: p["index"])
        if ps:
            ex.append(ps[0])
    fig, axes = plt.subplots(len(ex), 1, figsize=(8, 2.6 * len(ex)))
    for ax, p in zip(np.atleast_1d(axes), ex):
        im = Image.open(p["image_path"]).convert("RGB")
        ov = overlay(im, p["maps"][chosen])
        d = ImageDraw.Draw(ov)
        for b in p["boxes"]:
            d.rectangle(b, outline=(37, 99, 235), width=4)
        m = p["maps"][chosen]
        i, j = np.unravel_index(int(np.argmax(m)), m.shape)
        cx, cy = (j + 0.5) * p["size"][0] / p["grid"][1], (i + 0.5) * p["size"][1] / p["grid"][0]
        rr = max(6, p["size"][0] // 90)
        d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=(250, 204, 21), width=4)
        ax.imshow(ov)
        ax.set_axis_off()
        ax.set_title(f"{p['cls']} — {p['key']} P(yes) {p['p_yes']:.2f} — "
                     f"{'hit' if hit(m, p['inside']) >= 0.5 else 'miss'}  ({p['image'][-10:-4]})", fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "fig_6_10b_gradcam_examples.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
