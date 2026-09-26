"""
Score the views experiment: full photo vs tiles vs find-then-zoom vs oracle.

Protocol (eval_results/stage2_probe_preregistration.json, addendum_2_views):
  DEV   choose, per family, the aggregation with the best macro AUROC:
          '+full'  max over {full photo} U views
          'only'   max over the views alone (full photo if there are none)
  TEST  full vs the two finalists (+ oracle as a ceiling): paired macro-AUROC
        delta, cluster bootstrap over 20-frame blocks.
  CV    grouped 5-fold over all 847 frames for the same frozen variants,
        MCC thresholds per variant through the production combine() rule;
        out-of-fold macro-MCC delta vs full.
  BENGALURU  descriptive only (no labels).

Usage:
    venv/Scripts/python.exe scripts/stage2_views_report.py --stage dev
    venv/Scripts/python.exe scripts/stage2_views_report.py --stage final
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import multilabel_metrics as M  # noqa: E402
from scripts import stage2_probe_cv as CV  # noqa: E402
from scripts import stage2_probe_report as R  # noqa: E402
from scripts.stage2_probe import all_probe_types  # noqa: E402
from scripts import stage2_probe_rules as rules  # noqa: E402

EVAL_DIR = PROJECT_ROOT / "eval_results"
FAMILIES = ["tile_native", "tile_up", "zoom", "oracle"]
AGGS = ["+full", "only", "avgmax", "mean", "logitmean"]   # addendum_3: dev-only extension
BOOT = 2000


def agg_p(view_row: dict, fam: str, agg: str) -> dict:
    from scripts.stage2_views import aggregate
    full = view_row["full"]
    if fam == "full":
        return full
    return aggregate(full, view_row.get(fam) or [], agg)


def base_rows():
    d = json.loads((EVAL_DIR / "stage2_probe_raw_attain.json").read_text(encoding="utf-8"))
    rows = [r for r in d["rows"] if "error" not in r]
    R.split(rows)       # sets block / split
    return {r["image"]: r for r in rows}


def with_scores(base: dict, views: dict, fam: str, agg: str) -> list:
    """Base rows whose probe scores are replaced by the variant's aggregated
    scores, so every existing scorer (report, CV, combine) runs unchanged."""
    out = []
    for key, vr in views.items():
        if "error" in vr or key not in base:
            continue
        r = copy.copy(base[key])
        r["probe"] = {CV.VARIANT: {"p": agg_p(vr, fam, agg)}}
        out.append(r)
    return out


def macro_auroc(rows):
    return M.macro({c: {"auroc": M.roc_auc([R.probe_score(r, CV.VARIANT, c) for r in rows],
                                           [c in r["gt"] for r in rows])} for c in R.HEADLINE}, "auroc")


def per_class_auroc(rows, classes):
    return {c: M.roc_auc([R.probe_score(r, CV.VARIANT, c) for r in rows], [c in r["gt"] for r in rows])
            for c in classes}


def load_views(name):
    p = EVAL_DIR / name
    if not p.exists():
        ck = EVAL_DIR / (Path(name).stem + "_checkpoint.json")   # run in progress
        if not ck.exists():
            return None
        print(f"[note] {name} not final - reading its checkpoint")
        return {k: r for k, r in json.loads(ck.read_text(encoding="utf-8"))["rows"].items()}
    d = json.loads(p.read_text(encoding="utf-8"))
    return {r.get("key") or r.get("image"): r for r in d["rows"]}


def grounding_diag(views: dict, base: dict) -> dict:
    """How well the model's own boxes cover the annotated damage."""
    import scripts.stage2_views_experiment as E
    rec, area, zero, nbox = [], [], 0, []
    for key, vr in views.items():
        if "error" in vr or key not in base or "zoom_boxes" not in vr:
            continue
        gt = E.gt_boxes(base[key]["image_path"])
        bx = vr.get("zoom_boxes") or []
        nbox.append(len(bx))
        if not bx:
            zero += 1
        if gt:
            hit = sum(1 for g in gt if any(b[0] <= (g[0] + g[2]) / 2 <= b[2] and b[1] <= (g[1] + g[3]) / 2 <= b[3]
                                           for b in bx))
            rec.append(hit / len(gt))
        w, h = vr["size"]
        area.append(min(1.0, sum((b[2] - b[0]) * (b[3] - b[1]) for b in bx) / (w * h)))
    n = len(nbox)
    return {"frames": n, "no_usable_box": zero, "mean_crops": sum(nbox) / n if n else None,
            "annotated_damage_centres_inside_crops": sum(rec) / len(rec) if rec else None,
            "frame_area_covered": sum(area) / n if n else None}


def times(views: dict) -> dict:
    fams = {}
    for vr in views.values():
        for k, v in (vr.get("time_s") or {}).items():
            fams.setdefault(k, []).append(v)
    return {k: round(st.median(v), 2) for k, v in fams.items()}


def stage_dev():
    base = base_rows()
    dev = load_views("stage2_views_attain_dev.json")
    if dev is None:
        sys.exit("run the dev experiment first")
    out = {"n": len(dev), "variants": {}}
    full_rows = with_scores(base, dev, "full", "+full")
    out["variants"]["full"] = {"macro_auroc": macro_auroc(full_rows),
                               "per_class": per_class_auroc(full_rows, R.ALL_CLASSES)}
    for fam in FAMILIES:
        for agg in AGGS:
            rows = with_scores(base, dev, fam, agg)
            out["variants"][f"{fam}{agg}"] = {"macro_auroc": macro_auroc(rows),
                                              "per_class": per_class_auroc(rows, R.ALL_CLASSES)}
    out["chosen"] = {}
    for fam in FAMILIES:
        best = max(AGGS, key=lambda a: out["variants"][f"{fam}{a}"]["macro_auroc"])
        out["chosen"][fam] = f"{fam}{best}"
    out["grounding"] = grounding_diag(dev, base)
    out["median_time_s"] = times(dev)
    out = R.clean(out)
    (EVAL_DIR / "stage2_views_dev.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    L = ["| variant | macro AUROC | " + " | ".join(c.replace(" and utility cut", "") for c in R.ALL_CLASSES) + " |",
         "|---|---:|" + "---:|" * len(R.ALL_CLASSES)]
    for k, v in out["variants"].items():
        L.append(f"| {k}{' (chosen)' if k in out['chosen'].values() else ''} | {R.fmt(v['macro_auroc'])} | "
                 + " | ".join(R.fmt(v["per_class"][c]) for c in R.ALL_CLASSES) + " |")
    print("\n".join(L))
    print("grounding:", out["grounding"])
    print("median time (s):", out["median_time_s"])
    return 0


def stage_final():
    base = base_rows()
    dev = load_views("stage2_views_attain_dev.json")
    test = load_views("stage2_views_attain_test.json")
    devsum = json.loads((EVAL_DIR / "stage2_views_dev.json").read_text(encoding="utf-8"))
    chosen = devsum["chosen"]
    # One tiling finalist: the better tiling family on DEV (addendum_2).
    tile_best = max((chosen["tile_native"], chosen["tile_up"]),
                    key=lambda v: devsum["variants"][v]["macro_auroc"])
    finalists = ["full", tile_best, chosen["zoom"], chosen["oracle"]]

    def parse(v):
        if v == "full":
            return "full", "+full"
        for fam in FAMILIES:
            if v.startswith(fam):
                return fam, v[len(fam):]
        raise ValueError(v)

    out = {"finalists": finalists, "test": {}, "cv": {}}
    trows = {v: with_scores(base, test, *parse(v)) for v in finalists}
    by_key = {v: {r["image"]: r for r in trows[v]} for v in finalists}
    # Resample image KEYS by block, then score every variant on the same
    # resample (duplicates included) - a properly paired bootstrap.
    key_clusters = {}
    for r in trows["full"]:
        key_clusters.setdefault(r["block"], []).append(r["image"])
    for v in finalists:
        rows = trows[v]
        d = M.paired_cluster_bootstrap(
            lambda ks: macro_auroc([by_key["full"][k] for k in ks]),
            lambda ks, vv=v: macro_auroc([by_key[vv][k] for k in ks]),
            key_clusters, BOOT, 61)
        out["test"][v] = {"macro_auroc": macro_auroc(rows),
                          "per_class": per_class_auroc(rows, R.ALL_CLASSES),
                          "delta_vs_full": d}
    # CV over all 847 frames (dev + test outputs of the same variants)
    labels = {t.key: t.output_label for t in all_probe_types()}
    allviews = {**dev, **test}
    cv_rows = {v: with_scores(base, allviews, *parse(v)) for v in finalists}
    for r in cv_rows["full"]:
        r["block"] = (r["index"] - 1) // R.BLOCK_SIZE
    clall = R.clusters_of(cv_rows["full"])
    oof = {}
    for v in finalists:
        rows = cv_rows[v]
        for r in rows:
            r["block"] = (r["index"] - 1) // R.BLOCK_SIZE
        oof[v] = CV.run_cv(rows, set(CV.ELIGIBLE_CANDIDATES), labels)

    def pred(v):
        return lambda r: oof[v][r["image"]]["pred"]
    for v in finalists:
        pc = R.per_class_table(cv_rows["full"], pred(v), R.HEADLINE + [R.PATCH])
        out["cv"][v] = {
            "macro_mcc": M.macro({c: pc[c] for c in R.HEADLINE}, "mcc"),
            "per_class_mcc": {c: pc[c]["mcc"] for c in R.HEADLINE + [R.PATCH]},
            "delta_vs_full": M.paired_cluster_bootstrap(R.macro_stat(pred("full"), "mcc"),
                                                        R.macro_stat(pred(v), "mcc"), clall, BOOT, 71),
        }
    out["grounding_test"] = grounding_diag(test, base)
    out["median_time_s_test"] = times(test)
    # Bengaluru, descriptive
    ben = load_views("stage2_views_bengaluru.json")
    if ben:
        cfg = rules.load_config(PROJECT_ROOT / "configs" / "stage2_probe.json")
        bd = {"n": sum(1 for r in ben.values() if "error" not in r), "median_time_s": times(ben)}
        for v in finalists:
            if v.startswith("oracle"):
                continue
            fam, agg = parse(v)
            fires, shift = {}, {}
            for g, spec in cfg["groups"].items():
                s = [max(agg_p(r, fam, agg)[k] for k in spec["keys"]) for r in ben.values() if "error" not in r]
                s0 = [max(r["full"][k] for k in spec["keys"]) for r in ben.values() if "error" not in r]
                fires[g] = sum(1 for x in s if x >= spec["threshold"]) / len(s)
                shift[g] = st.median(s) - st.median(s0)
            bd[v] = {"share_over_production_threshold": fires, "median_score_shift_vs_full": shift}
        zr = [r for r in ben.values() if "error" not in r and "zoom_boxes" in r]
        bd["zoom_frames_with_no_usable_box"] = sum(1 for r in zr if not r["zoom_boxes"])
        bd["zoom_mean_crops"] = sum(len(r["zoom_boxes"]) for r in zr) / len(zr) if zr else None
        out["bengaluru"] = bd
    out = R.clean(out)
    (EVAL_DIR / "stage2_views_report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("test", "cv")}, indent=1)[:3000])
    for v in finalists:
        t, c = out["test"][v], out["cv"][v]
        print(f"{v:22s} test AUROC {R.fmt(t['macro_auroc'])}  delta {R.dci(t['delta_vs_full'])} "
              f"| CV macro MCC {R.fmt(c['macro_mcc'])} delta {R.dci(c['delta_vs_full'])} (p {R.fmt(c['delta_vs_full']['p'])})")
    return 0


def emit_config(variant: str) -> int:
    """Production probe config for a views variant, refitted on all 847
    frames exactly like stage2_probe_cv.py does for the full photo:
    eligibility from pooled AUROC lower bounds, MCC thresholds, Platt, and the
    Stage 1 safety-net threshold - all on the variant's aggregated scores."""
    base = base_rows()
    allviews = {**load_views("stage2_views_attain_dev.json"), **load_views("stage2_views_attain_test.json")}
    fam, agg = ("full", "+full") if variant == "full" else next(
        (f, variant[len(f):]) for f in FAMILIES if variant.startswith(f))
    rows = with_scores(base, allviews, fam, agg)
    cl = R.clusters_of(rows)
    auroc = {c: M.cluster_bootstrap(
        lambda rs, cc=c: M.roc_auc([R.probe_score(r, CV.VARIANT, cc) for r in rs], [cc in r["gt"] for r in rs]),
        cl, BOOT, 100 + i) for i, c in enumerate(R.ALL_CLASSES)}
    eligible = {c for c in CV.ELIGIBLE_CANDIDATES if auroc[c]["lo"] > 0.5}
    th, pl = CV.fit(rows, CV.VARIANT, R.ALL_CLASSES)
    cfg = CV.config_for(th, pl, len(rows), eligible)
    sn_keys = sorted({k for c in R.HEADLINE for k in R.PROBE_KEYS[c]})
    sn_scores = [max(r["probe"][CV.VARIANT]["p"][k] for k in sn_keys) for r in rows]
    sn_th = M.best_threshold(sn_scores, [bool(r["gt"]) for r in rows], "mcc")[0]
    cfg["stage1_safety_net"] = {"keys": sn_keys, "threshold": min(max(float(sn_th), 1e-6), 1 - 1e-6)}
    setup = json.loads((EVAL_DIR / "stage2_views_attain_test.json").read_text(encoding="utf-8"))["setup"]
    if fam == "full":
        cfg["views"] = {"mode": "full"}
    elif fam.startswith("tile"):
        cfg["views"] = {"mode": "tile", "aggregate": agg,
                        "upscale_limit": setup["upscale_limit"] if fam == "tile_up" else 1.0,
                        "tile": setup["grid"]}
    else:
        cfg["views"] = {"mode": "zoom", "aggregate": agg, "upscale_limit": setup["upscale_limit"],
                        "zoom": setup["zoom"]}
    cfg["provenance"] = {
        "tuned_on": f"Attain SMP WS_V2.0, all {len(rows)} frames, scores aggregated as {variant}",
        "eligible": sorted(eligible),
        "verify_only": sorted(set(CV.ELIGIBLE_CANDIDATES) - eligible),
        "pooled_auroc": {c: round(a["point"], 4) for c, a in auroc.items()},
        "report": "eval_results/stage2_views_report.json",
        "note": "thresholds tuned on vehicle-mounted Attain frames; not validated on Bengaluru photos",
    }
    labels = {t.key for t in all_probe_types()}
    rules.validate_config(cfg, labels)
    out = EVAL_DIR / f"stage2_probe_config_{variant.replace('+', 'plus')}.json"
    out.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"[config] {out.name}: eligible {sorted(eligible)}; views {cfg['views']}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["dev", "final", "emit"], required=True)
    ap.add_argument("--variant", default=None, help="for --stage emit, e.g. zoomavgmax")
    a = ap.parse_args()
    if a.stage == "emit":
        sys.exit(emit_config(a.variant))
    sys.exit(stage_dev() if a.stage == "dev" else stage_final())
