"""
Secondary analysis: grouped cross-validation of the per-type probe, and the
production configuration it implies.

Why, after the pre-registered single split: per-class AUROC moved a lot
between the 300-image DEV split (15 blocks of road) and TEST (Ravelling 0.78
-> 0.54, Weathering 0.51 -> 0.66). Fifteen blocks are too few to set
per-class thresholds or decide which classes the probe should be trusted
with. Every rule below was written into
eval_results/stage2_probe_preregistration.json (addendum_1) before this
script was run.

  1. 5-fold cross-validation over all 847 images, folds made of whole
     20-frame blocks (seed 20260924). Variant fixed to the pre-registered DEV
     choice. Thresholds (MCC) and Platt calibration are fitted on four folds
     and applied to the fifth, through the same combine() production runs.
     Metrics on the pooled out-of-fold predictions, with cluster-bootstrap
     intervals.
  2. Eligibility: a class is decided by the probe in production only if its
     pooled out-of-fold AUROC has a 95% lower bound above 0.5. Otherwise it
     is verify-only (the probe may remove it, never add it).
  3. The production configuration is refitted on all 847 images.
  4. The gate question: does the probe's own confidence rank the probe
     outputs' errors better than the field confidence does?

Also reported (not a decision): a Stage 1 safety net - how many distress
images that production Stage 1 calls Normal the probe would flag.

Usage:
    venv/Scripts/python.exe scripts/stage2_probe_cv.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import multilabel_metrics as M  # noqa: E402
from scripts import stage2_probe_report as R  # noqa: E402
from scripts.stage2_probe_rules import combine, validate_config  # noqa: E402
from scripts.stage2_probe import all_probe_types  # noqa: E402

EVAL_DIR = PROJECT_ROOT / "eval_results"
CV_SEED = 20260924
K = 5
BOOT = 2000
VARIANT = "min:name"          # pre-registered DEV choice
ELIGIBLE_CANDIDATES = R.HEADLINE + [R.PATCH]


def fit(rows, variant, classes):
    th, pl = {}, {}
    for c in classes:
        y = [c in r["gt"] for r in rows]
        s = [R.probe_score(r, variant, c) for r in rows]
        th[c] = M.best_threshold(s, y, "mcc")[0] if any(y) and not all(y) else 0.5
        pl[c] = M.fit_platt(s, y)
    return th, pl


def config_for(th, pl, n, eligible):
    cfg = R.build_config(VARIANT, th, pl, n)
    # non-eligible classes leave the calibrated groups -> verify-only
    cfg["groups"] = {g: v for g, v in cfg["groups"].items() if g in eligible}
    return cfg


def attain_of(out):
    att = {R.IRC_TO_ATTAIN[t] for t in out["types"] if t in R.IRC_TO_ATTAIN}
    if "Patching" in out["indicators"]:
        att.add(R.PATCH)
    return att


def run_cv(rows, eligible, labels):
    """Out-of-fold predictions for every row under the given eligibility."""
    blocks = sorted({r["block"] for r in rows})
    rng = random.Random(CV_SEED)
    rng.shuffle(blocks)
    fold_of = {b: i % K for i, b in enumerate(blocks)}
    oof = {}
    for k in range(K):
        train = [r for r in rows if fold_of[r["block"]] != k]
        hold = [r for r in rows if fold_of[r["block"]] == k]
        th, pl = fit(train, VARIANT, R.ALL_CLASSES)
        cfg = config_for(th, pl, len(train), eligible)
        validate_config(cfg)
        for r in hold:
            out = combine(r["v0"].get("distress_types") or [], r["probe"][VARIANT]["p"], cfg, labels)
            oof[r["image"]] = {"pred": attain_of(out), "conf": out["confidence"], "fold": k,
                               "types": out["types"]}
    return oof


def main() -> int:
    data = json.loads((EVAL_DIR / "stage2_probe_raw_attain.json").read_text(encoding="utf-8"))
    rows = [r for r in data["rows"] if "error" not in r]
    for r in rows:
        r["block"] = (r["index"] - 1) // R.BLOCK_SIZE
    labels = {t.key: t.output_label for t in all_probe_types()}
    cl = R.clusters_of(rows)
    out = {"variant": VARIANT, "k": K, "seed": CV_SEED, "n": len(rows),
           "n_blocks": len(cl), "rules": "eval_results/stage2_probe_preregistration.json addendum_1"}

    # ---- 1. out-of-fold, all candidate classes probe-decided ----
    oof_all = run_cv(rows, set(ELIGIBLE_CANDIDATES), labels)
    v0 = R.v0_pred

    def probe_all(r):
        return oof_all[r["image"]]["pred"]

    # per-class AUROC of the raw probe score (threshold-free, fold-invariant)
    auroc = {}
    for i, c in enumerate(R.ALL_CLASSES):
        def stat(rs, cc=c):
            return M.roc_auc([R.probe_score(r, VARIANT, cc) for r in rs], [cc in r["gt"] for r in rs])
        auroc[c] = M.cluster_bootstrap(stat, cl, BOOT, 100 + i)
    eligible = {c for c in ELIGIBLE_CANDIDATES if auroc[c]["lo"] > 0.5}
    out["auroc_pooled"] = auroc
    out["eligible_classes"] = sorted(eligible)
    out["not_eligible"] = sorted(set(ELIGIBLE_CANDIDATES) - eligible)
    print(f"[eligibility] probe-decided: {sorted(eligible)}; verify-only: {out['not_eligible']}")

    # ---- 2. out-of-fold with the eligibility rule applied ----
    oof_el = run_cv(rows, eligible, labels)

    def probe_el(r):
        return oof_el[r["image"]]["pred"]

    def summary(fn):
        pc = R.per_class_table(rows, fn, R.ALL_CLASSES)
        head = {c: pc[c] for c in R.HEADLINE}
        return {
            "per_class": pc,
            "macro_mcc": M.macro(head, "mcc"), "macro_f1": M.macro(head, "f1"),
            "macro_balanced_accuracy": M.macro(head, "balanced_accuracy"),
            "sample": M.sample_metrics([set(r["gt"]) for r in rows], [fn(r) for r in rows], R.HEADLINE),
            "ci_macro_mcc": M.cluster_bootstrap(R.macro_stat(fn, "mcc"), cl, BOOT, 7),
            "ci_per_class_mcc": {c: M.cluster_bootstrap(
                (lambda cc: lambda rs: M.binary_metrics([cc in r["gt"] for r in rs],
                                                        [cc in fn(r) for r in rs])["mcc"])(c),
                cl, BOOT, 200 + i) for i, c in enumerate(R.ALL_CLASSES)},
        }

    const = {c for c in R.HEADLINE if sum(c in r["gt"] for r in rows) / len(rows) >= 0.5}
    out["oof"] = {
        "v0_production": summary(v0),
        "probe_all_classes": summary(probe_all),
        "probe_eligible_only": summary(probe_el),
        "constant_prior": summary(lambda r: set(const)),
    }
    out["oof_paired"] = {}
    for name, fn in (("probe_all_classes", probe_all), ("probe_eligible_only", probe_el)):
        out["oof_paired"][name] = {
            "macro_mcc": M.paired_cluster_bootstrap(R.macro_stat(v0, "mcc"), R.macro_stat(fn, "mcc"), cl, BOOT, 11),
            "macro_f1": M.paired_cluster_bootstrap(R.macro_stat(v0, "f1"), R.macro_stat(fn, "f1"), cl, BOOT, 12),
            "mean_jaccard": M.paired_cluster_bootstrap(R.sample_stat(v0, "mean_jaccard"),
                                                       R.sample_stat(fn, "mean_jaccard"), cl, BOOT, 13),
            "per_class_mcc": {c: M.paired_cluster_bootstrap(
                (lambda cc: lambda rs: M.binary_metrics([cc in r["gt"] for r in rs], [cc in v0(r) for r in rs])["mcc"])(c),
                (lambda cc, f=fn: lambda rs: M.binary_metrics([cc in r["gt"] for r in rs], [cc in f(r) for r in rs])["mcc"])(c),
                cl, BOOT, 300 + i) for i, c in enumerate(R.HEADLINE + [R.PATCH])},
        }

    # ---- 3. gate: which confidence ranks the probe outputs' errors? ----
    def events(fn):
        return {
            "exact_headline_set": lambda r: (fn(r) & set(R.HEADLINE)) == (set(r["gt"]) & set(R.HEADLINE)),
            "jaccard_ge_0.5": lambda r: M.sample_metrics([set(r["gt"])], [fn(r)], R.HEADLINE)["mean_jaccard"] >= 0.5,
            "no_false_positive": lambda r: (fn(r) & set(R.HEADLINE)) <= set(r["gt"]),
        }
    gate = {}
    for ename, ev in events(probe_el).items():
        gate[ename] = {}
        for sname, sfn in (("probe_confidence", lambda r: oof_el[r["image"]]["conf"]),
                           ("field_confidence", lambda r: r["v0"].get("stage2_confidence_field") or 0.0)):
            gate[ename][sname] = M.cluster_bootstrap(
                lambda rs, s=sfn, e=ev: M.roc_auc([s(r) for r in rs], [e(r) for r in rs]), cl, 1000, 400)
        gate[ename]["n_right"] = sum(1 for r in rows if ev(r))
    out["gate_for_probe_outputs_oof"] = gate
    # reliability of the probe confidence (exact set)
    ev = events(probe_el)["exact_headline_set"]
    rel = []
    for lo, hi in ((0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)):
        rs = [r for r in rows if lo <= oof_el[r["image"]]["conf"] < hi]
        rel.append({"range": [lo, min(hi, 1)], "n": len(rs),
                    "mean_conf": sum(oof_el[r["image"]]["conf"] for r in rs) / len(rs) if rs else None,
                    "exact_right": sum(1 for r in rs if ev(r)) / len(rs) if rs else None})
    out["probe_confidence_reliability_oof"] = rel
    out["probe_confidence_ece_oof"] = M.expected_calibration_error(
        [oof_el[r["image"]]["conf"] for r in rows], [ev(r) for r in rows])

    # ---- 4. Stage 1 safety net (reported, not deployed by this script) ----
    def s1_normal(r):
        return r["stage1"]["stage1_label"] != "Distress"
    blocks = sorted(cl)
    rng = random.Random(CV_SEED)
    rng.shuffle(blocks)
    fold_of = {b: i % K for i, b in enumerate(blocks)}
    flag = {}
    for k in range(K):
        train = [r for r in rows if fold_of[r["block"]] != k]
        sc = [max(R.probe_score(r, VARIANT, c) for c in R.HEADLINE) for r in train]
        th = M.best_threshold(sc, [bool(r["gt"]) for r in train], "mcc")[0]
        for r in rows:
            if fold_of[r["block"]] == k:
                flag[r["image"]] = max(R.probe_score(r, VARIANT, c) for c in R.HEADLINE) >= th
    s1n = [r for r in rows if s1_normal(r)]
    out["stage1_safety_net_oof"] = {
        "stage1_says_normal": len(s1n),
        "of_which_have_distress": sum(1 for r in s1n if r["gt"]),
        "of_which_no_distress": sum(1 for r in s1n if not r["gt"]),
        "distress_flagged_by_probe": sum(1 for r in s1n if r["gt"] and flag[r["image"]]),
        "no_distress_flagged_by_probe": sum(1 for r in s1n if not r["gt"] and flag[r["image"]]),
        "stage1_recall_distress": sum(1 for r in rows if r["gt"] and not s1_normal(r)) / max(1, sum(1 for r in rows if r["gt"])),
        "note": "flagged = would be sent to expert review instead of auto-classified Normal",
    }

    # ---- 5. production configuration, refitted on all 847 ----
    th, pl = fit(rows, VARIANT, R.ALL_CLASSES)
    cfg = config_for(th, pl, len(rows), eligible)
    # Stage 1 safety net: any headline question's P(yes) at or above this
    # threshold means "there is distress here". Refitted on all images.
    sn_keys = sorted({k for c in R.HEADLINE for k in R.PROBE_KEYS[c]})
    sn_scores = [max(r["probe"][VARIANT]["p"][k] for k in sn_keys) for r in rows]
    sn_th = M.best_threshold(sn_scores, [bool(r["gt"]) for r in rows], "mcc")[0]
    cfg["stage1_safety_net"] = {
        "keys": sn_keys, "threshold": min(max(float(sn_th), 1e-6), 1 - 1e-6),
        "meaning": "Stage 1 said Normal but some headline type has P(yes) >= threshold: "
                   "route to expert review instead of auto-classifying Normal",
        "oof": out["stage1_safety_net_oof"],
    }
    cfg["provenance"] = {
        "tuned_on": f"Attain SMP WS_V2.0, all {len(rows)} images (refit after grouped {K}-fold CV)",
        "variant_chosen_on": "pre-registered DEV split (macro AUROC)",
        "threshold_objective": "MCC per group",
        "calibration": "Platt scaling per group",
        "eligibility": "probe decides a class only if its pooled out-of-fold AUROC 95% lower bound > 0.5",
        "eligible": sorted(eligible), "verify_only": out["not_eligible"],
        "oof_macro_mcc": out["oof"]["probe_eligible_only"]["macro_mcc"],
        "oof_macro_mcc_v0": out["oof"]["v0_production"]["macro_mcc"],
        "report": "eval_results/stage2_probe_cv.json",
        "note": "thresholds are tuned on vehicle-mounted Attain frames; not yet validated "
                "against labelled Bengaluru photos",
    }
    validate_config(cfg, set(labels))
    (EVAL_DIR / "stage2_probe_config_final.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    out["final_config"] = "eval_results/stage2_probe_config_final.json"

    out = R.clean(out)
    (EVAL_DIR / "stage2_probe_cv.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (EVAL_DIR / "stage2_probe_cv.md").write_text(render(out), encoding="utf-8")
    print(render(out))
    return 0


def render(o) -> str:
    f, ci, dci = R.fmt, R.ci, R.dci
    L = [f"# Stage 2 probe - grouped {o['k']}-fold cross-validation (secondary analysis)\n",
         f"{o['n']} Attain WS_V2.0 images in {o['n_blocks']} blocks of 20 frames; variant `{o['variant']}` "
         f"(pre-registered DEV choice); thresholds and calibration fitted out-of-fold. Rules fixed in "
         f"`{o['rules']}` before this ran.\n",
         "## Which classes can the probe be trusted with? (pooled AUROC, 95% cluster CI)\n",
         "| class | AUROC | eligible |", "|---|---:|---|"]
    for c, a in o["auroc_pooled"].items():
        el = "yes" if c in o["eligible_classes"] else ("diagnostic" if c == R.BLOCK else "no - verify-only")
        L.append(f"| {c} | {ci(a)} | {el} |")
    L += ["\n## Out-of-fold, all 847 images\n",
          "| method | macro MCC [95% CI] | macro F1 | macro bal. acc. | mean Jaccard | exact set |",
          "|---|---:|---:|---:|---:|---:|"]
    for m, x in o["oof"].items():
        L.append(f"| {m} | {ci(x['ci_macro_mcc'])} | {f(x['macro_f1'])} | {f(x['macro_balanced_accuracy'])} "
                 f"| {f(x['sample']['mean_jaccard'])} | {f(x['sample']['exact_match'], True)} |")
    L += ["\n| paired vs production | macro MCC delta | macro F1 delta | mean Jaccard delta |", "|---|---:|---:|---:|"]
    for m, x in o["oof_paired"].items():
        L.append(f"| {m} | {dci(x['macro_mcc'])} (p {f(x['macro_mcc']['p'])}) | {dci(x['macro_f1'])} | {dci(x['mean_jaccard'])} |")
    L += ["\n### Per class (out-of-fold)\n",
          "| class | prevalence | v0 P / R | v0 MCC | probe (eligible rule) P / R | probe MCC [95% CI] | delta MCC [95% CI] |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for c in R.HEADLINE + [R.PATCH, R.BLOCK]:
        a = o["oof"]["v0_production"]["per_class"][c]
        b = o["oof"]["probe_eligible_only"]["per_class"][c]
        d = o["oof_paired"]["probe_eligible_only"]["per_class_mcc"].get(c)
        L.append(f"| {c} | {f(a['prevalence'], True)} | {f(a['precision'], d=2)} / {f(a['recall'], d=2)} | {f(a['mcc'])} "
                 f"| {f(b['precision'], d=2)} / {f(b['recall'], d=2)} "
                 f"| {ci(o['oof']['probe_eligible_only']['ci_per_class_mcc'][c])} | {dci(d) if d else '-'} |")
    L += ["\n## Which confidence should gate the probe's outputs? (out-of-fold AUROC for the probe output being right)\n",
          "| correctness event | n right | probe confidence | field confidence |", "|---|---:|---:|---:|"]
    for e, x in o["gate_for_probe_outputs_oof"].items():
        L.append(f"| {e} | {x['n_right']} | {ci(x['probe_confidence'])} | {ci(x['field_confidence'])} |")
    L += ["\nProbe confidence reliability (exact headline set, out-of-fold): "
          + "; ".join(f"{b['range'][0]:.1f}-{b['range'][1]:.1f}: n={b['n']}, mean {f(b['mean_conf'])}, right {f(b['exact_right'], True)}"
                      for b in o["probe_confidence_reliability_oof"])
          + f". ECE {f(o['probe_confidence_ece_oof'])}.\n"]
    s = o["stage1_safety_net_oof"]
    L += ["## Stage 1 safety net (reported only)\n",
          f"Production Stage 1 recall on distress images: {f(s['stage1_recall_distress'], True)}. Of the "
          f"{s['stage1_says_normal']} images Stage 1 calls Normal, {s['of_which_have_distress']} carry annotated distress; "
          f"the probe (out-of-fold threshold) would flag {s['distress_flagged_by_probe']} of them for review, at the cost of "
          f"flagging {s['no_distress_flagged_by_probe']} of the {s['of_which_no_distress']} without distress.\n"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
