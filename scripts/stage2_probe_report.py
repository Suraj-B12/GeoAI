"""
Score the Stage 2 sandbox experiment (stage2_probe_experiment.py output).

Protocol - fixed before looking at test numbers
-----------------------------------------------
1. Split. Images are grouped into blocks of 20 consecutive frames (neighbours
   are near-duplicates from a moving vehicle), blocks are shuffled with a
   fixed seed, and 35% of blocks become DEV, the rest TEST. Nothing about the
   test images is used to make any choice.
2. On DEV only: pick the probe variant (highest macro AUROC over the five
   headline classes), tune one decision threshold per class (maximising MCC),
   and fit a Platt calibration per class.
3. On TEST, once: score production v0 and the chosen probe with those frozen
   choices. Every other variant is also reported on TEST as an ablation, with
   the DEV-chosen thresholds, and labelled as such.

Classes
-------
Headline (both methods can express them): Linear crack, Alligator crack,
Pothole, Raveling, Weathering. Two more are reported separately:
  Patch and utility cut - the probe can express it through the new Patching
    condition indicator; production v0 cannot (no such label), so it would be
    unfair to put it in the headline average.
  Block crack - IRC:82 §7.3.5 files block cracking under Transverse Cracking,
    so neither method can report it as a separate IRC label; the probe's
    "Block Cracking" sub-pattern question is scored as a diagnostic only.

Usage:
    venv/Scripts/python.exe scripts/stage2_probe_report.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import multilabel_metrics as M  # noqa: E402
from scripts.stage2_probe_rules import SCHEMA_VERSION, combine, validate_config  # noqa: E402

EVAL_DIR = PROJECT_ROOT / "eval_results"
PLOT_DIR = EVAL_DIR / "plots" / "probe"

HEADLINE = ["Linear crack", "Alligator crack", "Pothole", "Raveling", "Weathering"]
PATCH = "Patch and utility cut"
BLOCK = "Block crack"
ALL_CLASSES = HEADLINE + [PATCH, BLOCK]

# Production output (canonical IRC:82 names) -> Attain class. The same table
# 07_cross_dataset_eval.PIPELINE_TO_ATTAIN uses for IRC-era names.
IRC_TO_ATTAIN = {
    "Longitudinal Cracking": "Linear crack",
    "Transverse Cracking": "Linear crack",
    "Alligator Cracking": "Alligator crack",
    "Potholes": "Pothole",
    "Ravelling": "Raveling",
    "Hungry Surface": "Weathering",
    "Patching": PATCH,
}

# Probe question(s) whose P(yes) scores each Attain class. Linear crack is
# the larger of the two orientation questions: Attain does not split them.
PROBE_KEYS = {
    "Linear crack": ["Longitudinal Cracking", "Transverse Cracking"],
    "Alligator crack": ["Alligator Cracking"],
    "Pothole": ["Potholes"],
    "Raveling": ["Ravelling"],
    "Weathering": ["Hungry Surface"],
    PATCH: ["Patching"],
    BLOCK: ["Block Cracking"],
}

BLOCK_SIZE = 20
DEV_FRAC = 0.35
SPLIT_SEED = 20260923
BOOT = 2000


# ============================================================
# Helpers
# ============================================================

def v0_pred(row) -> set:
    return {IRC_TO_ATTAIN[t] for t in (row["v0"].get("distress_types") or [])
            if t in IRC_TO_ATTAIN}


def probe_score(row, variant: str, cls: str) -> float:
    p = row["probe"][variant]["p"]
    return max(p[k] for k in PROBE_KEYS[cls])


def split(rows):
    blocks = sorted({(r["index"] - 1) // BLOCK_SIZE for r in rows})
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(blocks)
    n_dev = round(DEV_FRAC * len(blocks))
    dev_blocks = set(blocks[:n_dev])
    for r in rows:
        r["block"] = (r["index"] - 1) // BLOCK_SIZE
        r["split"] = "dev" if r["block"] in dev_blocks else "test"
    return ([r for r in rows if r["split"] == "dev"],
            [r for r in rows if r["split"] == "test"])


def clusters_of(rows):
    c = defaultdict(list)
    for r in rows:
        c[r["block"]].append(r)
    return dict(c)


def per_class_table(rows, pred_fn, classes, score_fn=None) -> dict:
    out = {}
    for cls in classes:
        y = [cls in r["gt"] for r in rows]
        pr = [cls in pred_fn(r) for r in rows]
        m = M.binary_metrics(y, pr)
        if score_fn is not None:
            s = [score_fn(r, cls) for r in rows]
            m["auroc"] = M.roc_auc(s, y)
            m["ap"] = M.average_precision(s, y)
        out[cls] = m
    return out


def macro_stat(pred_fn, key, classes=HEADLINE):
    def f(rows):
        return M.macro(per_class_table(rows, pred_fn, classes), key)
    return f


def sample_stat(pred_fn, key, classes=HEADLINE):
    def f(rows):
        return M.sample_metrics([set(r["gt"]) for r in rows],
                                [pred_fn(r) for r in rows], classes)[key]
    return f


def r4(x):
    return None if x is None or x != x else round(x, 4)


def clean(d):
    if isinstance(d, dict):
        return {k: clean(v) for k, v in d.items()}
    if isinstance(d, list):
        return [clean(v) for v in d]
    if isinstance(d, float):
        return r4(d)
    return d


def build_config(variant: str, th: dict, platt: dict, n_dev: int) -> dict:
    """The production probe configuration implied by DEV-tuned choices."""
    sys_style, q_style = variant.split(":")

    def clamp(t):
        return min(max(float(t), 1e-6), 1 - 1e-6)

    groups = {c: {"kind": "distress", "keys": PROBE_KEYS[c], "threshold": clamp(th[c]),
                  "platt": list(platt[c]) if platt.get(c) else None}
              for c in HEADLINE}
    groups[PATCH] = {"kind": "indicator", "keys": PROBE_KEYS[PATCH], "label": "Patching",
                     "threshold": clamp(th[PATCH]),
                     "platt": list(platt[PATCH]) if platt.get(PATCH) else None}
    return {
        "schema_version": SCHEMA_VERSION,
        "variant": {"system_style": sys_style, "question_style": q_style},
        "groups": groups,
        "verify_threshold": 0.5,
        "provenance": {
            "tuned_on": f"Attain SMP WS_V2.0 DEV split ({n_dev} images, blocks of "
                        f"{BLOCK_SIZE}, seed {SPLIT_SEED}, {int(DEV_FRAC * 100)}% of blocks)",
            "threshold_objective": "MCC per group",
            "calibration": "Platt scaling per group on DEV",
            "group_names_are": "Attain classes the thresholds were tuned against",
            "note": "thresholds are tuned on vehicle-mounted Attain frames; they "
                    "have not been validated on handheld Bengaluru photos",
        },
    }


# ============================================================
# Main
# ============================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="stage2_probe_raw_attain.json")
    ap.add_argument("--out", default="stage2_probe_report")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    raw_path = EVAL_DIR / args.raw
    if not raw_path.exists():
        # allow scoring a run in progress from its checkpoint
        ck = EVAL_DIR / (Path(args.raw).stem + "_checkpoint.json")
        blob = json.loads(ck.read_text(encoding="utf-8"))
        data = {"config": blob["_config"], "rows": list(blob["rows"].values()),
                "partial": True}
    else:
        data = json.loads(raw_path.read_text(encoding="utf-8"))
    rows = [r for r in data["rows"] if "error" not in r and "v0" in r and r.get("probe")]
    n_err = sum(1 for r in data["rows"] if "error" in r)
    variants = list(rows[0]["probe"].keys())
    dev, test = split(rows)
    print(f"[data] {len(rows)} usable rows ({n_err} errors); dev {len(dev)} / test {len(test)}; "
          f"variants {variants}")

    report = {
        "source": args.raw,
        "partial_run": bool(data.get("partial")),
        "config": data.get("config"),
        "protocol": {
            "block_size": BLOCK_SIZE, "dev_fraction_of_blocks": DEV_FRAC,
            "split_seed": SPLIT_SEED, "bootstrap": f"cluster (block) bootstrap, B={BOOT}",
            "threshold_objective": "MCC, tuned on dev per class",
            "variant_selection": "highest dev macro AUROC over headline classes",
            "headline_classes": HEADLINE,
        },
        "n": {"usable": len(rows), "errors": n_err, "dev": len(dev), "test": len(test),
              "dev_no_distress": sum(1 for r in dev if not r["gt"]),
              "test_no_distress": sum(1 for r in test if not r["gt"])},
        "prevalence": {s: {c: r4(sum(c in r["gt"] for r in rs) / len(rs)) for c in ALL_CLASSES}
                       for s, rs in (("dev", dev), ("test", test))},
    }

    # ---------------- DEV: variant choice, thresholds, calibration ----------------
    dev_summary = {}
    thresholds = {}
    platt = {}
    for v in variants:
        sfn = (lambda vv: (lambda r, c: probe_score(r, vv, c)))(v)
        th = {}
        for cls in ALL_CLASSES:
            y = [cls in r["gt"] for r in dev]
            s = [sfn(r, cls) for r in dev]
            th[cls] = M.best_threshold(s, y, "mcc")[0] if any(y) and not all(y) else 0.5
        thresholds[v] = th
        platt[v] = {cls: M.fit_platt([sfn(r, cls) for r in dev], [cls in r["gt"] for r in dev])
                    for cls in ALL_CLASSES}
        pc = per_class_table(dev, lambda r, vv=v: {c for c in ALL_CLASSES
                                                  if probe_score(r, vv, c) >= thresholds[vv][c]},
                             ALL_CLASSES, sfn)
        dev_summary[v] = {
            "macro_auroc_headline": M.macro({c: pc[c] for c in HEADLINE}, "auroc"),
            "macro_mcc_headline_in_sample": M.macro({c: pc[c] for c in HEADLINE}, "mcc"),
            "per_class_auroc": {c: pc[c]["auroc"] for c in ALL_CLASSES},
        }
    chosen = max(variants, key=lambda v: dev_summary[v]["macro_auroc_headline"])
    report["dev"] = {"variants": dev_summary, "chosen_variant": chosen,
                     "thresholds": thresholds, "platt": platt}
    print(f"[dev] chosen variant: {chosen}  "
          + "  ".join(f"{v}: AUROC {dev_summary[v]['macro_auroc_headline']:.3f}" for v in variants))

    TH = thresholds[chosen]
    cfgs = {v: build_config(v, thresholds[v], platt[v], len(dev)) for v in variants}
    for v in variants:
        validate_config(cfgs[v])

    # Every probe decision goes through the SAME combine() production runs.
    # Cached per (variant, image): the bootstrap calls these thousands of times.
    _memo: dict = {}

    def v1_out(r, v=chosen):
        key = (v, r["image"])
        if key not in _memo:
            out = combine(r["v0"].get("distress_types") or [], r["probe"][v]["p"], cfgs[v])
            att = {IRC_TO_ATTAIN[t] for t in out["types"] if t in IRC_TO_ATTAIN}
            if "Patching" in out["indicators"]:
                att.add(PATCH)
            # Block crack is a diagnostic of the sub-pattern question only; it
            # does not reach an IRC label (it would be Transverse Cracking).
            if probe_score(r, v, BLOCK) >= thresholds[v][BLOCK]:
                att.add(BLOCK)
            _memo[key] = (att, out)
        return _memo[key]

    def v1_pred(r, v=chosen, th=None):
        return v1_out(r, v)[0]

    # Constant predictor: every class more common than not on DEV.
    const_set = {c for c in HEADLINE if report["prevalence"]["dev"][c] >= 0.5}
    report["constant_baseline_classes"] = sorted(const_set)

    # ---------------- TEST ----------------
    cl = clusters_of(test)
    methods = {
        "v0_production": (v0_pred, None),
        f"probe_{chosen}": (v1_pred, lambda r, c: probe_score(r, chosen, c)),
        "constant_prior": (lambda r: set(const_set), None),
    }
    test_res = {}
    for name, (fn, sfn) in methods.items():
        pc = per_class_table(test, fn, ALL_CLASSES, sfn)
        head = {c: pc[c] for c in HEADLINE}
        test_res[name] = {
            "per_class": pc,
            "macro_headline": {k: M.macro(head, k) for k in
                               ("f1", "mcc", "balanced_accuracy", "precision", "recall")},
            "micro_headline": M.binary_metrics(
                [c in r["gt"] for r in test for c in HEADLINE],
                [c in fn(r) for r in test for c in HEADLINE]),
            "sample_headline": M.sample_metrics([set(r["gt"]) for r in test],
                                                [fn(r) for r in test], HEADLINE),
            "ci": {
                "macro_mcc": M.cluster_bootstrap(macro_stat(fn, "mcc"), cl, BOOT, 1),
                "macro_f1": M.cluster_bootstrap(macro_stat(fn, "f1"), cl, BOOT, 2),
                "macro_ba": M.cluster_bootstrap(macro_stat(fn, "balanced_accuracy"), cl, BOOT, 3),
                "mean_jaccard": M.cluster_bootstrap(sample_stat(fn, "mean_jaccard"), cl, BOOT, 4),
                "per_class_mcc": {c: M.cluster_bootstrap(
                    (lambda cc: lambda rs: M.binary_metrics(
                        [cc in r["gt"] for r in rs], [cc in fn(r) for r in rs])["mcc"])(c),
                    cl, BOOT, 10 + i) for i, c in enumerate(ALL_CLASSES)},
            },
        }
        if sfn is not None:
            test_res[name]["macro_headline"]["auroc"] = M.macro(head, "auroc")
            test_res[name]["ci"]["macro_auroc"] = M.cluster_bootstrap(
                lambda rs: M.macro(per_class_table(rs, fn, HEADLINE, sfn), "auroc"), cl, BOOT, 5)
    report["test"] = test_res

    # Paired comparison probe vs production on the same test images
    a, b = v0_pred, v1_pred
    paired = {
        "macro_mcc": M.paired_cluster_bootstrap(macro_stat(a, "mcc"), macro_stat(b, "mcc"), cl, BOOT, 21),
        "macro_f1": M.paired_cluster_bootstrap(macro_stat(a, "f1"), macro_stat(b, "f1"), cl, BOOT, 22),
        "macro_balanced_accuracy": M.paired_cluster_bootstrap(
            macro_stat(a, "balanced_accuracy"), macro_stat(b, "balanced_accuracy"), cl, BOOT, 23),
        "mean_jaccard": M.paired_cluster_bootstrap(
            sample_stat(a, "mean_jaccard"), sample_stat(b, "mean_jaccard"), cl, BOOT, 24),
        "hamming_loss": M.paired_cluster_bootstrap(
            sample_stat(a, "hamming_loss"), sample_stat(b, "hamming_loss"), cl, BOOT, 25),
        "per_class_mcc": {c: M.paired_cluster_bootstrap(
            (lambda cc: lambda rs: M.binary_metrics([cc in r["gt"] for r in rs],
                                                    [cc in a(r) for r in rs])["mcc"])(c),
            (lambda cc: lambda rs: M.binary_metrics([cc in r["gt"] for r in rs],
                                                    [cc in b(r) for r in rs])["mcc"])(c),
            cl, BOOT, 30 + i) for i, c in enumerate(HEADLINE)},
    }
    # McNemar on per-image exact match over the headline classes
    hb = hc = 0
    for r in test:
        g = set(r["gt"]) & set(HEADLINE)
        ea = (a(r) & set(HEADLINE)) == g
        eb = (b(r) & set(HEADLINE)) == g
        hb += ea and not eb
        hc += eb and not ea
    paired["exact_match_mcnemar"] = {"v0_right_probe_wrong": hb, "probe_right_v0_wrong": hc,
                                     "p": M.mcnemar_exact(hb, hc)}
    report["test_paired_probe_minus_v0"] = paired

    # Ablation: every variant on TEST with its own DEV thresholds
    report["test_ablation_variants"] = {}
    for v in variants:
        fn = (lambda vv: (lambda r: v1_pred(r, vv)))(v)
        sfn = (lambda vv: (lambda r, c: probe_score(r, vv, c)))(v)
        pc = per_class_table(test, fn, HEADLINE + [PATCH, BLOCK], sfn)
        report["test_ablation_variants"][v] = {
            "macro_auroc": M.macro({c: pc[c] for c in HEADLINE}, "auroc"),
            "macro_mcc": M.macro({c: pc[c] for c in HEADLINE}, "mcc"),
            "macro_f1": M.macro({c: pc[c] for c in HEADLINE}, "f1"),
            "per_class_auroc": {c: pc[c]["auroc"] for c in HEADLINE + [PATCH, BLOCK]},
        }

    # Continuity with 07_cross_dataset_eval "tiers" (instance recall)
    def tier(rows, fn, classes):
        tot = hit = 0
        for r in rows:
            for c in classes:
                if c in r["gt"]:
                    tot += 1
                    hit += c in fn(r)
        return {"instances": tot, "recalled": hit, "recall": hit / tot if tot else float("nan")}
    report["test_tiers"] = {
        n: {"tier1_in_distribution": tier(test, fn, ["Linear crack", "Alligator crack", "Pothole"]),
            "tier2_zero_shot_expressible_by_both": tier(test, fn, ["Raveling", "Weathering"]),
            "tier2_zero_shot_all": tier(test, fn, ["Raveling", "Weathering", PATCH, BLOCK])}
        for n, (fn, _s) in methods.items() if n != "constant_prior"}

    # Output-shape diagnostics
    def labelsets(fn):
        from collections import Counter
        return Counter(" + ".join(sorted(fn(r) & set(HEADLINE))) or "(none)"
                       for r in test).most_common(8)
    report["test_output_shape"] = {
        "v0_most_common_label_sets": labelsets(a),
        "probe_most_common_label_sets": labelsets(b),
        "v0_raw_irc_lists_top": __import__("collections").Counter(
            ", ".join(r["v0"]["distress_types"] or []) for r in test).most_common(5),
        "v0_distinct_label_sets": len({frozenset(a(r)) for r in test}),
        "probe_distinct_label_sets": len({frozenset(b(r)) for r in test}),
    }

    # ---------------- Stage 1 and end-to-end ----------------
    def s1_distress(r):
        return r["stage1"]["stage1_label"] == "Distress"

    def s1_score(r):
        c = r["stage1"]["stage1_confidence"] or 0.0
        return c if s1_distress(r) else 1 - c

    def probe_any(r, v=chosen):
        return max(probe_score(r, v, c) for c in HEADLINE)

    y_dev = [bool(r["gt"]) for r in dev]
    th_any = M.best_threshold([probe_any(r) for r in dev], y_dev, "mcc")[0]
    y = [bool(r["gt"]) for r in test]
    report["stage1_test"] = {
        "n_distress": sum(y), "n_no_distress": len(y) - sum(y),
        "production_stage1": {**M.binary_metrics(y, [s1_distress(r) for r in test]),
                              "auroc": M.roc_auc([s1_score(r) for r in test], y)},
        "probe_max_score": {**M.binary_metrics(y, [probe_any(r) >= th_any for r in test]),
                            "auroc": M.roc_auc([probe_any(r) for r in test], y),
                            "threshold_from_dev": th_any},
        "note": "only the no-distress images are negatives; there are few of them, so "
                "specificity and AUROC here have wide uncertainty",
    }
    e2e = {}
    for n, fn in (("v0_production", a), (f"probe_{chosen}", b)):
        gated = (lambda f: (lambda r: f(r) if s1_distress(r) else set()))(fn)
        pc = per_class_table(test, gated, HEADLINE)
        e2e[n] = {"macro_mcc": M.macro(pc, "mcc"), "macro_f1": M.macro(pc, "f1"),
                  "per_class_recall": {c: pc[c]["recall"] for c in HEADLINE},
                  "ci_macro_mcc": M.cluster_bootstrap(macro_stat(gated, "mcc"), cl, BOOT, 41)}
    e2e["stage1_passes_test"] = sum(1 for r in test if s1_distress(r))
    report["end_to_end_test_with_production_stage1"] = e2e

    # ---------------- Confidence gate ----------------
    P = platt[chosen]

    def q(r, c):
        return M.apply_platt(probe_score(r, chosen, c), P[c])

    def conf_probe(r):
        return v1_out(r)[1]["confidence"]

    def conf_v0(r):
        return r["v0"].get("stage2_confidence_field") or 0.0

    events = {
        "exact_headline_set": lambda fn: (lambda r: (fn(r) & set(HEADLINE)) == (set(r["gt"]) & set(HEADLINE))),
        "no_false_positive": lambda fn: (lambda r: (fn(r) & set(HEADLINE)) <= set(r["gt"])),
        "jaccard_ge_0.5": lambda fn: (lambda r: M.sample_metrics([set(r["gt"])], [fn(r)], HEADLINE)["mean_jaccard"] >= 0.5),
    }
    gate = {}
    for mname, fn, cf in (("v0_production(field)", a, conf_v0), (f"probe_{chosen}", b, conf_probe)):
        g = {}
        for ename, ev in events.items():
            ok = [ev(fn)(r) for r in test]
            sc = [cf(r) for r in test]
            row = {"n_correct": sum(ok), "n_wrong": len(ok) - sum(ok),
                   "auroc": M.roc_auc(sc, ok),
                   "ci": M.cluster_bootstrap(lambda rs, e=ev, f=fn, c=cf: M.roc_auc(
                       [c(r) for r in rs], [e(f)(r) for r in rs]), cl, 1000, 51)}
            # error rate among auto-accepted at fixed review loads
            order = sorted(zip(sc, ok), key=lambda t: -t[0])
            ops = {}
            for load in (0.0, 0.3, 0.5, 0.7):
                k = round(len(order) * (1 - load))
                acc = order[:k]
                ops[f"review_{int(load * 100)}pct"] = {
                    "auto_accepted": k,
                    "error_rate_auto_accepted": (sum(1 for _s, o in acc if not o) / k) if k else None}
            row["operating_points"] = ops
            g[ename] = row
        gate[mname] = g
    # calibration quality of the per-class probabilities on TEST
    calib = {}
    for c in HEADLINE:
        yy = [c in r["gt"] for r in test]
        raw = [probe_score(r, chosen, c) for r in test]
        cal = [q(r, c) for r in test]
        calib[c] = {"ece_raw": M.expected_calibration_error(raw, yy),
                    "ece_platt_dev": M.expected_calibration_error(cal, yy)}
    # Added after the pre-registration record (a diagnostic, not a decision
    # criterion): what the LIVE threshold does, and whether the probe's image
    # confidence means what it says (of images scored >= t, >= t are right).
    from scripts.utils import CONFIDENCE_THRESHOLD as LIVE
    live = {}
    for mname, fn, cf in (("v0_production(field)", a, conf_v0), (f"probe_{chosen}", b, conf_probe)):
        ev = events["exact_headline_set"](fn)
        acc = [r for r in test if cf(r) >= LIVE]
        bins = []
        for lo, hi in ((0.0, 0.5), (0.5, 0.8), (0.8, 0.9), (0.9, 1.01)):
            rs = [r for r in test if lo <= cf(r) < hi]
            bins.append({"range": [lo, min(hi, 1.0)], "n": len(rs),
                         "mean_confidence": (sum(cf(r) for r in rs) / len(rs)) if rs else None,
                         "exact_set_correct": (sum(1 for r in rs if ev(r)) / len(rs)) if rs else None})
        live[mname] = {
            "threshold": LIVE,
            "review_load": 1 - len(acc) / len(test),
            "auto_accepted": len(acc),
            "exact_set_correct_among_accepted": (sum(1 for r in acc if ev(r)) / len(acc)) if acc else None,
            "no_false_positive_among_accepted": (sum(1 for r in acc if events["no_false_positive"](fn)(r))
                                                 / len(acc)) if acc else None,
            "reliability_bins_exact_set": bins,
        }
    report["confidence_at_live_threshold_test"] = live
    report["confidence_gate_test"] = {"gate": gate, "per_class_calibration": calib,
                                      "probe_confidence_definition":
                                      "product over headline classes of the Platt-calibrated "
                                      "probability of the decision taken"}

    # ---------------- Cost ----------------
    import statistics as st
    report["cost"] = {
        "v0_stage2_ms_median": st.median(r["v0"]["stage2_time_ms"] for r in rows),
        "probe_ms_median": {v: st.median(r["probe"][v]["time_ms"] for r in rows) for v in variants},
        "stage1_ms_median": st.median(r["stage1"]["stage1_time_ms"] for r in rows),
        "peak_reserved_gb_max": max((r.get("peak_reserved_gb") or 0) for r in rows),
        "min_yes_no_mass": {v: min(r["probe"][v]["min_yes_no_mass"] for r in rows) for v in variants},
    }

    report["uncalibrated_types_on_test"] = {
        "v0_named": sum(len([t for t in (r["v0"].get("distress_types") or [])
                             if t not in IRC_TO_ATTAIN and t != "Normal"]) for r in test),
        "kept_after_probe_verification": sum(len(v1_out(r)[1]["verified_uncalibrated"]) for r in test),
        "removed_by_probe_verification": sum(
            len([t for t in v1_out(r)[1]["removed_by_probe"] if t not in IRC_TO_ATTAIN]) for r in test),
        "note": "types Attain does not annotate (bleeding, rutting, edge breaking, ...) cannot "
                "be scored here; the rule only ever removes them",
    }
    cand = dict(cfgs[chosen])
    cand["provenance"] = dict(cand["provenance"], chosen_variant=chosen,
                              dev_macro_auroc=dev_summary[chosen]["macro_auroc_headline"],
                              report=f"eval_results/{args.out}.json")
    (EVAL_DIR / "stage2_probe_config_candidate.json").write_text(
        json.dumps(cand, indent=2), encoding="utf-8")
    report = clean(report)
    out_json = EVAL_DIR / f"{args.out}.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (EVAL_DIR / f"{args.out}.md").write_text(render_md(report, chosen), encoding="utf-8")
    if not args.no_plots:
        make_plots(report, test, chosen, TH)
    print(f"[written] {out_json}  and  {args.out}.md")
    print(render_md(report, chosen)[:6000])
    return 0


# ============================================================
# Markdown
# ============================================================

def fmt(x, pct=False, d=3):
    if x is None or x != x:
        return "-"
    return f"{100 * x:.1f}%" if pct else f"{x:.{d}f}"


def num(x, d=3, sign=False):
    """None/NaN-safe number formatting (NaNs are stored as null)."""
    if x is None or x != x:
        return "n/a"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def ci(c, d=3):
    return f"{num(c.get('point'), d)} [{num(c.get('lo'), d)}, {num(c.get('hi'), d)}]"


def dci(c, d=3):
    return f"{num(c.get('delta'), d, True)} [{num(c.get('lo'), d, True)}, {num(c.get('hi'), d, True)}]"


def render_md(R, chosen) -> str:
    v0, v1 = "v0_production", f"probe_{chosen}"
    T = R["test"]
    L = []
    L.append("# Stage 2: free-form list (production) vs per-type probing\n")
    L.append(f"Source `{R['source']}`{' (PARTIAL RUN)' if R['partial_run'] else ''}. "
             f"{R['n']['usable']} Attain WS_V2.0 images; DEV {R['n']['dev']} "
             f"({R['n']['dev_no_distress']} without distress), TEST {R['n']['test']} "
             f"({R['n']['test_no_distress']} without distress). Split by blocks of "
             f"{R['protocol']['block_size']} consecutive frames, seed {R['protocol']['split_seed']}. "
             f"Variant, thresholds and calibration chosen on DEV only; chosen variant: `{chosen}`. "
             f"Intervals: {R['protocol']['bootstrap']}.\n")
    L.append("## Headline (TEST, five classes both methods can express)\n")
    L.append("| metric | production v0 | probe | constant prior | delta probe - v0 [95% CI] | p |")
    L.append("|---|---:|---:|---:|---:|---:|")
    P = R["test_paired_probe_minus_v0"]
    for key, lab, pk in (("mcc", "macro MCC", "macro_mcc"), ("f1", "macro F1", "macro_f1"),
                         ("balanced_accuracy", "macro balanced acc.", "macro_balanced_accuracy")):
        d = P[pk]
        L.append(f"| {lab} | {fmt(T[v0]['macro_headline'][key])} | {fmt(T[v1]['macro_headline'][key])} "
                 f"| {fmt(T['constant_prior']['macro_headline'][key])} "
                 f"| {dci(d)} | {fmt(d['p'])} |")
    for key, lab in (("mean_jaccard", "mean Jaccard (per image)"), ("hamming_loss", "Hamming loss (lower better)")):
        d = P[key]
        L.append(f"| {lab} | {fmt(T[v0]['sample_headline'][key])} | {fmt(T[v1]['sample_headline'][key])} "
                 f"| {fmt(T['constant_prior']['sample_headline'][key])} "
                 f"| {dci(d)} | {fmt(d['p'])} |")
    L.append(f"| exact label set | {fmt(T[v0]['sample_headline']['exact_match'], True)} "
             f"| {fmt(T[v1]['sample_headline']['exact_match'], True)} "
             f"| {fmt(T['constant_prior']['sample_headline']['exact_match'], True)} "
             f"| McNemar {P['exact_match_mcnemar']['probe_right_v0_wrong']} vs "
             f"{P['exact_match_mcnemar']['v0_right_probe_wrong']} | {fmt(P['exact_match_mcnemar']['p'])} |")
    L.append(f"\nProbe macro AUROC (threshold-free): {ci(T[v1]['ci']['macro_auroc'])}. "
             f"Constant prior predicts {R['constant_baseline_classes']} on every image.\n")

    L.append("## Per class (TEST)\n")
    L.append("| class | prevalence | v0 P / R | v0 MCC | probe P / R | probe MCC [95% CI] | probe AUROC | delta MCC [95% CI] |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for c in HEADLINE + [PATCH, BLOCK]:
        a, b = T[v0]["per_class"][c], T[v1]["per_class"][c]
        cib = T[v1]["ci"]["per_class_mcc"][c]
        dd = P["per_class_mcc"].get(c)
        dtxt = dci(dd) if dd else "(not in headline)"
        L.append(f"| {c} | {fmt(a['prevalence'], True)} | {fmt(a['precision'], d=2)} / {fmt(a['recall'], d=2)} "
                 f"| {fmt(a['mcc'])} | {fmt(b['precision'], d=2)} / {fmt(b['recall'], d=2)} "
                 f"| {ci(cib)} | {fmt(b.get('auroc'))} | {dtxt} |")
    L.append("\n## Variant ablation (TEST, each with its own DEV thresholds; selection was on DEV)\n")
    L.append("| variant | dev macro AUROC | test macro AUROC | test macro MCC | test macro F1 | Patch AUROC | Block AUROC |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for v, x in R["test_ablation_variants"].items():
        L.append(f"| {v}{' (chosen)' if v == chosen else ''} | {fmt(R['dev']['variants'][v]['macro_auroc_headline'])} "
                 f"| {fmt(x['macro_auroc'])} | {fmt(x['macro_mcc'])} | {fmt(x['macro_f1'])} "
                 f"| {fmt(x['per_class_auroc'][PATCH])} | {fmt(x['per_class_auroc'][BLOCK])} |")
    L.append("\n## Output shape (TEST)\n")
    s = R["test_output_shape"]
    L.append(f"Distinct label sets: v0 {s['v0_distinct_label_sets']}, probe {s['probe_distinct_label_sets']}.\n")
    L.append("v0 raw lists: " + "; ".join(f"`{k}` x{n}" for k, n in s["v0_raw_irc_lists_top"]) + "\n")
    L.append("\n## Stage 1 and end to end (TEST)\n")
    s1 = R["stage1_test"]
    L.append(f"{s1['n_distress']} distress / {s1['n_no_distress']} no-distress images. {s1['note']}.\n")
    L.append("| detector | recall | specificity | MCC | AUROC |")
    L.append("|---|---:|---:|---:|---:|")
    for k in ("production_stage1", "probe_max_score"):
        x = s1[k]
        L.append(f"| {k} | {fmt(x['recall'], True)} | {fmt(x['specificity'], True)} | {fmt(x['mcc'])} | {fmt(x['auroc'])} |")
    e = R["end_to_end_test_with_production_stage1"]
    L.append(f"\nWith the production Stage 1 gate in front ({e['stage1_passes_test']} of {R['n']['test']} pass): "
             f"macro MCC v0 {ci(e[v0]['ci_macro_mcc'])}, probe {ci(e[v1]['ci_macro_mcc'])}.\n")
    L.append("\n## Confidence gate (TEST)\n")
    L.append("Each method's confidence judged on its OWN errors. Error rate = share of wrong outputs among the auto-accepted.\n")
    L.append("| method | correctness event | right / wrong | AUROC [95% CI] | err @0% review | @30% | @50% | @70% |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for m, g in R["confidence_gate_test"]["gate"].items():
        for ev, x in g.items():
            op = x["operating_points"]
            L.append(f"| {m} | {ev} | {x['n_correct']} / {x['n_wrong']} | {ci(x['ci'])} | "
                     + " | ".join(fmt(op[k]['error_rate_auto_accepted'], True) for k in
                                  ("review_0pct", "review_30pct", "review_50pct", "review_70pct")) + " |")
    L.append("\n## At the live threshold (TEST; diagnostic added after pre-registration)\n")
    L.append("| method | review load | auto-accepted | exact set right among accepted | no false positive among accepted |")
    L.append("|---|---:|---:|---:|---:|")
    for m, x in R["confidence_at_live_threshold_test"].items():
        L.append(f"| {m} @ {x['threshold']} | {fmt(x['review_load'], True)} | {x['auto_accepted']} "
                 f"| {fmt(x['exact_set_correct_among_accepted'], True)} | {fmt(x['no_false_positive_among_accepted'], True)} |")
    L.append("\nReliability (exact headline set):\n")
    L.append("| method | confidence bin | n | mean confidence | exact set right |")
    L.append("|---|---|---:|---:|---:|")
    for m, x in R["confidence_at_live_threshold_test"].items():
        for bn in x["reliability_bins_exact_set"]:
            L.append(f"| {m} | {bn['range'][0]:.1f}-{bn['range'][1]:.1f} | {bn['n']} | {fmt(bn['mean_confidence'])} "
                     f"| {fmt(bn['exact_set_correct'], True)} |")
    L.append("\n## Cost\n")
    c = R["cost"]
    L.append(f"Median Stage 2 time: v0 generation {c['v0_stage2_ms_median']:.0f} ms; probe "
             + ", ".join(f"{k} {v:.0f} ms" for k, v in c["probe_ms_median"].items())
             + f". Peak reserved GPU memory (whole process, all steps): {c['peak_reserved_gb_max']} GB. "
             f"Minimum probability mass on Yes/No tokens: {c['min_yes_no_mass']}.\n")
    return "\n".join(L) + "\n"


# ============================================================
# Plots
# ============================================================

def make_plots(R, test, chosen, TH):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plots] skipped: {e}")
        return
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    v0, v1 = "v0_production", f"probe_{chosen}"
    T = R["test"]
    classes = HEADLINE + [PATCH]
    # 1) per-class MCC with CIs
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = range(len(classes))
    for off, m, col, lab in ((-0.2, v0, "#9ca3af", "production (free-form list)"),
                             (0.2, v1, "#1c1917", f"per-type probe ({chosen})")):
        pts = [T[m]["ci"]["per_class_mcc"][c] for c in classes]
        ax.bar([x + off for x in xs], [p["point"] for p in pts], width=0.4, color=col, label=lab)
        ax.errorbar([x + off for x in xs], [p["point"] for p in pts],
                    yerr=[[p["point"] - p["lo"] for p in pts], [p["hi"] - p["point"] for p in pts]],
                    fmt="none", ecolor="#dc2626", capsize=3, lw=1)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([c.replace(" and utility cut", "") for c in classes], rotation=15)
    ax.set_ylabel("MCC (TEST)  -  0 = constant predictor")
    ax.set_title("Per-class Matthews correlation, Attain WS_V2.0 test split")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "per_class_mcc.png", dpi=300)
    plt.close(fig)
    # 2) ROC per class with the production operating point
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
    for ax, c in zip(axes.flat, classes):
        y = [c in r["gt"] for r in test]
        s = [probe_score(r, chosen, c) for r in test]
        pts = sorted(set(s), reverse=True)
        P = sum(y)
        N = len(y) - P
        fpr, tpr = [0.0], [0.0]
        for t in pts:
            tp = sum(1 for sc, yy in zip(s, y) if sc >= t and yy)
            fp = sum(1 for sc, yy in zip(s, y) if sc >= t and not yy)
            tpr.append(tp / P if P else 0)
            fpr.append(fp / N if N else 0)
        ax.plot(fpr, tpr, color="#1c1917", lw=1.5, label=f"probe AUROC {T[v1]['per_class'][c]['auroc']:.2f}")
        ax.plot([0, 1], [0, 1], color="#d6d3d1", lw=0.8)
        a = T[v0]["per_class"][c]
        ax.scatter([1 - a["specificity"]], [a["recall"]], color="#dc2626", zorder=3, s=25, label="production")
        b = T[v1]["per_class"][c]
        ax.scatter([1 - b["specificity"]], [b["recall"]], color="#16a34a", zorder=3, s=25, marker="s",
                   label="probe @ dev threshold")
        ax.set_title(f"{c.replace(' and utility cut', '')} (prev. {100 * a['prevalence']:.0f}%)", fontsize=9)
        ax.set_xlabel("false positive rate", fontsize=8)
        ax.set_ylabel("true positive rate", fontsize=8)
        ax.legend(fontsize=6, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "roc_per_class.png", dpi=300)
    plt.close(fig)
    print(f"[plots] {PLOT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
