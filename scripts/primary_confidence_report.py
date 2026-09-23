"""
Primary-type confidence: does the change hold up on labelled data?

Two changes went in together on 2026-09-23 and are evaluated together here,
on the same 407 labelled Attain images, paired image by image:

  1. PROMPT  - the v2 Stage 2 prompt now asks for the PRIMARY (most
               prominent) distress first. Before, the model listed labels in
               the order of the inspection protocol.
  2. GATE    - the review gate reads the probability of that first label
               (`primary`) instead of the geometric mean over every label
               (`field`). Secondary labels are shown beside the primary one.

Questions, in the order they have to be answered:

  A. Did the prompt change hurt the multi-label answer? (paired McNemar per
     correctness rule; the earlier run is the control)
  B. Is the first label now the dominant distress? Dominant = the annotated
     class with the largest summed box area.
  C. Does `primary` separate right primary labels from wrong ones, and what
     does the 0.80 gate cost with it?

Inputs are the calibration files written by scripts/calibrate_confidence.py.

Usage:
    venv/Scripts/python.exe scripts/primary_confidence_report.py \\
        --old calib_attain_407.json --new calib_attain_407_primary.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from scripts.calibrate_confidence import (  # noqa: E402
    attain_dominant_class, auc, auc_stderr, map_pred_to_dataset_space)
from scripts.irc82_taxonomy import canonicalize_to_irc  # noqa: E402
from scripts.utils import CONFIDENCE_THRESHOLD, EVAL_DIR  # noqa: E402

DEFINITIONS = ("no_false_positives", "any_overlap", "jaccard_50", "exact_match")
THRESHOLDS = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(name: str) -> dict:
    p = Path(name)
    if not p.is_absolute():
        p = EVAL_DIR / name
    d = json.loads(p.read_text(encoding="utf-8"))
    return {r["image"]: r for r in d["per_image"]
            if r.get("ran_stage2") and not r.get("error")}, d


def first_is_dominant(r: dict):
    """Is the first label the dominant annotated class? None if undecidable."""
    if not r.get("pred_raw"):
        return None
    dom = r.get("gt_dominant") or attain_dominant_class(r["image_path"]).get("gt_dominant")
    first = map_pred_to_dataset_space([r["pred_raw"][0]], "attain")
    if not dom or not first:
        return None
    return dom in first


def primary_geomean(r: dict):
    for tc in r.get("type_confidences") or []:
        if canonicalize_to_irc(tc["label"]) == r.get("primary_label"):
            return tc["geomean"]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="calib_attain_407.json")
    ap.add_argument("--new", default="calib_attain_407_primary.json")
    ap.add_argument("--out", default="primary_confidence")
    args = ap.parse_args()

    old, _ = load(args.old)
    new, new_meta = load(args.new)
    paired = sorted(k for k in old if k in new
                    and old[k].get("adjudicable") and new[k].get("adjudicable"))
    out = {"old": args.old, "new": args.new, "n_paired": len(paired),
           "threshold": CONFIDENCE_THRESHOLD}
    md = ["# Primary-type confidence: evaluation",
          "",
          f"Paired labelled Attain images: **{len(paired)}** "
          f"(old = `{args.old}`, unordered prompt; new = `{args.new}`, primary-first prompt). "
          f"New run: {new_meta.get('quantization_bits')}-bit, prompts `{new_meta.get('prompts_version')}`.",
          ""]

    # ---------------- A. multi-label answer, paired ----------------
    md += ["## A. Did the prompt change hurt the multi-label answer?", "",
           "| rule | old correct | new correct | old right, new wrong | old wrong, new right | McNemar p |",
           "|---|---:|---:|---:|---:|---:|"]
    out["answer_ab"] = {}
    for d in DEFINITIONS:
        o = sum(old[k]["correct"][d] for k in paired)
        n = sum(new[k]["correct"][d] for k in paired)
        b = sum(1 for k in paired if old[k]["correct"][d] and not new[k]["correct"][d])
        c = sum(1 for k in paired if not old[k]["correct"][d] and new[k]["correct"][d])
        p = mcnemar_p(b, c)
        out["answer_ab"][d] = {"old": o, "new": n, "b": b, "c": c, "p": round(p, 4)}
        md.append(f"| {d} | {o} ({100*o/len(paired):.1f}%) | {n} ({100*n/len(paired):.1f}%) "
                  f"| {b} | {c} | {p:.3f} |")
    lab_o = sum(len(old[k]["pred_raw"]) for k in paired) / len(paired)
    lab_n = sum(len(new[k]["pred_raw"]) for k in paired) / len(paired)
    same_set = sum(1 for k in paired if set(old[k]["pred_raw"]) == set(new[k]["pred_raw"]))
    out["labels_per_image"] = {"old": round(lab_o, 3), "new": round(lab_n, 3),
                               "same_label_set": same_set}
    md += ["", f"Labels per image: old {lab_o:.2f}, new {lab_n:.2f}. "
               f"Identical label set on {same_set}/{len(paired)} images "
               f"({100*same_set/len(paired):.0f}%): the prompt change mostly reordered labels.", ""]

    # ---------------- B. is the first label the dominant one? ----------------
    fo = Counter(old[k]["pred_raw"][0] for k in paired if old[k]["pred_raw"])
    fn = Counter(new[k]["pred_raw"][0] for k in paired if new[k]["pred_raw"])
    md += ["## B. Is the first label the dominant distress?", "",
           "Dominant = the annotated class with the largest summed bounding-box area "
           "(Attain vocabulary). Undecidable when the first label or the dominant "
           "class has no counterpart in the other vocabulary.", "",
           "| first label | old | new |", "|---|---:|---:|"]
    for lab in sorted(set(fo) | set(fn), key=lambda x: -(fo[x] + fn[x])):
        md.append(f"| {lab} | {fo[lab]} | {fn[lab]} |")
    dec = [(k, first_is_dominant(old[k]), first_is_dominant(new[k])) for k in paired]
    dec = [(k, a, b) for k, a, b in dec if a is not None and b is not None]
    do = sum(1 for _, a, _b in dec if a)
    dn = sum(1 for _, _a, b in dec if b)
    b = sum(1 for _, a, bb in dec if a and not bb)
    c = sum(1 for _, a, bb in dec if not a and bb)
    p = mcnemar_p(b, c)
    lo_o, hi_o = wilson(do, len(dec))
    lo_n, hi_n = wilson(dn, len(dec))
    out["first_is_dominant"] = {"n": len(dec), "old": do, "new": dn, "b": b, "c": c,
                                "p": round(p, 4),
                                "first_label_old": dict(fo), "first_label_new": dict(fn)}
    md += ["", f"First label = dominant class, on {len(dec)} decidable images: "
               f"old **{do}** ({100*do/len(dec):.1f}% [{100*lo_o:.0f}-{100*hi_o:.0f}]), "
               f"new **{dn}** ({100*dn/len(dec):.1f}% [{100*lo_n:.0f}-{100*hi_n:.0f}]); "
               f"{c} images improved, {b} got worse, McNemar p = {p:.3f}.", ""]

    # ---------------- C. primary confidence ----------------
    rows = [new[k] for k in paired
            if new[k].get("primary_correct") is not None and new[k].get("primary") is not None]
    span = sum(1 for k in paired if new[k].get("primary_span_found"))
    pos = [r for r in rows if r["primary_correct"]]
    neg = [r for r in rows if not r["primary_correct"]]
    md += ["## C. Does the primary confidence separate right from wrong primary labels?", "",
           f"Scored on the primary label alone: right if the dataset annotates it on "
           f"the image. {len(rows)} images have a primary label the dataset can judge "
           f"({len(pos)} right, {len(neg)} wrong). Primary label located in the token "
           f"stream on {span}/{len(paired)} images.", "",
           "| metric | AUC ± SE | mean (right) | mean (wrong) |", "|---|---:|---:|---:|"]
    out["primary_auc"] = {}
    for name, get in (("primary (joint)", lambda r: r["primary"]),
                      ("primary (geomean)", primary_geomean),
                      ("field (all labels)", lambda r: r["field"])):
        ps = [get(r) for r in pos if get(r) is not None]
        ns = [get(r) for r in neg if get(r) is not None]
        a = auc(ps, ns)
        se = auc_stderr(a, len(ps), len(ns))
        out["primary_auc"][name] = {"auc": round(a, 4) if a == a else None,
                                    "se": round(se, 4) if se == se else None,
                                    "n_pos": len(ps), "n_neg": len(ns)}
        md.append(f"| {name} | {a:.3f} ± {se:.3f} | "
                  f"{sum(ps)/max(len(ps),1):.3f} | {sum(ns)/max(len(ns),1):.3f} |")

    md += ["", "### Operating points for `primary` (joint)", "",
           "| threshold | auto-accepted | review load | auto-accept error (95% CI) | wrong caught |",
           "|---:|---:|---:|---:|---:|"]
    out["thresholds"] = {}
    for th in THRESHOLDS:
        acc = [r for r in rows if r["primary"] >= th]
        bad = [r for r in acc if not r["primary_correct"]]
        lo, hi = wilson(len(bad), len(acc))
        caught = len(neg) - len(bad)
        err = 100 * len(bad) / len(acc) if acc else float("nan")
        out["thresholds"][f"{th:.2f}"] = {
            "accepted": len(acc), "review_load_pct": round(100 * (1 - len(acc) / len(rows)), 1),
            "error_pct": round(err, 1) if acc else None, "ci": [round(100*lo, 1), round(100*hi, 1)],
            "caught": caught, "n_wrong": len(neg)}
        star = " ← live" if abs(th - CONFIDENCE_THRESHOLD) < 1e-9 else ""
        md.append(f"| {th:.2f}{star} | {len(acc)}/{len(rows)} | {100*(1-len(acc)/len(rows)):.1f}% "
                  f"| {err:.1f}% [{100*lo:.0f}-{100*hi:.0f}] | {caught}/{len(neg)} |")

    # Review load at the live threshold: old gate vs new gate, same images
    all_new = [new[k] for k in paired]
    all_old = [old[k] for k in paired]
    rl_old = sum(1 for r in all_old if r["field"] < CONFIDENCE_THRESHOLD) / len(paired)
    rl_new = sum(1 for r in all_new
                 if (r.get("primary") if r.get("primary") is not None else r["field"])
                 < CONFIDENCE_THRESHOLD) / len(paired)
    out["review_load_at_live"] = {"old_field_old_prompt": round(100 * rl_old, 1),
                                  "new_primary_new_prompt": round(100 * rl_new, 1)}
    md += ["", f"Stage 2 review load at the live {CONFIDENCE_THRESHOLD:.2f} threshold on these "
               f"{len(paired)} images: old gate (field, old prompt) **{100*rl_old:.1f}%**, "
               f"new gate (primary, new prompt) **{100*rl_new:.1f}%**.", ""]

    (EVAL_DIR / f"{args.out}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (EVAL_DIR / f"{args.out}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nWritten: {EVAL_DIR / (args.out + '.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
