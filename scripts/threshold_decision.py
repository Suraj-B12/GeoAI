"""
Decide the expert-review confidence threshold from calibration data.

Consumes one or more outputs of `scripts/calibrate_confidence.py` and answers
the only question that matters operationally: at which confidence value should
an image stop going to a human?

Why this is a separate script
-----------------------------
The calibration run is expensive and its result is a fixed artefact. The
threshold decision is cheap, and it is going to be revisited every time new
labelled data arrives. Keeping them apart means the decision can be re-derived
in a second, from stored data, under a different correctness definition or a
different cost assumption, without touching a GPU.

What a threshold actually trades
--------------------------------
Raising it sends more images to the expert: fewer mistakes are auto-accepted,
but reviewers spend time on predictions that were already correct. Lowering it
does the reverse. There is no universally right answer - it depends on the
relative cost of a wrong auto-accept versus an unnecessary review. So this
prints both costs at every candidate and does not pretend one number is
objectively optimal.

Two guards against fooling ourselves
------------------------------------
1. A threshold is only meaningfully better than another if the difference
   survives the sampling noise. Every rate is printed with a Wilson 95%
   interval, and the recommendation refuses to distinguish candidates whose
   intervals overlap.
2. The conclusion is reported under EVERY correctness definition, because the
   pilot for this work showed the definition - not the metric - drives the
   answer: field confidence scored AUC 1.000 under `no_false_positives` and
   0.500 under `jaccard_50` on the same images. Quoting one number without its
   definition is how the earlier "AUC 0.613" ended up unreproducible.

Usage:
    venv/Scripts/python.exe scripts/threshold_decision.py \
        --in calib_attain_407.json --in calib_rdd_india_250.json \
        --out threshold_decision.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils import CONFIDENCE_THRESHOLD, EVAL_DIR  # noqa: E402

DEFINITIONS = ("no_false_positives", "jaccard_50", "any_overlap", "exact_match")


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score interval - trustworthy at small n and near 0/1, unlike the
    normal approximation, which is exactly where threshold questions live."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def auc(pos: list, neg: list) -> float:
    if not pos or not neg:
        return float("nan")
    wins = sum(1.0 if p > n else (0.5 if p == n else 0.0) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def auc_stderr(a: float, n_pos: int, n_neg: int) -> float:
    if not (n_pos and n_neg) or a != a:
        return float("nan")
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    v = (a * (1 - a) + (n_pos - 1) * (q1 - a * a)
         + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg)
    return math.sqrt(v) if v > 0 else float("nan")


def load(paths: list) -> list:
    rows = []
    for raw in paths:
        p = Path(raw)
        if not p.is_absolute():
            p = EVAL_DIR / raw
        if not p.exists():
            print(f"  [skip] {p.name} not found")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        n = 0
        for r in d.get("per_image", []):
            if r.get("ran_stage2") and r.get("adjudicable"):
                r["_source"] = p.name
                r["_stage1_skipped"] = bool(d.get("stage1_skipped"))
                rows.append(r)
                n += 1
        print(f"  [load] {p.name}: {n} adjudicable Stage 2 rows "
              f"(dataset={d.get('dataset')}, "
              f"stage1_skipped={d.get('stage1_skipped')})")
    return rows


def analyse(rows: list, metric: str, definition: str, thresholds: list) -> dict:
    pos = [r[metric] for r in rows if r["correct"][definition]]
    neg = [r[metric] for r in rows if not r["correct"][definition]]
    a = auc(pos, neg)
    out = {"n": len(rows), "n_correct": len(pos), "n_wrong": len(neg),
           "auc": a, "auc_se": auc_stderr(a, len(pos), len(neg)), "rows": []}
    for th in thresholds:
        acc = [r for r in rows if r[metric] >= th]
        bad = [r for r in acc if not r["correct"][definition]]
        caught = [r for r in rows if r[metric] < th and not r["correct"][definition]]
        wasted = [r for r in rows if r[metric] < th and r["correct"][definition]]
        err_lo, err_hi = wilson(len(bad), len(acc)) if acc else (0.0, 0.0)
        out["rows"].append({
            "threshold": th,
            "auto_accepted": len(acc),
            "auto_accept_pct": 100 * len(acc) / len(rows) if rows else 0,
            "wrong_through": len(bad),
            "err_rate": 100 * len(bad) / len(acc) if acc else float("nan"),
            "err_lo": 100 * err_lo, "err_hi": 100 * err_hi,
            "wrong_caught": len(caught),
            "wrong_caught_pct": 100 * len(caught) / len(neg) if neg else float("nan"),
            "review_load": 100 * (len(rows) - len(acc)) / len(rows) if rows else 0,
            "correct_reviewed": len(wasted),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", action="append", required=True)
    ap.add_argument("--metric", default="field", choices=["field", "sequence"])
    ap.add_argument("--thresholds", default="0.70,0.72,0.75,0.78,0.80,0.82,0.85")
    ap.add_argument("--out", default="threshold_decision.md")
    args = ap.parse_args()
    thresholds = [float(t) for t in args.thresholds.split(",")]

    print("Loading calibration results:")
    rows = load(args.inputs)
    if not rows:
        print("No adjudicable rows found.")
        return 1

    groups = [("ALL POOLED", rows)]
    for ds in sorted({r["dataset"] for r in rows}):
        groups.append((ds, [r for r in rows if r["dataset"] == ds]))

    md = ["# Expert-review threshold decision", "",
          f"Metric: `{args.metric}` (Stage 2 confidence).  "
          f"Live threshold: **{CONFIDENCE_THRESHOLD}**.", "",
          f"Sources: {', '.join(sorted({r['_source'] for r in rows}))}", ""]

    L = "=" * 86
    for gname, grows in groups:
        if not grows:
            continue
        print("\n" + L)
        print(f"{gname}  -  n={len(grows)}")
        print(L)
        md += [f"## {gname} (n={len(grows)})", ""]

        for defn in DEFINITIONS:
            res = analyse(grows, args.metric, defn, thresholds)
            if res["n_wrong"] == 0 or res["n_correct"] == 0:
                print(f"\n{defn}: {res['n_correct']} right / {res['n_wrong']} wrong"
                      f"  - AUC undefined, only one class present")
                md += [f"### {defn}", "",
                       f"{res['n_correct']} right / {res['n_wrong']} wrong - "
                       f"AUC undefined (one class).", ""]
                continue

            se = res["auc_se"]
            # Perfect separation (AUC 0 or 1) drives the Hanley-McNeil variance
            # to zero, so the standard error comes back nan and every
            # comparison against 0.5 silently evaluates false - which labelled
            # a PERFECT separator "INVERTED". Handle it explicitly, and never
            # let the interval run outside [0, 1]: an AUC cannot.
            degenerate = not (se == se) or se == 0.0
            if degenerate:
                lo = hi = float("nan")
                ci_txt = "CI undefined, perfect separation"
                verdict = ("separates perfectly on this sample - with this few "
                           "negatives that is a sample-size artefact, not a result"
                           if res["auc"] > 0.5 else
                           "perfectly INVERTED on this sample")
            else:
                lo = max(0.0, res["auc"] - 1.96 * se)
                hi = min(1.0, res["auc"] + 1.96 * se)
                ci_txt = f"95% CI {lo:.3f}-{hi:.3f}"
                verdict = ("carries no information (95% CI spans 0.5)"
                           if lo <= 0.5 <= hi else
                           "informative" if lo > 0.5 else "INVERTED")
            print(f"\n{defn}: {res['n_correct']} right / {res['n_wrong']} wrong   "
                  f"AUC {res['auc']:.3f} ({ci_txt})  -> {verdict}")
            print(f"{'thresh':>7}{'accepted':>11}{'wrong thru':>12}"
                  f"{'err rate (95% CI)':>22}{'wrong caught':>14}{'review load':>13}")
            md += [f"### {defn}", "",
                   f"{res['n_correct']} right / {res['n_wrong']} wrong. "
                   f"AUC **{res['auc']:.3f}** ({ci_txt}) - {verdict}.", "",
                   "| threshold | auto-accepted | wrong through | error rate (95% CI) "
                   "| wrong caught | review load |",
                   "|---:|---:|---:|---:|---:|---:|"]

            for t in res["rows"]:
                star = " *" if abs(t["threshold"] - CONFIDENCE_THRESHOLD) < 1e-9 else "  "
                ci = (f"{t['err_rate']:.0f}% [{t['err_lo']:.0f}-{t['err_hi']:.0f}]"
                      if t["err_rate"] == t["err_rate"] else "n/a")
                print(f"{t['threshold']:>6.2f}{star}"
                      f"{t['auto_accepted']:>6}/{len(grows):<4}"
                      f"{t['wrong_through']:>12}{ci:>22}"
                      f"{t['wrong_caught']:>7}/{res['n_wrong']:<6}"
                      f"{t['review_load']:>11.0f}%")
                md.append(
                    f"| {t['threshold']:.2f}{'  <- live' if star == ' *' else ''} "
                    f"| {t['auto_accepted']}/{len(grows)} ({t['auto_accept_pct']:.0f}%) "
                    f"| {t['wrong_through']} | {ci} "
                    f"| {t['wrong_caught']}/{res['n_wrong']} "
                    f"({t['wrong_caught_pct']:.0f}%) | {t['review_load']:.0f}% |")
            md.append("")

            # Direct comparison of the two candidates actually on the table.
            cand = {t["threshold"]: t for t in res["rows"]}
            if 0.80 in cand and 0.75 in cand:
                a80, a75 = cand[0.80], cand[0.75]
                overlap = not (a80["err_hi"] < a75["err_lo"]
                               or a75["err_hi"] < a80["err_lo"])
                line = (f"0.80 -> 0.75: auto-accepts "
                        f"{a75['auto_accepted'] - a80['auto_accepted']:+d} images, "
                        f"lets through {a75['wrong_through'] - a80['wrong_through']:+d} "
                        f"more wrong ones, catches "
                        f"{a75['wrong_caught'] - a80['wrong_caught']:+d} fewer errors, "
                        f"review load {a80['review_load']:.0f}% -> {a75['review_load']:.0f}%.")
                note = ("  Error rates are NOT statistically distinguishable "
                        "at this sample size." if overlap else
                        "  The error-rate difference is outside sampling noise.")
                print("   " + line)
                print("   " + note.strip())
                md += [f"**0.80 vs 0.75.** {line}{note}", ""]

    out = EVAL_DIR / args.out
    out.write_text("\n".join(md), encoding="utf-8")
    print("\n" + L)
    print(f"Written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
