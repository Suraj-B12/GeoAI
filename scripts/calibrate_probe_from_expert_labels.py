"""
Fit per-type probe thresholds on EXPERT-LABELLED Bengaluru uploads (read-only).

Why this exists
---------------
On labelled Attain frames the per-type probe doubled macro MCC over the
free-form list (grouped CV, eval_results/stage2_probe_cv.md). But the
thresholds tuned there do not transfer to handheld Bengaluru close-ups: the
same config adds Ravelling to 98% of 204 production uploads. The P(yes)
values themselves look sensible (potholes: median 0.60 on Bengaluru vs 0.02
on Attain road strips), only the decision thresholds are domain-specific.

Production therefore runs the probe in SHADOW mode: every row stores every
type's P(yes) (raw_response.stage2_probe.p). Once experts have reviewed
enough uploads in the expert UI, this script fits Bengaluru thresholds from
those stored probabilities - no model, no GPU, no re-inference - and writes a
CANDIDATE config plus a report. It never touches configs/stage2_probe.json:
promoting a candidate is a deliberate manual step, taken only if the report
shows the probe beating the free-form list on held-out expert labels.

Method (same as the Attain analysis)
------------------------------------
* Ground truth: assessments.expert_corrected_types, canonicalised to IRC names.
* A type is calibrated only if it has at least --min-pos expert positives and
  --min-neg negatives; every other type stays verify-only.
* 5-fold cross-validation grouped by upload DAY (uploads from one outing are
  correlated), MCC thresholds + Platt fitted on four folds, applied to the
  fifth, through the production combine() rule; compared with the stored
  free-form list on the same rows. Cluster-bootstrap intervals by day.
* The candidate is refitted on all labelled rows.

Usage:
    venv/Scripts/python.exe scripts/calibrate_probe_from_expert_labels.py
    venv/Scripts/python.exe scripts/calibrate_probe_from_expert_labels.py --min-pos 20
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import httpx  # noqa: E402

from scripts import multilabel_metrics as M  # noqa: E402
from scripts import stage2_probe_rules as rules  # noqa: E402
from scripts.irc82_taxonomy import IRC82_DISTRESS_TAXONOMY, canonicalize_to_irc  # noqa: E402
from scripts.stage2_probe import all_probe_types  # noqa: E402
from scripts.stage2_views import aggregate  # noqa: E402

EVAL_DIR = PROJECT_ROOT / "eval_results"
LIVE_CONFIG = PROJECT_ROOT / "configs" / "stage2_probe.json"


def env_from_dotenv() -> dict:
    env = {}
    p = PROJECT_ROOT / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def fetch_reviewed() -> list:
    env = env_from_dotenv()
    url, key = env.get("SUPABASE_URL"), env.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        sys.exit("SUPABASE_URL / SUPABASE_SERVICE_KEY missing from .env")
    q = (f"{url}/rest/v1/assessments?select=id,processed_at,reviewed_at,distress_types,"
         f"expert_corrected_types,raw_response&expert_reviewed=eq.true&limit=10000")
    last = None
    for k in range(6):
        try:
            r = httpx.get(q, timeout=60, headers={"apikey": key, "Authorization": f"Bearer {key}"})
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            time.sleep(2 * (k + 1))
    sys.exit(f"could not read assessments: {last}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-pos", type=int, default=15)
    ap.add_argument("--min-neg", type=int, default=15)
    ap.add_argument("--out", default="stage2_probe_bengaluru")
    args = ap.parse_args()

    base = rules.load_config(LIVE_CONFIG) if LIVE_CONFIG.exists() else None
    types = all_probe_types()
    known = {t.key for t in types}
    labels = {t.key: t.output_label for t in types}

    rows = []
    for r in fetch_reviewed():
        rr = r.get("raw_response") or {}
        pr = (rr.get("stage2_probe") or {}).get("p") or (rr.get("stage1_safety_net") or {}).get("p")
        if not pr or set(pr) < {k for k in known if k in IRC82_DISTRESS_TAXONOMY}:
            continue    # reviewed before shadow mode: no stored probabilities
        gt = {canonicalize_to_irc(t) for t in (r.get("expert_corrected_types") or [])}
        gt = {t for t in gt if t in IRC82_DISTRESS_TAXONOMY}
        day = (r.get("processed_at") or r.get("reviewed_at") or "unknown")[:10]
        rec = (((rr.get("stage2_probe") or {}).get("views") or {}).get("recorded")
               or ((rr.get("stage1_safety_net") or {}).get("views") or {}).get("recorded"))
        scores = {"full": pr}
        if rec and rec.get("views"):
            # production decides on the full photo and RECORDS tile views;
            # rebuild the tiled score exactly as stage2_views.aggregate would
            scores[rec["mode"]] = aggregate(pr, rec["views"], rec.get("aggregate", "+full"))
            scores["_record_spec"] = {k: rec[k] for k in ("mode", "aggregate") if k in rec}
        rows.append({"id": r["id"], "day": day, "gt": gt, "p": pr, "scores": scores,
                     "generated": (rr.get("stage2_generated_types")
                                   or r.get("distress_types") or [])})
    print(f"[data] {len(rows)} expert-reviewed rows with stored probe probabilities")

    counts = {k: sum(1 for r in rows if k in r["gt"]) for k in IRC82_DISTRESS_TAXONOMY}
    eligible = [k for k, c in counts.items()
                if c >= args.min_pos and len(rows) - c >= args.min_neg]
    report = {"n_rows": len(rows), "positives_per_type": counts,
              "min_pos": args.min_pos, "min_neg": args.min_neg,
              "calibratable_types": eligible}
    if not eligible:
        need = {k: max(0, args.min_pos - c) for k, c in counts.items() if c > 0}
        report["status"] = "not enough expert labels yet"
        report["more_positives_needed"] = need
        (EVAL_DIR / f"{args.out}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 2

    days = sorted({r["day"] for r in rows})
    rng = random.Random(20260924)
    rng.shuffle(days)
    fold = {d: i % 5 for i, d in enumerate(days)}

    def fit(train, src):
        cfg = {"schema_version": rules.SCHEMA_VERSION,
               "variant": (base or {}).get("variant", {"system_style": "min", "question_style": "name"}),
               "verify_threshold": (base or {}).get("verify_threshold", 0.5), "groups": {}}
        for k in eligible:
            s_ = [r["scores"][src][k] for r in train]
            y = [k in r["gt"] for r in train]
            if not any(y) or all(y):
                continue
            cfg["groups"][k] = {"kind": "distress", "keys": [k],
                                "threshold": min(max(M.best_threshold(s_, y, "mcc")[0], 1e-6), 1 - 1e-6),
                                "platt": list(M.fit_platt(s_, y) or []) or None}
        return cfg

    def evaluate(src, pool):
        oof = {}
        for f in range(5):
            train = [r for r in pool if fold[r["day"]] != f]
            cfg = fit(train, src)
            if not cfg["groups"]:
                continue
            for r in pool:
                if fold[r["day"]] == f:
                    oof[r["id"]] = set(rules.combine(r["generated"], r["scores"][src], cfg, labels)["types"])
        scored = [r for r in pool if r["id"] in oof]
        clusters = defaultdict(list)
        for r in scored:
            clusters[r["day"]].append(r)
        per = {}
        for i, k in enumerate(eligible):
            gen = lambda rs, kk=k: M.binary_metrics([kk in r["gt"] for r in rs], [kk in set(r["generated"]) for r in rs])["mcc"]
            prb = lambda rs, kk=k: M.binary_metrics([kk in r["gt"] for r in rs], [kk in oof[r["id"]] for r in rs])["mcc"]
            per[k] = {"generated_mcc": gen(scored), "probe_mcc": prb(scored),
                      "mcc_delta": M.paired_cluster_bootstrap(gen, prb, dict(clusters), 2000, 50 + i),
                      "auroc": M.roc_auc([r["scores"][src][k] for r in scored], [k in r["gt"] for r in scored])}
        return {"n": len(scored), "per_type": per,
                "macro_mcc_generated": sum(v["generated_mcc"] for v in per.values()) / len(per),
                "macro_mcc_probe": sum(v["probe_mcc"] for v in per.values()) / len(per)}

    report["sources"] = {"full": evaluate("full", rows)}
    recorded = sorted({k for r in rows for k in r["scores"] if k not in ("full", "_record_spec")})
    for src in recorded:
        pool = [r for r in rows if src in r["scores"]]
        report["sources"][src] = evaluate(src, pool)
        report["sources"][f"full_on_same_{len(pool)}_rows"] = evaluate("full", pool)
    best = max(("full", *recorded), key=lambda s_: report["sources"][s_]["macro_mcc_probe"])
    report["best_source"] = best

    pool = rows if best == "full" else [r for r in rows if best in r["scores"]]
    final = fit(pool, best)
    if best == "full":
        final["views"] = {"mode": "full"}
    else:
        spec = next(r["scores"]["_record_spec"] for r in pool if "_record_spec" in r["scores"])
        live_views = (base or {}).get("views") or {}
        rec_spec = dict(live_views.get("record") or {})
        rec_spec.update(spec)
        final["views"] = rec_spec
    final["provenance"] = {"tuned_on": f"{len(pool)} expert-reviewed Bengaluru uploads",
                           "score_source": best,
                           "report": f"eval_results/{args.out}.json",
                           "calibrated_types": sorted(final["groups"])}
    if base and base.get("stage1_safety_net"):
        final["stage1_safety_net"] = base["stage1_safety_net"]
    rules.validate_config(final, known)
    (EVAL_DIR / f"{args.out}_config_candidate.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    report["status"] = ("candidate written; promote by copying to configs/stage2_probe.json and "
                        "setting STAGE2_MODE=probe ONLY if the best source's per-type MCC deltas "
                        "vs the free-form list are positive on held-out labels")
    (EVAL_DIR / f"{args.out}.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    for src, res in report["sources"].items():
        print(f"[{src}] n={res['n']} macro MCC free-form {res['macro_mcc_generated']:.3f} -> probe {res['macro_mcc_probe']:.3f}")
    print(f"best source: {best}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
