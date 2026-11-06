"""
Score the weight-space programming study (paper §6.11). No inference.

Reads
  C:\\GeoAI\\wsp\\analysis\\base_diagnose.json   base Stage 1 logits on DEV (for the
                                                recalibrated-base offset b*)
  C:\\GeoAI\\wsp\\final\\final_raw.json          the ONE final evaluation run
                                                (scripts/wsp_final_eval.py)
  eval_results/finetune/wsp_ledger.jsonl        passes used, test runs
  eval_results/finetune/wsp_dev_runs.json       the editor's dev record
Writes eval_results/finetune/wsp_report.json and wsp_report.md.

Definitions are those of eval_results/finetune/wsp_preregistration.json:
  score s = logit(Distress) - logit(Normal) at the first generated token
  production rule: Distress iff s > 0; expert review iff max(P) < 0.80
  recalibrated base: s + b*, b* maximising DEV MCC at the argmax rule
  (grid over midpoints of the dev scores; ties -> smallest |b|)
  CIs: cluster (20-frame block) bootstrap, B = 2000, paired for differences
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import multilabel_metrics as M  # noqa: E402
from scripts import wsp_core as W  # noqa: E402
from scripts.ft_common import FT_DIR, atomic_write_json, atomic_write_text, read_json  # noqa: E402

B = 2000
GATE = 0.80
RAW = W.WSP_DIR / "final" / "final_raw.json"


def sig(x):
    return 1 / (1 + math.exp(-x))


def best_offset(scores, y):
    """Constant offset b maximising MCC of (s + b > 0); ties -> smallest |b|."""
    srt = sorted(set(scores))
    cands = [0.0] + [-(a + b) / 2 for a, b in zip(srt, srt[1:])] + [-(srt[0] - 1), -(srt[-1] + 1)]
    best = None
    for b in cands:
        m = M.binary_metrics(y, [s + b > 0 for s in scores])["mcc"]
        key = (round(m, 12), -abs(b))
        if best is None or key > best[0]:
            best = (key, b, m)
    return best[1], best[2]


def rows_for(cfg_res: dict, test: list, offset: float = 0.0) -> list:
    out = []
    for r in test:
        x = cfg_res[r["key"]]
        s = x["logit_distress"] - x["logit_normal"] + offset
        out.append({"key": r["key"], "block": r["block"], "y": bool(r["gt"]), "s": s,
                    "pred": s > 0, "review": max(sig(s), 1 - sig(s)) < GATE})
    return out


def metric(name):
    def f(rs):
        y = [r["y"] for r in rs]
        if name == "auroc":
            return M.roc_auc([r["s"] for r in rs], y)
        if name == "review":
            return sum(r["review"] for r in rs) / len(rs)
        bm = M.binary_metrics(y, [r["pred"] for r in rs])
        return bm[{"recall": "recall", "specificity": "specificity", "mcc": "mcc"}[name]]
    return f


def clusters(rows):
    c = defaultdict(list)
    for r in rows:
        c[r["block"]].append(r)
    return dict(c)


def paired(rows_a, rows_b, name, seed):
    """Paired bootstrap on aligned rows (same keys, same order)."""
    pairs = defaultdict(list)
    for a, b in zip(rows_a, rows_b):
        assert a["key"] == b["key"]
        pairs[a["block"]].append((a, b))
    fa = lambda ps: metric(name)([p[0] for p in ps])  # noqa: E731
    fb = lambda ps: metric(name)([p[1] for p in ps])  # noqa: E731
    return M.paired_cluster_bootstrap(fa, fb, dict(pairs), B, seed)


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def stage2_ok(rec) -> bool:
    raw = rec.get("raw") or ""
    types = rec.get("distress_types") or []
    fields = all(f in raw.upper() for f in ("DISTRESS_TYPES", "SEVERITY", "DESCRIPTION"))
    return fields and bool(types) and not all(str(t).lower() in ("unknown", "other") for t in types)


def main() -> int:
    raw = read_json(RAW)
    if not raw or not raw.get("complete"):
        sys.exit(f"{RAW} missing or incomplete")
    P = W.pools()
    test = sorted(P["test"], key=lambda r: r["key"])
    dev = P["dev"]
    diag = read_json(W.WSP_DIR / "analysis" / "base_diagnose.json")
    dev_s = [diag["s1"][r["key"]]["logit_distress"] - diag["s1"][r["key"]]["logit_normal"] for r in dev]
    dev_y = [bool(r["gt"]) for r in dev]
    b_star, dev_mcc = best_offset(dev_s, dev_y)

    cfg = raw["configs"]
    R = {"base": rows_for(cfg["base"], test), "recalibrated_base": rows_for(cfg["base"], test, b_star),
         "edited": rows_for(cfg["final"], test), "random_control": rows_for(cfg["random"], test)}
    rep = {"n_test": len(test), "n_test_distress": sum(r["y"] for r in R["base"]),
           "n_test_clean": sum(not r["y"] for r in R["base"]),
           "recalibration": {"offset_b_star": b_star, "dev_mcc_at_b_star": dev_mcc,
                             "dev_mcc_at_0": M.binary_metrics(dev_y, [s > 0 for s in dev_s])["mcc"]},
           "test": {}, "paired": {}}
    for name, rows in R.items():
        cl = clusters(rows)
        rep["test"][name] = {m: M.cluster_bootstrap(metric(m), cl, B, i)
                             for i, m in enumerate(("auroc", "recall", "specificity", "mcc", "review"))}
        rep["test"][name]["top_token_is_label_share"] = (
            sum(cfg["final" if name == "edited" else "random" if name == "random_control" else "base"]
                [r["key"]]["top_is_label"] for r in test) / len(test))
    rep["paired"] = {
        "auroc_edited_minus_base": paired(R["base"], R["edited"], "auroc", 101),
        "mcc_edited_minus_recalibrated": paired(R["recalibrated_base"], R["edited"], "mcc", 102),
        "mcc_edited_minus_base": paired(R["base"], R["edited"], "mcc", 103),
        "specificity_edited_minus_base": paired(R["base"], R["edited"], "specificity", 104),
        "recall_edited_minus_base": paired(R["base"], R["edited"], "recall", 105),
        "review_edited_minus_base": paired(R["base"], R["edited"], "review", 106),
        "auroc_random_minus_base": paired(R["base"], R["random_control"], "auroc", 107),
        "mcc_random_minus_base": paired(R["base"], R["random_control"], "mcc", 108),
        "mcc_recalibrated_minus_base": paired(R["base"], R["recalibrated_base"], "mcc", 109),
    }
    # regression checks
    beng = raw["bengaluru"]
    reg = {"bengaluru": {}}
    for c, recs in beng.items():
        k = sum(1 for x in recs.values() if x["logit_distress"] > x["logit_normal"])
        reg["bengaluru"][c] = {"n": len(recs), "distress": k, "share": k / len(recs), "ci95": wilson(k, len(recs))}
    reg["bengaluru_share_change_pp"] = 100 * (reg["bengaluru"]["final"]["share"] - reg["bengaluru"]["base"]["share"])
    s2 = raw["stage2_dev"]
    reg["stage2_parse"] = {c: {"n": len(v), "failures": sum(1 for x in v.values() if not stage2_ok(x))}
                           for c, v in s2.items()}
    pr = raw["probe_dev"]
    keys = sorted(set(pr["base"]) & set(pr["final"]))
    types = sorted(pr["base"][keys[0]])
    reg["probe_dev_mean_abs_dp"] = {t: sum(abs(pr["final"][k][t] - pr["base"][k][t]) for k in keys) / len(keys)
                                    for t in types}
    reg["probe_dev_n"] = len(keys)
    rep["regression"] = reg
    rep["bake_check"] = raw.get("bake_check")
    rep["cost"] = raw.get("cost")

    # pre-registered rule
    pa = rep["paired"]
    better = (pa["auroc_edited_minus_base"]["lo"] > 0) or (pa["mcc_edited_minus_recalibrated"]["lo"] > 0)
    spec_ok = pa["specificity_edited_minus_base"]["lo"] <= 0 <= pa["specificity_edited_minus_base"]["hi"] \
        or pa["specificity_edited_minus_base"]["lo"] > 0
    b = reg["bengaluru"]
    beng_ok = reg["bengaluru_share_change_pp"] <= 20 and b["final"]["share"] <= 0.95
    s2_ok = reg["stage2_parse"]["final"]["failures"] <= reg["stage2_parse"]["base"]["failures"] + 1
    probe_ok = all(v <= 0.05 for v in reg["probe_dev_mean_abs_dp"].values())
    rep["verdict"] = {"better_criterion": better, "specificity_ok": spec_ok, "bengaluru_ok": beng_ok,
                      "stage2_ok": s2_ok, "probe_ok": probe_ok,
                      "adopt_as_better": bool(better and spec_ok and beng_ok and s2_ok and probe_ok)}
    led = W.ledger_rows()
    rep["passes"] = {"budget": W.BUDGET, "used": sum(1 for r in led if r.get("counted")),
                     "test_evaluation_runs": sorted({r["run_id"] for r in led if r.get("final")
                                                     and "test" in (r.get("pools") or [])})}
    rep["final_edit"] = read_json(W.FINAL_DIR / "manifest.json")
    atomic_write_json(FT_DIR / "wsp_report.json", rep)
    atomic_write_text(FT_DIR / "wsp_report.md", render_md(rep))
    print(json.dumps(rep["verdict"], indent=1))
    return 0


def ci(x, pct=True):
    f = (lambda v: f"{100 * v:.1f}%") if pct else (lambda v: f"{v:.3f}")
    return f"{f(x['point'])} [{f(x['lo'])}, {f(x['hi'])}]"


def dci(x, pct=False):
    f = (lambda v: f"{100 * v:+.1f} pp") if pct else (lambda v: f"{v:+.3f}")
    return f"{f(x['delta'])} [{f(x['lo'])}, {f(x['hi'])}], p = {x['p']:.3f}"


def render_md(r) -> str:
    t = r["test"]
    L = ["# Weight-space programming on the production model — held-out report", "",
         "Generated by `scripts/wsp_report.py` from the single final evaluation run "
         f"({', '.join(r['passes']['test_evaluation_runs'])}). Pre-registration: `wsp_preregistration.json`.", "",
         f"Test split: {r['n_test']} WS_V2.0 frames ({r['n_test_distress']} with distress, {r['n_test_clean']} without). "
         f"Forward passes used while editing: {r['passes']['used']} of {r['passes']['budget']}.", "",
         "| Model | AUROC | Recall | Specificity | MCC | Expert-review share |", "|---|---:|---:|---:|---:|---:|"]
    names = {"base": "Base (production)", "recalibrated_base": f"Recalibrated base (offset {r['recalibration']['offset_b_star']:+.2f})",
             "edited": "Edited weights (baked bf16)", "random_control": "Random-direction control"}
    for k, n in names.items():
        x = t[k]
        L.append(f"| {n} | {ci(x['auroc'], False)} | {ci(x['recall'])} | {ci(x['specificity'])} | "
                 f"{ci(x['mcc'], False)} | {ci(x['review'])} |")
    p = r["paired"]
    L += ["", "Paired differences (block bootstrap, B = 2000):", "",
          f"- AUROC, edited − base: {dci(p['auroc_edited_minus_base'])}",
          f"- MCC, edited − recalibrated base: {dci(p['mcc_edited_minus_recalibrated'])}",
          f"- MCC, edited − base: {dci(p['mcc_edited_minus_base'])}",
          f"- Recall, edited − base: {dci(p['recall_edited_minus_base'], True)}",
          f"- Specificity, edited − base: {dci(p['specificity_edited_minus_base'], True)}",
          f"- Expert-review share, edited − base: {dci(p['review_edited_minus_base'], True)}",
          f"- MCC, recalibrated − base: {dci(p['mcc_recalibrated_minus_base'])}",
          f"- Random control: AUROC − base {dci(p['auroc_random_minus_base'])}; MCC − base {dci(p['mcc_random_minus_base'])}",
          "", "Regression checks:", ""]
    g = r["regression"]
    for c, b in g["bengaluru"].items():
        L.append(f"- Bengaluru video segments, {c}: Distress on {b['distress']}/{b['n']} ({100 * b['share']:.1f}%, CI {b['ci95']})")
    L += [f"- Bengaluru Distress-share change, edited − base: {g['bengaluru_share_change_pp']:+.1f} pp",
          f"- Stage 2 parse failures on {g['stage2_parse']['base']['n']} dev frames: base {g['stage2_parse']['base']['failures']}, "
          f"edited {g['stage2_parse']['final']['failures']}",
          f"- Probe mean |ΔP(yes)| on {g['probe_dev_n']} dev frames: max over types "
          f"{max(g['probe_dev_mean_abs_dp'].values()):.4f}", ""]
    if r.get("bake_check"):
        L.append(f"Baked bf16 weights vs hook version, Stage 1 logits on {r['bake_check'].get('n')} practice frames: "
                 f"max abs difference {r['bake_check'].get('max_abs_diff')}.")
    if r.get("cost"):
        L.append(f"Cost of the baked layers: {json.dumps(r['cost'])}")
    v = r["verdict"]
    L += ["", "## Verdict against the pre-registered rule", "",
          f"- Better (AUROC CI above 0, or MCC beats the recalibrated base with CI above 0): **{v['better_criterion']}**",
          f"- Specificity not significantly lower: {v['specificity_ok']}",
          f"- Bengaluru shift within limits: {v['bengaluru_ok']}; Stage 2 parses: {v['stage2_ok']}; probe stable: {v['probe_ok']}",
          f"- **Adopt as better: {v['adopt_as_better']}**. Production is unchanged either way.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
