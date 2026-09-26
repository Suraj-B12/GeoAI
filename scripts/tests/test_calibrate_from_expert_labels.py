"""
End-to-end test of scripts/calibrate_probe_from_expert_labels.py (GPU-free).

Real expert labels for Bengaluru uploads do not exist yet, so the code path
that will turn them into thresholds would otherwise run for the first time on
the day they arrive. Here Attain frames stand in for expert-reviewed rows:
their annotations become `expert_corrected_types`, their stored probe scores
become `raw_response.stage2_probe.p` and the tile scores from the views
experiment become `views.recorded` - exactly the shape production writes in
shadow mode. The 20-frame block stands in for the upload day.

Checks: the script finds the rows, evaluates BOTH score sources (full photo
and recorded tiles) on the same rows, writes a candidate config that
validates, and records which source won.

Usage:
    venv/Scripts/python.exe scripts/tests/test_calibrate_from_expert_labels.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.calibrate_probe_from_expert_labels as C  # noqa: E402
from scripts import stage2_probe_rules as rules  # noqa: E402
from scripts.stage2_probe import all_probe_types  # noqa: E402

EVAL = PROJECT_ROOT / "eval_results"
TO_IRC = {"Linear crack": "Longitudinal Cracking", "Alligator crack": "Alligator Cracking",
          "Pothole": "Potholes", "Raveling": "Ravelling", "Weathering": "Hungry Surface"}


def fake_rows():
    raw = json.loads((EVAL / "stage2_probe_raw_attain.json").read_text(encoding="utf-8"))
    views = {}
    for name in ("stage2_views_attain_dev.json", "stage2_views_attain_test.json"):
        for r in json.loads((EVAL / name).read_text(encoding="utf-8"))["rows"]:
            views[r["key"]] = r
    out = []
    for r in raw["rows"]:
        v = views.get(r["image"])
        if "error" in r or not v or not v.get("tile_up"):
            continue
        out.append({
            "id": r["image"],
            "processed_at": f"2026-01-{(r['index'] - 1) // 20 + 1:02d}T00:00:00Z",
            "distress_types": r["v0"]["distress_types"],
            "expert_corrected_types": [TO_IRC[c] for c in r["gt"] if c in TO_IRC],
            "raw_response": {"stage2_generated_types": r["v0"]["distress_types"],
                             "stage2_probe": {"p": v["full"], "views": {"recorded": {
                                 "mode": "tile", "aggregate": "+full", "views": v["tile_up"]}}}},
        })
    return out


def main() -> int:
    rows = fake_rows()
    C.fetch_reviewed = lambda: rows
    sys.argv = ["calibrate", "--out", "_test_calibrate_expert", "--min-pos", "15"]
    rc = C.main()
    rep = json.loads((EVAL / "_test_calibrate_expert.json").read_text(encoding="utf-8"))
    cand = json.loads((EVAL / "_test_calibrate_expert_config_candidate.json").read_text(encoding="utf-8"))
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(f"[{'OK' if cond else 'FAIL'}] {name}" + ("" if cond else f" - {detail}"))
        ok = ok and cond

    check("script finished", rc == 0, str(rc))
    check("rows found", rep["n_rows"] == len(rows), f"{rep['n_rows']} vs {len(rows)}")
    check("both sources evaluated", {"full", "tile"} <= set(rep["sources"]), str(list(rep["sources"])))
    check("paired full baseline on the same rows",
          any(k.startswith("full_on_same_") for k in rep["sources"]))
    check("best source recorded", rep["best_source"] in rep["sources"])
    try:
        rules.validate_config(cand, {t.key for t in all_probe_types()})
        valid = True
    except ValueError as e:
        valid = False
        print("   ", e)
    check("candidate config validates", valid)
    check("candidate views match the winner",
          (cand["views"]["mode"] == "full") == (rep["best_source"] == "full"), str(cand["views"]))
    for s, res in rep["sources"].items():
        print(f"    {s}: n={res['n']} macro MCC free-form {res['macro_mcc_generated']:.3f} "
              f"-> probe {res['macro_mcc_probe']:.3f}")
    for f in ("_test_calibrate_expert.json", "_test_calibrate_expert_config_candidate.json"):
        (EVAL / f).unlink(missing_ok=True)
    print("ALL PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
