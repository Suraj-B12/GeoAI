"""
What would each Stage 2 gate do on real production uploads?

Every row processed since 2026-09-23 stores the whole-field confidence (the
gate) AND the first label's confidence, so the two gates can be compared on
the same photographs with no re-inference. These photographs have no expert
labels: this measures review load and agreement between gates, not accuracy.

Usage:
    venv/Scripts/python.exe scripts/production_gate_comparison.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import dotenv_values  # noqa: E402

from scripts.utils import CONFIDENCE_THRESHOLD, EVAL_DIR  # noqa: E402


def main() -> int:
    env = dotenv_values(PROJECT_ROOT / ".env")
    base, key = env["SUPABASE_URL"].rstrip("/"), env["SUPABASE_SERVICE_KEY"]
    req = urllib.request.Request(
        f"{base}/rest/v1/assessments?select=id,status,is_distressed,stage1_confidence,"
        "distress_types,raw_response&is_distressed=eq.true&order=processed_at.asc",
        headers={"apikey": key, "Authorization": f"Bearer {key}"})
    rows = []
    for r in json.load(urllib.request.urlopen(req, timeout=60)):
        rr = r.get("raw_response") or {}
        if rr.get("stage2_confidence_primary") is None or rr.get("stage2_confidence_field") is None:
            continue
        rows.append({"types": r["distress_types"] or [], "s1": r["stage1_confidence"] or 0,
                     "field": rr["stage2_confidence_field"],
                     "primary": rr["stage2_confidence_primary"]})
    th = CONFIDENCE_THRESHOLD
    n = len(rows)
    if not n:
        print("no rows with both confidences yet")
        return 1

    def load(metric, subset):
        return sum(1 for r in subset if r["s1"] < th or r[metric] < th)

    single = [r for r in rows if len(r["types"]) == 1]
    multi = [r for r in rows if len(r["types"]) > 1]
    both_pass = sum(1 for r in rows if r["field"] >= th and r["primary"] >= th)
    only_field = sum(1 for r in rows if r["field"] >= th > r["primary"])
    only_primary = sum(1 for r in rows if r["primary"] >= th > r["field"])
    out = {
        "n": n, "threshold": th,
        "review_load": {"field": load("field", rows), "primary": load("primary", rows)},
        "single_label": {"n": len(single), "field": load("field", single),
                         "primary": load("primary", single)},
        "multi_label": {"n": len(multi), "field": load("field", multi),
                        "primary": load("primary", multi)},
        "stage2_pass": {"both": both_pass, "only_field": only_field,
                        "only_primary": only_primary},
        "first_label": dict(Counter(r["types"][0] for r in rows if r["types"])),
        "mean": {"field": round(sum(r["field"] for r in rows) / n, 3),
                 "primary": round(sum(r["primary"] for r in rows) / n, 3)},
    }
    (EVAL_DIR / "production_gate_comparison.json").write_text(json.dumps(out, indent=2),
                                                              encoding="utf-8")
    pct = lambda k, m: f"{k}/{m} ({100 * k / m:.0f}%)" if m else "-"
    print(f"Distressed production rows with both confidences: {n}")
    print(f"Review load at {th:.2f} (either stage below): field gate {pct(out['review_load']['field'], n)}, "
          f"first-label gate {pct(out['review_load']['primary'], n)}")
    print(f"  single-label rows: field {pct(out['single_label']['field'], len(single))}, "
          f"first-label {pct(out['single_label']['primary'], len(single))}")
    print(f"  multi-label rows:  field {pct(out['multi_label']['field'], len(multi))}, "
          f"first-label {pct(out['multi_label']['primary'], len(multi))}")
    print(f"Stage 2 pass: both gates {both_pass}, only field {only_field}, only first-label {only_primary}")
    print(f"Mean confidence: field {out['mean']['field']}, first label {out['mean']['primary']}")
    print(f"First label: {Counter(r['types'][0] for r in rows if r['types']).most_common(6)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
