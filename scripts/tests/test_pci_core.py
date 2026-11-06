"""
Tests for app/pci_core.py (IRC:82-2015 rating arithmetic; pure, no database, no GPU).

  1. The Appendix-4 worked example (Table 5.1): per-row ratings 1.2/1.2/3.0/2.1/1.5/1.2/1.2 (IRC prints 1.3 for
     rut depth, not reproducible by any single rule: RESEARCH §1.5), weighted values, final 1.3 Fair.
  2. Every band boundary of every table, with the tolerance (x exactly at L / U, just inside, just outside).
  3. Truncation is float-safe (2 - 4/5 = 1.2, not 1.1), ratings stay inside their band's range.
  4. Rounding is Decimal half-up (0.75*1.5 -> 1.1, 0.75*1.2 -> 0.9, 1.25 -> 1.3).
  5. Bracket: not-assessed parameters at r = 1 / r = 3; one class only when both ends agree; Poor reachable.
  6. Quantities: coverage rules, skip-corrected E_lo/E_hi, runs (one per run of consecutive positives),
     not-assessed parameters never 0 %, nominal pothole area.

Usage:
    venv/Scripts/python.exe scripts/tests/test_pci_core.py
    venv/Scripts/python.exe -m pytest scripts/tests/test_pci_core.py -q
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import pci_core as P  # noqa: E402


def seg(k, labels=None, status="send", sev="Medium", run_key="P0", length=5.0, measure=None):
    if status in ("skip", "not_covered"):
        vc = "not_sent" if status == "skip" else "not_covered"
        labels = []
    elif labels is None:
        vc, labels = "normal", ["Normal"]
    elif labels == "unknown":
        vc, labels = "unknown", []
    else:
        vc = "distress"
    return {"segment_id": f"s{k}", "k": k, "status": status, "vlm_class": vc, "final_labels": labels,
            "final_severity": sev if vc == "distress" else "None", "length_m": length, "run_key": run_key,
            "measure_m": measure if measure is not None else 5.0 * k}


def test_app4_worked_example():
    v = dict(cracking=9, ravelling=8, potholes=0, shoving=0.09, patching=5, settlement=4, rut_depth=9)
    r = P.app4_final("highway", v)
    got = {x["param"]: (x["rating"], x["weighted"]) for x in r["rows"]}
    assert got == {"cracking": (1.2, 1.2), "ravelling": (1.2, 0.9), "potholes": (3.0, 1.5), "shoving": (2.1, 2.1),
                   "patching": (1.5, 1.1), "settlement": (1.2, 0.9), "rut_depth": (1.2, 1.2)}, got
    assert r["final"] == 1.3 and r["class"] == "Fair"
    # with IRC's printed 1.3 for rut depth the final is still 1.3 Fair
    w = [x["weighted"] for x in r["rows"]][:-1] + [1.3]
    assert P.mean1(w) == 1.3


def test_band_boundaries():
    for cat, tab in P.TABLES.items():
        for p, (L, U) in tab["params"].items():
            if cat == "urban" and p == "potholes":
                assert P.band(cat, p, 0.0) == "Good"
                assert P.band(cat, p, 1e-12) == "Good"          # within tolerance of NIL
                assert P.band(cat, p, 0.001) == "Fair"
                assert P.band(cat, p, 0.5) == "Fair"
                assert P.band(cat, p, 0.5 + 1e-12) == "Fair"
                assert P.band(cat, p, 0.5001) == "Poor"
                continue
            assert P.band(cat, p, L) == "Fair", (cat, p, L)
            assert P.band(cat, p, L - 1e-12) == "Fair", "tolerance at L"
            assert P.band(cat, p, L - 1e-6) == "Good"
            assert P.band(cat, p, U) == "Fair", (cat, p, U)
            assert P.band(cat, p, U + 1e-12) == "Fair"
            assert P.band(cat, p, U + 1e-6) == "Poor"
            assert P.band(cat, p, 0.0) == "Good"
            # e.g. 5 % computed as 100*1/20 in floating point
            x = 100.0 * (L / 100.0 * 20) / 20 if L else 0.0
            assert P.band(cat, p, x) in ("Fair", "Good")


def test_rating_values_in_band_and_truncation():
    assert P.rating_value("highway", "cracking", 9.0) == 1.2          # 2 - 4/5 = 1.1999.. in float
    assert P.trunc1(2 - (9 - 5) / 5) == 1.2
    assert P.trunc1(1.5555) == 1.5 and P.trunc1(2.0999999999) == 2.1   # 1e-9 guard
    for cat, tab in P.TABLES.items():
        for p, (L, U) in tab["params"].items():
            for i in range(0, 401):
                x = i * max(U, 1.0) * 1.5 / 400
                r, b = P.rating_value(cat, p, x), P.band(cat, p, x)
                lo, hi = {"Poor": (1.0, 1.0), "Fair": (1.1, 2.0), "Good": (2.1, 3.0)}[b]
                assert lo - 1e-12 <= r <= hi + 1e-12, (cat, p, x, r, b)
                assert abs(r * 10 - round(r * 10)) < 1e-9
    assert P.rating_value("urban", "potholes", 0.0) == 3.0
    assert P.rating_value("urban", "potholes", 0.25) == 1.5
    assert P.rating_value("urban", "potholes", 0.5) == 1.1           # 2 - 1 = 1.0 clamped to the Fair range
    assert P.rating_value("urban", "potholes", 5.0) == 1.0


def test_rounding():
    assert P.weighted(0.75, 1.5) == 1.1
    assert P.weighted(0.75, 1.2) == 0.9
    assert P.weighted(0.5, 2.5) == 1.3        # 1.25 half-up
    assert P.mean1([1.25]) == 1.3 and P.mean1([1.0, 1.1]) == 1.1   # 1.05 -> 1.1
    assert P.final_class(1.0) == "Poor" and P.final_class(1.04) == "Poor" and P.final_class(1.05) == "Fair"
    assert P.final_class(2.0) == "Fair" and P.final_class(2.05) == "Good" and P.final_class(0.8) == "Poor"
    assert P.class_range("Good", "Poor") == "Poor–Good" and P.class_range("Fair", "Fair") == "Fair"


def _q(n_seg, positives, skip=0, label="Potholes"):
    segs = [seg(k, [label] if k in positives else None) for k in range(n_seg)]
    segs += [seg(n_seg + j, status="skip") for j in range(skip)]
    return P.unit_quantities(segs, 5.0 * (n_seg + skip))


def test_bracket_and_poor_reachable():
    # every segment with potholes + alligator: worst possible assessed values
    segs = [seg(k, ["Potholes", "Alligator Cracking"]) for k in range(20)]
    q = P.unit_quantities(segs, 100.0)
    r = P.rate_unit(q, "urban")
    f = r["final"]
    # urban lo: cracking 1*1 + ravelling .75*1 + potholes .5*1 + settlement .75*1 + rut 1*1 -> 4.0/5 = 0.8
    assert f["lo"] == 0.8 and f["class_lo"] == "Poor", f
    # hi: assessed worst (1.0, 0.5) + unassessed at 3 (2.3, 2.3, 3.0) = 9.1/5 = 1.82 -> 1.8 Fair
    assert f["hi"] == 1.8 and f["class"] == "Poor–Fair", f
    assert set(r["not_assessed"]) == {"ravelling", "settlement", "rut_depth"}
    assert "not assessed" in f["label"]
    # clean road: Good only possible at hi end
    q = P.unit_quantities([seg(k) for k in range(20)], 100.0)
    f = P.rate_unit(q, "urban")["final"]
    assert f["hi"] == 2.4 and f["class_hi"] == "Good" and f["class_lo"] in ("Fair", "Poor"), f
    # not-assessed parameters are never a number
    row = {x["param"]: x for x in P.rate_unit(q, "highway")["params"]}
    assert row["patching"]["band"] == "not assessed" and "E_lo" not in row["patching"]
    assert row["cracking"]["band"] == "Good" and row["cracking"]["E_lo"] == 0.0


def test_quantities_and_skip_correction():
    q = _q(16, {3}, skip=4)                  # 16 classified (1 pothole), 4 skipped -> D = 100 m
    assert q["classified_m"] == 80 and q["skip_m"] == 20 and q["denominator_m"] == 100
    e = q["extent"]["potholes"]
    assert abs(e["lo"] - 5.0) < 1e-12 and abs(e["hi"] - (5 + 0.133 * 20)) < 1e-9
    assert abs(e["uncorrected"] - 6.25) < 1e-12
    assert q["coverage_status"] == "rated" and q["rated_80"] and q["rated_50"]
    assert q["runs"]["potholes"] == 1 and abs(q["runs_per_km"]["potholes"] - 10.0) < 1e-9
    assert abs(q["pothole_nominal_area_pct"] - 100 * 0.0314 / (100 * 3.5)) < 1e-12
    # coverage rules
    segs = [seg(k) for k in range(14)] + [seg(14 + j, status="not_covered") for j in range(6)]
    q = P.unit_quantities(segs, 100.0)
    assert abs(q["coverage"] - 0.7) < 1e-12 and q["coverage_status"] == "low_coverage"
    assert q["rated_50"] and not q["rated_80"]
    q = P.unit_quantities([seg(k) for k in range(9)], 100.0)
    assert q["coverage_status"] == "insufficient"
    assert P.rate_unit(q, "urban")["rated"] is False
    q = P.unit_quantities([seg(k, status="skip") for k in range(20)], 100.0)
    assert q["coverage_status"] == "insufficient"          # nothing classified
    # unanswered counts as imaged, not classified
    q = P.unit_quantities([seg(0, "unknown")] + [seg(k) for k in range(1, 20)], 100.0)
    assert q["unanswered_m"] == 5 and q["classified_m"] == 95


def test_runs():
    segs = [seg(k, ["Potholes"] if k in (1, 2, 3, 7, 9, 10) else None) for k in range(12)]
    rs = P.runs(segs, ["Potholes"])
    assert [(r["k_first"], r["n"]) for r in rs] == [(1, 3), (7, 1), (9, 2)]
    assert P.run_length_hist(rs) == {"1": 1, "2": 1, "3": 1}
    # a k jump or another pass splits a run; a skipped segment ends it
    segs = [seg(1, ["Potholes"]), seg(2, ["Potholes"], run_key="P1"), seg(4, ["Potholes"]),
            seg(5, status="skip"), seg(6, ["Potholes"])]
    assert len(P.runs(segs, ["Potholes"])) == 4
    # crack sub-split
    q = P.unit_quantities([seg(0, ["Alligator Cracking"]), seg(1, ["Transverse Cracking"])] +
                          [seg(k) for k in range(2, 20)], 100.0)
    assert q["extent"]["cracking"]["affected_m"] == 10 and q["extent"]["cracking_alligator"]["affected_m"] == 5
    assert q["extent"]["cracking_linear"]["affected_m"] == 5


def test_treatments():
    segs = [seg(0, ["Potholes"], sev="High"), seg(1, ["Bleeding"], sev="High"), seg(2, ["Longitudinal Cracking"],
                                                                                 sev="Low")]
    q = P.unit_quantities(segs, 15.0)
    t = {x["label"]: x for x in P.distress_treatments(q)}
    assert "straightway" in t["Potholes"]["treatment"] and t["Potholes"]["validated"]
    assert t["Bleeding"]["note"] == "label unvalidated" and "Mill out" in t["Bleeding"]["treatment"]
    assert t["Longitudinal Cracking"]["treatment"].startswith("Crack sealing")
    assert "investigate strengthening" in P.class_treatment("Poor–Fair", "urban")
    assert "Routine maintenance" in P.class_treatment("Good", "urban")
    assert "30-40 BC/SDBC" in P.class_treatment("Fair", "urban")
    assert P.class_treatment(None, "urban").startswith("Insufficient")


TESTS = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    failures = []
    for t in TESTS:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception:
            failures.append(t.__name__)
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - len(failures)}/{len(TESTS)} tests passed")
    sys.exit(1 if failures else 0)
