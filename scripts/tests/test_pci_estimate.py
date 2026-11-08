"""
Tests for the PCI ESTIMATE with a 90 % interval: app/pci_irc2023.pci_estimate and the estimate block of
app/pci_core.py (pci_estimate_unit, pothole inputs). Pure: no database, no GPU (critic, 2026-11-09).

  1. pci_estimate: zero-width inputs reproduce pci(); deterministic; a missing parameter -> None; inputs are
     validated and ordered; a single uncertain monotone parameter maps its 5/95 % ends onto the PCI ends.
  2. "range" (judgement range, e.g. the rut prior) holds at both ends and contains the "uniform" interval.
  3. Independence vs full correlation: the comonotone interval is wider (documents the assumption).
  4. Pothole inputs: confirmed runs are not certain (binomial lower bound), point / upper formulas.
  5. pci_estimate_unit: IRI needs >= 50 % IMU coverage; rut prior passed as "range"; a survey-calibrated value
     keeps an interval; entered values override; insufficient coverage -> None.

Usage:
    venv/Scripts/python.exe scripts/tests/test_pci_estimate.py
    venv/Scripts/python.exe -m pytest scripts/tests/test_pci_estimate.py -q
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import pci_core as C  # noqa: E402
from app import pci_irc2023 as P  # noqa: E402

TYPICAL = {"cracking": {"point": 2.8, "lo90": 0.0, "hi90": 11.7}, "ravelling": {"point": 0.12, "lo90": 0.0, "hi90": 2.6},
           "patching": {"point": 0.73, "lo90": 0.0, "hi90": 3.8}, "potholes": {"point": 0.0, "lo90": 0.0, "hi90": 0.8},
           "iri": {"point": 4.8, "lo90": 1.53, "hi90": 14.9},
           "rut_depth": {"point": 12.5, "lo90": 0.0, "hi90": 25.0, "dist": "uniform"}}


def fixed(values):
    return {p: {"point": v, "lo90": v, "hi90": v} for p, v in values.items()}


def test_zero_width_reproduces_pci_and_is_deterministic():
    for rc, vals, _ in P.EXAMPLES:
        if rc == "mdr":
            continue
        e = P.pci_estimate(rc, fixed(vals), n=300)
        want = P.pci(rc, vals)["pci"]
        for k in ("point", "median", "lo90", "hi90"):
            assert abs(e[k] - want) < 1e-9, (rc, k, e[k], want)
    a = P.pci_estimate("urban", TYPICAL, n=500)
    b = P.pci_estimate("urban", TYPICAL, n=500)
    assert a == b
    assert a["lo90"] <= a["median"] <= a["hi90"]
    assert P.pci_estimate("urban", {k: v for k, v in TYPICAL.items() if k != "iri"}) is None


def test_inputs_validated_and_ordered():
    bad = dict(TYPICAL, cracking={"point": -1.0, "lo90": 0.0, "hi90": 2.0})
    try:
        P.pci_estimate("urban", bad, n=50)
        raise AssertionError("negative input accepted")
    except ValueError:
        pass
    # lo90 above the point (or hi90 below it) is widened to include the point, never flipped
    rev = dict(TYPICAL, cracking={"point": 2.0, "lo90": 5.0, "hi90": 1.0})
    e = P.pci_estimate("urban", rev, n=200)
    assert e["lo90"] <= e["median"] <= e["hi90"]
    try:
        P.pci_estimate("mdr", fixed(dict(P.EXAMPLES[4][1])), n=50)
        raise AssertionError("MDR potholes must be Unavailable")
    except P.Unavailable:
        pass


def test_single_uncertain_parameter_maps_quantiles():
    base = {p: 0.0 for p in P.PARAMS}
    base["iri"] = 2.0
    inp = fixed(base)
    inp["cracking"] = {"point": 5.0, "lo90": 2.0, "hi90": 12.0}
    e = P.pci_estimate("urban", inp, n=4000)
    rest = sum(P.WEIGHTS[p] * P.pci_param("urban", p, base[p]) for p in P.PARAMS if p != "cracking")
    w = P.WEIGHTS["cracking"]
    assert abs(e["hi90"] - (rest + w * P.pci_param("urban", "cracking", 2.0))) < 0.15
    assert abs(e["lo90"] - (rest + w * P.pci_param("urban", "cracking", 12.0))) < 0.15
    assert abs(e["median"] - (rest + w * P.pci_param("urban", "cracking", 5.0))) < 0.15


def test_range_holds_at_both_ends():
    inp = fixed({"cracking": 1, "ravelling": 0, "potholes": 0, "patching": 0, "iri": 3.0, "rut_depth": 0})
    inp["rut_depth"] = {"point": 12.5, "lo90": 0.0, "hi90": 25.0, "dist": "range"}
    e = P.pci_estimate("urban", inp, n=1000)
    at = {r: P.pci("urban", {**{p: v["point"] for p, v in inp.items()}, "rut_depth": r})["pci"] for r in (0.0, 25.0)}
    assert abs(e["lo90"] - at[25.0]) < 1e-9 and abs(e["hi90"] - at[0.0]) < 1e-9
    assert e["range_params"] == ["rut_depth"] and e["median_assumes"]
    uni = dict(inp, rut_depth=dict(inp["rut_depth"], dist="uniform"))
    eu = P.pci_estimate("urban", uni, n=1000)
    assert e["lo90"] <= eu["lo90"] and e["hi90"] >= eu["hi90"]
    assert eu["lo90"] > at[25.0] and eu["hi90"] < at[0.0]            # uniform does NOT reach the ends
    # typical section: range is wider than uniform
    t = dict(TYPICAL, rut_depth=dict(TYPICAL["rut_depth"], dist="range"))
    a, b = P.pci_estimate("urban", TYPICAL, n=1500), P.pci_estimate("urban", t, n=1500)
    assert b["lo90"] <= a["lo90"] and b["hi90"] >= a["hi90"] and abs(a["median"] - b["median"]) < 1e-9


def test_independence_is_narrower_than_full_correlation():
    e = P.pci_estimate("urban", TYPICAL, n=1500)
    vals = []
    for i in range(1, 1000):
        u = i / 1000
        tot = 0.0
        for p in P.PARAMS:
            q = TYPICAL[p]
            x = (q["lo90"] + u * (q["hi90"] - q["lo90"]) if q.get("dist") == "uniform"
                 else P._split_normal_quantile(u, q["lo90"], q["point"], q["hi90"]))
            tot += P.WEIGHTS[p] * P.pci_param("urban", p, x)
        vals.append(tot)
    vals.sort()
    assert (vals[949] - vals[49]) > (e["hi90"] - e["lo90"]) + 5      # REVIEW_estimate.md E3: ~+15 points


def test_pothole_lower_bound_is_binomial():
    p = C.P_REAL_IF_CONFIRMED
    assert 0.70 < p < 0.78
    assert [C._binom_q05(k, p) for k in range(6)] == [0, 0, 0, 1, 1, 2]
    for k in range(1, 12):
        q = C._binom_q05(k, p)
        from math import comb
        cdf = lambda j: sum(comb(k, i) * p ** i * (1 - p) ** (k - i) for i in range(j + 1))
        assert cdf(q) >= 0.05 - 1e-12 and (q == 0 or cdf(q - 1) < 0.05)


# ---------------------------------------------------------------- pci_estimate_unit
def seg(k, probe=True, rough=True, labels=None):
    s = {"segment_id": f"s{k}", "k": k, "status": "send", "vlm_class": "distress" if labels else "normal",
         "final_labels": labels or ["Normal"], "final_severity": "Medium" if labels else "None", "length_m": 5.0,
         "run_key": "P0", "measure_m": 5.0 * k + 2.5}
    if probe:
        s["probe"] = {"Alligator Cracking": 0.05, "Longitudinal Cracking": 0.10, "Transverse Cracking": 0.08,
                      "Block Cracking": 0.02, "Ravelling": 0.05, "Patching": 0.03, "Potholes": 0.02, "Rutting": 0.05,
                      "Edge Cracking": 0.01, "Hairline Cracks": 0.05}
    if rough:
        s["roughness"] = {"duration_s": 0.625, "speed_mps": 8.0, "rms_av": 1.2, "rms_rot": 0.3, "iri_gc_uncal": 4.0,
                          "flags": [], "error": None, "jolt": 2.0}
    return s


def _unit(segs, **kw):
    old = os.environ.get("PCI_PARAM_CALIBRATION")
    os.environ["PCI_PARAM_CALIBRATION"] = str(Path(tempfile.gettempdir()) / "__no_pci_calibration__.json")
    try:
        q = C.unit_quantities(segs, 100.0, calibration=None)
        return q, C.pci_estimate_unit(segs, q, "urban", **kw)
    finally:
        if old is None:
            os.environ.pop("PCI_PARAM_CALIBRATION", None)
        else:
            os.environ["PCI_PARAM_CALIBRATION"] = old


def test_unit_full_inputs_and_rut_range():
    q, r = _unit([seg(k) for k in range(20)])
    assert r and r["estimate"] is not None, r
    assert r["inputs"]["rut_depth"]["dist"] == "range" and r["inputs"]["rut_depth"]["hi90"] == 25.0
    assert r["inputs"]["potholes"] == {"point": 0.0, "lo90": 0.0, "hi90": 0.0}
    e = r["estimate"]
    assert e["lo90"] <= e["median"] <= e["hi90"] and e["range_params"] == ["rut_depth"]


def test_unit_iri_needs_coverage():
    segs = [seg(k, rough=(k < 4)) for k in range(20)]                # IMU on 20 m of 100 m
    q, r = _unit(segs)
    assert "iri" not in r["inputs"] and r["estimate"] is None
    assert "< 50 %" in r["basis"]["iri"] and "Roughness (IRI)" in r["missing"]


def test_unit_potholes_and_entered():
    segs = [seg(k, labels=["Potholes"] if k in (3, 9, 15) else None) for k in range(20)]
    q, r = _unit(segs, impact_runs=3)
    assert q["runs"]["potholes"] == 3
    pin = r["inputs"]["potholes"]
    u_lo, u_pt, u_hi = C.UNITS_PER_POTHOLE
    # IRC units = real potholes x units per pothole; 3 jolt-confirmed runs: lower = binomial 5 % of 3 at 0.72 = 1
    assert pin["lo90"] == C._binom_q05(3, C.REAL_IF_JOLT[0]) * u_lo
    assert abs(pin["point"] - 3 * C.REAL_IF_JOLT[1] / C.RUN_RECALL_POINT * u_pt) < 1e-9
    assert pin["hi90"] >= 3 * C.REAL_IF_JOLT[2] / C.RUN_RECALL_LOW * u_hi
    q, r = _unit(segs, impact_runs=0)                                # no jolt: mostly false alarms (p 0.09)
    assert r["inputs"]["potholes"]["lo90"] == 0.0
    assert abs(r["inputs"]["potholes"]["point"] - 3 * C.REAL_IF_NO_JOLT[1] / C.RUN_RECALL_POINT * u_pt) < 1e-9
    q, r = _unit(segs, entered={"potholes": 10, "rut_depth": 4})     # potholes per km -> 1 per 100 m
    assert r["inputs"]["potholes"] == {"point": 1.0, "lo90": 1.0, "hi90": 1.0}
    assert r["inputs"]["rut_depth"]["lo90"] == r["inputs"]["rut_depth"]["hi90"] == 4.0
    assert "dist" not in r["inputs"]["rut_depth"]


def test_unit_survey_calibration_keeps_interval():
    d = Path(tempfile.mkdtemp(prefix="pci_est_"))
    cal = {"schema": 2, "edition": "IRC:82-2023", "probe_fp": "0123456789abcdef", "fitted_on": {"half": "calibration"},
           "params": {"cracking": {"source": "probe", "threshold": 0.05, "a": 0.0, "b": 4.0, "evidence": "test"}}}
    (d / "cal.json").write_text(json.dumps(cal), encoding="utf-8")
    segs = [seg(k) for k in range(20)]
    old = os.environ.get("PCI_PARAM_CALIBRATION")
    os.environ["PCI_PARAM_CALIBRATION"] = str(d / "cal.json")
    try:
        c = C.current_calibration()
        q = C.unit_quantities(segs, 100.0, calibration=c)
        r = C.pci_estimate_unit(segs, q, "urban")
    finally:
        if old is None:
            os.environ.pop("PCI_PARAM_CALIBRATION", None)
        else:
            os.environ["PCI_PARAM_CALIBRATION"] = old
    ci = r["inputs"]["cracking"]
    assert abs(ci["point"] - 4.0) < 1e-9 and ci["hi90"] > ci["point"]       # not a zero-width interval
    assert ci["lo90"] <= ci["point"] <= ci["hi90"]


def test_unit_insufficient():
    segs = [dict(seg(k), status="skip", vlm_class="not_sent", final_labels=[]) for k in range(20)]
    q, r = _unit(segs)
    assert r is None


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
