"""
Tests for app/pci_irc2023.py (IRC:82-2023 Appendix-2 PCI equations) and the 2023 section of app/pci_core.py.
Pure arithmetic: no database, no GPU.

  1. Transcription: every equation re-typed here INDEPENDENTLY from the printed page images
     (C:\\GeoAI\\pci\\irc82_2023\\appendix2_pages\\p076-p088.png, read 2026-11-08 by the critic) and compared with
     the module at many points (catches a changed digit).
  2. Printed curves: points read off the printed curves (log axes) within 5 PCI points.
  3. Design check: every usable roughness equation gives PCI ~80 at the Table 5.1-5.3 Excellent-to-Good IRI
     limit and ~40 at the Satisfactory-to-Fair limit (the property the rural SD / MSS readings fail).
  4. Worked examples A.1-C.3: highway within 0.6; urban = the known, documented gap; MDR printed PCI inside
     our range.
  5. Classes (Table 5.5 boundaries), weights, Unavailable equations, pci_range with unknown / ranged values.
  6. Edge cases: x = 0, IRI < 2, IRI below a rising curve's peak, huge values (no overflow), negative / NaN,
     extents > 100 %, fine-grid scan (every usable equation finite and inside [0, 100] after clamping).
  7. pci_core.pci_unit: source priority, single PCI only on the IRC scale, Unavailable bracketed, probe
     calibration without scores, validate_calibration rejects malformed files.

Usage:
    venv/Scripts/python.exe scripts/tests/test_pci_irc2023.py
    venv/Scripts/python.exe -m pytest scripts/tests/test_pci_irc2023.py -q
"""
from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import pci_core as C  # noqa: E402
from app import pci_irc2023 as P  # noqa: E402


def poly(c, x):
    return sum(ci * x ** (len(c) - 1 - i) for i, ci in enumerate(c))


# ---- re-typed from the page images (critic's own reading) ----
REF = {
    ("highway", "cracking"): lambda x: 7231 / (x ** 2 - 0.737 * x + 73.09),
    ("highway", "rut_depth"): lambda x: 7231 / (x ** 2 - 0.737 * x + 73.09),
    ("highway", "ravelling"): lambda x: 52.92 * math.exp(-0.02525 * x) + 44.1 * math.exp(-0.2899 * x),
    ("highway", "patching"): lambda x: 52.92 * math.exp(-0.02525 * x) + 44.1 * math.exp(-0.2899 * x),
    ("highway", "potholes"): lambda n: poly((-80.32, 1129, -3524, 3566), n) / poly((1, 5.791, -28.67, 35.72), n),
    ("highway", "iri"): lambda i: 100 / (i ** 1.91 - 3.542 * i + 4.315),
    ("mdr", "cracking"): lambda x: poly((-0.5662, 85.14, -3377, 5.182e4), x) / poly((1, -29.31, 523.4), x),
    ("mdr", "ravelling"): lambda x: poly((-0.5662, 85.14, -3377, 5.182e4), x) / poly((1, -29.31, 523.4), x),
    ("mdr", "rut_depth"): lambda x: poly((-0.5662, 85.14, -3377, 5.182e4), x) / poly((1, -29.31, 523.4), x),
    ("mdr", "patching"): lambda x: poly((-1.723e-3, -8.273e-2, 21.39, 419.1), x) / (x + 4.267),
    ("urban", "cracking"): lambda x: 100 * math.exp(-0.0534 * x) - 0.006641 * x,
    ("urban", "ravelling"): lambda x: 7231 / (x ** 2 - 0.737 * x + 73.09),
    ("urban", "patching"): lambda x: 7231 / (x ** 2 - 0.737 * x + 73.09),
    ("urban", "rut_depth"): lambda x: 7231 / (x ** 2 - 0.737 * x + 73.09),
    ("urban", "potholes"): lambda n: poly((-130.6, 2112, -9550, 14390), n) / poly((1, 6.619, -66.73, 144.1), n),
    ("urban", "iri"): lambda i: 100 / (i ** 1.91 - 3.542 * i + 4.315),
}
REF_IRI = {
    ("mdr", "SD"): ((-0.09535, 4.235, -74.43, 642.4, -2625, 4034), (1, -9.065, 20.71)),
    ("mdr", "OGPC"): ((-4.842, 121.5, -875.9, 2029), (1, -8.389, 19.85)),
    ("mdr", "MSS"): ((-2.686, 64.96, -437.2, 1046, -596.2), (1, -9.766, 31.09, -30.94)),
    ("mdr", "SDBC"): ((-4.468, 106.2, -694.7, 1507), (1, -6.778, 14.51)),
    ("mdr", "BC"): ((-2.52, 55.18, -292.4, 476.4, -5.565), (1, -7.458, 17.77, -12.09)),
    ("rural", "OGPC"): ((-4.02, 98.33, -670.6, 1444), (1, -8.215, 17.79)),
    ("rural", "SDBC"): ((25.16, 48.45), (1, -5.079, 7.595)),
}
# rural SD / MSS as printed (kept as candidates, not used)
REF_RURAL_PRINTED = {
    "SD": ((-25.68, 646.2, -4465, 9408), (1, -9.272, 21.13, 3.037)),          # IRI^3 reading of the garbled term
    "MSS_printed": ((-24.69, 593.2, -3663, 6786), (1, 7.173, -9.017, 15.43)),
}


def clamp(v):
    return max(0.0, min(100.0, v))


def test_transcription_digit_by_digit():
    xs = [0, 0.05, 0.3, 1, 2.6, 5, 7.5, 10, 14, 20, 35, 60, 99]
    for (rc, p), f in REF.items():
        for x in xs:
            if p == "iri":
                x = max(2.0, x)
            if p == "potholes" and x > 30:
                continue
            assert abs(P.pci_param(rc, p, x) - clamp(f(x))) < 1e-9, (rc, p, x)
    for (rc, s), (num, den) in REF_IRI.items():
        f = P.EQUATIONS[rc]["iri"][s]
        for i in [2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 8, 10, 12, 15, 18]:
            assert abs(f(i) - poly(num, i) / poly(den, i)) < 1e-9, (rc, s, i)
    for k, (num, den) in REF_RURAL_PRINTED.items():
        f = P.IRI_RURAL_CANDIDATES[k]
        for i in [2, 3, 4, 5, 6, 9]:
            assert abs(f(i) - poly(num, i) / poly(den, i)) < 1e-9, (k, i)
    # rural crack / ravel / rut / patch = the MDR equations (Table A2.2 Part B prints the same)
    for p in ("cracking", "ravelling", "rut_depth", "patching"):
        for x in xs:
            assert P.pci_param("rural", p, x) == P.pci_param("mdr", p, x)
    assert P.WEIGHTS == {"iri": 0.40, "potholes": 0.16, "rut_depth": 0.14, "cracking": 0.12, "ravelling": 0.10,
                         "patching": 0.08}
    assert abs(sum(P.WEIGHTS.values()) - 1.0) < 1e-12


# points read off the printed curves (critic, 2026-11-08; log x axes); tolerance 5 PCI points
CURVE_POINTS = [
    ("highway", "cracking", None, 1, 99), ("highway", "cracking", None, 10, 41),
    ("highway", "ravelling", None, 1, 85), ("highway", "ravelling", None, 10, 43),
    ("highway", "potholes", None, 1, 80), ("highway", "potholes", None, 3, 34), ("highway", "potholes", None, 10, 0),
    ("highway", "iri", None, 2, 100), ("highway", "iri", None, 3, 53), ("highway", "iri", None, 4, 23),
    ("mdr", "cracking", None, 10, 80), ("mdr", "cracking", None, 30, 21), ("mdr", "cracking", None, 100, 0),
    ("mdr", "patching", None, 0.1, 96.5), ("mdr", "patching", None, 1, 83), ("mdr", "patching", None, 10, 43),
    ("mdr", "iri", "SD", 3.0, 100), ("mdr", "iri", "SD", 4.5, 50), ("mdr", "iri", "SD", 5.0, 37),
    ("mdr", "iri", "SD", 7.0, 28), ("mdr", "iri", "OGPC", 5.0, 28), ("mdr", "iri", "OGPC", 7.0, 19),
    ("mdr", "iri", "MSS", 4.5, 32), ("mdr", "iri", "SDBC", 3.76, 50), ("mdr", "iri", "SDBC", 5.0, 23),
    ("mdr", "iri", "BC", 3.03, 70), ("mdr", "iri", "BC", 3.86, 31), ("mdr", "iri", "BC", 5.0, 25),
    ("mdr", "iri", "SD", 15.0, 0), ("mdr", "iri", "BC", 15.0, 0),
    ("urban", "cracking", None, 10, 58), ("urban", "potholes", None, 1, 80), ("urban", "potholes", None, 3, 36),
    ("urban", "ravelling", None, 10, 41), ("urban", "iri", None, 3, 53),
]


def test_printed_curve_points():
    for rc, p, s, x, y in CURVE_POINTS:
        v = P.pci_param(rc, p, x, s)
        assert abs(v - y) <= 5.0, (rc, p, s, x, v, y)


def test_iri_equations_hit_table5_limits():
    """Every usable roughness equation: ~80 at the EG limit, ~40 at the SF limit (Tables 5.1-5.3)."""
    for rc in ("mdr", "rural"):
        for s, f in P.EQUATIONS[rc]["iri"].items():
            if getattr(f, "unusable", None):
                continue
            a, b = P.IRI_RANGES[rc][s]
            tol = 6.0 if (rc, s) == ("rural", "SDBC") else 4.0          # MDR SD 36.7 at 4.98; rural SDBC 74.7 / 44.5
            assert abs(P.pci_param(rc, "iri", a, s) - 80) <= tol, (rc, s, a)
            assert abs(P.pci_param(rc, "iri", b, s) - 40) <= tol, (rc, s, b)
    for rc in ("highway", "urban"):
        assert abs(P.pci_param(rc, "iri", 2.55) - 80) <= 3 and abs(P.pci_param(rc, "iri", 3.30) - 40) <= 3
    # the two rural readings that fail it (why they are Unavailable)
    sd = P.IRI_RURAL_CANDIDATES["SD"]
    assert sd(4.27) < 65 and sd(4.98) < 20
    mss = P.IRI_RURAL_CANDIDATES["MSS_printed"]
    assert mss(2.0) < 50 and mss(3.79) < 1
    alt = P.IRI_RURAL_CANDIDATES["MSS_signs_flipped"]
    assert abs(alt(3.79) - 80) < 3 and abs(alt(4.27) - 40) < 3


def test_worked_examples():
    got = {}
    for rc, vals, printed in P.EXAMPLES:
        if rc == "mdr":
            r = P.pci_range(rc, vals)                       # potholes Unavailable -> bracketed
            assert r["missing"] == ["potholes"] and r["single"] is None
            assert r["lo"] - 1e-9 <= printed <= r["hi"] + 1e-9, (vals, r["lo"], r["hi"], printed)
            continue
        v = P.pci(rc, vals)["pci"]
        got.setdefault(rc, []).append((v, printed))
        assert abs(P.pci_range(rc, vals)["single"] - v) < 1e-9
    for v, printed in got["highway"]:
        assert abs(v - printed) <= 0.6, (v, printed)
    # urban: the documented gap (IRC82_2023.md §3); a change here means an urban equation changed
    for (v, _), want in zip(got["urban"], (94.50, 46.82, 27.63)):
        assert abs(v - want) < 0.01, (v, want)


def test_classes_and_recommendations():
    cases = [(100, "Excellent"), (90.01, "Excellent"), (90.0, "Good"), (80.0, "Satisfactory"), (60.0, "Fair"),
             (40.0, "Poor"), (20.01, "Poor"), (20.0, "Fail"), (0.0, "Fail")]
    for v, name in cases:
        assert P.pci_class(v) == name, (v, P.pci_class(v))
    assert set(P.RECOMMENDATION) == {"Excellent", "Good", "Satisfactory", "Fair", "Poor", "Fail"}
    # example recommendations as printed (A.2 / C.2: Minor rehabilitation; A.3: Major; A.1: Routine)
    assert P.pci("highway", P.EXAMPLES[0][1])["recommendation"] == "Routine maintenance"
    assert P.pci("highway", P.EXAMPLES[1][1])["recommendation"].startswith("Minor rehabilitation")
    assert P.pci("highway", P.EXAMPLES[2][1])["recommendation"].startswith("Major rehabilitation")


def test_unavailable_equations():
    for rc in ("mdr", "rural"):
        try:
            P.pci_param(rc, "potholes", 2)
            raise AssertionError("MDR/rural pothole equation must be Unavailable")
        except P.Unavailable:
            pass
    for s in ("SD", "MSS", "BC"):
        try:
            P.pci_param("rural", "iri", 3.0, s)
            raise AssertionError(f"rural {s} IRI must be Unavailable")
        except P.Unavailable:
            pass
    try:
        P.pci_param("mdr", "iri", 3.0, "XYZ")
        raise AssertionError("unknown surface must be a ValueError")
    except P.Unavailable:
        raise AssertionError("unknown surface is not Unavailable")
    except ValueError:
        pass
    r = P.pci_range("rural", {"iri": 3.0, "cracking": 1.0}, "MSS")
    assert "iri" in r["missing"] and r["single"] is None
    assert abs(r["hi"] - r["lo"] - 100 * (1 - 0.12)) < 1e-6
    # highway / urban IRI ignores the surface (one equation)
    assert P.pci_param("highway", "iri", 3, "SD") == P.pci_param("highway", "iri", 3)


def test_pci_range_semantics():
    r = P.pci_range("highway", {})
    assert r["lo"] == 0 and abs(r["hi"] - 100) < 1e-9 and r["single"] is None and len(r["missing"]) == 6
    full = dict(P.EXAMPLES[1][1])
    r = P.pci_range("highway", full)
    assert r["single"] is not None and abs(r["lo"] - r["hi"]) < 1e-9
    # ranged value: lo = best of the interval
    r = P.pci_range("highway", {**full, "cracking": (2, 8)})
    assert r["single"] is None and r["lo"] < r["hi"]
    row = next(x for x in r["rows"] if x["param"] == "cracking")
    assert abs(row["pci_best"] - P.pci_param("highway", "cracking", 2)) < 1e-9
    assert abs(row["pci_worst"] - P.pci_param("highway", "cracking", 8)) < 1e-9
    # non-monotone interior (MDR SD roughness rises between 4.86 and 5.31): the interval minimum is inside it
    w, b = P.pci_param_bounds("mdr", "iri", 4.7, 5.6, "SD")
    ends = [P.pci_param("mdr", "iri", x, "SD") for x in (4.7, 5.6)]
    inner_min = min(P.pci_param("mdr", "iri", 4.7 + 0.9 * i / 1000, "SD") for i in range(1001))
    assert w <= min(ends) and abs(w - inner_min) < 0.05, (w, ends, inner_min)


def test_edge_cases():
    for rc in P.ROAD_CLASSES:
        for p in P.PARAMS:
            for x in (0.0, 1e-12, 100.0, 1e6, 1e200):
                try:
                    v = P.pci_param(rc, p, x)
                except P.Unavailable:
                    continue
                assert 0.0 <= v <= 100.0 and math.isfinite(v), (rc, p, x, v)
            for bad in (-1.0, float("nan"), float("inf")):
                try:
                    P.pci_param(rc, p, bad)
                    raise AssertionError((rc, p, bad))
                except P.Unavailable:
                    raise AssertionError("bad value must be ValueError")
                except ValueError:
                    pass
    # x = 0 values
    assert abs(P.pci_param("highway", "cracking", 0) - 7231 / 73.09) < 1e-9
    assert abs(P.pci_param("highway", "ravelling", 0) - 97.02) < 1e-9
    assert abs(P.pci_param("mdr", "cracking", 0) - 51820 / 523.4) < 1e-9
    assert P.pci_param("urban", "cracking", 0) == 100.0
    # IRI below 2 = at 2; below a rising curve's peak = the peak (never better at a rougher IRI)
    for rc, s in (("highway", None), ("mdr", "BC"), ("rural", "OGPC"), ("rural", "SDBC")):
        assert P.pci_param(rc, "iri", 0.5, s) == P.pci_param(rc, "iri", 2.0, s)
    for rc, s in (("rural", "OGPC"), ("rural", "SDBC"), ("mdr", "OGPC")):
        vals = [P.pci_param(rc, "iri", 2 + 0.01 * i, s) for i in range(0, 300)]
        assert all(b <= a + 1e-9 for a, b in zip(vals, vals[1:])), (rc, s)
    # extents above 100 % are scored as 100 %
    assert P.pci_param("urban", "cracking", 150) == P.pci_param("urban", "cracking", 100)
    assert P.potholes_number(0.5) == 5.0


def test_fine_grid_scan():
    """No pole / exception inside the evaluated domain of any usable equation; all values clamped."""
    for rc in P.ROAD_CLASSES:
        for p in P.PARAMS:
            surfaces = list(P.EQUATIONS[rc]["iri"]) if (p == "iri" and isinstance(P.EQUATIONS[rc]["iri"], dict)) \
                else [None]
            for s in surfaces:
                top = {"iri": 20.0, "potholes": 50.0}.get(p, 100.0)
                n = 4000
                try:
                    vals = [P.pci_param(rc, p, top * i / n, s) for i in range(n + 1)]
                except P.Unavailable:
                    continue
                assert all(0 <= v <= 100 for v in vals)


# ---------------------------------------------------------------- pci_core 2023 section
def seg(k, labels=None, status="send", length=5.0, probe=None, iri=None):
    if status == "skip":
        vc, labels = "not_sent", []
    elif labels is None:
        vc, labels = "normal", ["Normal"]
    else:
        vc = "distress"
    s = {"segment_id": f"s{k}", "k": k, "status": status, "vlm_class": vc, "final_labels": labels,
         "final_severity": "Medium" if vc == "distress" else "None", "length_m": length, "run_key": "P0",
         "measure_m": 5.0 * k + 2.5}
    if probe is not None:
        s["probe"] = probe
    if iri is not None:
        s["iri"] = iri
    return s


def unit(segs, cal=None):
    q = C.unit_quantities(segs, 100.0, calibration=cal)
    return q


def test_pci_unit_uncalibrated_labels():
    segs = [seg(k, ["Longitudinal Cracking"] if k < 4 else (["Potholes"] if k == 10 else None)) for k in range(20)]
    q = unit(segs)
    out = C.pci_unit(q, "urban", calibration=None)
    rows = {r["param"]: r for r in out["params"]}
    assert rows["cracking"]["source"] == "labels" and rows["cracking"]["value_lo"] == 0.0     # best case 0 % area
    assert abs(rows["cracking"]["value_hi"] - 20.0) < 1e-9
    assert rows["potholes"]["source"] == "labels" and rows["potholes"]["value_point"] == 1.0
    for p in ("ravelling", "patching", "rut_depth", "iri"):
        assert rows[p]["source"] == "not_assessed" and not rows[p]["known"]
    pci = out["pci"]
    assert pci["single"] is None and "not assessed" in pci["single_note"]
    assert 0 <= pci["lo"] <= pci["hi"] <= 100
    # requester enters the four the images cannot give: still no single PCI (labels are not on the IRC scale)
    ent = {"ravelling": 1.0, "patching": 0.0, "rut_depth": 3.0, "iri": 2.8}
    out = C.pci_unit(q, "urban", entered=ent, calibration=None)
    assert out["pci"]["single"] is None and "uncalibrated" in out["pci"]["single_note"]
    # all six entered: the single PCI = the equations
    ent6 = dict(P.EXAMPLES[7][1])
    out = C.pci_unit(q, "urban", entered=ent6, calibration=None)
    # entered potholes are per km, scaled to the unit's length (100 m here)
    assert abs(out["pci"]["single"] - P.pci("urban", {**ent6, "potholes": ent6["potholes"] * 0.1})["pci"]) < 1e-9
    assert {r["source"] for r in out["params"]} == {"entered"}
    assert out["pci"]["single_note"] is None


def test_pci_unit_unavailable_and_insufficient():
    segs = [seg(k) for k in range(20)]
    q = unit(segs)
    # rural + BC + an entered IRI: bracketed, not a crash
    out = C.pci_unit(q, "rural", surface="BC", entered={"iri": 3.0}, calibration=None)
    row = next(r for r in out["params"] if r["param"] == "iri")
    assert row["known"] is False and "no roughness equation" in row["evidence"]
    # MDR potholes: Unavailable -> bracketed
    out = C.pci_unit(q, "mdr", calibration=None)
    row = next(r for r in out["params"] if r["param"] == "potholes")
    assert row["known"] is False and "Potholes" in out["pci"]["missing"]
    # insufficient coverage
    q2 = unit([seg(k, status="skip") for k in range(20)])
    assert C.pci_unit(q2, "urban", calibration=None)["pci"] is None


def _cal(params):
    return C.validate_calibration({"schema": 2, "edition": "IRC:82-2023", "probe_fp": "0123456789abcdef",
                                   "fitted_on": {"half": "calibration"}, "params": params})


def test_pci_unit_probe_and_priority():
    cal = _cal({"ravelling": {"source": "probe", "threshold": 0.5, "a": 0.0, "b": 20.0, "evidence": "x"},
                "cracking": {"source": "labels", "b": 0.5, "evidence": "y"}})
    segs = [seg(k, ["Transverse Cracking"] if k < 10 else None, probe={"Ravelling": 0.9 if k < 5 else 0.1})
            for k in range(20)]
    q = C.unit_quantities(segs, 100.0, calibration=cal)
    out = C.pci_unit(q, "highway", calibration=cal)
    rows = {r["param"]: r for r in out["params"]}
    assert rows["ravelling"]["source"] == "probe" and abs(rows["ravelling"]["value_point"] - 5.0) < 1e-9
    assert rows["cracking"]["source"] == "labels_calibrated" and abs(rows["cracking"]["value_hi"] - 25.0) < 1e-9
    assert rows["cracking"]["value_lo"] == 25.0          # calibrated: no best-case-0 widening
    # entered beats probe
    out = C.pci_unit(q, "highway", entered={"ravelling": 0.0}, calibration=cal)
    assert next(r for r in out["params"] if r["param"] == "ravelling")["source"] == "entered"
    # probe calibrated but no scores on the segments: not assessed, with the reason; cracking never falls back
    cal2 = _cal({"cracking": {"source": "probe", "threshold": 0.5, "b": 30.0}})
    segs2 = [seg(k, ["Transverse Cracking"]) for k in range(20)]
    q = C.unit_quantities(segs2, 100.0, calibration=cal2)
    row = next(r for r in C.pci_unit(q, "highway", calibration=cal2)["params"] if r["param"] == "cracking")
    assert row["source"] == "not_assessed" and "less than 50 %" in row["evidence"]
    # calibrated roughness (q["roughness"], from app.pci_iri via roughness_value) is the IRI source
    q = C.unit_quantities([seg(k) for k in range(20)], 100.0, calibration=None)
    q["roughness"] = {"value": 3.0, "measured_m": 100.0}
    row = next(r for r in C.pci_unit(q, "urban", calibration=None)["params"] if r["param"] == "iri")
    assert row["source"] == "roughness" and abs(row["pci_point"] - P.pci_param("urban", "iri", 3.0)) < 1e-9


def test_validate_calibration_rejects():
    good = {"schema": 2, "edition": "IRC:82-2023", "probe_fp": "0123456789abcdef", "params": {}}
    assert C.validate_calibration(good)["params"] == {}
    bad = [
        {**good, "schema": 1}, {**good, "edition": "IRC:82-2015"}, {**good, "probe_fp": "XYZ"},
        {**good, "params": {"iri": {"source": "probe", "threshold": 0.5}}},
        {**good, "params": {"ravelling": {"source": "labels"}}},
        {**good, "params": {"cracking": {"source": "probe", "threshold": 1.5}}},
        {**good, "params": {"cracking": {"source": "probe", "threshold": 0.5, "b": -1}}},
        {**good, "params": {"cracking": {"source": "probe", "threshold": 0.5, "keys": ["Potholes"]}}},
        {**good, "params": {"cracking": "probe"}},
        {**good, "params": {"cracking": {"source": "magic"}}},
    ]
    for c in bad:
        try:
            C.validate_calibration(c)
            raise AssertionError(c)
        except ValueError:
            pass


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
