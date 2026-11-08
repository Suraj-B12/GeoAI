"""
Tests for the pure functions of scripts/pci_survey_compare.py (no database, no GPU, no files outside a temp dir).

  1. read_survey: required columns, repeated ids, non-numeric / negative / > 100 % values, PCI outside 0-100,
     positions that are not lat/lon, a surface without a roughness equation for the road class; good file parses.
  2. split: blocks of 3 in CSV order, alternating calibration / test; test-half hash stable under CSV edits.
  3. post-hoc flag: a fit after an evaluate of the same file OR the same test half is post hoc.
  4. fit (synthetic rows, no DB): recovers a known label scale and a known probe threshold/scale; a parameter
     unrelated to the images is NOT written (not better than a constant); MDR potholes fitted in native units.
  5. metrics: perfect agreement, quadratic kappa equals scikit-learn's, bootstrap CI contains the point estimate
     and is deterministic; evaluate's constant = calibration-half MEAN (pre-registered).

Usage:
    venv/Scripts/python.exe scripts/tests/test_pci_survey.py
    venv/Scripts/python.exe -m pytest scripts/tests/test_pci_survey.py -q
"""
from __future__ import annotations

import random
import sys
import tempfile
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import pci_survey_compare as S  # noqa: E402

HDR = ("section_id,start_lat,start_lon,end_lat,end_lon,road_class,surface,cracking_pct,ravelling_pct,"
       "potholes_number,patching_pct,rut_depth_mm,iri_m_km,pci,other_score")


def _csv(text: str) -> Path:
    d = Path(tempfile.mkdtemp(prefix="pci_survey_test_"))
    p = d / "s.csv"
    p.write_text(text, encoding="utf-8")
    return p


def _expect_exit(text: str, needle: str):
    try:
        S.read_survey(_csv(text))
    except SystemExit as e:
        assert needle in str(e), (needle, str(e))
        return
    raise AssertionError(f"expected SystemExit with {needle!r}")


def row(i, **kw):
    v = {"cracking_pct": "", "ravelling_pct": "", "potholes_number": "", "patching_pct": "", "rut_depth_mm": "",
         "iri_m_km": "", "pci": "", "other_score": "", "road_class": "urban", "surface": "BC"}
    v.update({k: str(x) for k, x in kw.items()})
    return (f"S{i:03d},12.9,77.58,12.901,77.581,{v['road_class']},{v['surface']},{v['cracking_pct']},"
            f"{v['ravelling_pct']},{v['potholes_number']},{v['patching_pct']},{v['rut_depth_mm']},{v['iri_m_km']},"
            f"{v['pci']},{v['other_score']}")


def test_read_survey_errors_and_good_file():
    _expect_exit("section_id,start_lat\nS1,12.9\n", "needs columns")
    _expect_exit("\n".join([HDR, row(1), row(1)]), "repeated")
    _expect_exit("\n".join([HDR, row(1, cracking_pct="abc")]), "line 2")
    _expect_exit("\n".join([HDR, row(1, cracking_pct=-1)]), "cracking_pct must be >= 0")
    _expect_exit("\n".join([HDR, row(1, ravelling_pct=120)]), "<= 100")
    _expect_exit("\n".join([HDR, row(1, pci=120)]), "pci must be within 0-100")
    _expect_exit("\n".join([HDR, row(1, road_class="village")]), "road_class")
    _expect_exit("\n".join([HDR, row(1, surface="XYZ")]), "surface")
    _expect_exit("\n".join([HDR, row(1, road_class="rural", surface="BC")]), "no roughness equation")
    _expect_exit("\n".join([HDR, "S1,912.9,77.58,12.901,77.581,urban,BC,,,,,,,,"]), "latitude")
    _expect_exit("\n".join([HDR, "S1,,77.58,12.901,77.581,urban,BC,,,,,,,,"]), "position missing")
    _expect_exit(HDR + "\n", "no sections")
    rows = S.read_survey(_csv("﻿" + "\n".join([HDR, row(1, cracking_pct=3.5, pci="NA", potholes_number=2),
                                                     row(2, road_class="MDR", surface="sdbc", iri_m_km=3.1)])))
    assert len(rows) == 2 and rows[0]["survey"]["cracking"] == 3.5 and rows[0]["pci"] is None
    assert rows[0]["survey"]["potholes"] == 2 and rows[0]["survey"]["iri"] is None
    assert rows[1]["road_class"] == "mdr" and rows[1]["surface"] == "SDBC" and rows[1]["survey"]["iri"] == 3.1


def test_split_rule():
    rows = S.split([{"section_id": f"S{i}"} for i in range(14)])
    halves = [r["half"][0] for r in rows]
    assert "".join(halves) == "ccctttccctttcc"
    assert [r["block"] for r in rows] == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4]
    a = S.test_half_sha(rows)
    rows2 = S.split([{"section_id": f"S{i}", "extra": "edited"} for i in range(14)])
    assert S.test_half_sha(rows2) == a
    rows3 = S.split([{"section_id": f"S{i}"} for i in range(1, 15)])
    assert S.test_half_sha(rows3) != a


def test_post_hoc_flag():
    led = [{"cmd": "evaluate", "survey_sha256": "f1", "test_half_sha256": "t1"}]
    assert S.is_post_hoc(led, "f1", "zz")
    assert S.is_post_hoc(led, "edited-file", "t1")            # same test half, edited CSV
    assert not S.is_post_hoc(led, "f2", "t2")
    assert not S.is_post_hoc([{"cmd": "fit", "survey_sha256": "f1"}], "f1", "t1")
    assert not S.is_post_hoc([{"cmd": "evaluate", "survey_sha256": "f0"}], "f1", "t1")   # old ledger line


def _synthetic(n=24, seed=1):
    """Rows as measure() leaves them: 200 m sections of 40 x 5 m segments with probe scores."""
    rnd = random.Random(seed)
    rows = []
    for i in range(n):
        f_pot = rnd.choice([0.0, 0.05, 0.1, 0.15, 0.2, 0.3])
        f_rav = rnd.random()
        segs = []
        for k in range(40):
            segs.append({"segment_id": f"g{i}_{k}", "length_m": 5.0, "status": "send",
                         "probe": {"Potholes": 0.8 if k < round(f_pot * 40) else 0.2,
                                   "Ravelling": 0.9 if k < round(f_rav * 40) else 0.05}})
        crack_point = rnd.uniform(1, 30)
        rows.append({"section_id": f"S{i}", "survey": {
            "cracking": 0.5 * crack_point,                          # labels scale b = 0.5
            "potholes": 40.0 * f_pot * 0.2,                         # 40 potholes / km at f = 1, 0.2 km
            "ravelling": rnd.uniform(0, 30),                        # unrelated to the images
            "patching": None, "rut_depth": None, "iri": None},
            "pci": None, "road_class": "highway", "surface": None,
            "ours": {"segments": segs, "cracking_point": crack_point, "pothole_runs": 0, "skip_m": 0.0,
                     "report_length_m": 200.0, "road_class": "highway", "surface": None}})
    return S.split(rows)


def test_fit_recovers_known_calibration():
    rows = _synthetic()
    f = S.fit(rows, "0123456789abcdef")
    cr = f["params"]["cracking"]
    assert cr["source"] == "labels" and abs(cr["b"] - 0.5) < 0.03, cr
    po = f["params"]["potholes"]
    assert po["source"] == "probe" and 0.2 < po["threshold"] <= 0.8 and abs(po["b"] - 40) / 40 < 0.05, po
    assert "in-sample" in po["evidence"]
    assert "ravelling" not in f["params"] and f["report"]["ravelling"]["fitted"] is False
    assert "not better than a constant" in f["report"]["ravelling"]["reason"]
    assert f["report"]["patching"]["fitted"] is False                     # no survey values
    # only the calibration half was used
    assert f["report"]["cracking"]["n"] == sum(1 for r in rows if r["half"] == "calibration")
    # the written spec is a valid calibration
    from app import pci_core as C
    C.validate_calibration({"schema": 2, "edition": "IRC:82-2023", "probe_fp": "0123456789abcdef",
                            "params": f["params"]})


def test_fit_mdr_potholes_native_units():
    rows = _synthetic()
    for r in rows:
        r["road_class"] = r["ours"]["road_class"] = "mdr"
    f = S.fit(rows, "0123456789abcdef")
    rep = f["report"]["potholes"]
    assert rep["error_unit"] == "number", rep                            # never a mix of PCI points and counts
    assert f["report"]["cracking"]["error_unit"] == "PCI points"


def test_qkappa_matches_sklearn():
    from sklearn.metrics import cohen_kappa_score
    rnd = random.Random(3)
    for _ in range(20):
        a = [rnd.randrange(6) for _ in range(30)]
        b = [min(5, max(0, x + rnd.choice([-1, 0, 0, 1, 2]))) for x in a]
        ref = cohen_kappa_score(a, b, labels=list(range(6)), weights="quadratic")
        assert abs(S._qkappa(a, b) - ref) < 1e-9
    assert S._qkappa([2, 2, 2], [2, 2, 2]) is None                   # no spread: undefined, not 1


def test_metrics_sanity():
    rnd = random.Random(5)
    pairs = [(v, v, i // 3) for i, v in enumerate(rnd.uniform(5, 95) for _ in range(30))]
    m = S.metrics(pairs, 50.0)
    assert m["mae"] == 0 and m["bias"] == 0 and m["class_exact"] == 1 and abs(m["qkappa"] - 1) < 1e-12
    assert m["ci95"]["mae"] == [0.0, 0.0]
    noisy = [(s + rnd.gauss(3, 5), s, b) for _, s, b in pairs]
    m1, m2 = S.metrics(noisy, 50.0), S.metrics(noisy, 50.0)
    assert m1 == m2                                                       # seeded bootstrap
    lo, hi = m1["ci95"]["mae"]
    assert lo <= m1["mae"] <= hi and m1["constant_mae"] > 0
    lo, hi = m1["ci95"]["bias"]
    assert lo <= m1["bias"] <= hi
    assert m1["mae_minus_constant_ci95"][0] <= m1["mae_minus_constant_ci95"][1]


def test_evaluate_constant_is_calibration_mean():
    rows = _synthetic(12)
    for i, r in enumerate(rows):
        r["pci"] = [10.0, 20.0, 90.0][i % 3]
        r["ours"]["pci_report"] = {"lo": 30.0, "hi": 70.0, "single": None}
        r["ours"]["params"] = None
    res = S.evaluate(rows)
    calib = [r["pci"] for r in rows if r["half"] == "calibration"]
    assert abs(res["constant_from_calibration_half"] - sum(calib) / len(calib)) < 1e-12
    assert res["single_pci"] is None and res["insufficient"] is True
    assert res["range_midpoint"]["n"] == sum(1 for r in rows if r["half"] == "test")


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
