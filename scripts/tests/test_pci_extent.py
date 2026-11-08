"""
Tests for app/pci_extent.py (IRC:82-2023 extent calibration; pure, synthetic data, CPU only, no database).

  1. PAV gives a non-decreasing fit; interpolation is flat outside the knots.
  2. A synthetic monotone world (area rises with the score): the selection picks a curve, not the constant, and
     the fitted curve recovers the truth within tolerance; a world with no signal picks the constant.
  3. estimate_extents: length-weighted mean, interval contains the mean, lo >= 0, interval shrinks with
     section length (more segments) and widens with the Bengaluru domain factor (x/÷ 2 when KS D >= 0.2).
  4. Coverage of the 90 % interval on synthetic held-out blocks drawn from the fitting world is near 90 %.
  5. Pothole number = area % / 100 x length x lane width / 0.1 m2 (IRC:82-2023 eq. 2), scales linearly.
  6. Coverage rule: < 50 % scored length -> None; unscored segments still count in the length.
  7. validate_calibration rejects a decreasing curve, unknown parameter, foreign keys, missing residual model;
     label models demand labels.
  8. The stored Attain calibration (if present) validates and gives finite numbers.

Usage:
    CUDA_VISIBLE_DEVICES=-1 venv/Scripts/python.exe scripts/tests/test_pci_extent.py
    venv/Scripts/python.exe -m pytest scripts/tests/test_pci_extent.py -q
"""
from __future__ import annotations

import copy
import math
import random
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import pci_extent as PE  # noqa: E402


def _world(n_blocks=30, per=20, signal=True, seed=1):
    """Frames in blocks; each block has a latent severity; score and area depend on it."""
    rng = random.Random(seed)
    rows = []
    for b in range(n_blocks):
        sev = rng.random()
        for _ in range(per):
            s = min(0.99, max(0.005, sev * 0.9 + rng.gauss(0, 0.08))) if signal else rng.uniform(0.01, 0.9)
            area = max(0.0, (10.0 * sev if signal else 3.0) + rng.gauss(0, 2.0))
            rows.append({"block": b, "scores": {"Potholes": s, "Ravelling": s, "Patching": s,
                                                "Alligator Cracking": s, "Longitudinal Cracking": s * 0.5},
                         "labels": [], "area": area})
    return rows


def _target(r, p):
    return r["area"]


def _cal(rows, params=("potholes",), boot=60):
    cal, diag = PE.fit_calibration(rows, _target, params=params, n_boot=boot, seed=3, select_folds=5, select_seed=4)
    return PE.validate_calibration(cal), diag


def test_pav_and_interp():
    y = PE._pav([3.0, 1.0, 2.0, 5.0, 4.0], [1, 1, 1, 1, 1])
    assert all(b >= a - 1e-12 for a, b in zip(y, y[1:]))
    assert abs(y[0] - 2.0) < 1e-12 and abs(y[-1] - 4.5) < 1e-12
    assert PE._interp(-10, [0, 1], [2, 4]) == 2 and PE._interp(10, [0, 1], [2, 4]) == 4
    assert abs(PE._interp(0.5, [0, 1], [2, 4]) - 3.0) < 1e-12


def test_selection_signal_and_no_signal():
    cal, diag = _cal(_world(signal=True))
    assert diag["selection"]["potholes"]["chosen"] != "constant"
    spec = cal["params"]["potholes"]
    lo, hi = PE.frame_mean(spec, 0.05), PE.frame_mean(spec, 0.85)
    assert hi > lo + 5.0, (lo, hi)                         # truth: ~0.5 -> ~9.4
    cal0, diag0 = _cal(_world(signal=False, seed=5))
    spec0 = cal0["params"]["potholes"]
    assert abs(PE.frame_mean(spec0, 0.9) - PE.frame_mean(spec0, 0.05)) < 1.5


def test_estimate_basic_properties():
    cal, _ = _cal(_world())
    spec = cal["params"]["potholes"]
    sc = [{"Potholes": 0.1}] * 10 + [{"Potholes": 0.8}] * 10
    L = [5.0] * 10 + [15.0] * 10
    e = PE.estimate_extents(sc, L, calibration=cal)["potholes"]
    want = (sum(5 * PE.frame_mean(spec, 0.1) for _ in range(10)) + sum(15 * PE.frame_mean(spec, 0.8)
                                                                        for _ in range(10))) / 200.0
    assert abs(e["mean"] - want) < 1e-9
    assert 0.0 <= e["lo90"] <= e["mean"] <= e["hi90"]
    short = PE.estimate_extents(sc[:2], L[:2], calibration=cal)["potholes"]
    long_ = PE.estimate_extents(sc[:2] * 50, L[:2] * 50, calibration=cal)["potholes"]
    assert (long_["hi90"] - long_["lo90"]) < (short["hi90"] - short["lo90"])
    # Bengaluru domain factor: wider, and wider still with a large KS shift
    a = PE.estimate_extents(sc, L, calibration=cal, domain="attain")["potholes"]
    b = PE.estimate_extents(sc, L, calibration=cal, domain="bengaluru")["potholes"]
    cal2 = copy.deepcopy(cal)
    cal2["transfer"] = {"potholes": {"ks_D": 0.5}}
    c = PE.estimate_extents(sc, L, calibration=cal2, domain="bengaluru")["potholes"]
    assert b["domain_factor90"] == 1.5 and c["domain_factor90"] == 2.0
    assert (a["hi90"] - a["lo90"]) < (b["hi90"] - b["lo90"]) < (c["hi90"] - c["lo90"])
    assert a["mean"] == b["mean"] == c["mean"]


def test_coverage_on_synthetic_heldout():
    cal, _ = _cal(_world(n_blocks=40, seed=11), boot=100)
    test = _world(n_blocks=120, seed=99)
    cov = n = 0
    for b in range(120):
        fr = [r for r in test if r["block"] == b]
        e = PE.estimate_extents([r["scores"] for r in fr], [1.0] * len(fr), calibration=cal)["potholes"]
        t = sum(r["area"] for r in fr) / len(fr)
        cov += e["lo90"] <= t <= e["hi90"]
        n += 1
    assert 0.78 <= cov / n <= 1.0, cov / n


def test_pothole_number_eq2():
    cal, _ = _cal(_world())
    sc = [{"Potholes": 0.5}] * 20
    e = PE.estimate_extents(sc, [5.0] * 20, lane_width_m=3.5, calibration=cal)
    a = e["potholes"]["mean"]
    n = e["potholes_number"]
    assert abs(n["mean"] - a / 100.0 * 100.0 * 3.5 / 0.1) < 1e-9
    assert abs(n["area_m2_mean"] - a / 100.0 * 350.0) < 1e-9
    e2 = PE.estimate_extents(sc * 2, [5.0] * 40, lane_width_m=7.0, calibration=cal)
    assert abs(e2["potholes_number"]["mean"] - 4 * n["mean"]) < 1e-6
    assert n["lo90"] <= n["mean"] <= n["hi90"]


def test_coverage_rule():
    cal, _ = _cal(_world())
    sc = [{"Potholes": 0.5}] * 4 + [None] * 6
    e = PE.estimate_extents(sc, [5.0] * 10, calibration=cal)
    assert e["potholes"]["mean"] is None and e["potholes"]["status"] == "insufficient coverage"
    e = PE.estimate_extents([{"Potholes": 0.5}] * 6 + [None] * 4, [5.0] * 10, calibration=cal)
    assert e["potholes"]["mean"] is not None and abs(e["potholes"]["coverage"] - 0.6) < 1e-12
    assert e["potholes_number"]["road_area_m2"] == 6 * 5.0 * 3.5
    e = PE.estimate_extents([{"Potholes": 0.5}] * 6 + [None] * 4, [5.0] * 10, calibration=cal, total_length_m=50)
    assert e["potholes_number"]["road_area_m2"] == 50 * 3.5
    for bad in (dict(lane_width_m=0), dict(domain="mars")):
        try:
            PE.estimate_extents([{"Potholes": 0.5}], [5.0], calibration=cal, **bad)
            raise AssertionError(bad)
        except ValueError:
            pass


def test_validation():
    cal, _ = _cal(_world())
    bad = copy.deepcopy(cal)
    m = bad["params"]["potholes"]["model"]
    if m["kind"] == "bins":
        m["y"] = list(reversed(sorted(m["y"])))
        m["y"][0] += 1.0
        _raises(bad)
    bad = copy.deepcopy(cal)
    bad["params"]["rutting"] = bad["params"]["potholes"]
    _raises(bad)
    bad = copy.deepcopy(cal)
    bad["params"]["potholes"]["keys"] = ["Alligator Cracking"]
    _raises(bad)
    bad = copy.deepcopy(cal)
    del bad["params"]["potholes"]["residual"]
    _raises(bad)
    bad = copy.deepcopy(cal)
    bad["schema"] = 99
    _raises(bad)
    # a label model needs labels
    rows = _world()
    for i, r in enumerate(rows):
        r["labels"] = ["Potholes"] if i % 2 else []
    lab = copy.deepcopy(cal)
    sub = lab["params"]["potholes"]["model"]
    lab["params"]["potholes"]["model"] = {"kind": "label", "strata": {"0": sub, "1": sub}}
    lab["params"]["potholes"]["boot_y"] = [{"0": sub["y"], "1": sub["y"]}]
    PE.validate_calibration(lab)
    try:
        PE.estimate_extents([{"Potholes": 0.3}], [5.0], calibration=lab)
        raise AssertionError("labels required")
    except ValueError:
        pass
    e = PE.estimate_extents([{"Potholes": 0.3}], [5.0], calibration=lab, labels=[["Potholes"]])
    assert e["potholes"]["mean"] >= 0


def _raises(cal):
    try:
        PE.validate_calibration(cal)
    except ValueError:
        return
    raise AssertionError("validate_calibration accepted a bad calibration")


def test_stored_calibration():
    if not PE.DEFAULT_PATH.exists():
        print("  (no stored calibration; skipped)")
        return
    cal = PE.load_calibration()
    sc = [{k: 0.2 for k in ("Alligator Cracking", "Longitudinal Cracking", "Transverse Cracking", "Block Cracking",
                            "Ravelling", "Hungry Surface", "Patching", "Potholes")}] * 20
    e = PE.estimate_extents(sc, [5.0] * 20, calibration=cal, domain="bengaluru")
    for p in PE.PCI_PARAMS:
        assert math.isfinite(e[p]["mean"]) and 0 <= e[p]["lo90"] <= e[p]["mean"] <= e[p]["hi90"]
    assert e["potholes_number"]["mean"] >= 0


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
