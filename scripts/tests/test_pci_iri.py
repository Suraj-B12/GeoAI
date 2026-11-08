"""
Tests for app/pci_iri.py (roughness from the phone IMU; pure, synthetic data, CPU only, no database).

  1. Golden Car arithmetic: IRI by direct simulation of a random white-slope profile agrees with the
     closed-form PSD relation; a constant-slope (tilted plane) profile has IRI ~ 0.
  2. End to end with the Golden Car itself as the vehicle (rigid mount, no noise): the uncalibrated
     estimate recovers the profile's IRI (median ratio within 15 %) at 4 and 10 m/s, and rms / sqrt(v) is
     speed-invariant (white-slope road).
  3. A different car + mount resonance + noise + tilt + scale error + jitter: rank order of three roads of
     different roughness preserved at different speeds; absolute value NOT recovered (calibration problem).
  4. Track map: a 10 s stop in the middle of the drive does not smear distance; windows land on the right
     positions; flags for low speed / stop / IMU gaps; idle noise floor found during the stop.
  5. No calibration file -> iri None, status "uncalibrated"; report_field never shows a number.
  6. fit_calibration recovers known coefficients; method / unit / range mismatches refuse; a malformed file
     raises; read_imu_csv parses the recorder's format.
  7. iri_interval: uncalibrated scooter interval = the stated formula, never narrower than the vehicle term,
     scatter shrinks with length; calibrated = log-normal prediction interval; no extrapolation; the SENS_R
     table equals sensitivity.json; rut_prior (app/pci_rut.py) prior / flag / calibration bins.

Usage:
    CUDA_VISIBLE_DEVICES=-1 venv/Scripts/python.exe scripts/tests/test_pci_iri.py
    venv/Scripts/python.exe -m pytest scripts/tests/test_pci_iri.py -q
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import pci_iri as P  # noqa: E402

DX = 0.025
SEDAN = {"k1": 520.0, "k2": 40.0, "c": 3.6, "mu": 0.12}   # softer, less damped than the Golden Car


def _profile(gd, n=600.0, seed=0, w=2.0):
    return P.make_profile(n, DX, gd, w, rng=np.random.default_rng(seed))


def _sections(y, res, lo=30.0, hi=None):
    hi = hi or (len(y) * DX - 30)
    secs = [b for b in P.sections_along_session(res["segments"]) if b["s0_m"] >= lo and b["s1_m"] <= hi]
    true = [P.iri_from_profile(y[int((b["s0_m"] - 11) / DX):int(b["s1_m"] / DX)], DX) for b in secs]
    return secs, np.array(true)


def test_method_fp():
    fp = P.method_fp()
    assert len(fp) == 16 and all(c in "0123456789abcdef" for c in fp)
    assert P.method_fp(P.Params(band=(0.5, 25.0))) != fp
    assert P.method_fp(P.Params()) == fp


def test_golden_car_psd_vs_simulation():
    y = _profile(16e-6, n=2500, seed=1)
    gs = (2 * math.pi * 0.1) ** 2 * 16e-6
    sim, psd = P.iri_from_profile(y, DX), P.iri_from_slope_psd(gs)
    assert 0.9 < sim / psd < 1.1, (sim, psd)
    plane = np.arange(int(300 / DX)) * DX * 0.01        # 1 % grade, perfectly smooth
    assert P.iri_from_profile(plane, DX) < 0.01


def test_golden_car_end_to_end_and_speed_invariance():
    y = _profile(64e-6, seed=2)       # ISO class B-C, IRI ~4
    out = {}
    for v in (4.0, 10.0):
        s = P.synthetic_session(y, DX, np.full(len(y), v), P.GOLDEN_CAR, rng=np.random.default_rng(3))
        res = P.run_session(s["imu"], s["gnss"], s["segments"])
        secs, true = _sections(y, res)
        est = np.array([b["iri_gc_uncal"] for b in secs])
        ratio = float(np.median(est / true))
        assert 0.85 < ratio < 1.15, (v, ratio)
        out[v] = (np.mean([b["rms_av"] for b in secs]), np.mean([b["rms_av_norm"] for b in secs]))
        assert all(r["iri_status"] == "uncalibrated" and r["iri"] is None for r in res["segments"] if not r.get("error"))
    raw_ratio = out[10.0][0] / out[4.0][0]
    norm_ratio = out[10.0][1] / out[4.0][1]
    assert 1.3 < raw_ratio < 1.9, raw_ratio          # sqrt(2.5) = 1.58
    assert 0.85 < norm_ratio < 1.15, norm_ratio


def test_other_car_ranks_but_is_not_absolute():
    roads = {gd: _profile(gd, seed=int(gd * 1e6)) for gd in (8e-6, 64e-6, 512e-6)}
    speeds = {8e-6: 9.0, 64e-6: 6.0, 512e-6: 4.0}    # rough roads driven slower, as in town
    est, true = [], []
    for gd, y in roads.items():
        rng = np.random.default_rng(7)
        s = P.synthetic_session(y, DX, np.full(len(y), speeds[gd]), SEDAN, rng=rng, mount_hz=18.0, noise=0.03,
                                scale=1.1, tilt_deg=25.0, jitter_ns=2e5)
        res = P.run_session(s["imu"], s["gnss"], s["segments"])
        secs, t = _sections(y, res)
        est.append(np.mean([b["iri_gc_uncal"] for b in secs]))
        true.append(np.mean(t))
    assert list(np.argsort(est)) == list(np.argsort(true)), (est, true)
    assert est[2] > 3 * est[0]


def test_stop_and_flags():
    y = _profile(32e-6, n=400, seed=4)
    n = len(y)
    v = np.full(n, 6.0)
    i_stop = int(200 / DX)
    v[i_stop - 400:i_stop + 400] = 0.4                 # crawl ~20 m around 200 m ...
    v[i_stop - 4:i_stop + 4] = 0.01                    # ... and stand still ~16 s
    s = P.synthetic_session(y, DX, v, P.GOLDEN_CAR, rng=np.random.default_rng(5), noise=0.01)
    res = P.run_session(s["imu"], s["gnss"], s["segments"])
    by = {r["s0_m"]: r for r in res["segments"]}
    r = by[200.0]
    assert "low_speed" in r["flags"] and r["iri_gc_uncal"] is None and r["rms_av"] is not None
    ok = by[100.0]
    assert ok["flags"] == [] and abs(ok["speed_mps"] - 6.0) < 0.3 and ok["iri_gc_uncal"] > 0
    far = by[300.0]
    assert abs(far["speed_mps"] - 6.0) < 0.3, far
    assert res["idle"]["n_stops"] >= 1 and res["idle"]["rms_av"] < 0.05, res["idle"]
    agg = P.aggregate([by[200.0], by[100.0]])
    assert agg["n_used"] == 1 and agg["iri_status"] == "uncalibrated" and agg["iri"] is None


def test_imu_gap_and_uncovered():
    y = _profile(32e-6, n=300, seed=6)
    s = P.synthetic_session(y, DX, np.full(len(y), 6.0), P.GOLDEN_CAR, rng=np.random.default_rng(8))
    acc = s["imu"]["acc"]
    t = (acc["t_ns"] - acc["t_ns"][0]) / 1e9
    keep = ~((t > 20.0) & (t < 20.5))                 # 0.5 s hole at ~120 m
    s["imu"]["acc"] = {"t_ns": acc["t_ns"][keep], "xyz": acc["xyz"][keep]}
    s["segments"][3]["covered"] = False
    res = P.run_session(s["imu"], s["gnss"], s["segments"])
    hit = [r for r in res["segments"] if not r.get("error") and 118 <= r["s0_m"] <= 122]
    assert hit and all({"imu_gap", "filter_edge"} & set(r["flags"]) for r in hit), hit
    assert res["segments"][3]["error"].startswith("segment not covered")


def test_report_field_uncalibrated():
    f = P.report_field({"iri_status": "uncalibrated", "rms_av": 0.3, "speed_mps": 6, "iri_gc_uncal": 4.1})
    assert f["iri_m_per_km"] is None and f["status"] == "uncalibrated" and "NOT a measured IRI" in f["iri_gc_uncal_assumptions"]
    assert P.calibrated_iri({"rms_av": 0.3, "speed_mps": 6}, None) == (None, "uncalibrated")
    assert P.load_calibration(Path(tempfile.gettempdir()) / "no_such_iri_cal.json") is None


def test_fit_calibration_roundtrip():
    rng = np.random.default_rng(9)
    rows = []
    for _ in range(40):
        rms, v = rng.uniform(0.1, 1.0), rng.uniform(3, 12)
        iri = math.exp(1.2 + 0.9 * math.log(rms) - 0.4 * math.log(v) + rng.normal(0, 0.01))
        rows.append({"rms_av": rms, "speed_mps": v, "iri_ref": iri})
    cal = P.fit_calibration(rows, source={"survey": "synthetic"})
    assert abs(cal["coef"]["intercept"] - 1.2) < 0.05 and abs(cal["coef"]["log_rms_av"] - 0.9) < 0.03
    assert abs(cal["coef"]["log_speed"] + 0.4) < 0.05 and cal["r2_log"] > 0.99
    val, st = P.calibrated_iri({"rms_av": 0.5, "speed_mps": 6.0}, cal, unit_m=100)
    assert st == "calibrated" and abs(val - math.exp(1.2 + 0.9 * math.log(0.5) - 0.4 * math.log(6))) < 0.1
    assert P.calibrated_iri({"rms_av": 5.0, "speed_mps": 6.0}, cal, unit_m=100)[1] == "out_of_range"
    assert P.calibrated_iri({"rms_av": 0.5, "speed_mps": 6.0}, cal, unit_m=5)[1] == "unit_mismatch"
    assert P.calibrated_iri({"rms_av": 0.5, "speed_mps": 6.0}, {**cal, "method_fp": "0" * 16})[1] == "method_mismatch"
    with tempfile.TemporaryDirectory() as d:
        good = Path(d) / "cal.json"
        good.write_text(json.dumps(cal), encoding="utf-8")
        assert P.load_calibration(good)["coef"] == cal["coef"]
        bad = Path(d) / "bad.json"
        bad.write_text(json.dumps({**cal, "coef": {}}), encoding="utf-8")
        try:
            P.load_calibration(bad)
            raise AssertionError("malformed calibration accepted")
        except ValueError:
            pass
    try:
        P.fit_calibration(rows[:3], source={})
        raise AssertionError("fit on 3 rows accepted")
    except ValueError:
        pass


def test_read_imu_csv():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "imu.csv"
        f.write_text("t_ns,sensor,x,y,z,w,accuracy\n"
                     "300,acc,0.1,0.2,9.8,,3\n100,acc,0.0,0.0,9.7,,3\n100,acc,5,5,5,,3\n"
                     "200,gravity,0,0,9.81,,3\n150,game_rv,0.1,0.2,0.3,0.9,3\n", encoding="utf-8")
        imu = P.read_imu_csv(f)
        assert list(imu["acc"]["t_ns"]) == [100, 300] and imu["acc"]["xyz"][0, 2] == 9.7
        assert set(imu) == {"acc", "gravity", "game_rv"}


def test_iri_interval_uncalibrated():
    agg = {"rms_av": 0.8, "speed_mps": 6.0, "length_m": 100.0, "iri_status": "uncalibrated"}
    iv = P.iri_interval(agg)
    est = P.gc_iri_factor() * 0.8 / math.sqrt(6.0)
    r05, r50, r95 = P.SENS_R[6.0]
    assert iv["basis"] == "uncalibrated_scooter" and "scooter" in iv["vehicle"]
    assert abs(iv["point"] - est / r50) < 1e-9
    assert iv["lo90"] < iv["point"] < iv["hi90"] and iv["hi90"] / iv["lo90"] > 4
    k = math.exp(P.Z90 * P.REPEAT_SIGMA_LN_100M)
    import math as _m                                                  # physics and scatter in quadrature
    up = _m.sqrt(_m.log(r50 / r05) ** 2 + _m.log(k) ** 2)
    assert abs(iv["hi90"] - est / r50 * _m.exp(up)) < 1e-9 and iv["hi90"] < est / r05 * k
    km = P.iri_interval({**agg, "length_m": 1000.0})
    assert km["hi90"] / km["lo90"] < iv["hi90"] / iv["lo90"]          # scatter shrinks, bias does not
    assert km["hi90"] / km["lo90"] > (r95 / r05)                       # never narrower than the vehicle term
    assert P.iri_interval({"rms_av": None, "speed_mps": 6})["basis"] == "no_data"
    assert P.iri_interval({"rms_av": 0.5, "speed_mps": 1.0})["basis"] == "no_data"
    f = P.report_field(agg)
    assert f["iri_m_per_km"] is None and f["estimate"]["lo90"] < f["estimate"]["hi90"] and "scooter" in f["vehicle"]


def test_iri_interval_calibrated():
    rng = np.random.default_rng(11)
    rows = []
    for _ in range(30):
        rms, v = rng.uniform(0.3, 1.5), rng.uniform(3, 10)
        rows.append({"rms_av": rms, "speed_mps": v, "iri_ref": math.exp(0.5 + 0.8 * math.log(rms) + rng.normal(0, 0.2))})
    cal = P.fit_calibration(rows, terms=("log_rms_av",), source={"survey": "synthetic"})
    agg = {"rms_av": 0.8, "speed_mps": 6.0, "length_m": 100.0}
    iv = P.iri_interval(agg, cal)
    assert iv["basis"] == "calibrated"
    from scipy import stats
    want = math.exp(stats.t.ppf(0.95, cal["n"] - 2) * cal["residual_sd_log"] * math.sqrt(1 + 1 / cal["n"]))
    assert abs(iv["hi90"] / iv["point"] - want) < 1e-9 and abs(iv["point"] * iv["point"] - iv["lo90"] * iv["hi90"]) < 1e-6
    assert 0.1 < cal["residual_sd_log"] < 0.3
    # out of the fitted range -> falls back to the uncalibrated interval, never extrapolates
    assert P.iri_interval({**agg, "rms_av": 5.0}, cal)["basis"] == "uncalibrated_scooter"
    assert P.report_field(agg, cal)["status"] == "calibrated"


def test_sens_table_matches_sensitivity_json():
    f = PROJECT_ROOT / "eval_results" / "pci" / "iri" / "sensitivity.json"
    d = json.loads(f.read_text(encoding="utf-8"))
    assert {float(k) for k in d["by_speed"]} == set(P.SENS_R)
    for k, b in d["by_speed"].items():
        assert np.allclose(P.SENS_R[float(k)], (b["p05"], b["p50"], b["p95"]), atol=1e-4), k
    assert all(abs(v - 1.0) < 1e-3 for v in d["control_golden_car_R"].values())
    lo, mid, hi = P._sens_at(100.0)
    assert (lo, mid, hi) == P.SENS_R[12.0]


def test_rut_prior():
    from app import pci_rut as R
    pr = R.rut_prior(None)
    assert pr["is_prior"] and pr["point"] is None and (pr["lo90"], pr["hi90"]) == (0.0, 25.0) and pr["flag"] is None
    st = R.score_stats([0.1, 0.2, None, 0.9, float("nan"), 0.6])
    assert st["n"] == 4 and abs(st["max"] - 0.9) < 1e-12 and 0.6 < st["p90"] <= 0.9
    pr = R.rut_prior(st)
    assert pr["flag"] and (pr["lo90"], pr["hi90"]) == (0.0, 25.0) and not pr["conditioned_on_score"]
    cal = {"schema": R.CALIBRATION_SCHEMA, "stat": "p90", "probe_fp": "x", "source": {"survey": "synthetic"},
           "bins": [{"lo": 0.0, "hi": 0.3, "lo90": 0, "point": 3, "hi90": 8, "n": 12},
                    {"lo": 0.3, "hi": 1.0, "lo90": 4, "point": 12, "hi90": 22, "n": 9}]}
    assert R.validate_calibration(cal) == []
    assert R.rut_prior({"n": 5, "p90": 0.1}, cal)["hi90"] == 8 and R.rut_prior({"n": 5, "p90": 1.0}, cal)["point"] == 12
    assert R.rut_prior({"n": 0, "p90": None}, cal)["is_prior"]
    bad = {**cal, "bins": [{"lo": 0.0, "hi": 0.5, "lo90": 0, "point": 3, "hi90": 8, "n": 1}]}
    assert R.validate_calibration(bad)
    assert R.load_calibration(Path(tempfile.gettempdir()) / "no_such_rut_cal.json") is None


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
