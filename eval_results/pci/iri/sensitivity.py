"""
Sensitivity of the uncalibrated Golden-Car-equivalent index (app/pci_iri.iri_gc_uncal) to the assumptions
it makes, for the vehicle actually used: a SCOOTER with the phone on the handlebar (owner, 2026-11-09).

R = iri_gc_uncal / IRI_true, computed analytically (random-vibration integrals, Gaussian road with
displacement PSD Gd(n) = Gd0 (n/0.1)^-w on 0.01-10 cycles/m; R does not depend on Gd0):
  sigma_a^2 = s^2 Int (2 pi f)^4 |H_s,veh(f)|^2 |T_mount(f)|^2 |W(f)|^2 |B(f)|^2 Gd(f/v)/v df
  est       = gc_iri_factor * sigma_a / sqrt(v)
  IRI_true  = 1000 sqrt(2/pi)/V sqrt(Int (2 pi f)^2 |H_s - H_u|^2_GC sinc^2(f 0.25/V) Gd(f/V)/V df), V = 80 km/h
Parameters drawn uniformly from the ranges in PRIORS (engineering assumptions for a scooter, see
UNCERTAINTY.md §3 — not measured on our scooter). Writes sensitivity.json. A time-domain spot check
(pci_iri.synthetic_session) confirms the analytic R for a few draws without the wheelbase term.

    venv/Scripts/python.exe eval_results/pci/iri/sensitivity.py
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent))

import numpy as np  # noqa: E402

from app import pci_iri as P  # noqa: E402

PRIORS = {
    "body_hz": (1.2, 3.0),        # sprung (frame + rider) bounce frequency
    "body_zeta": (0.15, 0.50),    # its damping ratio
    "mu": (0.10, 0.40),           # unsprung / sprung mass (scooter engine-swingarm unit is heavy)
    "wheel_hz": (10.0, 25.0),     # wheel-hop frequency (small 10-12 in wheels, stiff tyres)
    "mount_hz": (18.0, 35.0),     # handlebar / holder resonance (a ~25 Hz line is seen in every drive)
    "mount_zeta": (0.05, 0.30),
    "front_weight": (0.6, 1.2),   # phone heave = a*front-wheel response + (1-a)*rear (a > 1: pitch lever)
    "wheelbase_m": (1.20, 1.35),
    "acc_scale": (0.97, 1.03),    # accelerometer gain (projected 1 g reads 9.69-9.72 vs 9.81)
    "waviness": (1.5, 2.8),       # road spectrum slope w (ISO 8608 nominal 2)
}
SPEEDS = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0]
N_DRAW = 3000
SEED = 20261109

f = np.linspace(0.05, 60.0, 12000)
BAND = P._band_power(f, P.DEFAULT, P.DEFAULT.band)
FAC = P.gc_iri_factor()
V = P.IRI_SPEED_MPS


def gd(n, w):
    out = np.zeros_like(n)
    m = (n >= 0.01) & (n <= 10.0)
    out[m] = (n[m] / 0.1) ** (-w)
    return out


def iri_true(w):
    fi = np.concatenate([np.linspace(0.2, 50, 40000), np.linspace(50.01, 230, 20000)])
    hs, hu = P.quarter_car_H(fi, P.GOLDEN_CAR)
    ma = np.sinc(fi * P.IRI_BASE_M / V) ** 2
    var = np.trapezoid((2 * np.pi * fi) ** 2 * np.abs(hs - hu) ** 2 * ma * gd(fi / V, w) / V, fi)
    return 1000 * math.sqrt(2 / math.pi) * math.sqrt(var) / V


def car_from(p):
    ws, wu = 2 * math.pi * p["body_hz"], 2 * math.pi * p["wheel_hz"]
    k2 = ws ** 2
    k1 = max(p["mu"] * wu ** 2 - k2, 1.0)
    return {"k1": k1, "k2": k2, "c": 2 * p["body_zeta"] * ws, "mu": p["mu"]}


def ratio(p, v):
    hs, _ = P.quarter_car_H(f, car_from(p))
    r = f / p["mount_hz"]
    z = p["mount_zeta"]
    T2 = (1 + (2 * z * r) ** 2) / ((1 - r ** 2) ** 2 + (2 * z * r) ** 2)
    a, L = p["front_weight"], p["wheelbase_m"]
    W2 = np.abs(a + (1 - a) * np.exp(-2j * np.pi * f * L / v)) ** 2
    var = p["acc_scale"] ** 2 * np.trapezoid((2 * np.pi * f) ** 4 * np.abs(hs) ** 2 * T2 * W2 * BAND * gd(f / v, p["waviness"]) / v, f)
    est = FAC * math.sqrt(var) / math.sqrt(v)
    return est / iri_true(p["waviness"])


def draw(rng):
    return {k: float(rng.uniform(*r)) for k, r in PRIORS.items()}


def main():
    rng = np.random.default_rng(SEED)
    # control: the Golden Car itself, rigid, w = 2 -> R should be ~1
    gc = {"body_hz": math.sqrt(63.3) / (2 * math.pi), "body_zeta": 6.0 / (2 * math.sqrt(63.3)), "mu": 0.15,
          "wheel_hz": math.sqrt((653.2 + 63.3) / 0.15) / (2 * math.pi), "mount_hz": 1e4, "mount_zeta": 0.1,
          "front_weight": 1.0, "wheelbase_m": 1.25, "acc_scale": 1.0, "waviness": 2.0}
    out = {"priors": PRIORS, "n_draw": N_DRAW, "seed": SEED, "vehicle": "scooter, phone on handlebar (owner 2026-11-09)",
           "control_golden_car_R": {str(v): ratio(gc, v) for v in (3.0, 6.0, 10.0)}, "by_speed": {}}
    allr = []
    draws = [draw(rng) for _ in range(N_DRAW)]
    for v in SPEEDS:
        R = np.array([ratio(p, v) for p in draws])
        allr.append(R)
        out["by_speed"][str(v)] = {q: float(np.percentile(R, int(q[1:]))) for q in ("p05", "p25", "p50", "p75", "p95")}
    allr = np.concatenate(allr)
    out["pooled"] = {q: float(np.percentile(allr, int(q[1:]))) for q in ("p05", "p25", "p50", "p75", "p95")}
    # one-at-a-time: each assumption at its range ends, others at mid-range, v = 6 m/s
    mid = {k: (a + b) / 2 for k, (a, b) in PRIORS.items()}
    oat = {}
    for k, (a, b) in PRIORS.items():
        oat[k] = [ratio({**mid, k: a}, 6.0), ratio({**mid, k: b}, 6.0)]
    out["one_at_a_time_v6"] = oat
    out["mid_R_v6"] = ratio(mid, 6.0)
    # time-domain spot check without the wheelbase term (front_weight = 1)
    spot = []
    srng = np.random.default_rng(7)
    for i in range(4):
        p = {**draw(srng), "front_weight": 1.0, "waviness": 2.0, "acc_scale": 1.0}
        y = P.make_profile(600, 0.025, 64e-6, 2.0, rng=np.random.default_rng(100 + i))
        s = P.synthetic_session(y, 0.025, np.full(len(y), 6.0), car_from(p), rng=np.random.default_rng(200 + i),
                                mount_hz=p["mount_hz"], mount_zeta=p["mount_zeta"])
        res = P.run_session(s["imu"], s["gnss"], s["segments"])
        secs = [b for b in P.sections_along_session(res["segments"]) if b["s0_m"] >= 30 and b["s1_m"] <= 570]
        true = [P.iri_from_profile(y[int((b["s0_m"] - 11) / 0.025):int(b["s1_m"] / 0.025)], 0.025) for b in secs]
        spot.append({"analytic_R": ratio(p, 6.0), "time_domain_R": float(np.mean([b["iri_gc_uncal"] for b in secs]) / np.mean(true))})
    out["time_domain_spot_check"] = spot
    (HERE / "sensitivity.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
