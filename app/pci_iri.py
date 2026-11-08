"""
Road roughness from the drive phone's IMU: features per 5 m video segment and per 100 m, an UNCALIBRATED
Golden-Car-equivalent IRI, an IRI estimate with a 90 % interval (iri_interval), and a calibrated IRI only
when a calibration file exists.

Survey vehicle: a SCOOTER with the phone on the handlebar/front (owner, 2026-11-09; survey.vehicle is empty
in the records). The Golden-Car index assumes a car; sensitivity.py quantifies what that costs (VEHICLE,
SENS_R, UNCERTAINTY.md).

Method and sources: eval_results/pci/iri/RESEARCH.md. Validation: eval_results/pci/iri/VALIDATION.md.

Pipeline (pure functions; file/DB I/O lives in scripts/iri_from_imu.py):
  1. imu.csv (`t_ns,sensor,x,y,z,w,accuracy`, t_ns = CLOCK_BOOTTIME, the clock of frames and GNSS fixes)
     -> acceleration projected on the Android gravity vector: a_v = acc . g_hat  (device axes, any mount)
  2. linear resample to a uniform FS grid; gaps > MAX_GAP_S split the signal into runs
  3. zero-phase Butterworth band-pass per run: 0.5-15 Hz (main), 0.5-5 Hz (body), 5-15 Hz (wheel);
     gyro minus its yaw component (projection on g_hat) -> roll+pitch rate, same band
  4. track distance s(t): the video pipeline's own s at each segment's frame time
     (s_cam = segment.s0_m - frame.near_edge_ahead_m) as anchors, GNSS-Doppler distance warped between them
  5. a segment's window = the times the vehicle's reference point (camera + axle_offset_m) is in [s0_m, s1_m)
  6. features over the window: RMS of each band, peak |a_v|, mean speed = length / time, rms / sqrt(v)
  7. iri_gc_uncal = K(v) * rms, K from the Golden Car under stated assumptions (gc_iri_factor)
  8. iri = calibration(rms, v) only from eval_results/pci/iri/iri_calibration.json; otherwise None and
     iri_status = "uncalibrated". Never a made-up number.

Speed normalisation (why rms / sqrt(v)): for a road whose profile-SLOPE spectrum is white (G_s, m/cycle;
ISO 8608 waviness w = 2), the temporal displacement PSD at speed v is G_s v / (2 pi f)^2, so for ANY linear
vehicle the body acceleration variance over a fixed temporal band is  sigma_a^2 = G_s v Int (2 pi f)^2
|H_s(f)|^2 |B(f)|^2 df  -> sigma_a proportional to sqrt(G_s v), and IRI proportional to sqrt(G_s).

Golden-Car IRI from a profile spectrum (Botshekan et al. 2020, eq. 43): IRI = sqrt(2/pi) sigma_rel / V with
sigma_rel^2 = G_s V Int |H_s - H_u|^2 sinc^2(f L/V) df at V = 80 km/h, L = 0.25 m moving average.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
from scipy import signal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CALIBRATION_PATH = PROJECT_ROOT / "eval_results" / "pci" / "iri" / "iri_calibration.json"
CALIBRATION_SCHEMA = "geoai.iri_calibration.v1"
METHOD_VERSION = "iri-1"

G0 = 9.80665
# Golden Car (ASTM E1926 / Sayers), per unit sprung mass. k1 = tyre, k2 = suspension, c = damper, mu = mu/ms.
GOLDEN_CAR = {"k1": 653.2, "k2": 63.3, "c": 6.0, "mu": 0.15}
IRI_SPEED_MPS = 80.0 / 3.6
IRI_BASE_M = 0.25

VEHICLE = ("scooter (two-wheeler), phone on the handlebar/front - owner's statement 2026-11-09; the drive "
           "records' survey.vehicle field is empty")

UNCAL_ASSUMPTIONS = (
    "Golden-Car-equivalent index, NOT a measured IRI. It assumes (1) the vehicle body responds like the Golden "
    "Car sprung mass, (2) the phone is rigid on that mass (no mount resonance, no pitch lever arm), (3) the "
    "road's profile-slope spectrum is white and Gaussian (ISO 8608 w = 2), (4) one wheel path, (5) exact "
    "accelerometer scale. The survey vehicle is a SCOOTER with the phone on the handlebar: (1), (2) and (4) are "
    "known to be false (single track, stiffer/lighter suspension, small wheels, rider, ~25 Hz handlebar mode); "
    "eval_results/pci/iri/sensitivity.py puts this index at 0.66-3.4x the true IRI (90 % of plausible scooters).")

# R = iri_gc_uncal / IRI_true for plausible scooters (eval_results/pci/iri/sensitivity.json, 3000 draws per
# speed, uniform priors on 10 assumptions; UNCERTAINTY.md §3): speed (m/s) -> (p05, p50, p95).
# scripts/tests/test_pci_iri.py checks these against sensitivity.json.
SENS_R = {3.0: (0.6623, 1.5203, 3.3585), 4.0: (0.6807, 1.5534, 3.2259), 5.0: (0.7119, 1.5613, 3.1338),
          6.0: (0.7479, 1.5719, 3.0775), 7.0: (0.7840, 1.5762, 3.0286), 8.0: (0.8311, 1.5994, 3.0012),
          10.0: (0.8967, 1.6249, 2.9471), 12.0: (0.9385, 1.6716, 2.9282)}
# Pass-to-pass scatter of a single 100 m value, ln units (UNCERTAINTY.md §4): from the only repeated blocks
# (explore.json: median |ln A/B| 0.18-0.27 over 5-6 blocks) -> sigma = 0.27 / (0.6745 sqrt 2) = 0.28; the
# larger end is used because 6 blocks cannot pin it down. Scales as sqrt(100 m / length).
REPEAT_SIGMA_LN_100M = 0.28
# Median idle (standing) RMS in the band over 13 drives (NOISE_FLOOR.md): the lower end of the interval
# treats this much of the signal as non-road noise.
NOISE_FLOOR_RMS = 0.257
Z90 = 1.6449


@dataclass(frozen=True)
class Params:
    fs: float = 200.0                       # uniform resampling rate (Hz); recorder runs at 200 Hz
    band: tuple = (0.5, 15.0)               # main band (Hz); 15 Hz keeps the ~25 Hz engine line out
    body_band: tuple = (0.5, 5.0)
    wheel_band: tuple = (5.0, 15.0)
    order: int = 4                          # Butterworth order (applied forward + backward)
    max_gap_s: float = 0.05                 # an acc gap longer than this splits the signal
    edge_s: float = 1.0                     # samples this close to a run edge are flagged
    min_speed_mps: float = 2.5              # below this no IRI estimate (features still stored)
    stop_speed_mps: float = 0.5             # a window whose GNSS speed dips below this is flagged
    axle_offset_m: float = 0.0              # reference point ahead (+) of the phone along the track
    min_samples: int = 20                   # fewer samples in a window -> no features
    idle_speed_mps: float = 0.3             # noise floor: GNSS speed below this ...
    idle_min_s: float = 3.0                 # ... for at least this long
    golden_car: tuple = tuple(sorted(GOLDEN_CAR.items()))
    version: str = METHOD_VERSION


DEFAULT = Params()


def method_fp(p: Params = DEFAULT) -> str:
    """16-hex fingerprint of everything that changes the stored numbers."""
    blob = json.dumps(asdict(p), sort_keys=True, default=list)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ============================================================ IMU

def read_imu_csv(path) -> dict[str, dict[str, np.ndarray]]:
    """{channel: {"t_ns": int64[], "xyz": float[n,3]}} sorted by time, duplicate stamps dropped."""
    import pandas as pd
    df = pd.read_csv(path, usecols=["t_ns", "sensor", "x", "y", "z"],
                     dtype={"t_ns": "int64", "sensor": "category", "x": "float64", "y": "float64", "z": "float64"})
    out = {}
    for ch, g in df.groupby("sensor", observed=True):
        t = g["t_ns"].to_numpy(np.int64)
        xyz = g[["x", "y", "z"]].to_numpy(np.float64)
        o = np.argsort(t, kind="stable")
        t, xyz = t[o], xyz[o]
        keep = np.concatenate([[True], np.diff(t) > 0])
        ok = keep & np.isfinite(xyz).all(1)
        out[str(ch)] = {"t_ns": t[ok], "xyz": xyz[ok]}
    return out


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = np.nan
    return v / n


def _interp_vec(t_new: np.ndarray, t: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(t_new, t, v[:, k]) for k in range(v.shape[1])])


def _lowpass_gravity(t_s: np.ndarray, acc: np.ndarray, fc: float = 0.2) -> np.ndarray:
    """Fallback gravity estimate when the recorder has no gravity channel: low-passed acc."""
    fs = 1.0 / np.median(np.diff(t_s))
    sos = signal.butter(2, fc, btype="low", fs=fs, output="sos")
    return np.column_stack([signal.sosfiltfilt(sos, acc[:, k]) for k in range(3)])


@dataclass
class Signals:
    t0_ns: int
    t: np.ndarray                 # uniform grid, seconds since t0_ns
    valid: np.ndarray             # False in gaps
    run_id: np.ndarray            # contiguous run index, -1 in gaps
    edge: np.ndarray              # within edge_s of a run edge
    av: np.ndarray                # band-passed vertical acceleration, main band (NaN in gaps)
    av_body: np.ndarray
    av_wheel: np.ndarray
    rot: np.ndarray               # band-passed roll+pitch rate magnitude (rad/s), NaN if no gyro
    av_raw_mean: float            # mean of acc . g_hat (m/s^2) - the accelerometer's idea of 1 g
    gravity_source: str


def _bandpass_runs(x: np.ndarray, run_id: np.ndarray, fs: float, band, order: int) -> np.ndarray:
    sos = signal.butter(order, band, btype="band", fs=fs, output="sos")
    out = np.full_like(x, np.nan)
    pad = 3 * (2 * len(sos) + 1) * 2
    for r in np.unique(run_id[run_id >= 0]):
        idx = np.flatnonzero(run_id == r)
        if len(idx) <= max(pad, int(2 * fs / band[0])):
            continue
        seg = x[idx]
        out[idx] = signal.sosfiltfilt(sos, seg - seg.mean(), padlen=min(len(seg) - 1, int(3 * fs / band[0])))
    return out


def prepare_signals(imu: Mapping[str, Mapping[str, np.ndarray]], p: Params = DEFAULT) -> Signals:
    acc = imu.get("acc")
    if acc is None or len(acc["t_ns"]) < 10 * p.fs:
        raise ValueError("imu has no usable 'acc' channel")
    t0 = int(acc["t_ns"][0])
    ta = (acc["t_ns"] - t0) / 1e9
    a = acc["xyz"]
    grav = imu.get("gravity")
    if grav is not None and len(grav["t_ns"]) >= 10:
        g = _interp_vec(ta, (grav["t_ns"] - t0) / 1e9, grav["xyz"])
        src = "gravity_sensor"
    else:
        g = _lowpass_gravity(ta, a)
        src = "lowpass_acc"
    ghat = _unit(g)
    av_raw = np.einsum("ij,ij->i", a, ghat)

    grid = np.arange(0.0, ta[-1], 1.0 / p.fs)
    # gap mask: grid points whose bracketing acc samples are further apart than max_gap_s
    j = np.clip(np.searchsorted(ta, grid), 1, len(ta) - 1)
    valid = (ta[j] - ta[j - 1]) <= p.max_gap_s
    run_id = np.full(len(grid), -1)
    starts = np.flatnonzero(valid & ~np.concatenate([[False], valid[:-1]]))
    for r, s in enumerate(starts):
        e = s
        while e < len(grid) and valid[e]:
            e += 1
        run_id[s:e] = r
    edge = np.zeros(len(grid), bool)
    ne = int(p.edge_s * p.fs)
    for r in np.unique(run_id[run_id >= 0]):
        idx = np.flatnonzero(run_id == r)
        edge[idx[:ne]] = True
        edge[idx[-ne:]] = True
    edge |= ~valid

    x = np.interp(grid, ta, av_raw)
    av = _bandpass_runs(x, run_id, p.fs, p.band, p.order)
    av_body = _bandpass_runs(x, run_id, p.fs, p.body_band, p.order)
    av_wheel = _bandpass_runs(x, run_id, p.fs, p.wheel_band, p.order)

    rot = np.full(len(grid), np.nan)
    gyro = imu.get("gyro")
    if gyro is not None and len(gyro["t_ns"]) >= 10 * p.fs:
        tg = (gyro["t_ns"] - t0) / 1e9
        w = _interp_vec(grid, tg, gyro["xyz"])
        gh = _unit(_interp_vec(grid, ta, ghat))
        w_perp = w - np.einsum("ij,ij->i", w, gh)[:, None] * gh
        comps = [_bandpass_runs(w_perp[:, k], run_id, p.fs, p.band, p.order) for k in range(3)]
        rot = np.sqrt(sum(c ** 2 for c in comps))
    return Signals(t0_ns=t0, t=grid, valid=valid, run_id=run_id, edge=edge, av=av, av_body=av_body,
                   av_wheel=av_wheel, rot=rot, av_raw_mean=float(np.nanmean(av_raw)), gravity_source=src)


# ============================================================ track distance

def gnss_distance(t_ns: np.ndarray, speed: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Cumulative distance (m) at each fix: trapezoid of Doppler speed; haversine step where speed is missing."""
    t = (np.asarray(t_ns, np.int64) - int(t_ns[0])) / 1e9
    sp = np.asarray(speed, float)
    la, lo = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    dlat, dlon = np.diff(la), np.diff(lo)
    h = np.sin(dlat / 2) ** 2 + np.cos(la[:-1]) * np.cos(la[1:]) * np.sin(dlon / 2) ** 2
    hav = 2 * 6371008.8 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
    dt = np.diff(t)
    dop = 0.5 * (sp[1:] + sp[:-1]) * dt
    step = np.where(np.isfinite(dop), dop, hav)
    step = np.where(np.isfinite(step), np.maximum(step, 0.0), 0.0)
    return np.concatenate([[0.0], np.cumsum(step)])


@dataclass
class TrackMap:
    """s(t) on the video pipeline's track-distance axis."""
    anchor_t: np.ndarray          # s since t0
    anchor_s: np.ndarray
    gnss_t: np.ndarray            # s since t0
    gnss_d: np.ndarray            # cumulative GNSS distance
    gnss_speed: np.ndarray

    def D(self, t):
        return np.interp(t, self.gnss_t, self.gnss_d)

    def speed(self, t):
        sp = np.where(np.isfinite(self.gnss_speed), self.gnss_speed, np.nan)
        ok = np.isfinite(sp)
        if ok.sum() < 2:
            return np.full(np.shape(t), np.nan)
        return np.interp(t, self.gnss_t[ok], sp[ok])

    def s(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, float)
        at, as_ = self.anchor_t, self.anchor_s
        D = self.D(t)
        Da = self.D(at)
        out = np.empty_like(t)
        i = np.searchsorted(at, t, side="right") - 1
        before, after = i < 0, i >= len(at) - 1
        out[before] = as_[0] + (D[before] - Da[0])
        out[after] = as_[-1] + (D[after] - Da[-1])
        mid = ~(before | after)
        im = i[mid]
        ds_a = as_[im + 1] - as_[im]
        dD = Da[im + 1] - Da[im]
        use_d = (dD > 0.5) & (dD > 0.2 * ds_a)
        frac_d = np.where(use_d, (D[mid] - Da[im]) / np.where(dD > 0, dD, 1.0), 0.0)
        frac_t = (t[mid] - at[im]) / (at[im + 1] - at[im])
        frac = np.clip(np.where(use_d, frac_d, frac_t), 0.0, 1.0)
        out[mid] = as_[im] + frac * ds_a
        return np.maximum.accumulate(out)

    def anchor_dt(self, t: np.ndarray) -> np.ndarray:
        """Length (s) of the anchor interval each time falls in (inf outside the anchors)."""
        i = np.searchsorted(self.anchor_t, t, side="right") - 1
        ok = (i >= 0) & (i < len(self.anchor_t) - 1)
        out = np.full(np.shape(t), np.inf)
        out[ok] = self.anchor_t[i[ok] + 1] - self.anchor_t[i[ok]]
        return out


def segment_anchors(segments: Iterable[Mapping]) -> tuple[np.ndarray, np.ndarray]:
    """(t_ns, s_cam) from segment records: frame.t_ns and s0_m - frame.near_edge_ahead_m.
    Only strictly increasing (t, s) pairs are kept."""
    pts = []
    for r in segments:
        t, s0, near = r.get("frame_t_ns"), r.get("s0_m"), r.get("near_edge_ahead_m")
        if t is None or s0 is None or near is None:
            continue
        pts.append((int(t), float(s0) - float(near)))
    pts.sort()
    keep_t, keep_s = [], []
    for t, s in pts:
        if keep_t and (t <= keep_t[-1] or s < keep_s[-1] - 0.01):
            continue
        keep_t.append(t)
        keep_s.append(max(s, keep_s[-1]) if keep_s else s)
    return np.array(keep_t, np.int64), np.array(keep_s, float)


def build_track_map(anchor_t_ns, anchor_s, gnss: Mapping[str, np.ndarray], t0_ns: int) -> TrackMap:
    if len(anchor_t_ns) < 2:
        raise ValueError("fewer than 2 segment anchors")
    gt = np.asarray(gnss["t_ns"], np.int64)
    o = np.argsort(gt, kind="stable")
    gt = gt[o]
    keep = np.concatenate([[True], np.diff(gt) > 0])
    gt = gt[keep]
    sp = np.asarray(gnss["speed_mps"], float)[o][keep]
    la = np.asarray(gnss["lat"], float)[o][keep]
    lo = np.asarray(gnss["lon"], float)[o][keep]
    d = gnss_distance(gt, sp, la, lo)
    return TrackMap(anchor_t=(np.asarray(anchor_t_ns, np.int64) - t0_ns) / 1e9, anchor_s=np.asarray(anchor_s, float),
                    gnss_t=(gt - t0_ns) / 1e9, gnss_d=d, gnss_speed=sp)


# ============================================================ Golden Car

def quarter_car_H(f: np.ndarray, car: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Sprung (H_s) and unsprung (H_u) displacement per unit road displacement, per unit sprung mass."""
    k1, k2, c, mu = car["k1"], car["k2"], car["c"], car["mu"]
    w = 2 * np.pi * np.asarray(f, float)
    a11 = -w ** 2 + 1j * w * c + k2
    a12 = -(1j * w * c + k2)
    a22 = -mu * w ** 2 + 1j * w * c + k2 + k1
    det = a11 * a22 - a12 * a12
    hs = (-a12 * k1) / det
    hu = (a11 * k1) / det
    return hs, hu


def _band_power(f: np.ndarray, p: Params, band) -> np.ndarray:
    """|B(f)|^2 of the zero-phase (forward-backward) Butterworth band-pass."""
    sos = signal.butter(p.order, band, btype="band", fs=p.fs, output="sos")
    _, h = signal.sosfreqz(sos, worN=f, fs=p.fs)
    return np.abs(h) ** 4


def acc_integral(p: Params = DEFAULT, car: Optional[Mapping] = None, band=None) -> float:
    """Int (2 pi f)^2 |H_s|^2 |B|^2 df: sigma_a^2 = G_s * v * this (white slope road)."""
    car = car or dict(p.golden_car)
    f = np.linspace(1e-3, p.fs / 2, 200001)
    hs, _ = quarter_car_H(f, car)
    return float(np.trapezoid((2 * np.pi * f) ** 2 * np.abs(hs) ** 2 * _band_power(f, p, band or p.band), f))


def rel_integral(car: Optional[Mapping] = None, V: float = IRI_SPEED_MPS, base_m: float = IRI_BASE_M) -> float:
    """Int |H_s - H_u|^2 sinc^2(f L / V) df: sigma_rel_velocity^2 = G_s * V * this."""
    car = car or GOLDEN_CAR
    f = np.concatenate([np.linspace(1e-4, 50, 200001), np.linspace(50.0001, 2000, 200001)])
    hs, hu = quarter_car_H(f, car)
    ma = np.sinc(f * base_m / V) ** 2
    return float(np.trapezoid(np.abs(hs - hu) ** 2 * ma, f))


def iri_from_slope_psd(Gs: float, car: Optional[Mapping] = None) -> float:
    """Golden-Car IRI (m/km) of a Gaussian road with white profile-slope PSD Gs (m/cycle)."""
    V = IRI_SPEED_MPS
    return 1000.0 * math.sqrt(2 / math.pi) * math.sqrt(Gs * V * rel_integral(car)) / V


_FACTOR_CACHE: dict = {}


def gc_iri_factor(p: Params = DEFAULT) -> float:
    """IRI_gc (m/km) = factor * rms_av / sqrt(v)  (rms in m/s^2, v in m/s), Golden Car + white slope road."""
    key = method_fp(p)
    if key not in _FACTOR_CACHE:
        Ia = acc_integral(p)
        Ir = rel_integral(dict(p.golden_car))
        V = IRI_SPEED_MPS
        _FACTOR_CACHE[key] = 1000.0 * math.sqrt(2 / math.pi) * math.sqrt(Ir / V) / math.sqrt(Ia)
    return _FACTOR_CACHE[key]


def iri_gc_uncal(rms_av: Optional[float], speed: Optional[float], p: Params = DEFAULT) -> Optional[float]:
    if rms_av is None or speed is None or not np.isfinite(rms_av) or not np.isfinite(speed) or speed < p.min_speed_mps:
        return None
    return float(gc_iri_factor(p) * rms_av / math.sqrt(speed))


# ============================================================ profile tools (validation / tests)

def make_profile(length_m: float, dx: float, gd_n0: float, w: float = 2.0, n0: float = 0.1,
                 rng: Optional[np.random.Generator] = None, n_min: float = 0.01, n_max: float = 10.0) -> np.ndarray:
    """Random profile (m) with displacement PSD Gd(n) = gd_n0 (n/n0)^-w (m^3, one-sided), n in cycles/m."""
    rng = rng or np.random.default_rng(0)
    N = int(round(length_m / dx))
    n = np.fft.rfftfreq(N, dx)
    G = np.zeros_like(n)
    band = (n >= n_min) & (n <= n_max)
    G[band] = gd_n0 * (n[band] / n0) ** (-w)
    dn = 1.0 / (N * dx)
    amp = np.sqrt(G * dn) / 2.0            # one-sided PSD -> var(y) = sum G dn (y = 2 Re sum X e^..)
    spec = amp * (rng.standard_normal(len(n)) + 1j * rng.standard_normal(len(n)))
    y = np.fft.irfft(spec, N) * N
    return y - y.mean()


def iri_from_profile(y: np.ndarray, dx: float, car: Optional[Mapping] = None, lead_m: float = 11.0) -> float:
    """IRI (m/km) of a profile by direct simulation of the Golden Car at 80 km/h (state-space, exact
    first-order-hold discretisation, 250 mm moving average). The first lead_m metres only settle the car."""
    car = car or GOLDEN_CAR
    V = IRI_SPEED_MPS
    k = max(int(round(IRI_BASE_M / dx)), 1)
    ys = np.convolve(y, np.ones(k) / k, mode="valid") if k > 1 else y
    sim = simulate_quarter_car(ys, dx / V, car)
    i0 = int(round(lead_m / dx))
    return float(np.mean(np.abs(sim["rel_velocity"][i0:])) / V * 1000.0)


def simulate_quarter_car(u: np.ndarray, h: float, car: Mapping[str, float]) -> dict:
    """Quarter car (per unit sprung mass) driven by road displacement u sampled every h seconds
    (linear between samples). Returns sprung acceleration and suspension relative velocity per sample.
    Starts on u[0] with the vertical velocity of the road's first 0.5 s."""
    from scipy.linalg import expm
    k1, k2, c, mu = car["k1"], car["k2"], car["c"], car["mu"]
    A = np.array([[0, 1, 0, 0], [-k2, -c, k2, c], [0, 0, 0, 1], [k2 / mu, c / mu, -(k1 + k2) / mu, -c / mu]], float)
    B = np.array([0, 0, 0, k1 / mu], float)
    M = np.zeros((6, 6))
    M[:4, :4] = A * h
    M[:4, 4] = B * h
    M[4, 5] = 1.0
    E = expm(M)
    Ad, B0, B1 = E[:4, :4], E[:4, 4], E[:4, 5]
    n = len(u)
    X = np.empty((n, 4))
    v0 = (u[min(n - 1, int(round(0.5 / h)))] - u[0]) / (min(n - 1, int(round(0.5 / h))) * h) if n > 1 else 0.0
    x = np.array([u[0], v0, u[0], v0])      # start moving with the road's initial slope (as Sayers' init)
    du = np.diff(u, append=u[-1])
    for i in range(n):
        X[i] = x
        x = Ad @ x + B0 * u[i] + B1 * du[i]
    acc = X @ A[1]
    return {"acc": acc, "rel_velocity": X[:, 1] - X[:, 3], "zs": X[:, 0]}


def drive_profile(y: np.ndarray, dx: float, speed_mps: np.ndarray, h: float, car: Mapping[str, float]) -> dict:
    """Drive a car over profile y (every dx m) with a time-varying speed. speed_mps: speed at each profile
    point. Returns uniform time grid t (step h), position x(t), sprung acceleration."""
    xs = np.arange(len(y)) * dx
    tt = np.concatenate([[0.0], np.cumsum(dx / np.asarray(speed_mps[:-1], float))])
    t = np.arange(0.0, tt[-1], h)
    x = np.interp(t, tt, xs)
    sim = simulate_quarter_car(np.interp(x, xs, y), h, car)
    return {"t": t, "x": x, "acc": sim["acc"]}


def synthetic_session(y: np.ndarray, dx: float, speed_mps: np.ndarray, car: Mapping[str, float], *,
                      rng: np.random.Generator, mount_hz: Optional[float] = None, mount_zeta: float = 0.1,
                      noise: float = 0.0, scale: float = 1.0, tilt_deg: float = 0.0, jitter_ns: float = 0.0,
                      seg_start_m: float = 20.0, seg_len: float = 5.0, near_m: float = 2.0, t0_ns: int = 10**14,
                      lat0: float = 12.95, lon0: float = 77.58) -> dict:
    """A recorded drive as the phone would see it, for validation and tests: device-frame acc (with
    gravity, a tilted mount, scale error, white noise, timestamp jitter) at 200 Hz, gravity at 50 Hz, gyro,
    1 Hz GNSS with Doppler speed along a straight road due north, and 5 m segment records whose anchors are
    frame times with the camera near_m behind the segment start. Track distance = profile position."""
    h = 1.0 / 1000.0
    d = drive_profile(y, dx, speed_mps, h, car)
    t, x, a = d["t"], d["x"], d["acc"]
    if mount_hz:
        wn = 2 * np.pi * mount_hz
        sysm = signal.cont2discrete(([wn ** 2], [1, 2 * mount_zeta * wn, wn ** 2]), h, method="bilinear")
        a = signal.lfilter(np.squeeze(sysm[0]), sysm[1], a)
    th = math.radians(tilt_deg)
    R = np.array([[1, 0, 0], [0, math.cos(th), -math.sin(th)], [0, math.sin(th), math.cos(th)]])
    ts = np.arange(0.0, t[-1], 0.005)
    ts_j = ts + (rng.normal(0, jitter_ns, len(ts)) / 1e9 if jitter_ns else 0)
    ts_j = np.sort(ts_j)
    a_s = np.interp(ts_j, t, a)
    world = np.column_stack([np.zeros_like(a_s), np.zeros_like(a_s), G0 + a_s])
    acc = scale * world @ R.T + rng.normal(0, noise, (len(ts_j), 3)) if noise else scale * world @ R.T
    tg = np.arange(0.0, t[-1], 0.02)
    grav = np.tile(R @ np.array([0, 0, G0]), (len(tg), 1))
    gyro = rng.normal(0, 0.002, (len(ts_j), 3))
    tf = np.arange(0.0, t[-1], 1.0)
    xf = np.interp(tf, t, x)
    vf = np.gradient(xf, tf) if len(tf) > 1 else np.zeros_like(tf)
    gnss = {"t_ns": (t0_ns + tf * 1e9).astype(np.int64), "speed_mps": vf,
            "lat": lat0 + xf / 111195.0, "lon": np.full(len(tf), lon0)}
    segs = []
    k = 0
    s0 = seg_start_m
    while s0 + seg_len <= x[-1] - 2:
        tf_k = float(np.interp(s0 - near_m, x, t))
        segs.append({"segment_id": f"syn_seg{k:05d}", "k": k, "s0_m": s0, "s1_m": s0 + seg_len,
                     "frame_t_ns": int(t0_ns + tf_k * 1e9), "near_edge_ahead_m": near_m, "covered": True})
        k += 1
        s0 += seg_len
    imu = {"acc": {"t_ns": (t0_ns + ts_j * 1e9).astype(np.int64), "xyz": acc},
           "gravity": {"t_ns": (t0_ns + tg * 1e9).astype(np.int64), "xyz": grav},
           "gyro": {"t_ns": (t0_ns + ts_j * 1e9).astype(np.int64), "xyz": gyro}}
    return {"imu": imu, "gnss": gnss, "segments": segs, "t0_ns": t0_ns}


def run_session(imu, gnss, segments, p: Params = DEFAULT, calibration: Optional[Mapping] = None) -> dict:
    """prepare_signals + track map + segment_estimates + idle noise floor for one drive."""
    sig = prepare_signals(imu, p)
    at, as_ = segment_anchors(segments)
    tmap = build_track_map(at, as_, gnss, sig.t0_ns)
    est = segment_estimates(sig, tmap, segments, p, calibration)
    return {"segments": est, "idle": idle_noise_floor(sig, tmap, p), "gravity_source": sig.gravity_source,
            "acc_one_g": sig.av_raw_mean, "n_anchors": len(at), "vehicle": VEHICLE}


# ============================================================ features

def window_features(sig: Signals, i0: int, i1: int, p: Params = DEFAULT) -> dict:
    sl = slice(i0, i1)
    av = sig.av[sl]
    ok = np.isfinite(av)
    n = int(ok.sum())
    out: dict[str, Any] = {"n_samples": n, "coverage": (n / max(i1 - i0, 1)) if i1 > i0 else 0.0}
    if n < p.min_samples:
        out.update(rms_av=None, rms_body=None, rms_wheel=None, rms_rot=None, peak_av=None)
        return out

    def rms(x):
        x = x[np.isfinite(x)]
        return float(np.sqrt(np.mean(x ** 2))) if len(x) >= p.min_samples else None

    out.update(rms_av=rms(av), rms_body=rms(sig.av_body[sl]), rms_wheel=rms(sig.av_wheel[sl]),
               rms_rot=rms(sig.rot[sl]), peak_av=float(np.nanmax(np.abs(av))))
    out["edge_frac"] = float(sig.edge[sl].mean())
    return out


def segment_estimates(sig: Signals, tmap: TrackMap, segments: Sequence[Mapping], p: Params = DEFAULT,
                      calibration: Optional[Mapping] = None) -> list[dict]:
    """One dict per segment record (keys segment_id, k, s0_m, s1_m). Segments the vehicle never covered
    inside the IMU record get error set."""
    s_grid = tmap.s(sig.t) + p.axle_offset_m
    v_grid = tmap.speed(sig.t)
    adt = tmap.anchor_dt(sig.t)
    out = []
    for r in segments:
        s0, s1 = r.get("s0_m"), r.get("s1_m")
        base = {"segment_id": r["segment_id"], "k": r.get("k"), "s0_m": s0, "s1_m": s1}
        if s0 is None or s1 is None or not s1 > s0:
            out.append({**base, "error": "segment has no s0_m/s1_m"})
            continue
        if not r.get("covered", True):
            out.append({**base, "error": "segment not covered (GNSS gap)"})
            continue
        i0 = int(np.searchsorted(s_grid, s0, "left"))
        i1 = int(np.searchsorted(s_grid, s1, "left"))
        if i0 == 0 or i1 >= len(s_grid) or i1 <= i0:
            out.append({**base, "error": "segment outside the IMU record"})
            continue
        dur = (i1 - i0) / p.fs
        speed = (s1 - s0) / dur
        f = window_features(sig, i0, i1, p)
        flags = []
        vmin = float(np.nanmin(v_grid[i0:i1])) if np.isfinite(v_grid[i0:i1]).any() else None
        if vmin is not None and vmin < p.stop_speed_mps:
            flags.append("stop_in_window")
        if speed < p.min_speed_mps:
            flags.append("low_speed")
        if f.get("edge_frac", 0) > 0:
            flags.append("filter_edge")
        if f["coverage"] < 0.95:
            flags.append("imu_gap")
        if float(np.max(adt[i0:i1])) > 5.0:
            flags.append("sparse_anchors")
        if f["rms_av"] is None:
            flags.append("too_few_samples")
        rec = {**base, "t0_ns": sig.t0_ns + int(round(sig.t[i0] * 1e9)), "t1_ns": sig.t0_ns + int(round(sig.t[i1 - 1] * 1e9)),
               "duration_s": dur, "speed_mps": speed, "speed_min_mps": vmin, **f}
        rec["rms_av_norm"] = (rec["rms_av"] / math.sqrt(speed)) if rec["rms_av"] is not None and speed > 0 else None
        ok_for_iri = rec["rms_av"] is not None and "low_speed" not in flags
        rec["iri_gc_uncal"] = iri_gc_uncal(rec["rms_av"], speed, p) if ok_for_iri else None
        rec["flags"] = flags
        rec["iri"], rec["iri_status"] = calibrated_iri(rec, calibration, unit_m=s1 - s0)
        out.append(rec)
    return out


def idle_noise_floor(sig: Signals, tmap: TrackMap, p: Params = DEFAULT) -> dict:
    """RMS of the main band where the GNSS speed stays below idle_speed_mps for >= idle_min_s (engine on,
    car standing). Excludes the first/last second of each stop and filter edges."""
    v = tmap.speed(sig.t)
    still = np.isfinite(v) & (v < p.idle_speed_mps) & ~sig.edge & np.isfinite(sig.av)
    vals, total = [], 0.0
    i, n = 0, len(still)
    trim = int(p.fs)
    while i < n:
        if not still[i]:
            i += 1
            continue
        j = i
        while j < n and still[j]:
            j += 1
        if (j - i) / p.fs >= p.idle_min_s + 2:
            seg = sig.av[i + trim:j - trim]
            vals.append(seg)
            total += len(seg) / p.fs
        i = j
    if not vals:
        return {"idle_s": 0.0, "rms_av": None, "n_stops": 0}
    x = np.concatenate(vals)
    return {"idle_s": total, "rms_av": float(np.sqrt(np.mean(x ** 2))), "n_stops": len(vals)}


# ============================================================ aggregation

def aggregate(rows: Sequence[Mapping], p: Params = DEFAULT, calibration: Optional[Mapping] = None) -> dict:
    """Combine segment rows (any set: a 100 m section, a pass) into one value set.
    rms: sqrt of the time-weighted mean square; speed: total length / total time;
    iri_gc_uncal: length-weighted mean of the segments' values (IRI is a length average).
    Rows with error, no rms or low speed are left out (counted)."""
    use = [r for r in rows if not r.get("error") and r.get("rms_av") is not None
           and "low_speed" not in (r.get("flags") or [])]
    res: dict[str, Any] = {"n_segments": len(rows), "n_used": len(use), "vehicle": VEHICLE}
    if not use:
        res.update(length_m=0.0, duration_s=0.0, speed_mps=None, rms_av=None, rms_rot=None, rms_av_norm=None,
                   iri_gc_uncal=None, iri=None, iri_status="no_data")
        return res
    L = np.array([r["s1_m"] - r["s0_m"] for r in use], float)
    T = np.array([r["duration_s"] for r in use], float)
    rms = np.array([r["rms_av"] for r in use], float)
    res["length_m"] = float(L.sum())
    res["duration_s"] = float(T.sum())
    res["speed_mps"] = float(L.sum() / T.sum())
    res["rms_av"] = float(np.sqrt(np.sum(rms ** 2 * T) / T.sum()))
    rot = [(r["rms_rot"], r["duration_s"]) for r in use if r.get("rms_rot") is not None]
    res["rms_rot"] = float(np.sqrt(sum(a * a * t for a, t in rot) / sum(t for _, t in rot))) if rot else None
    res["rms_av_norm"] = res["rms_av"] / math.sqrt(res["speed_mps"])
    gc = [(r["iri_gc_uncal"], r["s1_m"] - r["s0_m"]) for r in use if r.get("iri_gc_uncal") is not None]
    res["iri_gc_uncal"] = float(sum(a * l for a, l in gc) / sum(l for _, l in gc)) if gc else None
    res["iri"], res["iri_status"] = calibrated_iri(res, calibration, unit_m=res["length_m"])
    return res


def sections_along_session(rows: Sequence[Mapping], section_m: float = 100.0, p: Params = DEFAULT,
                           calibration: Optional[Mapping] = None) -> list[dict]:
    """100 m blocks of one drive on its own track distance: block b = floor(s0_m / section_m)."""
    blocks: dict[int, list] = {}
    for r in rows:
        if r.get("s0_m") is None:
            continue
        blocks.setdefault(int(r["s0_m"] // section_m), []).append(r)
    out = []
    for b in sorted(blocks):
        agg = aggregate(blocks[b], p, calibration)
        out.append({"block": b, "s0_m": b * section_m, "s1_m": (b + 1) * section_m,
                    "segment_ids": [r["segment_id"] for r in blocks[b]], **agg})
    return out


# ============================================================ calibration

def load_calibration(path: Optional[Path] = None) -> Optional[dict]:
    """The calibration file, or None when there is none. A malformed file raises (never silently ignored)."""
    path = Path(path) if path else CALIBRATION_PATH
    if not path.exists():
        return None
    cal = json.loads(path.read_text(encoding="utf-8"))
    problems = validate_calibration(cal)
    if problems:
        raise ValueError(f"{path}: invalid IRI calibration: {'; '.join(problems)}")
    return cal


def validate_calibration(cal: Mapping) -> list[str]:
    p = []
    if cal.get("schema") != CALIBRATION_SCHEMA:
        p.append(f"schema must be {CALIBRATION_SCHEMA}")
    if cal.get("model") != "log_linear":
        p.append("model must be log_linear")
    terms = cal.get("terms")
    if not isinstance(terms, list) or not terms or any(t not in ("log_rms_av", "log_speed", "log_rms_rot") for t in terms):
        p.append("terms must be a non-empty subset of log_rms_av, log_speed, log_rms_rot")
    coef = cal.get("coef") or {}
    for t in ["intercept"] + (terms if isinstance(terms, list) else []):
        if not isinstance(coef.get(t), (int, float)) or not math.isfinite(coef[t]):
            p.append(f"coef.{t} missing or not a number")
    if not isinstance(cal.get("method_fp"), str):
        p.append("method_fp missing")
    if not isinstance(cal.get("unit_m"), (int, float)) or cal["unit_m"] <= 0:
        p.append("unit_m must be > 0")
    rng = cal.get("fit_range") or {}
    for k in ("rms_av", "speed_mps"):
        v = rng.get(k)
        if not (isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v) and v[0] < v[1]):
            p.append(f"fit_range.{k} must be [lo, hi]")
    for k in ("n", "residual_sd_log", "source"):
        if k not in cal:
            p.append(f"{k} missing")
    return p


UNIT_TOLERANCE = 0.5  # a calibration fitted on 100 m units applies to 50-150 m aggregates


def calibrated_iri(rec: Mapping, cal: Optional[Mapping], unit_m: Optional[float] = None,
                   p: Params = DEFAULT) -> tuple[Optional[float], str]:
    """(IRI m/km, status). status: calibrated | uncalibrated | method_mismatch | unit_mismatch |
    out_of_range | no_data. Only 'calibrated' carries a number."""
    if cal is None:
        return None, "uncalibrated"
    if cal.get("method_fp") != method_fp(p):
        return None, "method_mismatch"
    rms, v = rec.get("rms_av"), rec.get("speed_mps")
    if rms is None or v is None or rms <= 0 or v <= 0:
        return None, "no_data"
    if unit_m is not None and abs(unit_m - cal["unit_m"]) > UNIT_TOLERANCE * cal["unit_m"]:
        return None, "unit_mismatch"
    rng = cal["fit_range"]
    if not (rng["rms_av"][0] <= rms <= rng["rms_av"][1] and rng["speed_mps"][0] <= v <= rng["speed_mps"][1]):
        return None, "out_of_range"
    x = {"log_rms_av": math.log(rms), "log_speed": math.log(v)}
    if "log_rms_rot" in cal["terms"]:
        rr = rec.get("rms_rot")
        if rr is None or rr <= 0:
            return None, "no_data"
        x["log_rms_rot"] = math.log(rr)
    c = cal["coef"]
    y = c["intercept"] + sum(c[t] * x[t] for t in cal["terms"])
    return float(math.exp(y)), "calibrated"


def fit_calibration(rows: Sequence[Mapping], *, terms: Sequence[str] = ("log_rms_av", "log_speed"),
                    unit_m: float = 100.0, source: Mapping, p: Params = DEFAULT, notes: str = "") -> dict:
    """Fit log IRI_ref = a + sum b_t * term on calibration-half sections.
    rows: {"rms_av", "speed_mps", "iri_ref", ["rms_rot"]} per section (aggregate() output + the survey's IRI).
    `source` must say where iri_ref came from (survey file, split rule, section ids)."""
    use = [r for r in rows if r.get("iri_ref") and r["iri_ref"] > 0 and r.get("rms_av") and r.get("speed_mps")
           and ("log_rms_rot" not in terms or r.get("rms_rot"))]
    if len(use) < len(terms) + 5:
        raise ValueError(f"need at least {len(terms) + 5} calibration sections, got {len(use)}")
    cols = {"log_rms_av": [math.log(r["rms_av"]) for r in use], "log_speed": [math.log(r["speed_mps"]) for r in use],
            "log_rms_rot": [math.log(r["rms_rot"]) if r.get("rms_rot") else float("nan") for r in use]}
    X = np.column_stack([np.ones(len(use))] + [cols[t] for t in terms])
    y = np.log([r["iri_ref"] for r in use])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    dof = max(len(use) - X.shape[1], 1)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    cal = {
        "schema": CALIBRATION_SCHEMA, "model": "log_linear", "method_fp": method_fp(p), "unit_m": float(unit_m),
        "terms": list(terms), "coef": {"intercept": float(beta[0]), **{t: float(b) for t, b in zip(terms, beta[1:])}},
        "n": len(use), "residual_sd_log": float(math.sqrt(np.sum(res ** 2) / dof)),
        "r2_log": (1 - float(np.sum(res ** 2)) / ss_tot) if ss_tot > 0 else None,
        "fit_range": {"rms_av": [float(min(r["rms_av"] for r in use)), float(max(r["rms_av"] for r in use))],
                      "speed_mps": [float(min(r["speed_mps"] for r in use)), float(max(r["speed_mps"] for r in use))]},
        "source": dict(source), "notes": notes,
    }
    problems = validate_calibration(cal)
    if problems:
        raise ValueError("; ".join(problems))
    return cal


def _sens_at(v: float) -> tuple[float, float, float]:
    vs = sorted(SENS_R)
    v = min(max(float(v), vs[0]), vs[-1])
    return tuple(float(np.interp(v, vs, [SENS_R[x][i] for x in vs])) for i in range(3))


def iri_interval(agg: Mapping, calibration: Optional[Mapping] = None, p: Params = DEFAULT) -> dict:
    """IRI estimate (m/km) with a 90 % interval for an aggregate (aggregate() output: a 100 m section, a km).

    Calibrated (a calibration applies to this aggregate): point = calibrated IRI, interval = log-normal
    prediction interval from the calibration's residual sd (t quantile, sqrt(1 + 1/n)).
    Otherwise UNCALIBRATED: point = iri_gc_uncal / R50(v); lo90 = noise-floor-corrected iri_gc_uncal / R95(v)
    shrunk by the pass-to-pass scatter; hi90 = iri_gc_uncal / R05(v) grown by it. R = the scooter
    sensitivity ratios (SENS_R). The interval covers vehicle/mount physics within the stated ranges, the
    noise floor and pass-to-pass scatter; it does NOT cover a vehicle outside those ranges, wheel-path
    position, GNSS distance errors, non-Gaussian roads (potholes, speed breakers) or a phone that moved in
    its holder (UNCERTAINTY.md §5)."""
    base = {"vehicle": VEHICLE, "unit": "m/km"}
    rms, v, L = agg.get("rms_av"), agg.get("speed_mps"), agg.get("length_m") or 0.0
    if calibration is not None:
        val, st = calibrated_iri(agg, calibration, unit_m=L or None, p=p)
        if st == "calibrated":
            from scipy import stats as _st
            n = int(calibration["n"])
            dof = max(n - 1 - len(calibration["terms"]), 1)
            half = float(_st.t.ppf(0.95, dof)) * calibration["residual_sd_log"] * math.sqrt(1 + 1 / n)
            return {**base, "point": val, "lo90": val * math.exp(-half), "hi90": val * math.exp(half),
                    "basis": "calibrated", "detail": f"log-linear calibration (n = {n}, residual sd "
                    f"{calibration['residual_sd_log']:.3f} ln), source {calibration.get('source')}"}
    if rms is None or v is None or v < p.min_speed_mps or rms <= 0:
        return {**base, "point": None, "lo90": None, "hi90": None, "basis": "no_data",
                "detail": "no usable roughness samples at >= %.1f m/s" % p.min_speed_mps}
    est = gc_iri_factor(p) * rms / math.sqrt(v)
    est_nf = gc_iri_factor(p) * math.sqrt(max(rms * rms - NOISE_FLOOR_RMS ** 2, (0.1 * rms) ** 2)) / math.sqrt(v)
    r05, r50, r95 = _sens_at(v)
    sig = REPEAT_SIGMA_LN_100M * math.sqrt(100.0 / max(L, 100.0)) if L else REPEAT_SIGMA_LN_100M
    k = math.exp(Z90 * sig)
    return {**base, "point": est / r50, "lo90": est_nf / r95 / k, "hi90": est / r05 * k,
            "basis": "uncalibrated_scooter", "iri_gc_uncal": est,
            "detail": (f"Golden-Car-equivalent {est:.2f} divided by the scooter response ratio R "
                       f"(p05/p50/p95 = {r05:.2f}/{r50:.2f}/{r95:.2f} at {v:.1f} m/s), noise floor "
                       f"{NOISE_FLOOR_RMS} m/s2 at the low end, pass-to-pass scatter x/{k:.2f}. Not a measured "
                       "IRI; replaced by the calibrated value once the survey calibration exists.")}


def report_field(agg: Mapping, calibration: Optional[Mapping] = None) -> dict:
    """What the condition report shows for roughness: the calibrated IRI, or the uncalibrated estimate with
    its interval, always with the vehicle and the basis."""
    iv = iri_interval(agg, calibration)
    if iv["basis"] == "calibrated":
        return {"iri_m_per_km": round(iv["point"], 2), "lo90": round(iv["lo90"], 2), "hi90": round(iv["hi90"], 2),
                "status": "calibrated", "vehicle": VEHICLE, "detail": iv["detail"],
                "rms_av": agg.get("rms_av"), "speed_mps": agg.get("speed_mps")}
    return {"iri_m_per_km": None, "status": agg.get("iri_status") or "uncalibrated",
            "estimate": None if iv["point"] is None else {k: round(iv[k], 2) for k in ("point", "lo90", "hi90")},
            "estimate_basis": iv["basis"], "vehicle": VEHICLE,
            "note": "Roughness not measured: uncalibrated phone-IMU estimate from a scooter, shown as a range "
                    "(eval_results/pci/iri/UNCERTAINTY.md); the calibrated value replaces it after the survey.",
            "detail": iv["detail"], "rms_av": agg.get("rms_av"), "speed_mps": agg.get("speed_mps"),
            "iri_gc_uncal": agg.get("iri_gc_uncal"), "iri_gc_uncal_assumptions": UNCAL_ASSUMPTIONS}
