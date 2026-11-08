# IRI from the survey phone — estimate and honest uncertainty (no reference IRI yet)

Written 2026-11-09. Function: `app/pci_iri.py: iri_interval(agg, calibration=None) -> {point, lo90, hi90,
basis, vehicle, detail, ...}`; `report_field(agg, calibration=None)` wraps it for the report.

**Survey vehicle: a SCOOTER (two-wheeler), phone mounted on the handlebar/front** — owner's statement
2026-11-09. The drive records' `survey.vehicle` field is empty. Everything written before this date in
RESEARCH.md / VALIDATION.md that says "car" or "windshield" is wrong about the vehicle (the numbers there
stand; their interpretation is corrected here and in RESEARCH.md §6).

## 1. Literature: how far are uncalibrated phone IRI values from reference IRI?

All opened in this session (pages that could not be fetched are marked).

| Source | Setting | Finding relevant to an uncalibrated estimate |
|---|---|---|
| Setiawan & Nurdin, Lontar Komputer 10(2):62–72, 2019 (PDF from distantreader.org/stacks/journals/lontar/lontar-48004.pdf, text read) | Bandung; **Yamaha Mio scooter** and Daihatsu Ayla car, phone in a holder, same route, 100 m sections, 12–31 km/h, accelerometer-only IRI (Zang-type ∬\|a\|dt²/S) | Car mean IRI **2.95** m/km (2.30–3.43, 10 sections); scooter mean **5.73** (3.57–8.88, 10 sections) — **1.9× the car on the same road**, and far more scattered. Agreement with a visual condition class: car 9/10, motorcycle "6 of 19" (text; the table lists 10 rows). Authors blame the motorcycle's stiffer suspension. |
| Certad, Aghanouri, Gorospe, Olaverri-Monreal, arXiv 2606.03427 (June 2026, PDF text read) | One car, one phone (Galaxy A56, dashboard), two commercial apps (Road IRI by Definitics, India; RoadLab Pro), 381 km, 100 m | Same drives, same phone: mean IRI **1.15 vs 4.26 m/km** (factor 3.7); ICC 0.15–0.41; strong proportional bias (Bland–Altman slope 0.89); app IRIs did not consistently correlate with vertical-acceleration RMS. |
| Islam & Buttlar, NEXTRANS 098IY04 final report (Purdue, PDF text read) | Honda CR-V, dashboard mount, 50 mph, 0.1-mile sections, double integration + ProVAL, vs inertial profiler | Without calibration 17/20 sections within ±10 in/mi (±0.16 m/km) on smoother roads; on the rough site **underpredicted**; a fitted correction IRI = 0.95·x + 58 in/mi (+0.92 m/km) was needed. One car, one speed. |
| Hoang & Nguyen, Transport and Communications Science Journal 70(4), 2019 (PDF text read) | RoadLab Pro, car, Hanoi | Same lane, same phone: IRI **2.98 → 4.77 m/km (+60 %) from 20 to 50 km/h**; between-mount SD 0.18 / 0.27 / 0.46 m/km at 20 / 35 / 50 km/h. |
| Botshekan et al., Data-Centric Engineering 1:e16, 2020 (Cambridge HTML read) | Vehicle parameters *identified* from many trips (half-car), phone position free | After identification, absolute relative error peaks near **10 %** vs laser IRI (calibrated on one track, validated on another). The best physics-based case — it needs the vehicle identification we do not have. |
| Douangphachanh & Oneyama, ICINCO 2014 (PDF text read) | 4 vehicles × 4 phone placements, per-vehicle regression vs VIMS | Calibrated per vehicle/phone: R² 0.58–0.82. |
| Yeganeh et al., arXiv 1902.10699 (PDF text read) | Car, calibrated model | R² 0.76 vs Road Surface Profiler. |
| Zang et al., Sensors 18:914, 2018 (PMC read) | Bicycle handlebar, displacement method | r = 0.893 vs laser profiler (n = 9, 20 m); repeated groups differ by 0.7–1.8 m/km. |

Not found: any peer-reviewed **calibrated** IRI model for scooters/motorcycles with error bands against a
profiler. Search summaries mention a car + motorcycle study with r up to 0.862 against a Roughometer
(ResearchGate, HTTP 403 — not read) and Tai et al. 2010 (motorcycle anomaly detection, 78.5 % precision,
cited in Yeganeh et al.) — neither gives an uncalibrated IRI error.

**Reading of the evidence:** uncalibrated phone IRI is off by a vehicle-, mount- and speed-dependent factor
that is commonly 0.5–2× for cars and was ~2× (upward) for a scooter in the one direct comparison; two
commercial apps on the same car differed by 3.7×. Calibrated systems reach R² ≈ 0.6–0.8 / ~10 % — but only
after per-vehicle calibration or identification.

## 2. Plausibility prior: IRI on Bengaluru / Indian urban bituminous roads

| Source (opened) | Road | IRI (m/km) |
|---|---|---|
| Anjana & Jamakhandi, IJRTI 2(7), 2017 (PDF text read) | 4 links, Silk Board junction (PCI 80–87) | **1.53, 1.63, 1.82, 1.86** (MERLIN) |
| Jagadeesh et al., IJERT 2(11), 2013 (page read via fetch) | Hosur Road arterial, 4 surveys 2011–12 | roughness 2237–2893 mm/km (instrument not stated) → **3.1–3.9** with IRC:SP:16's RI = 630·IRI^1.12 |
| Shivananda & Khatua, IOP Conf. Ser. MSE 1255:012020, 2022 (PDF text read) | North Bengaluru, Carbin app in a car (itself an uncalibrated phone app) | NH **2.44**, SH **4.25**, urban road **5.6** |
| IRC:SP:16-2019 (PDF text read), Table 3.1 / 3.2 | Bituminous (BC/SDBC) | NH/SH: Good < 2.55, Fair 2.55–3.30, Poor > 3.30; MDR BC < 2.81 / 2.81–3.55 / > 3.55 |

Published Bengaluru values span **~1.5–5.6 m/km** (three studies, five-plus roads, one of them itself an
uncalibrated phone app). IRC:SP:16 asks RTRRMS calibration sites to cover "IRI 2.0 m/km to 12.0 m/km". This
prior is reported beside the estimate; it is **not** multiplied into the interval (it is too thin, and
using it would make the output mostly the prior).

## 3. Synthetic sensitivity analysis for a scooter (`sensitivity.py` → `sensitivity.json`)

R = iri_gc_uncal / IRI_true, by random-vibration integrals (checked against time-domain simulation on 4
draws: 2.93/2.79, 1.44/1.41, 1.68/1.70, 1.43/1.43). Ten assumptions drawn uniformly, 3,000 draws per
speed. The ranges are **engineering assumptions for a scooter**, not measurements of ours (no opened source
gives scooter suspension parameters; the ~25 Hz handlebar line is our own observation, NOISE_FLOOR.md D1):

| Assumption | Range | Golden Car |
|---|---|---|
| body (frame + rider) bounce | 1.2–3.0 Hz, ζ 0.15–0.50 | 1.27 Hz, ζ 0.38 |
| unsprung / sprung mass | 0.10–0.40 (engine on the swing-arm) | 0.15 |
| wheel hop (10–12 in wheels) | 10–25 Hz | 11 Hz |
| handlebar/holder resonance | 18–35 Hz, ζ 0.05–0.30 | rigid |
| phone heave = a·front + (1−a)·rear wheel response | a 0.6–1.2 (>1: pitch lever ahead of the CG), wheelbase 1.20–1.35 m | a = 1 |
| accelerometer gain | 0.97–1.03 | 1 |
| road spectrum waviness w | 1.5–2.8 | 2 |

Control: the Golden Car itself gives R = 1.000 at 3, 6, 10 m/s.

| speed | R p05 | p50 | p95 |
|---|---|---|---|
| 3 m/s | 0.66 | 1.52 | 3.36 |
| 6 m/s | 0.75 | 1.57 | 3.08 |
| 10 m/s | 0.90 | 1.62 | 2.95 |
| pooled 3–12 m/s | 0.77 | 1.59 | 3.08 |

So the uncalibrated index from a scooter most likely **over-reads** IRI by ~1.6× (consistent with the
1.9× in Setiawan & Nurdin), and plausibly anywhere from 0.66× to 3.4×. One-at-a-time at 6 m/s (others at
mid-range, R_mid = 1.62): road waviness (2.52 at w 1.5 → 1.06 at w 2.8), phone position/pitch a (1.21 → 2.28),
body frequency (0.97 → 2.33), handlebar resonance (2.09 at 18 Hz → 1.51 at 35 Hz), body damping
(1.42 → 2.02), unsprung mass (1.31 → 1.77); wheelbase, gain, wheel hop and mount damping move R < 0.1.

## 4. The interval (`iri_interval`)

**Uncalibrated (now), basis `uncalibrated_scooter`:**
- point = iri_gc_uncal / R50(v)
- lo90 = iri_gc_uncal(noise-floor-subtracted, 0.257 m/s²) / R95(v) / k
- hi90 = iri_gc_uncal / R05(v) · k
- k = exp(1.645 · 0.28 · sqrt(100 m / length)): pass-to-pass scatter of a single pass. 0.28 (ln) comes
  from the only repeated blocks (explore.json, median |ln A/B| 0.18–0.27 over 5–6 blocks, exploratory); the
  larger end is used. Vehicle bias is common to the whole survey and is NOT reduced by averaging; only the
  scatter term shrinks with length.

On the 43 drives (`interval_summary.py` → `interval_summary.json`):

| unit | n | point, 5/50/95 % | lo90 (median) | hi90 (median) | hi/lo | PCI_IRI (urban) at median hi / point / lo |
|---|---|---|---|---|---|---|
| 100 m own-track blocks | 383 | 3.3 / 5.4 / 10.3 | 1.69 | 17.6 | 10.6× | 0.6 / 9.7 / 100 (≥ 100 at IRI ≤ 2) |
| ~1 km stretches | 28 | 4.6 / 5.9 / 8.1 | 2.51 | 13.7 | 5.6× | 1.0 / 7.7 / 82 |

**Honest consequence: without a calibration the interval does not meaningfully tighten PCI_IRI.** For a
typical 100 m section the IRI is somewhere between ~1.7 and ~18 m/km, which maps onto almost the whole
PCI_IRI scale; per km it is ~2.5–14 m/km (PCI_IRI ~1–82). The point estimates (median 5.4 m/km per 100 m)
fall inside the published Bengaluru range (1.5–5.6) at its upper end, which is plausible for roads chosen
for their distress but proves nothing. Every hi90 is above the IRC "Poor" limit 3.30 and 87 % of 100 m lo90
values are below the "Good" limit 2.55: the condition class from roughness is undetermined.

**Calibrated (after the survey), basis `calibrated`:** point = exp(a + b ln rms + c ln v), 90 % prediction
interval exp(± t₀.₉₅,ₙ₋ₚ · s_res · sqrt(1 + 1/n)). With s_res ≈ 0.2–0.3 (ln), typical of R² 0.6–0.8
calibrations above, the interval would be about ×/÷1.4–1.7 — roughly 3× narrower than now.

## 5. What the interval does NOT cover

- a scooter (or rider weight, tyre pressure, phone holder) outside the §3 ranges — the ranges are judgement;
- the scooter riding outside the wheel path (IRI is defined per wheel path; the scooter's lateral position
  varies and is unknown);
- non-Gaussian roads: potholes, speed breakers, manholes (the random-vibration inversion assumes a
  Gaussian profile; one hump dominates a 100 m RMS);
- GNSS distance / segment-position errors (±10 m) and segments that straddle a stop;
- a phone that shifted in its holder during a drive;
- speeds below 2.5 m/s (excluded) or above 12 m/s (clamped to the 12 m/s ratios);
- the noise floor beyond the median idle level (NOISE_FLOOR.md: stops are not representative of moving noise).

## 6. What would tighten it now (cheapest first)

1. **A small calibration set** following IRC:SP:16-2019 Table 5.1 for response-type devices: 8–10 straight
   sites of ≥ 100 m (≥ 1 km total) spanning smooth to rough, 3–5 scooter runs each at a steady speed, with a
   reference IRI per site (MERLIN or straight-edge-derived profile, a Class 1/2 profiler, or agency IRI if a
   stretch has a recent NHAI/BBMP survey). That is exactly `fit_calibration()`'s input.
2. **Repeated passes** (3–5 same-lane passes over a few km, two speeds) to replace the 0.28 scatter guess and
   to measure the speed exponent.
3. Logging the vehicle (`survey.vehicle`) and the mount position in every drive record.
