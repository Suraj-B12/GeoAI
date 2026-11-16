# The PCI estimate with a 90 % interval (2026-11-09)

Why: the strict IRC:82-2023 range (unknown parameters counted as PCIᵢ 0 and 100) was ~79 points wide. The owner
asked for a tighter, defensible number. The estimate replaces "unknown" with the best evidence for each parameter,
each with a 90 % interval its evidence supports (reviewed: REVIEW_estimate.md), and combines them by Monte Carlo (`app/pci_irc2023.pci_estimate`, 1,500 draws
per 100 m section, independent parameters). The strict range is kept as the worst case.

| Parameter (weight) | Input per 100 m section | Evidence |
|---|---|---|
| Cracking (0.12) | Attain-calibrated % of road area from the probe scores of all crack types (`app/pci_extent.py`) | TEST: ranks stretches (ρ 0.59), interval held 27/28; Bengaluru widening x2 |
| Ravelling (0.10) | Attain-calibrated % area, interval widened a further x2 | FAILED coverage (0.79 < 0.80) — flagged |
| Patching (0.08) | Attain-calibrated % area | interval only (26/28) |
| Potholes (0.16) | number = runs of pothole frames: lower = runs confirmed by an IMU jolt; point = 0.43 x runs; upper = 1.72 x runs + gate-skip allowance | Attain label precision 0.43 / recall 0.25; scooter jolt rate implies 36–58 % of labelled runs are real |
| Rut depth (0.14) | uniform prior 0–25 mm (`app/pci_rut.py`) | published Bengaluru rut depths 0–23 mm; no phone signal measures rutting |
| Roughness / IRI (0.40) | scooter accelerometer → uncalibrated Golden-Car index / sensitivity ratio, 90 % interval from a 10-assumption sensitivity analysis + pass-to-pass scatter (`app/pci_iri.iri_interval`) | 100 m interval typically 1.7–17.6 m/km; no calibrated motorcycle IRI model exists in the literature |

Precedence: value entered from site > survey calibration (`param_calibration.json`, `iri/iri_calibration.json`) >
the estimates above. Km rows and the stretch = length-weighted mean of the 100 m sections' median, 5 % and 95 %
values (section errors taken as fully correlated: conservative, right for shared calibration error).

## Result on the sample stretches (urban)
| Report | Strict range | Estimate (90 % interval) | Class |
|---|---|---|---|
| 34 (1.55 km) | 9–85 | 52 (40–83) | Fair (Poor–Good) |
| 37 (1.0 km) | 15–89 | 53 (43–86) | Fair (Fair–Good) |
| 38 (1.0 km) | 15–97 | 55 (45–87) | Fair (Fair–Good) |
| 33 (8.3 km) | 5–89 | 52 (41–81) | Fair (Fair–Good) |

Width by parameter in a typical section (report 34, PCI points between the parameter's own 5 % and 95 % ends):
roughness 39.6, rut depth 12.4, potholes 6.2, cracking 4.0, patching 1.5, ravelling 0.9. **Roughness is the gap.**

## What narrows it (in order of effect)
1. A reference IRI on 8–10 straight sites ≥ 100 m (smooth to rough), 3–5 scooter runs each → `fit_calibration`;
   expected ~3x narrower IRI interval. The road survey's IRI does this automatically (`pci_survey_compare.py fit`).
2. Rut depth measured on site (straight edge) or by the survey → entered / calibrated.
3. Expert labels on video segments → pothole precision/recall on Bengaluru data instead of Attain's.

## Update after the critic review and fixes (2026-11-09, final)
Pothole units (runs × P(real | jolt) 0.74 / P(real | no jolt) 0.09, run recall 0.5–0.7, 1–4 units per pothole),
correlated image extents, rut depth as a range that must hold at both ends, IRI physics and scatter combined in
quadrature, speed-matched jolt threshold, extents over imaged segments only. Final sample results (ids 49–56):
50 (31–81), 50 (31–86), 47 (29–82), 50 (31–85), 50 (34–82), 54 (39–89), 50 (31–85), 55 (33–93) — all Fair.
The earlier table above (52 (40–83) …) was before these corrections and is superseded.

## Update 2026-11-16: joint stretch estimate, "likely range", distress photo examples

Owner's request: the 90 % interval (about 50 points wide) is too wide to use; aim for 10-20 points. Photos of
every distress type in the report.

**What was changed (code: `pci_irc2023.pci_estimate_joint`, `pci_chainage.estimate_mean`, `pci_iri._rho_sys`,
`pci_core.POTHOLE_RHO_SECTIONS`; tests `scripts/tests/test_pci_estimate.py`, 6 new):**

1. Km rows and the stretch are no longer the mean of the sections' own 5 % / 95 % ends (= every error fully
   shared between sections). The sections are drawn jointly, 3,000 draws, Gaussian copula per parameter:
   - roughness: correlation between sections = variance of the scooter response ratio R over (R + pass-to-pass
     scatter), 0.60-0.75 on the samples (R is one unknown for the whole survey; the scatter is per section);
   - potholes: 0.5 (judgement: P(real) and recall constants are shared, which runs are real and how big each
     pothole is are per section). Sensitivity 0.25 -> 1.0: median and likely range < 1 point, 5 % end up to 3.7;
   - image extents (one probe calibration) and the rut-depth prior: 1.0, as before; the rut range still has to
     hold at both ends for the 90 % interval.
2. A **likely range** is reported beside the 90 % interval: the central 50 % of the draws (25-75 %). The true
   PCI is as likely inside it as outside. This is the headline range in the PDF, CSV (`pci_est_lo50/hi50`),
   saved-report list and WebGIS; the 90 % interval and the strict range stay next to it.
3. Annex: besides one frame per pothole run and per worst section (<= 24), up to 4 example frames of every
   distress type the model named on the stretch (most confident, each from a different 100 m; <= 60 in total).

**Result (samples regenerated, ids 57-64; before: ids 49-56):**

| Report | Length | Before: median (90 %) | Now: median, likely range (width), 90 % (width) |
|---|---|---|---|
| 57 | 8.31 km | 50 (31-81) | 50, 44-57 (13), 35-74 (39) |
| 58 | 1.55 km | 50 (31-86) | 50, 44-58 (14), 34-79 (45) |
| 59 | 1.00 km | 47 (29-82) | 47, 41-54 (13), 31-76 (45) |
| 60 | 1.00 km | 50 (31-85) | 51, 44-59 (15), 34-79 (45) |
| 61 | 1.00 km | 50 (34-82) | 51, 46-57 (11), 37-78 (41) |
| 62 | 1.02 km | 54 (39-89) | 55, 49-61 (12), 41-82 (41) |
| 63 | 1.00 km | 50 (31-85) | 51, 45-58 (13), 34-78 (44) |
| 64 | 0.80 km | 55 (33-93) | 55, 48-65 (17), 36-88 (52) |

The likely range is 11-17 points wide. The 90 % interval narrowed by 6-11 points (correct correlation), and is
still 39-52 wide.

**What was NOT done, and why (do not re-try without new data):**

- A 90 % interval of 10-20 points is not reachable from the data we hold. Roughness has weight 0.40 and the
  urban PCI_IRI curve falls from ~100 at IRI 2 to ~12 at IRI 5 m/km; the scooter's response ratio R is known
  only to a factor ~4 (0.77-3.08). Even combining that physics prior with the published Bengaluru IRI values as
  a second prior (8 road values, ln-sd 0.49; product ln-sd 0.32) leaves the median section's PCI_IRI between
  ~6 and ~62 at 90 % (22 PCI points from roughness alone). The population prior was again NOT used: it is thin,
  old (2011-2022) and would make the output mostly the prior (UNCERTAINTY.md §2).
- Blind identification of the scooter from the acceleration spectra (to narrow R without a reference) was
  explored (2,244 five-second windows, all drives, by speed bin). Findings: (a) 3-15 Hz is roughly flat,
  consistent with road waviness w ~ 2; (b) below ~1.5 Hz the spectrum does not fall as a road-heave signal
  should (flat instead of +20 dB/decade): that band is not road response (rider motion, pitch/gravity leakage)
  and inflates the 0.5-15 Hz RMS on smooth roads; (c) a spectral peak moves with speed (about 1.6-2.5 Hz at
  2-3.5 m/s, 3-4 Hz at 6.5-8 m/s, about 0.5-0.7 cycles/m), compatible with wheel rotation (circumference
  ~1.5-1.9 m) or the wheelbase filter. No body-bounce resonance could be read off without a model that also
  explains (b) and (c). Not usable as evidence yet; a lead for later (raising the band's lower edge to ~1.5 Hz
  is the first thing to test once a reference IRI exists).

**What makes the 90 % interval 10-20 points (cheapest first):**
1. Reference IRI on 8-10 straight sites (>= 100 m each, smooth to rough), 3-5 scooter passes each ->
   `pci_iri.fit_calibration` (or the survey: `pci_survey_compare.py fit`). Expected roughness interval x/÷ 1.4-1.7
   instead of x/÷ ~3: about 15-20 PCI points at 90 %.
2. Rut depth with a 2-3 m straight edge on the same sites (removes ~12 points of range).
3. Two measurements of the scooter itself narrow R without any road reference: bounce frequency and damping
   (push down on the seat with the rider on, record the phone for 10 s) and the phone's position along the
   wheelbase. These are the 2nd and 3rd largest terms of R (body frequency 0.97-2.33, position 1.21-2.28).
