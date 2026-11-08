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
