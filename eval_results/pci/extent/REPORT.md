# Probe scores → IRC:82-2023 extent (% area) and pothole number — Attain calibration (2026-11-09)

Pre-registered in `PREREG.md` (sha256 in `PREREG.sha256`, 2026-11-08T18:38Z) before the DEV fit and the single TEST
run; TEST is spent for this question. Code `scripts/pci_attain_extent_calibration.py`, model `app/pci_extent.py`,
calibration `attain_extent_calibration.json`, raw results `dev_fit.json`, `test_results.json`, `cv_results.json`,
`transfer.json`. (Written by the coordinator from the workstream's final report; numbers as reported there.)

## Headline
- The old "% of length flagged" rule overstates area ~10x: TEST block MAE 70 pp for cracking (true mean 7.5 %),
  50 pp ravelling, 10 pp patching, 12 pp potholes; calibrated estimates are within ~1–3 pp.
- Cracking: interval only — ranks stretches (Spearman 0.59 [0.26, 0.82]), 90 % interval held on 27/28 blocks, MAE
  2.95 vs 3.11 for a constant (difference CI [−1.11, +0.74] includes 0, criterion C1 fails).
- Ravelling: FAILS — coverage 22/28 = 0.79 (< 0.80), not better than a constant, rho 0.20 [−0.19, 0.54].
- Patching, potholes: interval only (flat curves; coverage 26/28, 27/28).
- Transfer Attain → Bengaluru is weak: our cracking scores resemble Attain frames WITHOUT cracks (KS D 0.10 vs
  those, 0.61 vs all Attain); pothole / patching scores are higher on ours (D 0.23 / 0.17; shadows). Declared
  widening for Bengaluru: x/÷2 where D ≥ 0.20, x/÷1.5 otherwise — assumptions, not measurements.
- The pothole number from the Attain curve is NOT usable (its floor, 0.072 % area, is Attain's base rate: a perfect
  road would still get ~2.5 potholes per 100 m). Potholes stay on label runs.

## Mapping (MAPPING.md, IRC:82-2023 text)
Cracking = alligator + linear + block (App-1 item 1; §7.3.5.1 block = transverse form); edge and hairline cracks are
not in App-1's PCI list and not labelled in Attain. Ravelling = Raveling only (§7.5.2; hungry surface §7.2.4 and
stripping §7.5.1 are separate). Patching = patch and utility cut (§7.5.4). Potholes eq. (2), §7.5.3.4. All Stage 2
labels of a frame were considered; the label-flag model lost DEV selection (cracking) or was not eligible.

## Ground truth / ROI
Union box area inside a fixed per-frame-size road trapezoid (checked by eye on 21 frames, `roi_check/`), mean ROI
0.81 of the image. Whole-data means (% of ROI): cracking 7.69, ravelling 0.46, patching 0.69, potholes 0.134.
Full-image ROI agrees within ~20 %; the box-hull ROI triples every extent (rejected).

## TEST (28 blocks of 20 frames, 547 frames, block bootstrap B = 2000)
| | true mean % | MAE | Spearman | constant MAE | % flagged MAE | 90 % coverage | verdict |
|---|---|---|---|---|---|---|---|
| Cracking | 7.45 | 2.95 | 0.59 | 3.11 | 70.1 | 27/28 | interval only |
| Ravelling | 0.49 | 0.69 | 0.20 | 0.61 | 50.3 | 22/28 | FAILS |
| Patching | 0.77 | 0.92 | 0.40 | 0.92 | 10.5 | 26/28 | interval only |
| Potholes | 0.15 | 0.15 | 0.41 | 0.15 | 12.0 | 27/28 | interval only |
Grouped 5-fold CV (847 frames, 43 blocks) agrees: only cracking carries real signal.

## Viewpoint (car windshield vs our scooter)
Our frames: 3840x2160, pitched 18–24° down, mostly near road, horizon at or above the top edge; the rider's
shadow, strong shade patches and close traffic are common. Attain 1920x1080 frames show 20–50 m of road with the
horizon mid-frame. Shadows push the pothole / patching probes up; the continuous survey has far fewer distressed
frames than Attain (collected for distress) — the largest bias is this prior shift.

## Limitations
Box area is not measured ground area; small splits (DEV 15 blocks, TEST 28); Tehran / car / distress-selected vs
Bengaluru / scooter / continuous; one Attain frame treated as one 5 m segment; one lane imaged; third-party copy
of IRC:82-2023.
