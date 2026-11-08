# Roughness from the phone IMU — validation without a reference IRI

Method: app/pci_iri.py (method_fp printed by scripts/iri_from_imu.py), reasoning in RESEARCH.md.
Script: eval_results/pci/iri/validate_iri.py -> validation.json. There is **no reference IRI** for any
Bengaluru road yet; nothing below measures accuracy. It measures whether the feature is physically sane,
repeatable and speed-normalised, so that a later calibration on the survey has something stable to map.

## Part A — Pre-registered criteria

**Written 2026-11-08T09:30Z, before any real-data validation statistic was computed.** At this time the
only real-data numbers seen were: one drive's spectrum (used to set the 15 Hz band, RESEARCH §5) and that
drive's counts of segments with features. The synthetic checks in scripts/tests/test_pci_iri.py had been
run (they use other roads/seeds than V1 below). Criteria are not changed after results are seen; any later
analysis is labelled exploratory.

Unit for all 100 m statements: aggregate() of the segments (rms = sqrt of time-weighted mean square,
speed = length / time, iri_gc_uncal = length-weighted mean). Segments with an error, no rms or the
`low_speed` flag (< 2.5 m/s) are excluded everywhere. Bootstrap CIs: 2,000 resamples, clustered by drive
(V2, V5) or by drive pair (V3, V4), percentile 95 %.

### V1 Synthetic quarter-car check (known IRI)
12 roads, 600 m each: Gd(n0 = 0.1 c/m) ∈ {4, 16, 64, 256}·10⁻⁶ m³ × waviness w ∈ {1.8, 2.0, 2.3}, seed =
road index. True IRI per 100 m by direct Golden-Car simulation of the profile (iri_from_profile).
- Vehicle A: the Golden Car itself, rigid mount, no noise, 6 m/s.
- Vehicle B: different car (k1 520, k2 40, c 3.6, mu 0.12 per unit sprung mass), 18 Hz mount resonance
  (ζ 0.1), white noise 0.03 m/s² per axis, accelerometer scale +10 %, 25° tilt, 0.2 ms timestamp jitter,
  constant speed per road drawn uniform 3–10 m/s (seed 99).
- **S1 PASS if** Vehicle A on the w = 2.0 roads: median |iri_gc_uncal / IRI_true − 1| ≤ 0.10.
- **S2 PASS if** Vehicle B, all 12 roads: Spearman ρ(iri_gc_uncal, IRI_true) over 100 m sections ≥ 0.90.
- S3 (report only): Vehicle B median and IQR of iri_gc_uncal / IRI_true (expected biased: the calibration problem).
- **S4 PASS if** Vehicle B on the w = 2.0, Gd = 64·10⁻⁶ road driven at 3, 6 and 10 m/s: coefficient of
  variation of the road-mean rms_av_norm across the three speeds ≤ 0.15. (Reported for w = 1.8 and 2.3
  without criterion: rms/sqrt(v) is exact only for w = 2.)

### V2 Idle noise floor
Idle = GNSS Doppler speed < 0.3 m/s for ≥ 3 s (first/last second dropped), engine assumed running.
- **PASS if** median over drives of idle rms_av ≤ 0.25 × median rms_av of all moving segments (all drives).
- Not testable if fewer than 3 drives have an idle period.

### V3 Repeatability between passes over the same road, same direction
Pairs: segments of two different drives with centres < 6 m apart and camera headings < 30° apart (the same
same-road rule as app/pci_chainage.py), both usable. Each segment of drive A takes its nearest such partner
in drive B. 100 m blocks: consecutive k of drive A in groups of 20 (k // 20); a block counts when ≥ 16 of its
segments are matched; block value = aggregate of A's matched segments vs aggregate of their partners.
Primary feature rms_av_norm.
- **R1 PASS if** Spearman ρ between the two passes over all blocks ≥ 0.70 (needs ≥ 10 blocks, else not testable).
- **R2 PASS if** median |ln(A/B)| over blocks ≤ ln 1.25 (median disagreement within ±25 %).
- R3 (report only): the same ρ at 5 m segment level; R1/R2 for iri_gc_uncal and rms_av (raw).

### V4 Speed dependence and its normalisation
Within-location (removes road-type confounding): for the V3 segment pairs, OLS slope β of
ln(rms_av_A / rms_av_B) on ln(v_A / v_B), pairs with |ln(v_A/v_B)| ≥ ln 1.1.
- **PASS (sqrt(v) normalisation adequate) if** |β − 0.5| ≤ 0.15 or the 95 % CI of β contains 0.5.
- Report only: Spearman ρ of rms_av and of rms_av_norm with speed over all segments (confounded by road type).

### V5 Pothole-labelled segments
Positives: segments whose linked assessment's distress_types contains 'Potholes'. The camera sees up to
~30 m ahead of the segment, so each segment's score = max rms_av_norm over segments k … k+6 of the same drive
(≥ 4 usable). Negatives: classified segments with no pothole label (assessment present).
- **PASS if** AUC(score; positives vs negatives) ≥ 0.60 and its 95 % CI (bootstrap clustered by drive) excludes 0.5.
- Report only: the same with peak_av, and with the segment's own value (no look-ahead window).
- Caveat stated in advance: pothole labels are unvalidated VLM output and look over-called (shadows) —
  CLAUDE.md, PCI open items — so a FAIL may be the labels' fault and a PASS is weak evidence.

## Part B — Results (run 2026-11-08, method_fp e9e1bc016ba5fa70, database geoai)

Data: 43 drives, 8,402 segment rows; 8,110 usable (93 not covered = GNSS gaps, 26 outside the IMU record,
173 below 2.5 m/s). Flags: sparse_anchors 90, stop_in_window 35. Segment speeds 5th/50th/95th percentile
3.6 / 6.8 / 10.0 m/s (13 / 25 / 36 km/h). Raw numbers: validation.json (pre-registered), explore.json
(exploratory). Run log: run.log.

| Check | Pre-registered rule | Result | Verdict |
|---|---|---|---|
| S1 Golden Car, w = 2 roads | median abs. rel. error ≤ 0.10 | 0.025 (ratios 0.94–1.01 per road) | **PASS** |
| S2 other car + mount + noise, 12 roads | Spearman ≥ 0.90 (48 sections) | 0.991 | **PASS** |
| S3 other car, absolute | report | median ratio 1.00, IQR 0.88–1.10 | — |
| S4 speed normalisation, w = 2 | CV across 3/6/10 m/s ≤ 0.15 | 0.004 (w = 1.8: 0.047, w = 2.3: 0.067) | **PASS** |
| V2 idle noise floor | median idle ≤ 0.25 × median moving | 0.257 / 0.657 = **0.39** (13 drives, 167 s idle) | **FAIL** |
| V3 R1 repeatability, 100 m | Spearman ≥ 0.70, ≥ 10 blocks | only 3 blocks (2 drive pairs) | **not testable** |
| V3 R2 repeatability, 100 m | median |ln A/B| ≤ ln 1.25 | 3 blocks | **not testable** |
| V3 R3 segment level | report | Spearman 0.46 (n = 90 segment pairs) | — |
| V4 within-location speed exponent | |β − 0.5| ≤ 0.15 or CI ∋ 0.5 | β = 0.15, CI [−0.06, 3.34], 60 pairs from 4 drive pairs | PASS by the letter — **uninformative** (CI spans everything) |
| V5 pothole-labelled segments | AUC ≥ 0.60 and CI > 0.5 | AUC 0.684 [0.650, 0.718] (731 pos / 6,069 neg) | **PASS** |
| V5 variants | report | peak_av 0.663 [0.633, 0.694]; own segment 0.712 [0.680, 0.749] | — |

### What this does and does not show

- **The arithmetic is right.** With the Golden Car as the vehicle the pipeline (gravity projection, tilt,
  resampling, band-pass, distance mapping, aggregation) recovers the profile's IRI within 2.5 % (median); and
  rms/sqrt(v) is speed-invariant on white-slope roads, ±5–7 % for w = 1.8 / 2.3.
- **The synthetic "other car" is NOT evidence that absolute values transfer.** Its ratio came out 1.00, but
  it is still a quarter car with similar modes; a real car has pitch, seats, a windshield suction mount and
  a different body response. S3 says nothing about the survey car.
- **Noise floor FAIL.** While the car stands (GNSS speed < 0.3 m/s) the 0.5–15 Hz band still carries a median
  0.26 m/s² RMS against 0.66 moving — about 15 % of the moving variance at the median, more on smooth roads.
  The idle level varies 10× between drives (0.13 to 1.50 m/s²); the 1.50 drive has most of it at 0.5–5 Hz
  (explore.json), which looks like movement while "standing" (Doppler speed is unreliable below ~0.5 m/s, or
  the phone was handled), not engine vibration. Consequences: (i) on smooth roads the feature is partly
  engine/mount noise, compressing the low end of any calibration; (ii) the calibration should include an
  intercept (it does: log-linear with intercept) and the survey's calibration half must span smooth roads.
  Not done (would be a change after seeing results): subtracting a per-drive noise floor.
- **Repeatability could not be tested.** Only 4 pairs of drives cover the same road in the same direction
  (175 matched segments), giving 3 qualifying 100 m blocks. Exploratory (explore.json): relaxing to ≥ 8
  matched segments per block gives 5–6 blocks with Spearman 0.70 (6 m) / 0.94 (10 m) and median
  disagreement 19–31 % — too few to conclude anything. At 5 m the two passes agree only moderately
  (Spearman 0.46), as expected for single wheel-path GNSS-positioned 5 m windows (position σ 4–6 m).
  **Needed:** 3–5 repeated same-lane passes over a few km (ideally at two speeds) before the survey.
- **Speed exponent unknown.** The within-location estimate rests on 60 segment pairs from 4 drive pairs and
  its CI is useless. Across all segments rms_av rises weakly with speed (Spearman +0.12) and rms_av_norm
  falls (−0.21), suggesting sqrt(v) over-corrects (true exponent < 0.5), but that comparison is confounded by
  road type. The calibration therefore fits the speed exponent itself (log_speed term) rather than trusting
  sqrt(v); the repeated-pass drives above would pin it down.
- **Potholes PASS (weakly informative).** Segments the VLM labelled 'Potholes' have higher roughness
  (median rms/sqrt(v) 0.64 vs 0.39 over the look-ahead window; AUC 0.68–0.71). The labels are unvalidated
  and look over-called, so this says the feature responds to *something* the camera also sees, not that it
  measures IRI.

### Descriptive (not IRI): Golden-Car-equivalent index on the survey drives

383 own-track 100 m blocks with ≥ 16 usable segments: iri_gc_uncal 5th/25th/50th/75th/95th percentile =
5.1 / 6.6 / 7.9 / 9.5 / 13.5 "m/km"; rms_av 0.49 / 0.82 / 1.46 m/s² (5th/50th/95th). These include the
noise floor above and every assumption in RESEARCH §4; they must not be shown as IRI.

### Bottom line

The feature is computed correctly, responds to road damage, and has a defined calibration path. It is **not
yet shown to be repeatable** (not enough repeated passes) and its noise floor is high on smooth roads. No
IRI may be reported until (a) repeated passes show 100 m repeatability and (b) the survey's calibration half
supplies reference IRI.
