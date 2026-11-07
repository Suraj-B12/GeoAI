# Validation — A→B chainage and IRC:82-2015 condition rating

Generated 2026-11-07T17:09:54+00:00 by `venv/Scripts/python.exe scripts/pci_validation.py --database geoai --redo samples` on database `geoai` (data as of 2026-10-30T15:05:00.064449Z). Every number below is from that run (`eval_results/pci/validation.json`).

## 1. IRC arithmetic

- App-4 worked example (Table 5.1): ratings cracking 1.2, ravelling 1.2, potholes 3.0, shoving 2.1, patching 1.5, settlement 1.2, rut_depth 1.2; final **1.3 Fair** (IRC prints 1.3 Fair). IRC prints 1.3 for the rut-depth row; the inferred rule gives 1.2 (RESEARCH §1.5); final 1.3 Fair either way.
- Band boundaries: 116 checks, 0 failures.
- Truncation 2−(9−5)/5 → 1.2; rounding 0.75·1.5 → 1.1, 0.75·1.2 → 0.9.
- Table 5.3 bracket, every segment pothole + alligator: 0.8–1.8 (Poor–Fair); no distress: 1.4–2.4 (Fair–Good). Poor is reachable at the low end.

## 2. Chainage on synthetic drives (no smoothing)

| Shape | true m | measured m | error m |
|---|---|---|---|
| straight_1000 | 1000.0 | 1000.0 | -0.0 |
| arc_r100_90deg | 157.08 | 157.06 | -0.02 |
| turn90_r15 | 623.56 | 623.45 | -0.12 |
| turn180_r8 | 625.13 | 624.7 | -0.44 |

Correlated 2-D AR(1) GNSS noise on a 1,000 m straight drive (50 seeds each): length error (m).

| noise | mean | sd | p90 | max |
|---|---|---|---|---|
| sigma4_corr50m | 63.36 | 7.0 | 69.47 | 84.36 |
| sigma4_corr100m | 31.96 | 5.62 | 38.41 | 43.53 |
| sigma4_corr200m | 16.78 | 5.15 | 23.42 | 24.69 |
| sigma6_corr50m | 151.36 | 12.41 | 166.78 | 196.48 |
| sigma6_corr100m | 73.88 | 9.06 | 83.38 | 98.2 |
| sigma6_corr200m | 37.45 | 7.91 | 47.4 | 50.87 |

The AR(1) rows measure the method on RAW unsmoothed noise; they are not the chainage error of our data (see Statement 2).

- A/B swap: length difference 0.0 m, pothole E_lo difference 0.0.
- Rule tests: test_refusals pass; test_via_points_and_swap_symmetry pass; test_gap_rules pass; test_parallel_road_not_joined pass; test_flyover_refused_level_crossing_used pass; test_u_turn_and_loop_attribution pass.

## 3. Test–retest on real overlaps

Rule: nearest segment of each later drive within 6 m, |dheading| < 30 deg, |d alt| <= 4 m. Matched pairs 84 over 4 drive pairs; both classified 67.

| Label | n | agreement | κ | positive (earlier / later / both) |
|---|---|---|---|---|
| Distress | 67 | 0.925 | 0.756 | 15 / 10 / 10 |
| Potholes | 67 | 0.910 | 0.674 | 12 / 10 / 8 |
| Cracking | 67 | 0.925 | 0.505 | 6 / 5 / 3 |

κ 95 % block-bootstrap interval (blocks of 20 consecutive pairs, 8 blocks): Distress [0.0, 0.962] (10 % of resamples have positives in only one drive, where κ is exactly 0); Potholes [0.0, 0.908] (10 % of resamples have positives in only one drive, where κ is exactly 0); Cracking [0.0, 0.848] (10 % of resamples have positives in only one drive, where κ is exactly 0). The lower ends of 0 are that artefact, not a measured lower agreement: with so few blocks the interval is **not informative**. Positive agreement with Wilson 95 % intervals (n = positives):

- Distress: later drive also positive when the earlier was 10/15 (0.67 [0.42, 0.85]); earlier also positive when the later was 10/10 (1.00 [0.72, 1.00]).
- Potholes: later drive also positive when the earlier was 8/12 (0.67 [0.39, 0.86]); earlier also positive when the later was 8/10 (0.80 [0.49, 0.94]).
- Cracking: later drive also positive when the earlier was 3/6 (0.50 [0.19, 0.81]); earlier also positive when the later was 3/5 (0.60 [0.23, 0.88]).

Drive pairs (hours apart): 0.9 h, 37 pairs; 4.86 h, 29 pairs; 0.22 h, 15 pairs; 17.32 h, 3 pairs.

100 m blocks fully matched in both drives and rated in both: **2** (2 drive pairs).
- cracking: band agreement 1.000, mean |ΔE_lo| 0.0 percentage points (n = 2; not interpretable).
- potholes: band agreement 0.500, mean |ΔE_lo| 5.0 percentage points (n = 2; not interpretable).
- bracketed final class agreement 1.000 (n = 2; not interpretable — do not quote).
- Matched-segment offsets (bounded by the 6 m rule): along-track p50 0.76 m, p90 2.02 m; cross-track p50 1.58 m.

## 4. Independent length (OpenStreetMap)

Status: done. Map data (c) OpenStreetMap contributors, ODbL 1.0 (https://www.openstreetmap.org/copyright), via the Overpass API. OSM routes are shortest paths on an undirected drivable-way graph through our own points, snapped to the nearest OSM vertex; they approximate the driven route, they do not survey it.

**Read as inconclusive.** The waypoints snap to the nearest OSM *vertex* (not the nearest way), median 5.5–19.2 m away per sample; a snap onto a parallel or side road forces a detour, so the OSM route is often longer than the road driven (ours / OSM 0.21–1.00). This is not an independent length measurement and no chainage-length accuracy is derived from it.

| Sample | surveyed m (ours) | OSM route m | ours / OSM | waypoint snap p50 / p90 m |
|---|---|---|---|---|
| longest_drive | 8309.6 | 10342.6 | 0.8034 | 9.8 / 29.1 |
| random_1km_2026-10-27_1 | 1332.8 | 1412.3 | 0.9437 | 8.1 / 23.1 |
| random_1km_2026-10-27_2 | 953.8 | 1166.1 | 0.8179 | 19.2 / 30.8 |
| random_1km_2026-10-27_3 | 942.4 | 4415.4 | 0.2134 | 12.2 / 19.3 |
| random_1km_2026-10-29_1 | 770.3 | 910.6 | 0.8459 | 11.2 / 28.1 |
| random_1km_2026-10-29_2 | 910.7 | 1262.0 | 0.7216 | 12.8 / 23.8 |
| random_1km_2026-10-29_3 | 1003.9 | 1001.9 | 1.0020 | 5.5 / 12.6 |
| two_gaps | 632.6 | 724.9 | 0.8727 | 6.4 / 23.7 |

Time-adjacent gaps: OSM route / straight line over 34 of 36 gaps with an OSM route: median 1.095, p90 1.709, max 3.167. This supports printing gaps as lower bounds ('at least x m'); the same snapping caveat applies.

## 5. Run lengths (consecutive positive segments of one drive)

| Label | positive segments | runs | mean segments/run | p90 | share of positives in runs ≥ 3 |
|---|---|---|---|---|---|
| Potholes | 770 | 502 | 1.53 | 3.0 | 0.295 |
| Cracking (any) | 761 | 428 | 1.78 | 3.0 | 0.424 |
| Alligator Cracking | 148 | 108 | 1.37 | 3.0 | 0.290 |
| Longitudinal Cracking | 720 | 405 | 1.78 | 3.0 | 0.415 |
| Transverse Cracking | 275 | 195 | 1.41 | 2.0 | 0.262 |
| Bleeding | 355 | 282 | 1.26 | 2.0 | 0.158 |
| Edge Breaking | 81 | 67 | 1.21 | 2.0 | 0.037 |

## 6. Class sensitivity (all 100 m blocks of all drives, Table 5.3)

- Blocks 411; coverage {'rated': 404, 'low_coverage': 4, 'insufficient': 3}; rated under the 80 % rule 404, under the 50 % rule 408.
- Bracketed final class: {'Poor–Fair': 189, 'Fair–Good': 144, 'Poor–Good': 75}; single class (both ends agree) 0.
- Low end: {'Poor': 264, 'Fair': 144}; high end: {'Good': 219, 'Fair': 189}; bracket width {'n': 408, 'mean': 1.1, 'sd': 0.14, 'min': 1.0, 'p50': 1.0, 'p90': 1.4, 'max': 1.5}.
- Cracking band {'Good': 164, 'Fair': 90, 'Poor': 77, 'Fair–Good': 60, 'Poor–Fair': 17} (E_lo vs E_hi band differs on 77); pothole band {'Poor': 240, 'Poor–Good': 118, 'Good': 50} (differs on 118).
- Σwr/Σw class {'Fair–Good': 317, 'Poor–Good': 91}; assessed-only {'Fair': 139, 'Poor': 110, 'Fair–Good': 78, 'Good': 40, 'Poor–Fair': 36, 'Poor–Good': 5}; no skip correction {'Fair–Good': 215, 'Poor–Fair': 193}.
- blocks of each drive in k order (no routing, one pass), 100 m by centre distance, tail < 50 m merged; Table 5.3 (urban).

## Sample reports (pre-registered rule, `samples_rule.json`)

| id | rule | requested trip m | routed m | coverage | gaps | cracking band (E) | pothole band (E) | runs/km | overall (secondary) |
|---|---|---|---|---|---|---|---|---|---|
| 17 | longest_drive | – | 8309.6 | 0.992 | 0 (≥ 0 m) | Poor (18.1–22.4) | Poor (5.6–9.9) | 7.0 | Poor–Fair (ravelling, settlement and depression, rut depth not assessed) |
| 18 | random_1km_2026-10-27_1 | 1000 | 1551.7 | 0.867 | 3 (≥ 219.0 m) | Fair–Good (4.5–6.0) | Poor (12.3–13.9) | 16.4 | Poor–Fair (ravelling, settlement and depression, rut depth not assessed) |
| 19 | random_1km_2026-10-27_2 | 1000 | 1004.3 | 0.961 | 1 (≥ 50.5 m) | Poor (24.9–25.6) | Poor (12.4–13.1) | 19.7 | Poor–Fair (ravelling, settlement and depression, rut depth not assessed) |
| 20 | random_1km_2026-10-27_3 | 1000 | 1004.4 | 0.940 | 1 (≥ 62.0 m) | Fair (7.9–9.2) | Poor (16.4–17.7) | 22.2 | Poor–Fair (ravelling, settlement and depression, rut depth not assessed) |
| 21 | random_1km_2026-10-29_1 | 1000 | 1003.7 | 0.782 | 2 (≥ 233.4 m) | Good (1.3–2.0) | Poor (5.1–5.8) | 6.4 | Fair–Good (ravelling, settlement and depression, rut depth not assessed) |
| 22 | random_1km_2026-10-29_2 | 1000 | 1024.3 | 0.888 | 1 (≥ 113.6 m) | Good (0.5–3.5) | Poor (1.1–4.0) | 1.1 | Fair–Good (ravelling, settlement and depression, rut depth not assessed) |
| 23 | random_1km_2026-10-29_3 | 1000 | 1003.9 | 0.996 | 0 (≥ 0 m) | Poor (27.0–30.4) | Poor (2.0–5.4) | 4.0 | Poor–Fair (ravelling, settlement and depression, rut depth not assessed) |
| 24 | two_gaps | – | 795.7 | 0.795 | 2 (≥ 163.1 m) | Fair (6.3–6.6) | Poor (7.1–7.4) | 12.6 | Poor–Fair (ravelling, settlement and depression, rut depth not assessed) |

Requested = metres along the drives' own trip (rule b); routed = the A→B path the router chose (it may use other drives, links and gaps, so it can differ).

## Statements for the paper (from the numbers above)

1. **The Appendix-4 overall class cannot be determined from this pipeline.** On 408 rated 100 m blocks the bracketed final is a single class on 0 blocks; the bracket width is at least 1.0 on every block (median 1.0). The overall class must not be quoted as a condition. Only the **cracking** band carries graded information ({'Good': 164, 'Fair': 90, 'Poor': 77, 'Fair–Good': 60, 'Poor–Fair': 17}). The **pothole** band per 100 m is in effect a flag: {'Poor': 240, 'Poor–Good': 118, 'Good': 50} — Fair on 0 blocks, because one flagged 5 m segment is 5 % of a section against an urban Poor limit of 0.5 %. Read it as 'pothole flagged / not flagged' and report the amount as runs per km (per block: {'n': 408, 'mean': 12.2, 'sd': 14.08, 'min': 0.0, 'p50': 10.0, 'p90': 30.3, 'max': 70.0}; 240 blocks with at least one run). Never report a distribution of pothole bands.
2. **The synthetic AR(1) noise result is not our chainage error.** It measures the method on raw, unsmoothed GNSS noise; the real segment centres come from the video pipeline's own fitted track (centre path / s-path 0.999 median over 42 drives; not independent). No chainage-LENGTH accuracy is claimed: the OpenStreetMap comparison in §4 is inconclusive (vertex snapping), so an independent length reference is still missing. Gaps are printed as straight lines; the OSM route over the time-adjacent gaps is median 1.095x and up to 3.167x the straight line, so chainage after a gap can be short by up to about that factor times the gap (same snapping caveat).
3. **Repeatability between drives is measured only at the segment level.** 67 matched segment pairs, κ (Distress) 0.756 [0.0, 0.962], 8 blocks — not informative; only 2 overlapping 100 m blocks, so class repeatability is NOT measured. The paired drives are 0.9 h apart (37 pairs), 4.86 h apart (29 pairs), 0.22 h apart (15 pairs), 17.32 h apart (3 pairs) — same phone, same model, not different conditions.
4. **Skip correction for potholes.** Audited skips: 13/161 distressed (0.081 [0.048, 0.133]), 9/161 with a pothole label (0.056 [0.030, 0.103], Wilson 95 %). Under the urban NIL rule the 13.3 % bound makes 118 of 408 blocks 'Poor–Good' for potholes with no pothole label at all; with the pothole-specific upper limit instead: {'Poor': 240, 'Poor–Good': 112, 'Good': 50, 'Fair–Good': 6}.

## 7. Limitation

There are **0** expert labels on the video segments and no Bengaluru validation of any label. Everything above measures arithmetic, geometry and repeatability between drives — not accuracy against ground truth. Repeatability between drives of the same model on the same road is not correctness.
