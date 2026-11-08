# Critic review — the PCI ESTIMATE with a 90 % interval (2026-11-09)

Scope: `eval_results/pci/ESTIMATE.md`, `app/pci_irc2023.pci_estimate`, `app/pci_core.pci_estimate_unit` /
`_extent_inputs` (+ pothole / ravelling constants), `app/pci_chainage.py` (estimate_mean, _run_impact, jolt
threshold), `app/pci_extent.py` + `eval_results/pci/extent/*`, `app/pci_iri.iri_interval` +
`eval_results/pci/iri/{UNCERTAINTY,NOISE_FLOOR}.md`, `app/pci_rut.py` + `RUT_PRIOR.md`. Survey vehicle: scooter.

Evidence used: the 51 rated 100 m sections of the three longest drives (A = first segment, B = segment 400, urban),
built read-only from the production DB `geoai` in a scratch process (no writes, no server touched); their
estimate inputs were re-run through variants of `pci_estimate` (scratch scripts `analyse.py`, `analyse2.py`,
`why_missing.py`). "Median section" = median over these 51 sections of (median, lo90, hi90).

| Variant (same inputs) | median | lo90 | hi90 | width |
|---|---|---|---|---|
| as built (independent, split-normal, rut uniform) | 55.0 | 44.3 | 87.9 | 44.0 |
| rut prior as a judgement range (ends), **now the default** | 55.0 | 42.1 | 96.5 | 54.1 |
| all six parameters fully correlated (comonotone) | 54.1 | 38.0 | 99.4 | 60.6 |
| IRI split-log-normal instead of split-normal, same 5/50/95 % | 56.2 | 45.0 | 89.6 | 44.8 |

Point contributions (median section, weight × PCIᵢ at the point inputs): cracking 10.3, ravelling 9.9, potholes
16.0, patching 7.9, rut 4.6, IRI 5.4 → **the headline "Fair ~55" is mostly "no image distress found" (44 of 55
points) plus an assumed IRI (5.4) and the middle of an assumed rut range (4.6).**

## Findings, by severity

### High

**E1. Pothole number: a "run" is not an IRC pothole unit, and the Attain rates are used inconsistently (report).**
- IRC:82-2023 eq. (2) and §7.5.3.4: number = pothole AREA / 0.1 m² (a ~300 mm "foot-sized" unit; 0.5 m² = 5).
  The code counts one run of pothole-labelled 5 m frames as ONE unit. A run holds one or more potholes of unknown
  size; common Bengaluru potholes of 0.3–1 m² are 3–10 units. Urban PCI_pothole: 1 → 80.3, 3 → 40.1, 5 → 30.9.
  So the "upper = 1.72 × runs" is not an upper bound in IRC units.
- Precision 0.43 / recall 0.25 are FRAME-level rates of the free-form Stage 2 list on Attain (car, Tehran,
  distress-selected). Expected true count given n detections is n × P / R = 1.72 n — that is the point estimate
  under those rates, not the upper bound; "point = 0.43 n" counts only the true detections and ignores the 75 %
  the label misses. The two numbers are used inconsistently with each other.
- **n = 0 (35 of 51 sections):** the interval is [0, skip allowance] — recall is ignored exactly where it matters.
  With recall 0.25 and independent misses, zero detections are still consistent (p ≥ 0.05) with up to ~10 real
  potholes. Allowing hi90 = 10 for n = 0 lowers lo90 by ~9–10 PCI points on those sections (scratch run:
  e.g. 54.5 (45.0–87.4) → 51.0 (35.3–85.0)).
- Frame-level rates do not map to runs: a pothole seen in 2–4 consecutive forward frames raises run-level recall
  (and shadows raise run-level false positives) by unknown amounts.
- Fix (decision for the coordinator): state the unit plainly ("label runs, not IRC units"), widen the upper end by
  an area factor (or show the pothole parameter as a range with an explicit "size not measured"), give n = 0 a
  recall-based upper end, and replace Attain frame rates by run-level rates from expert labels on video.

**E2. The rut prior was turned into a probability distribution its own source says it is not — FIXED.**
`RUT_PRIOR.md` §3 / `pci_rut.PRIOR_BASIS`: 0–25 mm is "a judgement range covering every value found, NOT a 90 %
probability interval … the PCI range should be computed from the two ends". `pci_estimate_unit` drew it uniformly,
so the "90 %" PCI interval averaged rut away (a road with no rut, PCI_rut 98.9, and one with 25 mm, 10.6, were
both inside the draws but the interval did not reach either). Now `dist: "range"`: the interval is the envelope of
the runs with rut fixed at each end; the median still uses the middle and says so (`median_assumes`). Effect on the
median section: 44.3–87.9 → 42.1–96.5. Sample stretches: 55.8 (41.7–94.9), 55.6 (42.4–95.0), 56.3 (41.7–95.8).
This WIDENS the interval; it is what the evidence supports (5 roads in the literature).

**E3. Independence across parameters is not "somewhat" narrower — it is ~15 points narrower (report).**
With all six parameters at the same quantile the median section's interval is 38.0–99.4 (width 60.6) vs 44.3–87.9
(44.0). Real dependence: cracking / ravelling / patching come from the same probe scores, the same Attain
calibration and the same declared domain factor, so their errors are positively correlated; IRI and potholes are
physically correlated (rough roads hold potholes). ESTIMATE.md and the docstring said "would widen the interval
somewhat" (docstring now says "markedly"). Suggest: treat the three Attain-extent parameters as one comonotone
group (shared calibration error), the rest independent, and print the fully-correlated interval as the
"worst case" beside the strict IRC range.

**E4. The jolt "confirmation" is weak and confounded with speed breakers (report + lower bound FIXED).**
- Threshold = 90th percentile of non-pothole-labelled segments, so per segment 10 % exceed by construction; the
  ±1 window (≥ 3 segments) gives a base rate of **23.8 % measured** (7,371 non-pothole segments with both
  neighbours; 27 % if independent), more for longer runs.
- **The threshold is speed-confounded:** share of segments above it by speed: < 4 m/s **27.2 %**, 4–7 m/s 13.5 %,
  7–10 m/s 9.6 %, > 10 m/s 4.2 %. Slow segments are where scooters cross speed breakers, rough junctions and
  manholes; there the window base rate is ~60 %. A shadow-labelled "pothole" just before a speed breaker is
  "confirmed" almost always. Observed: 17 of 21 runs on the sample stretches were "confirmed" — about what the
  base rate plus slow riding predicts, so the confirmation carries little evidence.
- AUC 0.69 is agreement between the IMU and the model's own labels (precision 0.43) on the same 43 drives that set
  the threshold — not a validation against truth. The "36–58 % of labelled runs are real" requires an assumed
  hit rate of 0.6–0.9 for real potholes, never measured.
- The code used confirmed runs as a hard LOWER bound (`lo90 = confirmed`): a section with 2 confirmed runs could
  never be rated with 0 or 1 pothole although each confirmation is false ≥ 24–27 % of the time. **FIXED:**
  P(real | confirmed) = 0.43 / (0.43 + 0.57 × 0.27) = 0.74 (the most favourable case, hit rate 1); lower bound = 5 %
  quantile of Binomial(confirmed, 0.74) → 0, 0, 0, 1, 1, 2 for 0–5 confirmed runs; point = 0.74 × confirmed + 0.43 ×
  unconfirmed. Suggest (chainage, report): speed-normalised jolt (`rms_av_norm` / per-speed-bin percentile) and a
  "near a speed breaker" exclusion before any confirmation is used.

**E5. Sections without an estimate are silently dropped from km rows and the stretch (report).**
4 of 19 and 5 of 19 sections on two sample stretches had no estimate: one for insufficient coverage, the other
eight because cracking / ravelling / patching were missing — fewer than 50 % of the section's length has probe scores (gate-skipped frames have no stored
image, so no probe score). `estimate_mean` averages only the rest; `sections_used` is recorded, but the km/stretch
class is printed without saying that the skipped (often blurred / rough / fast) sections are not in it. That is a
selection bias toward well-imaged sections. Suggest: give such sections the strict-range treatment for the three
extent parameters (bounds 0 and the Attain upper curve end) instead of dropping them, or print "x of y sections"
next to every km/stretch estimate.

### Medium

**E6. The IRI interval is wider than "90 %" by construction, and its shape matters (report).**
`iri_interval`: lo90 = noise-floor-corrected / R95 / k, hi90 = uncorrected / R05 × k — two independent 5 % ends
multiplied (plus the noise-floor subtraction on one side only). Under its own assumptions (ln-spread of R and
the pass-to-pass σ 0.28 combined in quadrature) the 90 % ratio hi/lo would be ≈ ×5.3 instead of ×10.6 for a 100 m
section. It is conservative, not misleading — but it is not a 90 % interval; call it "≥ 90 % under the stated
ranges" or combine in quadrature. The R ranges themselves are engineering judgement (uniform over 10 assumed
ranges), so "90 %" carries no frequency meaning anyway. Shape: drawing IRI split-log-normal with the same
quantiles moves the median section +1.2 / +0.7 / +1.7 (median / lo / hi).
IRI is 39.6 of ~44 width points: as ESTIMATE.md says, roughness is the gap. The **median** depends on R50 = 1.59,
a synthetic value; the published scooter-vs-car factor (1.9, one study) would move PCI_IRI only a few points
because the curve is already low at IRI ≈ 5 (PCI_IRI ≈ 10–15).

**E7. A survey-calibrated probe value replaced the interval with zero width — FIXED.** `param_calibration.json`
stores no residual spread, yet `pci_estimate_unit` set lo90 = hi90 = point for a calibrated parameter, making the
interval narrower than its evidence (the fit's own in-sample MAE is nonzero). Now the uncalibrated interval is kept
around the calibrated value (widened to include it); with no prior interval the basis says "no interval".
Better (survey script, coordinator): store residual 5/95 % quantiles from the calibration half and use them.
Also: a `labels`-source calibration (scale b for cracking / pothole runs) is ignored by the estimate (only
`probe` is applied), while `pci_unit` uses it — inconsistent (report).

**E8. The rut calibration file was never read by the estimate — FIXED.** `pci_rut.rut_prior(stats)` was called
without `calibration`, so `rut_calibration.json` would have been ignored after the survey. Now
`rut_prior(stats, calibration=pci_rut.load_calibration())` (None while absent; an invalid file raises).

**E9. IRI interval from a few metres of IMU data was reported as a full-section interval — FIXED.** The estimate
used any IMU rows; the pass-to-pass term uses max(L, 100 m), so 10 m of data got the 100 m scatter. Now the
estimate requires IMU data on ≥ 50 % of the imaged length (the same rule as `roughness_value`), else no IRI and the
basis says why. (No sample section changed: all had enough IMU data.)

**E10. Ravelling: ×2 widening after the pre-registered coverage test failed (report).** PREREG says a failed C4
means "the interval must not be used without widening; say so" — it does not say by how much, and TEST is spent,
so ×2 is unvalidated (it is chosen after seeing 22/28). Impact is small (ravelling width 0.6–0.9 points). Wording:
say "widened ×2 by judgement, coverage after widening unknown".

**E11. Cracking evidence does not transfer as worded (report).** ESTIMATE.md quotes "ranks stretches (ρ 0.59),
interval held 27/28" — Attain TEST numbers. On our video the cracking scores look like Attain frames WITHOUT cracks
(KS D 0.61 vs all Attain; transfer.json); the mean prediction is 2.9 % vs 8.7 % on Attain. Say "on Attain" every
time and that ranking on Bengaluru is unknown.

**E12. Stretch aggregate (estimate_mean) — defensible, with two caveats (report).** Averaging the sections'
5 % and 95 % ends = treating section errors as fully correlated; that is right for the dominant shared errors
(scooter ratio R, Attain calibration, rut prior) and conservative for the rest; mean of medians = median of the
mean under that assumption. Caveats: E5 (dropped sections), and the label "90 %" for a km/stretch is then also
"≥ 90 %". No change needed.

### Low
- **L1.** Inputs were not checked: a negative value or lo90 > point flipped the split-normal side. FIXED
  (validated; lo/hi widened to include the point).
- **L2.** `width_by_param` for a range/uniform parameter uses its full range while others use 5–95 % ends; rut's
  share is overstated by ~10 % relative. Acceptable once "range" semantics are explicit.
- **L3.** All sections use the same MC seed, so MC noise is common across sections (consistent with E12).
  n = 1500 gives ~0.3-point MC error on the ends.
- **L4.** Double use of data: the extent study's TEST run is clean (PREREG hash verified, sha256 677ebc7a…); the
  pothole P/R 0.43/0.25 come from the §6.8 Attain test (used as constants, not tuned — acceptable, but they are
  frame-level, see E1); the IMU AUC and the jolt threshold share the same 43 drives (E4).
- **L5.** Wording in ESTIMATE.md: "each with an honest 90 % interval" — not true for rut (judgement range), IRI
  (≥ 90 % under assumed ranges), potholes (E1), ravelling (failed). "Km rows ... conservative" is fine.

## Fixes applied (only `app/pci_irc2023.py` and the estimate block of `app/pci_core.py`)
1. `pci_estimate`: input validation + ordering (L1); new `dist: "range"` = envelope at both ends with
   `median_assumes` (E2); new output fields `range_params`, `median_assumes`; all existing field names kept
   (point, median, lo90, hi90, class, class_lo, class_hi, recommendation, width_by_param, draws); docstring on
   independence (E3). Every run uses the same random stream (common random numbers) so envelopes are comparable.
2. `pci_estimate_unit`: rut prior passed as `"range"` (E2) and the rut calibration loaded (E8); IMU coverage
   ≥ 50 % for IRI (E9); calibrated probe values keep an interval (E7); pothole lower bound = binomial 5 % quantile
   of confirmed runs with P(real | confirmed) 0.74, point 0.74 × confirmed + 0.43 × unconfirmed, basis text names
   the run-vs-IRC-unit caveat (E4/E1). New constants `JOLT_FALSE_CONFIRM_MIN`, `P_REAL_IF_CONFIRMED`,
   helper `_binom_q05`.

## For the coordinator
E1 (pothole units / recall / n = 0), E3 (comonotone group for the extents; print the correlated worst case),
E4 (speed-normalised jolt, speed-breaker exclusion — pci_chainage), E5 (dropped sections — estimate_mean),
E6 (IRI interval in quadrature or relabel — pci_iri), E7 (store calibration residuals — survey script), E10–E11
and L5 wording in ESTIMATE.md; re-generate the sample reports after these fixes (intervals widen: ~42–95 on the
three sample stretches).
