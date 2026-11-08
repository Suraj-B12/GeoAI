# Critic review — IRC:82-2023 PCI (equations, pci_core 2023 section, chainage/routes wiring, survey script)

Reviewer: tests + critic workstream, 2026-11-08. Method: every Appendix-2 equation re-read from the page images
(`C:\GeoAI\pci\irc82_2023\appendix2_pages\p076-p088.png`, crops upscaled) and re-typed independently in
`scripts/tests/test_pci_irc2023.py`; every equation evaluated on fine grids (extent 0–100 step 0.01, pothole number
0–50 step 0.001, IRI 2–20 step 0.001); denominators' real roots computed; worked examples recomputed; each IRI
equation evaluated at the Table 5.1–5.3 class limits. Line numbers refer to the files as of this review.

**Transcription verdict:** every printed coefficient in `app/pci_irc2023.py` matches the page images digit by digit
(highway, MDR, rural, urban; all 9 roughness equations). Tables 5.1–5.5 ranges, weights and class limits match the
text. The problems are in the *printed equations themselves* and in how values enter them.

## A key design fact found (used as a check throughout)

Every usable Appendix-2 curve gives PCI ≈ 80 at the Table 5.1–5.3 "Excellent to Good" limit and ≈ 40 at the
"Satisfactory to Fair" limit: MDR SD 77.8/36.7, OGPC 78.9/40.3, MSS 79.6/39.4, SDBC 79.4/40.7, BC 80.1/39.7; rural
OGPC 81.5/40.8, SDBC 74.7/44.5; highway/urban IRI 79.4/41.5; cracking/rut/ravelling (highway, MDR, urban) 76–79 /
41–45. Exceptions: MDR patching (56.5/33.0 — but it follows its own printed curve) and the two rural roughness
equations below.

## Findings, by severity

### Critical

**C1. `category="mdr"` crashed every report (HTTP 500) — FIXED by the coordinator in 5b27b0a/working tree (`_RENEWAL["mdr"]`); `test_report_mdr_category` now passes.** `pci_core._RENEWAL` (2015 section, ~line 740) has no
`"mdr"` key; `class_treatment()` raises `KeyError: 'mdr'` in `build_report` → `unit()`. `CATEGORIES` and
`validate_request` accept "mdr" since the 2023 change. Scenario: the WebGIS offers "Major District Road" → the
request fails. Reproduced by the new `test_report_mdr_category` (FAILS). Fix (coordinator; 2015 section, not
mine): `_RENEWAL["mdr"] = _RENEWAL["rural"]` (Table 9.1 groups MDR/RR) and check every other 2015 dict keyed by
category (`TABLES` already copies rural).

### High

**H1. Rural SD and MSS roughness equations as printed are not usable — FIXED (now `Unavailable`).**
- MSS printed `IRI³ + 7.173 IRI² − 9.017 IRI + 15.43`: PCI 48 at IRI 2, 5.9 at 3, 0.6 at the Excellent-to-Good
  limit 3.79, and it rises again 0.2 → 3.1 between IRI 4.2 and 8.6. An Excellent rural MSS road scored Fail.
  With both middle signs flipped (`− 7.173 IRI² + 9.017 IRI`, the alternating sign pattern of every other
  denominator) it gives 79.0 / 38.4 at the class limits, like every other curve — two sign errors, unconfirmed.
- SD printed denominator `IRI² − 9.272 IRI² + 21.13 IRI + 3.037`: read as IRI³ (as the code did) it gives 60.8 /
  14.7 at the class limits and dips to 14.5 at IRI 4.9 then rises to 27.8 at 6 (non-monotone by 13 points); the
  literal reading has a pole at IRI 2.69.
- **IRC82_2023.md says "the cubic reading is the only one that follows the printed curve" — not supported:** the
  rural roughness curve on p. 85 of our copy does not render (garbled glyphs), so no rural roughness reading can be
  checked against a curve. The record should be corrected.
- Both candidates are kept in `IRI_RURAL_CANDIDATES` for the record; owner should confirm against an official copy.

**H2. Uncalibrated label quantities produced a "single PCI" and a bracket that is not a bracket — FIXED.**
Cracking from labels is "% of surveyed LENGTH (upper bound of % area)" (`pci_core` pci_unit), yet it was used as
`value_lo` (best case) and in `single`. Scenario: a requester enters ravelling, patching, rut depth and IRI; every
unit then printed a single PCI with cracking extent = % of 5 m segments with any crack (one hairline crack per
segment → 100 % → PCI_crack ≈ 0.7), i.e. systematically too low, and `lo`/`hi` claimed to bound it. Pothole runs
are not the IRC number (area / 0.1 m²) either (one small pothole = 0.3; a 0.5 m² one = 5). Fix: uncalibrated label
cracking now brackets from 0 % area (best) to the length % (worst); `single` is None whenever any parameter comes
from uncalibrated labels, with `pci.single_note` saying why. Calibrated labels (`labels_calibrated`) are unchanged.

**H3. `entered` values were applied unchanged to every unit — wrong for the pothole COUNT — FIXED by the coordinator (entered potholes = per km, scaled to unit length); tests adapted.**
`pci_unit` uses `entered[p]` for each 100 m section, km row and the stretch (evidence string: "for the whole
stretch"). Extents (%), rut depth and IRI are intensive and survive this; the pothole number is per section.
Scenario: requester enters "potholes: 12" for a 3 km stretch → every 100 m section gets 12 potholes → PCI_pothole 0
on every section and km row. Fix (coordinator/UI + pci_chainage): define the entered pothole value as *per km*
and scale by unit length (as the probe path does: `b × f × length_km`), and say so on the form.

**H4. The pre-registered headline metric is unreachable with our data (protocol; report).** The rule's headline
is "MAE between our single PCI and the survey PCI". After H2, and even before it, a single PCI needs all six
parameters; rut depth has no source except `entered`, IRI only after the IMU calibration, ravelling/patching only
after a probe calibration that beats a constant. In practice every test section will fall back to the "range
midpoint", whose value is `known part + 50 × unknown weight` — not a prediction. Suggest a **pre-data addendum**
to `survey_split_rule.json` (still before any survey data): headline = agreement of the partial PCI over the
parameters we assess (`assessed_part`, renormalised) with the survey's PCI recomputed on the same parameters from
the survey's own per-parameter values (the survey CSV carries them), plus per-parameter MAE; the full-PCI MAE only
when a single PCI exists.

**H5. `fit` wrote straight into the file production reads live — FIXED.** `current_calibration()` re-reads
`eval_results/pci/param_calibration.json` on mtime change, so a `fit` changed live reports before any blind
evaluation. Now `fit` writes `param_calibration_candidate.json`; `evaluate` measures our reports WITH that
candidate (env `PCI_PARAM_CALIBRATION`), and only when it was fitted on the same survey file (previously it used
whatever production had, or another file's fit, while printing "evaluating uncalibrated"); new `promote` copies
the candidate to the live path only if the ledger holds an evaluate of exactly that candidate on that file.

**H6. Rising left ends of roughness curves — FIXED.** Rural OGPC rises 86.6 → 100.3 between IRI 2 and 3.4, rural
SDBC 68.7 → 98.0 between 2 and 2.8 (MDR OGPC flat 2.0–2.08): a smoother road scored lower than a rougher one.
Now IRI below the curve's peak on [2, 20] is scored as the peak (`IRI_PEAK`, generalising the existing "IRI < 2 →
2" rule). Interior non-monotone stretches are kept as printed (MDR SD +1.4 between 4.86 and 5.31, MDR OGPC +0.6 at 6–8;
the printed curves show them).

### Medium

**M1. `pci_range` took best/worst from the interval ends only — FIXED.** With non-monotone curves the interval
minimum can be inside (MDR SD roughness on [4.7, 5.6]). New `pci_param_bounds()` samples 400 points;
`pci_range` and `pci_unit` (`pci_best`/`pci_worst`) use it, so `single` is always inside `[lo, hi]`.

**M2. `stretch.pci` is off the IRC scale for stretches > 1 km (report).** `build_report` calls `pci_unit` on the
whole stretch (`stretch = unit(0.0, L, ...)`), so its pothole number is the count over the whole stretch (e.g. 30 runs
over 5 km → PCI_pothole 0). `summary.pci` correctly uses the km-row mean. Fix: drop `pci` from the stretch unit
or set it to None when L > 1000 m, so neither the PDF nor the UI can show it.

**M3. Two stretch PCIs that differ by construction (report).** `summary.pci` (km rows) and `summary.pci_sections`
(100 m) differ systematically because pothole numbers per section scale with section length (IRC allows 100–1000
m). Label both with their basis wherever shown; the survey comparison is consistent (it builds A→B on each survey
section, one km row). `pci_mean` should also carry a `single_note` and say "of the RATED km rows" (unrated rows
are dropped from the mean; `units_rated/units` is there but not in the `basis` text).

**M4. Surface not validated per road class (report).** `validate_request` accepts any of SD/OGPC/MSS/SDBC/BC for
any class. Rural + BC (no equation) used to raise `ValueError` inside `pci_param` → 500 on any IRI value; now
`Unavailable` → bracketed (FIXED in pci_irc2023). Still: return 422 for rural + BC, and `_meta_2023` should not
offer rural SD/MSS as usable (their roughness PCI is now not computable) — owner of the meta fn.

**M5. Survey `fit` adopted parameters that were not better than a constant — FIXED.** It wrote the best
candidate whatever its MAE, with "(NOT better than a constant)" in the evidence, and production then used it.
Now such parameters are not written (report shows `best_candidate` and the reason). Evidence text now says
"in-sample, best of N candidates": with ~11,000 (t, b) candidates per parameter on the calibration half the
in-sample MAE is optimistic; only the test half is evidence.

**M6. Survey `fit` b grid could not express realistic scales — FIXED.** `B_GRID` stopped at 20, but `b = 100 ×
area factor` for % parameters (needs up to 100) and probe potholes are per km at f = 1 (hundreds). Now a log grid
0.01–1000 (299 values) plus the old fine 0.05 steps. `test_fit_recovers_known_calibration` recovers b = 40/km.

**M7. Mixed units in the fit error — FIXED.** `_pci_err` returned |count difference| for MDR/rural potholes
(Unavailable) and PCI points otherwise, averaged together and labelled "mae_pci". Now `_errs()` uses PCI points
only if every section has a usable equation, else native units for all; the unit is reported (`error_unit`).

**M8. Constant baseline deviated from the pre-registration — FIXED.** The rule says "calibration-half mean survey
PCI"; `evaluate` used the median. Now the mean.

**M9. Post-hoc detection keyed on file bytes only — FIXED.** Any edit of the CSV (extra column, corrected value)
made a fit after an evaluate look pre-registered. The ledger now also stores a hash of the test half's section
ids; `is_post_hoc()` matches either. Remaining gap (report): a run with `--out-dir` keeps its own ledger, so an
evaluate in a scratch dir on the REAL survey file is invisible to a later production fit — never use `--out-dir`
with the real survey, or always append to `eval_results/pci/survey_ledger.jsonl`.

**M10. Split leakage/scope (protocol statement, report).** Alternating 3-section blocks put calibration and test
sections side by side on the same roads (a pothole can straddle a block edge). The test half therefore measures
transfer within the surveyed roads, not to new roads, and block-bootstrap CIs (blocks of 3) understate
uncertainty if condition is correlated over > 300 m. State this in the paper; it is not a code bug.

**M11. Worked-example wording (IRC82_2023.md, report).** (a) the urban gap (−1.6/−10.0/−7.1) is attributed to "a
different roughness curve"; the IRI term alone would need PCI_IRI ≈ 92 / 32 / 22 at IRI 2.4 / 6 / 7 against 88 /
7 / 5 from the equation, and no other printed roughness curve gives that set — say "unexplained". (b) Evidence for
the MDR pothole reading: with a linear term −19.25 PN (value 0 at PN = 10, its curve's end) and surface BC, the
MDR examples reproduce: B.1 88.06 vs 89.45, B.2 47.27 vs 47.17, B.3 21.77 vs 22.44 (with −18.9: 88.12/47.60/22.32).
This supports "a linear term is missing" but is a reconstruction; keep it Unavailable unless the owner confirms.

### Low

- **L1.** Overflow: `pci_param(..., 1e100)` raised `OverflowError` (IRI⁵, PN³). FIXED: values are capped (`X_MAX`;
  extents at 100 %, where every curve is already ~0); extents > 100 % are scored as 100 %.
- **L2.** `ZeroDivisionError → 0.0` silently. No pole lies in the evaluated domain (MDR MSS has a real pole at IRI
  1.958, MDR BC at 1.149 — both below the IRI-2 floor; no positive root for any pothole/extent denominator), so it
  now raises `ValueError` instead of inventing a value.
- **L3.** Unknown parameters are bracketed at PCI_i = 100, but several curves never reach 100 (highway ravelling/
  patching max 97.0, cracking 99.1, MDR cracking 99.0): `hi` is up to ~0.3 points loose. Harmless; could use the
  curve maximum.
- **L4.** `validate_calibration` raised AttributeError for a non-object parameter entry — FIXED (ValueError).
- **L5.** A probe calibration with no scores for a unit fell to "not assessed" with a generic text — FIXED: the
  evidence now says the probe is calibrated but scores cover < 50 % (and gives the scored metres).
- **L6.** Survey `read_survey` accepted negative values, % > 100, non-lat/lon positions and rural/MDR + a surface
  without a roughness equation (later crashing `_pci_err`) — FIXED with line-numbered errors.
- **L7.** Survey `fit`'s `probe_fraction` has no ≥ 50 % scored-length rule while production's `probe_value` has one;
  a section with one scored segment counts in the fit. Align when the IRI fit is added.
- **L8.** `video_probe_scores.py`: `probe_fp` hashes model/variant/keys/max_pixels but not the question texts or
  `stage2_probe.py`; a wording change keeps the same fp (stale scores). Rows that errored are never retried (the
  NOT EXISTS skips them) — delete error rows to retry. `load_graph`'s cache key does not include the probe table,
  so scores written after the graph is cached appear only after another data change (scoring is finished, so
  latent).
- **L9.** `P23.NAME` says "…and phone sensors" even when IRI is not assessed; wording only.
- **L10.** `pci.assessed_part` is a renormalised mean over known parameters; if shown, it must not be called PCI.

## roughness_value() (pci_core, phone-IMU IRI hook) — reviewed at the coordinator's request

Sound overall: no calibration → None; IMU rows are independent of the image gate, so gate-skipped segments count
(right); bins are length-weighted (IRI is a length average); ≥ 50 % calibrated length required. Issues:
- **R-a (Medium).** Bins are `floor(measure / unit_m)` and `calibrated_iri` rejects a bin whose used length differs
  from `unit_m` by > 50 % (`UNIT_TOLERANCE`). A bin loses length when segments are `low_speed`, errored or
  not covered, so bins with stops (junctions, signals, traffic — often the roughest places) drop out as
  `unit_mismatch`, biasing the unit IRI toward smoothly driven stretches; and a report's last partial section
  (< 50 m) never gets an IRI. Report `bins` counts (already returned) in the evidence text so this is visible.
- **R-b (Low).** Segments from different passes (attached same-direction passes filling holes) land in one bin and
  are aggregated together (rms, speed across two drives/cars). Acceptable if the calibration is per method, but
  say so; better: aggregate per (pass, bin) then length-weight.
- **R-c (Low).** Bin origin is the report's relative measure from A, so bins match report sections only when
  `unit_m` = 100; the survey fit aggregates over survey sections of 100–1000 m — fine within the 50 % tolerance only
  if the IRI fit uses ~100 m units too.
- **R-d (Low).** Docstring has a garbled duplicated clause ("(bins of the calibration's unit_m) each bin ->").
- `pci_unit` uses it only when `q["roughness"]` is truthy; the evidence string "calibrated on survey sections
  (iri_calibration.json)" should carry the calibration's fitted_on/post-hoc flag like the probe evidence does.

## Fixes applied (only in the three allowed files)

`app/pci_irc2023.py`
1. Rural SD and MSS roughness → `Unavailable`; printed/candidate readings kept in `IRI_RURAL_CANDIDATES` (H1).
2. A surface with no equation for the class (rural BC) → `Unavailable`, unknown surface name → ValueError (M4).
3. IRI below a curve's peak scored as the peak (`IRI_PEAK`, `_equation`) (H6).
4. `X_MAX` caps (no overflow; extents ≤ 100 %) (L1); pole → ValueError instead of 0 (L2).
5. New `pci_param_bounds()`; `pci_range` uses it (M1).

`app/pci_core.py` (2023 section only)
6. Uncalibrated label cracking brackets from 0 % area; `single` None for uncalibrated labels; `pci.single_note` (H2).
7. `pci_unit` best/worst from `pci_param_bounds` (M1).
8. Probe-calibrated-but-unscored evidence text (L5); `validate_calibration` type check (L4).

`scripts/pci_survey_compare.py`
9. Candidate file + `promote`; evaluate uses only the candidate fitted on the same file (H5).
10. Not-better-than-constant parameters not written; in-sample wording (M5); log b grid (M6); unit-consistent
    errors (M7); constant = calibration-half mean (M8); test-half hash in ledger + `is_post_hoc` (M9);
    `read_survey` value/position/surface checks (L6); docstring no longer claims an IRI fit.

## Items for the coordinator

H4 (pre-data addendum to the split rule), M2 (stretch.pci),
M3 (`pci_mean` note/basis), M4 (422 for rural+BC; meta surfaces), M9 (ledger + out-dir), M11 + H1 wording in
`IRC82_2023.md`, R-a…R-d (roughness_value), L7 when adding the IRI fit, and the owner's confirmation of the
rural SD/MSS and MDR pothole equations against an official IRC copy.
