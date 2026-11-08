# Pre-registration — Attain extent calibration (written 2026-11-09, before any fit and before any TEST number)

Hash of this file: `PREREG.sha256` (the `test` stage refuses to run if this file changed). Looked at so far:
DEV frames only (score-vs-area bins, per frame size) and ~21 Attain frames + 6 video frames by eye for the ROI.

## Data and split
- Attain SMP WS_V2.0, 847 frames; probe P(yes) from `eval_results/stage2_probe_raw_attain.json`, variant
  `min:name` (production), Stage 2 list `v0.distress_types`.
- Split = `scripts/stage2_probe_report.split`: blocks of 20 consecutive frames, seed 20260923, 35 % of blocks DEV
  (15 blocks, 300 frames), the rest TEST (28 blocks, 547 frames). Nothing from TEST is used for any choice.
- Target per frame = 100 × (union box area of the parameter ∩ ROI) / ROI area, ROI = `trapezoid` (per frame size,
  `ROI_TRAPEZOID` in the script). Parameters fitted and tested: cracking, ravelling, patching, potholes
  (area %); `ravelling_incl_weathering` also fitted/tested as a sensitivity, never used for a PCI.
- Unit of evaluation = a TEST block (20 frames ≈ a stretch); truth = mean of its frames' targets.

## Model (app/pci_extent.py) — fixed now
- Feature s = max P(yes) over the parameter's keys (MAPPING.md §3), x = logit(clip(s, 1e-4, 1 − 1e-4)).
- Candidates: `constant` (dev mean); `bins3`, `bins5`, `bins8` = quantile bins of x on the fitting frames, bin
  means of the target made non-decreasing by pool-adjacent-violators, piecewise-linear between bin centres
  (median x of each bin), flat beyond the end centres; `bins3_label` = a separate `bins3` curve per Stage 2
  label flag (MAPPING §3), a candidate only if both strata hold ≥ 30 fitting frames.
- Selection (DEV only): grouped 5-fold CV over DEV blocks (fold = shuffled block position mod 5, seed 20260924),
  criterion = mean absolute error of block means (blocks of 20) on held-out folds; lowest wins, ties → the
  simpler model (order: constant, bins3, bins5, bins8, bins3_label). The winner is refitted on all DEV frames.
- Section estimate = length-weighted mean of the per-frame predictions.
- Calibration uncertainty: 400 block-bootstrap refits on DEV (bin edges fixed from the full DEV fit).
- Residual dispersion: from the CV out-of-fold DEV residuals. On the square-root scale, block residual
  u = √T − √Ŷ (T, Ŷ = block means); τ20 = RMS of u over DEV blocks with ≥ 10 frames. Other section sizes n
  (frames / 5 m segments): τ(n) = τ20 · √(g(n)), g(n) = [(1+(n−1)ρ)/n] / [(1+19ρ)/20] for n ≤ 20 and 20/n
  beyond (20-frame blocks independent); ρ = intra-block correlation of raw OOF frame residuals, from
  var(block mean) = σ²(1+(n−1)ρ)/n, clipped to [0, 1].
- 90 % interval = 5th/95th percentile of the mixture over the bootstrap curves × 41 normal quantiles of u
  (T = max(0, √Ŷ_b + u)²). Point estimate = the full-DEV curve.
- Bengaluru application only (not part of the Attain test): an extra multiplicative factor for the viewpoint /
  domain change, 90 % range ×/÷ 1.5, raised to ×/÷ 2 for a parameter whose Attain-vs-video KS D ≥ 0.20. This is a
  declared assumption, not an estimate.

## Baselines (on the same TEST blocks)
(a) constant = DEV mean frame target; (b) current "% of frames flagged" rule: cracking / potholes = % of frames
whose Stage 2 list names the type (what the A→B report does today, `pci_core` labels); ravelling = % of frames
with P(yes) ≥ 0.08878 (production Attain threshold); patching = % with P(yes) ≥ 0.5; (c) the a + b·f form of
`pci_core.probe_value` with a, b least-squares on DEV blocks (f = fraction flagged by (b)).

## Pass criteria (per parameter, on TEST blocks, block bootstrap B = 2000, seed 20260923)
- C1 beats constant: block MAE(model) < MAE(constant) AND the 95 % CI of MAE(model) − MAE(constant) is < 0.
- C2 beats the current rule: MAE(model) < MAE(% flagged) (CI reported).
- C3 ranks stretches: Spearman ρ(model, truth) over TEST blocks with 95 % CI lower bound > 0.
- C4 honest interval: coverage of the 90 % interval ≥ 0.80 AND its Wilson 95 % CI contains 0.90.
- Reported, not a criterion: bias, MAE vs baseline (c), mean interval width, ROI sensitivity, grouped 5-fold CV
  over all 847 frames (secondary).

Verdict per parameter: **calibrated** if C1, C3 and C4 pass; **interval only** if C4 passes but C1 or C3 fails
(the estimate is then no better than a constant and is reported as such); **fails** if C4 fails (the interval
must not be used without widening; say so). No re-fit or re-tuning after the TEST run; any later change is an
addendum with a new hash and TEST then counts as spent.
