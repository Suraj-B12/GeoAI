# Gate validation: pre-registered protocol and choice rule

Written 2026-10-28, before any held-out number of this study was computed. Not edited afterwards
(the file's SHA-256 is recorded in results.json).

## Disclosure

Before writing this, I had read the discriminator's single-split leave-one-session-out point
estimates (C:\GeoAI\analysis\discriminator\FINDINGS.json item 2, out\full_batch_extra.json):
probe_frac alone held-out recall 0.85 / 0.88 / 1.00 and pass 0.63 / 0.50 / 0.67 (d47c / f428 / 293d);
probe_max_logit alone recall 0.90 / 0.93 / 0.86, pass 0.65 / 0.60 / 0.62. No CIs, no combinations.
In-sample (all 813): probe_frac at 90 % recall passes 61.4 %, probe_max_logit 63.1 %.

## Data and reference

batch_001: 813 send-all segments of three sessions (f428 230, 293d 55, d47c 528), all answered.
Reference = the VLM (distress = any valid non-Normal label, `analyze.answer_class`), exactly as
`geoai gate-calibrate`. Scores = the records' `gate.scores` (full precision, from records/*.json).

## Folds and calibration

Leave-one-session-out: calibrate on two sessions with the shipping exact solver
(`gate._solve_any_of`, the one `gate.calibrate` uses) at target recall 0.9, apply the thresholds
(`>=`) to the held-out session.

Candidate rules:
- V0 default thresholds (probe_frac 0.05, evidence 30, crack_mm 300), not calibrated (reference only)
- V1 probe_frac alone (calibrated)
- V2 probe_max_logit alone (calibrated)
- V3 probe_frac OR probe_max_logit (calibrated jointly, exact)
- V4 the shipping any-of rule over probe_frac, evidence, crack_mm (what `gate-calibrate` does)

Metrics per held-out session and pooled over the three held-out sessions (813 segments): recall
(share of VLM-positive segments the rule passes) and pass rate (share of all segments it passes;
audits not counted). Sent rate with audit_frac 0.10 = pass + 0.10 x (1 - pass), for information.

## Uncertainty

Block bootstrap, blocks of 20 consecutive segments (k order) within each session, B = 2000,
seed 0. Each replicate resamples the blocks of the two training sessions (each session keeps its
number of blocks), re-calibrates, and resamples the blocks of the held-out session to evaluate.
95 % percentile intervals. Differences between rules are paired (same replicate indices).

## Choice rule (decides what ships)

V1 (probe_frac alone) is the default: it is a gate signal of the current calibration schema
(`geoai.gate_calibration.v1`, GATE_SIGNALS).

An alternative Vx (V2, V3, V4) replaces V1 only if ALL hold on the pooled held-out data:
1. pass rate of Vx <= pass rate of V1 - 0.03, and the paired 95 % CI of (pass_Vx - pass_V1) lies wholly below 0;
2. recall of Vx >= recall of V1 - 0.02;
3. the held-out recall of Vx is >= 0.80 in every session.

Consequences fixed in advance:
- evidence and crack_mm stay OFF unless V4 wins under the rule.
- If V2 or V3 wins, it needs probe_max_logit as a gate signal, i.e. a schema/contract change. Then
  V1 still ships (the task keeps the schema unchanged) and the win is reported as a decision for the owner.
- The shipped thresholds are calibrated on all three sessions (gate-calibrate on batch_001), not
  the fold thresholds.
