# Noise floor re-check (follow-up to VALIDATION.md V2, which FAILED: idle/moving = 0.39 > 0.25)

Vehicle: **scooter (two-wheeler), phone on the handlebar/front** — owner's statement 2026-11-09; the drive
records' `survey.vehicle` is empty. VALIDATION.md and RESEARCH.md v1 wrongly assumed a car with a windshield
mount.

## Part A — criteria (written 2026-11-08T18:40Z, before any corrected statistic was computed)

Known at this time: the V2 numbers (idle median 0.257 m/s², moving median 0.657, 13 drives with idle) and a
diagnostic look at idle vs moving spectra per drive (`$TEMP` script, numbers below in Part B D1): idle
segments show peaks at ~57 Hz / ~26–31 Hz (engine, outside the band) and, in some drives, at 2–2.5 Hz
(rider / scooter motion while stopped). No corrected roughness value, AUC or repeatability was computed.

Variants (rms_av of each segment; floor f in m/s², same 0.5–15 Hz band):
- **C1 per-drive subtraction:** rms_c = sqrt(max(rms² − f_d², (0.1·rms)²)); f_d = the drive's idle rms_av,
  drives without an idle period use the median over drives that have one.
- **C2 global subtraction:** the same with f = median idle rms over drives (one number for the survey).

Adoption as the default point estimate requires ALL of:
- (a) pothole AUC (VALIDATION V5 primary: max rms_av_norm over k…k+6) drops by ≤ 0.02;
- (b) same-direction segment-level repeatability (VALIDATION R3, 6 m / 30°, Spearman) drops by ≤ 0.05;
- (c) **representativeness:** the drive's idle floor f_d ≤ the drive's 5th percentile of moving segment
  rms_av in ≥ 80 % of drives with an idle period. (If idle noise exceeds the smoothest moving segments, the
  stop is not a sample of the moving noise — e.g. feet-down rocking on a scooter — and subtracting it would
  remove road signal.)
If not adopted, the C2 floor is used only for the lower end of the uncertainty interval (the "all of the
floor is noise" case), and the raw value stays the point estimate.
Reported regardless: median 100 m iri_gc_uncal before / after, number of repeated passes, repeatability.

## Part B — results

Run 2026-11-08 (method_fp e9e1bc016ba5fa70), `noise_floor.py` -> `noise_floor.json`.

**D1 diagnosis (spectra, 13 drives with an idle period).** While stopped, the strongest lines are at
~54–64 Hz and ~26–31 Hz (engine at idle; outside the 0.5–15 Hz band, so not the cause), but the in-band
idle energy is broadband, and in 3 drives (d47c, b81d, 9fb5) peaks at 1.2–2.9 Hz — the scooter rocking /
rider's feet down / handling, not engine. While moving, every drive shows a line at **24–25 Hz that does
not move with speed** (a handlebar/holder/frame mode excited by engine and road, consistent with a scooter
handlebar mount) plus ~64–70 Hz (engine); both outside the band. The 15 Hz band edge is therefore still
right for a scooter.

| | raw | C1 per-drive | C2 global (0.257 m/s²) |
|---|---|---|---|
| (a) pothole AUC (CI) | 0.684 (0.650–0.718) | 0.665 (0.622–0.709) — Δ −0.019 pass | 0.682 (0.649–0.715) — Δ −0.002 pass |
| (b) R3 repeatability, Spearman (90 pairs) | 0.459 | **0.182 — FAIL** | 0.450 — pass |
| (c) idle ≤ moving 5th pct, share of drives | **9/13 = 69 % — FAIL (needs 80 %)** | | |
| 100 m iri_gc_uncal, 5th/50th/95th pct (383 blocks) | 5.10 / 7.94 / 13.53 | 3.12 / 6.88 / 12.56 | 4.41 / 7.33 / 13.02 |

**Verdict: neither variant adopted** (criterion (c) fails for both; C1 also fails (b)). In 4 of 13 drives
the "idle" level is above the drive's smoothest 5 % of moving segments (d47c idle 1.50 vs moving p5 0.39) —
stops on a scooter are not samples of the moving noise. Per the pre-registered rule the C2 floor is used
only as the lower end of the uncertainty interval: subtracting it lowers the median 100 m index by 8 %
(7.94 → 7.33), i.e. the noise floor can explain at most a modest part of the high index; the rest is
vehicle response (UNCERTAINTY.md).

**Repeated passes (no new drives since 2026-10-30; 45 sessions, 43 with segments):** 175 same-direction
segment pairs (6 m, 30°) from 4 drive pairs; 3 qualifying 100 m blocks from 2 drive pairs — 100 m
repeatability still not measurable. Segment level Spearman 0.46 (n = 90).
