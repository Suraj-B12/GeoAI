# Critic review round 2 — backend code and sample reports (2026-11-07)

**STATUS: DONE** (verdict in §D)

Reviewer: critic agent, read-only on project files. Scratch scripts and page images are in the session
scratchpad `critic2\` (`indep.py`, `rate.py`, `drop.py`, `drop2.py`, `edge2.py`, `annex.py`, `r12_p*.png`,
`r5_p*.png`). Data: `geoai_test` (SELECT only, `default_transaction_read_only = on`). Severity: BLOCKER
(output wrong or misleading as built), MAJOR (fix before the paper / production), MINOR.

## A. Checked and correct

- **Tests:** the 4 PCI files pass for me too (CPU, `CUDA_VISIBLE_DEVICES=-1`): **27 passed** in 31 s.
- **IRC arithmetic, recomputed independently.** I wrote my own code with exact fractions: my own bands,
  truncation, half-up rounding and bracket, not the project's functions.
  - **182 rated units** of the 9 stored reports (sections, km rows, stretches): **0 mismatches** in final
    lo/hi or class.
  - **App-4 worked example:** 1.2 / 1.2 / 3.0 / 2.1 / 1.5 / 1.2 / 1.2 → **1.3 Fair**, the same as
    `C.app4_final`.
  - `TABLES` (5.1/5.2/5.3) and `WEIGHTS` match RESEARCH §1.2–1.5 cell by cell, including the urban
    pothole row NIL / >0–≤0.5 / >0.5.
  - A boundary value of exactly 5 % falls in Fair, as "5-15" says.
- **Chainage, recomputed from SQL** (haversine between consecutive `centre_lat/lon`, my own section split):
  | Sample | Mine | Stored | Gap (mine / stored) |
  |---|---|---|---|
  | 4, longest drive, 1,657 segments | 8,310.04 m | 8,309.6 m (−0.005 %) | — |
  | 6 | 1,004.1 m | 1,004.3 m | 50.5 / 50.5 m |
  | 12 | 404.36 m | 404.4 m | 11.9 / 11.9 m |

  Segment counts are identical. The per-100 m coverage, pothole E and cracking E all agree, except where
  one boundary segment falls into the neighbouring section (equirectangular vs haversine measure, ≤ 0.4 m).
  That shifts one segment = 5 percentage points in a 100 m section. It is not a bug, but see #11.
- **Round-1 fixes I confirmed in the code:**
  - #1/#2 bracket over all table parameters, "not assessed" never 0;
  - #6 skip-corrected range, 80 % rule plus a 50 % line;
  - #7 no smoothing;
  - #8 same/cross/gap edge rules, Δalt refusal, switch cost;
  - #10 survey-form fields only "as entered";
  - #11 rate key, keyed save, cap, dedupe, CSV guard, text limits, `sample` set server-side;
  - #12 `data_as_of` + stale + frozen PDF + frame SHA;
  - #14 Decimal / `floor(10x+1e-9)`;
  - #15 refusals and tail merge;
  - #16 Proforma columns, including Drainage/Shoulder and Total as "not assessed";
  - #17 treatment heading.
- **Edge cases that answered correctly:**
  - A = B and A ≈ B (3 m) → 422; a 25 m path → 422 "at least 50 m";
  - outside the bbox → 422; NaN → 422; 9 via → 422; 121 chars → 422; emoji → 422;
  - Kannada → 422 (deliberate, STATUS question 1);
  - chainage without a direction → 422; lane 0 / `true` → 422; bad category → 422; list body → 400;
  - save without configured keys → 403, wrong key → 401;
  - rate limit: per `Cf-Connecting-Ip` from loopback (the 11th call → 429, another IP → 200); a
    non-loopback peer cannot dodge it by rotating the header (31st call → 429);
  - reversing A/B gives the same length.

## B. Coordinator item — Poor potholes, overall "Fair–Good" (report 12, 404 m, 24.7 pothole runs/km)

What the PDF shows now (pages `r12_p01`–`p05`):

- The summary table does lead with the bands: Cracking 2.5 % **Good**; Potholes 27.1 % **Poor**;
  10 runs = 24.7/km.
- But the next line prints, in **green bold**: "Overall … 1.1–2.1 → **Fair–Good** (ravelling, settlement
  and depression, rut depth not assessed). Secondary to the per-parameter bands above."
- Nothing on page 1 says why Poor potholes can coexist with Good.
- The route figure (p. 2) and the strip chart's first row colour the track **by the bracketed final
  class**. Two of the four 100 m sections, each with Poor potholes (4.8 % and 23.8 %), are drawn green.
- The only "rectify straightway" text is a grey footnote at the end of §8 (`ANY_CLASS_TEXT`), plus a cell
  in the per-distress table.

### Finding B1 — MAJOR: the overall class reads as the verdict, and the map paints Poor-pothole sections green

App. 4 weights potholes 0.50, and three of the five urban parameters are bracketed, so the arithmetic is
right. Poor potholes cost only 0.5 × (3 − 1) / 5 = 0.2 of the final value. With cracking Good and the
bracket's high end, the final reaches 2.2 (Good).

Validation §6 shows **0 of 408 blocks** with a single class, and a bracket width ≥ 1.0 on every block.
The final class therefore always spans two classes and carries little information. The colour is what a
reader will remember.

**Fix (exact wording and placement):**

1. **Summary (page 1), straight after the headline table and before "Overall".** Add a boxed line in
   `INK` (not a class colour), printed whenever a headline band is Poor (`potholes` or `cracking`):
   > **Potholes: Poor band (27.1 % of surveyed length, 24.7 runs per km, est.).** IRC:82-2015 §5.3: distress
   > that adversely influences riding quality, such as potholes, "should be rectified straightway",
   > whatever the overall rating. Under Appendix 4 the overall value is an average in which potholes weigh
   > 0.50 of 4.00, so a stretch with Poor potholes can still come out Fair or Good overall.
2. **"Overall" line.** Print it in muted text, never in a class colour. Reword it:
   > Overall (Appendix 4, secondary): 1.1–2.1, Fair–Good — a bracket, because ravelling, settlement and
   > depression and rut depth are not assessed (3 of 5 parameters, 2.5 of 4.0 weight). The bracket here is
   > always about one class wide; use the per-parameter bands above.
3. **Route figure and strip chart.** Colour the track by the **pothole band** (the headline). Optionally
   add a second small-multiple map for the cracking band. Move the bracketed final class to the last
   strip row, labelled "Overall (App. 4, bracketed; secondary)", and never use it to colour the map. The
   legend must show the band colours only.
4. **§8 Treatments.** Move `ANY_CLASS_TEXT` from the footnote to the top of the section, in body size,
   and repeat it in the km-row block whenever potholes are present.
5. **JSON.** Add `summary.priority_notes: [..]` holding the same text, so the WebGIS shows it beside the
   class. Add `summary.final.note` carrying the bracket explanation.

### Finding B2 — MAJOR: the report JSON has no per-section segment attribution

Only a stretch-wide `segment_ids` list exists (`pci_chainage.py:844`). The UI must guess which frames
feed which section, and its "nearest section" rule disagrees with the backend in three cases:

- attached other-pass segments, which are projected into a ±30 m window (`_attach_other`);
- path segments the backend drops (#1 below);
- boundary segments.

**Fix:** add one top-level table that both the PDF and the UI use:

```
attribution: [[segment_id, measure_m, section_index, km_index, pass_id, used: bool]]
```

It costs about 60 B per segment (≈ 300 KB for 25 km). Keep it out of the stored `summary`.

## C. Findings

### 1. MAJOR — path segments are discarded when two path passes share a 100 m section

**Cause.**

- `build_report` treats each path pass (P0, P1, …) as a *competing candidate* per section
  (`pci_chainage.py:698-729`) and keeps only the one with the most classified length.
- Path passes never overlap: they are consecutive pieces of the same path, split at a gap or a switch.
- So wherever a gap or switch falls inside a section, the smaller piece is dropped. Coverage drops, its
  distress is ignored, and the dropped piece is then listed as "also surveyed" (it was not a second
  survey).

**Evidence.**

- **Samples:** 2, 1 and 1 path segments dropped (samples 5, 9, 11). In sample 5, section 0 lists P1 as
  "also surveyed".
- **300 random A/B pairs on the real graph** (`drop2.py`): 70 routable. **41 of 70 reports drop path
  segments**: 589 of 36,874 in total, with a maximum of **49 of 689 (7 %)** on one report. Stretches with
  switches and gaps are the ones hit.

**Fix.** Make the union of all path-pass segments in a section one candidate that is always used. Let
other (attached) passes compete only for measure ranges the path does not cover. Alternatively, choose
per pass on sub-ranges of measure. Then:

- list "also surveyed" only for passes that overlap the used ones in measure;
- add a test: a gap at 150 m keeps every path node in `segment_ids`.

### 2. MAJOR — the save endpoint lets anyone guess the operator key without a limit

`pci_save` calls `_check_key` **before** `_limit` (`pci_routes.py:526-527`). In my test, 30 wrong keys in
a row gave 30 × 401 and never 429. The endpoint is public through the tunnel, and the same keys control
the worker in `app.main`.

**Fix:** call `_limit(request, "save")` first. Add a separate bucket of about 5 failed keys per 10 min per
client, and log the failures.

### 3. MAJOR (paper) — pothole labels on Bengaluru video look unreliable on the annex frames; the report presents "Poor potholes" as the headline

**Evidence.** I looked at the 12 annex frames of report 12 (pages 6–7).

- At least 4 of the 12 frames marked "pothole run" show no visible pothole:
  - `seg00209`, which shows a pedestrian and a construction heap;
  - `seg00212`, which shows parked motorbikes;
  - `seg00213`;
  - `seg00228`, which shows a kerb and edge.
- Others do show potholes (`seg00010`, `seg00027`, `seg00033`).
- This was a look by eye, not a labelled measurement.
- In section 2 the model says 60 % of the 100 m contains potholes.

**Fix.**

- The report already says "assessed (weak)" and gives the Attain precision of 0.43. Add to the pothole
  headline row: "label unvalidated on Bengaluru video (0 expert labels)".
- **The paper must not report any pothole band or runs/km as a condition finding** until the expert
  Video tab has labels. A 50-frame blind check of pothole-run frames by an expert would give a first
  precision.
- Apply the same caution to cracking.

### 4. MAJOR (paper) — the validation report leaves out three things

The numbers in `validation.md` match `validation.json` and the scripts (I spot-checked: run lengths 770 /
502, retest n = 67, the sensitivity counts). What is missing:

- **(a) Synthetic AR(1) noise.** It adds **+63 m per km at σ 4 m and +151 m per km at σ 6 m** (correlation
  length 50 m). The report states this without interpretation. A reader will take it as the chainage
  error. The real centres come from the video pipeline's own track (centre-path ÷ s-path 1.0007), so
  the synthetic numbers measure the method on raw noise, not our data.
  - State this.
  - Do not quote any chainage-length accuracy until the OSM part is done (it is "not done" at present).
  - The PDF's "±10 m" is a *position* statement. Add "chainage length accuracy: not yet measured"
    (`ACCURACY`, `pci_chainage.py:67`).
- **(b) Test–retest.**
  - 67 matched pairs over 4 drive pairs; κ has no CI.
  - The band agreement comes from **2 blocks** ("potholes band agreement 0.500"). That is not a rate.
  - Report κ with a block-bootstrap CI.
  - Say how far apart in time the paired drives are. Same day and minutes apart is not test–retest
    under different conditions.
  - The paper may say: "repeatability measured on 67 segment pairs (κ distress 0.76); too few
    overlapping 100 m blocks (2) to measure class repeatability".
- **(c) Bracket width.** "Single class 0 of 408; width ≥ 1.0 on every block" is the most important
  result for the paper. State it as such: the App-4 overall class cannot be determined from this
  pipeline; only per-parameter bands can.

### 5. MAJOR — the skip correction turns any section with a skipped segment and no pothole label into "Poor–Good" potholes

**Cause.** Under the urban NIL rule, E_hi = 13.3 % × skip / D. One skipped 5 m segment in 100 m gives
0.67 % > 0.5, which is Poor. That is why validation §6 shows the pothole band **Poor–Good on 118 of 408
blocks**. This is a strict upper bound, but it uses the all-distress rate as the pothole rate.

**Data.** Of the audited skips, 9 of 161 carried potholes (5.6 %, Wilson upper ≈ 10 %; `geoai_test`
audit rows, n = 13 distressed). That per-type rate *is* roughly estimable for potholes, though n is small.

**Fix.**

- Keep 13.3 % as the bound.
- Print "Poor–Good" pothole bands as "Good (no pothole label); Poor if skipped segments hold a pothole".
- Add the 5.6 % / 10 % pothole-specific line as a sensitivity.
- The strip chart draws a Poor–Good band in the Poor–Fair orange (`pci_report_pdf.py:187`) and its
  legend has no Poor–Good entry. Fix both.

### 6. MINOR — the annex chainage is the run start, not the pictured segment

**Cause.** `_annex` prints `r["measure_m"]` (the run's minimum measure) but pictures `segment_ids[0]`
(the lowest k). On a pass driven towards A, those differ.

**Evidence.** Sample 5 has 6 of 24 frames off by **5–20 m**. The lat/lon in the same caption is the
pictured segment's, so the caption is inconsistent.

**Fix:** print the pictured segment's own measure. Say "road ahead = towards A" when the pass runs
towards A.

### 7. MINOR — the annex "section rated Poor" reason hides the bracket

The reason is "section rated Poor (Km 0+100)". The section is actually Poor–Fair overall, with a Poor
pothole band.

**Fix:** use "pothole band Poor in Km 0+100–0+200" or "cracking band Poor …". Also fix the doubled
parentheses: "Km 0+095 (Km 12+255 (12.255))".

### 8. MINOR — the method text "the class never depends on it [the interpolation]" is wrong for the final class

The per-parameter *band* does not depend on the interpolated decimals, but the App-4 final does. For
example, cracking at 2.5 % gives r = 2.5 and puts the stretch's high end at 2.1, which is Good. A
different interpolation could make it Fair.

**Fix** (`_method_text`, `pci_chainage.py:917`): "the band never depends on it; the final value and,
near 1.0/2.0, the final class do".

### 9. MINOR — a user-built out-and-back is double-counted

**Evidence.** A = B with one via point (`edge2.py`): 910 m, `segments_used` 183 against **92 unique
segments**, and 28 pothole runs against 12 one-way.

**Fix:** refuse a path that revisits a node, or count each segment once, and say so.

### 10. MINOR — text validation

- **Format characters accepted.** U+202E (RTL override) and U+200B (zero-width) pass. They can spoof a
  name in the WebGIS and the CSV. Reject Unicode category `Cf`.
- **Contradictory message.** The emoji refusal says "use English or Kannada script", but Kannada is
  refused.
- **Negative road chainage.** `chainage_at_a_km` 0 with decreasing chainage gives "−0.100 (before
  Km 0)" without a warning. Refuse it when the chainage at B would be < 0.

### 11. MINOR — a single boundary segment moves E by 5 points in a 100 m section

Up to 21 segment centres fall into one 100 m section, and coverage is clamped to 1.0. That is correct
but worth one sentence in the method: "± one segment (5 %) at section boundaries".

Stored report 12 has `data_as_of` truncated to seconds (`…20:29:50.000000Z`). It was saved before the
precision fix, which is harmless here but should not happen again.

### 12. MINOR — the route test mutates the shared scratch database

`test_pci_routes.py:225` runs `UPDATE video_segments SET status = status`, which bumps `updated_at`, and
nothing reverts it. After it runs:

- sample report 4 on `geoai_test` reads **stale = true (1 changed segment)**;
- the graph's `data_as_of` moved from 20:29:50 to 20:44:16, so every later save hashes differently.

(I ran the suite as instructed, so this happened again.)

**Fix:** restore `updated_at` in a `finally`, or touch a segment of a report made only for the test.

### 13. MINOR — presentation details

**Route figure:**

- It has 100 m ticks but **no chainage numbers**, except "Km 1".
- Fair (amber) and Poor–Fair (orange) are hard to tell apart.
- Fair–Good and Good are both green.

**Number marking:**

- The Proforma and 100 m tables print bare numbers. Add "(est.)" to the column heads ("Potholes (%,
  est.)").
- Add "label unvalidated" to the pothole and crack headline rows, as Bleeding already has.

**Wording:**

- "click snapped 0 m" should read "entered point snapped 0 m to the nearest segment".

**Sample selection:**

- The 1 km samples are not 1 km of road: sample 5 is a 1,552 m path with 3 gaps (218 m), which drives
  the earlier drives in reverse. That is fine, but `pci_samples` should record the requested and the
  routed length.

## D. Verdict

**The IRC arithmetic and the chainage arithmetic are correct.** I reproduced them independently: 182
units with 0 mismatches, the App-4 example, and lengths within 0.005 %. Round-1 fixes #1–#17 are
implemented, and the 27 tests pass.

**Not ready for production or the paper as is.** Fix these first:

- **#1:** path segments are silently dropped at gaps and switches in 41 of 70 random reports.
- **B1:** the green overall class and route map on Poor-pothole stretches.
- **#2:** unlimited key guessing on the save endpoint.
- **B2:** no per-section attribution, so the UI and the backend disagree.

For the paper:

- Present only per-parameter bands, as a screening tool, with "labels unvalidated on Bengaluru" (#3).
- Report that the App-4 overall class is always a ≥ 1-class bracket (#4c).
- Do not claim chainage-length accuracy (#4a) or class repeatability (#4b).

After those fixes it is reasonable to deploy as a screening/prioritisation aid.
