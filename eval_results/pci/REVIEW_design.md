# Critic review of `DESIGN.md` (v1, 2026-11-07)

Reviewer: critic agent, read-only. Sources:

- `DESIGN.md` and `RESEARCH.md`;
- the IRC:82-2015 PDF: pages 11, 12 and 81 rendered and checked by eye; the text of PDF pages 37–38
  (§7.5.3) and 69–70 (Appendix 1, Proforma 1);
- the production local PostgreSQL, SELECT only (scripts in the session scratchpad `critic/`);
- `app/gis_routes.py`, `app/main.py`, paper §8.

Severity scale: **BLOCKER** (the output would be wrong or misleading as designed), **MAJOR** (must be
fixed before results go in the paper), **MINOR**.

## A. Checked and correct

- **Tables and worked example.** RESEARCH §1.2–1.5 matches the rendered pages: Table 5.1 (p. 11),
  Tables 5.2/5.3 (p. 12) and the App-4 worked example (p. 81). That includes the urban pothole row
  "NIL (0) / >0 and ≤0.5 / >0.5" and the missing patching and shoving rows in Table 5.3.
- **Interpolation.** It reproduces 6 of the 7 App-4 rows:
  - cracking 2−4/5 = 1.2;
  - ravelling 2−7/9 = 1.22 → 1.2;
  - shoving 3−0.9 = 2.1;
  - patching 2−4/9 = 1.55 → 1.5;
  - settlement 2−3/4 = 1.25 → 1.2;
  - rut depth gives 1.2, where the example prints 1.3.

  The final value is (8.9 or 9.0)/7, which gives 1.3 Fair either way. The class never depends on the
  interpolation, so labelling the decimal as "inferred" is enough.
- **Table choice.** Table 5.3 is the right default for Bengaluru city streets. Highway and rural can
  be selected per road, which is right for NH/SH stretches inside the city.
- **Patching.** It is correctly left out of the Table 5.3 rating (that table has no patching row).
- **Edge Breaking.** It is correctly kept out of the rating and reported only (Proforma-1 column;
  IRC §7.5.4 is disintegration, not cracking).
- **Naming.** "IRC:82-2015 distress-based condition rating (estimated…)" with no "PCI" is right. IRC
  uses "pavement condition index" only once (§9.3) and never defines it.
- **Migration number.** `009` is free (migrations 000–008 exist).

## B. Data checks (production `geoai`, SELECT only, 2026-11-07)

| Check | Result |
|---|---|
| Segment spacing inside a drive | 8,402 segments, 43 drives. Consecutive centres: median 5.01 m, p99 5.02 m, **max 5.1 m**, no k gaps. Centre-path length ÷ s-path length = **1.0007** |
| Heading (`record.camera.heading_deg`) | missing on 93 segments (= the not_covered ones); present on all others |
| Expert-reviewed **video** segments | **0** |
| `condition_indicators` on video (Patching) | **never present** (0 of 8,402) |
| Final labels on distressed segments (n = 1,287) | Potholes **770**, Longitudinal 720, Bleeding 355, Transverse 275, Alligator 148, Edge Breaking 81, Streaking 73, Hungry Surface 72, Edge Cracking 27. **Ravelling 0, Settlement 0, Shallow Depression 0, Shoving 0, Hairline 0** |
| Status × Stage 1 | send: Normal 5,615 / Distress 1,274. audit: Normal 148 / Distress 13. skip 1,259. not_covered 93 |
| Design applied to blocks of k along each drive (Table 5.3, 20 and 200 segments) | **100 m blocks:** 421 total, 34 insufficient (< 50 %), 81 at 50–80 % coverage. Final **Good 105 / Fair 282 / Poor 0**. Pothole band Poor 238 / Good 149. Cracking band Poor 92 / Fair 120 / Good 175. **1 km blocks:** 42 total, 1 insufficient. Final **Fair 40 / Good 1 / Poor 0**. Pothole band **Poor on 40 of 41** |
| Other drives within 15 m (every 2nd segment) | same heading 125 (91 at < 6 m, 25 at 6–12 m, 9 at 12–15 m); opposite 12; crossing 31 |
| U-turns / self-revisits | 3 drives turn > 150° within 60 m; 1 drive revisits its own track |
| Survey form fields | `road_name` is "test 1"/"test 2" and `direction` is `increasing_chainage` on **all 43** drives: form defaults |
| Gaps between consecutive drives (time order) | 3–648 m apart, Δt 22 s – 30 min. Within a day the vehicle kept driving unrecorded |

## C. Findings

### 1. BLOCKER: undetectable parameters enter the rating as "Good", so Table 5.3 can never return Poor

**Evidence.**

- Ravelling and Settlement/Shallow Depression appear on **0 of 8,402** segments.
- On Attain the free-form list never named ravelling (CLAUDE.md, paper §6.8). For settlement, the
  design itself says "depth not visible".
- Under the design these parameters therefore always score E = 0 → r = 3.0. The minimum urban final
  value is (1·1 + 0.75·3 + 0.5·1 + 0.75·3)/4 = **1.5 Fair**. Poor is unreachable whatever the road
  looks like.
- The simulation confirms it: 0 Poor among 387 rated 100 m blocks and 41 km blocks.
- Good needs r_crack + 0.5·r_pothole ≥ 3.7, so any pothole label rules out Good. The final class is
  therefore a proxy for "no pothole label and < ~4 % cracking".

**Fix.**

- A parameter counts as *assessed* only if the pipeline has shown recall for that type: a stated
  per-type eligibility, e.g. the probe eligibility in `stage2_probe_config_final.json` or Bengaluru
  expert labels.
- Every other parameter, including rut depth, is "not assessed". It is **bracketed**: compute the
  App-4 final with all M table parameters, once with each unassessed r = 1.0 and once with r = 3.0.
- Print one class only when both ends agree; otherwise print e.g. "Fair–Good (ravelling, settlement,
  rut depth not assessed)".
- Make the per-parameter bands (cracking, potholes) the headline, and the combined final value
  secondary.
- Never let "not detected by this pipeline" turn into 0 %. The same applies to Patching on Tables
  5.1/5.2: 0 indicators ever stored means it is always not assessed.

### 2. MAJOR: rut depth is dropped from the mean instead of bracketed (covered by #1's fix)

Rut depth has weight 1.00. Averaging over 4 parameters instead of 5 changes the scale: the all-Good
maximum is 2.25 instead of 2.40. It also hides a swing of (3−1)/5 = 0.4 in the 5-parameter final,
which is enough to change the class. RESEARCH §5.3's "N of M" rule is not what App. 4 specifies:
App. 4 averages over *all* the table's parameters.

**Fix:** use the bracket in #1. Keep "averaged over N assessed" only as a sensitivity line.

### 3. MAJOR: the forward-looking frame breaks both the extent estimate and the ±10 m claim

**Evidence.**

- Paper §8.1: "The phone shows road from 1.44 m to about 23–33 m ahead".
- Paper §8.9(4): "a label can come from outside it; positive segments cluster more than chance
  (P(positive | previous positive) 0.43–0.60 against base rates 0.18–0.38)".
- The segment footprint is 2.3–7.3 m ahead of the camera (`frame.near/far_edge_ahead_m`), but the VLM
  sees the whole frame.

**Consequences.**

- One pothole 20 m ahead is labelled on several consecutive segments, which inflates E_p roughly by
  the run length.
- Its chainage is attributed up to ~25 m behind its true position.
- DESIGN §1.8's "±10 m" covers only GNSS plus segment length. It is not honest for the *distress*
  position.

**Fix.**

- Measure the run-length distribution of consecutive positives per type and report it beside E_p.
- Offer a "runs" count, where consecutive positive segments count as one occurrence.
- Change the accuracy statement to: "segment position ±10 m (GNSS σ 4–6 m, latency uncorrected);
  the labelled distress may lie up to ~30 m ahead of the segment in the travel direction".
- In the photo annex, say the frame shows road ahead of the marked chainage.

### 4. MAJOR: the pothole parameter is saturated, and the "lower bound" is not a bound

**Evidence.**

- Potholes are on 770 of 1,287 distressed segments, and the pothole band is Poor on 40 of 41 km.
- One pothole segment in a 100 m section gives 5 % ≫ 0.5 %, so every 100 m section with any pothole
  label is Poor.
- The lower bound n·0.0314 m² / (L·3.5) gives ~0.02 % → Fair. The two bounds disagree almost always,
  so the output is "Fair–Poor" nearly everywhere.
- IRC §7.5.3.4 (PDF p. 38) "A small pothole is defined as 25 mm deep and 200 mm wide" is a
  *severity class*. §7.5.3.1 says potholes are "of varying sizes", so 200 mm is not a minimum.
- Because of #3, n_pothole_segments over-counts potholes. A "lower bound" built on it is not a lower
  bound.

**Fix.**

- Call it the "nominal small-pothole area", not a lower bound.
- Use the **surveyed lane width**, not the carriageway width: one lane is imaged.
- Make the primary pothole figure the count of pothole runs per km (est.), next to the IRC band.
- Report that the pothole labels have no Bengaluru validation (0 expert labels). On Attain, alligator
  recall is 8 %. Pothole precision on Bengaluru video is unknown, and 60 % of distress segments
  labelled pothole is high enough to need a check before any class is quoted.

### 5. MAJOR: "% of length" overstates area for *all* localised distress, not just potholes

DESIGN §2 (via RESEARCH §5.3.2) says cracking and ravelling are "usually extensive along length".
That fails for transverse and longitudinal cracks (995 segment-labels). One transverse crack across a
3.5 m lane affects ~1 m² of a 17.5 m² segment (~6 %), but it counts as 100 % of the segment.

**Fix.**

- State "upper bound on % area for localised cracks" for cracking too.
- Report E_crack split into alligator (areal) and linear cracks.
- Remove the claim from §2 and from paper §8.9(5), which says "as IRC:82 section ratings by length
  do". IRC never defines a by-length denominator: RESEARCH §1.7 itself concludes it is area
  (Proforma 2 records m²).

### 6. MAJOR: the denominator is biased by the gate, and the 50 % threshold is unjustified

**Evidence.**

- The gate skips smoother road: audited skips were distressed 8.1 % vs 19.7 % of sent segments.
- With classified_m as the denominator, E_p is biased upward wherever skips occur.
- Sections that are mostly skipped (probably good road) become "insufficient", so rated sections
  are selected toward distress.
- RESEARCH §5.1.7 recommended 80 %; DESIGN uses 50 % with no reason given.
- 81 of the 387 rated 100 m blocks fall at 50–80 % coverage.

**Fix.**

- Put the skip correction in the main line, not only in a sensitivity line. Show the range from
  "skip = 0 % distress" to "skip at the audited rate's upper CI" (13.3 %).
- Per-type skip rates cannot be estimated: only 13 audited segments were distressed. Say so.
- Count skip_m as *imaged* coverage.
- Justify the threshold, or print both the 50 % and 80 % results and state the choice.

### 7. MAJOR: the chainage smoothing solves a non-problem and adds a bias; validation 3(a) is circular

**Evidence.**

- Centres are already spaced along the camera's s-path: max step 5.1 m, ratio 1.0007. GNSS error
  here is a slowly varying offset, not point jitter.
- A centred 5-node moving average **shortens** curves. My simulation on 5 m sampling:
  - −2 to −5 m per 90° junction turn;
  - −6 to −12 m per 180° turn.
- The planned synthetic test uses i.i.d. σ = 4–6 m noise. On that, a raw 1 km path grows by ~690 m,
  so the test will "prove" smoothing is needed. With realistic correlated noise (ρ = 0.99) raw is
  +3 m and the moving average −4 m.
- Validation 3(a), "smoothed length vs Σ length_m", compares the path with itself, because length_m
  comes from the same s-path.

**Fix.**

- Drop the moving average on surveyed runs and use the centres as they are. If smoothing is kept,
  measure its curve bias.
- Test with *correlated* noise (AR(1), correlation length ≥ 50 m).
- Replace 3(a) with an independent length reference, e.g. OpenStreetMap way lengths for the surveyed
  streets (ODbL, attribute it), or measured distances between landmarks on imagery.

### 8. MAJOR: the survey-network edges create false paths

**Overlap edges** ("any heading within 15 m"):

- They join parallel service roads and the opposite carriageway, and the road under a flyover joins
  the deck above. Bengaluru has many flyovers, and 2-D distance there is ~0.
- 34 same-heading pairs sit 6–15 m apart.
- Dijkstra may hop between them, and with no switching penalty the path can zig-zag between two drives
  on one road.

**Gap edges** ("drive end ↔ any segment of another drive within 600 m", straight line):

- They allow shortcuts through city blocks.
- Even at cost ×3, a gap edge can beat a surveyed detour.
- A straight line is only a *lower bound* on the unrecorded road distance.

**Fix.**

- Join two drives only at real crossings (heading difference > 30°, within ~8 m) or on the same road
  (|Δheading| < 30° and lateral offset < ~5 m).
- Add a per-switch penalty.
- Flag a > 4 m `alt_ellipsoid_m` difference as a possible grade separation and refuse the link.
- Gap edges: only end → start of the **time-next drive** with Δt < 10 min. That reproduces the
  vehicle's actual trip: 42 such pairs, 3–648 m. Allow other gaps only through an explicit via point.
- Print gap lengths as "≥ x m (straight line)". Chainage after a gap carries unknown extra error, so
  say "±10 m on surveyed runs; +unknown per gap".

### 9. MAJOR: attribution is ambiguous on self-overlapping paths, U-turns and two carriageways

**Evidence.**

- Projecting "every segment within 12 m" onto the nearest point of the line has two problems:
  - when the path revisits a road (via points, loops), all segments land on one pass and the other
    pass shows as a gap;
  - out-and-back drives (3 drives with U-turns) put both passes on the same metres, and the "latest
    drive" rule keeps both, so they are counted twice.
- Opposite-heading segments are kept, which mixes two directions of a divided road.
- "Latest drive per 100 m" without a coverage condition can replace a fully covered earlier drive
  with a 20 % sliver.

**Fix.**

- Attribute path nodes by their order along the path.
- Project other segments only into a window of measure around the path's local position.
- Select data per **pass**, i.e. per run of one drive in one direction, not per drive.
- Default to same-direction segments, with opposite ones as an option labelled "other carriageway".
- Choose the drive by coverage first and date second.

### 10. MAJOR: placeholder survey-form values would be printed as facts

`road_name` = "test 1"/"test 2" and `direction` = `increasing_chainage` on all 43 drives are form
defaults. DESIGN §1.5 and §8 print "direction per drive" and the weather from `record.survey`.

**Fix:**

- Derive direction relative to A→B from the heading.
- Print survey-form fields only with the label "as entered on the survey form".
- Never use `survey.direction` for chainage sign.
- Treat the road name and number as user input on the report request.

### 11. MAJOR: the open POST endpoint can be abused, and the rate limit is effectively global

**Evidence.**

- The design rate-limits with slowapi. The default key `get_remote_address` (app/main.py:121) is the
  TCP peer, and through the Cloudflare tunnel every remote user is 127.0.0.1. So "10/minute" is one
  bucket for the whole internet: one person can lock everyone out.
- It also still allows 14,400 stored reports a day, with no cap on storage.
- `prepared_by` and the road name are free text printed on a PDF served from our host. Someone could
  generate an official-looking "Prepared by: BBMP Executive Engineer" report.
- The CSV export of road names is open to spreadsheet formula injection (values starting with `= + - @`).

**Fix.**

- Key the limit on `Cf-Connecting-Ip` only when the peer is loopback, as `classify_rate_key` already
  distinguishes.
- Make `/path` and report preview stateless. Persist (`POST /report` save) only with the operator API
  key, or dedupe by a hash of the inputs, with a hard cap on stored rows.
- Print `prepared_by` as "entered by requester, unverified", or drop it from the open endpoint.
- Prefix CSV cells that start with formula characters with `'`.
- Validate input:
  - lat/lon finite and inside the data bbox + 1 km;
  - width 2–30 m;
  - `chainage_at_a_km` 0–3000;
  - text ≤ 120 chars;
  - via ≤ 8.
- Set `sample` server-side only.
- Embed a Unicode font, so that Kannada road names do not print as empty boxes.

### 12. MAJOR: "snapshot" contradicts "map and report never disagree"

Once an expert review lands, the stored report and the map differ by design.

**Fix.**

- Print "data as of <timestamp>, N segments reviewed at that time".
- `GET` returns `stale: true` plus the count of segments changed since the snapshot.
- Store the frame SHA-256 and embed the annex thumbnails in the stored JSON or PDF, so a later image
  change cannot alter an old report.

### 13. MAJOR: the validation plan misses what can be measured now, and risks cherry-picking

**Add:**

- **Test–retest.** Use the ~180 segments where another drive runs the same heading within 6 m.
  Measure:
  - label agreement on matched segments (κ);
  - E_p and class agreement on overlapping 100 m stretches;
  - chainage difference.

  This is the only reliability evidence possible without ground truth (paper §8.9: "repeatability
  between drives is not yet measured").
- **Independent length.** OSM, see #7.
- **Gap error.** Straight line vs OSM route length for the 42 time-adjacent gaps.
- **Run lengths.** Run-length distribution per type, see #3.
- **Class sensitivity.** Class under the bracket (#1) and under the skip correction (#6).

**Sample reports.** Choose the stretches by a pre-stated rule (random start points plus the named
cases) *before* looking at their ratings. "A short urban stretch with potholes" chosen after viewing
is a showcase, not evidence.

**Limitation to state.** There are **0** expert labels on video and no Bengaluru validation of *any*
label. That includes Bleeding (355 segments), which the report would print as a Proforma-1
percentage.

### 14. MINOR: arithmetic details

- **Truncation.** In floating point, `2 − (9−5)/5` can come out as 1.1999… and truncate to 1.1. Use
  `Decimal`, or `floor(x·10 + 1e-9)/10`.
- **Boundaries.** With 5 m segments, E_p hits exactly 5, 10 and 15 % very often, so compare band
  boundaries with an explicit tolerance and unit-test the boundary values.
- **Rounding.** State the order of rounding (w·r rounded, then the mean rounded). It matches the App-4
  print (1.125 → 1.1).

### 15. MINOR: short stretches and tails

- A = B, or a stretch shorter than ~50 m: refuse with a message.
- A final sub-section shorter than 50 m should merge into the one before. A 20 m tail of 4 segments
  makes one label 25 %.
- Say that a stretch shorter than 1 km is rated as a part-km. IRC records "for each km length"
  (App. 1, PDF p. 69).

### 16. MINOR: label mapping and Proforma completeness

- **Upheaval.** RESEARCH maps Upheaval to Settlement; DESIGN omits it. Pick one. IRC §7.4.6 is
  "Settlement and upheaval".
- **Hairline cracks.** Counting them as cracking is defensible (they sit under Cracks in §7.3), but
  note that they have never been labelled.
- **Proforma-1 columns.** "Drainage and Shoulder condition" and "Total" (PDF p. 70) must appear and be
  marked "not assessed". "Patch Work (%)" and "Depression (%)" must show "not assessed", not 0.
- **Survey timing.** IRC §5.3 asks for urban surveys twice a year, before and after the monsoon.
  Print the season of the survey dates (Oct 2026, during the NE monsoon).

### 17. MINOR: treatment text

Treatment lines chosen from model-judged severity, e.g. "mill out and repair" for high bleeding,
read as instructions. Head the section "Candidate treatments per IRC:82 (to be confirmed by field
inspection)". Print treatments only for distresses whose labels have some validation, or flag the
others as "label unvalidated".

## D. Verdict

The IRC arithmetic is correct (tables, bands, weights, App-4 formula, classification). It can be
built as designed, apart from the float-truncation detail.

The rating **as designed is not reasonable to publish**:

- Two of the four urban parameters are ones the pipeline has never detected. Counting them as Good
  makes Poor mathematically unreachable (#1).
- The pothole parameter is saturated and inflated by the look-ahead (#3, #4).
- The chainage network can produce false paths (#8, #9).

With the fixes to #1–#13 the design is reasonable to build:

- **Rating:** bracket every unassessed or unvalidated parameter, make per-parameter bands and pothole
  runs per km the headline, and use honest position and extent statements.
- **Chainage:** temporal gap links and stricter overlap links, per-pass attribution, and no moving
  average.
- **API:** a keyed save and a per-client rate limit.
- **Validation:** test–retest and an independent length reference.

The paper should present the result as a screening tool, with the bracketed class and the
measured repeatability. It should not present the result as a condition rating of Bengaluru roads
until expert labels exist.
