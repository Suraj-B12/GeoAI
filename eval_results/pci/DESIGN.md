# A→B Chainage and IRC:82-2015 Condition Rating — Design v2 (2026-11-07)

v1 was reviewed by the critic agent (`REVIEW_design.md`, 1 BLOCKER, 12 MAJOR, 4 MINOR). v2 adopts every
fix; the "Review #" column/notes say where. IRC numbers: `RESEARCH.md` §1.2–1.5, checked visually against
PDF pp. 11, 12, 81 (all correct; the App-4 rut-depth row "1.3" is IRC's own inconsistency).

**Name:** "IRC:82-2015 distress-based condition rating — screening estimate from automated image
classification". Never "PCI" alone, never ASTM. The paper presents it as a **screening tool with a
bracketed class**, not as a condition rating of Bengaluru roads, until expert labels exist (Review D).

## 0. Data facts (production `geoai`, 2026-11-07; Review §B)

- 8,402 segments, 43 drives (26/27/29 Oct 2026), ≈ 42 km, one phone, one lane. Consecutive centres are
  5.0 m apart (max 5.1 m; centre-path ÷ s-path = 1.0007). Heading present on all but the 93 not_covered.
- Status: send 6,889 · audit 161 · skip 1,259 · not_covered 93. Audited skips distressed 8.1 % [4.8, 13.3]
  vs 19.7 % of sent (paper §8.7).
- **0 expert-reviewed video segments. 0 Patching indicators. Ravelling, Settlement, Shallow Depression,
  Shoving, Hairline: 0 labels.** Distressed segments (1,287): Potholes 770, Longitudinal 720, Bleeding 355,
  Transverse 275, Alligator 148, Edge Breaking 81, …
- Survey-form `road_name` ("test 1/2") and `direction` (`increasing_chainage`) are **form defaults on all
  43 drives**: never printed as facts, never used for chainage sign (Review #10).
- Camera sees ≈ 1.4–33 m ahead; segment footprint 2.3–7.3 m ahead; positives cluster (P(pos|prev pos)
  0.43–0.60 vs base 0.18–0.38) (Review #3).
- Free-form list validity on Attain test (the list the video pipeline uses; `stage2_probe_report.md`):
  linear crack P 0.89 / R 0.82, MCC 0.257; alligator R 0.08; pothole P 0.43 / R 0.25, MCC 0.219;
  ravelling 0 / 0. Stage 1 recall 51.8 %.

## 1. Chainage (Review #7, #8, #9, #15)

1. **Graph.** Nodes = segments with a position.
   - *Track* edges: consecutive k within a drive.
   - *Same-road* links between drives: |Δheading| < 30° (or > 150° = same road, opposite direction) and
     distance < 6 m; *crossing* links: |Δheading| ≥ 30°, distance < 8 m. Refuse a link when
     |Δ alt_ellipsoid_m| > 4 m (possible flyover). Each switch between drives costs +50 m.
   - *Gap* edges: only end of a drive → start of the **time-next drive** of the same device with
     Δt < 10 min (the vehicle's real trip; 42 such pairs, 3–648 m). Cost = straight-line × 1.5; flagged.
     No other gaps; a user who wants another join adds a via point (which may create an explicit gap ≤
     300 m, flagged the same way).
2. **Snap** A, B, via (≤ 8) to the nearest node; refuse > 40 m (422 with the distance). Refuse A ≈ B or
   path < 50 m, path > 25 km, no path.
3. **Path** = Dijkstra through the via points. **No smoothing**: chainage = cumulative distance between the
   path's segment centres (Review #7: centres are already 5 m apart along the camera's path; a moving
   average shortens curves). Gaps: straight line, printed "≥ x m (straight line)".
4. **Attribution per pass.** The path is a sequence of *passes* (a run of one drive in one direction). Path
   nodes take their measure from their order on the path. Segments of *other* passes join a path pass
   only through same-road links, projected into a ±30 m measure window around the linked node (no
   global nearest-point projection, so loops and U-turns cannot collapse). Default: same travel
   direction only; opposite-direction passes are listed as "other direction / possibly other
   carriageway" and not rated.
5. **Overlap choice per 100 m section:** the pass with the most classified length; ties → latest date.
   Others listed "also surveyed on …". Never averaged.
6. **Direction** relative to A→B is derived from camera heading vs path bearing (per pass).
7. **Two chainages.** Relative "Km 0+000 at A … Km x+yyy at B" always. Road chainage only when the user
   enters `chainage_at_a_km` **and** whether road chainage increases towards B: road = at_A ± m/1000,
   printed "Km 12+350 (12.350), calibrated by requester at A (unverified)". Else "Road chainage not
   calibrated (no km-stone reference)".
8. **Accuracy statement:** "Segment position ±10 m on surveyed runs (phone GNSS σ 4–6 m, receiver latency
   not corrected, 5 m segments); unknown extra error after each gap. A labelled distress may lie up to
   ~30 m *ahead* of the marked segment in the travel direction (forward-looking camera)."

## 2. Units, coverage, estimators (Review #3, #4, #5, #6, #15)

- Units: **100 m sub-sections** from A (a tail < 50 m merges into the previous one), **km rows** (IRC
  "kilometre-wise"; a stretch < 1 km is a part-km, said so), **whole stretch**. Sum quantities, then rate.
- Per unit: `length_m`, `imaged_m` (send+audit+skip), `classified_m` (send/audit with a Normal/Distress
  answer), `skip_m`, `not_covered_m`, `gap_m`, `unknown_m`.
- Coverage = imaged_m / length_m. **Rated if coverage ≥ 80 %** (RESEARCH §5.1.7; stated as our choice);
  50–80 % rated but flagged "low coverage"; < 50 % or classified_m = 0 → "Insufficient data".
  The JSON also holds the 50 %-rule result so the paper can report both.
- For parameter p: `affected_m[p]` = classified length whose final labels hold a label of p.
  **Skip-corrected range** (main line): denominator D = classified_m + skip_m;
  `E_lo = 100·affected/D` (skips distress-free), `E_hi = 100·(affected + 0.133·skip_m)/D` (skips at the
  audited rate's upper CI). Per-type skip rates are not estimable (13 distressed audits): stated.
  Also stored: uncorrected `100·affected/classified_m`.
- **Runs:** consecutive positive segments of one pass = one occurrence. Report runs per km per type and the
  run-length distribution (Review #3). Potholes: **runs per km is the primary figure**.
- Every % is "% of surveyed length with the distress visible (est.)" and is an **upper bound on % area for
  localised distress, cracks included** (Review #5). Cracking split into alligator (areal) and linear.
- Nominal pothole area (sensitivity, not a bound): runs × 0.0314 m² (IRC §7.5.3.4 *small*-severity pothole,
  200 mm) / (D × surveyed lane width; default 3.5 m, stated).

## 3. Label → IRC parameter, and "assessed" (Review #1, #16)

A parameter is **assessed** only if the labels feeding it have measured recall > 0 on a labelled set.

| IRC parameter | Final labels | Status (v2) | Evidence |
|---|---|---|---|
| Cracking | Alligator, Longitudinal, Transverse, Edge, Hairline Cracks | assessed (weak) | linear crack MCC 0.257; alligator R 0.08 (Attain) |
| Potholes | Potholes | assessed (weak) | MCC 0.219, P 0.43, R 0.25 (Attain) |
| Ravelling | Ravelling | **not assessed** | R 0 on Attain; 0 labels on video |
| Settlement and Depression | Settlement, Shallow Depression (Upheaval n/a in taxonomy) | **not assessed** | never evaluated; 0 labels |
| Shoving (5.1) | Shoving | **not assessed** | never evaluated; 0 labels |
| Patching (5.1, 5.2) | Patching indicator | **not assessed** | 0 indicators stored on video |
| Rut depth (mm) | — | **not assessed** | needs a 3 m straight edge |
| Proforma-1 only (reported, not rated) | Bleeding, Edge Breaking (+ Stripping, Hungry Surface, Streaking, Slippage, Corrugation, Rutting presence) | reported, "label unvalidated" | — |

Not-assessed parameters are **never 0 %**: printed "not assessed".

## 4. Rating (Tables 5.1/5.2/5.3 + Appendix 4; Review #1, #2, #14)

- Table by category: urban 5.3 (default), highway 5.1, MDR/rural 5.2.
- Per assessed parameter: band for E_lo and for E_hi (class range if they differ), rating value by the
  App-4-inferred interpolation (Poor 1.0; Fair 2 − (x−L)/(U−L); Good 3 − x/L; urban pothole Good = 3.0,
  Fair 2 − x/0.5), truncated with `floor(10x + 1e-9)/10`, clamped to its band. Band limits compared with
  tolerance 1e-9 (E hits 5/10/15 % exactly often); boundary values unit-tested.
- Weighted = w·r rounded half-up to 1 decimal with `Decimal` (App-4 prints 0.75·1.5 = 1.125 → 1.1 and
  0.75·1.2 = 0.9); pinned by the App-4 test.
- **Final = mean over ALL parameters of the table** (App. 4). Not-assessed parameters are **bracketed**:
  final_lo with their r = 1.0, final_hi with r = 3.0 (combined with E_hi / E_lo of the assessed ones), each
  rounded to 1 decimal and classed (≤ 1.0 Poor, 1.1–2.0 Fair, ≥ 2.1 Good).
  Print one class only when both ends agree, else "Poor–Fair" / "Fair–Good" / "Poor–Good"
  ("ravelling, settlement, rut depth not assessed").
- **Headline = the per-parameter bands of the assessed parameters** (cracking, potholes) + pothole runs/km;
  the bracketed final class is secondary. Sensitivity lines: Σwr/Σw; "averaged over assessed only";
  uncorrected (no skip correction); 50 % coverage rule.
- App-4 example unit test: 1.2 / 1.2 / 3.0 / 2.1 / 1.5 / 1.2 / 1.2 (printed 1.3) → 1.3 Fair.

## 5. Treatments (Review #17)

Heading: "Candidate treatments per IRC:82-2015 — to be confirmed by field inspection. No label in this
report has been validated against Bengaluru expert labels." Class text (RESEARCH §6.1 / Table 9.1 by
category) per km and stretch, using the bracket (e.g. "Poor–Fair: investigate strengthening need if Poor
is confirmed (Table 9.1 note); else periodic renewal …"). Per distress present: RESEARCH §6.2 line by worst
visually judged severity, with clause; potholes add "rectify straightway (§5.3)"; alligator adds "pit/core
to find the cause (§7.3.3.5)". Unvalidated types marked "label unvalidated".

## 6. Validation (Review #13) — script `scripts/pci_validation.py` → `eval_results/pci/validation.{json,md}`

1. IRC arithmetic: App-4 example; every band boundary of each table; truncation/rounding; bracket.
2. Chainage, synthetic: straight, arc, 90° and 180° turns; **correlated** GNSS noise (AR(1), correlation
   length ≥ 50 m, σ 4 and 6 m) → length error; A/B swap symmetry; via; refusals.
3. **Test–retest** on real overlaps (same-road links, same direction): label agreement (κ, per type
   Distress / Pothole / Cracking), E_p and band agreement on overlapping 100 m stretches, chainage offset of
   matched segments.
4. **Independent length**: OpenStreetMap (Overpass, ODbL; attribute) way lengths for the surveyed path of
   the sample stretches if reachable; gap straight-line vs OSM route length for the 42 time-adjacent gaps.
   If OSM cannot be reached, say "not done", never substitute.
5. Run-length distribution per type.
6. Class sensitivity over all 100 m sections of all drives: bracket width, E_lo vs E_hi band changes,
   coverage-rule 50 vs 80 %, Σwr/Σw.
7. Limitation stated: 0 expert labels on video; no Bengaluru validation of any label.

## 7. API (`geoai.pci.v1`, in the GIS router; Review #11, #12)

- Rate limit keyed on `Cf-Connecting-Ip` when the TCP peer is loopback (as `classify_rate_key`), else the
  peer. Input bounds: lat/lon finite and inside the data bbox + 1 km; via ≤ 8; width 2–30 m;
  chainage_at_a_km 0–3000; text ≤ 120 chars; category enum.
- `POST /gis/pci/path` → path GeoJSON, snapped points, lengths, passes, gaps (stateless, 30/min).
- `POST /gis/pci/report` → full report JSON (stateless preview, 10/min).
- `POST /gis/pci/report.pdf` / `.csv` → the same, rendered (stateless download, 10/min).
- `POST /gis/pci/reports` (save) — **operator API key required**; hard cap 200 stored; dedupe by input
  hash. `sample` is set server-side only (by the sample script).
- `GET /gis/pci/reports` (list), `GET /gis/pci/reports/{id}` (JSON + `stale` flag + count of segments whose
  label/review changed since `data_as_of`), `/{id}.pdf` (the stored PDF bytes, frozen), `/{id}.csv`,
  `/{id}.geojson`.
- Free text (road name/number, prepared_by) printed as "entered by requester, unverified". CSV cells
  starting with `= + - @` prefixed with `'`. PDF uses an embedded Unicode font that covers Kannada
  (e.g. Windows Nirmala UI) or refuses unsupported scripts with a clear message.
- Storage `research.pci_reports` (migration 009, local PG): id, created_at, input JSON, input hash,
  report JSON, pdf bytea, sample flag, data_as_of, title.

## 8. Report content (PDF, A4; JSON the same)

Header (Proforma-1 fields: road name/number *as entered*, category & table, Km from/to (relative; road
chainage if calibrated), A/B lat-lon, direction (derived), lane imaged (one), surveyed lane width,
surface type *as entered*, survey dates and **season** (IRC §5.3 asks pre- and post-monsoon), weather *as
entered on the survey form*, method, device, model/pipeline version (P4, prompts v2 id, gate config),
report id, data as of, generated at IST, prepared by *unverified*) → **Summary** (headline bands for
cracking & potholes, pothole runs/km, bracketed final class, length, coverage, gaps) → route figure
(vector track coloured by section band, chainage ticks every 100 m / 1 km, A/B, gaps dashed, north arrow,
scale bar; no basemap) → strip chart → km-wise Proforma-1 table (all columns, "not assessed" where so,
Drainage/Shoulder and Total "not assessed") + App-4 rating table with bracket → 100 m table → candidate
treatments → photo annex (Poor sections and pothole runs, ≤ 24 frames embedded, each: chainage,
lat/lon, IST time, labels, confidence, review state, "frame shows road ahead of the marked chainage") →
method → limitations (RESEARCH §5.4 + Review findings: area vs length, look-ahead, not-assessed
parameters, model validity numbers, 0 expert labels, gate skip, GNSS) → sign-off ("Recorded by / Checked
by", blank). No overlapping text: verified by a PDF text-box overlap test over every page.

## 9. Sample reports (Review #13)

Chosen by a rule fixed **before** any rating is looked at, in `scripts/pci_samples.py` (seed 20261107):
(a) the longest drive end to end; (b) three 1 km stretches starting at random path nodes (seeded) along
the time-ordered trip of 27 Oct and 29 Oct; (c) one stretch that crosses ≥ 2 time-adjacent gaps, chosen at
random among such stretches. Saved with `sample=true`; regenerated, never edited.

## 10. Ownership

Backend agent: `app/pci_core.py`, `app/pci_chainage.py`, `app/pci_report_pdf.py`, `app/pci_routes.py`, one
include line in `app/gis_routes.install`, `migrations/009_pci_reports.sql`, `scripts/setup_local_db.py` +
`scripts/tests/test_local_db.py` (count), `scripts/tests/test_pci_*.py`, `scripts/pci_*.py`,
`eval_results/pci/validation.*`, `STATUS_backend.md`.
Frontend agent: scratch copy of `webgis/`, tested behind a harness on port 8010; copied into `webgis/` only
on the coordinator's word (served live). `STATUS_frontend.md`.
Critic: read-only, `REVIEW_*.md`. Coordinator: paper §9, CLAUDE.md, memory, commits, production restart.
