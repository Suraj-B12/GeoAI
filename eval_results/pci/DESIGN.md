# A→B Chainage and IRC:82-2015 Condition Rating — Design (v1, 2026-11-07)

Source of every IRC number: `eval_results/pci/RESEARCH.md` (§ refs below = that note). Tables 5.1–5.3 and
the Appendix-4 example were checked visually against the PDF pages (PDF p. 11, 12, 81) on 2026-11-07:
all values in RESEARCH §1.2–1.5 are correct, including the printed rut-depth row "1.3" (an IRC
inconsistency, not a reading error).

Name of the output everywhere: **"IRC:82-2015 distress-based condition rating (estimated from
automated image classification)"**. Never "PCI" alone and never "ASTM PCI". In the UI the tool may be
labelled "Condition report (IRC:82 rating)"; the paper calls it the *IRC rating*, with "PCI" only as
"the IRC:82 rating, which plays the role a PCI plays elsewhere".

## 0. Data facts this design rests on (measured 2026-11-07, production `geoai` DB)

- 8,402 video segments in 43 drives (26, 27, 29 Oct 2026), ≈ 42 km, one phone, one lane, ~5 m each.
- Status: send 6,889 · audit 161 · skip 1,259 · not_covered 93. Skip = gate did not send (never
  classified). Audited skips were distressed 8.1 % [4.8, 13.3] vs 19.7 % of sent (paper §8.7).
- `chainage_start_km` is NULL on every segment (no start chainage was entered on the survey form).
- GNSS 1σ (`record.camera.h_sigma_m`): median 4.0 m, p90 6.1 m.
- Drives are clips: consecutive clips are 3–650 m apart (median ≈ 80 m) with no frames in between.
- Repeat coverage is rare: 34 of 2,230 ~20 m grid cells hold more than one drive.
- `record.survey` has road_id / road_name ("test 1", "test 2": placeholders), lane "L1",
  direction "increasing_chainage", weather.
- Final labels (expert override) come from the same code the map uses (`app/gis_routes._segment_feature`:
  `final_labels`, `vlm_class`, `status`, severity, review state). The report must use exactly those, so the
  map and the report never disagree.

## 1. Chainage

1. **Survey network.** Nodes = segments with a position (any status; not_covered too, they carry
   geometry). Edges:
   - *track*: consecutive k within a drive, length = distance between centres;
   - *overlap/junction*: segments of different drives within 15 m (any heading: allows turns at
     junctions and repeat drives), length = distance;
   - *gap*: drive end ↔ any segment of another drive (and drive start ↔ …) within 600 m, length =
     straight-line distance, cost ×3 (prefer surveyed road), flagged `gap=true`.
2. **Snap A, B** to the nearest node; refuse (HTTP 422, message with the distance) when > 40 m away:
   "pick a point on a surveyed road (nearest surveyed road is 63 m away)". Optional `via` points
   (max 8) to force the path along a chosen road; each snapped the same way.
3. **Path** = Dijkstra A → (via…) → B on cost. Refuse if no path or path > 25 km.
4. **Reference line** = node centres along the path in a local metric frame (equirectangular about A),
   smoothed by a centred moving average of 5 nodes (≈ 25 m) *within* each surveyed run (never across a
   gap; end points kept). Gap pieces are straight lines. Chainage = cumulative length along it, 0 at A.
   (Raw GNSS path length grows with jitter; smoothing removes most of it — measured in §6.)
5. **Attribution.** Every segment within 12 m (perpendicular) of the reference line and, when both
   headings exist, within 45° of the line's local bearing *or its reverse* is projected onto it →
   measure m, extent [m − L/2, m + L/2] clipped to [0, total]. Opposite-heading segments are another
   carriageway direction only on divided roads, which we cannot tell; v1 keeps them but records the
   direction per drive in the report.
6. **Overlapping drives.** Per 100 m section, if more than one drive covers it, use the latest drive
   (survey date of the condition), list the others ("also surveyed on …"). Never average across dates.
7. **Two chainages printed.** *Relative*: "Km 0+000 at A … Km x+yyy at B" (always). *Road chainage*: only
   when the user enters the km-stone chainage at A (`chainage_at_a_km`) and the direction (increasing /
   decreasing): `road = chainage_at_a_km ± m/1000`. Printed "Km 12+350 (12.350)", marked "calibrated by
   the user at A". Without it the report says "Road chainage not calibrated (no km-stone reference)".
8. **Accuracy statement**: "±10 m (phone GNSS σ ≈ 4–6 m plus 5 m segment length); gap lengths are
   straight-line estimates".

## 2. Sections and quantities

- **100 m sub-sections** from A (last one shorter), **km rows** (IRC "kilometre-wise", §5.3), and the
  **whole stretch**. Quantities are summed, then rated (never average ratings).
- Per unit, using attributed segment extents clipped to the unit:
  - `imaged_m` = length of send/audit/skip; `not_covered_m`; `gap_m` (no segment at all);
  - `classified_m` = send/audit segments whose final answer is Normal or Distress (model or expert);
  - `skip_m`; `unknown_m` (sent, no valid answer);
  - per IRC parameter p: `affected_m[p]` = classified length whose final labels contain any label of p.
- **Estimator** `E_p = 100 · affected_m[p] / classified_m` ("% of classified length showing the
  distress", est.).
- **Rated only if** `classified_m ≥ 50 %` of the unit's length (our threshold, stated) — else
  "Insufficient data", no rating, shown grey.
- **Sensitivity lines** (always computed, shown in the report):
  - *gate-skipped as distress-free*: `100 · affected_m / (classified_m + skip_m)`;
  - *potholes at IRC minimum size*: `100 · n_pothole_segments · 0.0314 m² / (classified_m · width_m)`
    (π·0.1², the "small" pothole, IRC §7.5.3.4; width = entered carriageway/lane width, else 3.5 m,
    stated);
  - final value as Σwr/Σw; final without settlement.

## 3. Label → IRC parameter

| IRC parameter | Our final labels | Assessed? |
|---|---|---|
| Cracking | Alligator, Longitudinal, Transverse, Edge, Hairline Cracks (Block/Reflection map here in our taxonomy) | yes (est.) |
| Ravelling | Ravelling | yes (est.) |
| Potholes | Potholes | yes (est., upper bound) |
| Shoving (5.1 only) | Shoving | yes (est.) |
| Patching (5.1, 5.2) | Patching condition indicator, if stored on the segment | yes if present, else "not detected by this pipeline" |
| Settlement and Depression | Settlement, Shallow Depression | low confidence (depth not visible) |
| Rut depth (mm) | — | **not assessed** (needs a 3 m straight edge) |
| Proforma-1 only, not rated | Bleeding, Edge Breaking, Stripping, Hungry Surface, Streaking, Slippage, Corrugation, Rutting (presence) | reported |

## 4. Rating (IRC:82-2015 Tables 5.1/5.2/5.3 + Appendix 4)

- Table chosen by road category: urban → 5.3 (default, Bengaluru streets), highway → 5.1,
  MDR/rural → 5.2. Printed.
- Band lookup with the printed limits (L = Good/Fair limit, U = Fair/Poor limit): `x < L` Good,
  `L ≤ x ≤ U` Fair, `x > U` Poor; urban potholes: 0 Good, `0 < x ≤ 0.5` Fair, `> 0.5` Poor.
- Rating value (interpolation inferred from the App-4 example, RESEARCH §1.5, labelled as such):
  Poor 1.0; Fair `2 − (x − L)/(U − L)`; Good `3 − x/L` (urban potholes Good = 3.0, Fair `2 − x/0.5`),
  truncated to 1 decimal, clamped to the band.
- Weighted rating = w · r rounded to 1 decimal (as the example prints), weights App. 4.
- **Final = mean of weighted ratings over the parameters assessed** (App. 4 formula), rounded to
  1 decimal; class ≤ 1.0 Poor, 1.1–2.0 Fair, ≥ 2.1 Good. Report states "averaged over N of M
  parameters; rut depth not assessed".
- Unit test: the App-4 example → 1.2/1.2/3.0/2.1/1.5/1.2/(1.2 vs printed 1.3) and final 1.3 Fair.

## 5. Treatments

Per unit: class text (RESEARCH §6.1, Table 9.1 by category) + one line per distress present, chosen by
the worst *visually judged* severity among its segments (RESEARCH §6.2), each with its clause. Potholes
always add "rectify straightway (§5.3)". Poor adds "detailed investigation for strengthening (Table 9.1
note)". Alligator adds "pit/core to find the cause (§7.3.3.5)". Severity is labelled "visually judged
from images; widths and depths not measured".

## 6. Validation (what we can honestly measure now)

1. IRC arithmetic: App-4 example, every band boundary (inclusive/exclusive), each table.
2. Chainage on synthetic geometry with known length (straight, arc, L-turn), with simulated GNSS noise
   σ = 4 and 6 m: bias of raw vs smoothed length; A/B swap symmetry; via points; refusal > 40 m.
3. Real data: (a) smoothed path length vs sum of segment `length_m` along the path; (b) repeat drives
   over the same road: difference of projected chainage for matched segments; (c) gap share.
4. Rating sensitivity on the sample stretches: how often the class changes under each sensitivity
   line (skip, pothole-size, Σwr/Σw, no settlement).
5. NOT available: agreement with an engineer's IRC survey or with expert-labelled sections (1 expert
   label in total). Stated as the main limitation; the paper gives the protocol (expert-label every
   segment of ~10 random 100 m sections; compare E_p and class).

## 7. API (schema `geoai.pci.v1`) — added to the GIS router, no API key (like /gis/*), rate-limited

- `POST /gis/pci/path` `{a:{lat,lon}, b:{lat,lon}, via?:[…]}` → path line GeoJSON, snapped A/B (+ snap
  distance), total length, surveyed / gap length, drives used. Fast (no rating). 422 with message on refusal.
- `POST /gis/pci/report` `{a, b, via?, road:{name, number?, category:"urban"|"highway"|"rural",
  carriageway_width_m?, lane?, surface_type?, chainage_at_a_km?, chainage_direction?}, prepared_by?}`
  → full report JSON, saved; returns `id`. 10/minute. Input lengths bounded, text escaped in the PDF.
- `GET /gis/pci/reports` → list (id, title, road, length, final class, created, `sample` flag).
- `GET /gis/pci/reports/{id}` JSON · `/gis/pci/reports/{id}.pdf` · `/gis/pci/reports/{id}.csv`
  (100 m sections, UTF-8 BOM) · `/gis/pci/reports/{id}.geojson` (sections as coloured lines).
- Reports are **snapshots**: the JSON stores every input number and the data timestamp, so a PDF
  re-rendered later shows the same numbers. Storage: `research.pci_reports` (migration 009, local PG).
- Sample reports: a script generates 4–5 real stretches (short urban stretch with potholes, a 1 km
  run, a stretch across clip gaps, the longest drive) flagged `sample=true`; regenerated, never edited.

## 8. Report content (PDF, A4; JSON has the same)

Header (Proforma-1 fields: road name/number, category & table, Km from/to (relative and calibrated),
A/B lat-lon, direction, lane, carriageway width, surface type, survey dates & weather, method, device,
pipeline/model version, report id, generated at IST, prepared by) → Summary (final rating & class,
length, coverage, Poor/Fair/Good/insufficient length, potholes, key actions) → route figure (vector
track coloured by section class with chainage ticks, A/B, north arrow, scale bar; no basemap) → strip
chart → km-wise Proforma-1 table + App-4 rating table → 100 m section table → treatments → photo annex
(Poor sections and pothole segments, ≤ 24 frames, each with chainage, lat/lon, IST time, labels,
confidence, review state) → method → limitations (RESEARCH §5.4 incl. measured model error) →
sign-off ("Recorded by / Checked by", blank). Every estimated number carries "(est.)". No text may
overlap; tables wrap; verified by a PDF text-box overlap test.

## 9. Ownership

- Backend agent: `app/pci_core.py`, `app/pci_chainage.py`, `app/pci_report_pdf.py`, `app/pci_routes.py`,
  one `include_router` line in `app/gis_routes.install`, `migrations/009_pci_reports.sql`,
  `scripts/setup_local_db.py` + `scripts/tests/test_local_db.py` (count), `scripts/tests/test_pci_*.py`,
  `scripts/pci_*.py` (samples, validation), `eval_results/pci/*` results.
- Frontend agent: a scratch copy of `webgis/` → tested behind a harness on port 8010 → copied into
  `webgis/` only when the coordinator says so (`webgis/` is served live).
- Critic agent: read-only, writes `eval_results/pci/REVIEW_*.md`.
- Coordinator: paper, CLAUDE.md, memory, commits, production restart.
