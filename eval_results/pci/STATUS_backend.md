# STATUS — backend agent (PCI / chainage report)

Updated: 2026-11-07

## Done
- **Real data in `geoai_test` (for the frontend harness):** `scripts/pci_copy_to_test_db.py` copied
  45 sessions, 8,402 segments, 7,050 assessments from production `geoai` (read-only transaction) into
  `geoai_test` in 9 s. photo_id set NULL on copies; image_path unchanged (frames resolve to the real files).
  **`scripts/tests/test_local_db.py` / `test_expert_review.py` DROP `geoai_test`** — after running them, re-run
  `venv/Scripts/python.exe scripts/pci_copy_to_test_db.py` (idempotent, ~10 s; `--check` says present/absent).
  `test_pci_routes.py` re-copies by itself when the copy is missing.
- `migrations/009_pci_reports.sql` (research.pci_reports), registered in `scripts/setup_local_db.py`;
  `test_local_db.py` count 10 → 11. Applied to `geoai_test`.
- `app/pci_core.py` + `scripts/tests/test_pci_core.py` **8/8 pass** (App-4 example 1.3 Fair; all boundaries).
- `app/pci_chainage.py` + `scripts/tests/test_pci_chainage.py` **10/10 pass**. Real data: graph 8,402 nodes,
  edges track 8,359 / same 208 / cross 15 / gap 36; built 3.5 s (cached); 8.3 km report in 0.1 s.
- `app/pci_report_pdf.py` + `scripts/tests/test_pci_pdf.py` **4/4 pass** (no text-span overlap, nothing off
  page, all §8 sections). Looked at pages by eye (route figure, strip chart, Proforma table, App-4 tables,
  100 m table, treatments, photo annex) — layout fine. Real 8.3 km drive: 19 pages, 1.1 MB, ~5 s.
- `app/pci_routes.py` + one line in `app/gis_routes.install` + `scripts/tests/test_pci_routes.py` **5/5 pass**.

- `scripts/pci_samples.py` — rule + chosen inputs written FIRST to `eval_results/pci/samples_rule.json`
  (8 samples: longest drive, 3 random 1 km per day for 27 and 29 Oct, one two-gap stretch; 1 refused draw recorded).
  Run on `geoai_test`: sample reports ids 4–11 saved (sample=true); all 8 stored PDFs pass the overlap check.
  **Coordinator, production:** `venv/Scripts/python.exe scripts/pci_samples.py --database geoai` (reuses the same
  selection; dedupes; replaces older sample rows of the same label).
  Design choice: §9(b) "three 1 km stretches … of 27 Oct and 29 Oct" read as three PER DAY (six).

- `scripts/pci_validation.py` → `eval_results/pci/validation.{json,md}`, checkpoints in
  `eval_results/pci/validation_parts/`, OSM downloads cached in `C:\GeoAI\pci\osm_cache\` (ODbL). Parts arithmetic,
  synthetic, retest, runs, sensitivity done on production `geoai` (read-only). Part `samples` says "not done"
  until the coordinator runs pci_samples on production; then `--redo samples`.
- Tests: pytest of the four PCI test files **27 passed**; `validate_all.py` **18/18**; `test_gis_routes.py` **160/160**
  (it uses its own `geoai_gis_test`). NOT re-run: `test_local_db.py` / `test_expert_review.py` (they DROP `geoai_test`,
  which the frontend harness is using right now). Only change affecting them: SCHEMA_FILES count 10 → 11 (asserted).
  Coordinator: run them when the frontend is done, then re-run `scripts/pci_copy_to_test_db.py`.

## Round 2 (REVIEW_backend.md) — progress
- [x] #1 BLOCKER path segments dropped: rewritten selection (path always used; attached passes fill only
  ranges the path did not image; not_covered path segments replaced there; revisits counted once (#9)).
  New tests `test_overlap_path_always_used_others_fill_holes`, `test_gap_mid_section_keeps_every_path_segment`,
  `test_out_and_back_counted_once` → test_pci_chainage 12/12.
- [x] B2 `attribution` table + `attribution_columns` in the report JSON; annex uses the used segments and the
  pictured segment's own measure (#6), reason prints the band(s) (#7).
- [x] B1 JSON: `summary.priority_notes`, `summary.final.note`; `rating.params[].band_note` when the worse end
  comes only from skipped segments (#5). PDF: boxed priority line on page 1 (IRC §5.3 quote, 0.50 of 4.00 weight,
  "labels unvalidated, Attain P 0.43 / R 0.25, 0 expert labels"), Overall line muted + note, route map and strip
  chart coloured by the POTHOLE band, Overall moved to the last strip row "(secondary)", legend = band colours
  (distinct hues; violet = "Good on labels, Poor via skips"), riding-quality rule at the top of §8 and in every
  km block with potholes, "(est.)" in column heads, route chainage labels every 100/200/500 m/1 km by length,
  "entered point snapped x m to the nearest segment", annex caption without doubled parentheses.
- [x] #2 save: rate limit BEFORE the key check + failed-key bucket (5 wrong keys / 10 min / client → 429, the right
  key too while locked out; failures logged). Test `test_wrong_keys_are_rate_limited`.
- [x] #10 text: Unicode Cc/Cf/Co/Cs/Cn refused (RTL override, zero-width); emoji and Kannada messages both say
  "write … in English letters"; decreasing road chainage that would go below 0 at B → 422.
- [x] #8 method sentence; #11 "± one segment at section boundaries" sentence; ACCURACY adds "chainage LENGTH
  accuracy has not yet been measured against an independent reference".
- [x] #12 test hygiene: test_pci_routes restores the bumped `updated_at` (trigger disabled for the restore).
- [x] Validation: κ with block-bootstrap CI + hours between paired drives; audited-skip pothole rate 9/161
  (0.056 [0.030, 0.103] Wilson) and 118/408 blocks "Poor–Good" for potholes only from the 13.3 % skip bound;
  "Statements for the paper" section (bracket never a single class 0/408; AR(1) noise ≠ our chainage error; test–retest
  67 pairs / 2 blocks → class repeatability not measured). OSM part DONE but read as **inconclusive** (vertex
  snapping; ours/OSM 0.21–1.00) — no chainage-length accuracy claimed.
- [x] geoai_test re-copied (clean: 0 bumped rows); samples regenerated on geoai_test → ids 18–25 (all 8 PDFs: 0
  overlaps); `pci_samples.py` now records requested vs routed length in `samples_generated_<db>.json`.
- [x] Final checks: 4 PCI test files **30 passed**; test_gis_routes **160/160**; validate_all **18/18**; looked at
  page 1 + route of samples 19 and 25 and synthetic PDFs by eye.

## In progress
Nothing. Backend round 2 complete.

## API additions in round 2 (frontend: please switch to these)
- `report.attribution` + `report.attribution_columns` = `["segment_id","measure_m","section","km","pass","used","role"]`,
  one row per candidate segment (path, attached, opposite); `used=true` rows are exactly what the numbers use;
  role ∈ path | revisit (counted once) | attached | attached, range already covered | opposite direction |
  path, not imaged; range filled by an attached pass. Use it instead of guessing the nearest section.
- `summary.priority_notes: [str]` (show beside the class; boxed on PDF page 1); `summary.final.note`.
- `rating.params[].band_note` (e.g. "Good (no pothole label); Poor if skipped segments hold a pothole").
- sections: `passes_used: [{pass, session_id, date_ist}]` (path always used; `chosen_pass` = main one).
- annex rows: `pass`, `measure_m`/`chainage` = the pictured segment's own.
- Map colour convention in the PDF: POTHOLE band (headline), never the overall class.
- Save: 429 after 5 wrong keys / 10 min per client (also for the right key while locked out).
- Text fields: format/control characters (RTL override, zero-width) → 422.

## Production steps for the coordinator
1. Restart production to load `/gis/pci/*`; migration 009 self-applies (or `scripts/setup_local_db.py`).
2. `venv/Scripts/python.exe scripts/pci_samples.py --database geoai` (same pre-registered selection).
3. `venv/Scripts/python.exe scripts/pci_validation.py --database geoai --redo samples` (fills the samples table).
4. When the frontend no longer uses geoai_test: run `test_local_db.py` / `test_expert_review.py`, then re-run
   `scripts/pci_copy_to_test_db.py`.

## API as built (for the frontend) — all JSON bodies; errors are `{"detail": "..."}`
Base: the GIS server. Harness: mount `app.pci_routes.router` (or `app.gis_routes.install`) on port 8010 with
`DATABASE_URL` → `geoai_test`.

- `GET /gis/pci/meta` → `{schema, name, data_as_of, bbox:[w,s,e,n], bbox_margin_km, categories:[{id,name,table,title,params}],
  default_category:"urban", default_lane_width_m:3.5, limits:{via_max:8, snap_max_m:40, min_path_m:50, max_path_m:25000,
  explicit_gap_max_m:300, lane_width_m:[2,30], chainage_at_a_km:[0,3000], text_max:120, stored_max:200, rate_per_minute},
  assessment, fields}`
- Request body for path/report/save: `{a:{lat,lon}, b:{lat,lon}, via?:[{lat,lon}] (≤8), category?: "urban"|"highway"|"rural",
  lane_width_m?: 2–30, chainage_at_a_km?: 0–3000 + chainage_increases_towards_b: bool (both or neither),
  road_name?, road_number?, surface_type?, prepared_by?, title?: text ≤120}`. Unknown field → 400.
  Bad value → 422. A/B/via > 40 m from the survey → 422 "… m from the nearest surveyed segment (limit 40 m)".
  Kannada/other Indic text → 422 (no complex-script shaping available; see Design questions).
  No surveyed path → 422 (message suggests a via point; via legs may cross an explicit gap ≤ 300 m).
- `POST /gis/pci/path` (30/min) → `{schema, length_m, chainage_a, chainage_b, snapped:[{name,node,segment_id,distance_m,lat,lon,
  snapped_lat,snapped_lon}], gaps:[{from_m,to_m,straight_m,kind:"gap"|"explicit_gap",label:"≥ x m (straight line)",...}], gap_m,
  switches, passes:[{id,kind,session_id,device,k_first,k_last,from_m,to_m,direction,date_ist,segments}], other_passes, opposite_passes,
  geometry:{surveyed: MultiLineString, gaps: MultiLineString}, geojson: FeatureCollection(kind surveyed|gap)}`
- `POST /gis/pci/report` (10/min) → report JSON: `{schema, name, report_id:null, generated_at, generated_at_ist, data_as_of, input,
  header{road_name, road_number, category, category_name, table, table_title, km_from, km_to, road_chainage_from/to:{km,text}|null,
  road_chainage_note, a, b, via, directions, lane_imaged, lane_width_m, surface_type, survey_dates_ist, season,
  weather_as_entered_on_survey_form, method, devices, model, prepared_by, prepared_by_note},
  summary{length_m, chainage_a, chainage_b, coverage, coverage_status, gap_m, gaps, headline{cracking{band,E_lo,E_hi},
  potholes{...}, pothole_runs, pothole_runs_per_km, pothole_nominal_area_pct, pothole_nominal_area_band},
  final{lo,hi,class_lo,class_hi,class,label,n_params,n_assessed}|null, sections_by_class, km_by_class, segments_used,
  segments_expert_reviewed}, path{…as /path…, vertices:[[lon,lat,m]], steps}, sections:[unit], km:[unit], stretch: unit,
  treatment_heading, treatment_any_class, annex:[{segment_id, reason, measure_m, chainage, lat, lon, time_ist, labels, severity,
  confidence, review_state, image_sha256, note}], accuracy, method[], limitations[], segment_ids[], assessment}`.
  unit = `{from_m, to_m, chainage_from, chainage_to, road_chainage_from/to, quantities{length_m, imaged_m, classified_m, skip_m,
  not_covered_m, unanswered_m, gap_m, coverage, coverage_status: rated|low_coverage|insufficient, rated_80, rated_50,
  extent{cracking|cracking_alligator|cracking_linear|potholes|ravelling|…: {lo,hi,uncorrected,affected_m}}, reported{Bleeding,…},
  runs{potholes,cracking}, runs_per_km, run_length_hist, pothole_nominal_area_pct, worst_severity, labels_present},
  rating{rated, coverage_status, table, params:[{param,name,unit,weight,assessed,band,E_lo,E_hi,band_lo,band_hi,rating_best,
  rating_worst,weighted_best,weighted_worst,evidence}], not_assessed, final, sensitivity, headline}, treatment{class, distress[]},
  geometry: MultiLineString (sections and km only)}`; sections also `{index, chosen_pass{pass,session_id,date_ist},
  also_surveyed[], other_direction[]}`; km also `{km, part_km, sections}`.
  Class strings use an EN DASH: "Poor–Fair". Insufficient → `rating.final = null`.
- `POST /gis/pci/report.pdf` / `.csv` (10/min) → file download (stateless preview, "Preview (not stored)").
- `POST /gis/pci/reports` (save) — header `X-API-Key: <operator key>`; 401 wrong/missing, 403 if the server has no
  API_KEYS; 409 when 200 non-sample reports are stored → `{id, created: bool, input_hash, created_at, links{json,pdf}}`
  (same input + same data → same id, created false).
- `GET /gis/pci/reports` → `{schema, count, reports:[{id, created_at, title, sample, sample_rule, data_as_of, category, input,
  summary, links{json,pdf,csv,geojson}}]}` (samples first, then newest).
- `GET /gis/pci/reports/{id}` → `{schema, id, created_at, title, sample, sample_rule, data_as_of, stale, changed_segments,
  stale_note, report}`; `/{id}.pdf` stored bytes (frozen); `/{id}.csv`; `/{id}.geojson` (sections with class/bands, gap, A/B,
  annex points). 404 for anything else.

## Production (for the coordinator)
- Migration 009 is applied automatically on first use (`app.pci_routes.ensure_schema`, like 008 via expert_review);
  to record it in the ledger run `venv/Scripts/python.exe scripts/setup_local_db.py` (applies only 009, idempotent).
- The new routes load on the next production restart (gis_routes.install includes them).

## Design questions (resolved with the safest option; please confirm)
1. **Kannada in the PDF.** Nirmala UI covers Kannada glyphs, but ReportLab here has no complex-script shaping
   (uharfbuzz not installed). Checked by eye: "ಹೊ" printed with a detached dotted-circle sign and "ಸ್ತೆ" as "ಸತೆ"
   (wrong letters). Resolution: **refuse Indic text with a clear 422 message** (DESIGN §7 allows refusing). Proposal:
   `uv pip install uharfbuzz` and enable ReportLab shaping, then lift the refusal (Nirmala is already registered).
2. **Gap edges 36, not 42.** DESIGN §1.1 says "42 such pairs"; 42 is every consecutive pair of the 43 drives. With
   the stated rule (same device, Δt < 10 min) there are 36 on the copied data. Implemented the rule as written.
3. **Saving with no API_KEYS configured** (dev mode) is refused (403) rather than open, since the design requires
   the operator key.
4. **"Same travel direction"** for attaching other passes = within 30° of the linked path node's camera heading
   (symmetric under A/B swap). The path's own passes are always rated, whichever way the vehicle drove; their
   direction is reported ("vehicle travelled towards B / towards A").
5. Runs per km use the surveyed denominator D = classified + skipped length (the same D as E).
