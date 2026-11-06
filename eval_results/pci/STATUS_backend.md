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

## In progress
- OSM part of pci_validation (detached process, log `eval_results/pci/validation_run.log`; resumable:
  `venv/Scripts/python.exe scripts/pci_validation.py --database geoai` reuses finished parts and cached OSM files)

## Next
final STATUS

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
