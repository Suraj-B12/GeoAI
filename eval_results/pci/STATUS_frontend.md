# STATUS — frontend (A→B condition report in the WebGIS)

**Deployed 2026-11-07 (coordinator copied webgis_dev → webgis/ with ?v=20261107 cache-busting), harness stopped
(port 8010 free; no background processes left).**

Updated: 2026-11-07 — **DONE** (all 7 steps). UI runs on the REAL `/gis/pci/*` routes (harness mode "real",
geoai_test with the backend's 8 sample reports). Not committed, not copied into `webgis/` (coordinator's call).

## Done
- `webgis/` copied to `eval_results/pci/webgis_dev/` (production `webgis/` untouched).
- Harness `C:\Users\PRO-LA~1\AppData\Local\Temp\claude\69f5d1fa5ca9c74f\8a7b1584-25d6-4c77-a4ce-be4c2864d62e\scratchpad\pci_front\harness.py`
  (port 8010, DATABASE_URL → geoai_test, GIS_APP_DIR → webgis_dev, API_KEYS = harness-only key `harness-operator-key`).
  Uses `app.pci_routes` when importable (now: mode "real"); `PCI_MOCK=1` forces the old DESIGN §7 mock.
  `/__harness` = mode; `/__t/overlap.js`, `/__t/helpers.js` = test scripts (scratchpad only).
- webgis_dev changes: `pci.js` (new), `pci.css` (new); hooks in `app.js` (`window.GeoAIApp`; map onSelect ignored in
  report mode; `#pciPanel` as an evidence-card obstacle), `map.js` (`#pciPanel` in availableViewport), `index.html` (2 includes).
- Features: "Condition report" action (icon-only ≤ 820 px) → docked "Report a stretch" card (desktop: below the GEOAI
  bar, toolbar shifted right of it; phone: bottom sheet, stats rail hidden, auto-folds while picking). Tabs New / Saved.
  Pick A, B, ≤ 8 via (add/remove), draggable markers, snapped points, path (surveyed solid, gaps dashed with
  "≥ x m (straight line)" tooltip), lengths/gaps/passes; API refusals shown in the card (role=alert).
  Form: road name/number, category (urban 5.3 default / highway 5.1 / rural 5.2), lane width (sent only when changed,
  so the report keeps "default 3.5 m (not measured)"), surface, optional km-stone calibration + direction, prepared by;
  free text marked "entered by requester, unverified".
  Preview: per-parameter table (assessed: band + E_lo–E_hi; not assessed: "not assessed" + evidence, never 0),
  pothole runs/km, bracketed final class + rating range, length, coverage, gaps, expert-reviewed count, screening note;
  100 m strip chart (cracking lo/hi, pothole runs, class/band cells, clickable); "Colour by" final class / cracking /
  potholes; sections list (click → map highlight + zoom, chosen pass / also surveyed / other direction, candidate
  treatment in a details, frames grid with ★ for PDF-annex frames, frame viewer with labels, time, position,
  confidence, review state, look-ahead note, full-resolution link); header/accuracy/limitations details.
  Map: sections coloured by band (7 classes incl. brackets), ticks every 100 m (500 m > 3 km), km labels.
  Downloads PDF/CSV (POST, same body as the preview) and JSON (the preview document); saved: stored PDF/CSV/JSON.
  Saved tab: list (Sample tag + rule text), open → map + summary + stale tag/note; save needs operator key
  (password field, sessionStorage only; wrong key → "The operator key was not accepted"; duplicate input → "already saved as report N").
  Esc: cancels picking, then closes; focus returns to the action.

## Tests (in-app browser, harness 8010, real API on geoai_test) — all PASS
| # | Test | Result |
|---|---|---|
| 1 | Pick A→B on the longest drive (map clicks) → `POST /gis/pci/path` | 1.63 km / 2.73 km paths, 1 pass, snapped 0 m, prompts A→B→done |
| 2 | Via point add → 3 points, path recomputed; remove → back to A,B | pass |
| 3 | Refusal: A 165 m / 638 m off the road | API message shown in the card (role=alert), no alert() |
| 4 | Gap path: end of drive i → start of time-adjacent drive i+1 (36 such pairs found) | 404 m, 1 gap "≥ 11.9 m", dashed line + tooltip, 2 passes |
| 5 | Generate (`POST /gis/pci/report`) with name/number/surface/prepared-by + calibration Km 12.350 decreasing | header "Km 12+350 → Km 11+946 · calibrated by requester at A (unverified)", free text "entered by requester, unverified", lane note "default 3.5 m (not measured)" |
| 6 | Kannada road name | backend refusal shown in the card ("…write the name in English letters") |
| 7 | Headline: Cracking/Potholes band + E range; Ravelling/Settlement/Rut depth "not assessed" + evidence; final "Poor–Fair, rating 0.9–2.0" | pass (never 0 %) |
| 8 | Section click → map highlight + fit, frames grid (20 segs/section via nearest-section assignment), frame viewer | pass; section inside the free map area at all sizes |
| 9 | Downloads preview: PDF 512 kB (4.6 s), CSV 2 kB, JSON 125 kB, names `IRC82-condition_<road>_<len>m_<date>_preview.*` | pass |
| 10 | Save with wrong key → "The operator key was not accepted"; harness key → "Saved as report 12"; key only in sessionStorage (`geoaiPciOperatorKey`), not localStorage | pass |
| 11 | Saved tab: 8 samples ("Sample" + rule text) + saved report; open sample → map + summary + rule note; stored PDF 511 kB / CSV / JSON (`…_no11.*`) | pass |
| 12 | Esc: cancels picking, then closes the card, focus → "Condition report"; tabs ArrowLeft/Right; 43 focusables all named | pass |
| 13 | Existing features with report closed: filters, legend, Survey, drive menu, replay (strip + evidence), per-km report, theme | pass; opening the report closes replay/evidence; survey clicks don't select in report mode |
| 14 | Overlap detector (`scratchpad\pci_front\overlap.js`: floating cards × panel, text leaves inside the panel clipped to the scroll box, horizontal overflow, SVG labels, page h-scroll) + section-visible check, states build / picking / minimised / result / saved / saved-open × light/dark at 1440×900, 1024×768, 820×1180, 375×812 | **48/48 OK** after fixes (fixed: toolbar/notice/zoom collisions, phone minimised sheet jumping up, strip-chart label overlap, params table h-overflow, section out of view) |
| 15 | Console | no JS errors from the new code; only expected 422/401 network entries and MapLibre style warnings (pre-existing) |
Screenshots (Claude tool-results): `C:\Users\PRO-LAB-3\.claude\projects\C--Users-PRO-LAB-3-Documents-Capstone\8a7b1584-25d6-4c77-a4ce-be4c2864d62e\tool-results\mcp-Claude_Browser-blob-1793997398011-3k0w4f.jpg` (1440 result),
`…-1793997212625-4t2972.jpg` (1440 section + frames), `…-1793997451084-eksf81.jpg` (sample open), `…-1793997610856-ltn046.jpg` (820), `…-1793997664686-lmxbnp.jpg` (375), `…-1793996717089-6rz6d7.jpg` (375 frames, mock).

## Round 2 — coordinator review + critic (REVIEW_backend §B1/B4), 2026-11-07 — DONE
Fixes:
1. Notices never cover the card: while the report is open, `#notice` and `#basemapNotice` are placed by pci.js
   (desktop: right of the card below the toolbar, left of the dashboard card, stacked; phone: stacked under the
   dashboard heading, the sheet starts below them). Inline styles are removed on close. Also fixed two overlaps
   of the EXISTING app with the report closed (found by the extended detector): desktop `#notice` over the drive
   control + map tools → placed below them; phone `#notice`/`#basemapNotice` on top of each other and the zoom
   buttons → stacked above the bottom controls.
2. Parameter table: fixed layout (38/30/32 %), range wraps — no horizontal overflow at any width.
3. Title: the "Condition report" action is icon-only (aria-label + tooltip); with a text label the title went to
   3 lines at 1024–1440 (measured: icon 1 line, "Report A→B" 2, "Condition report" 3). Title now 1 line desktop, 2 on phones.
4. Minimise: labelled button "▾ Hide" / "▴ Show panel" (dark when hidden) + a one-line summary; folding re-fits the
   whole A→B path (or the selected section) into the free map area (verified at 375: path fully visible).
5. Critic: when the pothole band contains Poor, a "PRIORITY" box quotes IRC §5.3 ("should be rectified straightway")
   above the KPIs; the final class is muted with the Appendix-4 note (potholes weigh 0.50…); each assessed parameter
   shows "labels unvalidated · <Attain evidence> · <n> Bengaluru expert labels". Default colouring = worst assessed
   parameter band ("Colour by" adds Final class / Cracking / Potholes); a Fair–Good stretch with Poor potholes is
   drawn Poor. `summary.priority_notes` (strings or {text}) are shown when present; the IRC §5.3 line is added only
   if none mentions potholes.
6. Frames per section: uses `sections[].segment_ids`, else top-level `attribution` rows
   ([segment_id, measure_m, section, km, pass, used] or objects; used=true, ordered by measure), else the
   nearest-section fallback. STATUS_backend's "API as built" does not list `attribution` / `priority_notes` yet →
   re-test once it does.
Detector extended: all fixed children of <body>, dashboard heading/stats/filters, #notice (forced visible in every
check), #basemapNotice, map status, legend, map tools, zoom, scale, drive control, toolbar, evidence card, strip;
plus "title ≤ 2 lines", "selected section inside the free map area", "whole path visible when minimised".
Re-run (real API): states closed / build / picking / minimised / result / saved / saved-open × light/dark at
1440×900, 1024×768, 820×1180, 375×812 → **56/56 OK**. Functional re-run at 1440: refusal message, gap path
(1 gap ≥ 11.9 m, 2 passes), generate (priority + final note shown, sections Poor), PDF/CSV/JSON downloads,
wrong operator key → clear error; no JS errors.
Screenshots: `…tool-results\mcp-Claude_Browser-blob-1793998646989-ncu5w8.jpg` (1440, priority, notices beside the card),
`…tool-results\mcp-Claude_Browser-blob-1793998544715-70zn6n.jpg` (375 minimised, path fully visible).

## Round 3 — real backend fields (2026-11-07)
- Harness restarted (it held the old backend modules); mode real, geoai_test.
- pci.js now reads `attribution` through `attribution_columns` (used rows per section, ordered by measure) — verified:
  gap stretch section 1 = 19 used attribution rows = 19 frames shown; `summary.priority_notes` shown in the PRIORITY
  box (UI's own §5.3 line only as fallback); `summary.final.note` shown under the KPIs (UI's Appendix-4 sentence only
  when the backend sends neither); `rating.params[].band_note` shown as a note row under its parameter.
- "Survey data temporarily unavailable…" in the round-2 1440 screenshot was NOT a bug: the test helper forces
  `#notice` visible in every check to test notice placement (`T.withNotice`). A clean reload shows no notice
  ("45 drives · data loaded"), and the harness log has 0 errors/503s.
- Final run after the backend re-copied geoai_test and regenerated samples (ids 18–25; harness restarted) — all PASS, no JS errors:
  - Sample "two_gaps" (no. 25): 2 gaps, 8 sections, 130 attribution rows, PRIORITY box, not stale; section 1 frames
    = 16 used attribution rows; stored PDF 512 kB / CSV / JSON downloaded as `IRC82-condition_stretch_796m_2026-11-07_no25.*`.
  - Fresh A→B across a time-adjacent drive gap: 254.5 m, gap 143.6–155.5 m (inside section 100–200, "≥ 12 m (straight line)"),
    2 passes; via add/remove; PRIORITY (backend priority_notes) + final.note; gap section shows 19 frames = 19 used
    attribution rows; preview PDF/CSV/JSON; wrong key → "not accepted"; harness key → saved as report 28;
    key only in sessionStorage. Refusal 165 m off-road shown in the card. Esc closes, focus back on the action.
  - No report in this data carries a `band_note`, so that row was only checked in code, not on screen.
  - Overlap detector (all floating elements, #notice forced visible, title ≤ 2 lines, section in view, whole path
    visible when minimised), states closed/build/picking/minimised/result/saved/saved-open × light/dark at
    1440×900, 1024×768, 820×1180, 375×812 → **56/56 OK**.

## Round 4 — critic REVIEW_frontend.md (F1–F11 frontend items), 2026-11-07 — DONE
Changes in `webgis_dev/pci.js` + `pci.css` (backup of the round-3 file: scratchpad `pci_front\pci.before_r3.js`):
- F1 Generate errors: shown in the footer next to the button (role=alert) AND in the form (scrolled into view);
  focus stays on Generate; API messages capitalised, "Km 0.050" → "Km 0+050". Test: negative road chainage →
  "Not generated: Road chainage would be negative at B (Km 0+050 at A, …)", in view, focus = generate.
- F2/F11.1 colour: default "Potholes band (as the PDF)" by the OBSERVED band (E_lo / band_lo); the worse band that
  skipped frames could add is a dashed edge on the map, a thin bar in the strip, "Good / Poor?" on the chip, and the
  backend `band_note` in every section row. Options: Potholes / Cracking / Worst assessed (observed) / Overall
  (App. 4, secondary). Legend = every colour drawn + the dashed-edge meaning. Clean road is no longer purple.
- F3/F9: while the card is open the dashboard stats + filter chips are hidden (note in the card: the report uses
  every drive, map filters not applied), survey points/clusters at 22 % opacity in greyscale, tracks greyscale,
  selection/coverage hidden; restored on close. Phones: dashboard heading hidden while open; "▴ Taller" toggle
  (sheet to ≤ 88 % height, below the toolbar). Free map: 1024×768 490×494 px (was 490×195); 820×1180 724×431;
  375×812 279×257 with the sheet body 240 px (was 180), 324 px when taller.
- F5: pothole row shows the nominal-area band ("Fair (0.028 % of lane area, sensitivity only)") and the backend
  `resolution_note` (params → headline → summary; UI sentence only as fallback).
- F6: PRIORITY box lists the pothole runs as chips (`summary.pothole_runs_list`; fallback: consecutive used pothole
  frames from `attribution`), "Rests on N runs, M frames", ≤ 2 runs → "Possible pothole — check the frame(s)"
  (skipped if the backend note already says so). A chip opens the section and focuses that frame.
- F7: focus is remembered before every re-render and restored (section, frame, via, pick, colour-by, details);
  opening a frame focuses the `<figure>` (tabindex −1); image alt carries labels + chainage.
- F8: saved list leads with `headline_text` (fallback "Potholes Poor · 3.1 runs/km · Cracking Fair"), overall last;
  minimised summary leads with the bands.
- F10: light band tokens darkened — white-on-chip contrast good 6.48, fair–good 5.65, fair 5.76, poor–fair 6.04,
  poor 6.01, poor–good 5.95, insufficient 4.99 (dark theme 5.75–9.98); panel text ≥ 11 px (strip 10.5–11);
  targets ≥ 32 px, 44 px on phones (tabs, buttons, chips, summaries, thumbnails).
- F11: "≥" on path/stretch length and Km B when gaps; "Surveyed length"; snapped distance 1 decimal; road chainage
  without the "(12.350)" duplicate; chainage on thumbnails + frame caption (+ road Km when calibrated); treatment
  clause "(IRC:82 …)"; after Save, downloads use the stored report (`…_no<N>.*`); operator key only stored after a
  successful save and cleared on 401; sample rule shows requested vs routed length; strip pothole axis has 0.
- Bug found and fixed while testing: JSON download after save failed ("Cannot access 'id' before initialization").
Tests (harness restarted with the backend's round-3 code; real fields `resolution_note`, `pothole_runs_list`,
`headline_text`, regenerated samples):
- Functional 1440: fresh A→B across a gap (254.5 m, gap 143.6–155.5 m inside section 100–200, "≥ 254.5 m");
  8 run chips = 8 `pothole_runs_list` entries, chip → frame "Potholes · Km 0+045" focused; gap section 19 frames;
  map lines all observed band; preview PDF/CSV/JSON; save → "already saved as report 28" (dedupe), then stored
  PDF/CSV/JSON `…_no28.*`; sample no. 37 opens with 8 run chips; saved list leads with headline_text; no JS errors.
- Overlap: own detector + the critic's `critic3\detect.js` (structural containers #workspace/#initialLoader/
  skip-link/dashboard-overview filtered) in states closed/build/picking/minimised/result/(tall on ≤ 820)/saved/
  saved-open × light/dark at 1440×900, 1024×768, 820×1180, 375×812 → **60/60 OK** (incl. critic small-font,
  small-target, h-overflow, svg-text checks).

## Next
- Coordinator: review, copy `pci.js`, `pci.css`, and the small `app.js`/`map.js`/`index.html` diffs into `webgis/`.
- Harness still runs on 8010 (background); stop it with TaskStop or by killing the python on port 8010.

## Open problems / notes for the coordinator
- Browser caches pci.js/pci.css: when copying into `webgis/`, users may need a hard reload (index.html has no version query).
- A report with 24.7 pothole runs/km rated "Fair–Good" (stretch 404 m, Kanakapura test): backend/critic should check the pothole band (Table 5.3 uses % area; runs are only reported).
- Kannada road names are refused by the backend PDF renderer (clear message shown in the card).
- Report sections carry geometry but no segment list: the UI assigns `report.segment_ids` to the nearest section
  (loads the passes' segments via `/gis/video/segments.geojson?session=…&geom=point`). A per-section
  `segment_ids` from the backend would be used automatically if added.
- The "Colour by" choice is UI-only (final bracketed class default). Sample titles from the backend are generic
  ("Km 0+000 to Km 0+796 (urban)"); the UI adds the rule text.
- On a 375×812 phone the free map above the sheet is ~200 px tall; the sheet can be minimised (– button) and
  folds automatically while picking.
- Screenshot capture in the in-app browser timed out intermittently (hidden pane); layout was verified by the JS detector.
