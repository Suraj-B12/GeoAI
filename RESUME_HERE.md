# RESUME HERE — IRC:82-2023 PCI build-out (started 2026-11-08)

User request (2026-11-08): one final condition score to compare with an actual road survey; move the A->B
report from IRC:82-2015 to **IRC:82-2023 (PCI 0-100)**; regenerate the sample reports, neat, only what a road
engineer needs; run agents in parallel and finish.

Decided: survey format unknown -> build to IRC:82-2023, accept any per-section score; calibrate on part of the
survey, report agreement only on the blind rest (pre-registered: eval_results/pci/survey_split_rule.json).

## Done (coordinator)
- `app/pci_irc2023.py` equations + `eval_results/pci/IRC82_2023.md` (transcription, worked-example check).
- `research.video_probe_scores` (migration 010) filled for all 7,050 segments (`scripts/video_probe_scores.py`).
- `app/pci_core.py` 2023 section (`pci_unit`, calibration loader), `app/pci_chainage.py` (unit `pci`, summary
  `pci`, request `surface`/`entered`, probe scores on segments), `app/pci_routes.py` validate_request.
- `scripts/pci_survey_compare.py` check/fit/evaluate — wiring verified on a SYNTHETIC survey
  (C:\GeoAI\pci\synthetic_survey_wiring_test.csv, outputs in a scratch dir only).

## Workstreams in flight (each has its own files + STATUS file)
| Workstream | Files it owns | STATUS |
|---|---|---|
| IRI from phone IMU | app/pci_iri.py, scripts/iri_from_imu.py, migrations/011_segment_roughness.sql, scripts/tests/test_pci_iri.py | eval_results/pci/iri/STATUS.md |
| PDF + CSV (2023, neat) | app/pci_report_pdf.py, app/pci_treatments_2023.py, CSV fn in app/pci_routes.py, test_pci_pdf.py | eval_results/pci/STATUS_pdf_2023.md |
| WebGIS panel | webgis/pci.js, pci.css, index.html version, meta fn in app/pci_routes.py | eval_results/pci/STATUS_ui_2023.md |
| Tests + critic | test_pci_irc2023.py, test_pci_survey.py, new tests in test_pci_routes.py, fixes in pci_irc2023 / pci_core 2023 / survey script | eval_results/pci/STATUS_review_2023.md, REVIEW_2023.md |

## Steps left (coordinator)
1. DONE: IRI agent finished (no IRI until survey calibration; idle-noise check FAILED, repeatability not
   testable — eval_results/pci/iri/VALIDATION.md). Migration 011 registered; pci_core.roughness_value +
   pci_chainage._roughness_rows hook it in (bins of the calibration unit_m).
2. TODO after the critic agent finishes with scripts/pci_survey_compare.py: add the IRI fit to `fit`
   (measure(): raw rows via PC._roughness_rows(); per calibration section pci_iri.aggregate() -> rms_av, speed;
   iri_ref = survey iri_m_km; pci_iri.fit_calibration(unit_m = median section length); write
   eval_results/pci/iri/iri_calibration.json, or OUT/iri_calibration.json + PCI_IRI_CALIBRATION with --out-dir).
3. Full test run; restart production (see CLAUDE.md "Restarting production"); regenerate samples
   (`scripts/pci_samples.py --database geoai --regenerate`, `scripts/pci_validation.py --database geoai --redo samples`).
4. Paper §9 -> 2023, CLAUDE.md, memory; commit + push; delete this file.
