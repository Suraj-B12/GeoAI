# STATUS — IRC:82-2023 tests + critic (workstream "Tests + critic")

Owner files: eval_results/pci/REVIEW_2023.md, scripts/tests/test_pci_irc2023.py, scripts/tests/test_pci_survey.py,
new test functions in scripts/tests/test_pci_routes.py; fixes only in app/pci_irc2023.py, the 2023 section of
app/pci_core.py, scripts/pci_survey_compare.py.

## Step 1 — read + grid check (done 2026-11-08)
- Re-read all 2023 equations from appendix2_pages/p076-p088 (crops upscaled): every coefficient in
  app/pci_irc2023.py matches the print. Rural roughness curve image (p085) is corrupt (does not render).
- Grid scan done (extent 0-100 step 0.01, PN 0-50 step 0.001, IRI 2-20 step 0.001). Findings -> REVIEW_2023.md.

## Next
- fixes, tests, review file (see below as they land).

## Step 2 — fixes + new tests (done 2026-11-08)
- Fixes applied in app/pci_irc2023.py, pci_core 2023 section, scripts/pci_survey_compare.py (list in REVIEW_2023.md).
- New: scripts/tests/test_pci_irc2023.py (13/13), scripts/tests/test_pci_survey.py (8/8),
  test_pci_routes.py: test_validation_irc2023_inputs (PASS), test_report_mdr_category (FAIL: KeyError 'mdr'
  in pci_core._RENEWAL — 2015 section, not mine to edit; reported to coordinator).

## Step 3 — full PCI suite run (done 2026-11-08)
- test_pci_irc2023 13/13, test_pci_survey 8/8, test_pci_core 8/8, test_pci_chainage 13/13, test_pci_iri 9/9,
  test_pci_pdf 12/12, test_pci_routes 8/9 (FAIL test_save_dedupe_cap_stored_and_stale: headline_text no longer
  starts with "PCI " — owned by the PDF/CSV/UI agents' in-flux change, not touched), validate_all 18/18.
- Survey/routes tests adapted to the coordinator's per-km entered potholes and the _RENEWAL['mdr'] fix.
- roughness_value() reviewed (REVIEW_2023.md section "roughness_value()").

## DONE — scripts/pci_survey_compare.py released to the coordinator (IRI fit can be added).

## Round 2 — PCI ESTIMATE critic (2026-11-09) — DONE
- eval_results/pci/REVIEW_estimate.md (E1-E12, L1-L5); fixes in pci_irc2023.pci_estimate and the pci_core
  estimate block (range semantics for the rut prior, rut calibration loaded, IRI coverage >= 50 %, calibrated
  values keep an interval, binomial pothole lower bound, input validation). Output field names unchanged.
- New scripts/tests/test_pci_estimate.py 11/11. Full PCI set: chainage 13, core 8, estimate 11, extent 8,
  irc2023 13, iri 13, pdf 13, survey 8, routes 9/9 — all pass.
