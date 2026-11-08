# RESUME HERE — tighten the IRC:82-2023 PCI range (started 2026-11-09)

User (2026-11-09): the PCI range (~79 points) is useless; consider all distress sub-labels; tighten it, make it
robust and sure, use everything; finish the project.

Approach: replace "unknown = PCI_i 0..100" with calibrated per-parameter estimates + honest 90 % intervals, and
combine them by Monte Carlo into a PCI estimate with a 90 % interval (`app/pci_irc2023.pci_estimate`, done). Keep the
strict range as the worst case. Survey calibration (scripts/pci_survey_compare.py) still overrides later.

## Workstreams
| Workstream | Owner files | STATUS |
|---|---|---|
| Attain-calibrated extents (cracking/ravelling/patching % area, pothole number via eq. 2) | app/pci_extent.py, scripts/pci_attain_extent_calibration.py, scripts/tests/test_pci_extent.py, eval_results/pci/extent/* | eval_results/pci/extent/STATUS.md |
| IRI interval + rut-depth prior (IRI agent resumed) | app/pci_iri.py (+ app/pci_rut.py), eval_results/pci/iri/*, eval_results/pci/rut/* | eval_results/pci/iri/STATUS.md |
| Coordinator: pci_estimate (done), wire into pci_core.pci_unit + report JSON (`pci.estimate`), then PDF/UI agents show estimate + interval + "other distresses" table, critic, regenerate samples 33-40, paper §9.9, commit | | this file |

## Progress 2026-11-09
- DONE + committed 59f43ca: extents (Attain), IRI interval (scooter), rut prior, pothole jolt confirmation,
  pci_estimate_unit / estimate_mean, ESTIMATE.md. Samples: estimate ~52 (40-83) vs strict 9-85; IRI = most width.
- RUNNING: PDF/CSV agent (estimate display, impact column, other-distress table -> preview_2023b), WebGIS agent,
  critic (REVIEW_estimate.md, test_pci_estimate.py); band-view probe GPU job (C:\GeoAI\pci\probe_run\band_full.log).
- THEN: restart production, regenerate samples (no bash `source .env`!), verify live, paper §9.9, CLAUDE.md, commit.
