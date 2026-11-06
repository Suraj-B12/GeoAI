# RESUME HERE — Chainage + IRC:82 condition rating (A→B report in the WebGIS), started 2026-11-07

User request (2026-11-07): research chainage and PCI per IRC; in the GIS platform pick point A and B and get
chainage + condition rating for the stretch; downloadable report with everything a road engineer needs, no
overlaps; sample reports present in the portal; a critic agent that finds flaws until reasonable; multiple
agents; tests; paper + metrics updated; honest, no made-up results.

| Workstream | Owner | Files | Status file |
|---|---|---|---|
| R research | research agent (done) | `eval_results/pci/RESEARCH.md` | done |
| D design | coordinator | `eval_results/pci/DESIGN.md` | critic review → `eval_results/pci/REVIEW_design.md` |
| B backend | backend agent | `app/pci_*.py`, `migrations/009_pci_reports.sql`, `scripts/pci_*.py`, `scripts/tests/test_pci_*.py`, one line in `app/gis_routes.install` | `eval_results/pci/STATUS_backend.md` |
| F frontend | frontend agent | scratch copy of `webgis/` (copied into `webgis/` only on the coordinator's word: it is served live) | `eval_results/pci/STATUS_frontend.md` |
| C critic | critic agent | `eval_results/pci/REVIEW_*.md` (read-only otherwise) | its review files |
| P paper/deploy | coordinator | paper §9, CLAUDE.md, memory, commit, production restart | this file |

Steps left: critic on design → fix DESIGN → backend + frontend in parallel → critic on code + sample
reports → fix rounds → sample reports in production DB → paper §9 → restart production → commit/push.
Delete this file when done.
