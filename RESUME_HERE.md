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

**Checkpoint rules (usage limits stop every agent at once):**
- Every agent rewrites its STATUS file after each finished step: done / in progress / next, file list, test
  results, open problems. A new session must be able to resume from the STATUS file alone.
- Agents write code in small, complete steps (each file importable, tests runnable). No half-edited file
  is left overnight.
- Long computations (sample reports, validation) run as scripts that write their results atomically
  (temp file + rename) and skip work already done.
- The coordinator commits and pushes after each milestone (design, backend, frontend, reviews, paper),
  so the work is in git and on GitHub, not only on disk.
- After a reset: read this file + every `eval_results/pci/STATUS_*.md` + `REVIEW_*.md`, then resume the
  same agents with SendMessage (or start new ones from the STATUS files).

Committed so far: `b0b0b2d` (research, design v1, this file), `b46fc92` (design v2 after critic review).
2026-11-07: backend agent and frontend agent started in parallel on DESIGN v2 (frontend works in
`eval_results/pci/webgis_dev/`, harness port 8010; backend tests on port 8012 / TestClient, writes only to
`geoai_test`). Next: critic agent on code + sample reports once both STATUS files say done.

Steps left: critic on design → fix DESIGN → backend + frontend in parallel → critic on code + sample
reports → fix rounds → sample reports in production DB → paper §9 → restart production → commit/push.
Delete this file when done.
