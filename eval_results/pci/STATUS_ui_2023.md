# STATUS — WebGIS condition-report panel → IRC:82-2023 PCI (UI workstream)

Owner files: `webgis/pci.js`, `webgis/pci.css`, `webgis/index.html` (version string only), `pci_meta()` in
`app/pci_routes.py`, this file. Scratch: `%TEMP%\claude\…\scratchpad\pci_ui23\` (dev copy `webgis_dev\`, harness
`harness.py` on 127.0.0.1:8012 → geoai_test, backups `pci.before.js/.css`).

Rule: all edits go to `scratchpad\pci_ui23\webgis_dev` first; the live `webgis/` is replaced only after the
harness tests pass (live page is served from disk).

## Steps
| # | Step | State |
|---|---|---|
| 0 | Read 2023 backend + schema, start harness (8012, geoai_test, real routes) | done 2026-11-08 |
| 1 | Form: road class incl. MDR, surface selector, optional measured values, chainage wording | coded in webgis_dev |
| 2 | Result card (headline PCI, 6-row table, not-assessed line), map colours + legend, saved list | coded in webgis_dev |
| 3 | Remove 2015 wording/bands from the panel | coded (old saved 2015 reports: notice + 'Generate again') |
| 4 | Tests: headless Edge via CDP (`scratchpad\pci_ui23\cdp.mjs` + `steps.mjs`) on harness 8012: desktop 1440×900 light/dark, phone 375×812; urban/rural/MDR; invalid input; old saved report → regenerate; overlap detector 0 issues, no JS errors | done 2026-11-08 |
| 5 | Copy to live webgis/, bump ?v= | done 2026-11-08 (pci.js/css ?v=20261108); live smoke test passed (single PCI 61 Satisfactory with entered values). Harness stopped. |

## Backend findings (for the coordinator, not fixed here)
- `category=mdr` → HTTP 500: `KeyError: 'mdr'` in `pci_core.class_treatment` (`_RENEWAL[category]`).
- `category=rural` + `surface=BC` + entered IRI → HTTP 500 (`pci_param` raises ValueError "IRI equation for BC
  not given for rural roads"; `validate_request` accepts the pair). UI offers SD/OGPC/MSS/SDBC only for rural.
- Saved reports in geoai_test (ids 3–10) are IRC:82-2015 (no `summary.pci`); `headline_text()` still 2015.

- `pci_meta()` now also returns `irc2023` (road classes with surfaces/default, surface names, params with unit/weight/
  entered limits, classes with recommendations); the UI falls back to the same constants when it is absent
  (live server needs a restart to serve it).

## Still open (backend)
- Live: `category=mdr` still HTTP 500 (KeyError 'mdr' in `class_treatment`); the UI shows "The server could not produce this (HTTP 500)".
- Live `/gis/pci/meta` has no `irc2023` yet (code is in `pci_routes.py`; UI uses identical fallback constants).
- Saved samples 17–24 are 2015 reports: list shows "Saved before the PCI update — no PCI"; opening one offers "Generate again with the PCI". Regenerate samples to fix.
- `summary.priority_notes`, `limitations`, `treatment*` text still cite IRC:82-2015; the panel no longer shows priority/treatment, but shows limitations in the header details.
- Entered potholes are applied unchanged to every unit (100 m section, km row, stretch) — semantics "number per unit" need a decision.
- Visual review screenshots: `scratchpad\pci_ui23\shots_{desk,desk2,dark,phone,mdr,live}\*.png`.

## Round 2 — PCI estimate with a 90 % interval (2026-11-09) — DONE, deployed (?v=20261109)
- Headline: "PCI ≈ 52 · 90 % interval 40–83 (Poor to Good) · Fair", scale bar (interval box + median tick),
  Table 5.5 recommendation for the median class (+ "confirm on site" when the interval spans classes), one line
  naming the widest input (mean `width_by_param` over the sections), strict range as a small secondary line.
- Table "Typical 100 m section": estimate with 90 % interval (length-weighted mean of the sections' inputs;
  potholes per 100 m), PCI_i at the estimate (from `meta.irc2023.curves`, sampled from `pci_param`; "—" until
  the server serves them), weight under the name, short basis.
- Map/strip/list coloured by the median-estimate class; sections without an estimate grey dashed; colour-by
  selector hidden in estimate mode. Pothole flags carry "jolt felt" / "no jolt" badges + one-line explanation.
  Saved list "PCI ≈ 52 (40–83) · Fair" (strict range / "no PCI" for older reports).
- Fallback: no `summary.pci_estimate` → previous strict-range UI (tested on harness and live server).
- `pci_meta()` adds `irc2023.curves` (~54 KB, cached). Needs the server restart.
- Tests: harness fixture mode (`/__mode?fixture=1` answers from `C:\GeoAI\pci\preview_2023b\r34.json`, since
  geoai_test has no probe scores, so no estimates); desktop light/dark, 375 px, fixture on/off; overlap 0, no JS errors.
- Backend wording still about a car: header.method "forward-looking, windshield mount" and header.lane_imaged
  "the lane the vehicle drove in" (app/pci_chainage.py) — the survey vehicle is a scooter (handlebar mount).
