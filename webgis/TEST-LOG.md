# Verification — 30 October 2026

This log covers the separate **drive-video-only map workspace**, including default Potholes, Video/Survey navigation, filter-aware metrics, no citizen/walking layer, anchored cards and sliding navigation. The latest layout makes the map fill the viewport and moves controls into compact floating groups. The supplied sample remains unchanged.

## Automated checks

- Final JavaScript data, UI, map and basemap suites after the full-map revision: **69/69 passed**, including **37 UI tests**.
- The suite includes **1,440 filter combinations** compared with the supplied Python API's filtering implementation.
- Live HTTP API checks against the supplied mock: **19/19 passed**.
- Sample consistency checks: **58/58 passed**, including SHA-256 checks for all **1,664 manifest-listed files**.
- Latest regression checks cover the absence of citizen requests/controls, metadata-selected Potholes on entry/reset, Video/Survey navigation, current-filter summary metrics, route inspection versus session filtering, exports, replay, request races and recovery.
- Motion-controller regressions cover anchored-card lifecycle, mutual exclusion, keyboard dismissal and focus restoration, delayed closing, reduced-motion immediate closing, and evidence entrance/exit boundaries.
- Additional cases cover final expert fields, null images/positions, pending/skipped/not-covered semantics, metadata styling, paging/caps, image concurrency, keyboard synchronization and mutually exclusive per-km coverage totals.

Saved outputs accompany the package in `verification/`. These results establish the tested local behavior, not production latency or the absence of every possible defect.

## Handoff acceptance and scope

| Item | Status | Evidence and limits |
|---|---|---|
| Static `/gis/app/`, same-origin API | Passed against mock | Empty `apiBase`; bundled static assets. Real FastAPI host remains a deployment check. |
| Drive list and route context | Passed | Three drives with API road/date/state/length/distress. Inspecting or replaying one retains all route tracks. |
| Default Potholes selection | Passed | 131 video segments, 652.3 m matching length, 111 awaiting review. Distress is 100% of this filtered selection. Total route context remains 4.06 km. |
| Full survey reference totals | Passed | With label restriction removed: 813 segments, 4,060.31 m surveyed, 20.08% distressed, 389 awaiting review. |
| Reference filters | Passed | Potholes 131; Medium+High distress 75; high-confidence distress 47; needs-review 389. Non-pothole reference cases remove the default label restriction. |
| Citizen/walking layer | Intentionally excluded | No runtime citizen request, layer toggle, pending-review control, citizen metric or citizen export. This follows the latest requested scope. |
| Anchored controls and navigation | Passed | Survey opens below the surveyed-roads total on the left; About and Filters open below their triggers; Settings, Export, Drive and Legend use the same card lifecycle. The detailed report remains modal. Sliding active navigation, keyboard dismissal/focus return, mutual exclusion and reduced-motion closing are covered. |
| Evidence card | Passed functional checks | Resized image, final labels, original correction, confidence/review state, local time and GNSS uncertainty. Sub-second LAN timing is not measured here. |
| Missing/unknown/corrected states | Passed fixtures | Isolated synthetic data; original sample unchanged. Missing frames never imply Normal. |
| Virtualized strip | Passed | Bounded visible nodes, chronological segment order, thumbnail/map synchronization and keyboard stepping. Replay follows matching frames when filters are active. |
| Playback and images | Passed | 1–5 segments/s, synchronized vehicle/selection, resized media only, maximum six concurrent image requests. |
| Per-km report | Passed | Longest drive has bins of 1,000 + 1,000 + 637.5 m. Coverage counts metres once; overlapping labels remain separate. Complete-drive scope is stated. |
| Video CSV | Passed | Current filters, UTF-8 BOM and § characters preserved. Export disabled while loading/stale. |
| Ward overlay | Passed functional checks | Historical 198-ward file, names and loaded-video counts/distressed length. Historical status and centre-based assignment documented. |
| API failure/recovery | Passed fixtures | Clear notice, disabled stale actions, 30-second retry and stale-response guards. |
| Credentials/local paths | Passed authored-assets scan | No browser secrets or absolute workstation paths. Original media only via explicit link. |
| Current full-map visual confirmation | Passed at reviewed sizes | Desktops 1440×900 and 900×850; phones 375×812, 375×667 and 375×600. Reviewed default map, Survey, replay, expanded evidence and dark presentation. The map fills the viewport, navigation remains reachable with Legend open, and browser hit-testing confirms Fit and zoom remain reachable during replay. Phone replay hides metrics to preserve route space. Earlier layout measurements are recorded separately below. |

## Current full-map design review

The map now extends to every viewport edge. Only **Video** and **Survey** remain in primary navigation; Filters and Export are separate actions. Desktop navigation sits at the upper left, with compact title/actions and two-by-two metric cards at the upper right. Drive inspection and Settings float independently. Main GEOAI branding, typography, map styling and anchored-card motion are retained.

Current screenshots were reviewed independently at 1440×900 and 900×850 desktop sizes, and 375×812, 375×667 and 375×600 phone sizes. They cover the map, Survey card, replay, expanded evidence and dark presentation. The map bounds were measured as the complete viewport. No clipped actions or horizontal overflow were found in these reviewed states. On phones, the narrow metric card keeps the main values and labels readable; longer supporting notes use ellipses. Replay hides metrics and filter chips, leaving a clear route area above the compact evidence card and player. Closing replay restores the metrics. A legacy bottom offset was removed so filter chips keep their natural compact height.

Replay-specific checks found that the player could cover map navigation controls. Phone replay now places Fit and zoom above the compact evidence card. Below 640 px height, the title/actions card temporarily hides and map controls move beneath the primary navigation. The 375×667 and 375×600 captures show the vehicle, evidence, player, Fit and zoom together, with successful browser hit-tests on the map controls. At 900×850 the narrower player leaves the right-side controls clear. Expanded desktop evidence is positioned below the selected vehicle without covering it, and its remaining metadata scrolls within the card. Expanded phone evidence shows labels, confidence and metadata without dashboard interference.

Evidence, player and legend overlays live outside the map container, so opening them does not raise the whole map over the navigation. The legend-open browser check confirmed that navigation remained reachable. Static screenshots establish final geometry; controller tests and browser interactions establish opening, closing, focus and playback behavior.

Motion durations match the reference interface: navigation 400 ms with `cubic-bezier(.22,1,.36,1)`, Survey/Filters/About 300 ms with an 8 px vertical offset, utility cards 250 ms, and evidence entrance/desktop width 550 ms. Reduced motion removes transitions and close delays. Playback frame updates do not repeat the evidence entrance. Floating cards retain near-opaque surfaces to avoid WebGL backdrop artifacts.

## Earlier verification history

Previous revisions were reviewed at desktop 855×853 and 1440×900, and phones 375×812 and 375×667. Those checks established the drive-only scope, readable two-row metrics, expanded evidence content, internal card scrolling, the frame-strip behavior and anchored controls. A native-browser check also caught and fixed keyboard focus returning before the modal report had closed. These results remain useful regression history; their older inset-map screenshots and exact spacing measurements are not evidence for the current full-map geometry.

## Re-run

Place the original `gis_sample/` beside `webgis/`, then start the mock as described in README. Optional test dependencies are separate from the application:

```sh
npm install
npm test
python3 verification/api-integration.py http://127.0.0.1:8765
python3 verification/sample-truth.py
```

Copy only `webgis/` to the production server. Test code, package metadata and verification logs are optional handoff evidence.

## Earlier repository integration review

The repository handoff and original sample ZIP match the sources used for development. A fresh install from the pinned dependency lock and `npm test` passed all 66 tests, including 36 UI tests. The added cases cover portable Python selection and all advanced detector scores, preserving zero values while showing missing values as unavailable. An explicit Python override also passed the five sample-contract tests. Windows launcher selection is tested with controlled fixtures; native Windows execution is not claimed. All 19 HTTP checks and 58 sample checks were rerun from the repository copy.

## Receiving-environment checks

Run the receiving team's API suite and smoke tests against its real FastAPI/VLM host. Confirm data, internet access, large-volume rendering and LAN preview latency. Deployment and acceptance on the real VLM server remain separate receiving-environment checks.

Latest Survey placement check: the card is left-aligned beneath the road total with a 12 px gap, including at 375×667. The numeric distance uses the same Rostex Outline family as GEO; units and the secondary label remain in Urbanist. No horizontal overflow was observed. All 69 automated tests passed after this update.

## Typography and spacing update

The right-hand metrics now use a content-sized container and wrapped notes. At 1440×900 each metric cell measures about 121 px high with 14 px labels, 36 px values and 12 px supporting text; the outlined distance is 32 px. At 900×850 notes wrap without clipping, and Export remains clickable above other overlays.

Phone checks at 375×667 cover the larger Survey and Filter cards without horizontal overflow. The 375×600 replay keeps its vehicle, evidence, frame count, speed selector and playback buttons visible. Evidence spacing follows the measured player height, and the map adjusts the selection after the drive menu finishes closing. Expanded evidence remains inside the viewport and loads a 1280 px preview. The browser recorded no console errors. All 69 automated tests pass. Earlier screenshots and measurements are retained as layout history.
