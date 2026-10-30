# GEOAI Video road surveys

A separate, read-only road-engineering map workspace for the `geoai.gis.v1` drive-video API. **This application contains drive surveys only.** It does not load citizen or walking observations, and it does not change the main GEOAI website.

The initial view selects **Potholes** from the server's metadata. The full video taxonomy remains available in Filters for the handoff's inspection and coverage workflows. Reset filters returns to the pothole view.

## Run with the sample

Keep the supplied `gis_sample/` folder beside `webgis/`, unchanged. From their parent folder:

```sh
python3 gis_sample/mock_server.py --app webgis
```

On Windows, use `python` instead of `python3` if needed. Open `http://127.0.0.1:8765/gis/app/`. Use the supplied mock: a plain static file server cannot provide filtering, summaries and image endpoints.

## Install on the existing GIS server

Copy the complete `webgis/` folder into the Capstone repository root. The existing FastAPI server described in the handoff mounts it at `/gis/app/`. Keep `config.js` unchanged:

```js
window.GIS_CONFIG = { apiBase: "" };
```

The browser calls the existing read-only `/gis/*` API. No additional backend, database connection, credentials or build step is required. Application assets use relative paths. For separately hosted development, `apiBase` can point to a reachable GIS API origin with suitable CORS settings. Deployment to the VLM server is separate from this repository.

## Map and controls

The map fills the entire viewport. GEOAI navigation floats at the upper left; the title, actions and current-filter metrics sit in compact cards at the upper right on desktop. Drive inspection and map settings remain separate floating controls. On phones, the metric card uses two columns with map space visible beside it. Replay temporarily hides the phone metrics and filter chips to keep the route visible, then restores them when closed.

- **Video** returns to the map, evidence and route player. **Survey** opens the analysis card on the left, directly beneath the surveyed-roads total. These are the only primary navigation items; the active highlight slides between them.
- **Filters** and **Export** are separate dashboard actions. They open cards beside their controls. The GEOAI logo opens About the data; the map gear opens layers and light/dark presentation. The legend opens upward from its map button.
- The floating metric card describes **current filters**: matching road length, distress within that selection, segments awaiting expert review and classified length. They do not claim to describe the whole route when a distress filter is active.
- Select the **surveyed routes** distance capsule to open **Inspect a drive**. Choosing a drive focuses it without hiding the other route tracks or silently changing the observation filters. Use the explicit session filter to restrict observations and statistics to one drive.
- A segment opens a compact photo card. **View details** shows final labels, model/expert provenance, confidence, local time, position uncertainty, frame quality and detector details. Expanding the card pauses playback and temporarily hides the strip to leave room for the evidence. Closing restores the player.
- **Replay drive** opens the virtualized strip of matching frames, ordered along the chosen drive. Select a thumbnail or use left/right keys within the strip; playback supports 1–5 segments per second. The vehicle, segment highlight, card and strip remain synchronized.
- **Per-km report** describes the inspected drive, or all drives if none is inspected. Its filtered summary and complete-drive kilometre statistics are explicitly distinguished: the session API does not accept the shared observation filters. Coverage bars count each metre once; overlapping distress labels appear separately.
- **Export** downloads video CSV with the current filters or opens the print-friendly survey report. There is no citizen export or citizen layer control.

## Motion and keyboard behavior

The navigation and cards follow the main GEOAI interface: a 400 ms sliding navigation highlight, 300 ms vertical entry/exit for Survey, Filters and About, and 250 ms utility-card motion. Evidence enters once over 550 ms and changes width over 550 ms on desktop; playback frame changes do not restart its entrance. The detailed per-km report remains a centered modal with a print-friendly layout.

Opening a control closes the other control cards. Escape or a close button dismisses a card and returns keyboard focus to its trigger. Clicking or moving focus outside a nonmodal card dismisses it without interrupting the new target. Controls retain visible keyboard focus. The system's reduced-motion preference removes transitions and close delays. Cards remain within the viewport and scroll internally when their content is taller than the available space. The metrics card grows with its labels and supporting notes, which wrap instead of being truncated. Typography is larger across navigation, filters, Survey, reports, evidence and replay; the outlined distance total is the main left-side highlight.

## Interpreting the sample

The default Potholes view contains **131 matching video segments**, **652.3 m** of matching road length and **111** results awaiting expert review. Its distress percentage is **100% of the current selection**, because this selection contains potholes. The separate route context remains **4.06 km across three drives**.

With the label filter removed, the supplied snapshot contains 813 video segments, 4,060.31 surveyed metres, 815.31 distressed metres (20.08%) and 389 results awaiting expert review. These are sample observations, not hard-coded application values.

Surveyed means covered send/audit/skip segments. Not covered means missing evidence and never Normal. Normal and Distress are model results, not ground truth. Expert corrections use the API's final fields and show the original model answer struck through. Confidence below the metadata threshold is marked for review. Position uncertainty is the supplied one-sigma value; times appear in IST. Ellipsoidal height is never labelled as mean sea level.

## Included features

- Metadata-driven final-label colours, severity sizes, taxonomy and confidence thresholds.
- Drive tracks, low-zoom video clusters, canvas footprints at street zoom, coverage lines, severity heatmap and historical wards with hover totals.
- Date, multi-label, severity, review, confidence, session, model-class and gate-status filters; authoritative API summaries and filtered CSV.
- Compact and expanded evidence cards, virtualized thumbnails, synchronized route playback and per-km reporting.
- Full-viewport desktop/phone map, floating metrics and controls, keyboard navigation, filter URLs, light/dark maps and a printable tabular report. Phone replay temporarily hides the metric card; closing the player restores it.
- Loading, empty, missing-image and error states, request cancellation, stale-response protection and 30-second recovery after a connection failure.

The primary basemap uses the main GIS's keyless OpenFreeMap/MapLibre approach, with OpenStreetMap fallback. Map imagery needs internet access; drive images come from the configured GIS API. Code, fonts, runtime libraries and historical ward boundaries are bundled locally. Floating controls use near-opaque surfaces to avoid WebGL backdrop-blur artifacts.

## Performance and resilience

An unscoped view loads at most 5,000 video features; a larger catalogue initially scopes to a drive. A selected drive/date range can page up to 20,000 features. Incomplete views are identified, and summary totals still describe all matches. Narrow filters for a complete map. Compatible loaded data updates immediately; the API remains authoritative for statistics and exports.

The strip renders visible frames with a small overscan. At most six image requests run concurrently, with nearby previews preloaded during replay. Automatic loading uses only API-provided thumbnails and previews. Original-resolution media is opened solely through the explicit full-resolution link. No routine background polling occurs.

## Scope and limitations

- Playback shows the pipeline's selected frames, not a continuous MP4 stream. The current API does not expose original video playback.
- Citizen/walking observations from the broader handoff are intentionally excluded following the requested scope correction.
- The bundled 198-ward boundary dataset is historical, from 2012. It is not the current administrative delimitation. Ward figures use loaded video features and segment centres; crossing segments are not clipped to boundaries. See [source and licence](data/README.md).
- All original sample segments have actual status `send`. Pending, skipped, missing-position and expert-corrected states are exercised with isolated fixtures; the sample is unchanged.
- Filter URLs do not preserve the selected frame or camera position. A time slider is not included. Printing produces tables, not a captured map image.
- Production-volume rendering, the real FastAPI host and LAN image latency still require validation in the receiving environment. Local mock tests cannot establish those timings.

## API requests

No additions are required for this dashboard. Optional future improvements would benefit from filtered per-km aggregates, original-video stream/timestamps and explicit MSL altitude fields.

## Verification

See [TEST-LOG.md](TEST-LOG.md) for executed tests and their limits. Optional tests and saved logs are provided in the package's `verification/` folder beside `webgis/`; they are not required on the production server.

## Screenshots

These captures show the current full-map layout.

![Full-map video survey workspace](screenshots/typography-desktop.jpg)

![Survey card beneath the surveyed-roads total](screenshots/typography-survey-desktop.jpg)

![Phone map and floating controls](screenshots/fullmap-mobile.jpg)

![Phone route player with clear map space](screenshots/fullmap-player-mobile.jpg)

## Licences

Project code retains the original GEOAI MIT licence in `LICENSE`. Leaflet 1.9.4, MarkerCluster 1.5.3, Leaflet.heat 0.2.0, MapLibre GL 5.24.0 and its Leaflet bridge 0.1.4 are pinned, with upstream licences under `vendor/`. Urbanist uses SIL Open Font License (`fonts/Urbanist-OFL.txt`). Rostex Outline is the existing project brand asset. Ward attribution is in `data/README.md`. OpenFreeMap/OpenMapTiles/OpenStreetMap attribution appears on the map, with fallback attribution when applicable.
