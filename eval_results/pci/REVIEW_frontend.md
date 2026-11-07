DONE — verdict at the end

# Critic review round 3 — WebGIS condition-report UI, production sample PDFs, validation.md (2026-11-07)

Severity: BLOCKER (wrong/misleading as built), MAJOR (fix before deploying to live /gis/app/ or the paper), MINOR.
Harness http://127.0.0.1:8010/gis/app/ (geoai_test). Production read-only GETs only. Scratch: `scratchpad\critic3\`.

## Findings

### F1. MAJOR — "Report not generated" errors appear below the fold; at 1440×900 the user sees nothing happen
Evidence: harness 1440×900, A 12.88446,77.55222 → B 12.87916,77.54556, Km-stone 0.050 decreasing → Generate.
The backend correctly refuses ("road chainage would be negative at B …"), but `#pciNew .pci-error` is rendered at
the end of the form: rect top 1160 px while `#pciBody` shows 274–812 px, `scrollTop` 0. The Generate button
(in the sticky footer) just re-enables; a sighted user sees no reaction (the alert is only announced to screen readers).
Same for every `S.error` (Kannada name, lane width, 429, 503).
Fix: in `generate()`'s catch, after `render()`, `$('pciNew').querySelector('.pci-error')?.scrollIntoView({block:'nearest'})`
and focus it (tabindex=-1), or render the error in the footer next to the button. Also capitalise the backend
message and print chainage as "Km 0+050", not "Km 0.050".

### F2. MAJOR — default map/strip colour is mostly "Poor–Good" purple on clean road, the legend has no purple entry, and the section rows hide the backend's `band_note`
Evidence: same preview (1.03 km, 1 gap). Sections 1–7 and 9: "potholes 0 runs", cracking 0.0–x %, yet the chip and map
read **Poor–Good** (e.g. section 1: `skip_m` 50 of 105 m, pothole `E_lo 0 / E_hi 6.33`). DOM count of report lines:
8 × `.band-poorgood` (#7d6ba6), 1 × `.band-poor`, 1 × `.band-none`. `legendHTML()` uses `LEGEND.band` for
"Worst assessed parameter"/"Cracking"/"Potholes", which lists Good, Fair–Good, Fair, Poor–Fair, Poor, Insufficient —
**no Poor–Good**, so 80 % of the drawn stretch has no legend entry. `worstBand()` ranks a bracket by its worse end, so
"Poor–Good" (no pothole label, only skipped frames) beats "Good" and becomes the section colour. The backend now sends
`band_note` "Good (no potholes label); Poor if skipped segments hold a pothole" on every such section param
(REVIEW_backend #5 fix) — the UI shows `band_note` only for stretch params, never in the section rows, tooltips or strip.
Fix: (a) add `['poorgood','Poor–Good (undecided: skipped frames)']` and `fairgood/poorfair` to every legend;
(b) in the section row print the `band_note` instead of the bare "Poor–Good" (e.g. "Potholes: Good (no label) · Poor
if the 50 m skipped hold a pothole"); (c) for "Worst" colouring use `band_uncorrected` (the observed band) and draw the
skip-bound with a hatched/dashed edge, or rank Poor–Good below Poor/Poor–Fair; (d) show skipped metres per section.

### F3. MAJOR — the survey layer stays drawn under the report in label colours that collide with the band palette
Evidence: 1440×900, report open with the restored filter (Potholes, drive test1). Survey label colours from `/gis/meta`:
Normal #16a34a (≈ band Good #176b4a), Potholes #7f1d1d (≈ band Poor #a84235), Bleeding #6d28d9 / Streaking #a855f7
(≈ band Poor–Good #7d6ba6). The pothole survey markers sit on and beside the A→B line; `pci.css` dims nothing on desktop
(only pointer-events are disabled). The dashboard card also keeps showing "Matching road length 159.8 m · Distress 100 %"
for the old filter, which has nothing to do with the stretch.
Fix: while `body.pci-open`, set the survey pane to `opacity:.25; filter:grayscale(1)` (keep it as a picking guide) and
collapse/replace the dashboard stats with "Condition report open — survey filters are not applied to the report"
(the report uses every drive regardless of filter; say so in the card).

### F4. MAJOR (PDF, backend) — §7 table prints "Good / Poor if skips" for cracking whose upper end is only Fair
Evidence: preview PDF of the F1 stretch (`scratchpad\critic3\pvh_p04.png`), rows 0 and 2: Cracking E 0.0–6.3 / 0.0–7.3 %
(Table 5.3 cracking limits 5/15 → upper end **Fair**; JSON `band_hi:"Fair"`, `band_note:"Good (no cracking label); Fair
if skipped segments hold cracks"`), but the cell reads **"Good / Poor if skips"**. Cause `app/pci_report_pdf.py:170`
hard-codes "Poor": `"Good / Poor if skips" if short and p["band_lo"] == "Good"`.
Fix: `f"Good / {p['band_hi']} if skips"`; add a test with a cracking E_hi between 5 and 15.

### F5. MAJOR (screen + PDF + paper) — the pothole band is effectively binary: any flagged 5 m segment ⇒ "Poor"; Fair is unreachable per 100 m, and nothing says so
Evidence: Table 5.3 urban pothole limits `[0, 0.5]` %. One flagged 5 m segment in a 100 m section is 5 % ≫ 0.5 %, so a
section can only be Good (no label), Poor (any label) or Poor–Good (skips). validation.md §6 confirms: pothole band
{Poor 240, Poor–Good 118, Good 50} on 408 blocks — **Fair 0**. At stretch level Fair needs < 1 flagged segment per km.
In the F1 preview the stretch is "Potholes **Poor** 1.1–4.0 %" from **one** 2-segment run in 915 m, while the nominal
pothole area is 0.001 % → **Fair** (PDF p. 1, `pvh_p01.png`). The UI shows only "Poor" + a red PRIORITY box; the nominal
area band is not on screen at all. DESIGN §3 calls runs/km "the primary figure", yet both the UI table and the PDF lead
with the % band.
Fix: state it next to the band on screen and in the PDF: "With 5 m segments one flagged segment is 5 % of a 100 m
section, so the pothole band reads Poor whenever any pothole is flagged; read it as 'potholes flagged: yes/no' and use
runs per km for amount." Show the nominal-area band on screen beside the % band. Paper: report pothole results as
runs/km and flagged/not flagged, never as an IRC band distribution.

### F6. MAJOR — the PRIORITY box ("rectify straightway") can rest on a single false-positive run, and the UI does not link the frame(s) behind it
Evidence: F1 preview. The only pothole run (2 segments, Km 0+780, `…F0D1_8130_seg00450`, confidence 0.96) is the
first photo in the PDF annex (`pvh_p06.png`): it shows a vehicle's shadow and a lane arrow on smooth asphalt — no
pothole visible. That run alone makes the stretch "Potholes: Poor", the red PRIORITY box, the "Potholes present:
rectify straightway" line in §8 and the red section on the map. The box text does say "unvalidated", but the engineer
gets no way to check it from the box.
Fix: in the PRIORITY box print the evidence size ("rests on 1 run, 2 frames") and add a button "Check the N frames"
that opens those frames (annex ids are in `report.annex`). When runs ≤ 2, word it as "Possible pothole — check the
frame before acting" instead of an IRC instruction.

### F7. MAJOR (accessibility) — every action re-renders the panel with `innerHTML`, so keyboard focus falls to `<body>`
Evidence (1440×900, preview open): focus `[data-act=section][data-index="2"]` + activate → `document.activeElement`
= `BODY`; same after activating a frame thumbnail (`.pci-thumb`), Pick/Move, + Via point, Remove, Save. Only the
"Colour by" select restores focus. A keyboard or screen-reader user who opens a section is thrown back to the top of
the page; the next Tab goes to the GEOAI bar, not to the frames that just opened.
Fix: in `render()`, remember `document.activeElement`'s `data-act` + `data-index`/`data-id` (or id) before replacing
the HTML and re-focus the matching element afterwards (`{preventScroll:true}`); after opening a frame, focus the
`<figure>` (tabindex=-1). The frame image alt "Road frame for this segment" should carry the label and chainage.

### F8. MAJOR — the Saved-reports list and the minimised summary lead with the bracketed overall class (B1 not carried through)
Evidence: Saved tab (harness, 10 reports): every row reads "<length> · **final Poor–Fair** · urban · <date>" — the overall
class is the only condition shown, though B1/the PDF call it "secondary, never a verdict". The minimised card says
"`<len> · final Fair–Good · panel hidden`" (`updateMinSummary()`), i.e. a Poor-pothole stretch is summarised as
"Fair–Good". Also: 10 of 10 saved reports show the red PRIORITY box and "Potholes: Poor" (see F5) — on the list the
user cannot tell the stretches apart at all.
Fix: list rows and the minimised summary show "Potholes <band> · <runs/km> runs/km · Cracking <band>" (from
`summary.headline`), with the overall class, if at all, last and muted ("overall, secondary: Poor–Fair").

### Layout sweep (own detector, `scratchpad\critic3\detect.js`)
Detector: pairwise rect overlap of every visible floating element (panel, GEOAI bar, dashboard heading/stats/filters,
toolbar, #notice (forced visible), #basemapNotice, legend, map tools, zoom, scale, attribution, drive control,
detail panel, strip, settings, all fixed/absolute children of body), off-screen check, horizontal overflow and
text-leaf overlap inside the panel (head, KPIs, point rows, footer, section rows, tabs), SVG strip-label overlap,
page h-scroll, font size < 10.5 px, touch targets < 32 px at ≤ 820 px. States: closed, picking A, path built, result,
section + frame, minimised, saved list, saved-open, saved-open + #notice; light and dark; 1440×900, 1024×768,
820×1180, 375×812 (72 runs). **Result: 0 overlaps, 0 off-screen, 0 overflow, 0 page h-scroll in every run.**
Remaining items are F9–F10 and minors below.

### F9. MAJOR (usability) — the free map area is small at 1024×768 and on phones, and the report text window on a phone is 180 px
Evidence: `GeoAIApp.map.availableViewport()` with a result open: 1440×900 → 902×386 px; **1024×768 → 490×195 px**
(the dashboard heading + 4 stats for the unrelated survey filter cover the right half down to y = 485);
375×812 → 279×228 px, and the sheet's scroll box `#pciBody` is **180 px** tall (563–744): the per-parameter table
needs 3–4 scrolls before the PRIORITY box (screenshot `…tool-results\mcp-Claude_Browser-blob-1794015286660-6soyc2.jpg`).
The GEOAI bar, the dashboard heading (2-line title), the basemap notice and the sheet's own head/tabs/footer use the rest.
Fix: while the report is open, hide `.dashboard-stats`/`.active-filters` on desktop too (they describe the survey
filter, not the report) and shrink the heading to one line; on phones hide the dashboard heading and add an
"Expand" state (sheet to ~85 % height) next to Hide.

### F10. MINOR (accessibility) — contrast and target size
- Light theme, white 11 px bold text on band chips: Fair–Good 3.73:1, Fair 3.72:1, Poor–Fair 4.43:1, Insufficient
  3.26:1 (< 4.5:1). `.pci-final` coloured text uses the same colours on white. Dark theme passes (≥ 5.75).
- Many 9–10 px texts (eyebrow 9 px, evidence `em` 10 px, KPI `small` 10 px, strip axis ≈ 8 px).
- At ≤ 820 px: "Clear points" 72×15 px, "Show all N frames" 105×15, "Report header…" summary 764×15 — below the
  WCAG 2.2 24 px minimum. Fix: darker band tokens in light mode (e.g. Fair #8a5d00, Fair–Good #55701a, none #6b7076),
  min 11 px text, `min-height:32px` on text buttons/summaries in the panel.

### F11. MINOR — UI items
1. Screen and PDF colour the stretch differently: the PDF route/strip uses the **pothole band** ("the overall class is
   never used for colour"); the UI default is "Worst assessed parameter" and offers "Final class (bracketed)". A user
   comparing the screen with the PDF sees different maps. Make the UI default "Potholes band" (as the PDF) and drop
   or rename "Final class" to "Overall (App. 4, secondary)".
2. "Length A→B 1.03 km" (KPI and path box) and the km labels after a gap are printed as exact, although the gap is a
   straight-line lower bound (validation §4: OSM route / straight line median 1.10, p90 1.71, max 3.17). Print
   "≥ 1.03 km" when `gaps > 0`, and mark chainage after the first gap "(≥)" or add the PDF's "unknown extra error
   after each gap" to the KPI.
3. Path box label "Surveyed runs" clashes with "pothole runs" (consecutive positive segments). Use "Surveyed length".
4. Snapped distance rounding differs: UI "snapped 3 m" (2.5 → `toFixed(0)`), PDF "snapped 2 m" for the same point.
5. Section rows show "road Km 12+350 (12.350)" — the doubled form REVIEW_backend #7 asked to remove (also in the
   CSV `road_chainage_*` and PDF §5).
6. Frame viewer and thumbnails show time and lat/lon but **no chainage**; the engineer works in Km. `attribution`
   has `measure_m` per segment — show "Km 0+780 (road Km 13+130)" in the caption and on each thumbnail.
7. Candidate-treatment details drop the IRC clause (`distress[].clause`, e.g. "7.5.3.5; 10.3.8; 11.4") shown in the PDF.
8. After saving a preview, PDF/CSV still POST the preview and are named `…_preview.*` with "Preview (not stored)"
   on every page, though the report is now no. N. After a successful save, switch downloads to `/reports/{id}.pdf`.
9. A wrong operator key is stored in sessionStorage on every keystroke (`onPanelInput`) and stays there after the
   401. Store it only after a successful save; clear it on 401.
10. Saved list: "Rule: a random 1 km stretch … (draw 1)" on a 1.55 km report (sample 19/2). Say "random stretch
    starting 1 km … routed length 1.55 km" or record requested vs routed length (REVIEW_backend #13, still open).
11. Error/refusal strings come straight from the backend in lower case ("road chainage would be negative…",
    "B is 291 m from the nearest surveyed segment…"). Capitalise in the UI.
12. Strip chart "Potholes · runs" axis shows only the max ("1") and no 0; fine for one value, unclear when the max is 3.

### F12. MINOR — production sample PDFs (reports 1, 5, 8; read-only GETs; pages in `scratchpad\critic3\p1_*`, `p5_*`, `p8_*`)
Round-2 fixes verified on the PDFs: pothole-band colouring of route and strip with a violet "Good on labels, Poor via
skips" class; boxed PRIORITY text in ink; muted "Overall (Appendix 4, secondary)"; "(est.)" in column heads; annex
captions carry per-frame chainage (not re-checked against `measure_m`); "entered point snapped x m to the nearest segment"; §8 heading leads with
"rectify straightway". Remaining:
1. "Name of the road: **not entered (entered by requester, unverified)**" (all three) — contradictory; print
   "not entered" alone.
2. §8 heading cites "§5.3 **p. 6**" while every other line cites "§5.3 p. 7"; and "Table 9.1 **)" has a stray `**`.
3. "1 runs" (report 1 p. 3, preview p. 1) — pluralisation.
4. Direction row "P1: vehicle travelled **unknown**" (preview) — "direction unknown".
5. For a stretch ≤ ~1 km the "Whole stretch" and the single "Km row" blocks in §5, §6 and §8 are identical — print
   one, or say "(= whole stretch)".
6. Annex frames picked as "potholes band Poor–Good in Km 0+200–0+300" show Bleeding/Streaking labels (report 5,
   Km 0+275) — they do not illustrate anything about potholes. Pick the gate-skipped frames instead, or drop them.
7. Report 1 route figure (8.3 km, p. 2): the track fills ~15 % of the frame and 83 tick marks make it look hatched;
   no basemap or road names, so it cannot be used to locate a section. Use 500 m ticks above 3 km (as the UI does)
   and a landscape/fitted frame.
8. Strip legend: violet means "Good on the labels, Poor only if skips" for the band rows, but the last row
   ("Overall, App. 4") also uses violet for Poor–Good caused by the not-assessed parameters — a different meaning.
9. Annex evidence (report 5, p. 6): of 5 "pothole run" frames, Km 0+050 and 0+135 show only tree shadows on asphalt;
   preview Km 0+780 (F6) shows a vehicle shadow. Consistent with REVIEW_backend #3 (shadows as potholes) — it should be
   named as a known failure mode in Limitations ("dappled shade and vehicle shadows are often labelled Potholes").

## C. validation.md — overstated or unsupported

### V1. MAJOR (paper) — Statement 1 says "only the per-parameter bands (cracking, potholes) carry information"; for potholes this is not true
§6 itself shows the pothole band is {Poor 240, Poor–Good 118, Good 50} on 408 blocks and **never Fair** (see F5): at
5 m segments × 100 m blocks it is a "any pothole label?" flag, and Poor–Good means "no label, some skips". Restate:
"the cracking band carries graded information; the pothole band per 100 m is equivalent to 'pothole flagged / not
flagged'; use runs per km for amount". Add the per-block distribution of runs/km.

### V2. MINOR — 100 m test–retest numbers on n = 2 are printed like results
"cracking: band agreement 1.000 … potholes: band agreement 0.500 … bracketed final class agreement 1.000" are on
**2 blocks**. Statement 3 correctly says class repeatability is not measured, but the bullets should carry "(n = 2;
not interpretable)" or be removed, so nobody quotes "class agreement 1.000".

### V3. MINOR — κ confidence intervals all start at exactly 0.0 (degenerate resamples)
Distress [0.0, 0.962], Potholes [0.0, 0.908], Cracking [0.0, 0.848]. `validation.json`: 8 bootstrap blocks. In
`scripts/pci_validation.py:65` κ is exactly 0 whenever a resample has positives in only one of the two drives
(b1 = 0 ⇒ agree = pe ⇒ κ = 0), which happens often with 8 blocks and 5–15 positives — the lower bound is that
artefact, not a measured lower agreement. Say so, report the share of such resamples, and do not quote "κ 0.756"
without "[0, 0.96], 8 blocks, not informative".

### V4. MINOR — §4 gap ratio and the chainage claim
"OSM route / straight line median 1.095, p90 1.709, max 3.167 … supports printing gaps as lower bounds" — it also
means chainage after a gap can be short by up to ~2× the gap. Statement 2 should say so explicitly (and the UI should
mark post-gap chainage, F11.2). Same vertex-snapping caveat as the length table, so even the ratio is uncertain.

### V5. MINOR — sample table "random_1km" rows
Row 2 is 1,551.7 m with 3 gaps; the rule name says 1 km. Add requested vs routed length (REVIEW_backend #13).

Not overstated (checked): §1 arithmetic, §2 framed as method-on-raw-noise, §4 framed as inconclusive, §7 limitation
(0 expert labels; repeatability ≠ correctness).

## Verdict

No BLOCKER in the UI as built: the A→B flow works end to end on the real routes (path, via add/remove, 40 m refusal,
gap path, generate, sections, frames, PDF/CSV/JSON, wrong key, saved samples), the existing app still works (filters,
legend, replay — closed when the report opens, per-km report, theme), the console shows only the expected 422/401 and
MapLibre style warnings, and my own detector finds **0 overlaps / clipping / h-scroll in 72 runs** (9 states × 2 themes
× 4 sizes).

MAJOR before copying into the live `/gis/app/`:
- F1 errors from Generate render below the fold;
- F2 default colouring paints clean road "Poor–Good" purple with no legend entry and hides `band_note`;
- F3 the survey layer and filter stats stay under the report in colliding colours;
- F7 focus is lost on every action;
- F8 the saved list and minimised summary lead with the overall class;
- F9 the free map area at 1024×768 and the phone text window are too small;
- F4 (backend PDF) "Good / Poor if skips" printed for cracking whose upper end is Fair.

MAJOR for the paper and for what an engineer reads (F5, F6, V1): with 5 m segments the pothole band is a yes/no flag
(Fair never occurs; 10/10 stored reports show "Potholes Poor" + PRIORITY), and in the reports I looked at, the frames
behind PRIORITY include shadows labelled as potholes. Present potholes as runs/km plus "flagged", and link the
PRIORITY box to its frames.

It is reasonable to deploy to the live `/gis/app/` as a screening aid after F1–F4, F7 and F8 are fixed, with F5/F6
wording added. F9–F12 can follow.
