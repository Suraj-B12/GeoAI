# Rut depth — a prior interval per section (IRC:82-2023 PCI weight 0.14)

Written 2026-11-09. Function: `app/pci_rut.py: rut_prior(rutting_score_stats, calibration=None)` and
`score_stats(scores)`. Survey vehicle: **scooter, phone on the handlebar/front** (owner, 2026-11-09).

**This is a prior, not a measurement.** It says what range of rut depths is plausible on Bengaluru urban
bituminous roads before anything is measured, so the PCI can carry a bounded range for rut depth instead
of "unknown" (0–∞ mm). It must be shown as "prior (literature)" in the report.

## 1. Can our data see rut depth? No.

- **Camera:** a forward-looking phone camera at ~1 m height cannot resolve a depth of 5–20 mm spread across a
  ~0.5–1 m wide wheel path. Rut depth is measured with a 2–3 m straight edge or a transverse profiler.
- **IMU on a scooter:** car-based rut detection uses roll / roll rate. A 2025 MDPI paper (Applied Sciences
  16(7):3534, https://www.mdpi.com/2076-3417/16/7/3534 — HTTP 403, read only as a search-result summary)
  reports roll angle / angular velocity as "critical indicators" of rutting for a dedicated in-car
  terminal and says rut-depth research with crowdsourced devices such as smartphones "remains scarce". The
  physics needs a left and a right wheel in different wheel paths. **A scooter is single-track**: it rides
  in or between wheel paths, its roll is steering/lean, so phone roll carries no rut-depth information. Not used.
- **'Rutting' probe score** (`research.video_probe_scores`, probe_fp 7e47e0ab123dddb5, 7,095 scores): P(yes)
  5th/50th/95th/99th percentile 0.009 / 0.056 / 0.264 / 0.448; 0.6 % of segments above 0.5. The free-form
  Stage 2 list names Rutting on **0** video segments. The probe has never been checked against a rut
  measurement (Attain has no rutting class; no expert label; no survey yet). So it **does not change the
  interval** until a calibration exists; a section whose p90 reaches 0.5 is only flagged for a site look.

## 2. What the literature says about rut depths on Bengaluru / Indian urban bituminous roads

| Source (opened) | Where | Instrument | Rut depth |
|---|---|---|---|
| Anjana S. & Jamakhandi L., "Functional Evaluation of Urban Roads – A Study in Bengaluru", IJRTI 2(7), 2017 (https://www.ijrti.org/papers/IJRTI1707033.pdf, text read) | 4 links at Silk Board junction (MERLIN IRI 1.53–1.86 m/km, PCI 80–87) | 2.5 m wooden straight edge, 12 sample units per road | two roads **nil**; two roads **10–23 mm** (road 1: 10–23, mean 15; road 3: 12–18, mean 15) — the text calls them "moderate" |
| Jagadeesh H.S., Harikeerthan M.K., Abhimanyu K.S., "Performance Assessment and Pavement Management of Urban Roads Using GIS in Bangalore City", IJERT 2(11), 2013 (https://www.ijert.org/performance-assessment-and-pavement-management-of-urban-roads-using-gis-in-bangalore-city-2, read via fetch summary) | Hosur Road, Silk Board – Nayandanahally, 11.7 km, 4-lane arterial | not stated | **2.5 → 2.9 → 10 → 10.8 mm** (Feb 2011 → Apr 2012) |
| IRC:82-2023 Table 5.3 (urban), as transcribed in app/pci_irc2023.py `RANGES` | — | — | Excellent–Good < 5 mm, Satisfactory–Fair 5–10 mm, Poor–Fail > 10 mm |
| IRC:37 (bituminous pavement design): 20 mm rut at end of design life | — | — | 20 mm — **secondary sources only** (search summaries of course slides / infralens.in; IRC:37 itself not opened) |

Five Bengaluru roads, all inside **0–23 mm**; two of five with no rut at all; a heavily trafficked arterial
going from 2.5 to 10.8 mm in 14 months. This is far too little for a statistical interval or a "typical"
value.

## 3. The prior

- **Without calibration (now):** `lo90 = 0 mm, hi90 = 25 mm, point = None`. 0 because rut-free urban roads
  are common in the sample; 25 mm = the largest single reading (23 mm) rounded up, which also covers the
  IRC:37 20 mm design limit. The keys are named lo90/hi90 for the PCI interface, but it is a **judgement
  range** covering every value found, not a 90 % probability interval. No point value: the sample does not
  support one, and the PCI range should be computed from the two ends.
- **Effect on the urban PCI_rut (IRC:82-2023 eq., `_f7231`):** 0 mm → 98.9, 25 mm → 10.6 — the range is still
  wide, but it is bounded (unknown would be 0–100).
- **With calibration (after the survey):** `eval_results/pci/rut/rut_calibration.json`, schema
  `geoai.rut_calibration.v1`: bins of a section statistic of the Rutting score (`p90`, `mean` or `max`) tiling
  [0, 1], each with `lo90`, `point`, `hi90` (mm) and `n`, fitted from the survey's measured rut depths on the
  calibration half, plus `source` and `probe_fp`. Only then does the image score move the interval.
- **What the survey must record for this:** rut depth (mm) per section and lane, with the method (straight
  edge length / profiler), on the calibration half; sections must span rut-free and rutted roads.

## 4. What it does not cover
Lane differences (IRC:82-2023 averages lanes of one direction), service roads, concrete roads, and any
change since the 2011–2017 measurements. A road visibly deformed beyond 25 mm (e.g. shoving at bus stops)
would be outside the prior: the probe flag is the only hint, and it is unvalidated.
