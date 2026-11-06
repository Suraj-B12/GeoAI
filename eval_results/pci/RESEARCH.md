# Chainage and IRC:82-2015 Condition Rating for the WebGIS A→B Report — Research Note

Written 2026-11-07. This note is research only: nothing in the code base was changed.

**How to read this note.** Every statement carries one of three tags:

- **[IRC]**: what IRC:82-2015 actually says. Each one cites the clause or table and the page, given
  as "p. N (PDF M)", where N is the number printed on the page and M is the PDF page index.
- **[PRACTICE]**: common practice elsewhere (MoRTH/NHAI/NHIDCL/NRRDA templates, ASTM, the
  literature). A web source is given for each.
- **[INFERENCE]**: my own reasoning or recommendation. It is not in any standard.

**Primary source.** IRC:82-2015, *Code of Practice for Maintenance of Bituminous Road Surfaces (First
Revision)*, Indian Roads Congress, June 2015. Local copy: `IRC 82/irc.gov.in.082.2015.pdf`, 81 pages, all
read. The tables are a vector text layer, not images. I extracted them with `pdftotext -layout` and
then checked every row of Tables 5.1–5.3 and Appendix 4 against the per-string page coordinates
(pypdf). The `-layout` output shifts the value columns of Tables 5.2 and 5.3 by one row. The coordinate
check removes that shift, so the values below are the corrected, row-aligned ones. Cells that could
still be read more than one way are flagged in §1.9.

Copyright: the document reads "All Rights Reserved … no part … shall be reproduced". The tables are
quoted here for internal implementation. The paper should cite them (IRC:82-2015, Table 5.x, p. N)
rather than reprint whole pages.

---

## 1. What IRC:82-2015 specifies

### 1.1 Where condition rating sits in the code [IRC]

- **§5.1, p. 4 (PDF 11).** "the first step towards planning of maintenance operations is the evaluation of
  the existing pavement surface". The visual survey identifies "type, location and magnitude of the
  distress as described in Appendix-1 and recorded in a standard format as in Proforma 1". For more
  precise project-level data, actual measurements follow Appendix 2 / Proforma 2. "The rating of pavement
  may be assigned as per criteria given in Tables 5.1 to 5.3 for different categories of roads."
- **Note to Table 5.1, p. 4.** The distresses are "to be observed visually as described in Appendix – 1 and
  to be noted in a standard format as given in Proforma 1."
- **Text after Table 5.3, p. 5 (PDF 12).** When different distresses fall in different rating categories, the
  overall rating uses "an appropriate weightage with regard to the significance of a particular
  distress", given in **Appendix-4**.
- **The term "pavement condition index".** It appears exactly once, in §9.3, p. 42: stretches "due for
  periodic renewal on the basis of the pavement condition index and periodicity". IRC:82 never defines
  a separate "PCI". What it defines is the **distress-based rating on a 1–3 scale** (Poor / Fair / Good).
  Do not call our output an "ASTM PCI". Call it the "IRC:82-2015 distress-based condition rating"
  (see §3).

### 1.2 Table 5.1 — Pavement Distress Based Rating for **Highways** (p. 4, PDF 11) [IRC]

| Defects (type) | Range of Distress | | |
|---|---|---|---|
| Cracking (%) | >10 | 5 to 10 | <5 |
| Ravelling (%) | >10 | 1 to 10 | <1 |
| Potholes (%) | >1 | 0.1 to 1 | <0.1 |
| Shoving (%) | >1 | 0.1 to 1 | <0.1 |
| Patching (%) | >10 | 1 to 10 | <1 |
| Settlement and Depression (%) | >5 | 1 to 5 | <1 |
| Rut depth (mm) using 3 m straight edge | >10 | 5 to 10 | <5 |
| **Rating** | **1** | **1.1 - 2** | **2.1 - 3** |
| **Condition** | **Poor** | **Fair** | **Good** |

### 1.3 Table 5.2 — Pavement Distress Based Rating for **MDR(s) and Rural Roads (ODR and VR)** (p. 5, PDF 12) [IRC]

| Defects | Range of Distress | | |
|---|---|---|---|
| Cracking (%) | >20 | 10-20 | < 10 |
| Ravelling (%) | >20 | 10-20 | <10 |
| Pothole (%) | >1 | 0.5 to 1 | <0.5 |
| Patching (%) | >20 | 5-20 | <5 |
| Settlement and depression (%) | >5 | 2 to 5 | <2 |
| **Rating** | **1** | **1.1 - 2** | **2.1 - 3** |
| **Condition** | **Poor** | **Fair** | **Good** |

Table 5.2 has no shoving row and no rut-depth row.

An independent check that these values are right: the NRRDA/MoRD *Managing Maintenance of Rural
Roads — Guidance Note for States* (Oct 2014), Table 1, gives the same thresholds for rural roads:

- potholes "Upto 0.5 / 0.5 to 1.0 / Above 1.0";
- patching "Upto 5 / 5 to 20 / Above 20";
- cracking and ravelling "Upto 10 / 10 to 20 / Above 20".

It adds a PMGSY "Pavement Condition Index 4 to 5 / 2 to 3 / 1", which is a different, 1–5 scale
defined in the PMGSY Operations Manual. Source:
<https://cdnbbsr.s3waas.gov.in/s3d9ff90f4000eacd3a6c9cb27f78994cf/uploads/2025/02/20250227831088188.pdf>, p. 4.

### 1.4 Table 5.3 — Pavement Distress Based Rating for **Urban Roads** (p. 5, PDF 12) [IRC] — the table that applies to Bengaluru city streets

| Defect | Range of Distress | | |
|---|---|---|---|
| Cracking % | >15 | 5-15 | <5 |
| Ravelling % | >10 | 5 to 10 | <5 |
| Potholes % | >0.5 | >0 and ≤0.5 | NIL (0) |
| Settlement % | >5 | 1-5 | <1 |
| Rut depth (mm) using 3 m straight edge | >10 | 5-10 | <5 |
| **Rating** | **1** | **1.1 - 2** | **2.1 - 3** |
| **Condition** | **Poor** | **Fair** | **Good** |

Table 5.3 has no patching row and no shoving row. **On an urban road a single pothole of any size
takes the pothole parameter out of "Good"**, because Good means NIL.

### 1.5 Appendix 4 — Utility for Rating of Pavement Based on Quantity of Distress (pp. 73–74, PDF 80–81) [IRC]

What the appendix states:

- The ratings come from an Excel utility "available to the users in the CD enclosed with this document".
  We do not have the CD.
- The utility "assigns the rating value to each individual parameter from 1, 1.1 to 2.0 and 2.1 to 3.0
  which corresponds to Poor, Fair and Good".
- A fixed weight multiplies each parameter's rating. The "Final Rating Value is calculated by taking
  **the average of the Weighted Rating Values** of all parameters".
- For the other road categories, "final rating … may also be worked out, using the same approach".

Fixed weights (p. 73):

| S.No. | Parameter | Weightage (Fixed) (Multiplier Factor) |
|---|---|---|
| 1 | Cracking | 1.00 |
| 2 | Ravelling | 0.75 |
| 3 | Potholes | 0.50 |
| 4 | Shoving | 1.00 |
| 5 | Patching | 0.75 |
| 6 | Settlement | 0.75 |
| 7 | Rut Depth | 1.00 |

Worked example for Highways, Table 5.1 (p. 74), row-aligned:

| Distress Type | Input (%) | Rating as per Table 5.1 | Weightage | Weighted Rating Value |
|---|---|---|---|---|
| Cracking (%) | 9.00 | 1.2 | 1.00 | 1.2 |
| Ravelling (%) | 8.00 | 1.2 | 0.75 | 0.9 |
| Potholes (%) | 0.00 | 3.0 | 0.50 | 1.5 |
| Shoving (%) | 0.09 | 2.1 | 1.00 | 2.1 |
| Patching (%) | 5.00 | 1.5 | 0.75 | 1.1 |
| Settlements (%) | 4.00 | 1.2 | 0.75 | 0.9 |
| Rut Depth (%) [sic; Table 5.1 uses mm] | 9.00 | 1.3 | 1.00 | 1.3 |
| **Final Rating Value** | | | | **1.3** |
| **Condition** | | | | **Fair** |

The arithmetic checks out: (1.2+0.9+1.5+2.1+1.1+0.9+1.3)/7 = 9.0/7 = 1.29, which rounds to 1.3 (Fair).

**How the utility interpolates inside a band.** IRC does not state this. I inferred it from the worked
example [INFERENCE]. Write the band limits as L (Good/Fair boundary) and U (Fair/Poor boundary):

- Poor (x > U): r = 1.0.
- Fair (L ≤ x ≤ U): r = 2 − (x − L)/(U − L), then truncated to one decimal.
- Good (x < L): r = 3 − x/L, then truncated to one decimal.

This reproduces 6 of the 7 example rows exactly: cracking 1.2, ravelling 1.2, potholes 3.0, shoving 2.1,
patching 1.5 (1.556 truncated), settlement 1.2 (1.25 truncated). It gives **1.2 for rut depth, where the
example prints 1.3**, although rut depth and cracking have the same input (9) and the same bands
(5 to 10).

I also tested the alternative, a linear map onto the printed 1.1–2.0 and 2.1–3.0 spans. It fits worse:
4 of 7 when truncated, 3 of 7 when rounded. No single rule reproduces all seven rows, so the utility
may use a different rule for rut depth, or the example has a typo.

**Recommendation:** implement the band lookup, which is unambiguous IRC, plus this interpolation. Label
the decimal as "interpolated per the Appendix-4 worked example". The class (Poor/Fair/Good) never
depends on the interpolation.

**A quirk to disclose [INFERENCE from the stated formula].** The formula is the plain mean of
wᵢ·rᵢ, not Σwᵢrᵢ/Σwᵢ, so the final value is compressed:

- An all-"Good" highway (every r = 3.0) scores 3·5.75/7 = **2.46**.
- An all-"Poor" highway scores 5.75/7 = **0.82**, which is below the "1" of Table 5.1.
- With the five urban parameters, the range is 2.40 to 0.80.

Classify the final value with the same bands: ≤1.0 Poor, 1.1–2.0 Fair, ≥2.1 Good. Report the IRC
formula as specified, and show Σwr/Σw beside it only as a sensitivity line.

### 1.6 Serviceability indicators, Tables 5.4 and 5.5 (p. 6, PDF 13) [IRC] — not measurable by us

| Indicator | Highways L1 Good / L2 Fair / L3 Poor | Urban L1 / L2 / L3 |
|---|---|---|
| Roughness (max. permissible), mm/km | 1800 / 2400 / 3200 | 1800 / 2600 / 3400 |
| Skid Number (ASTM-274), minimum desirable | 60 / 50 / 40 SN | 65 / 55 / 45 SN |

Note on p. 6: "Level 1 is expected to match with new pavement condition, level 2 is the in-service minimum
desirable level and level 3 is the warrant for intervention". Per §5.2, these "are to be read in
conjunction with" Table 5.1 (highways) and Table 5.3 (urban). The report must state that roughness and
skid resistance were not assessed.

### 1.7 Survey method, inspection unit and chainage in IRC:82 [IRC]

| Item | What IRC:82 says | Where |
|---|---|---|
| Frequency | At least two surveys a year (before and after the monsoon) for highways and urban roads; at least one for other roads, preferably after the monsoon | §5.3, p. 6 |
| Speed | "a vehicle travelling at a speed of 5 km/hour supplemented by inspection of the critical locations by walking". Appendix 1 says 8–10 km/h with "a well experienced/trained team of 3-4 persons" (the two clauses differ) | §5.3 p. 6; App. 1 p. 62 |
| **Recording unit** | "The data collected should be recorded methodically, **kilometer wise**." Distresses are "recorded in individual percentage for **each km length** in Proforma 1" | §5.3 p. 6; App. 1 p. 62 |
| Who | "a team of experienced engineers at a responsible level"; automated survey vehicles are also allowed and are "recommended" for highways | §5.3 pp. 6–7 |
| Distresses recorded (App. 1 and 2) | Alligator/fatigue cracking; longitudinal & transverse cracking; bleeding; depressions/settlements; potholes; patching; ravelling; rutting | pp. 62, 64 |
| Proforma 1 header | Name of the Road; Km … to Km …; Chainage of Test Section Km … to Km …; Carriageway Width (m); Type of Surface; Date of Observation; Weather Condition | p. 63 |
| **Proforma 1 columns** | Chainage From / To; Cracks (%); Patch Work (%); Potholes (%); Bleeding (%); Depression (%); Edge Breaking (%); Ravelling (%); Shoving (%); Rut Depth (mm); Drainage and Shoulder condition; Total; signed "Recorded by" and "Checked by" | p. 63 |
| Proforma 2 (measured) | Adds Name **and Number** of the Road, and Chainage of Sub-Section. Per distress: Type, Size (m, L×B), Area (m²); pothole Depth (mm); rut depth "To be measured using 3 m straight edge" | p. 65 |
| Image-based measurement | "Pavement Surface Imaging Technique … pavement cameras … generate the **calibrated images (with known length and width)** … measurement of … cracking, ravelling, patching and pot holes is done by using the Image Processing software" | App. 3 §B.2, p. 70 |
| LHS/RHS, lanes | **Not specified in IRC:82.** Proforma 1/2 have no side or lane column | — |
| Denominator of "%" | **Not defined explicitly.** Proforma 2 records areas (m²), which implies % = distressed area ÷ pavement area of the section; "% length" is never stated | [INFERENCE] |

Our forward-looking phone frames are **not** the calibrated top-down images that App. 3 §B.2 describes.
That is the root of most limitations in §5.

### 1.8 Severity definitions (Section 7) [IRC] — used for treatment choice, not for the Tables 5.x rating

The Tables 5.1–5.3 rating uses **extent only (% or mm)**. Severity enters through the treatment
choice in §7.

| Distress | Severity rule (verbatim numbers) | Clause, page |
|---|---|---|
| Bleeding/fatty surface | Low: "isolated spots measuring less than 5 m² area in a lane km … total area less than 1%"; Medium: "area is in the range of 1 to 5%"; High: "Extensive bleeding … showing initiation of shoving" | 7.2.1.4, pp. 9–10 |
| Smooth surface | Skid number 30–40 = low; below 30 = high | 7.2.2.4, p. 11 |
| Streaking, hungry surface, hairline cracks, reflection cracks, corrugation, shoving, shallow depression, settlement/upheaval, stripping, edge breaking | "Not applicable" | §7 |
| Hairline cracks | width "less than one mm" (a definition, not a severity) | 7.3.2.1, p. 13 |
| Alligator/map cracking | Low: "depth and width … 1 to 3 mm"; Medium: "3 to 6 mm", interconnected, slight spalling, no pumping; High: "more than 6 mm", pieces loose or missing, pumping | 7.3.3.4, pp. 15–16 |
| Longitudinal cracking | Low "1-3 mm wide and infrequent"; Medium "3-6 mm wide"; High "greater than 6 mm wide and are frequent (numerous)" | 7.3.4.4, p. 17 |
| Transverse cracking | same widths as longitudinal | 7.3.5.4, p. 18 |
| Edge cracking | Low: no breakup or loss of material; Medium: "some breakup … (upto 10% of the affected length)"; High: "more than 10% of the affected length"; location "0.3 to 0.5 m inside of pavement edge" | 7.3.6, pp. 18–19 |
| Slippage | Low: isolated locations in a lane; High: along the wheel path in the entire lane or carriageway | 7.4.1.4, p. 21 |
| Rutting | Low "4-10 mm deep"; High "more than 10 mm" (note: Table 5.1/5.3 bands are <5 / 5–10 / >10) | 7.4.2.4, p. 23 |
| Corrugation | "usually shallow (about 25 mm)", wave spacing "2-3 m" (a definition) | 7.4.3.1, p. 23 |
| Shallow depression | "dipping about 25 mm or more below the profile" (a definition) | 7.4.5.1, p. 25 |
| Ravelling | Low: "some loss of fines … initial stage of binder wearing out"; Medium: "loose particles exists … rough surface"; High: "surface is too rough with loss of aggregates" | 7.5.2.4, p. 29 |
| Potholes | Small: "25 mm deep and 200 mm wide"; Medium: "25 to 50 mm deep and 500 mm wide"; Large: "greater than 50 mm deep and 500 mm width" | 7.5.3.4, p. 31 |
| Edge breaking | location "upto 30 cm from the edge" | 7.5.4.2, p. 32 |

### 1.9 Cells I am less than certain about

- **Tables 5.2 and 5.3: row alignment.** The values in §1.3 and §1.4 were confirmed by text coordinates,
  and the rows sit on separate y-positions. Some strings in Table 5.1 (the ravelling, potholes, shoving
  and patching value strings) and in Table 5.3 (cracking and settlement) were not positioned by pypdf.
  For those I used the `-layout` order. That order is consistent with every positioned row and with
  NRRDA's Table 1 for the rural case.
  **I could not view the page as an image:** no PDF rasteriser was available, and the browser preview
  timed out. A 1-minute visual check of PDF pages 11–12 is recommended before coding.
- **Table 5.3, pothole middle band.** It is printed as ">0 and ≤0.5". The "≤" glyph appears in the
  text layer.
- **Appendix 4, rut-depth example row (1.3).** It is not reproducible from the other rows (see §1.5).
- **Contents page.** It lists "Appendix 1 Pavement Condition Assessment/Visual Surface Condition
  Assessment … 62", but the section numbering on that page is garbled in the text layer. This has no
  effect on the content.

### 1.10 Maintenance policy tied to the rating [IRC]

| Rule | Text | Where |
|---|---|---|
| Rating decides activity | "Type of maintenance activity to be undertaken shall depend upon condition rating, as per Tables 5.1 to 5.3." | §5.3, p. 6 |
| Riding-quality distress first | Where distress "adversely influences the riding quality significantly and affects smooth flow of traffic, these should be rectified straightway"; for "minor cracking, and ravelling, optimal preventive strategy should be adopted" | §5.3, pp. 6–7 |
| Rural renewal | "periodic renewal treatments may be decided when condition rating reaches to a value of 2 to 1" | §5.3, p. 7 |
| Highway/urban renewal | "periodic renewal may be undertaken at a serviceability level of 2" | §5.3, p. 7 |
| Preventive | "Preventive maintenance shall be undertaken before pavement rating drops to 2"; "when drop in quality of pavement surface is less than 30% as quantified in Tables 5.1 to 5.3"; treatments "shall be effective before the pavement condition rating drops below 2" | §5.3 p. 7; §8.2.4 p. 34 |
| Preventive is not for poor pavements | "A thin overlay may not be considered as a preventive maintenance, when it is applied to a distressed pavement in poor condition. Slurry seal/microsurfacing may not be considered as preventive maintenance, when placed on an extensively cracked surface." | §8.2.3, p. 33 |
| Renewal priority | roads ranked "in the ascending order of condition rating" | §9.5, pp. 43–44 |
| Rehabilitation out of scope | strengthening via IRC:81 (Benkelman), IRC:37, IRC:115 (FWD) | §2.4, pp. 2–3 |

**Table 9.1 — Suggested Renewal Treatment on Flexible Pavement (p. 43, PDF 50)**

| Road Category | Condition Rating* | Roughness, mm/km | Suggested Treatment** (Thickness, mm) |
|---|---|---|---|
| NH | 2 | 2400 | 30-40 BC |
| SH | 2 | 2800 | 30 BC/30 SDBC |
| MDR/RR | 2-1 | 4000 | SD/PC/MSS/SDBC (20-25) |
| Urban Road | 2 | 2600 | 30-40 BC/SDBC/2 Coat Micro Surfacing |

\* "When the condition of road is rated as 1, it is recommended that a detailed investigation is carried out to
assess the strengthening requirement, if any. In case, the pavement is found to be structurally adequate, then
the periodic renewal will suffice."
\*\* "Wherever feasible, recycling of wearing course may be preferred"

Abbreviations, as used in IRC practice:

- BC = Bituminous Concrete
- SDBC = Semi-Dense BC
- SD = Surface Dressing
- PC = Premix Carpet
- MSS = Mix Seal Surfacing

The preventive-treatment warrants are in Tables 8.1–8.7 (pp. 35–41). Each one lists climate, traffic,
conditions addressed and limitations, for: crack filling/sealing; fog seal; slurry seal;
microsurfacing; surface dressing; thin hot/cold-mix overlays (15–25 mm); ultrathin friction courses
(10–20 mm). Every table lists "structural failure (… extensive fatigue cracking …)" as a limitation.

---

## 2. Chainage — conventions and GPS practice

### 2.1 Indian conventions [PRACTICE unless tagged]

- **Format.** MoRTH/NHAI/NHIDCL documents write chainage in kilometres to three decimals. Examples:
  "NH-10 Sevoke-Rangpo section from Km 0.000 to 52.100" (NHIDCL NSV EOI, Dec 2024); NSV output fields
  "Start Chainage 0.500 / End Chainage 1.500 (in km)".
  NRRDA's ILO/World Bank strip-chart form heads its columns **"0+000 … 1+000"**, i.e. km+m with
  100 m columns (NRRDA Guidance Note 2014, road condition inventory form).
  IRC:82 Proforma 1 uses "Km ……… to Km ………" and "Chainage From / To" [IRC, p. 63].
  The usual written form is **"Km 12+350"**, equivalent to 12.350 km.
- **Kilometre and 200 m stones.** IRC:8-1980 (*Type Designs for Highway Kilometre Stones*, 2nd rev.) sets
  the km stones. A MoRTH circular (28-01-1987, compendium 603.10) says National Highways show
  distances on km stones installed per IRC:8-1980.
  IRC:26-1967 is the *Type Design for 200-Metre Stones* (numerals, black on white). I could not open
  the IRC:26 text itself; a secondary summary (infralens.in) describes 200 m stones between km stones.
  **Not verified:** the exact inscription rules, and whether hectometre (100 m) stones are prescribed
  for Indian roads.
  Sources: <https://morth.gov.in/sites/default/files/comprehensive_compendium_circular/603.10-28.01.1987.pdf>,
  <https://morth.gov.in/sites/default/files/comprehensive_compendium_circular/603.17-25.10.2016-Setting%20up%20and%20Kilometer%20Stones%20%20on%20National%20Highways%20dt%20on%2025th%20October%202016..pdf>,
  <https://infralens.in/code/IRC-26>.
- **Direction and lanes.** The NHIDCL NSV template uses Direction "Increasing (chainage)" or "Decreasing
  (chainage)", and "Lane Number: L1, L2, R1, R2, etc., L1 being 1st lane on the left from centreline of
  carriageway, L2 being 2nd lane on the left from centerline". LHS/RHS is relative to the direction of
  increasing chainage. Source: NHIDCL, *EOI for NSV Survey of NH-10*, Annexure I (Output Format from
  NSV), pp. 70–73: <https://www.nhidcl.com/sites/default/files/tender/nsv_rfp_merged.pdf>.
- **NSV reporting interval.** The NHIDCL Annexure I sample tables show:
  - Visual condition (ravelling, potholes, edge break, cracking, disintegration, depression, bleeding,
    patching, drain, shoulder): per lane in **500 m** rows (0.000–0.500, 0.500–1.000, …), each on a
    **1–5 scale**. Examples: cracking % area "1 Very Poor (>30%) … 5 Very Good (<5%)"; ravelling
    "1 (>30%), 2 (11–30%), 3 (6–10%), 4 (1–5%), 5 (0%)"; potholes by **count**, "1 (>5), 2 (3–5), 3 (2),
    4 (1), 5 (0)"; patching "1 (>30%), 2 (16–30%), 3 (6–15%), 4 (2–5%), 5 (<2%)".
  - IRI: per **100 m** (left/right wheel path and lane IRI).
  - Rutting: in mm, left/right/average, per **500 m**.
  - Surveys: every six months.

  This is an NHAI/NHIDCL data template, **not IRC:82**. Its bands differ from Table 5.1. The **interval**
  is inferred from the sample rows; the text gives no rule.
  NHAI news (Oct 2025): NSV on 20,933 km, data to NHAI "Data Lake"/RAMS. Sources:
  <https://www.businesstoday.in/amp/industry/infra/story/nhai-to-deploy-ai-survey-vehicles-on-20000-km-highways-for-real-time-monitoring-499215-2025-10-22>.
  The JICA manual gives NSV accuracy targets (cracks ≥2 mm detectable, rut ±5 mm):
  <https://openjicareport.jica.go.jp/pdf/12113015_07.pdf>.
- **Other IRC documents named in the brief.** I did not obtain the texts of IRC:SP:19 (project
  preparation), IRC:SP:16 (roughness) or IRC:SP:20 (rural roads). Nothing in this note depends on them;
  verify before citing them in the paper.

### 2.2 GPS-based chainage in practice [PRACTICE / INFERENCE]

- **Linear referencing.** Chainage is the distance along a reference centreline (polyline) from a fixed
  origin. A GPS fix is turned into chainage by projecting it onto the polyline (nearest point, with a
  heading check) and reading the cumulative length at that point. This is standard LRS practice: it is
  how NSV/RAMS geo-reference data. Path length along a raw GPS track is **not** the same thing: GNSS
  jitter makes the cumulative distance longer than the true distance.
- **Smartphone GNSS error.** gps.gov states that GPS smartphones are "typically accurate to within a
  4.9 m (16 ft.) radius under open sky", and worse near buildings, bridges and trees. The page notes it
  has not been updated since Feb 2025. Bengaluru streets have urban canyons and tree cover. Expect
  5–15 m errors along the road [INFERENCE], i.e. one to three of our 5 m segments.
  Source: <https://www.gps.gov/systems/gps/performance/accuracy/>.
- **Map matching.** Snapping to a road network with an HMM (Newson & Krumm, 2009, ACM SIGSPATIAL) is
  the usual way to handle noisy tracks and junctions. This is from recall; the 2009 citation is
  standard but was not re-checked here.
- **What our data has now.** In `GeoAI_Video/pipeline/geoai/segment_records.py`, chainage =
  `start_chainage_km ± (s − origin)/1000`, with the sign taken from the survey form's
  `increasing_chainage` / `decreasing_chainage`. It is null when there is no start chainage or direction.
  `s` is the camera path distance. This is **drive chainage**: it equals road chainage only if the survey
  form's start chainage was read from a km stone at the drive's first frame.

---

## 3. PCI methods elsewhere [PRACTICE]

### 3.1 ASTM D6433 (roads and parking lots), structure only

- **Sample units.** The section is divided into sample units. My recollection for asphalt roads is
  ~2,500 ± 1,000 ft² (≈ 225 ± 90 m²). Wikipedia says "approximately 5000 square feet", which may be the
  airfield standard D5340. **Unverified; check the standard.**
- **Survey.** About 20 asphalt distress types (Wikipedia lists 19, including alligator cracking,
  potholes, rutting, weathering/ravelling). Each has Low/Medium/High severity, and its quantity is
  measured as area, length or count.
- **Density** = quantity ÷ sample-unit area × 100.
- **Deduct values.** Each (type, severity, density) gives a **deduct value (DV)** from published curves.
  The DVs are combined iteratively: the allowable number of deducts is
  m = 1 + (9/98)(100 − HDV) [recall, unverified]; q = the number of deducts > 2; then the **corrected
  deduct value (CDV)**.
- **Result.** PCI = 100 − max CDV, on a 0–100 scale.
- **Classes** (Wikipedia): 86–100 Good; 71–85 Satisfactory; 56–70 Fair; 41–55 Poor; 26–40 Very Poor;
  11–25 Serious; 0–10 Failed.
- **Why we should not reproduce it.**
  - The deduct curves exist only as figures in the copyrighted, paid ASTM standard. Wikipedia: "The
    ASTM does not include the formulae of these curves, but they are recalculated by researchers."
    Digitised curves are unofficial re-creations of licensed content.
  - The method needs **measured area/length and severity per distress per sample unit**, which we do
    not have.
  - The user asked for IRC compliance. IRC:82 has its own rating.

  Source: <https://en.wikipedia.org/wiki/Pavement_condition_index>.

### 3.2 Indian indices

| Index | Scale | Basis | Status for us |
|---|---|---|---|
| **IRC:82 distress-based rating** (Tables 5.1–5.3 + App. 4) | 1–3 (Poor/Fair/Good) | % distress per km, rut mm | **Use this**: it is the IRC-specified condition rating |
| PMGSY PCI (OMMAS) | 1–5 (4–5 Good, 2–3 Fair, 1 Poor) | visual; method in the PMGSY Operations Manual (not obtained) | rural roads only; method unverified |
| NHAI/NHIDCL NSV visual ratings | 1–5 per distress | % area / pothole count per 500 m per lane | data template, not a combined index |
| Roughness (IRC:SP:16; Tables 5.4/5.5) | mm/km (BI) or IRI m/km | instrumented | not measurable from our video |
| Benkelman (IRC:81) / FWD (IRC:115) | deflection | structural | out of scope |

### 3.3 Image/video-based PCI literature [PRACTICE]

- **Majidifard, Adu-Gyamfi & Buttlar (2020).** "Deep machine learning approach to develop a new asphalt
  pavement condition index", *Construction and Building Materials*, doi:10.1016/j.conbuildmat.2020.118513
  (arXiv:2004.13314).
  - Data: 7,237 Google Street View images in 9 distress classes.
  - Method: YOLO detects and classifies the distress, and a U-Net segmentation estimates **density**.
    The two feed an index that is validated against **PASER**, not against an ASTM PCI.
  - Relevance: this is the closest analogue to our forward-view imagery. It needed a segmentation
    model to get extent; detection alone does not give extent.
  - Source: <https://arxiv.org/abs/2004.13314v1>.
- **YOLOv8 PCI from dash-cam imagery (2026).** "AI-Driven Pavement Condition Assessment from Dash-Cam
  Imagery: A Comparative Analysis of YOLOv8-Based PCI Estimation, Manual Inspections, and Automated PASER
  Ratings in Urban Networks", *Infrastructures* 11(6):207, doi:10.3390/infrastructures11060207.
  - Results (from the abstract via search; full text returned 403): Pearson r = 0.962 and
    Spearman 0.955 against manual PCI, with r² = 0.86. The automated score runs ~1.5 PCI points high.
    Distress detection accuracy was 84.6% over 4 classes (longitudinal, transverse, alligator, pothole).
  - Stated limitation: 2-D images cannot capture crack width or pothole depth.
  - Agreement with a commercial PASER product was "substantially lower".
  - **Not verified:** authors, sample size, how area was approximated.
- **Xu, Zhao, Loney, Li & Visentin (2025).** "Zero-Shot Image-Based Large Language Model Approach to Road
  Pavement Monitoring", arXiv:2504.06785.
  - Method: LLM prompts aligned to the Irish PSCI 1–10 scale, applied to Google Street View images.
  - Results: claims high agreement with official PSCI, "even surpassing expert evaluations"; no
    numbers on the abstract page.
  - Relevance: the nearest analogue to our VLM approach. It rates a *visual class* per image rather
    than computing area-based densities.
  - Source: <https://arxiv.org/abs/2504.06785>.

**Takeaway [INFERENCE].** Published image-based PCI work either:

- (a) segments distress pixels and assumes a camera geometry to get area/density, or
- (b) rates each image directly on a visual scale (PASER/PSCI).

None of these studies gets depth or width from a forward view; that limitation is stated in the
Infrastructures paper. Our pipeline gives **presence per 5 m segment**, plus an image-judged
severity. That supports **extent-along-length**, not area.

---

## 4. What a road-engineer's condition report contains [IRC + PRACTICE]

A condition report should contain the items below. Items marked IRC come from Proforma 1/2; the rest
are practice or inference.

1. **Header:**
   - Name and Number of the Road (IRC P1/P2); road category and therefore the IRC table used
     (5.1 / 5.2 / 5.3);
   - Chainage of Test Section "Km … to Km …" (IRC P1), with the lat/lon of points A and B;
   - direction (Increasing/Decreasing chainage) and lane surveyed (L1/R1 …) [NSV];
   - Carriageway Width (m) (IRC P1; "not measured" if unknown);
   - Type of Surface (IRC P1);
   - Date(s) of Observation (IRC P1, one line per drive used);
   - Weather Condition (IRC P1);
   - survey method and vehicle/camera, with the model and pipeline version (P4, prompt id, config hash);
   - report generation date.
2. **Method statement:** how chainage was derived, the inspection-unit length, how each % was
   computed, which IRC parameters were **not assessed** and why.
3. **Section-wise table in Proforma 1 column order:** Chainage From | To | Cracks (%) | Patch Work (%) |
   Potholes (%) | Bleeding (%) | Depression (%) | Edge Breaking (%) | Ravelling (%) | Shoving (%) |
   Rut Depth (mm) | Drainage and Shoulder condition | Total. For our data, add "segments covered / expected".
4. **Rating table per km (or per section):**
   - for each IRC parameter: input %, band (Poor/Fair/Good), rating, weight, weighted rating;
   - Final Rating Value and Condition (App. 4 layout);
   - a strip chart along chainage.
5. **Recommended maintenance treatment** per section and per dominant distress/severity, quoting IRC:82
   clause numbers (§6 below).
6. **Photos:** a representative frame for each Poor/Fair sub-section and for every pothole, each with
   chainage, lat/lon and timestamp.
7. **Limitations and data-quality statement** (§5.4), including the model's measured recall.
8. **Sign-off:** "Recorded by" / "Checked by" (IRC P1/P2), plus an expert-review status per section.

---

## 5. RECOMMENDATION

### 5.1 Chainage for an A→B selection [INFERENCE, built on §2]

1. **Build one reference line per road.** Use the road's centreline from the GIS layer, OSM or a
   cleaned reference drive. Smooth and densify it to 1 m and store its cumulative length. Every drive
   on that road is referenced to this line, not to its own track. Path length along a raw GPS track
   grows with jitter, and two drives on the same road would otherwise give two different chainages.
2. **Snap A and B** (the user's clicks) to the reference line by perpendicular projection:
   - this gives `m_A` and `m_B`, the measures along the line;
   - if `m_B < m_A`, the selection runs against the line's direction: swap the ends for the
     arithmetic and set direction = Decreasing;
   - reject a click that is more than ~25 m from the line, so the user picks a point on the road.
3. **Snap each 5 m segment.** Project the segment's GPS centre onto the line, with a heading check:
   the segment bearing must agree with the line's bearing within ~45°, or with its reverse, which marks
   the opposite carriageway. That gives `m_seg`. Segment extent = `m_seg ± length_m/2`. Use `s0_m`/`s1_m`
   only for the length, never as the road measure.
4. **Report chainage in two forms:**
   - **Relative chainage:** "Km 0+000 at A" up to "Km x+yyy at B" (= `m − m_A`). This is always
     available.
   - **Road chainage:** `start_chainage + (m − m_origin)/1000`, given *only* when a km-stone origin has
     been calibrated, i.e. the survey form's `start_chainage_km` read at a km stone, or a stone's mapped
     location. Write it as "Km 12+350" and also in decimal "12.350" (NHAI style). Do not present GPS
     chainage as official km-stone chainage when no calibration exists.
5. **Resolution and error.** Print chainage to the metre. State the accuracy as **±10 m** in the report:
   smartphone GNSS gives ~5 m in open sky and worse under trees or between buildings, plus the
   segment's own 5 m length. Recommend recalibrating at each passed km stone, which removes the
   cumulative error.
6. **Overlapping drives.** If several drives cover the same metres in the same direction, prefer the
   latest drive (the condition of the road on its survey date). Show the survey date per section. Never
   average across dates; condition changes, especially across a monsoon.
7. **Coverage.** For each section, coverage = covered metres ÷ section metres, counting only usable
   (classified or reviewed) segments. Mark a section **"insufficient data"** when coverage < 80%. The
   80% threshold is my choice, not IRC; it must be stated.

### 5.2 Inspection unit [INFERENCE grounded in IRC + practice]

- **The IRC unit is the kilometre.** §5.3 says "kilometer wise", and App. 1 says "for each km length".
  Use **km-wise rows** for the IRC-format Proforma-1 table and rating, with the A→B stretch truncated at
  its ends.
- **Add 100 m sub-sections** for the strip chart and for treatment location. Precedents:
  - NRRDA/ILO strip chart, columns 0+000…1+000 (100 m);
  - NSV IRI per 100 m;
  - App. 1 asks for "location" of distress.

  A 100 m sub-section holds 20 of our 5 m segments, so each segment = 5% of its length.
- **Rate the whole A→B stretch** as well, from its length-weighted percentages.
- **Aggregate quantities, not ratings.** A km's % = (affected metres in the km) ÷ (covered metres in
  the km). Rate that %. Do not average the ratings of the sub-sections. App. 4 rates quantities.

### 5.3 Per-section IRC rating from per-5 m labels [INFERENCE]

**Road category → table.** Bengaluru city streets use **Table 5.3 (Urban)**. Let the operator pick
Highway (5.1) or MDR/Rural (5.2) per road, and print the table used.

**Map our labels to IRC parameters.** Use the union per segment: a segment counts once per parameter.

| IRC rating parameter | Our IRC:82 labels that count | Measurable? |
|---|---|---|
| Cracking % | Alligator, Longitudinal, Transverse, Edge, Hairline, Reflection cracking (Block → transverse per our mapping) | extent along length: yes; area: no |
| Ravelling % | Ravelling (Stripping and Hungry Surface reported separately, not counted) | extent along length |
| Potholes % | Pothole | presence and approximate count; area: no |
| Patching % (Tables 5.1/5.2 only) | Patching (condition indicator) | extent along length |
| Settlement/Depression % | Settlement, Shallow Depression, Upheaval | weak: depth not visible; flag low confidence |
| Shoving % (Table 5.1 only) | Shoving | extent along length; rare |
| Rut depth (mm) | Rutting | **Not assessed.** The image severity is not a 3 m straight-edge measurement |
| (P1 columns, not rated) Bleeding %, Edge Breaking % | Bleeding/Fatty Surface, Edge Breaking | report only |

**Estimator.** For each section and parameter p:

```
E_p (% length affected) = 100 × Σ length_m of usable segments flagged with p
                               ÷ Σ length_m of usable segments in the section
```

1. **Use E_p as the input "%" in Tables 5.1–5.3.** Label it everywhere as **"estimated: % of surveyed
   length showing the distress"**.
   - The rationale: it is the same extent-by-eye judgement that IRC's visual Appendix-1 survey makes
     from a slow-moving vehicle, but at a fixed 5 m granularity.
   - **It is an upper bound on % area** for localised distresses: one pothole flags a whole 5 m × lane
     segment. On Table 5.3, any pothole segment already puts the pothole parameter in Fair or Poor,
     whatever its area.
2. **Show a lower-bound sensitivity for potholes**, the parameter where the gap between length and area
   is largest:

   ```
   A_low = n_pothole_segments × 0.0314 m²      (π·0.1², the IRC "small" 200 mm pothole, §7.5.3.4)
   %_low = A_low ÷ (section length × lane width)
   ```

   This needs a lane width; use 3.5 m if the carriageway width is unknown and say so.
   - Report the pothole class at both bounds.
   - If the two bounds disagree, print "Fair–Poor (area not measured)".
   - Do the same for patching and settlement on the tables that use them, if wanted.
   - Cracking and ravelling are usually extensive along length; E_p alone is acceptable, with the
     label above.
3. **Band lookup.** Use the exact boundaries of the chosen table:
   - "5 to 10" includes 5 and 10;
   - "<5" is Good;
   - ">10" is Poor;
   - urban potholes: NIL = Good, >0 and ≤0.5 = Fair, >0.5 = Poor.
4. **Rating value.** Interpolate per §1.5 (truncate to 1 decimal) and label it "interpolated (Appendix-4
   example)".
5. **Weights and final value.** Weights are from App. 4: cracking 1.00, ravelling 0.75, potholes 0.50,
   shoving 1.00, patching 0.75, settlement 0.75, rut depth 1.00. Final value = **mean of wᵢ·rᵢ over
   the parameters assessed**, as App. 4 specifies.
   - Urban: cracking, ravelling, potholes and settlement assessed; rut depth not assessed, so n = 4.
     Settlement is low confidence: offer a second line without it (n = 3).
   - State in the report: "Rut depth (Table 5.3) not assessed; final value averaged over N of 5
     parameters".
   - Show Σwr/Σw as a sensitivity line only.
   - Classify: final ≤ 1.0 → Poor; 1.1–2.0 → Fair; ≥ 2.1 → Good. A value between 1.0 and 1.1 can only
     arise from rounding; round to 1 decimal first.
6. **Severity.** Do not use severity in the rating; the Tables 5.x are extent-only. Use the image-judged
   severity (Low/Med/High, with the IRC mm thresholds *as judged from the image*) only to choose the
   treatment text in §6. Label it "visually judged severity, widths/depths not measured".
7. **Confidence and review state.**
   - Expert-reviewed labels override model labels.
   - A segment in `expert_review` without a review counts with its model labels, but the section shows
     "x of y segments unreviewed".
   - `rejected_non_pavement` and failed segments are excluded from the denominator and reduce coverage.
   - A Stage-1 "Normal" segment that the safety net flagged counts as distress-free, with a count of
     flagged segments shown.
8. **Every number printed carries "(est.)".** The final rating line reads:
   "IRC:82-2015 Table 5.3 distress-based rating, estimated from automated image classification — not an
   Appendix-1/2 survey".

### 5.4 Limitations the report must state [from our measured results + §1–3]

1. **Area.** IRC % inputs are distress area ÷ pavement area (implied by Proforma 2). We measure the
   **fraction of 5 m segments** in which the model saw the distress. This overstates localised
   distresses (potholes, patches) and has 5% resolution per 100 m.
2. **Not measured:**
   - rut depth (3 m straight edge);
   - crack width and pothole depth (severity is visually judged);
   - roughness and skid resistance (Tables 5.4/5.5);
   - carriageway width (unless entered);
   - drainage and shoulder condition.
3. **Detection error, measured on Attain with our production pipeline:**
   - Stage 1 recall on distress frames is **51.8%**, so cracking/ravelling % are likely
     **under**-estimated by missed frames;
   - the Stage 2 free-form list is often exactly "Longitudinal + Transverse Cracking", which can
     **over**-state cracking on frames that have other distress;
   - alligator cracking is named on **8%** of frames that have it, so fatigue-cracking treatments
     (§7.3.3.5, structural investigation) are under-triggered;
   - ravelling, weathering and patching were never named by the free-form list on Attain.

   The safety net and expert review mitigate this; they do not remove it. These numbers come from
   another country's dataset (Attain, Tehran); **no Bengaluru validation exists yet** (one expert label
   so far).
4. **One frame per 5 m, one lane, one direction.** Distress outside the camera's view or in other
   lanes is not seen; report the lane surveyed.
5. **Chainage.**
   - GPS-derived, ±~10 m;
   - relative to A unless calibrated to km stones;
   - the drive date can differ between sections.
6. **Not an IRC:82 survey.** IRC asks for surveys by experienced engineers (§5.3), with at least two a
   year on urban roads. The report is a **screening/prioritisation aid**. Treatment decisions need a
   field check (§7.3.3.5 itself asks for pits or cores for alligator cracking).
7. **Appendix-4 interpolation.** Inferred from IRC's worked example (6 of 7 rows reproduced). The
   Poor/Fair/Good class does not depend on it.

Suggested validation before the paper quotes any rating [INFERENCE]: have an expert label every 5 m
segment of ~10 randomly chosen 100 m sub-sections. Compare (a) E_p from expert labels with E_p from
model labels, and (b) the class agreement per parameter. Report it with the block-bootstrap CI
convention used elsewhere in the paper.

---

## 6. Treatment text per condition class and per distress (for the report generator) [IRC, paraphrased with clause refs]

### 6.1 By overall condition class

| Class (final rating) | Urban road (Table 5.3) | Highway (5.1) | MDR/Rural (5.2) | IRC basis |
|---|---|---|---|---|
| **Good (2.1–3.0)** | Routine maintenance (pothole filling, crack repair, patching, markings). Consider **preventive maintenance** (crack sealing, fog seal, slurry seal, microsurfacing, chip seal, thin surfacing, ultrathin friction course) so the rating does not drop below 2 | same | same | §4.1(a),(b); §5.3 p. 7; §8.2.4; §8.2.6; Tables 8.1–8.7 |
| **Fair (1.1–2.0)** | At rating ≈ 2: **periodic renewal**, suggested "30-40 BC/SDBC/2 Coat Micro Surfacing" (roughness trigger 2600 mm/km). Preventive treatments are effective only *before* the rating drops below 2; slurry seal/microsurfacing is not preventive on an extensively cracked surface | Renewal at 2: NH "30-40 BC", SH "30 BC/30 SDBC" | Renewal "when condition rating reaches … 2 to 1": "SD/PC/MSS/SDBC (20-25)" | §5.3 p. 7; §8.2.3; §8.2.4; Table 9.1 |
| **Poor (≤1.0)** | "detailed investigation … to assess the strengthening requirement". If structurally adequate, periodic renewal suffices; otherwise rehabilitation/strengthening per IRC:81 / IRC:115 / IRC:37 (outside IRC:82) | same | same | Table 9.1 footnote *; §2.4; §9.3 |
| Any class | Distress that "adversely influences the riding quality significantly" (e.g. potholes) "should be rectified straightway". Recycling of the wearing course is preferred where feasible | | | §5.3 p. 6; Table 9.1 ** |

### 6.2 By distress and visually judged severity

| Distress | Severity | Treatment (IRC:82) | Clause |
|---|---|---|---|
| Potholes | any (small/medium/large) | Fill with open- or dense-graded premix: mixes for immediate use, storable cold mixes (cutback IRC:116 / emulsion IRC:100), ready-made mixes, or patching machines. Cut the deteriorated edges first. Fill potholes deeper than 75 mm in layers of 25–75 mm | 7.5.3.5; 10.3.8; 11.4 |
| Alligator cracking | L/M/H | Find the cause (pit or core for structural adequacy and subsurface moisture). Crack sealing with bitumen emulsions, or with rubberised/modified bitumen; milling and surface recycling. If structurally inadequate: designed overlay | 7.3.3.5 |
| Longitudinal cracking | Low/Medium | Crack sealing, preferably with rubberised bitumen | 7.3.4.5(a) |
| Longitudinal cracking | High | Remove and replace the cracked layer with a fresh overlay | 7.3.4.5(b) |
| Transverse cracking | any | Slurry seal or rubberised bitumen | 7.3.5.5 |
| Hairline cracks | — | Fog seal, liquid rejuvenating agents, slurry seal, microsurfacing | 7.3.2.5 |
| Edge cracking | any | Improve edge drainage, replace shoulder material with permeable material; with heavy truck traffic, a 75–100 mm structural overlay | 7.3.6.5 |
| Reflection cracking / cracks generally | — | Structurally sound: fill with low-viscosity binder (cut-back, emulsion, rubberised bitumen per ASTM D5078); wide cracks: slurry seal or sand-bitumen premix; fine and extensive: fog seal; SAM/SAMI | 7.3.7.5; 10.3.9, Table 10.5 |
| Ravelling | Low | Fog seal, sand seal, seal coat, slurry seal or microsurfacing | 7.5.2.5(a) |
| Ravelling | Medium | Seal coat, slurry seal or microsurfacing in the affected area | 7.5.2.5(b) |
| Ravelling | High | Single or multiple liquid seal coat, slurry seal, microsurfacing, or a thin overlay (mix seal surfacing / open-graded premix carpet) | 7.5.2.5(c) |
| Hungry surface | — | Fog seal / liquid rejuvenators; slurry seal or microsurfacing 2–4 mm | 7.2.4.5 |
| Stripping | — | Hot coarse sand (≥150 °C) rolled in (surface dressing); liquid seal; else remove and relay; rejuvenating sealants, slurry seal, microsurfacing | 7.5.1.5 |
| Bleeding | Low | Heated coarse sand (passing 1.18 mm), light rolling | 7.2.1.5(a) |
| Bleeding | Medium | Pre-coated chips (4.75–1.18 mm, 1% bitumen), light rolling | 7.2.1.5(b) |
| Bleeding | High | Mill out and repair; RAP plus fresh aggregate, then slurry seal/microsurfacing; or open-graded premix carpet | 7.2.1.5(c) |
| Smooth surface | — | Surface dressing, premix carpet or microsurfacing with non-polishing aggregate; slurry seal | 7.2.2.5 |
| Rutting | — (depth not measured) | Fill ruts with premix and compact (tack coat first); microsurfacing fills ruts ≤ 40 mm | 7.4.2.5; 10.3.4 (microsurfacing) |
| Corrugation | — | Scarify and recompact, then new surfacing; or cut high spots / sand-bitumen premix with a drag spreader | 7.4.3.5 |
| Shoving | — | Remove down to a firm base and lay a stable premix patch | 7.4.4.5 |
| Slippage | L/H | Remove the surface around the area to well-bonded material; tack coat; premix patch | 7.4.1.5 |
| Shallow depression | — | Fill with premix and compact to profile | 7.4.5.5 |
| Settlement / upheaval | — | Excavate a defective fill and rebuild; under-drains where drainage failed; designed pavement if thickness is inadequate | 7.4.6.5 |
| Edge breaking | — | Remove to a regular section; rebuild pavement and shoulder together; shoulder slope 1% steeper than camber; improve drainage | 7.5.4.5 |
| Streaking | — | Remove the streaked surface and apply a new surface | 7.2.3.5 |
| Patching (condition indicator) | — | No treatment of its own; counts towards the Table 5.1/5.2 patching parameter | Tables 5.1–5.2 |

---

## 7. Sources

**IRC and Indian agencies**

- IRC:82-2015, Indian Roads Congress (local PDF). Clauses and pages as cited above.
- NRRDA (MoRD), *Managing Maintenance of Rural Roads — Guidance Note for States*, Oct 2014, Table 1 and
  the road condition inventory form:
  <https://cdnbbsr.s3waas.gov.in/s3d9ff90f4000eacd3a6c9cb27f78994cf/uploads/2025/02/20250227831088188.pdf>
- NHIDCL, *EOI for NSV Survey of NH-10 Sevoke–Rangpo, Km 0.000 to 52.100*, Dec 2024, Annexure I (Output
  Format from NSV): <https://www.nhidcl.com/sites/default/files/tender/nsv_rfp_merged.pdf>
- MoRTH circular 603.10 (28-01-1987), on km stones per IRC:8-1980:
  <https://morth.gov.in/sites/default/files/comprehensive_compendium_circular/603.10-28.01.1987.pdf>
- MoRTH circular 603.17 (25-10-2016), on setting up km stones on NHs:
  <https://morth.gov.in/sites/default/files/comprehensive_compendium_circular/603.17-25.10.2016-Setting%20up%20and%20Kilometer%20Stones%20%20on%20National%20Highways%20dt%20on%2025th%20October%202016..pdf>
- IRC:26 summary (secondary): <https://infralens.in/code/IRC-26>
- NHAI NSV deployment (news, Oct 2025):
  <https://www.businesstoday.in/amp/industry/infra/story/nhai-to-deploy-ai-survey-vehicles-on-20000-km-highways-for-real-time-monitoring-499215-2025-10-22>
- JICA NSV manual: <https://openjicareport.jica.go.jp/pdf/12113015_07.pdf>

**GNSS, PCI and literature**

- gps.gov, *GPS Accuracy*: <https://www.gps.gov/systems/gps/performance/accuracy/>
- Wikipedia, *Pavement condition index*: <https://en.wikipedia.org/wiki/Pavement_condition_index>
- Majidifard, Adu-Gyamfi, Buttlar (2020), CBM, doi:10.1016/j.conbuildmat.2020.118513:
  <https://arxiv.org/abs/2004.13314v1>
- *Infrastructures* 11(6):207, doi:10.3390/infrastructures11060207 (abstract via search only)
- Xu et al. (2025), arXiv:2504.06785: <https://arxiv.org/abs/2504.06785>

**Not verified:**

- the ASTM D6433 sample-unit size and allowable-deduct formula (from recall);
- the IRC:26 / IRC:8 inscription details;
- IRC:SP:16, SP:19 and SP:20 contents;
- the PMGSY Operations Manual PCI method;
- the Newson & Krumm citation details;
- the authors and sample size of the *Infrastructures* 2026 paper.
