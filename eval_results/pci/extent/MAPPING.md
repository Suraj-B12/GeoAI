# Which distress feeds which IRC:82-2023 parameter (extent calibration)

Source text: `C:\GeoAI\pci\irc82_2023\IRC82-2023.txt` (the third-party hosted copy described in
`eval_results/pci/IRC82_2023.md`; the owner should confirm against an official IRC copy). Clause numbers below are
from that copy. Written 2026-11-09 before any fit.

## 1. What IRC:82-2023 counts in each parameter

| Parameter (§5.1, Table 5.4 weight) | What the code says | Clause |
|---|---|---|
| **Cracking extent %** (0.12) | Appendix-1 (manual survey list, starred = "used for calculation of PCI"): "Cracking (Alligator or fatigue cracking, Longitudinal and transverse cracking)". §7.1 (ii) lists the crack family: hair-line, alligator, longitudinal, edge, shrinkage, reflection. §7.3.5.1: transverse cracks include "interconnected cracks forming series of large blocks" (block cracking is filed under transverse). | App-1 item 1; §7.1 (ii); §7.3.3–7.3.7 |
| **Ravelling extent %** (0.10) | §7.5.2.1: "progressive separation and dissociation of fine aggregate particles and binder from the bituminous surface". §7.5.2.4 low severity = "some loss of fines associated with initial stage of binder wearing out". **Stripping** (§7.5.1) and **hungry surface** (§7.2.4, a *surface defect*: "loss of fine aggregates from the surface or the appearance of dry surface with fine cracks") are separate defects; App-1 stars only "Ravelling". | §7.5.2, §7.5.1, §7.2.4, App-1 item 6 |
| **Patching extent %** (0.08) | §7.5.4 "Patch/Patch Deterioration": repaired damaged or excavated areas; "applications of sealant without aggregate are not to be recorded as patches" (§7.5.4.3). | §7.5.4, App-1 item 5 |
| **Pothole number** (0.16) | eq. (2): number = pothole surface area (m²) / 0.1 m²; §7.5.3.4 "pothole unit" = 0.1 m² (≈ 300 mm diameter). | §5.2 eq. (2), §7.5.3.4 |
| Extent basis | eq. (1): extent = area affected by the distress / surface area of the pavement section; "surface area" = all lanes of one carriageway, 100–1000 m. | §5.2 eq. (1) and note; note under Table 5.3 |

## 2. Decisions

1. **Cracking = alligator + longitudinal + transverse, block cracking included (as transverse, §7.3.5.1).**
   Edge cracking and hairline cracks are cracks (§7.1 ii) but are not in the App-1 PCI list; Attain does not
   label them, so their effect cannot be measured here. They are NOT used as probe keys in the calibration
   (sensitivity only, not fitted).
2. **Ravelling = Attain "Raveling" only.** IRC:82-2023 files hungry surface (§7.2.4) and stripping (§7.5.1)
   as separate defects; App-1 stars only ravelling for the PCI. Attain "Weathering" (loss of fines / aged binder)
   matches §7.2.4 hungry surface — the mapping the project has used since 2026-09 (CLAUDE.md, Attain table). It
   overlaps §7.5.2.4 "low severity" ravelling in wording, so a **sensitivity** target
   `ravelling_incl_weathering` (Raveling ∪ Weathering) is computed and reported, never used for the PCI.
   On Attain it changes the mean extent from 0.46 % to 3.7 % of the road ROI: the choice matters a lot.
3. **Patching = Attain "Patch and utility cut"** (parsed with the no-space trap `'Patch and utility cut- Low'`).
4. **Potholes = Attain "Pothole"**: area fraction of the box union, box count and connected-component count
   per frame. The IRC number needs metric area; see §4.
5. **Excluded:** Faded marking, Lane shoulder drop-off, Manhole (road features, not distress). They are used
   only to define the `box_hull` ROI variant.
6. **Rut depth and IRI** are not image extents; not part of this calibration.

## 3. Our model outputs per parameter (probe keys and Stage 2 labels)

The per-type probe (`configs/stage2_probe.json`, variant `min:name`, full photo) gives P(yes) for 20 IRC types
on every frame. The feature of a parameter is the **largest P(yes) over its keys** (the same rule as
`app/pci_core.PROBE_KEYS` and the probe report).

| Parameter | Probe keys (feature = max P(yes)) | Stage 2 free-form labels considered (all labels of the list, not only the first) |
|---|---|---|
| cracking | Alligator Cracking, Longitudinal Cracking, Transverse Cracking, Block Cracking | any of Alligator / Longitudinal / Transverse / Edge Cracking, Hairline Cracks anywhere in `distress_types` |
| ravelling | Ravelling | Ravelling anywhere in the list |
| ravelling_incl_weathering (sensitivity) | Ravelling, Hungry Surface | Ravelling or Hungry Surface |
| patching | Patching | none (the list never carries Patching; it is a condition indicator, never produced on Attain) |
| potholes | Potholes | Potholes anywhere in the list |

`app/pci_core.PROBE_KEYS["cracking"]` also lists Edge Cracking and Hairline Cracks; this calibration does not
(decision 1). The label flag is a candidate second feature; whether it is used is decided on DEV only by the
pre-registered selection rule (PREREG.md).

## 4. Area basis

- Ground truth = union area of the parameter's boxes inside a road ROI / ROI area (pixels). Boxes over-state the
  area of thin linear cracks; IRC crack extent is the AFFECTED area (the cracked zone, §5.2 eq. 1 "area of pavement
  section affected"), which a box around a crack run approximates — it is a proxy, stated as such.
- Pixel-area fraction is an unbiased estimate of ground-area fraction only if distress is spread uniformly over
  the visible road (near rows hold fewer m² per pixel than far rows). Annotation completeness falls with distance
  (far defects are less often boxed), so the Attain truth leans to under-statement.
- Attain annotations are incomplete in places (seen by eye in roi_check/: some visible cracks unboxed); the
  target is "annotated area", a lower-leaning proxy of true area.
