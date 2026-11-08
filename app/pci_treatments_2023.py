"""
IRC:82-2023 maintenance text for the A->B condition report (PDF): Table 5.5 by PCI class, Table 10.1 renewal
and the distress-specific treatments of Section 7. Every clause number and wording was checked against the
extracted 2023 text (C:\\GeoAI\\pci\\irc82_2023\\IRC82-2023.txt, 2026-11-08); texts are short paraphrases.

    TABLE_5_5                      [(class, PCI range text, recommendation)]
    class_recommendation(cls)      the Table 5.5 recommendation of one class
    renewal_note(road_class)       Table 10.1 row + its note on PCI < 60
    distress_rows(labels, worst)   [{label, severity, treatment, clause}] for the labels found
"""
from __future__ import annotations

from typing import Iterable, Mapping, Optional

# Table 5.5 (p. 15). The printed "Renewal (as per section 9.5)" points to a section the 2023 text does not have;
# periodic renewal is Section 10 (Table 10.1), so the report cites §10.
TABLE_5_5 = (
    ("Excellent", ">90 to 100", "Routine maintenance"),
    ("Good", ">80 to 90", "Preventive maintenance"),
    ("Satisfactory", ">60 to 80", "Renewal (periodic renewal, §10)"),
    ("Fair", ">40 to 60", "Minor rehabilitation (based on structural evaluation)"),
    ("Poor", ">20 to 40", "Major rehabilitation / structural overlay"),
    ("Fail", "0 to 20", "Reconstruction"),
)
CLASS_ORDER = tuple(c for c, _, _ in TABLE_5_5)

# Table 10.1 (p. 57): suggested renewal at PCI 60-80, by road category
RENEWAL_10_1 = {
    "highway": "NH: 30-40 mm BC; SH: 30 mm BC / 30 mm SDBC",
    "mdr": "SD / PC / MSS / SDBC, 20-25 mm",
    "rural": "SD / PC / MSS / SDBC, 20-25 mm",
    "urban": "30-40 mm BC / SDBC, or 2-coat microsurfacing",
}


def class_recommendation(cls: Optional[str]) -> Optional[str]:
    for c, _r, rec in TABLE_5_5:
        if c == cls:
            return rec
    return None


def renewal_note(road_class: str) -> str:
    return (f"Table 10.1 (renewal at PCI 60-80): {RENEWAL_10_1.get(road_class, RENEWAL_10_1['urban'])}; recycling "
            "of the wearing course preferred where feasible. Below PCI 60: detailed investigation of the "
            "strengthening requirement first; if structurally adequate, periodic renewal suffices.")


# Section 7 treatments (clause = the 2023 "Treatment" sub-clause). Severity keys are the labels' visually judged
# severity (Low / Medium / High); "any" applies to all.
TREATMENT: dict[str, dict] = {
    "Potholes": {"clause": "7.5.3.5, 13.4",
                 "any": "Cut the deteriorated portion and remove loose material, then fill with open- or "
                        "dense-graded premix: hot/cold mix for immediate use, storable cold mix, ready mix, or "
                        "mechanised patching"},
    "Alligator Cracking": {"clause": "7.3.3.5",
                           "any": "Find the cause first (test pit or core: structural adequacy, subsurface "
                                  "moisture). Crack sealing with bitumen emulsion or rubberised/modified bitumen, "
                                  "or milling and surface recycling; a designed overlay if structurally weak"},
    "Longitudinal Cracking": {"clause": "7.3.4.5",
                              "Low": "Crack sealing, preferably with rubberised bitumen",
                              "Medium": "Crack sealing, preferably with rubberised bitumen",
                              "High": "Remove the cracked layer and replace it with a fresh overlay"},
    "Transverse Cracking": {"clause": "7.3.5.5", "any": "Slurry seal or rubberised bitumen"},
    "Hairline Cracks": {"clause": "7.3.2.5",
                        "any": "Fog seal, liquid rejuvenating agent, slurry seal or microsurfacing"},
    "Edge Cracking": {"clause": "7.3.6.5",
                      "any": "Improve edge drainage; replace poor shoulder material with permeable material; "
                             "with substantial truck traffic a 75-100 mm hot-mix structural overlay"},
    "Ravelling": {"clause": "7.5.2.5",
                  "Low": "Fog seal, sand seal or seal coat, slurry seal or microsurfacing",
                  "Medium": "Seal coat, slurry seal or microsurfacing over the affected area",
                  "High": "Liquid seal coat(s), slurry seal, microsurfacing, or a thin overlay (mix seal "
                          "surfacing / open-graded premix carpet)"},
    "Patching": {"clause": "7.5.4.3",
                 "any": "Deteriorated patches: full-depth or deep (≥ 100 mm) patch repair; a thin surface patch "
                        "(< 25 mm) is only temporary"},
    "Rutting": {"clause": "7.4.2.5",
                "any": "Fill the ruts with open- or dense-graded premix over a tack coat and compact to level"},
}
TREATMENT["Block Cracking"] = dict(TREATMENT["Transverse Cracking"])      # IRC 7.3.5.1: transverse "large blocks"

# Distresses outside the IRC:82-2023 PCI (reported only; labels unvalidated on Bengaluru video)
OTHER_TREATMENT: dict[str, dict] = {
    "Bleeding": {"clause": "7.2.1.5",
                 "Low": "Heated coarse sand (passing 1.18 mm) to blot excess bitumen, light rolling",
                 "Medium": "Pre-coated chips (4.75-1.18 mm, 1 % bitumen), single layer, light rolling",
                 "High": "Mill out and repair; RAP with fresh aggregate then slurry seal / microsurfacing, or "
                         "low-bitumen open-graded premix carpet"},
    "Edge Breaking": {"clause": "7.5.5.6",
                      "any": "Cut to a regular section; rebuild pavement and shoulder together with thorough "
                             "compaction; shoulder slope 1 % steeper than camber; improve drainage"},
    "Hungry Surface": {"clause": "7.2.4.5",
                       "any": "Fog seal / liquid rejuvenator; or slurry seal / microsurfacing 2-4 mm"},
    "Streaking": {"clause": "7.2.3.5", "any": "Remove the streaked surface and apply a new surface"},
    "Stripping": {"clause": "7.5.1.5",
                  "any": "Surface dressing: hot coarse sand (≥ 150 °C) rolled in, or a liquid seal; else remove and "
                         "relay the mix; rejuvenating sealant, slurry seal or microsurfacing"},
    "Slippage": {"clause": "7.4.1.5",
                 "any": "Remove the surface to well-bonded material; tack coat; premix patch"},
    "Corrugation": {"clause": "7.4.3.5",
                    "any": "Scarify and re-compact, then new surfacing; or cut high spots / sand-bitumen premix "
                           "with a drag spreader, then roll"},
    "Rutting": TREATMENT["Rutting"],
    "Shoving": {"clause": "7.4.4.5", "any": "Remove to a firm base and lay a stable premix patch"},
    "Shallow Depression": {"clause": "7.4.5.5", "any": "Fill with premix and compact to the surrounding profile"},
}
OTHER_ORDER = ("Bleeding", "Edge Breaking", "Hungry Surface", "Stripping", "Streaking", "Slippage", "Corrugation",
               "Rutting", "Shoving", "Shallow Depression")


def other_treatment(label: str, severity: Optional[str] = None) -> Optional[dict]:
    t = OTHER_TREATMENT.get(label)
    if not t:
        return None
    text = t.get(severity) if severity in t else (t.get("any") or t.get("Medium"))
    return {"treatment": text, "clause": t["clause"]}

# Labels whose treatment is printed only when that parameter has a value in the report
PARAM_OF = {"Potholes": "potholes", "Alligator Cracking": "cracking", "Longitudinal Cracking": "cracking",
            "Transverse Cracking": "cracking", "Hairline Cracks": "cracking", "Edge Cracking": "cracking",
            "Block Cracking": "cracking", "Ravelling": "ravelling", "Patching": "patching", "Rutting": "rut_depth"}
ORDER = ("Potholes", "Alligator Cracking", "Longitudinal Cracking", "Transverse Cracking", "Block Cracking",
         "Edge Cracking", "Hairline Cracks", "Ravelling", "Patching", "Rutting")


def distress_rows(labels: Iterable[str], worst: Optional[Mapping[str, str]] = None) -> list[dict]:
    """One row per label found, in ORDER; severity = the worst visually judged severity (or 'not judged')."""
    worst = worst or {}
    have = set(labels)
    out = []
    for lab in ORDER:
        if lab not in have:
            continue
        t = TREATMENT[lab]
        sev = worst.get(lab)
        text = t.get(sev) if sev in t else (t.get("any") or t.get("Medium"))
        out.append({"label": lab, "severity": sev or "not judged", "treatment": text, "clause": t["clause"]})
    return out
