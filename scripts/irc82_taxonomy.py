"""
IRC:82-2015 compliant pavement distress taxonomy and severity criteria.

Single source of truth for all distress classification in this project.
All output labels, severity ratings, and report names align to the
"Code of Practice for Maintenance of Bituminous Road Surfaces (First
Revision)" published by the Indian Roads Congress, June 2015.

PDF source: IRC 82/irc.gov.in.082.2015.pdf
Verified against PDF Section 7.2-7.5 (pages 9-32 in the document numbering).

Scope: PURE VISION-ONLY subset (Option B).
  Excludes Smooth Surface (7.2.2 — severity needs skid number measurement)
  Excludes Reflection Cracking (7.3.7 — requires knowledge of underlying layer)

Categories (per IRC:82 Section 7.1):
  - Surface Defects     (Section 7.2)
  - Cracks              (Section 7.3)
  - Deformation         (Section 7.4)
  - Disintegration      (Section 7.5)

Each entry includes:
  irc_section:   the exact subsection number in IRC:82-2015
  description:   what the distress IS (symptoms from the IRC text)
  visual_cues:   what to look for in a photograph
  severity:      L/M/H criteria per IRC, or None if IRC declares
                 "severity not applicable" for that type
  severity_note: how to estimate visually when the IRC criterion uses mm
  treatment_ref: section number for the IRC-prescribed treatment
  legacy_codes:  prior labels in our pipeline (RDD codes etc) that map here
"""

from __future__ import annotations
from typing import Optional


# ============================================================
# The IRC:82-2015 taxonomy — verified entries
# ============================================================

IRC82_DISTRESS_TAXONOMY: dict[str, dict] = {

    # -------------------- 7.2 SURFACE DEFECTS --------------------

    "Bleeding": {
        "category": "Surface Defects",
        "irc_section": "7.2.1",
        "description": (
            "A surface having a thin film of excess or free bituminous binder, "
            "creating a shiny, glass-like reflecting surface. Becomes soft in "
            "hot weather and slippery when wet."
        ),
        "visual_cues": (
            "Shiny / glossy / mirror-like patches on the pavement. Often "
            "concentrated along wheel paths. Black, wet-looking even when dry."
        ),
        "severity": {
            "Low":    "Isolated spots < 5 m² per lane-km, < 1% of lane area",
            "Medium": "Affected area 1-5% of lane",
            "High":   "Extensive bleeding showing initiation of shoving",
        },
        "severity_note": "Estimate area coverage visually; default to higher tier if uncertain.",
        "treatment_ref": "7.2.1.5",
        "legacy_codes": ["Bleeding", "Fatty Surface"],
    },

    "Streaking": {
        "category": "Surface Defects",
        "irc_section": "7.2.3",
        "description": (
            "Alternating lean and heavy lines of bitumen, either longitudinally "
            "or transversely, due to non-uniform binder application."
        ),
        "visual_cues": (
            "Parallel dark/light stripes across pavement, either along the road "
            "or across it. Lines of richer bitumen show as darker bands."
        ),
        "severity": None,  # IRC: severity not applicable
        "severity_note": "IRC:82-2015 declares severity not applicable for streaking.",
        "treatment_ref": "7.2.3.5",
        "legacy_codes": ["Streaking"],
    },

    "Hungry Surface": {
        "category": "Surface Defects",
        "irc_section": "7.2.4",
        "description": (
            "Surface characterized by loss of fine aggregates or appearance of "
            "a dry surface with fine cracks. Indicates insufficient bitumen."
        ),
        "visual_cues": (
            "Dry, gray, sandy or porous-looking surface. Fine aggregate "
            "missing from top layer. Often with very fine surface cracks."
        ),
        "severity": None,
        "severity_note": "IRC:82-2015 declares severity not applicable for hungry surface.",
        "treatment_ref": "7.2.4.5",
        "legacy_codes": ["Hungry Surface", "Weathering/Oxidation"],
    },

    # -------------------- 7.3 CRACKS --------------------

    "Hairline Cracks": {
        "category": "Cracks",
        "irc_section": "7.3.2",
        "description": (
            "Cracks present in a narrow area with width less than one millimetre. "
            "Generally isolated and not interconnected; appear as short fine "
            "cracks at close intervals."
        ),
        "visual_cues": (
            "Very fine surface lines (< 1 mm wide). Short, often isolated. "
            "May disappear in hot weather. Distinct from longitudinal/transverse "
            "by absence of clear orientation."
        ),
        "severity": None,
        "severity_note": "IRC:82-2015 declares severity not applicable for hairline cracks.",
        "treatment_ref": "7.3.2.5",
        "legacy_codes": ["Hairline Crack", "Hairline Cracks"],
    },

    "Alligator Cracking": {
        "category": "Cracks",
        "irc_section": "7.3.3",
        "description": (
            "A series of interconnected cracks forming small irregular blocks "
            "resembling the skin of an alligator. Also called map cracking, "
            "fatigue cracking, or mesh cracking."
        ),
        "visual_cues": (
            "Interconnected web pattern. Small irregular polygonal blocks "
            "(typically < 30 cm across). Often along wheel paths."
        ),
        "severity": {
            "Low":    "Width/depth 1-3 mm, isolated, not interconnected, no distortion",
            "Medium": "Width/depth 3-6 mm, interconnected, slight spalling, no pumping",
            "High":   "Width/depth > 6 mm, moderate-to-severe spalling, loose pieces, pumping visible",
        },
        "severity_note": "Visually estimate crack width vs benchmark (1mm ≈ paper edge, 6mm ≈ pencil tip).",
        "treatment_ref": "7.3.3.5",
        "legacy_codes": ["Alligator Crack (D20)", "Alligator Crack", "D20"],
    },

    "Longitudinal Cracking": {
        "category": "Cracks",
        "irc_section": "7.3.4",
        "description": (
            "Cracks parallel to the road centerline. Often appear at joints "
            "between paving lanes or between pavement and paved shoulders."
        ),
        "visual_cues": (
            "Straight or slightly curved cracks running along the direction "
            "of travel. May be single or multiple parallel lines."
        ),
        "severity": {
            "Low":    "Width 1-3 mm, infrequent",
            "Medium": "Width 3-6 mm",
            "High":   "Width > 6 mm, frequent / numerous",
        },
        "severity_note": "Visually estimate crack width. Frequency = how many separate cracks per lane-meter.",
        "treatment_ref": "7.3.4.5",
        "legacy_codes": ["Longitudinal Crack (D00)", "Longitudinal Crack", "D00"],
    },

    "Transverse Cracking": {
        "category": "Cracks",
        "irc_section": "7.3.5",
        "description": (
            "Cracks perpendicular to the road, or as interconnected cracks "
            "forming a series of large blocks across the road. Usually from "
            "low-temperature shrinkage of bituminous mix."
        ),
        "visual_cues": (
            "Cracks running across the lane perpendicular to travel direction. "
            "Often regularly spaced (thermal cracking)."
        ),
        "severity": {
            "Low":    "Width 1-3 mm, infrequent",
            "Medium": "Width 3-6 mm",
            "High":   "Width > 6 mm, frequent / numerous",
        },
        "severity_note": "Same width estimation as longitudinal cracking.",
        "treatment_ref": "7.3.5.5",
        "legacy_codes": ["Transverse Crack (D10)", "Transverse Crack", "D10"],
    },

    "Edge Cracking": {
        "category": "Cracks",
        "irc_section": "7.3.6",
        "description": (
            "Cracks parallel to the outer edge of the pavement, normally "
            "within 0.3-0.5 m inside the pavement edge."
        ),
        "visual_cues": (
            "Cracks along the shoulder side of the lane. May show edge "
            "breakup or loss of material at higher severity."
        ),
        "severity": {
            "Low":    "No breakup or loss of material",
            "Medium": "Some breakup / loss of material (up to 10% of affected length)",
            "High":   "Considerable breakup / loss of material (> 10% of affected length)",
        },
        "severity_note": "Estimate the percentage of crack length showing material loss.",
        "treatment_ref": "7.3.6.5",
        "legacy_codes": ["Edge Crack", "Edge Crack (D08)", "D08"],
    },

    # 7.3.7 Reflection Cracking — EXCLUDED (not vision-only, needs substrate knowledge)

    # -------------------- 7.4 DEFORMATION --------------------

    "Slippage": {
        "category": "Deformation",
        "irc_section": "7.4.1",
        "description": (
            "Relative movement between the wearing course and the layer "
            "beneath, characterized by crescent-shaped cracks pointing in "
            "the direction of wheel thrust."
        ),
        "visual_cues": (
            "Crescent (half-moon) shaped cracks. Apex points in the "
            "direction of vehicle thrust (often uphill on grades)."
        ),
        "severity": {
            "Low":  "Slippage at isolated locations in a lane",
            "High": "Slippage along wheel path in entire lane or carriageway",
        },
        "severity_note": "IRC:82-2015 uses only Low / High for slippage (no Medium).",
        "treatment_ref": "7.4.1.5",
        "legacy_codes": ["Slippage"],
    },

    "Rutting": {
        "category": "Deformation",
        "irc_section": "7.4.2",
        "description": (
            "Longitudinal depression or groove in the pavement along the "
            "wheel path, caused by permanent deformation of one or more "
            "pavement layers under traffic."
        ),
        "visual_cues": (
            "Visible parallel depressions in the wheel paths. Often shows "
            "water pooling after rain. Tire tracks worn into pavement."
        ),
        "severity": {
            "Low":  "4-10 mm deep",
            "High": "> 10 mm deep",
        },
        "severity_note": (
            "IRC:82-2015 uses only Low / High for rutting (no Medium). "
            "Estimate depth visually using reference objects in frame."
        ),
        "treatment_ref": "7.4.2.5",
        "legacy_codes": ["Rutting", "Rutting (D30)", "D30"],
    },

    "Corrugation": {
        "category": "Deformation",
        "irc_section": "7.4.3",
        "description": (
            "Regular undulations (ripples) across the bituminous surface, "
            "usually shallow (~25 mm) with wave spacing of 2-3 m. "
            "Different from larger depressions caused by sub-grade weakness."
        ),
        "visual_cues": (
            "Washboard-like ripples perpendicular to travel direction. "
            "Often near intersections or on grades where vehicles brake."
        ),
        "severity": None,
        "severity_note": "IRC:82-2015 declares severity not applicable for corrugation.",
        "treatment_ref": "7.4.3.5",
        "legacy_codes": ["Corrugation", "Washboarding"],
    },

    "Shoving": {
        "category": "Deformation",
        "irc_section": "7.4.4",
        "description": (
            "Plastic movement within bituminous layers resulting in bulging "
            "of the pavement surface. Typically at points where traffic "
            "starts/stops (intersections, bus stops) or on grades/curves."
        ),
        "visual_cues": (
            "Raised bulges or wave-fronts in the pavement surface. Often "
            "near intersections, bus stops, or sharp curves. May appear "
            "with crescent-shaped slippage cracks."
        ),
        "severity": None,
        "severity_note": "IRC:82-2015 declares severity not applicable for shoving.",
        "treatment_ref": "7.4.4.5",
        "legacy_codes": ["Shoving"],
    },

    "Shallow Depression": {
        "category": "Deformation",
        "irc_section": "7.4.5",
        "description": (
            "Isolated low areas of limited size, dipping about 25 mm or "
            "more below the surrounding profile, where water normally "
            "becomes stagnant."
        ),
        "visual_cues": (
            "Bowl-like dips in the pavement, often holding water. "
            "Boundaries less sharp than a pothole — no broken edges."
        ),
        "severity": None,
        "severity_note": "IRC:82-2015 declares severity not applicable for shallow depression.",
        "treatment_ref": "7.4.5.5",
        "legacy_codes": ["Shallow Depression", "Depression"],
    },

    "Settlement": {
        "category": "Deformation",
        "irc_section": "7.4.6",
        "description": (
            "Relatively large deformations of the pavement compared to "
            "shallow depressions. Often followed by extensive cracking "
            "over the affected region. Includes upheaval (raised areas)."
        ),
        "visual_cues": (
            "Large-scale sagging or rising areas of pavement, typically "
            "extending several meters. Cracking is often present in the "
            "deformed region."
        ),
        "severity": None,
        "severity_note": "IRC:82-2015 declares severity not applicable for settlements.",
        "treatment_ref": "7.4.6.5",
        "legacy_codes": ["Settlement", "Settlements", "Upheaval"],
    },

    # -------------------- 7.5 DISINTEGRATION --------------------

    "Stripping": {
        "category": "Disintegration",
        "irc_section": "7.5.1",
        "description": (
            "Separation of the bitumen film from the surface of aggregate "
            "particles, due to the presence of moisture. Leads to loss of "
            "bond and cohesion in the mixture."
        ),
        "visual_cues": (
            "Exposed aggregate with no bitumen coating visible. Patches "
            "where bitumen has been washed/peeled away. Often on roads "
            "with poor drainage."
        ),
        "severity": None,
        "severity_note": "IRC:82-2015 declares severity not applicable for stripping.",
        "treatment_ref": "7.5.1.5",
        "legacy_codes": ["Stripping", "Water Damage"],
    },

    "Ravelling": {
        "category": "Disintegration",
        "irc_section": "7.5.2",
        "description": (
            "Progressive separation and dissociation of fine aggregate "
            "particles and binder from the bituminous surface. Fine "
            "aggregates wear away first, followed by coarse aggregates."
        ),
        "visual_cues": (
            "Rough, jagged appearance. Surface looks pock-marked or "
            "pitted. Loose aggregate may be visible on/near the surface."
        ),
        "severity": {
            "Low":    "Some loss of fines, initial stage of binder wearing out",
            "Medium": "Loose particles present, some loss, binder worn to a rough surface",
            "High":   "Surface too rough with loss of aggregates",
        },
        "severity_note": "Visually assess surface texture roughness.",
        "treatment_ref": "7.5.2.5",
        "legacy_codes": ["Ravelling", "Raveling"],
    },

    "Potholes": {
        "category": "Disintegration",
        "irc_section": "7.5.3",
        "description": (
            "Bowl-shaped cavities of varying sizes in the bituminous "
            "surface or extending into the binder/base course, caused by "
            "localized disintegration of material."
        ),
        "visual_cues": (
            "Bowl-shaped holes with broken or jagged edges. Visible "
            "exposed pavement layers below. Often hold water after rain."
        ),
        "severity": {
            "Small":  "Up to 25 mm deep AND up to 200 mm wide",
            "Medium": "25-50 mm deep AND 200-500 mm wide",
            "Large":  "> 50 mm deep AND > 500 mm wide",
        },
        "severity_note": (
            "IRC:82-2015 uses 'Small / Medium / Large' (NOT Low/Medium/High). "
            "Estimate dimensions visually using reference objects (vehicle "
            "tire ≈ 250 mm wide, lane width ≈ 3.5 m)."
        ),
        "treatment_ref": "7.5.3.5",
        "legacy_codes": ["Pothole (D40)", "Pothole", "D40"],
    },

    "Edge Breaking": {
        "category": "Disintegration",
        "irc_section": "7.5.4",
        "description": (
            "Common defect where the edge of the bituminous surface gets "
            "broken in an irregular way. Distress restricted to within "
            "30 cm from the pavement edge. May peel off in chunks."
        ),
        "visual_cues": (
            "Ragged, irregular pavement edge with chunks missing or "
            "broken away. Distinct from edge cracking (which has cracks "
            "but no material loss)."
        ),
        "severity": None,
        "severity_note": "IRC:82-2015 declares severity not applicable for edge breaking.",
        "treatment_ref": "7.5.4.5",
        "legacy_codes": ["Edge Breaking", "Edge Break-up"],
    },
}


# ============================================================
# Helpers
# ============================================================

# Pre-build lookup from any historical / legacy label -> canonical IRC name
_LEGACY_TO_IRC: dict[str, str] = {}
for _canonical, _entry in IRC82_DISTRESS_TAXONOMY.items():
    _LEGACY_TO_IRC[_canonical.lower()] = _canonical
    for _legacy in _entry.get("legacy_codes", []):
        _LEGACY_TO_IRC[_legacy.lower()] = _canonical

# Also map a few aliases that might appear in model outputs but aren't in legacy_codes
_ALIASES = {
    "block crack (d43)": "Alligator Cracking",  # Block crack closest IRC analog
    "block crack": "Alligator Cracking",
    "d43": "Alligator Cracking",
    "polishing": "Bleeding",  # Polishing isn't in vision-only IRC; map to closest visible
    "polishing of aggregates": "Bleeding",
    "weathering": "Hungry Surface",
    "oxidation": "Hungry Surface",
    "weathering/oxidation": "Hungry Surface",
    "patch": "Skin Patch",  # Note: patches were not in 7.2-7.5 but section 8.6
    "inlaid patch (d44)": "Skin Patch",
    "inlaid patch": "Skin Patch",
    "d44": "Skin Patch",
    "open joint (d50)": "Edge Breaking",  # closest visible analog
    "d50": "Edge Breaking",
}
# We do NOT add Skin Patch to IRC82_DISTRESS_TAXONOMY because patches are repair
# work products, not distresses, in IRC:82. They are catalogued as treatments.
# But the canonicalizer should still accept the legacy label without rejecting it.
PATCH_LABEL_TREATED_AS_VALID = "Skin Patch"


def canonicalize_to_irc(label: str) -> Optional[str]:
    """
    Map any free-text or legacy distress label to its canonical IRC:82 name.

    Returns:
        The canonical IRC name (e.g. "Potholes", "Alligator Cracking").
        Returns None ONLY for explicit non-distress markers ("Normal",
        "Unknown", "Other Distress", "Other", empty/whitespace).
        For unrecognised distress labels, returns the input string unchanged
        with leading/trailing whitespace stripped — so the parser can decide
        whether to flag for expert review.
    """
    if label is None:
        return None
    s = label.strip()
    if not s:
        return None
    low = s.lower()
    if low in {"normal", "no distress detected", "unknown", "other distress",
               "other", "unknown distress", "none"}:
        return None
    if low in _LEGACY_TO_IRC:
        return _LEGACY_TO_IRC[low]
    if low in _ALIASES:
        mapped = _ALIASES[low]
        if mapped == "Skin Patch":
            return PATCH_LABEL_TREATED_AS_VALID
        return mapped
    # Try stripping parenthetical RDD-style codes "Foo (D00)" -> "Foo"
    if "(" in s:
        bare = s.split("(", 1)[0].strip()
        if bare.lower() in _LEGACY_TO_IRC:
            return _LEGACY_TO_IRC[bare.lower()]
        if bare.lower() in _ALIASES:
            mapped = _ALIASES[bare.lower()]
            if mapped == "Skin Patch":
                return PATCH_LABEL_TREATED_AS_VALID
            return mapped
    return s  # unrecognised — pass through, caller can decide


def is_irc_valid(label: str) -> bool:
    """True if `label` is exactly a canonical IRC name."""
    return label in IRC82_DISTRESS_TAXONOMY


def severity_for(label: str) -> Optional[dict]:
    """Return the severity criteria dict for a distress, or None if N/A."""
    entry = IRC82_DISTRESS_TAXONOMY.get(label)
    if not entry:
        return None
    return entry.get("severity")


def render_taxonomy_for_prompt() -> str:
    """
    Render the IRC:82-2015 taxonomy as a system-prompt block for the VLM.
    Grouped by IRC category, with description + visual cues per type.
    """
    by_cat: dict[str, list[str]] = {}
    for name, entry in IRC82_DISTRESS_TAXONOMY.items():
        by_cat.setdefault(entry["category"], []).append(name)

    lines = []
    lines.append("Pavement distress types per IRC:82-2015 (Indian Roads Congress "
                 "Code of Practice for Maintenance of Bituminous Road Surfaces). "
                 "These are the ONLY valid distress labels:")
    lines.append("")
    cat_order = ["Surface Defects", "Cracks", "Deformation", "Disintegration"]
    for cat in cat_order:
        if cat not in by_cat:
            continue
        section = IRC82_DISTRESS_TAXONOMY[by_cat[cat][0]]["irc_section"].split(".")[0:2]
        lines.append(f"{cat.upper()} (IRC:82 Section {'.'.join(section)}):")
        for name in by_cat[cat]:
            e = IRC82_DISTRESS_TAXONOMY[name]
            lines.append(f"  - {name} (IRC:82 §{e['irc_section']}): {e['description']}")
            lines.append(f"      Visual cues: {e['visual_cues']}")
        lines.append("")
    return "\n".join(lines).strip()


def render_severity_criteria_for_prompt() -> str:
    """
    Render the IRC:82-2015 severity criteria block for the VLM prompt.
    Includes quantitative mm-based thresholds where IRC provides them.
    """
    lines = []
    lines.append("Severity rating per IRC:82-2015 (use these EXACT criteria — not "
                 "subjective judgement). Estimate visually using reference objects "
                 "(coin ≈ 25mm, tire width ≈ 250mm, lane width ≈ 3.5m). "
                 "When scale is uncertain, default to the HIGHER severity tier.")
    lines.append("")
    for name, entry in IRC82_DISTRESS_TAXONOMY.items():
        sev = entry.get("severity")
        if not sev:
            lines.append(f"  - {name}: severity not applicable per IRC:82 §{entry['irc_section']} "
                         f"(report presence only).")
            continue
        crit_strs = [f'{tier}={crit}' for tier, crit in sev.items()]
        lines.append(f"  - {name} (IRC:82 §{entry['irc_section']}):")
        for tier, crit in sev.items():
            lines.append(f"      {tier:8s}: {crit}")
    return "\n".join(lines).strip()


# Self-test on import
def _validate():
    """Sanity check the taxonomy."""
    cats = {e["category"] for e in IRC82_DISTRESS_TAXONOMY.values()}
    assert cats == {"Surface Defects", "Cracks", "Deformation", "Disintegration"}, \
        f"Unexpected category set: {cats}"
    # Every entry must have required keys
    required = {"category", "irc_section", "description", "visual_cues",
                "severity_note", "treatment_ref", "legacy_codes"}
    for name, entry in IRC82_DISTRESS_TAXONOMY.items():
        missing = required - set(entry.keys())
        assert not missing, f"{name} missing keys: {missing}"
    # Verify canonicalizer roundtrip
    assert canonicalize_to_irc("Pothole (D40)") == "Potholes"
    assert canonicalize_to_irc("Longitudinal Crack (D00)") == "Longitudinal Cracking"
    assert canonicalize_to_irc("Raveling") == "Ravelling"
    assert canonicalize_to_irc("Normal") is None
    assert canonicalize_to_irc("Unknown") is None
    assert canonicalize_to_irc("") is None
    assert canonicalize_to_irc("  Potholes  ") == "Potholes"

_validate()


if __name__ == "__main__":
    print(f"IRC:82-2015 vision-only taxonomy: {len(IRC82_DISTRESS_TAXONOMY)} distress types")
    print()
    print(render_taxonomy_for_prompt())
    print()
    print("=" * 70)
    print(render_severity_criteria_for_prompt())
