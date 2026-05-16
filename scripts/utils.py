"""
Shared constants, label maps, prompt templates, and utilities
for the Pavement Distress Classification pipeline.

This module is the single source of truth for:
  - Dataset paths
  - Label maps (GAPs binary + RDD multi-class)
  - System / user prompt templates for both stages
  - Response parsing logic
  - Dynamic system resource detection
"""

import os
import sys
from pathlib import Path


# ============================================================
# Dynamic System Resource Detection
# ============================================================

def detect_system_resources() -> dict:
    """
    Detect available CPU, RAM, and GPU resources dynamically.
    Returns a dict with scaling recommendations.
    """
    import psutil

    info = {}

    # --- CPU ---
    try:
        info["cpu_available"] = len(os.sched_getaffinity(0))
    except (AttributeError, NotImplementedError):
        info["cpu_available"] = psutil.cpu_count(logical=False) or os.cpu_count() or 1
    info["cpu_physical"] = psutil.cpu_count(logical=False) or os.cpu_count() or 1
    info["cpu_logical"] = psutil.cpu_count(logical=True) or os.cpu_count() or 1

    # --- RAM ---
    ram = psutil.virtual_memory()
    info["ram_total_gb"] = round(ram.total / (1024 ** 3), 2)
    info["ram_available_gb"] = round(ram.available / (1024 ** 3), 2)

    # --- Platform ---
    info["platform"] = sys.platform

    # --- GPU ---
    info["gpu_name"] = None
    info["gpu_total_gb"] = 0.0
    info["gpu_free_gb"] = 0.0
    info["has_flash_attn"] = False
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            free, total = torch.cuda.mem_get_info(0)
            info["gpu_name"] = props.name
            info["gpu_total_gb"] = round(total / 1e9, 2)
            info["gpu_free_gb"] = round(free / 1e9, 2)
            info["gpu_compute_capability"] = f"{props.major}.{props.minor}"
            info["gpu_multiprocessors"] = props.multi_processor_count
            # Flash Attention requires Ampere+ (compute >= 8.0)
            info["supports_flash_attn"] = props.major >= 8
        try:
            import flash_attn  # noqa: F401
            info["has_flash_attn"] = True
        except ImportError:
            pass
    except ImportError:
        pass

    # --- Auto-scaling recommendations ---
    physical = info["cpu_physical"]
    info["num_workers"] = min(physical, 8)
    info["parallel_processes"] = min(physical, 8)
    info["pin_memory"] = info["gpu_name"] is not None

    # Chunk size for mmap loading: use 60% of available RAM
    safe_ram_bytes = int(info["ram_available_gb"] * 0.6 * 1024 ** 3)
    # Each GAPs image is 1*160*160*4 bytes (float32) = 102,400 bytes
    info["max_samples_in_ram"] = max(1000, safe_ram_bytes // 102_400)

    return info


def print_system_info():
    """Print system resource info for debugging."""
    info = detect_system_resources()
    print("=" * 50)
    print("SYSTEM RESOURCES")
    print("=" * 50)
    print(f"  CPU: {info['cpu_physical']} physical / {info['cpu_logical']} logical cores")
    print(f"  RAM: {info['ram_available_gb']:.1f} GB available / {info['ram_total_gb']:.1f} GB total")
    if info["gpu_name"]:
        print(f"  GPU: {info['gpu_name']}")
        print(f"       {info['gpu_free_gb']:.1f} GB free / {info['gpu_total_gb']:.1f} GB total")
        print(f"       Flash Attention: {'installed' if info['has_flash_attn'] else 'not installed'}")
    else:
        print("  GPU: None")
    print(f"  Scaling: {info['num_workers']} workers, {info['max_samples_in_ram']} samples/chunk")
    print("=" * 50)
    return info

# ============================================================
# Paths  (auto-detect: works whether run from project root or scripts/)
# ============================================================
# If PROJECT_ROOT env var is set, use it (for portability across machines).
# Otherwise, derive from this file's location (scripts/ is one level below root).
PROJECT_ROOT = Path(os.environ.get(
    "PROJECT_ROOT",
    str(Path(__file__).resolve().parent.parent),
))

# Raw datasets
GAPS_RAW_ROOT = PROJECT_ROOT / "GAPs v2" / "v2" / "NORMvsDISTRESS_50k_160"
RDD_RAW_ROOT = PROJECT_ROOT / "RDD" / "RDD_SPLIT"

# Converted images / training JSONs
DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = DATA_DIR / "images"

# Outputs
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EVAL_DIR = PROJECT_ROOT / "eval_results"

# ============================================================
# Label Maps
# ============================================================

# --- Stage 1: GAPs binary detection ---
GAPS_LABEL_MAP = {
    0: "Normal",
    1: "Distress",
}
GAPS_LABEL_TO_ID = {v: k for k, v in GAPS_LABEL_MAP.items()}

# --- Stage 2: RDD distress type classification ---
# Official RDD2022 benchmark classes (0-3). Class 4 = noise, mapped to "Other".
RDD_LABEL_MAP = {
    0: "Longitudinal Crack (D00)",
    1: "Transverse Crack (D10)",
    2: "Alligator Crack (D20)",
    3: "Pothole (D40)",
    4: "Other Distress",  # noise class from contributor submissions
}
RDD_LABEL_TO_ID = {v: k for k, v in RDD_LABEL_MAP.items()}

# Short codes for programmatic use
RDD_SHORT_CODES = {
    0: "D00",
    1: "D10",
    2: "D20",
    3: "D40",
    4: "Other",
}

# ============================================================
# Distress taxonomy — IRC:82-2015 compliant (vision-only subset)
#
# All distress labels are sourced from scripts/irc82_taxonomy.py which is
# verified against the Indian Roads Congress Code of Practice IRC:82-2015
# Section 7 (Types of Pavement Distress).
#
# Single source of truth. To change the taxonomy, edit irc82_taxonomy.py.
# ============================================================
from scripts.irc82_taxonomy import (
    render_taxonomy_for_prompt as _render_irc_taxonomy,
    render_severity_criteria_for_prompt as _render_irc_severity,
    canonicalize_to_irc as _canonicalize_to_irc,
    IRC82_DISTRESS_TAXONOMY,
)

ALL_DISTRESS_TYPES = _render_irc_taxonomy()
IRC_SEVERITY_CRITERIA = _render_irc_severity()

# ============================================================
# Prompt Templates
# ============================================================

# --- Stage 1: Binary Detection (GAPs) ---
# Tone: assertive but professional. Step-by-step inspection checklist forces a
# deliberate observation pass before the model commits to a label. Negative-
# motivation framing ("missing damage costs road repairs and citizen safety")
# pushes carefulness without melodrama.
STAGE1_SYSTEM_PROMPT = (
    "You are an expert pavement condition inspector reviewing road photos for "
    "a city's road maintenance team. Your decisions are escalated to engineers "
    "and translated into real budget for repairs.\n\n"
    "Errors here have real cost. Missing actual damage ('Normal' on a damaged "
    "road) leaves citizens driving on unsafe surfaces. Falsely flagging clean "
    "roads ('Distress' on intact pavement) wastes inspection time. Both matter.\n\n"
    "Before you answer, run this inspection checklist on the image:\n"
    "  1. CRACKS — any visible lines breaking the surface (any orientation, any pattern)?\n"
    "  2. PATTERN — is there an interconnected mesh / alligator-skin texture?\n"
    "  3. HOLES — any depressions, pits, or bowl-shaped voids in the surface?\n"
    "  4. EDGES — damage along lane edges or shoulders?\n"
    "  5. SURFACE TEXTURE — loose aggregate, raveling, or unusual roughness?\n\n"
    "If ANY of those are present, answer Distress. Otherwise answer Normal. "
    "Do NOT confuse with: shadows, lane markings, wet patches, or oil stains — "
    "those are not pavement distress.\n\n"
    "Respond with exactly one word: Normal or Distress."
)

STAGE1_USER_PROMPT = (
    "<image>\n"
    "Run the inspection checklist on this pavement image. "
    "Is this pavement in normal condition, or does it show signs of distress? "
    "Respond with exactly one word: Normal or Distress."
)


# ============================================================
# Stage 0 — pavement pre-filter
# ============================================================
# Runs BEFORE Stage 1. Designed to be CONSERVATIVE: only rejects an image
# when it is clearly not a road/pavement photo (selfies, indoor scenes,
# random objects, signs alone, sky). Real but damaged/poor-quality road
# photos must pass through, even when they look bad — false rejects cost
# real data, false accepts cost only a downstream Normal/Unknown classification.
#
# Output protocol:
#   YES     — image contains visible road/pavement surface, send to Stage 1
#   NO      — image contains no road/pavement, mark rejected_non_pavement
#   UNSURE  — model is uncertain. Treated as YES (let it through, operator
#             can still review in dashboard if confidence is low).
PAVEMENT_FILTER_SYSTEM_PROMPT = (
    "You are screening photos submitted to a road maintenance reporting "
    "system. Your job is to decide whether a photo contains visible road "
    "or pavement surface that an inspector should look at.\n\n"

    "DEFINITION OF 'PAVEMENT' (be GENEROUS):\n"
    "  Any visible road, street, footpath, parking lot, driveway, or "
    "vehicle path surface. Includes:\n"
    "    - Asphalt (black) — paved or unpaved sections\n"
    "    - Concrete (gray) — slabs, footpaths, kerbs\n"
    "    - Brick paving, cobblestones, stone\n"
    "    - Bituminous surfaces of any age, dry or wet\n"
    "    - Damaged surfaces: cracked, broken, potholed, raveling, eroded\n"
    "    - Muddy or unpaved roads with visible road bed\n"
    "    - Distant pavement in the frame (even partial) — counts\n"
    "    - Pavement visible at any angle: top-down, oblique, eye-level\n"
    "    - Poor lighting, blur, or low quality — still counts if pavement is visible\n\n"

    "NOT PAVEMENT (clearly reject only these):\n"
    "    - People (selfies, group photos, faces) WITHOUT any visible road\n"
    "    - Animals alone, indoor scenes, food, hands, body parts\n"
    "    - Pure sky, water, fields, walls, ceilings\n"
    "    - Signboards, billboards, posters photographed close-up alone\n"
    "    - Vehicles photographed close-up with no road visible\n"
    "    - Documents, screens, books, art\n"
    "    - Random objects (toys, tools, clothes) without road context\n\n"

    "DECISION RULE — BE GENEROUS, ERR ON THE SIDE OF KEEPING:\n"
    "  • If you see ANY pavement surface anywhere in the image → YES\n"
    "  • If pavement is partially visible behind other things → YES\n"
    "  • If the image is blurry / dark / low-quality but pavement is "
    "discernible → YES\n"
    "  • If there's NO pavement visible anywhere in the image → NO\n"
    "  • If you genuinely cannot tell → UNSURE (do not say NO unless certain)\n\n"

    "Respond with exactly one word: YES, NO, or UNSURE."
)

PAVEMENT_FILTER_USER_PROMPT = (
    "<image>\n"
    "Does this image contain any visible road or pavement surface? "
    "Be generous — partial / damaged / blurry pavement still counts. "
    "Reply with exactly one word: YES, NO, or UNSURE."
)

# --- Stage 2: Distress Type Classification (IRC:82-2015) ---
# Few-shot examples use IRC:82 canonical names exclusively.
STAGE2_FEW_SHOT_EXAMPLES = """
Here are examples of correct IRC:82-2015 compliant analysis:

Example 1 - An image showing a long crack running parallel to the road, ~2mm wide, single crack:
DISTRESS_TYPES: Longitudinal Cracking
SEVERITY: Low
DESCRIPTION: A single longitudinal crack ~2mm wide runs along the wheel path with no secondary damage (IRC:82 §7.3.4).

Example 2 - An image showing interconnected web-pattern cracks with a bowl-shaped hole nearby:
DISTRESS_TYPES: Alligator Cracking, Potholes
SEVERITY: High
DESCRIPTION: Severe alligator cracking (>6mm with spalling) and an adjacent pothole indicate advanced structural failure (IRC:82 §7.3.3, §7.5.3).

Example 3 - An image showing a crack across the lane plus loose surface aggregate:
DISTRESS_TYPES: Transverse Cracking, Ravelling
SEVERITY: Medium
DESCRIPTION: A 4mm transverse crack and visible aggregate loss indicating early-stage ravelling (IRC:82 §7.3.5, §7.5.2).
""".strip()

STAGE2_SYSTEM_PROMPT = (
    "You are an expert pavement distress analyst classifying damage for a "
    "city's road maintenance team. Your output drives where repair crews are "
    "sent and how budget is spent.\n\n"
    "ALL OUTPUT MUST CONFORM TO IRC:82-2015 — the Indian Roads Congress Code "
    "of Practice for Maintenance of Bituminous Road Surfaces. The taxonomy "
    "below is the complete and ONLY list of valid distress labels. Severity "
    "uses the IRC:82 quantitative criteria.\n\n"
    "Wrong labels send the wrong fix. Missed potholes hit citizens tomorrow. "
    "Be precise — name only what you can clearly see in the image.\n\n"
    "FORBIDDEN: 'Other Distress', 'Other', 'Unknown', 'Various'. "
    "If genuinely no distress is visible, output 'DISTRESS_TYPES: Normal - "
    "No distress detected'. If a distress is visible but you cannot identify "
    "which IRC:82 type it is, pick the closest type from the taxonomy and "
    "report Low severity — never use generic non-IRC labels.\n\n"
    f"{ALL_DISTRESS_TYPES}\n\n"
    f"{IRC_SEVERITY_CRITERIA}\n\n"
    "Inspection protocol (run silently before answering):\n"
    "  STEP 1 — CRACK ORIENTATION\n"
    "    • Lines along the road direction → Longitudinal Cracking (IRC §7.3.4)\n"
    "    • Lines across the road direction → Transverse Cracking (IRC §7.3.5)\n"
    "    • Very fine lines < 1mm with no clear orientation → Hairline Cracks (IRC §7.3.2)\n"
    "  STEP 2 — CRACK PATTERN\n"
    "    • Interconnected web / alligator-skin → Alligator Cracking (IRC §7.3.3)\n"
    "    • Cracks along the pavement edge (within ~30 cm) → Edge Cracking (IRC §7.3.6)\n"
    "    • Crescent / half-moon shaped → Slippage (IRC §7.4.1)\n"
    "  STEP 3 — SURFACE DEFECTS\n"
    "    • Bowl-shaped cavity with broken edges → Potholes (IRC §7.5.3)\n"
    "    • Bowl-shaped dip WITHOUT broken edges (still smooth pavement) → Shallow Depression (IRC §7.4.5)\n"
    "    • Rough/pock-marked, loss of aggregate → Ravelling (IRC §7.5.2)\n"
    "    • Exposed aggregate, no bitumen coating → Stripping (IRC §7.5.1)\n"
    "    • Shiny / glossy / wet-looking patches → Bleeding (IRC §7.2.1)\n"
    "    • Dry, sandy, gray surface → Hungry Surface (IRC §7.2.4)\n"
    "    • Alternating dark/light bitumen lines → Streaking (IRC §7.2.3)\n"
    "  STEP 4 — DEFORMATION\n"
    "    • Longitudinal depression in wheel path → Rutting (IRC §7.4.2)\n"
    "    • Washboard ripples across the lane → Corrugation (IRC §7.4.3)\n"
    "    • Raised bulges / wave fronts in surface → Shoving (IRC §7.4.4)\n"
    "    • Large-scale sagging or rising area → Settlement (IRC §7.4.6)\n"
    "  STEP 5 — EDGE FAILURE\n"
    "    • Ragged pavement edge with material missing → Edge Breaking (IRC §7.5.4)\n"
    "  STEP 6 — RULE OUT FALSE POSITIVES\n"
    "    • Shadows, lane markings, oil stains, skid marks, manhole covers, "
    "drainage grates — these are NOT pavement distress.\n\n"
    f"{STAGE2_FEW_SHOT_EXAMPLES}\n\n"
    "Now analyze the provided image. List every IRC:82 distress type you "
    "observe (an image may contain multiple). Respond in this exact format:\n"
    "DISTRESS_TYPES: <comma-separated list of canonical IRC:82 names from the taxonomy above>\n"
    "SEVERITY: <Low / Medium / High — OR Small / Medium / Large for potholes — "
    "OR 'N/A' for types where IRC:82 declares severity not applicable>\n"
    "DESCRIPTION: <one sentence describing what you observe, citing IRC:82 section>"
)

STAGE2_USER_PROMPT = (
    "<image>\n"
    "Run the 6-step IRC:82 inspection protocol on this pavement image. "
    "Identify every IRC:82 distress type present. Be precise — use only "
    "canonical IRC:82 names from the taxonomy. Provide type(s), severity "
    "per IRC:82 quantitative criteria, and a one-sentence description."
)

# --- Stage 2 training response template ---
# During fine-tuning, we use a simplified but structured response.
def build_stage2_training_response(class_ids: list[int]) -> str:
    """
    Build the expected model response for a training sample.

    Parameters
    ----------
    class_ids : list[int]
        Unique YOLO class IDs found in the image's label file.

    Returns
    -------
    str
        Structured response string for the training conversation.
    """
    if not class_ids:
        return (
            "DISTRESS_TYPES: Normal - No distress detected\n"
            "SEVERITY: None\n"
            "DESCRIPTION: The pavement surface appears to be in good condition "
            "with no visible signs of damage or deterioration."
        )

    # NEVER emit "Other Distress" in training labels (RDD class 4 is the noise
    # bucket). The model learns to default to it for ambiguous cases, which
    # produces useless output downstream. Drop class 4 entirely; if every
    # class is class-4, fall back to the Normal response.
    actionable_ids = [cid for cid in class_ids if cid != 4]
    if not actionable_ids:
        return (
            "DISTRESS_TYPES: Normal - No distress detected\n"
            "SEVERITY: None\n"
            "DESCRIPTION: The pavement surface shows minor deterioration but no "
            "specific distress type can be clearly identified."
        )

    type_names = []
    for cid in sorted(set(actionable_ids)):
        name = RDD_LABEL_MAP.get(cid, "Unknown Distress")
        type_names.append(name)

    types_str = ", ".join(type_names)

    # Heuristic severity based on count / type
    has_pothole = 3 in class_ids
    num_types = len(set(class_ids))
    total_annotations = len(class_ids)

    if has_pothole or total_annotations >= 5:
        severity = "High"
    elif num_types >= 2 or total_annotations >= 3:
        severity = "Medium"
    else:
        severity = "Low"

    # Build description
    desc_parts = []
    for name in type_names:
        desc_parts.append(name.split(" (")[0].lower())
    desc_str = " and ".join(desc_parts)
    description = f"The pavement shows signs of {desc_str} damage."

    return (
        f"DISTRESS_TYPES: {types_str}\n"
        f"SEVERITY: {severity}\n"
        f"DESCRIPTION: {description}"
    )


# ============================================================
# Response Parsing
# ============================================================

def parse_pavement_filter_response(text: str) -> str:
    """
    Parse the pavement pre-filter (Stage 0) response.

    Returns:
        'yes'    — image contains pavement, send to Stage 1
        'no'     — image clearly contains no pavement, reject
        'unsure' — model can't tell; treated as 'yes' downstream (do not reject)
    """
    if not text:
        return "unsure"
    t = text.strip().upper()
    # Strip common formatting (period, quotes, newlines)
    for c in '.,!?\'"`':
        t = t.replace(c, "")
    t = t.split()[0] if t.split() else ""
    if t in ("YES", "Y", "PAVEMENT", "ROAD"):
        return "yes"
    if t in ("NO", "N", "NOT"):
        return "no"
    if t in ("UNSURE", "MAYBE", "UNCERTAIN", "UNKNOWN"):
        return "unsure"
    # Unparseable — default to UNSURE (which means "let it through" downstream).
    # Never default to NO on unparseable, that would silently lose real data.
    return "unsure"


def parse_stage1_response(text: str) -> int:
    """
    Parse the model's Stage-1 (binary) response.

    Returns
    -------
    int
        0 = Normal, 1 = Distress, -1 = unparseable.
    """
    text_lower = text.strip().lower()

    # Exact match first
    if text_lower in ("normal", "normal."):
        return 0
    if text_lower in ("distress", "distress.", "distressed"):
        return 1

    # Keyword search
    has_normal = "normal" in text_lower
    has_distress = "distress" in text_lower or "damage" in text_lower or "crack" in text_lower or "pothole" in text_lower

    if has_distress and not has_normal:
        return 1
    if has_normal and not has_distress:
        return 0
    if has_distress and has_normal:
        # If both keywords present, check which appears first
        idx_normal = text_lower.find("normal")
        idx_distress = min(
            text_lower.find("distress") if "distress" in text_lower else 9999,
            text_lower.find("damage") if "damage" in text_lower else 9999,
            text_lower.find("crack") if "crack" in text_lower else 9999,
        )
        return 0 if idx_normal < idx_distress else 1

    return -1  # unparseable


def parse_stage2_response(text: str) -> dict:
    """
    Parse the model's Stage-2 (type classification) response.

    Returns
    -------
    dict with keys:
        - distress_types: list[str]
        - severity: str
        - description: str
        - raw: str (original response)
    """
    result = {
        "distress_types": [],
        "severity": "Unknown",
        "description": "",
        "raw": text.strip(),
    }

    # Try to extract structured fields
    lines = text.strip().split("\n")
    for line in lines:
        line_stripped = line.strip()
        if line_stripped.upper().startswith("DISTRESS_TYPES:"):
            types_str = line_stripped.split(":", 1)[1].strip()
            if "normal" in types_str.lower() or "no distress" in types_str.lower():
                result["distress_types"] = ["Normal"]
            else:
                # Canonicalize every emitted label to IRC:82-2015 name.
                # canonicalize_to_irc returns None for {Unknown, Other, Other Distress,
                # Normal, empty} — those get filtered out. Unrecognised distress strings
                # pass through unchanged so the parser caller can flag them.
                raw_types = [
                    t.strip() for t in types_str.split(",") if t.strip()
                ]
                canonical_types = []
                for raw in raw_types:
                    irc_name = _canonicalize_to_irc(raw)
                    if irc_name is None:
                        continue  # 'Other', 'Unknown', etc — drop
                    if irc_name not in canonical_types:
                        canonical_types.append(irc_name)
                result["distress_types"] = canonical_types
        elif line_stripped.upper().startswith("SEVERITY:"):
            result["severity"] = line_stripped.split(":", 1)[1].strip()
        elif line_stripped.upper().startswith("DESCRIPTION:"):
            result["description"] = line_stripped.split(":", 1)[1].strip()

    # Fallback: if no structured output, try keyword extraction.
    # Keywords map to canonical IRC:82 names (the parser stays IRC-aligned).
    if not result["distress_types"]:
        text_lower = text.lower()
        keyword_to_irc = {
            "longitudinal crack": "Longitudinal Cracking",
            "transverse crack": "Transverse Cracking",
            "alligator crack": "Alligator Cracking",
            "fatigue crack": "Alligator Cracking",
            "mesh crack": "Alligator Cracking",
            "map crack": "Alligator Cracking",
            "pothole": "Potholes",
            "block crack": "Alligator Cracking",  # closest IRC visible analog
            "rutting": "Rutting",
            "raveling": "Ravelling",
            "ravelling": "Ravelling",
            "bleeding": "Bleeding",
            "fatty surface": "Bleeding",
            "edge crack": "Edge Cracking",
            "edge breaking": "Edge Breaking",
            "edge break": "Edge Breaking",
            "shallow depression": "Shallow Depression",
            "depression": "Shallow Depression",
            "settlement": "Settlement",
            "upheaval": "Settlement",
            "shoving": "Shoving",
            "corrugation": "Corrugation",
            "washboard": "Corrugation",
            "slippage": "Slippage",
            "stripping": "Stripping",
            "streaking": "Streaking",
            "hungry surface": "Hungry Surface",
            "hairline": "Hairline Cracks",
            "weathering": "Hungry Surface",  # closest visible analog
            "oxidation": "Hungry Surface",
        }
        for keyword, label in keyword_to_irc.items():
            if keyword in text_lower and label not in result["distress_types"]:
                result["distress_types"].append(label)

    if not result["distress_types"]:
        result["distress_types"] = ["Unknown"]

    return result


# ============================================================
# Confidence Threshold (single source of truth)
# ============================================================
# Below this threshold, assessments are flagged for expert review.
# This value should be calibrated on the validation set after fine-tuning.
CONFIDENCE_THRESHOLD = 0.80


def map_stage2_to_rdd_ids(distress_types: list[str]) -> list[int]:
    """
    Map parsed distress type strings back to RDD class IDs.
    Returns list of matched IDs; unmatched types are skipped.
    """
    ids = []
    for dt in distress_types:
        dt_lower = dt.lower()
        if "d00" in dt_lower or "longitudinal" in dt_lower:
            ids.append(0)
        elif "d10" in dt_lower or "transverse" in dt_lower:
            ids.append(1)
        elif "d20" in dt_lower or "alligator" in dt_lower or "fatigue" in dt_lower:
            ids.append(2)
        elif "d40" in dt_lower or "pothole" in dt_lower:
            ids.append(3)
    return sorted(set(ids))
