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
# Comprehensive distress taxonomy (for few-shot / open-world)
# This is injected into the system prompt so the model can
# recognise distress types BEYOND the RDD training labels.
# ============================================================
ALL_DISTRESS_TYPES = """
Known pavement distress types (classify into one or more):

CRACKING:
  - Longitudinal Crack (D00): Cracks running parallel to the road direction.
  - Transverse Crack (D10): Cracks running perpendicular to the road direction.
  - Alligator Crack (D20): Interconnected cracks forming a pattern resembling alligator skin. Also called fatigue cracking or mesh cracking.
  - Block Crack (D43): Rectangular cracks that divide the pavement into blocks, typically caused by thermal contraction.
  - Edge Crack: Cracks along the edge of the pavement, usually within 30 cm of the shoulder.
  - Reflective Crack: Cracks in overlay surfaces that mirror joints or cracks in the underlying layer.

SURFACE DEFORMATION:
  - Pothole (D40): Bowl-shaped holes in the pavement surface caused by wear and weathering.
  - Rutting: Longitudinal surface depressions in the wheel path caused by repeated traffic loads.
  - Shoving: Longitudinal displacement of the pavement surface, often near intersections.
  - Depression: Localized low areas in the pavement surface.

SURFACE DEFECTS:
  - Raveling: Loss of aggregate particles from the pavement surface.
  - Bleeding: Excess asphalt binder on the pavement surface creating a shiny, reflective area.
  - Polishing: Smooth, slippery pavement surface from traffic wear.

PATCHES AND REPAIRS:
  - Inlaid Patch (D44): Repair patches that are flush with the surrounding pavement.
  - Applied Patch: Repair material applied on top of existing pavement.
  - Utility Cut Patch: Patches from utility work (water, gas, electrical).

JOINT DEFECTS:
  - Open Joint (D50): Gaps or separations at pavement joints.
  - Joint Seal Damage: Deterioration of sealant material in pavement joints.

OTHER:
  - Weathering/Oxidation: General surface deterioration from sun and weather exposure.
  - Water Damage: Damage caused by water infiltration (stripping, pumping).
""".strip()

# ============================================================
# Prompt Templates
# ============================================================

# --- Stage 1: Binary Detection (GAPs) ---
STAGE1_SYSTEM_PROMPT = (
    "You are an expert pavement condition inspector. "
    "Your task is to examine pavement images and determine whether "
    "the pavement shows any signs of distress or damage. "
    "Respond with exactly one word: Normal or Distress."
)

STAGE1_USER_PROMPT = (
    "<image>\n"
    "Examine this pavement image carefully. "
    "Is this pavement in normal condition, or does it show signs of distress? "
    "Respond with exactly one word: Normal or Distress."
)

# --- Stage 2: Distress Type Classification (RDD) ---
# Includes few-shot text examples to ground the model's output format.
STAGE2_FEW_SHOT_EXAMPLES = """
Here are examples of correct analysis:

Example 1 - An image showing a long crack running parallel to the road with minor surface wear:
DISTRESS_TYPES: Longitudinal Crack (D00)
SEVERITY: Low
DESCRIPTION: The pavement shows a single longitudinal crack running along the wheel path with no secondary damage.

Example 2 - An image showing interconnected cracks forming a web pattern with a nearby bowl-shaped hole:
DISTRESS_TYPES: Alligator Crack (D20), Pothole (D40)
SEVERITY: High
DESCRIPTION: The pavement shows severe alligator cracking with an adjacent pothole indicating advanced structural failure.

Example 3 - An image showing a crack running across the road width with a patched repair nearby:
DISTRESS_TYPES: Transverse Crack (D10), Inlaid Patch (D44)
SEVERITY: Medium
DESCRIPTION: The pavement shows a transverse crack perpendicular to traffic direction with an adjacent repair patch.
""".strip()

STAGE2_SYSTEM_PROMPT = (
    "You are an expert pavement distress analyst. "
    "Your task is to identify the specific type(s) of pavement distress "
    "visible in the image. Use your knowledge of pavement engineering "
    "to classify the distress accurately.\n\n"
    f"{ALL_DISTRESS_TYPES}\n\n"
    f"{STAGE2_FEW_SHOT_EXAMPLES}\n\n"
    "Now analyze the provided image. Respond with a structured analysis in this exact format:\n"
    "DISTRESS_TYPES: <comma-separated list of distress types found>\n"
    "SEVERITY: <Low / Medium / High>\n"
    "DESCRIPTION: <one sentence describing what you observe>"
)

STAGE2_USER_PROMPT = (
    "<image>\n"
    "Analyze this pavement image. Identify all types of distress present. "
    "Provide the distress type(s), severity estimate, and a brief description."
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

    type_names = []
    for cid in sorted(set(class_ids)):
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
                result["distress_types"] = [
                    t.strip() for t in types_str.split(",") if t.strip()
                ]
        elif line_stripped.upper().startswith("SEVERITY:"):
            result["severity"] = line_stripped.split(":", 1)[1].strip()
        elif line_stripped.upper().startswith("DESCRIPTION:"):
            result["description"] = line_stripped.split(":", 1)[1].strip()

    # Fallback: if no structured output, try keyword extraction
    if not result["distress_types"]:
        text_lower = text.lower()
        keyword_map = {
            "longitudinal crack": "Longitudinal Crack (D00)",
            "transverse crack": "Transverse Crack (D10)",
            "alligator crack": "Alligator Crack (D20)",
            "fatigue crack": "Alligator Crack (D20)",
            "mesh crack": "Alligator Crack (D20)",
            "pothole": "Pothole (D40)",
            "block crack": "Block Crack (D43)",
            "rutting": "Rutting",
            "raveling": "Raveling",
            "bleeding": "Bleeding",
            "patch": "Repair Patch",
            "open joint": "Open Joint (D50)",
            "edge crack": "Edge Crack",
            "depression": "Depression",
        }
        for keyword, label in keyword_map.items():
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
