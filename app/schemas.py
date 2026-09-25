"""
Pydantic models for the Pavement Distress Classification API.
"""

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel, Field
from scripts.utils import CONFIDENCE_THRESHOLD  # noqa: E402 — single source of truth


class ClassificationResponse(BaseModel):
    """Response from the /classify endpoint."""

    # Stage 1: Binary detection
    is_distressed: bool = Field(
        description="Whether the pavement shows signs of distress"
    )
    stage1_label: str = Field(
        description="Stage 1 result: 'Normal' or 'Distress'"
    )
    stage1_confidence: float = Field(
        description="Stage 1 confidence (0.0-1.0): softmax probability of the predicted class"
    )

    # Stage 2: Type classification (only populated if distressed)
    distress_types: list[str] = Field(
        default_factory=list,
        description="Distress types identified, in the order the model listed "
                    "them; the first is shown as the main one (empty if normal)"
    )
    primary_distress_type: Optional[str] = Field(
        default=None,
        description="The label the model listed first, distress_types[0]. "
                    "stage2_confidence refers to this label only when "
                    "stage2_confidence_mode is 'primary'"
    )
    secondary_distress_types: list[str] = Field(
        default_factory=list,
        description="The other distress types the model listed"
    )
    severity: str = Field(
        default="None",
        description="Estimated severity: 'None', 'Low', 'Medium', or 'High'"
    )
    description: str = Field(
        default="",
        description="Brief description of the observed distress"
    )
    stage2_confidence: float = Field(
        default=0.0,
        description="Stage 2 confidence (0.0-1.0) read by the review gate. Its "
                    "definition is set by STAGE2_CONFIDENCE_MODE: 'primary' = "
                    "probability of the primary label, 'field' = geometric mean "
                    "over all labels, 'sequence' = over the whole output"
    )
    stage2_confidence_primary: Optional[float] = Field(
        default=None,
        description="Probability of the primary label alone (recorded in every mode)"
    )
    stage2_confidence_probe: Optional[float] = Field(
        default=None,
        description="Per-type probe confidence: probability that every calibrated type "
                    "decision is right (STAGE2_MODE=probe only; recorded in every mode)"
    )
    stage2_mode: Optional[str] = Field(
        default=None,
        description="Which path produced distress_types for this image: 'probe' "
                    "(per-type Yes/No probe) or 'generate' (free-form list)"
    )
    stage1_safety_net: Optional[dict] = Field(
        default=None,
        description="When Stage 1 said Normal and the safety net is on: whether the probe "
                    "flagged distress (then needs_expert_review is true), the strongest type "
                    "and its P(yes)."
    )
    condition_indicators: list[str] = Field(
        default_factory=list,
        description="IRC:82 condition indicators seen (e.g. 'Patching', Tables 5.1-5.3). "
                    "Not distress types, so never part of distress_types."
    )

    # Expert review flag
    needs_expert_review: bool = Field(
        default=False,
        description="True if confidence is below threshold and needs human verification"
    )

    # Metadata
    processing_time_ms: float = Field(
        description="Total processing time in milliseconds"
    )
    stage1_time_ms: float = Field(
        description="Stage 1 inference time in milliseconds"
    )
    stage2_time_ms: float = Field(
        default=0.0,
        description="Stage 2 inference time in milliseconds (0 if skipped)"
    )

    # Raw responses (for debugging)
    stage1_raw: str = Field(
        default="",
        description="Raw model response from Stage 1"
    )
    stage2_raw: str = Field(
        default="",
        description="Raw model response from Stage 2 (empty if skipped)"
    )


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""

    status: str = Field(description="Server status: 'ok' or 'error'")
    model_loaded: bool = Field(description="Whether the model is loaded")
    model_name: str = Field(description="Name/path of the loaded model")
    device: str = Field(description="GPU device being used")
    adapter_loaded: bool = Field(description="Whether a LoRA adapter is loaded")
    adapter_disabled_in_config: bool = Field(
        default=False,
        description="DISABLE_ADAPTER env var is true — production runs the base model only.",
    )
    prompts_version: str = Field(
        default="v2",
        description="Active prompt set: v1 (plain baseline) or v2 (Improved Baseline + IRC:82).",
    )
    taxonomy: str = Field(
        default="IRC:82-2015",
        description="Distress taxonomy in use for label canonicalization.",
    )
    pavement_filter_enabled: bool = Field(
        default=True,
        description="Stage 0 pre-filter active in the worker pipeline.",
    )
    model_family: str = Field(
        default="unknown",
        description="Resolved model family from the checkpoint config (e.g. qwen2_5_vl, qwen3_vl).",
    )
    model_class: str = Field(
        default="unknown",
        description="transformers class actually used to load the model.",
    )
    quantization_bits: int = Field(
        default=-1,
        description="Quantization actually in effect (0=bf16). -1 when unknown.",
    )
    oom_fallback_used: bool = Field(
        default=False,
        description="True if the loader had to drop precision to fit VRAM.",
    )
    max_image_pixels: int = Field(
        default=0,
        description="Effective processor pixel cap (may be auto-reduced from the requested value).",
    )
    stage2_confidence_mode: str = Field(
        default="sequence",
        description="Stage 2 confidence metric: 'sequence' (all tokens), 'field' "
                    "(DISTRESS_TYPES only), 'primary' or 'probe'.",
    )
    stage2_mode: str = Field(
        default="generate",
        description="Requested Stage 2 type decision: 'generate' or 'probe'.",
    )
    stage2_mode_effective: str = Field(
        default="generate",
        description="What actually runs: 'probe' (labels from the probe) or 'shadow' "
                    "(probe recorded, labels unchanged) only if its config loaded and the "
                    "fast/slow parity self-test passed at startup; otherwise 'generate'.",
    )
    stage1_safety_net: bool = Field(
        default=False,
        description="Stage 1 'Normal' answers are checked by the probe and routed to "
                    "expert review when it sees distress (STAGE1_SAFETY_NET).",
    )
    stage2_probe: Optional[dict] = Field(
        default=None,
        description="Probe status: enabled, reason if not, parity self-test, "
                    "variant and threshold provenance.",
    )
    vram_used_gb: float = Field(
        default=0.0,
        description="VRAM in use after model load, in GB.",
    )


class ExpertCorrectionRequest(BaseModel):
    """Request body for submitting an expert correction."""

    assessment_id: str = Field(description="UUID of the assessment being corrected")
    corrected_types: list[str] = Field(
        description="Expert-verified distress types (e.g. ['Pothole (D40)', 'Alligator Crack (D20)'])"
    )
    corrected_severity: str = Field(
        description="Expert-verified severity: 'Low', 'Medium', or 'High'"
    )
    expert_notes: str = Field(default="", description="Optional notes from the expert")


class RetrainStatusResponse(BaseModel):
    """Response from the /retrain/status endpoint."""

    is_running: bool = Field(description="Whether a retraining job is currently running")
    progress_pct: float = Field(default=0.0, description="Progress percentage (0-100)")
    current_step: int = Field(default=0, description="Current training step")
    total_steps: int = Field(default=0, description="Total training steps")
    eta_seconds: float = Field(default=0.0, description="Estimated time remaining in seconds")
    pending_corrections: int = Field(default=0, description="Number of corrections awaiting retraining")
    last_retrain_at: str = Field(default="", description="ISO timestamp of last completed retrain")


class ErrorResponse(BaseModel):
    """Error response."""

    error: str = Field(description="Error message")
    detail: str = Field(default="", description="Detailed error information")
