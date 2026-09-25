"""
Model loading and inference logic for the Pavement Distress Classifier.

Provides a thread-safe singleton that:
  1. Loads Qwen2.5-VL with optional LoRA adapter
  2. Extracts REAL confidence scores from model logits (no heuristics)
  3. Runs the two-stage pipeline (detection -> classification)
  4. Flags low-confidence results for expert review
"""

import math
import os
import threading
import time

import torch
from PIL import Image

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import confidence as _conf
from scripts.irc82_taxonomy import canonicalize_to_irc
from scripts.model_loader import load_vlm
from scripts.utils import (
    CONFIDENCE_THRESHOLD,
    PAVEMENT_FILTER_SYSTEM_PROMPT,
    PAVEMENT_FILTER_USER_PROMPT,
    parse_pavement_filter_response,
    parse_stage1_response,
    parse_stage2_response,
)

# Prompt set selection
# ===================
# Production = "improved_baseline" (v2 IRC prompts: deep persona + stakes
# + 6-step IRC:82 inspection protocol + quantitative severity criteria).
# Set PROMPTS_VERSION=v1 to fall back to the plain Stage-2 prompts (kept
# for paper/research comparison only — see scripts/utils.py).
_PROMPTS_VERSION = os.environ.get("PROMPTS_VERSION", "v2").lower()
if _PROMPTS_VERSION in ("v2", "improved", "improved_baseline"):
    from scripts.utils_v2_prompts import (
        STAGE1_SYSTEM_PROMPT_V2 as STAGE1_SYSTEM_PROMPT,
        STAGE1_USER_PROMPT_V2 as STAGE1_USER_PROMPT,
        STAGE2_SYSTEM_PROMPT_V2 as STAGE2_SYSTEM_PROMPT,
        STAGE2_USER_PROMPT_V2 as STAGE2_USER_PROMPT,
    )
    _ACTIVE_PROMPTS = "v2_improved_baseline"
elif _PROMPTS_VERSION == "v2_primary_first":
    # v2 plus a "list the most prominent distress first" rule. Tested on
    # 2026-09-23 and NOT adopted (see scripts/utils_v2_prompts.py). For
    # reproducing that experiment only.
    from scripts.utils_v2_prompts import (
        STAGE1_SYSTEM_PROMPT_V2 as STAGE1_SYSTEM_PROMPT,
        STAGE1_USER_PROMPT_V2 as STAGE1_USER_PROMPT,
        STAGE2_SYSTEM_PROMPT_V2_PRIMARY_FIRST as STAGE2_SYSTEM_PROMPT,
        STAGE2_USER_PROMPT_V2 as STAGE2_USER_PROMPT,
    )
    _ACTIVE_PROMPTS = "v2_improved_baseline_primary_first"
else:
    from scripts.utils import (
        STAGE1_SYSTEM_PROMPT,
        STAGE1_USER_PROMPT,
        STAGE2_SYSTEM_PROMPT,
        STAGE2_USER_PROMPT,
    )
    _ACTIVE_PROMPTS = "v1_plain"
print(f"[PavementClassifier] Active prompt set: {_ACTIVE_PROMPTS}")

# Stage 2 confidence metric
# =========================
# 'field' (default since 2026-09-17) = geometric mean over the DISTRESS_TYPES
#     value tokens only. This is what the confidence is supposed to mean: how
#     sure is the model about the CLASSIFICATION.
# 'sequence' = geometric mean over EVERY generated token, including the
#     free-text DESCRIPTION. Kept selectable so the pre-2026-09-17 results
#     stay reproducible, but it should not be used as a gate.
#
# Why the default moved (scripts/calibrate_confidence.py, n=37 Attain images,
# eval_results/confidence_calibration_qwen25vl7b.json):
#
#       metric    mean(correct)  mean(wrong)   AUC     auto-accept @0.80
#       field        0.8295        0.8185     0.613        27/37
#       sequence     0.7670        0.7927     0.231         4/37
#
# AUC 0.5 means the score carries no information. 'sequence' scored 0.231 —
# it ranked WRONG predictions ABOVE right ones, because the geomean was
# dominated by free-text prose where a confidently-worded mistake reads as
# high probability and a hedged correct answer reads as low. A gate built on
# an inverted signal is worse than no gate: it sends good predictions to the
# expert and waves bad ones through. 'field' is only weakly positive (0.613),
# but it is on the correct side of 0.5.
#
# Both values are computed and returned on EVERY call regardless of this
# setting, and the worker writes both into raw_response — so the threshold
# can be re-calibrated from real expert corrections without re-running
# inference, and switching modes never loses the other number.
# Stage 1 generation budget. The answer is one word; 12 tokens leaves room for
# a stray "Distress." or a short preamble while keeping the step count low.
# Override via env if a future model needs more headroom.
STAGE1_MAX_NEW_TOKENS = int(os.environ.get("STAGE1_MAX_NEW_TOKENS", "12"))

#
# 'primary' (2026-09-23) = joint probability of the FIRST label only.
#     Secondary labels would no longer lower the score. Recorded on every row
#     but NOT the production gate: on 407 labelled Attain images it did not
#     separate right first labels from wrong ones (AUC 0.468, joint; 0.476,
#     geomean), while 'field' did (0.635). eval_results/primary_confidence.md.
#     Falls back to 'field' (recorded per row) when no label can be matched.
#
# 'probe' (2026-09-23) = the probability, from the per-type probe, that every
#     calibrated type decision is right (scripts/stage2_probe_rules.py). Only
#     meaningful with STAGE2_MODE=probe; when the probe is unavailable for an
#     image the gate falls back to 'field' and the row records that it did.
STAGE2_CONFIDENCE_MODE = os.environ.get("STAGE2_CONFIDENCE_MODE", "field").lower()
if STAGE2_CONFIDENCE_MODE not in ("sequence", "field", "primary", "probe"):
    print(f"[PavementClassifier] WARNING: unknown STAGE2_CONFIDENCE_MODE "
          f"{STAGE2_CONFIDENCE_MODE!r} — falling back to 'field'")
    STAGE2_CONFIDENCE_MODE = "field"
print(f"[PavementClassifier] Stage 2 confidence mode: {STAGE2_CONFIDENCE_MODE}")

# Stage 2 type decision
# =====================
# 'generate' = the free-form DISTRESS_TYPES list the v2 prompt produces. On
#     Attain it was "Longitudinal Cracking, Transverse Cracking" for 78% of
#     images and almost never named alligator cracking, potholes on wide
#     frames, ravelling or weathering.
# 'probe'    = the same generation (still used for severity and the
#     description), plus one Yes/No question per IRC:82 type scored from the
#     model's own probabilities; the probe decides the calibrated types and
#     may only remove the others (scripts/stage2_probe_rules.py). Thresholds
#     come from STAGE2_PROBE_CONFIG, written by scripts/stage2_probe_report.py
#     from the DEV split. Any failure - missing or invalid config, a fast/slow
#     parity mismatch at load, an exception on an image - falls back to
#     'generate' for that image and records why.
# 'shadow'   = the probe runs and everything it would decide is recorded on
#     the row (raw_response.stage2_probe, applied=false), but distress_types
#     and the gate stay those of 'generate'. The production setting since
#     2026-09-24: on Attain the probe doubled macro MCC (cross-validated), but
#     its Attain-tuned thresholds fire on almost every Bengaluru close-up
#     (Ravelling on 98% of 204 uploads), and no labelled Bengaluru photos
#     exist to set thresholds on. Shadow mode collects the probabilities so
#     scripts/calibrate_probe_from_expert_labels.py can fit Bengaluru
#     thresholds from expert reviews without re-running the model.
STAGE2_MODE = os.environ.get("STAGE2_MODE", "generate").lower()
if STAGE2_MODE not in ("generate", "probe", "shadow"):
    print(f"[PavementClassifier] WARNING: unknown STAGE2_MODE {STAGE2_MODE!r} "
          f"— falling back to 'generate'")
    STAGE2_MODE = "generate"
STAGE2_PROBE_CONFIG = os.environ.get(
    "STAGE2_PROBE_CONFIG",
    str(Path(__file__).resolve().parent.parent / "configs" / "stage2_probe.json"))
if not Path(STAGE2_PROBE_CONFIG).is_absolute():
    # relative to the project root, not to wherever uvicorn was launched from
    STAGE2_PROBE_CONFIG = str(Path(__file__).resolve().parent.parent / STAGE2_PROBE_CONFIG)
print(f"[PavementClassifier] Stage 2 mode: {STAGE2_MODE}")

# Stage 1 safety net
# ==================
# Production Stage 1 called 48% of Attain images WITH annotated distress
# "Normal" (recall 51.8%, 847 images). When it says Normal, the probe asks
# about each headline type; if any P(yes) reaches the configured threshold the
# image goes to expert review instead of being auto-classified Normal. It can
# only ADD human review - it never changes a label. On Attain (out-of-fold) it
# flagged 313 of the 371 missed distress images and 22 of 78 clean ones; on
# 229 Bengaluru uploads it would have re-routed 1 of the 6 auto-accepted
# Normals. Needs the probe (any STAGE2_MODE) and a config with a
# stage1_safety_net section; otherwise it is off and /health says why.
STAGE1_SAFETY_NET = os.environ.get("STAGE1_SAFETY_NET", "false").lower() in ("1", "true", "yes")
print(f"[PavementClassifier] Stage 1 safety net: {'on' if STAGE1_SAFETY_NET else 'off'}")


def _release_cuda_cache() -> None:
    """Hand cached allocator blocks back between inference steps. On Windows
    WDDM an over-grown reserve pages to system RAM instead of failing."""
    try:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

# Sentinel for PavementClassifier.reload(): distinguishes "adapter argument was
# not supplied" from "adapter explicitly set to None (run the base model)".
_UNCHANGED = object()

# Weight storage formats the loader supports. Quantization changes WHAT IS
# STORED, not what is computed: bnb_4bit_compute_dtype is bfloat16, so every
# matmul dequantizes back to bf16 first. Activations, attention, the KV cache
# and the logits we read confidence from are bf16 in all three modes.
QUANTIZATION_LABELS = {
    0: "bf16 (no quantization)",
    4: "4-bit NF4",
    8: "8-bit int8",
}

QUANTIZATION_OPTIONS = [
    {"bits": 0, "label": QUANTIZATION_LABELS[0], "weights_gb": 16.6,
     "note": "Highest fidelity, 16 bits per weight. NOT measured at the "
             "production 1024x1024 image cap: the resolution study ran at "
             "4-bit, peaking at 19.8 GB. bf16 adds ~12 GB of weights, which "
             "very likely exceeds the 24 GB card and drops into the slow "
             "paging regime (~25x slower). Measure before using in production."},
    {"bits": 4, "label": QUANTIZATION_LABELS[4], "weights_gb": 4.3,
     "note": "Production default. NF4 levels sit at the quantiles of a normal "
             "distribution, block-wise over 64 weights, with the block scales "
             "themselves quantized. About 4.13 bits per weight effective."},
    {"bits": 8, "label": QUANTIZATION_LABELS[8], "weights_gb": 8.6,
     "note": "Middle ground. Rarely the right pick: int8 has neither bf16 "
             "fidelity nor the compactness of NF4."},
]


class PavementClassifier:
    """
    Thread-safe pavement distress classifier for Qwen-family VLMs.

    Model-agnostic: the concrete class is resolved from the checkpoint config
    by scripts/model_loader.py, so MODEL_PATH can point at Qwen2.5-VL or
    Qwen3-VL (or any family transformers maps) without code changes.

    Extracts real confidence scores from model logits:
      - Stage 1: softmax probability over "Normal" vs "Distress" tokens
      - Stage 2: geometric mean of per-token probabilities, computed over
        either the DISTRESS_TYPES field only ('field', default) or the whole
        generation ('sequence', legacy). Both are always reported.
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        adapter_path: str = None,
        quantization_bits: int = 4,
    ):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.quantization_bits = quantization_bits   # remember for reload
        self._lock = threading.Lock()
        self._model = None
        self._processor = None
        self._device = None
        self._loaded = False
        self._load_info: dict = {}

        # Cached token IDs for Stage 1 confidence extraction
        self._normal_token_ids = []
        self._distress_token_ids = []

        # Per-type probe (STAGE2_MODE=probe). Rebuilt on every model load,
        # because it holds references to the model object.
        self._prober = None
        self._probe_cfg = None
        self._probe_labels: dict = {}
        self._probe_status: dict = {"requested_mode": STAGE2_MODE, "enabled": False}

        self._load_model(quantization_bits)

    def reload(
        self,
        quantization_bits: int | None = None,
        adapter_path=_UNCHANGED,
    ) -> dict:
        """Rebuild the model in place with a different precision and/or adapter.

        Tears the current model down entirely and reloads from scratch. This is
        deliberately heavier than the peft set_adapter()/load_adapter() path,
        which has edge cases on 4-bit quantized bases (peft #2586 territory),
        and it is the ONLY correct way to change quantization: the weight
        storage format is fixed at load time and cannot be converted in place.

        Takes ~30-60s. The lock serialises against predict_*, so an in-flight
        image blocks the swap rather than corrupting it — but the worker should
        be stopped first so nothing sits waiting on a 60s reload mid-batch.

        quantization_bits:
            None -> keep the current setting
            0    -> bf16, no quantization (~16.6 GB for a 7B, best fidelity)
            4    -> NF4 + double quant (~4.3 GB, production default)
            8    -> int8 (~8.6 GB)
        adapter_path:
            omitted -> keep the current adapter
            None or "" -> base model only
            str     -> directory containing adapter_config.json

        Everything is validated BEFORE the working model is torn down, and a
        failed load restores the previous configuration, so a bad argument can
        never leave the process with no model at all.

        Returns the runtime descriptor (see runtime_info()).
        """
        from datetime import datetime, timezone

        # ---- validate first, mutate nothing ----
        if quantization_bits is None:
            quantization_bits = self.quantization_bits
        if quantization_bits not in (0, 4, 8):
            raise ValueError(
                f"quantization_bits must be 0 (bf16), 4 or 8 - got {quantization_bits!r}"
            )

        if adapter_path is _UNCHANGED:
            new_adapter_path = self.adapter_path
        else:
            new_adapter_path = adapter_path or None
            if new_adapter_path is not None:
                from pathlib import Path
                ap = Path(new_adapter_path)
                if not ap.is_dir():
                    raise ValueError(f"adapter path is not a directory: {new_adapter_path}")
                if not (ap / "adapter_config.json").exists():
                    raise ValueError(
                        f"adapter path missing adapter_config.json: {new_adapter_path}"
                    )

        prev_bits, prev_adapter = self.quantization_bits, self.adapter_path
        if (quantization_bits == prev_bits
                and new_adapter_path == prev_adapter
                and self._loaded):
            # Nothing to do. Reloading anyway costs a minute of GPU time and
            # drops the worker for no reason.
            info = self.runtime_info()
            info["reloaded"] = False
            info["reason"] = "already running this configuration"
            return info

        with self._lock:
            print(f"[PavementClassifier] Reload: quantization {prev_bits} -> "
                  f"{quantization_bits}, adapter {prev_adapter!r} -> {new_adapter_path!r}")

            # Free the old model BEFORE loading the new one - otherwise we need
            # 2x VRAM for the swap moment. Drop refs + empty cache. The prober
            # holds its own reference to the model (and a float32 copy of the
            # Yes/No head rows): if it survived, the old weights could not be
            # freed and the reload would need two models' worth of VRAM.
            self._prober = None
            self._probe_cfg = None
            self._model = None
            self._processor = None
            self._loaded = False
            try:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            self.adapter_path = new_adapter_path
            self.quantization_bits = quantization_bits
            try:
                self._load_model(quantization_bits)
            except Exception:
                # The requested configuration did not load. Restore the one
                # that was working so the API does not come back with a dead
                # model, then re-raise so the operator sees the real error.
                print(f"[PavementClassifier] Reload FAILED at {quantization_bits}-bit - "
                      f"restoring previous configuration ({prev_bits}-bit)")
                self.adapter_path = prev_adapter
                self.quantization_bits = prev_bits
                self._load_model(prev_bits)
                raise

        info = self.runtime_info()
        info["reloaded"] = True
        info["previous"] = {"quantization_bits": prev_bits, "adapter_path": prev_adapter}
        info["loaded_at"] = datetime.now(timezone.utc).isoformat()
        return info

    def reload_adapter(self, new_adapter_path: str | None) -> dict:
        """Backwards-compatible wrapper: swap the adapter, keep the precision.

        Kept because /operator/adapters/switch and its dashboard control have
        already shipped against this signature.
        """
        return self.reload(quantization_bits=None, adapter_path=new_adapter_path)

    def runtime_info(self) -> dict:
        """Everything an operator needs to know about what is loaded right now.

        `quantization_bits` is the REQUESTED setting; `quantization_effective`
        is what the loader actually ended up with, which differs when the OOM
        fallback ladder fires. The dashboard shows both so a silent downgrade
        is visible rather than hidden.
        """
        info = dict(self._load_info)
        vram = {}
        try:
            if torch.cuda.is_available():
                free_b, total_b = torch.cuda.mem_get_info()
                vram = {
                    "total_gb": round(total_b / 1024 ** 3, 2),
                    "free_gb": round(free_b / 1024 ** 3, 2),
                    "used_gb": round((total_b - free_b) / 1024 ** 3, 2),
                    "gpu_name": torch.cuda.get_device_name(0),
                }
        except Exception:
            pass
        return {
            "model_path": self.model_path,
            "adapter_path": self.adapter_path,
            "quantization_bits": self.quantization_bits,
            "quantization_label": QUANTIZATION_LABELS.get(
                self.quantization_bits, f"{self.quantization_bits}-bit"),
            "quantization_effective": info.get("quantization_bits", self.quantization_bits),
            "oom_fallback_used": info.get("oom_fallback_used", False),
            "model_family": info.get("model_type", "unknown"),
            "model_class": info.get("model_class", "unknown"),
            "max_image_pixels": info.get("max_pixels", 0),
            "stage2_confidence_mode": STAGE2_CONFIDENCE_MODE,
            "stage2_mode": STAGE2_MODE,
            "stage2_mode_effective": ("probe" if self._prober is not None and STAGE2_MODE == "probe"
                                      else "shadow" if self._prober is not None and STAGE2_MODE == "shadow"
                                      else "generate"),
            "stage1_safety_net": self.safety_net_active,
            "stage2_probe": self.probe_status,
            "prompts_version": _ACTIVE_PROMPTS,
            "device": self._device,
            "is_loaded": self._loaded,
            "vram": vram,
            "options": QUANTIZATION_OPTIONS,
        }

    def _load_model(self, quantization_bits: int) -> None:
        """Load model, processor, and cache target token IDs.

        Delegates to scripts/model_loader.load_vlm so the production path and
        the evaluation path (scripts/03_baseline_eval.py) load models through
        identical code — otherwise an A/B comparison between two models could
        be confounded by a difference in how they were loaded.
        """
        # Cap visual tokens at the processor level. RoadSide / Cloudinary photos
        # can be 3000x3000+ — without this, attention O(n^2) blows past 24GB VRAM
        # for high-res inputs (confirmed empirically with a 3456x3456 image).
        # ~2200px max dim keeps inference under 7s/img on A5000 4-bit.
        # Match the values used in scripts/03_baseline_eval.py for consistency.
        max_pixels = int(os.environ.get("MAX_IMAGE_PIXELS", str(2200 * 2200)))
        min_pixels = int(os.environ.get("MIN_IMAGE_PIXELS", str(256 * 28)))

        self._model, self._processor, self._load_info = load_vlm(
            model_path=self.model_path,
            quantization_bits=quantization_bits,
            adapter_path=self.adapter_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            # Self-test is worth 2s at startup: it catches a broken
            # template/processor pairing before any citizen photo is marked
            # 'failed' in the DB. Disable only for fast local iteration.
            run_self_test=os.environ.get("SKIP_MODEL_SELFTEST", "").lower()
            not in ("1", "true", "yes"),
        )
        self._device = self._load_info["device"]
        self._loaded = True

        # Cache token IDs for "Normal" and "Distress" for fast confidence extraction.
        # A word may tokenize into multiple sub-tokens; we use the FIRST sub-token
        # as the discriminating signal (the model decides on the first token).
        tokenizer = self._processor.tokenizer
        self._normal_token_ids = tokenizer.encode("Normal", add_special_tokens=False)
        self._distress_token_ids = tokenizer.encode("Distress", add_special_tokens=False)
        if not self._normal_token_ids or not self._distress_token_ids:
            print("[PavementClassifier] WARNING: Failed to encode Normal/Distress tokens — Stage 1 confidence will be 0.0")
        print(f"[PavementClassifier] Token IDs — Normal: {self._normal_token_ids}, Distress: {self._distress_token_ids}")
        print(f"[PavementClassifier] Model loaded on: {self._device}")
        self._init_probe()

    def _init_probe(self) -> None:
        """Build the per-type prober when STAGE2_MODE is 'probe' or 'shadow'
        or the Stage 1 safety net is on, or record why not.

        Never raises: a probe that cannot be trusted is switched off and the
        pipeline keeps running on the free-form list. The fast shared-prefix
        path is checked against the slow one-pass-per-question path before it
        is enabled.
        """
        self._prober = None
        self._probe_cfg = None
        self._probe_labels = {}
        st = {"requested_mode": STAGE2_MODE, "enabled": False,
              "applies_labels": STAGE2_MODE == "probe",
              "stage1_safety_net_requested": STAGE1_SAFETY_NET,
              "stage1_safety_net_active": False,
              "config_path": STAGE2_PROBE_CONFIG}
        if STAGE2_MODE == "generate" and not STAGE1_SAFETY_NET:
            st["reason"] = "STAGE2_MODE=generate and STAGE1_SAFETY_NET off"
            self._probe_status = st
            return
        try:
            from scripts import stage2_probe_rules as rules
            from scripts.stage2_probe import TypeProber, all_probe_types
            cfg = rules.load_config(STAGE2_PROBE_CONFIG)
            types = all_probe_types()
            rules.validate_config(cfg, known_keys={t.key for t in types})
            prober = TypeProber(self._model, self._processor, types,
                                cfg["variant"]["system_style"],
                                cfg["variant"]["question_style"])
            parity = prober.parity_check()
            st["parity"] = parity
            if not parity["ok"]:
                raise RuntimeError(f"fast and slow probe paths disagree: {parity}")
            self._prober = prober
            self._probe_cfg = cfg
            self._probe_labels = {t.key: t.output_label for t in types}
            sn_active = STAGE1_SAFETY_NET and bool(cfg.get("stage1_safety_net"))
            st.update(enabled=True, reason=None, variant=cfg["variant"],
                      provenance=cfg.get("provenance"),
                      stage1_safety_net_active=sn_active)
            if STAGE1_SAFETY_NET and not sn_active:
                st["stage1_safety_net_reason"] = "config has no stage1_safety_net section"
            print(f"[PavementClassifier] Stage 2 probe ENABLED in {STAGE2_MODE!r} mode "
                  f"({cfg['variant']}, parity max diff {parity['max_abs_diff']}, "
                  f"{parity['fast_s']}s per image; safety net "
                  f"{'on' if sn_active else 'off'})")
        except Exception as e:
            st["reason"] = f"{type(e).__name__}: {e}"
            print(f"[PavementClassifier] WARNING: Stage 2 probe DISABLED, using the "
                  f"free-form list instead - {st['reason']}")
        finally:
            _release_cuda_cache()
        self._probe_status = st

    @property
    def probe_status(self) -> dict:
        return dict(self._probe_status)

    def _build_inputs(self, image: Image.Image, system_prompt: str, user_prompt: str):
        """Build model inputs from image and prompts. Returns (inputs, input_length)."""
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt.replace("<image>\n", "")},
                ],
            },
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        input_length = inputs["input_ids"].shape[1]
        return inputs, input_length

    def _run_inference_with_scores(
        self, image: Image.Image, system_prompt: str, user_prompt: str,
        max_new_tokens: int = 200,
    ) -> tuple[str, tuple, torch.Tensor]:
        """
        Run inference and return (decoded_text, score_tensors, generated_ids).

        Each score tensor has shape (vocab_size,) — one per generated token.
        Stage 0/1 pass smaller max_new_tokens (8-20) for speed.
        """
        inputs, input_length = self._build_inputs(image, system_prompt, user_prompt)

        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                output_scores=True,
                return_dict_in_generate=True,
            )

        # Decode text
        generated_ids = output.sequences[:, input_length:]
        response = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Extract per-token logit tensors (tuple of tensors, one per step)
        scores = output.scores  # tuple of (batch=1, vocab_size) tensors

        return response.strip(), scores, generated_ids[0]

    def _compute_stage1_confidence(self, scores: tuple, generated_ids: torch.Tensor) -> float:
        """
        Compute Stage 1 confidence from the FIRST generated token's logits.

        Takes the softmax probability over the "Normal" and "Distress" first-sub-tokens,
        then returns the probability of whichever the model actually chose.

        Returns a float in [0.0, 1.0]. This is a RAW model probability —
        no inflation, no heuristics, no manipulation.
        """
        if not scores or len(scores) == 0:
            return 0.0

        first_logits = scores[0][0]  # shape: (vocab_size,)

        # Get the first sub-token IDs
        normal_id = self._normal_token_ids[0] if self._normal_token_ids else None
        distress_id = self._distress_token_ids[0] if self._distress_token_ids else None

        if normal_id is None or distress_id is None:
            return 0.0

        # Extract logits for the two target tokens
        target_logits = torch.tensor(
            [first_logits[normal_id].item(), first_logits[distress_id].item()],
            dtype=torch.float32,
        )

        # Softmax over just these two classes
        probs = torch.softmax(target_logits, dim=0)

        # Which did the model actually generate?
        actual_first_token = generated_ids[0].item()

        if actual_first_token == normal_id:
            confidence = probs[0].item()
        elif actual_first_token == distress_id:
            confidence = probs[1].item()
        else:
            # Model generated something other than Normal/Distress as first token.
            # Fall back to the max token probability in the full vocabulary.
            full_probs = torch.softmax(first_logits.float(), dim=0)
            confidence = full_probs[actual_first_token].item()

        return round(confidence, 4)

    # ------------------------------------------------------------------
    # Stage 2 confidence
    # ------------------------------------------------------------------
    # The maths lives in scripts/confidence.py so that production and the
    # offline evaluation scripts cannot drift apart. These methods stay as
    # thin delegates because scripts/validate_all.py asserts their presence
    # here, and because the call sites read better with them.

    def _token_log_probs(self, scores: tuple, generated_ids: torch.Tensor) -> list:
        """Per-token log-probability of each generated token. See confidence.py."""
        return _conf.token_log_probs(scores, generated_ids)

    @staticmethod
    def _geomean(log_probs: list) -> float:
        """Geometric mean of probabilities = exp(mean(log p_i))."""
        return _conf.geomean(log_probs)

    def _compute_stage2_sequence(self, scores, generated_ids) -> float:
        return _conf.sequence_confidence(scores, generated_ids)

    def _compute_sequence_confidence(self, scores: tuple,
                                     generated_ids: torch.Tensor) -> float:
        """Geometric mean over EVERY generated token (legacy Stage 2 metric).

        Includes the free-text DESCRIPTION, which is high-entropy prose; this
        is why it was measured anti-correlated with correctness. Retained for
        continuity and for the paper comparison table, not used as the gate.
        """
        return _conf.sequence_confidence(scores, generated_ids)

    def _find_field_token_span(self, generated_ids: torch.Tensor,
                               field: str = "DISTRESS_TYPES"):
        """Token span covering the VALUE of a structured output field.

        None when absent or under two tokens; the caller then falls back to
        whole-sequence confidence rather than trusting a one-token estimate.
        """
        return _conf.find_field_token_span(
            self._processor.tokenizer, generated_ids, field)

    def _compute_field_confidence(self, scores: tuple, generated_ids: torch.Tensor,
                                  field: str = "DISTRESS_TYPES") -> tuple:
        """Confidence over the classification tokens only.

        Returns (confidence, used_field_span); on a failed span lookup the
        whole-sequence value comes back with used_field_span=False so the
        fallback is recorded rather than silent.
        """
        return _conf.field_confidence(
            self._processor.tokenizer, scores, generated_ids, field)

    def _stage2_confidences(self, scores: tuple, generated_ids: torch.Tensor,
                            parsed: dict) -> dict:
        """Every Stage 2 confidence for one generation, plus the gated value.

        All of them are returned on every call, whatever the mode, so the gate
        can be re-calibrated from stored rows without re-running inference.
        """
        seq = self._compute_sequence_confidence(scores, generated_ids)
        fld, field_ok = self._compute_field_confidence(scores, generated_ids)
        type_confs = _conf.type_confidences(
            self._processor.tokenizer, scores, generated_ids)
        types = parsed.get("distress_types") or []
        entry = _conf.primary_entry(
            type_confs, types[0] if types else None, canonicalize_to_irc)
        # No matchable label (a "Normal" answer, a keyword-fallback parse):
        # use the whole-field value and record that the fallback happened.
        primary = entry["joint"] if entry else fld
        # 'probe' is resolved later in predict_stage2 (it needs the probe's
        # output); until then it gates like 'field', its fallback.
        gated = {"sequence": seq, "field": fld, "primary": primary}.get(
            STAGE2_CONFIDENCE_MODE, fld)
        return {
            "stage2_confidence": gated,
            "stage2_confidence_primary": primary,
            "stage2_primary_span_found": entry is not None,
            "stage2_confidence_field": fld,
            "stage2_confidence_sequence": seq,
            "stage2_field_span_found": field_ok,
            # Per label, in output order. Labels after the first are
            # conditional on the ones before them - see confidence.py.
            "stage2_type_confidences": type_confs,
        }

    def predict_is_pavement(self, image: Image.Image) -> dict:
        """
        Stage 0 — pavement pre-filter. Runs BEFORE Stage 1.

        Designed to be CONSERVATIVE: only rejects images that are clearly
        NOT road/pavement. Damaged, blurry, partial, or low-quality pavement
        photos always pass through. False rejects cost real data; false
        accepts only cost a downstream Normal/Unknown classification.

        Returns:
            {
                'is_pavement': bool,        # True if 'yes' OR 'unsure'
                'decision':    str,         # 'yes' | 'no' | 'unsure'
                'raw':         str,         # full VLM response
                'time_ms':     float,
            }
        Thread-safe.
        """
        image = image.convert("RGB")
        with self._lock:
            t0 = time.time()
            raw, _scores, _ids = self._run_inference_with_scores(
                image, PAVEMENT_FILTER_SYSTEM_PROMPT,
                PAVEMENT_FILTER_USER_PROMPT, max_new_tokens=8,
            )
            dt = (time.time() - t0) * 1000.0
        decision = parse_pavement_filter_response(raw)
        # 'unsure' is treated as pavement — never reject on uncertainty.
        is_pavement = decision in ("yes", "unsure")
        return {
            "is_pavement": is_pavement,
            "decision": decision,
            "raw": raw,
            "time_ms": round(dt, 1),
        }

    def predict_stage1(self, image: Image.Image) -> dict:
        """
        Run Stage 1 only: binary detection (Normal vs Distress).

        Returns dict with stage 1 fields only. Thread-safe.
        """
        image = image.convert("RGB")

        with self._lock:
            s1_start = time.time()
            # Stage 1 answers with a single word ("Normal" / "Distress"), so a
            # 200-token budget buys nothing and costs real time: measured 4.7s
            # at max_new_tokens=200 vs 1.2s at a small budget, for the same
            # 2-token output. Confidence reads only the FIRST token's logits,
            # so the smaller budget cannot change the score. A model that
            # rambles past the budget produces an unparseable response, which
            # already defaults to Distress + 0.0 confidence -> expert review.
            stage1_raw, s1_scores, s1_gen_ids = self._run_inference_with_scores(
                image, STAGE1_SYSTEM_PROMPT, STAGE1_USER_PROMPT,
                max_new_tokens=STAGE1_MAX_NEW_TOKENS,
            )
            s1_time = (time.time() - s1_start) * 1000

            stage1_id = parse_stage1_response(stage1_raw)

            # -1 means unparseable — don't silently treat as Normal.
            # Flag for expert review with zero confidence instead.
            if stage1_id == -1:
                print(f"[PavementClassifier] WARNING: Unparseable stage1 response: {stage1_raw!r}")
                is_distressed = True  # Err on side of caution
                s1_confidence = 0.0
            else:
                is_distressed = stage1_id == 1
                s1_confidence = self._compute_stage1_confidence(s1_scores, s1_gen_ids)

        return {
            "is_distressed": is_distressed,
            "stage1_label": "Distress" if is_distressed else "Normal",
            "stage1_confidence": s1_confidence,
            "needs_expert_review": s1_confidence < CONFIDENCE_THRESHOLD,
            "stage1_time_ms": round(s1_time, 1),
            "stage1_raw": stage1_raw,
        }

    def _stage2_result_is_uninformative(self, parsed: dict) -> bool:
        """
        Decide whether the parsed Stage 2 result is too vague to commit.

        Triggers cascade fallback (re-run with adapter disabled, base model only)
        if any of:
          - parsed.distress_types is empty
          - parsed.distress_types == ['Unknown']  (parser already filtered 'Other Distress')
          - any element is 'Unknown' / 'Other Distress' / 'Other'

        The fallback restores the model's pretrained zero-shot taxonomy
        (Block Crack, Raveling, Weathering, Edge Crack, etc.) which the
        QLoRA fine-tune narrowed away. We only run the second pass when the
        primary pass failed to commit to a specific type — saves ~5s/image
        on the common case where the adapter does its job.
        """
        types = parsed.get("distress_types") or []
        if not types:
            return True
        bad = {"unknown", "other distress", "other", "unknown distress"}
        return all(t.strip().lower() in bad for t in types)

    def _has_active_adapter(self) -> bool:
        """True iff a LoRA adapter is currently attached to the model."""
        from peft import PeftModel
        return isinstance(self._model, PeftModel) and self.adapter_path is not None

    def predict_stage2(self, image: Image.Image) -> dict:
        """
        Run Stage 2 with adapter-cascade fallback.

        First pass: full model (with LoRA adapter active if present). The
        fine-tuned adapter is highly accurate on the four RDD classes
        (D00/D10/D20/D40) but had its broader taxonomy capability narrowed
        during fine-tuning.

        Cascade: if the first pass returns 'Unknown' / empty / 'Other
        Distress', re-run the SAME image with the adapter temporarily
        disabled (PeftModel.disable_adapter context manager). The base model
        retains the pretrained zero-shot taxonomy and can name types like
        Block Crack, Raveling, Weathering — useful for real-world photos
        outside the RDD distribution.

        Returns dict with stage 2 fields plus 'stage2_used_fallback' bool.
        Thread-safe.
        """
        image = image.convert("RGB")

        with self._lock:
            s2_start = time.time()

            # Pass 1: with adapter (if loaded)
            stage2_raw, s2_scores, s2_gen_ids = self._run_inference_with_scores(
                image, STAGE2_SYSTEM_PROMPT, STAGE2_USER_PROMPT
            )
            parsed = parse_stage2_response(stage2_raw)
            confs = self._stage2_confidences(s2_scores, s2_gen_ids, parsed)

            used_fallback = False
            primary_raw = stage2_raw

            # Pass 2 (fallback): only fires if pass 1 is uninformative AND
            # we actually have an adapter to disable. Without an adapter,
            # there's nothing to gain from a second pass.
            if (
                self._has_active_adapter()
                and self._stage2_result_is_uninformative(parsed)
            ):
                try:
                    print(f"[PavementClassifier] Stage 2 cascade: pass 1 was '{parsed.get('distress_types')}', re-running with adapter disabled")
                    with self._model.disable_adapter():
                        fb_raw, fb_scores, fb_gen_ids = self._run_inference_with_scores(
                            image, STAGE2_SYSTEM_PROMPT, STAGE2_USER_PROMPT
                        )
                    fb_parsed = parse_stage2_response(fb_raw)

                    # Only swap if fallback gave something more specific
                    if not self._stage2_result_is_uninformative(fb_parsed):
                        used_fallback = True
                        stage2_raw = fb_raw
                        s2_scores = fb_scores
                        s2_gen_ids = fb_gen_ids
                        parsed = fb_parsed
                        confs = self._stage2_confidences(fb_scores, fb_gen_ids, parsed)
                        print(f"[PavementClassifier] Stage 2 cascade: fallback recovered '{fb_parsed.get('distress_types')}'")
                except Exception as e:
                    # Fallback errors are non-fatal — keep the primary result.
                    print(f"[PavementClassifier] Stage 2 cascade fallback FAILED: {e}")

            # Per-type probe (STAGE2_MODE 'probe' or 'shadow'). Any failure
            # keeps the free-form answer for this image and says so in the row.
            probe_out, probe_err = None, None
            if self._prober is not None and STAGE2_MODE in ("probe", "shadow"):
                _release_cuda_cache()  # the generation's reserve, before the probe
                try:
                    from scripts import stage2_probe_rules as rules
                    t_p = time.time()
                    res = self._prober.probe(image)
                    p = {r["key"]: r["p_yes"] for r in res}
                    probe_out = rules.combine(parsed["distress_types"], p,
                                              self._probe_cfg, self._probe_labels)
                    probe_out["p"] = {k: round(v, 5) for k, v in p.items()}
                    probe_out["min_yes_no_mass"] = round(min(r["yes_no_mass"] for r in res), 5)
                    probe_out["time_ms"] = round((time.time() - t_p) * 1000, 1)
                    probe_out["variant"] = self._probe_cfg["variant"]
                except Exception as e:
                    probe_out = None
                    probe_err = f"{type(e).__name__}: {e}"
                    print(f"[PavementClassifier] Stage 2 probe FAILED on this image, "
                          f"keeping the free-form list - {probe_err}")
                finally:
                    _release_cuda_cache()

            s2_time = (time.time() - s2_start) * 1000

        generated_types = parsed["distress_types"]
        # Shadow mode records the probe's answer but does not use it.
        applied = probe_out is not None and STAGE2_MODE == "probe"
        probe_conf = None
        if probe_out is not None:
            probe_out["applied"] = applied
            probe_conf = probe_out["confidence"]
            if not probe_out["types"]:
                # Stage 1 said distressed but no type survived. Never
                # auto-accept "distressed, type unknown": force review.
                probe_conf = 0.0
                probe_out["forced_review"] = "no distress type reported"
        if applied:
            types = probe_out["types"]
            indicators = probe_out["indicators"]
        else:
            types, indicators = generated_types, []
        confs = dict(confs)
        confs["stage2_confidence_probe"] = probe_conf
        if STAGE2_CONFIDENCE_MODE == "probe":
            if applied and probe_conf is not None:
                confs["stage2_confidence"] = probe_conf
                conf_source = "probe"
            else:
                confs["stage2_confidence"] = confs["stage2_confidence_field"]
                conf_source = "field (probe unavailable)"
        else:
            conf_source = STAGE2_CONFIDENCE_MODE
        return {
            "distress_types": types,
            # The first label is shown as the main one and the rest beside it.
            # Free-form mode: the model's listing order. Probe mode: the type
            # it is most confident is present. Neither is a measured "most
            # prominent" (nothing here measures extent): utils_v2_prompts.py.
            "primary_distress_type": types[0] if types else None,
            "secondary_distress_types": types[1:],
            "condition_indicators": indicators,
            "severity": parsed["severity"],
            "description": parsed["description"],
            # Every metric is always reported so the review threshold can be
            # calibrated from real data instead of guessed, and so switching
            # STAGE2_CONFIDENCE_MODE never loses the other numbers.
            **confs,
            "stage2_confidence_mode": STAGE2_CONFIDENCE_MODE,
            "stage2_confidence_source": conf_source,
            # Which path produced distress_types for THIS image. In shadow
            # mode this is 'generate' and stage2_probe.applied is false.
            "stage2_mode": "probe" if applied else "generate",
            "stage2_probe_mode": STAGE2_MODE if probe_out is not None else None,
            "stage2_generated_types": generated_types,
            "stage2_probe": probe_out,
            "stage2_probe_error": probe_err,
            "stage2_time_ms": round(s2_time, 1),
            "stage2_raw": stage2_raw,
            "stage2_used_fallback": used_fallback,
            "stage2_primary_raw": primary_raw if used_fallback else None,
        }

    @property
    def safety_net_active(self) -> bool:
        return (STAGE1_SAFETY_NET and self._prober is not None
                and bool((self._probe_cfg or {}).get("stage1_safety_net")))

    def stage1_safety_check(self, image: Image.Image) -> dict | None:
        """Stage 1 said Normal - does the probe see distress anyway?

        Returns None when the safety net is off. Otherwise a dict with
        `flagged` (route to expert review), the strongest type, its P(yes),
        the threshold and every type's P(yes). A failure returns
        {"flagged": False, "error": ...}: the Normal decision then stands
        exactly as it would without the safety net, and the row says why.
        Thread-safe.
        """
        if not self.safety_net_active:
            return None
        from scripts import stage2_probe_rules as rules
        image = image.convert("RGB")
        with self._lock:
            t0 = time.time()
            try:
                res = self._prober.probe(image)
                p = {r["key"]: r["p_yes"] for r in res}
                out = rules.safety_net(p, self._probe_cfg) or {"flagged": False}
                out["p"] = {k: round(v, 5) for k, v in p.items()}
            except Exception as e:
                out = {"flagged": False, "error": f"{type(e).__name__}: {e}"}
                print(f"[PavementClassifier] Stage 1 safety net FAILED on this image "
                      f"(Normal decision stands) - {out['error']}")
            finally:
                _release_cuda_cache()
            out["time_ms"] = round((time.time() - t0) * 1000, 1)
        return out

    def predict(self, image: Image.Image) -> dict:
        """
        Run the full two-stage classification pipeline with real confidence scores.

        Returns dict matching ClassificationResponse schema.
        Internally calls predict_stage1() and predict_stage2().
        """
        total_start = time.time()

        # --- Stage 1: Binary Detection ---
        s1 = self.predict_stage1(image)

        result = {
            "is_distressed": s1["is_distressed"],
            "stage1_label": s1["stage1_label"],
            "stage1_confidence": s1["stage1_confidence"],
            "distress_types": [],
            "severity": "None",
            "description": "",
            "stage2_confidence": 0.0,
            "needs_expert_review": s1["needs_expert_review"],
            "stage1_time_ms": s1["stage1_time_ms"],
            "stage2_time_ms": 0.0,
            "stage1_raw": s1["stage1_raw"],
            "stage2_raw": "",
        }

        # --- Stage 2: Type Classification (only if distressed) ---
        if s1["is_distressed"]:
            s2 = self.predict_stage2(image)
            result["distress_types"] = s2["distress_types"]
            result["severity"] = s2["severity"]
            result["description"] = s2["description"]
            result["stage2_confidence"] = s2["stage2_confidence"]
            result["stage2_time_ms"] = s2["stage2_time_ms"]
            result["stage2_raw"] = s2["stage2_raw"]
            result["stage2_used_fallback"] = s2.get("stage2_used_fallback", False)
            result["stage2_primary_raw"] = s2.get("stage2_primary_raw")
            result["primary_distress_type"] = s2.get("primary_distress_type")
            result["secondary_distress_types"] = s2.get("secondary_distress_types") or []
            result["stage2_confidence_primary"] = s2.get("stage2_confidence_primary")
            result["stage2_confidence_field"] = s2.get("stage2_confidence_field")
            result["stage2_confidence_sequence"] = s2.get("stage2_confidence_sequence")
            result["stage2_confidence_mode"] = s2.get("stage2_confidence_mode")
            result["stage2_confidence_probe"] = s2.get("stage2_confidence_probe")
            result["stage2_mode"] = s2.get("stage2_mode")
            result["condition_indicators"] = s2.get("condition_indicators") or []
            result["stage2_generated_types"] = s2.get("stage2_generated_types")
            result["stage2_probe"] = s2.get("stage2_probe")
            result["stage2_probe_error"] = s2.get("stage2_probe_error")

            # Flag for expert review if EITHER stage has low confidence
            result["needs_expert_review"] = (
                s1["stage1_confidence"] < CONFIDENCE_THRESHOLD
                or s2["stage2_confidence"] < CONFIDENCE_THRESHOLD
            )
        else:
            # Stage 1 said Normal: the safety net may route it to a human.
            sn = self.stage1_safety_check(image)
            result["stage1_safety_net"] = sn
            if sn and sn.get("flagged"):
                result["needs_expert_review"] = True

        total_time = (time.time() - total_start) * 1000
        result["processing_time_ms"] = round(total_time, 1)

        return result

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device(self) -> str:
        return self._device or "unknown"

    @property
    def has_adapter(self) -> bool:
        return self.adapter_path is not None

    @property
    def load_info(self) -> dict:
        """What actually got loaded — family, class, real quantization, pixel
        budget, whether the OOM fallback fired, self-test result. Reported by
        /health so the operator sees the true config, not the requested one."""
        return dict(self._load_info)

    @property
    def confidence_mode(self) -> str:
        return STAGE2_CONFIDENCE_MODE


# ============================================================
# Global singleton
# ============================================================
_classifier: PavementClassifier = None


def get_classifier(
    model_path: str = None,
    adapter_path: str = None,
    quantization_bits: int = 4,
) -> PavementClassifier:
    """Get or create the global classifier singleton.

    Adapter loading respects DISABLE_ADAPTER env var:
      DISABLE_ADAPTER=true    → no LoRA loaded, pure base model (production default
                                 for the Improved Baseline pipeline)
      DISABLE_ADAPTER unset   → ADAPTER_PATH env var governs (legacy behaviour;
                                 used for paper experiments / A-B runs only)
    """
    global _classifier
    if _classifier is None:
        model_path = model_path or os.environ.get(
            "MODEL_PATH", "Qwen/Qwen2.5-VL-7B-Instruct"
        )
        disable_adapter = os.environ.get("DISABLE_ADAPTER", "").lower() in ("1", "true", "yes")
        if disable_adapter:
            adapter_path = None
            print("[PavementClassifier] DISABLE_ADAPTER=true -> running pure base model "
                  "(Improved Baseline pipeline)")
        else:
            adapter_path = adapter_path or os.environ.get("ADAPTER_PATH", None)
            if adapter_path == "":
                adapter_path = None
        quant = int(os.environ.get("QUANTIZATION_BITS", str(quantization_bits)))
        _classifier = PavementClassifier(model_path, adapter_path, quant)
    return _classifier
