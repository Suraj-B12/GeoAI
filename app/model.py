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
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.utils import (
    CONFIDENCE_THRESHOLD,
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
else:
    from scripts.utils import (
        STAGE1_SYSTEM_PROMPT,
        STAGE1_USER_PROMPT,
        STAGE2_SYSTEM_PROMPT,
        STAGE2_USER_PROMPT,
    )
    _ACTIVE_PROMPTS = "v1_plain"
print(f"[PavementClassifier] Active prompt set: {_ACTIVE_PROMPTS}")


class PavementClassifier:
    """
    Thread-safe pavement distress classifier using Qwen2.5-VL.

    Extracts real confidence scores from model logits:
      - Stage 1: softmax probability over "Normal" vs "Distress" tokens
      - Stage 2: geometric mean of per-token probabilities (sequence confidence)
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

        # Cached token IDs for Stage 1 confidence extraction
        self._normal_token_ids = []
        self._distress_token_ids = []

        self._load_model(quantization_bits)

    def reload_adapter(self, new_adapter_path: str | None) -> dict:
        """Hot-swap the LoRA adapter without restarting the server.

        Tears down the current model entirely (PeftModel + base model)
        and rebuilds from scratch with the new adapter. Safer than
        peft's set_adapter()/load_adapter() which has edge cases on
        4-bit quantized bases (peft #2586 territory). Takes ~30s.

        new_adapter_path:
            - None or "" -> base model only (no adapter)
            - path to dir with adapter_config.json -> load that adapter

        Returns: {"adapter_path": str|None, "device": str, "loaded_at": iso}
        """
        from datetime import datetime, timezone

        new_adapter_path = new_adapter_path or None
        if new_adapter_path is not None:
            # Validate the directory has the right files BEFORE tearing
            # down the working model. A bad path here would otherwise leave
            # us with no model loaded.
            from pathlib import Path
            p = Path(new_adapter_path)
            if not p.is_dir():
                raise ValueError(f"adapter path is not a directory: {new_adapter_path}")
            if not (p / "adapter_config.json").exists():
                raise ValueError(
                    f"adapter path missing adapter_config.json: {new_adapter_path}"
                )

        with self._lock:
            print(f"[PavementClassifier] Hot-swap adapter: {self.adapter_path!r} -> {new_adapter_path!r}")

            # Free the old model BEFORE loading the new one — otherwise we
            # need 2x VRAM for the swap moment. Drop refs + empty cache.
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
            self._load_model(self.quantization_bits)

        return {
            "adapter_path": self.adapter_path,
            "device": self._device,
            "loaded_at": datetime.now(timezone.utc).isoformat(),
        }

    def _load_model(self, quantization_bits: int) -> None:
        """Load model, processor, and cache target token IDs."""
        print(f"[PavementClassifier] Loading model: {self.model_path}")
        print(f"[PavementClassifier] Quantization: {quantization_bits}-bit")

        # Reduce CUDA memory fragmentation for large model inference
        os.environ.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF",
            "max_split_size_mb:512,garbage_collection_threshold:0.9",
        )

        if quantization_bits == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        elif quantization_bits == 8:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            bnb_config = None

        # Select best attention implementation dynamically:
        # Flash Attention 2 > SDPA > Eager
        attn_impl = "sdpa"  # safe default (PyTorch native)
        try:
            import flash_attn  # noqa: F401
            if torch.cuda.is_available():
                cc = torch.cuda.get_device_properties(0).major
                if cc >= 8:  # Ampere+ (A6000, RTX 30xx/40xx)
                    attn_impl = "flash_attention_2"
                    print(f"[PavementClassifier] Using Flash Attention 2 (compute {cc}.x)")
        except ImportError:
            print("[PavementClassifier] flash-attn not installed, using SDPA")

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=attn_impl,
        )

        if self.adapter_path:
            from peft import PeftModel
            print(f"[PavementClassifier] Loading adapter: {self.adapter_path}")
            self._model = PeftModel.from_pretrained(self._model, self.adapter_path)
            # Do NOT call merge_and_unload() on a quantized model —
            # known peft bug #2586 produces broken weights when merging
            # QLoRA adapters onto a 4-bit base. Keep as PeftModel for inference.

        # Cap visual tokens at the processor level. RoadSide / Cloudinary photos
        # can be 3000x3000+ — without this, attention O(n^2) blows past 24GB VRAM
        # for high-res inputs (confirmed empirically with a 3456x3456 image).
        # ~2200px max dim keeps inference under 7s/img on A5000 4-bit.
        # Match the values used in scripts/03_baseline_eval.py for consistency.
        max_pixels = int(os.environ.get("MAX_IMAGE_PIXELS", str(2200 * 2200)))
        min_pixels = int(os.environ.get("MIN_IMAGE_PIXELS", str(256 * 28)))
        self._processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        print(f"[PavementClassifier] Processor pixel limits: min={min_pixels}, max={max_pixels}")
        self._model.eval()
        self._device = str(next(self._model.parameters()).device)
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
        self, image: Image.Image, system_prompt: str, user_prompt: str
    ) -> tuple[str, tuple, torch.Tensor]:
        """
        Run inference and return (decoded_text, score_tensors, generated_ids).

        Each score tensor has shape (vocab_size,) — one per generated token.
        """
        inputs, input_length = self._build_inputs(image, system_prompt, user_prompt)

        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=200,
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

    def _compute_sequence_confidence(self, scores: tuple, generated_ids: torch.Tensor) -> float:
        """
        Compute sequence-level confidence as the geometric mean of per-token probabilities.

        This is equivalent to exp(mean(log(p_i))) and represents how "sure" the model
        was about the entire generated response. Range: [0.0, 1.0].

        No inflation, no manipulation — pure model probability.
        """
        if not scores or len(scores) == 0:
            return 0.0

        log_probs = []
        num_tokens = min(len(scores), generated_ids.shape[0])

        for i in range(num_tokens):
            logits = scores[i][0].float()  # (vocab_size,)
            token_id = generated_ids[i].item()

            # Skip special/padding tokens
            if token_id <= 0:
                continue

            log_prob = torch.log_softmax(logits, dim=0)[token_id].item()

            # Guard against -inf
            if math.isfinite(log_prob):
                log_probs.append(log_prob)

        if not log_probs:
            return 0.0

        # Geometric mean of probabilities = exp(mean of log-probs)
        avg_log_prob = sum(log_probs) / len(log_probs)
        confidence = math.exp(avg_log_prob)

        # Clamp to [0, 1] (should already be, but safety)
        return round(max(0.0, min(1.0, confidence)), 4)

    def predict_stage1(self, image: Image.Image) -> dict:
        """
        Run Stage 1 only: binary detection (Normal vs Distress).

        Returns dict with stage 1 fields only. Thread-safe.
        """
        image = image.convert("RGB")

        with self._lock:
            s1_start = time.time()
            stage1_raw, s1_scores, s1_gen_ids = self._run_inference_with_scores(
                image, STAGE1_SYSTEM_PROMPT, STAGE1_USER_PROMPT
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
            s2_confidence = self._compute_sequence_confidence(s2_scores, s2_gen_ids)
            parsed = parse_stage2_response(stage2_raw)

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
                        s2_confidence = self._compute_sequence_confidence(fb_scores, fb_gen_ids)
                        parsed = fb_parsed
                        print(f"[PavementClassifier] Stage 2 cascade: fallback recovered '{fb_parsed.get('distress_types')}'")
                except Exception as e:
                    # Fallback errors are non-fatal — keep the primary result.
                    print(f"[PavementClassifier] Stage 2 cascade fallback FAILED: {e}")

            s2_time = (time.time() - s2_start) * 1000

        return {
            "distress_types": parsed["distress_types"],
            "severity": parsed["severity"],
            "description": parsed["description"],
            "stage2_confidence": s2_confidence,
            "stage2_time_ms": round(s2_time, 1),
            "stage2_raw": stage2_raw,
            "stage2_used_fallback": used_fallback,
            "stage2_primary_raw": primary_raw if used_fallback else None,
        }

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

            # Flag for expert review if EITHER stage has low confidence
            result["needs_expert_review"] = (
                s1["stage1_confidence"] < CONFIDENCE_THRESHOLD
                or s2["stage2_confidence"] < CONFIDENCE_THRESHOLD
            )

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
