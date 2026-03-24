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
    STAGE1_SYSTEM_PROMPT,
    STAGE1_USER_PROMPT,
    STAGE2_SYSTEM_PROMPT,
    STAGE2_USER_PROMPT,
    parse_stage1_response,
    parse_stage2_response,
)


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
        self._lock = threading.Lock()
        self._model = None
        self._processor = None
        self._device = None
        self._loaded = False

        # Cached token IDs for Stage 1 confidence extraction
        self._normal_token_ids = []
        self._distress_token_ids = []

        self._load_model(quantization_bits)

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

        self._processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
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

    def predict_stage2(self, image: Image.Image) -> dict:
        """
        Run Stage 2 only: distress type classification.

        Returns dict with stage 2 fields only. Thread-safe.
        """
        image = image.convert("RGB")

        with self._lock:
            s2_start = time.time()
            stage2_raw, s2_scores, s2_gen_ids = self._run_inference_with_scores(
                image, STAGE2_SYSTEM_PROMPT, STAGE2_USER_PROMPT
            )
            s2_time = (time.time() - s2_start) * 1000

            s2_confidence = self._compute_sequence_confidence(s2_scores, s2_gen_ids)
            parsed = parse_stage2_response(stage2_raw)

        return {
            "distress_types": parsed["distress_types"],
            "severity": parsed["severity"],
            "description": parsed["description"],
            "stage2_confidence": s2_confidence,
            "stage2_time_ms": round(s2_time, 1),
            "stage2_raw": stage2_raw,
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
    """Get or create the global classifier singleton."""
    global _classifier
    if _classifier is None:
        model_path = model_path or os.environ.get(
            "MODEL_PATH", "Qwen/Qwen2.5-VL-7B-Instruct"
        )
        adapter_path = adapter_path or os.environ.get("ADAPTER_PATH", None)
        if adapter_path == "":
            adapter_path = None
        quant = int(os.environ.get("QUANTIZATION_BITS", str(quantization_bits)))
        _classifier = PavementClassifier(model_path, adapter_path, quant)
    return _classifier
