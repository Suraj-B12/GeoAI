"""
Family-agnostic VLM loader with VRAM guardrails and a load-time self-test.

Single source of truth for "how do we put a vision-language model on the GPU".
Used by app/model.py (production) and scripts/03_baseline_eval.py (evaluation,
which scripts/07_cross_dataset_eval.py imports) so a model swap changes ONE
file instead of three.

Why this exists
---------------
The pipeline was hardcoded to `Qwen2_5_VLForConditionalGeneration`. Swapping
to Qwen3-VL (or any other family) meant editing every load site and risking
silent drift between the production path and the evaluation path — which would
invalidate any A/B comparison. This module resolves the correct class from the
model config, so `MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct` is the only change
needed.

Guardrails implemented here
---------------------------
1. Class resolution from config.model_type (AutoModelForImageTextToText), with
   an explicit fallback to the Qwen2.5-VL class so existing behaviour is
   byte-identical if auto-resolution ever fails.
2. VRAM pre-flight: parameter count is fetched from the HF Hub metadata (no
   download), converted to an estimated footprint, and compared against free
   VRAM. Warns loudly, and can auto-downgrade the pixel budget.
3. OOM fallback ladder: bf16 -> 4-bit NF4 -> clear error. An OOM at load time
   is recoverable; an OOM 40 minutes into a batch run is not.
4. Processor kwargs compatibility: min_pixels/max_pixels are Qwen-specific.
   Unsupported kwargs are detected and dropped instead of crashing.
5. Load-time self-test: a synthetic image is pushed through the full
   apply_chat_template -> processor -> generate path. Catches template and
   processor mismatches at startup (2 seconds) rather than on the first real
   citizen photo (which would be marked 'failed' in the DB).
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

import torch
from PIL import Image

# Families we have actually exercised in this project. Anything else loads with
# a warning — it is not blocked, but it is not silently trusted either.
VERIFIED_FAMILIES = {
    "qwen2_5_vl": "Qwen2.5-VL (original production family)",
    "qwen3_vl": "Qwen3-VL (upgrade candidate)",
}

# Bytes per parameter by load mode, including runtime overhead observed in
# practice (bnb 4-bit stores quant constants + absmax alongside the weights).
_BYTES_PER_PARAM = {
    0: 2.00,   # bf16
    8: 1.10,   # int8
    4: 0.62,   # nf4 + double quant
}


def resolve_model_class(model_path: str) -> tuple[Any, str]:
    """
    Pick the right *ForConditionalGeneration class for this checkpoint.

    Returns (class, model_type). Never raises for an unknown family — falls
    back to the Qwen2.5-VL class, which is what the pipeline has always used.
    """
    from transformers import AutoConfig, AutoModelForImageTextToText
    from transformers import Qwen2_5_VLForConditionalGeneration

    model_type = "unknown"
    try:
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model_type = getattr(cfg, "model_type", "unknown")
    except Exception as e:
        print(f"[loader] WARNING: could not read config ({e}); assuming qwen2_5_vl")
        return Qwen2_5_VLForConditionalGeneration, "qwen2_5_vl"

    if model_type in VERIFIED_FAMILIES:
        print(f"[loader] Model family: {model_type} — {VERIFIED_FAMILIES[model_type]}")
    else:
        print(f"[loader] WARNING: untested model family '{model_type}'. "
              f"Verified families: {sorted(VERIFIED_FAMILIES)}. "
              f"Run scripts/smoke_test_model.py before trusting any output.")

    try:
        from transformers.models.auto.modeling_auto import (
            MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES as _MAP,
        )
        if model_type in _MAP:
            return AutoModelForImageTextToText, model_type
    except Exception:
        pass

    print(f"[loader] '{model_type}' not in the image-text-to-text mapping; "
          f"falling back to Qwen2_5_VLForConditionalGeneration")
    return Qwen2_5_VLForConditionalGeneration, model_type


def estimate_param_count(model_path: str, timeout_s: float = 10.0) -> Optional[int]:
    """
    Parameter count from HF Hub metadata — no weights downloaded.

    Returns None when offline, when the repo is local, or when the Hub does not
    expose safetensors metadata. Callers must treat None as "unknown", never
    as zero.
    """
    if os.path.isdir(model_path):
        return None
    try:
        from huggingface_hub import model_info
        info = model_info(model_path, timeout=timeout_s)
        st = getattr(info, "safetensors", None)
        if st and getattr(st, "total", None):
            return int(st.total)
    except Exception as e:
        print(f"[loader] param-count lookup skipped ({type(e).__name__}: {e})")
    return None


def vram_status() -> dict:
    """Free / total VRAM in GB. Zeros when CUDA is unavailable."""
    if not torch.cuda.is_available():
        return {"available": False, "free_gb": 0.0, "total_gb": 0.0, "name": "cpu"}
    free, total = torch.cuda.mem_get_info(0)
    return {
        "available": True,
        "free_gb": round(free / 1e9, 2),
        "total_gb": round(total / 1e9, 2),
        "name": torch.cuda.get_device_properties(0).name,
    }


def preflight_vram(
    model_path: str,
    quantization_bits: int,
    max_pixels: int,
) -> dict:
    """
    Compare the estimated footprint against free VRAM BEFORE loading.

    Returns a dict with the verdict and (when tight) a recommended max_pixels.
    Advisory only — it never blocks a load, because the estimate can be wrong
    and the OOM ladder in load_vlm() is the real safety net.
    """
    vram = vram_status()
    params = estimate_param_count(model_path)
    out = {
        "gpu": vram["name"],
        "free_gb": vram["free_gb"],
        "total_gb": vram["total_gb"],
        "params_b": round(params / 1e9, 2) if params else None,
        "verdict": "unknown",
        "recommended_max_pixels": max_pixels,
    }

    if not vram["available"] or params is None:
        print(f"[loader] VRAM pre-flight: skipped (gpu={vram['name']}, "
              f"params={'unknown' if params is None else params})")
        return out

    weights_gb = params * _BYTES_PER_PARAM.get(quantization_bits, 2.0) / 1e9
    # Activation headroom scales with the visual-token count. One token per
    # 28x28 px block; measured overhead is roughly 0.5 MB/token at bf16 for
    # a single-image, ~200-token generation.
    vis_tokens = max_pixels / 784.0
    activations_gb = max(1.5, vis_tokens * 0.5 / 1024.0)
    needed_gb = weights_gb + activations_gb
    out.update(
        weights_gb=round(weights_gb, 2),
        activations_gb=round(activations_gb, 2),
        needed_gb=round(needed_gb, 2),
        visual_tokens=int(vis_tokens),
    )

    margin = vram["free_gb"] - needed_gb
    if margin >= 2.0:
        out["verdict"] = "ok"
        print(f"[loader] VRAM pre-flight OK: need ~{needed_gb:.1f} GB "
              f"({weights_gb:.1f} weights + {activations_gb:.1f} activations), "
              f"free {vram['free_gb']:.1f} GB on {vram['name']}")
    elif margin >= 0:
        out["verdict"] = "tight"
        # Trim the pixel budget so activations fit with 2 GB to spare.
        budget_gb = max(1.0, vram["free_gb"] - weights_gb - 2.0)
        out["recommended_max_pixels"] = max(
            256 * 28, int(budget_gb * 1024.0 / 0.5 * 784.0)
        )
        print(f"[loader] VRAM pre-flight TIGHT: need ~{needed_gb:.1f} GB, "
              f"free {vram['free_gb']:.1f} GB. Recommended max_pixels "
              f"{out['recommended_max_pixels']} (was {max_pixels}).")
    else:
        out["verdict"] = "insufficient"
        print(f"[loader] VRAM pre-flight INSUFFICIENT: need ~{needed_gb:.1f} GB, "
              f"free only {vram['free_gb']:.1f} GB. Will fall back to 4-bit on OOM.")
    return out


def _select_attn_impl() -> str:
    """Flash Attention 2 when installed and the GPU is Ampere+, else SDPA."""
    try:
        import flash_attn  # noqa: F401
        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
            cc = torch.cuda.get_device_properties(0).major
            print(f"[loader] Using Flash Attention 2 (compute {cc}.x)")
            return "flash_attention_2"
    except ImportError:
        print("[loader] flash-attn not installed — using SDPA (correct, just slower)")
    except Exception as e:
        print(f"[loader] flash-attn probe failed ({e}) — using SDPA")
    return "sdpa"


def _build_quant_config(quantization_bits: int):
    from transformers import BitsAndBytesConfig
    if quantization_bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    if quantization_bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def load_processor(model_path: str, min_pixels: int, max_pixels: int):
    """
    Load the processor, degrading gracefully when pixel kwargs are unsupported.

    min_pixels/max_pixels are Qwen-family kwargs. Other families reject them,
    so try the full form first, then the `size` dict form, then bare.
    """
    from transformers import AutoProcessor

    attempts = [
        ("min_pixels/max_pixels",
         dict(min_pixels=min_pixels, max_pixels=max_pixels)),
        ("size dict",
         dict(size={"shortest_edge": min_pixels, "longest_edge": max_pixels})),
        ("defaults", {}),
    ]
    last_err = None
    for label, kwargs in attempts:
        try:
            proc = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True, **kwargs
            )
            print(f"[loader] Processor pixel limits applied via: {label}")
            return proc, label
        except (TypeError, ValueError) as e:
            last_err = e
            continue
    raise RuntimeError(f"Could not load processor for {model_path}: {last_err}")


def self_test(model, processor, verbose: bool = True) -> dict:
    """
    Push a synthetic image through the whole path before trusting the model.

    Catches chat-template mismatches, processor/model disagreement about the
    image token, and dtype/device faults — at startup, in ~2 seconds, instead
    of on the first real photo (which would land in the DB as 'failed').

    Returns {'ok': bool, 'error': str|None, 'elapsed_s': float}.
    """
    t0 = time.time()
    try:
        img = Image.new("RGB", (112, 112), color=(128, 128, 128))
        messages = [
            {"role": "system", "content": "You are a test harness."},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "Reply with the single word OK."},
            ]},
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[text], images=[img], padding=True,
                           return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=4, do_sample=False)
        decoded = processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0]
        elapsed = time.time() - t0
        if verbose:
            print(f"[loader] Self-test PASSED in {elapsed:.1f}s "
                  f"(model replied {decoded.strip()!r})")
        return {"ok": True, "error": None, "elapsed_s": round(elapsed, 2),
                "reply": decoded.strip()}
    except Exception as e:
        elapsed = time.time() - t0
        print(f"[loader] Self-test FAILED after {elapsed:.1f}s: "
              f"{type(e).__name__}: {e}")
        return {"ok": False, "error": f"{type(e).__name__}: {e}",
                "elapsed_s": round(elapsed, 2), "reply": None}


def load_vlm(
    model_path: str,
    quantization_bits: int = 4,
    adapter_path: Optional[str] = None,
    min_pixels: int = 256 * 28,
    max_pixels: int = 2200 * 2200,
    run_self_test: bool = True,
    allow_oom_fallback: bool = True,
    empty_cache_after_load: bool = True,
) -> tuple[Any, Any, dict]:
    """
    Load a VLM (+ optional LoRA adapter) with every guardrail applied.

    Returns (model, processor, info). `info` records what actually happened —
    resolved class, final quantization, final pixel budget, whether the OOM
    fallback fired, self-test result — so /health and the eval JSONs can report
    the true configuration rather than the requested one.
    """
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "max_split_size_mb:512,garbage_collection_threshold:0.9",
    )

    model_cls, model_type = resolve_model_class(model_path)
    pre = preflight_vram(model_path, quantization_bits, max_pixels)
    if pre["verdict"] == "tight" and pre["recommended_max_pixels"] < max_pixels:
        print(f"[loader] Auto-reducing max_pixels {max_pixels} -> "
              f"{pre['recommended_max_pixels']} to fit available VRAM")
        max_pixels = pre["recommended_max_pixels"]

    attn_impl = _select_attn_impl()
    info = {
        "model_path": model_path,
        "model_type": model_type,
        "model_class": model_cls.__name__,
        "attn_implementation": attn_impl,
        "requested_quantization_bits": quantization_bits,
        "quantization_bits": quantization_bits,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "oom_fallback_used": False,
        "preflight": pre,
    }

    # Ladder: requested precision first; on OOM, drop to 4-bit (which always
    # fits on a 24 GB card for models in the 7-14B range).
    ladder = [quantization_bits]
    if allow_oom_fallback and quantization_bits != 4:
        ladder.append(4)

    model = None
    last_err: Optional[BaseException] = None
    for bits in ladder:
        try:
            label = "bf16 (no quantization)" if bits == 0 else f"{bits}-bit"
            print(f"[loader] Loading {model_path} ({label})...")
            load_kwargs: dict[str, Any] = dict(
                device_map="auto",
                dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation=attn_impl,
            )
            qcfg = _build_quant_config(bits)
            if qcfg is not None:
                load_kwargs["quantization_config"] = qcfg
            model = model_cls.from_pretrained(model_path, **load_kwargs)
            info["quantization_bits"] = bits
            info["oom_fallback_used"] = bits != quantization_bits
            break
        except torch.cuda.OutOfMemoryError as e:
            last_err = e
            print(f"[loader] OOM loading at {bits}-bit. Clearing cache and "
                  f"{'retrying at 4-bit' if bits != ladder[-1] else 'giving up'}.")
            model = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            # Non-OOM failures are not retryable by changing precision.
            raise RuntimeError(
                f"Failed to load {model_path} as {model_cls.__name__}: "
                f"{type(e).__name__}: {e}"
            ) from e

    if model is None:
        raise RuntimeError(
            f"Could not fit {model_path} in VRAM at any precision "
            f"({vram_status()}). Last error: {last_err}"
        )

    if adapter_path:
        from peft import PeftModel
        print(f"[loader] Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        # NEVER merge_and_unload() on a quantized base — peft #2586 silently
        # drops the LoRA delta with no error.

    processor, pixel_mode = load_processor(model_path, min_pixels, max_pixels)
    info["processor_pixel_mode"] = pixel_mode

    model.eval()
    info["device"] = str(next(model.parameters()).device)

    # CPU/disk offload check. device_map="auto" silently offloads layers when
    # accelerate thinks VRAM is short. The model still produces correct output,
    # just 10-30x slower — which is invisible in accuracy metrics and fatal to
    # any timing comparison between two models. Surface it loudly.
    device_map = getattr(model, "hf_device_map", None)
    info["hf_device_map_summary"] = None
    info["fully_on_gpu"] = True
    if isinstance(device_map, dict) and device_map:
        placements: dict[str, int] = {}
        for dev in device_map.values():
            placements[str(dev)] = placements.get(str(dev), 0) + 1
        info["hf_device_map_summary"] = placements
        offloaded = {d: n for d, n in placements.items()
                     if d in ("cpu", "disk") or str(d).startswith("cpu")}
        if offloaded:
            info["fully_on_gpu"] = False
            print(f"[loader] *** WARNING: {sum(offloaded.values())} module(s) "
                  f"OFFLOADED to {sorted(offloaded)} — inference will be MUCH "
                  f"slower and timing comparisons will be invalid. "
                  f"Placement: {placements}")
        else:
            print(f"[loader] Device placement: {placements} (fully on GPU)")

    if run_self_test:
        info["self_test"] = self_test(model, processor)
        if not info["self_test"]["ok"]:
            raise RuntimeError(
                f"Model loaded but failed its self-test — refusing to serve it. "
                f"{info['self_test']['error']}"
            )

    # ---- Release the allocator's load-time scratch blocks -----------------
    # MEASURED, NOT THEORETICAL (RTX A5000 24GB, Qwen2.5-VL-7B bf16):
    #   before empty_cache: 16.65 GB tensors but 32.35 GB RESERVED, device
    #                       free 0.00 GB -> Stage 1 inference 23.5 s
    #   after  empty_cache: 16.65 GB tensors, 16.84 GB reserved, 7.54 GB free
    #                       -> Stage 1 inference 4.7 s   (5x faster)
    #
    # Sharded loading leaves ~16 GB of freed-but-cached blocks in the caching
    # allocator. On Windows WDDM the driver does not OOM when reservations
    # exceed physical VRAM — it silently spills into system RAM over PCIe, so
    # the model keeps working and just runs 5x slower for the life of the
    # process. Nothing in the logs says so; the only visible symptom is
    # reserved > total. Releasing the blocks here is free and permanent.
    if torch.cuda.is_available() and empty_cache_after_load:
        import gc
        reserved_before = torch.cuda.memory_reserved(0) / 1e9
        gc.collect()
        torch.cuda.empty_cache()
        reserved_after = torch.cuda.memory_reserved(0) / 1e9
        info["reserved_gb_before_cleanup"] = round(reserved_before, 2)
        info["reserved_gb_after_cleanup"] = round(reserved_after, 2)
        if reserved_before - reserved_after > 0.5:
            print(f"[loader] Released {reserved_before - reserved_after:.1f} GB of "
                  f"cached allocator blocks ({reserved_before:.1f} -> "
                  f"{reserved_after:.1f} GB reserved)")
        total_gb = vram_status()["total_gb"]
        if reserved_after > total_gb:
            print(f"[loader] *** WARNING: {reserved_after:.1f} GB still reserved on a "
                  f"{total_gb:.1f} GB card — CUDA is spilling into system RAM and "
                  f"inference will be several times slower than it should be.")

    if torch.cuda.is_available():
        v = vram_status()
        # Three different numbers, because they answer different questions:
        #   allocated = tensors actually live right now (the real model cost)
        #   reserved  = what the caching allocator holds (allocated + free blocks)
        #   device    = total - free, which also counts other processes
        # Reporting only the last one made a healthy load look like a full card.
        info["vram_allocated_gb"] = round(torch.cuda.memory_allocated(0) / 1e9, 2)
        info["vram_reserved_gb"] = round(torch.cuda.memory_reserved(0) / 1e9, 2)
        info["vram_used_gb"] = round(v["total_gb"] - v["free_gb"], 2)
        info["vram_total_gb"] = v["total_gb"]
        print(f"[loader] Loaded on {info['device']} — "
              f"{info['vram_allocated_gb']:.1f} GB tensors / "
              f"{info['vram_reserved_gb']:.1f} GB reserved / "
              f"{info['vram_used_gb']:.1f} GB device-wide of {v['total_gb']:.1f} GB")

    return model, processor, info
