"""
Step 3: Evaluate base Qwen2.5-VL-7B-Instruct BEFORE fine-tuning.

Runs the model on test images for both stages:
  - Stage 1 (GAPs): Binary detection accuracy
  - Stage 2 (RDD): Distress type classification accuracy

Saves metrics to eval_results/baseline_results.json and
confusion matrix PNGs to eval_results/confusion_matrices/.

Usage:
    python scripts/03_baseline_eval.py --stage 1 --max-samples 500
    python scripts/03_baseline_eval.py --stage 2 --max-samples 500
    python scripts/03_baseline_eval.py --stage both
    python scripts/03_baseline_eval.py --stage 2 --quantization-bits 0
"""

import argparse
import gc
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# --- Graceful interrupt handling ---
_interrupted = False

def _signal_handler(sig, frame):
    global _interrupted
    if _interrupted:
        print("\nForce quit.")
        sys.exit(1)
    _interrupted = True
    print("\n\n*** Ctrl+C detected — finishing current image, then saving partial results... ***")
    print("*** Press Ctrl+C again to force quit (loses all data). ***\n")

signal.signal(signal.SIGINT, _signal_handler)

# Processor pixel limits — controls visual token count for Qwen2.5-VL.
# RDD images are 512x512 (262k pixels), GAPs are 160x160 (25k pixels).
# These bounds prevent explosion on unexpectedly large images while
# keeping small images usable.
MAX_PIXELS = 2200 * 2200  # ~4.84M upper bound
MIN_PIXELS = 256 * 28     # 7168 — low enough for 160x160 GAPs images

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils import (
    DATA_DIR,
    EVAL_DIR,
    GAPS_LABEL_MAP,
    RDD_LABEL_MAP,
    STAGE1_SYSTEM_PROMPT,
    STAGE1_USER_PROMPT,
    STAGE2_SYSTEM_PROMPT,
    STAGE2_USER_PROMPT,
    map_stage2_to_rdd_ids,
    parse_stage1_response,
    parse_stage2_response,
)


def print_system_info():
    """Print GPU/CPU info for logging."""
    print(f"\n{'='*60}")
    print("System Info")
    print(f"{'='*60}")
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        vram_total = gpu.total_memory / 1024**3
        vram_used = torch.cuda.memory_allocated(0) / 1024**3
        vram_reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"GPU: {gpu.name} ({vram_total:.1f} GB VRAM)")
        print(f"VRAM: {vram_used:.1f} GB allocated, {vram_reserved:.1f} GB reserved")
    else:
        print("GPU: None (CPU only)")
    import psutil
    ram = psutil.virtual_memory()
    print(f"RAM: {ram.used / 1024**3:.1f} / {ram.total / 1024**3:.1f} GB")
    print(f"CPU cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical")
    print(f"{'='*60}\n")


def print_vram_usage(label: str = ""):
    """Print current VRAM usage."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"  VRAM [{label}]: {alloc:.1f} GB allocated, {reserved:.1f} GB reserved")


def load_model(model_path: str, adapter_path: str = None, quant_bits: int = None):
    """
    Load Qwen2.5-VL model with optional LoRA adapter.

    quant_bits: 0 = fp16/bf16 (no quantization), 4 = 4-bit, 8 = 8-bit.
    Falls back to QUANTIZATION_BITS env var, then defaults to 4.
    """
    from transformers import BitsAndBytesConfig

    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "max_split_size_mb:512,garbage_collection_threshold:0.9",
    )

    if quant_bits is None:
        quant_bits = int(os.environ.get("QUANTIZATION_BITS", "4"))

    bnb_config = None
    if quant_bits == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif quant_bits == 8:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # Select best attention implementation
    attn_impl = "sdpa"
    try:
        import flash_attn  # noqa: F401
        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
            attn_impl = "flash_attention_2"
            print("Using Flash Attention 2")
    except ImportError:
        pass

    quant_label = f"{quant_bits}-bit" if bnb_config else "fp16/bf16 (no quantization)"
    print(f"Loading model: {model_path}")
    print(f"  Attention: {attn_impl} | Quantization: {quant_label}")

    load_kwargs = dict(
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    if bnb_config:
        load_kwargs["quantization_config"] = bnb_config

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        **load_kwargs,
    )

    if adapter_path:
        from peft import PeftModel
        print(f"Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )

    model.eval()
    print_vram_usage("after model load")
    print(f"Processor pixel limits: min={MIN_PIXELS}, max={MAX_PIXELS}")
    print(f"Model device: {next(model.parameters()).device}")

    return model, processor


def run_inference(model, processor, image: Image.Image,
                  system_prompt: str, user_prompt: str,
                  max_new_tokens: int = 150) -> str:
    """Run a single inference and return the model's text response."""
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

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Get the device from the model's first parameter (safe for device_map="auto")
    device = next(model.parameters()).device

    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    # Decode only the new tokens
    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Free VRAM from intermediates
    del inputs, output_ids, generated_ids
    torch.cuda.empty_cache()

    return response.strip()


def _compute_and_save_stage1(y_true, y_pred, unparseable_count, total_time,
                             test_data_len, tag):
    """Compute Stage 1 metrics and save confusion matrix. Crash-safe."""
    stage1_labels = [0, 1]
    stage1_names = [GAPS_LABEL_MAP[0], GAPS_LABEL_MAP[1]]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, labels=stage1_labels,
                           average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, labels=stage1_labels,
                       average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=stage1_labels,
                  average="macro", zero_division=0)
    report = classification_report(
        y_true, y_pred,
        labels=stage1_labels,
        target_names=stage1_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=stage1_labels)

    results = {
        "stage": "stage1_binary",
        "total_samples": len(y_true),
        "total_available": test_data_len,
        "partial": _interrupted,
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "unparseable_responses": unparseable_count,
        "avg_inference_time_sec": total_time / max(len(y_true), 1),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }

    print(f"\n--- Stage 1 Results ({len(y_true)}/{test_data_len} samples) ---")
    print(f"Accuracy:    {acc:.4f}")
    print(f"Precision:   {prec:.4f}")
    print(f"Recall:      {rec:.4f}")
    print(f"F1 (macro):  {f1:.4f}")
    print(f"Unparseable: {unparseable_count}/{len(y_true)}")
    print(f"Avg time:    {results['avg_inference_time_sec']:.2f}s/image")

    # Confusion matrix plot
    cm_dir = EVAL_DIR / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=stage1_names, yticklabels=stage1_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Stage 1 - Binary Detection ({tag.title()}, n={len(y_true)})")
    fig.tight_layout()
    cm_filename = f"{tag}_stage1_cm.png"
    fig.savefig(cm_dir / cm_filename, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to: {cm_dir / cm_filename}")

    return results


def _compute_and_save_stage2(y_true_primary, y_pred_primary, exact_match_count,
                             total_time, test_data_len, tag):
    """Compute Stage 2 metrics and save confusion matrix. Crash-safe."""
    # Use the fixed RDD label set so report doesn't crash on missing classes
    rdd_label_ids = sorted(RDD_LABEL_MAP.keys())  # [0, 1, 2, 3] for D00/D10/D20/D40
    all_seen = sorted(set(y_true_primary + y_pred_primary))

    # Include -1 (Normal/unparseable) if it appeared, plus all RDD labels
    all_labels = sorted(set(all_seen + rdd_label_ids))
    label_names = []
    for lid in all_labels:
        if lid == -1:
            label_names.append("Unparseable/Normal")
        else:
            label_names.append(RDD_LABEL_MAP.get(lid, f"Class_{lid}"))

    acc = accuracy_score(y_true_primary, y_pred_primary)
    f1 = f1_score(y_true_primary, y_pred_primary, labels=all_labels,
                  average="macro", zero_division=0)
    report = classification_report(
        y_true_primary, y_pred_primary,
        labels=all_labels,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true_primary, y_pred_primary, labels=all_labels)

    results = {
        "stage": "stage2_type_classification",
        "total_samples": len(y_true_primary),
        "total_available": test_data_len,
        "partial": _interrupted,
        "primary_accuracy": float(acc),
        "f1_macro": float(f1),
        "exact_match_rate": exact_match_count / max(len(y_true_primary), 1),
        "avg_inference_time_sec": total_time / max(len(y_true_primary), 1),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "label_order": [str(l) for l in all_labels],
    }

    print(f"\n--- Stage 2 Results ({len(y_true_primary)}/{test_data_len} samples) ---")
    print(f"Primary Accuracy: {acc:.4f}")
    print(f"F1 (macro):       {f1:.4f}")
    print(f"Exact Match Rate: {results['exact_match_rate']:.4f}")
    print(f"Avg time:         {results['avg_inference_time_sec']:.2f}s/image")

    # Confusion matrix plot
    cm_dir = EVAL_DIR / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                xticklabels=label_names, yticklabels=label_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Stage 2 - Distress Type ({tag.title()}, n={len(y_true_primary)})")
    fig.tight_layout()
    cm_filename = f"{tag}_stage2_cm.png"
    fig.savefig(cm_dir / cm_filename, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to: {cm_dir / cm_filename}")

    return results


def evaluate_stage1(model, processor, max_samples: int = 0, tag: str = "baseline") -> dict:
    """Evaluate Stage 1 (GAPs binary detection) on test set."""
    test_json_path = DATA_DIR / "gaps_test.json"
    if not test_json_path.exists():
        print(f"ERROR: {test_json_path} not found. Run 02_build_training_data.py first.")
        return {}

    with open(test_json_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    if max_samples > 0:
        test_data = test_data[:max_samples]

    print(f"\n{'='*60}")
    print(f"Stage 1 Evaluation: GAPs Binary Detection")
    print(f"Test samples: {len(test_data)}")
    print(f"{'='*60}")

    y_true, y_pred = [], []
    unparseable_count = 0
    skipped_count = 0
    error_count = 0
    total_time = 0

    for i, entry in enumerate(tqdm(test_data, desc="Stage 1 Eval")):
        if _interrupted:
            print(f"\n  Interrupted at sample {i}/{len(test_data)}.")
            break

        img_path = Path(entry["images"][0])
        if not img_path.exists():
            skipped_count += 1
            if skipped_count <= 3:
                print(f"\n  WARNING: Image not found: {img_path}")
            continue

        gt_label = entry["conversations"][1]["value"]  # "Normal" or "Distress"
        gt_id = 0 if gt_label == "Normal" else 1

        try:
            image = Image.open(img_path).convert("RGB")

            # Log dimensions of first image
            if i == 0:
                w, h = image.size
                print(f"  First image: {img_path.name} ({w}x{h})")

            start = time.time()
            response = run_inference(
                model, processor, image,
                STAGE1_SYSTEM_PROMPT, STAGE1_USER_PROMPT,
                max_new_tokens=20,  # Stage 1 only needs "Normal" or "Distress"
            )
            elapsed = time.time() - start
            total_time += elapsed
            image.close()

            # Log first 3 responses for debugging
            if i < 3:
                print(f"  Sample {i}: GT={gt_label} | Response={response!r} | {elapsed:.1f}s")

            pred_id = parse_stage1_response(response)
            if pred_id == -1:
                unparseable_count += 1
                pred_id = 0  # Default to Normal for unparseable

            y_true.append(gt_id)
            y_pred.append(pred_id)

        except Exception as e:
            error_count += 1
            if error_count <= 5:
                print(f"\n  ERROR on sample {i} ({img_path.name}): {e}")
            continue

        # Log every 100 samples
        if (i + 1) % 100 == 0:
            running_acc = accuracy_score(y_true, y_pred)
            avg_time = total_time / len(y_true)
            print(f"  [{i+1}/{len(test_data)}] Acc: {running_acc:.4f} | "
                  f"Avg: {avg_time:.2f}s/img | Errors: {error_count} | Skipped: {skipped_count}")

    # Summary before metrics
    print(f"\n  Processed: {len(y_true)} | Skipped: {skipped_count} | "
          f"Errors: {error_count} | Unparseable: {unparseable_count}")

    if not y_true:
        print("  No samples evaluated. Skipping metrics.")
        return {}

    return _compute_and_save_stage1(y_true, y_pred, unparseable_count,
                                    total_time, len(test_data), tag)


def _checkpoint_path(tag: str) -> Path:
    """Return the checkpoint file path for Stage 2."""
    return EVAL_DIR / f".stage2_checkpoint_{tag}.json"


def _save_checkpoint(path: Path, data: dict):
    """Save checkpoint with .bak fallback for crash safety.

    Atomic rename via Path.replace() is unreliable on Windows when any other
    process (Search Indexer, Defender, OneDrive, an open File Explorer window)
    has even a read handle on the destination — confirmed empirically.
    Instead: rename current json -> .bak (cheap rename, doesn't need write
    access on the target), then write fresh json. If we crash mid-write the
    .bak from the previous successful save survives.

    Save failures are non-fatal: a transient file lock should not kill a 6h+
    eval run. We log the warning and continue — the eval state stays in RAM
    and the next interval will retry.
    """
    bak = path.with_suffix(".json.bak")
    try:
        if path.exists():
            try:
                if bak.exists():
                    bak.unlink()
                path.rename(bak)
            except OSError:
                # If we can't rotate the bak, just overwrite directly
                pass
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except OSError as e:
        print(f"  [warn] checkpoint save failed: {e} — continuing in-memory only")


def _load_checkpoint(path: Path) -> dict | None:
    """Load checkpoint, falling back to .json.bak if the primary is missing/corrupt."""
    candidates = [path]
    bak = path.with_suffix(".json.bak")
    if bak.exists():
        candidates.append(bak)
    for cand in candidates:
        if not cand.exists():
            continue
        try:
            with open(cand, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "resume_index" in data and "y_true_primary" in data:
                if cand != path:
                    print(f"  [info] primary checkpoint unreadable, falling back to {cand.name}")
                return data
        except (json.JSONDecodeError, OSError, KeyError):
            continue
    return None


CHECKPOINT_INTERVAL = 50  # Save every 50 images


def evaluate_stage2(model, processor, max_samples: int = 0, tag: str = "baseline") -> dict:
    """Evaluate Stage 2 (RDD distress type classification) on test set.

    Supports checkpoint-based resumption: saves progress every 50 images
    so a crash or reboot doesn't lose work. On restart, automatically
    resumes from the last checkpoint.
    """
    test_json_path = DATA_DIR / "rdd_test.json"
    if not test_json_path.exists():
        print(f"ERROR: {test_json_path} not found. Run 02_build_training_data.py first.")
        return {}

    with open(test_json_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    if max_samples > 0:
        test_data = test_data[:max_samples]

    # --- Check for existing checkpoint ---
    ckpt_path = _checkpoint_path(tag)
    ckpt = _load_checkpoint(ckpt_path)

    if ckpt and ckpt.get("total_samples") == len(test_data):
        resume_index = ckpt["resume_index"]
        y_true_primary = ckpt["y_true_primary"]
        y_pred_primary = ckpt["y_pred_primary"]
        exact_match_count = ckpt["exact_match_count"]
        skipped_count = ckpt["skipped_count"]
        error_count = ckpt["error_count"]
        total_time = ckpt["total_time"]
        print(f"\n  *** RESUMING from checkpoint at sample {resume_index}/{len(test_data)} ***")
        print(f"  Already processed: {len(y_true_primary)} | "
              f"Skipped: {skipped_count} | Errors: {error_count}")
    else:
        if ckpt:
            print(f"\n  Checkpoint found but sample count changed "
                  f"({ckpt.get('total_samples')} -> {len(test_data)}). Starting fresh.")
        resume_index = 0
        y_true_primary, y_pred_primary = [], []
        exact_match_count = 0
        skipped_count = 0
        error_count = 0
        total_time = 0

    print(f"\n{'='*60}")
    print(f"Stage 2 Evaluation: RDD Distress Type Classification")
    print(f"Test samples: {len(test_data)} (starting from index {resume_index})")
    print(f"Checkpoint interval: every {CHECKPOINT_INTERVAL} images")
    print(f"{'='*60}")

    for i in tqdm(range(resume_index, len(test_data)), desc="Stage 2 Eval",
                  initial=resume_index, total=len(test_data)):
        if _interrupted:
            print(f"\n  Interrupted at sample {i}/{len(test_data)}.")
            break

        entry = test_data[i]
        img_path = Path(entry["images"][0])
        if not img_path.exists():
            skipped_count += 1
            if skipped_count <= 3:
                print(f"\n  WARNING: Image not found: {img_path}")
            continue

        gt_response = entry["conversations"][1]["value"]
        gt_parsed = parse_stage2_response(gt_response)
        gt_rdd_ids = map_stage2_to_rdd_ids(gt_parsed["distress_types"])
        gt_primary = gt_rdd_ids[0] if gt_rdd_ids else -1

        try:
            image = Image.open(img_path).convert("RGB")

            if i == resume_index:
                w, h = image.size
                print(f"  First image: {img_path.name} ({w}x{h})")

            start = time.time()
            response = run_inference(
                model, processor, image,
                STAGE2_SYSTEM_PROMPT, STAGE2_USER_PROMPT,
                max_new_tokens=150,
            )
            elapsed = time.time() - start
            total_time += elapsed
            image.close()

            if i < resume_index + 3:
                print(f"  Sample {i}: GT_IDs={gt_rdd_ids} | Response={response[:120]!r}... | {elapsed:.1f}s")

            pred_parsed = parse_stage2_response(response)
            pred_rdd_ids = map_stage2_to_rdd_ids(pred_parsed["distress_types"])
            pred_primary = pred_rdd_ids[0] if pred_rdd_ids else -1

            y_true_primary.append(gt_primary)
            y_pred_primary.append(pred_primary)

            if set(pred_rdd_ids) == set(gt_rdd_ids):
                exact_match_count += 1

        except Exception as e:
            error_count += 1
            if error_count <= 5:
                print(f"\n  ERROR on sample {i} ({img_path.name}): {e}")
            continue

        # --- Checkpoint every N images ---
        if len(y_true_primary) > 0 and len(y_true_primary) % CHECKPOINT_INTERVAL == 0:
            _save_checkpoint(ckpt_path, {
                "resume_index": i + 1,
                "y_true_primary": y_true_primary,
                "y_pred_primary": y_pred_primary,
                "exact_match_count": exact_match_count,
                "skipped_count": skipped_count,
                "error_count": error_count,
                "total_time": total_time,
                "total_samples": len(test_data),
            })
            # Deep memory clean every checkpoint interval — prevents the
            # allocator-fragmentation slowdown empirically observed during long
            # eval runs on the A5000 (step time ballooning 5s -> 60s+ around
            # sample 3850 of Stage 2). gc.collect() releases Python refs to
            # tensors; empty_cache() then returns CUDA memory to the allocator.
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            running_acc = accuracy_score(y_true_primary, y_pred_primary)
            avg_time = total_time / len(y_true_primary)
            print(f"  [{i+1}/{len(test_data)}] Acc: {running_acc:.4f} | "
                  f"Avg: {avg_time:.2f}s/img | Checkpoint saved + cache cleared",
                  flush=True)

        elif (i + 1) % 100 == 0:
            running_acc = accuracy_score(y_true_primary, y_pred_primary)
            avg_time = total_time / len(y_true_primary)
            print(f"  [{i+1}/{len(test_data)}] Acc: {running_acc:.4f} | "
                  f"Avg: {avg_time:.2f}s/img | Errors: {error_count} | Skipped: {skipped_count}",
                  flush=True)

    # --- Final checkpoint (covers Ctrl+C case too) ---
    final_index = i + 1 if len(test_data) > 0 else resume_index
    if y_true_primary:
        _save_checkpoint(ckpt_path, {
            "resume_index": final_index,
            "y_true_primary": y_true_primary,
            "y_pred_primary": y_pred_primary,
            "exact_match_count": exact_match_count,
            "skipped_count": skipped_count,
            "error_count": error_count,
            "total_time": total_time,
            "total_samples": len(test_data),
        })

    print(f"\n  Processed: {len(y_true_primary)} | Skipped: {skipped_count} | "
          f"Errors: {error_count}")

    if not y_true_primary:
        print("  No samples evaluated. Skipping metrics.")
        return {}

    results = _compute_and_save_stage2(y_true_primary, y_pred_primary,
                                       exact_match_count, total_time,
                                       len(test_data), tag)

    # Clean up checkpoint on successful completion
    if not _interrupted and ckpt_path.exists():
        ckpt_path.unlink()
        print(f"  Checkpoint cleared (evaluation complete).")

    return results


def main():
    parser = argparse.ArgumentParser(description="Baseline evaluation of Qwen2.5-VL")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="Model name or path")
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="Path to LoRA adapter (for post-finetune eval)")
    parser.add_argument("--stage", type=str, default="both", choices=["1", "2", "both"],
                        help="Which stage to evaluate")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max samples to evaluate (0 = all)")
    parser.add_argument("--quantization-bits", type=int, default=None,
                        choices=[0, 4, 8],
                        help="0 = fp16 (no quantization), 4 = 4-bit, 8 = 8-bit. "
                             "Overrides QUANTIZATION_BITS env var. Default: 4")
    args = parser.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    print_system_info()

    model, processor = load_model(args.model, args.adapter_path, args.quantization_bits)

    suffix = "finetuned" if args.adapter_path else "baseline"
    results_path = EVAL_DIR / f"{suffix}_results.json"

    # Load existing results so running --stage 1 then --stage 2 separately
    # accumulates into the same file instead of overwriting
    all_results = {}
    if results_path.exists():
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                all_results = json.load(f)
            print(f"Loaded existing results from: {results_path}")
        except (json.JSONDecodeError, OSError):
            all_results = {}

    try:
        if args.stage in ("1", "both"):
            results_s1 = evaluate_stage1(model, processor, args.max_samples)
            if results_s1:
                all_results["stage1"] = results_s1

        if args.stage in ("2", "both") and not _interrupted:
            results_s2 = evaluate_stage2(model, processor, args.max_samples)
            if results_s2:
                all_results["stage2"] = results_s2
        elif _interrupted and args.stage == "both":
            print("\nSkipping Stage 2 — interrupted during Stage 1.")

    except Exception as e:
        print(f"\n!!! Unexpected error: {e}")
        traceback.print_exc()

    finally:
        # Always save whatever we have
        if all_results:
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2)
            partial_tag = " (PARTIAL — interrupted)" if _interrupted else ""
            print(f"\nResults saved to: {results_path}{partial_tag}")
        else:
            print("\nNo results to save.")


if __name__ == "__main__":
    main()
