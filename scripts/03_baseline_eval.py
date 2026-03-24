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
"""

import argparse
import json
import sys
import time
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


def load_model(model_path: str, adapter_path: str = None):
    """
    Load Qwen2.5-VL model with optional LoRA adapter.

    Uses 4-bit quantization for memory efficiency.
    Dynamically selects Flash Attention 2 if available.
    """
    import os
    from transformers import BitsAndBytesConfig

    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "max_split_size_mb:512,garbage_collection_threshold:0.9",
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Select best attention implementation
    attn_impl = "sdpa"
    try:
        import flash_attn  # noqa: F401
        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
            attn_impl = "flash_attention_2"
            print(f"Using Flash Attention 2")
    except ImportError:
        pass

    print(f"Loading model: {model_path} (attn: {attn_impl})")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )

    if adapter_path:
        from peft import PeftModel
        print(f"Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        # Do NOT merge_and_unload() on a quantized model (peft bug #2586).
        # Inference works correctly as PeftModel.

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    model.eval()
    return model, processor


def run_inference(model, processor, image: Image.Image, system_prompt: str, user_prompt: str) -> str:
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
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    # Decode only the new tokens
    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()


def evaluate_stage1(model, processor, max_samples: int = 0, tag: str = "baseline") -> dict:
    """Evaluate Stage 1 (GAPs binary detection) on test set."""
    test_json_path = DATA_DIR / "gaps_test.json"
    if not test_json_path.exists():
        print("ERROR: gaps_test.json not found. Run 02_build_training_data.py first.")
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
    total_time = 0

    for i, entry in enumerate(tqdm(test_data, desc="Stage 1 Eval")):
        # Load image (absolute path stored in JSON)
        img_path = Path(entry["images"][0])
        if not img_path.exists():
            continue

        image = Image.open(img_path).convert("RGB")

        # Get ground truth from the conversation
        gt_label = entry["conversations"][1]["value"]  # "Normal" or "Distress"
        gt_id = 0 if gt_label == "Normal" else 1

        # Run inference
        start = time.time()
        response = run_inference(model, processor, image, STAGE1_SYSTEM_PROMPT, STAGE1_USER_PROMPT)
        elapsed = time.time() - start
        total_time += elapsed

        # Parse response
        pred_id = parse_stage1_response(response)
        if pred_id == -1:
            unparseable_count += 1
            pred_id = 0  # Default to Normal for unparseable

        y_true.append(gt_id)
        y_pred.append(pred_id)

        # Log every 100 samples
        if (i + 1) % 100 == 0:
            running_acc = accuracy_score(y_true, y_pred)
            avg_time = total_time / (i + 1)
            print(f"  [{i+1}/{len(test_data)}] Accuracy: {running_acc:.4f} | Avg time: {avg_time:.2f}s/img")

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(
        y_true, y_pred,
        target_names=[GAPS_LABEL_MAP[0], GAPS_LABEL_MAP[1]],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)

    results = {
        "stage": "stage1_binary",
        "total_samples": len(y_true),
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "unparseable_responses": unparseable_count,
        "avg_inference_time_sec": total_time / max(len(y_true), 1),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }

    # Print results
    print(f"\n--- Stage 1 Results ---")
    print(f"Accuracy:    {acc:.4f}")
    print(f"Precision:   {prec:.4f}")
    print(f"Recall:      {rec:.4f}")
    print(f"F1 (macro):  {f1:.4f}")
    print(f"Unparseable: {unparseable_count}/{len(y_true)}")
    print(f"Avg time:    {results['avg_inference_time_sec']:.2f}s/image")

    # Save confusion matrix plot
    cm_dir = EVAL_DIR / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Distress"],
                yticklabels=["Normal", "Distress"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Stage 1 - Binary Detection Confusion Matrix ({tag.title()})")
    fig.tight_layout()
    cm_filename = f"{tag}_stage1_cm.png"
    fig.savefig(cm_dir / cm_filename, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to: {cm_dir / cm_filename}")

    return results


def evaluate_stage2(model, processor, max_samples: int = 0, tag: str = "baseline") -> dict:
    """Evaluate Stage 2 (RDD distress type classification) on test set."""
    test_json_path = DATA_DIR / "rdd_test.json"
    if not test_json_path.exists():
        print("ERROR: rdd_test.json not found. Run 02_build_training_data.py first.")
        return {}

    with open(test_json_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    if max_samples > 0:
        test_data = test_data[:max_samples]

    print(f"\n{'='*60}")
    print(f"Stage 2 Evaluation: RDD Distress Type Classification")
    print(f"Test samples: {len(test_data)}")
    print(f"{'='*60}")

    # For Stage 2, we evaluate per-image multi-label accuracy
    # using the dominant (first) class as the primary label
    y_true_primary, y_pred_primary = [], []
    exact_match_count = 0
    total_time = 0

    for i, entry in enumerate(tqdm(test_data, desc="Stage 2 Eval")):
        # Load image (absolute path stored in JSON)
        img_path = Path(entry["images"][0])
        if not img_path.exists():
            continue

        image = Image.open(img_path).convert("RGB")

        # Parse ground truth from the training response
        gt_response = entry["conversations"][1]["value"]
        gt_parsed = parse_stage2_response(gt_response)
        gt_rdd_ids = map_stage2_to_rdd_ids(gt_parsed["distress_types"])
        gt_primary = gt_rdd_ids[0] if gt_rdd_ids else -1  # -1 = Normal

        # Run inference
        start = time.time()
        response = run_inference(model, processor, image, STAGE2_SYSTEM_PROMPT, STAGE2_USER_PROMPT)
        elapsed = time.time() - start
        total_time += elapsed

        # Parse prediction
        pred_parsed = parse_stage2_response(response)
        pred_rdd_ids = map_stage2_to_rdd_ids(pred_parsed["distress_types"])
        pred_primary = pred_rdd_ids[0] if pred_rdd_ids else -1

        y_true_primary.append(gt_primary)
        y_pred_primary.append(pred_primary)

        # Exact match: all predicted types match all ground truth types
        if set(pred_rdd_ids) == set(gt_rdd_ids):
            exact_match_count += 1

        if (i + 1) % 100 == 0:
            running_acc = accuracy_score(y_true_primary, y_pred_primary)
            avg_time = total_time / (i + 1)
            print(f"  [{i+1}/{len(test_data)}] Primary Acc: {running_acc:.4f} | Avg time: {avg_time:.2f}s/img")

    # Build label list for classification report
    all_labels = sorted(set(y_true_primary + y_pred_primary))
    label_names = []
    for lid in all_labels:
        if lid == -1:
            label_names.append("Normal")
        else:
            label_names.append(RDD_LABEL_MAP.get(lid, f"Class_{lid}"))

    acc = accuracy_score(y_true_primary, y_pred_primary)
    f1 = f1_score(y_true_primary, y_pred_primary, average="macro", zero_division=0)
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
        "primary_accuracy": float(acc),
        "f1_macro": float(f1),
        "exact_match_rate": exact_match_count / max(len(y_true_primary), 1),
        "avg_inference_time_sec": total_time / max(len(y_true_primary), 1),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "label_order": [str(l) for l in all_labels],
    }

    print(f"\n--- Stage 2 Results ---")
    print(f"Primary Accuracy: {acc:.4f}")
    print(f"F1 (macro):       {f1:.4f}")
    print(f"Exact Match Rate: {results['exact_match_rate']:.4f}")
    print(f"Avg time:         {results['avg_inference_time_sec']:.2f}s/image")

    # Confusion matrix
    cm_dir = EVAL_DIR / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                xticklabels=label_names, yticklabels=label_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Stage 2 - Distress Type Confusion Matrix ({tag.title()})")
    fig.tight_layout()
    cm_filename = f"{tag}_stage2_cm.png"
    fig.savefig(cm_dir / cm_filename, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to: {cm_dir / cm_filename}")

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
    args = parser.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    model, processor = load_model(args.model, args.adapter_path)

    all_results = {}

    if args.stage in ("1", "both"):
        results_s1 = evaluate_stage1(model, processor, args.max_samples)
        if results_s1:
            all_results["stage1"] = results_s1

    if args.stage in ("2", "both"):
        results_s2 = evaluate_stage2(model, processor, args.max_samples)
        if results_s2:
            all_results["stage2"] = results_s2

    # Save all results
    suffix = "finetuned" if args.adapter_path else "baseline"
    results_path = EVAL_DIR / f"{suffix}_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to: {results_path}")


if __name__ == "__main__":
    main()
