"""
Step 6: Incremental retraining on expert corrections.

Instead of retraining on ALL data (12-24 hours), this loads the existing
LoRA adapter and fine-tunes ONLY on new expert-corrected samples,
mixed with a small random sample from the original training data
to prevent catastrophic forgetting.

Expected time: ~5-30 minutes depending on number of corrections.

Usage:
    python scripts/06_incremental_retrain.py --corrections-file data/expert_corrections.json
    python scripts/06_incremental_retrain.py --corrections-file data/expert_corrections.json --mix-ratio 0.1

The corrections file is a JSON list of:
  [
    {
      "image_path": "/absolute/path/to/image.jpg",
      "distress_types": ["Pothole (D40)", "Alligator Crack (D20)"],
      "severity": "High"
    },
    ...
  ]
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils import (
    DATA_DIR,
    OUTPUTS_DIR,
    PROJECT_ROOT,
    STAGE2_SYSTEM_PROMPT,
    STAGE2_USER_PROMPT,
    build_stage2_training_response,
    RDD_LABEL_TO_ID,
)

SEED = 42
random.seed(SEED)


def build_correction_entry(correction: dict) -> dict:
    """Convert an expert correction into a ShareGPT training entry."""
    # Map type names back to class IDs for building training response
    class_ids = []
    for dtype in correction["distress_types"]:
        # Try direct lookup
        cid = RDD_LABEL_TO_ID.get(dtype)
        if cid is not None:
            class_ids.append(cid)
        else:
            # For custom types not in RDD, build a custom response
            pass

    # Build the expected response
    if class_ids:
        response = build_stage2_training_response(class_ids)
    else:
        # Custom types — build response manually
        types_str = ", ".join(correction["distress_types"])
        severity = correction.get("severity", "Medium")
        desc_parts = [t.split(" (")[0].lower() for t in correction["distress_types"]]
        description = f"The pavement shows signs of {' and '.join(desc_parts)} damage."
        response = (
            f"DISTRESS_TYPES: {types_str}\n"
            f"SEVERITY: {severity}\n"
            f"DESCRIPTION: {description}"
        )

    # Normalize image path
    img_path = correction["image_path"].replace("\\", "/")

    return {
        "conversations": [
            {"from": "human", "value": STAGE2_USER_PROMPT},
            {"from": "gpt", "value": response},
        ],
        "system": STAGE2_SYSTEM_PROMPT,
        "images": [img_path],
    }


def main():
    parser = argparse.ArgumentParser(description="Incremental retrain on expert corrections")
    parser.add_argument("--corrections-file", type=str, required=True,
                        help="Path to JSON file with expert corrections")
    parser.add_argument("--mix-ratio", type=float, default=0.1,
                        help="Ratio of original training data to mix in (0.1 = 10%%)")
    parser.add_argument("--adapter-path", type=str,
                        default=str(OUTPUTS_DIR / "qwen25vl-qlora-gaps-rdd"),
                        help="Path to existing LoRA adapter to continue from")
    parser.add_argument("--output-dir", type=str,
                        default=str(OUTPUTS_DIR / "qwen25vl-incremental"),
                        help="Output directory for the new adapter")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of epochs for incremental training (keep low)")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate (lower than initial training)")
    args = parser.parse_args()

    # Load expert corrections
    corrections_path = Path(args.corrections_file)
    if not corrections_path.exists():
        print(f"ERROR: Corrections file not found: {corrections_path}")
        return

    with open(corrections_path, "r", encoding="utf-8") as f:
        corrections = json.load(f)

    print(f"Loaded {len(corrections)} expert corrections")

    # Convert to training entries
    correction_entries = []
    for c in corrections:
        entry = build_correction_entry(c)
        correction_entries.append(entry)

    # Mix in original training data to prevent forgetting
    original_train_path = DATA_DIR / "combined_train.json"
    mix_entries = []
    if original_train_path.exists() and args.mix_ratio > 0:
        with open(original_train_path, "r", encoding="utf-8") as f:
            original_data = json.load(f)
        mix_count = max(1, int(len(correction_entries) * args.mix_ratio / (1 - args.mix_ratio)))
        mix_count = min(mix_count, len(original_data))
        mix_entries = random.sample(original_data, mix_count)
        print(f"Mixing in {len(mix_entries)} samples from original training data")

    # Combine and shuffle
    all_entries = correction_entries + mix_entries
    random.shuffle(all_entries)
    print(f"Total incremental training samples: {len(all_entries)}")

    # Save incremental training data
    incr_data_path = DATA_DIR / "incremental_train.json"
    with open(incr_data_path, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, indent=2, ensure_ascii=False)
    print(f"Saved to: {incr_data_path}")

    # Update dataset_info.json to include incremental dataset
    info_path = DATA_DIR / "dataset_info.json"
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    info["incremental_train"] = {
        "file_name": "incremental_train.json",
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "system": "system",
            "images": "images",
        },
    }

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    # Generate incremental training YAML
    incr_yaml = f"""### Incremental Retrain Config (auto-generated)
### Trains ONLY on expert corrections + small mix of original data
### Starting from existing adapter weights

model_name_or_path: Qwen/Qwen2.5-VL-7B-Instruct
adapter_name_or_path: {args.adapter_path}
trust_remote_code: true

stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: 64
lora_alpha: 128
lora_dropout: 0.1

quantization_bit: 4
quantization_method: bitsandbytes

dataset_dir: {str(DATA_DIR).replace(chr(92), '/')}
dataset: incremental_train
template: qwen2_5_vl
cutoff_len: 2048
overwrite_cache: true

output_dir: {args.output_dir.replace(chr(92), '/')}
logging_steps: 10
save_steps: 50
save_total_limit: 2
plot_loss: true
overwrite_output_dir: true

per_device_train_batch_size: 2
gradient_accumulation_steps: 2
learning_rate: {args.lr}
num_train_epochs: {args.epochs}
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true

weight_decay: 0.01
label_smoothing_factor: 0.1
freeze_vision_tower: true

report_to: none
"""

    yaml_path = PROJECT_ROOT / "configs" / "incremental_retrain.yaml"
    yaml_path.write_text(incr_yaml, encoding="utf-8")
    print(f"\nIncremental training config saved to: {yaml_path}")

    print(f"\nTo start incremental retraining, run:")
    print(f"  llamafactory-cli train {yaml_path}")
    print(f"\nExpected time: ~{max(5, len(all_entries) // 10)} minutes")
    print(f"New adapter will be saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
