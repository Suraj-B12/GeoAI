"""
Step 2: Build LLaMA-Factory ShareGPT-format training JSONs
for BOTH datasets (GAPs binary + RDD multi-class).

This script creates:
  - data/gaps_train.json        (Stage 1: binary detection training)
  - data/gaps_valid.json        (Stage 1: binary detection validation)
  - data/gaps_test.json         (Stage 1: binary detection test)
  - data/rdd_train.json         (Stage 2: distress type training)
  - data/rdd_valid.json         (Stage 2: distress type validation)
  - data/rdd_test.json          (Stage 2: distress type test)
  - data/combined_train.json    (Both stages merged for single training run)
  - data/combined_valid.json    (Both stages merged for validation)
  - data/dataset_info.json      (LLaMA-Factory dataset registry)

Usage:
    python scripts/02_build_training_data.py
"""

import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils import (
    DATA_DIR,
    GAPS_LABEL_MAP,
    IMAGE_DIR,
    RDD_LABEL_MAP,
    RDD_RAW_ROOT,
    STAGE1_SYSTEM_PROMPT,
    STAGE1_USER_PROMPT,
    STAGE2_SYSTEM_PROMPT,
    STAGE2_USER_PROMPT,
    build_stage2_training_response,
)

SEED = 42
random.seed(SEED)


# ============================================================
# GAPs Dataset -> ShareGPT
# ============================================================

def build_gaps_sharegpt(split_name: str) -> list[dict]:
    """
    Read the GAPs manifest CSV and convert to ShareGPT format.
    Expects the manifest from step 01.
    """
    manifest_path = DATA_DIR / f"gaps_{split_name}_manifest.csv"
    if not manifest_path.exists():
        print(f"  WARNING: Manifest not found: {manifest_path}")
        print(f"  Run 01_convert_gaps_to_images.py first!")
        return []

    entries = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_id = int(row["label_id"])
            label_name = row["label_name"]  # "Normal" or "Distress"

            # Use absolute path — avoids all relative-path ambiguity
            img_path = row["absolute_path"]

            entry = {
                "conversations": [
                    {
                        "from": "human",
                        "value": STAGE1_USER_PROMPT,
                    },
                    {
                        "from": "gpt",
                        "value": label_name,
                    },
                ],
                "system": STAGE1_SYSTEM_PROMPT,
                "images": [img_path],
            }
            entries.append(entry)

    return entries


# ============================================================
# RDD Dataset -> ShareGPT
# ============================================================

def parse_yolo_labels(label_path: Path) -> list[int]:
    """
    Parse a YOLO-format label file and return list of class IDs.
    Empty files / missing files return an empty list (= Normal).
    """
    if not label_path.exists():
        return []

    class_ids = []
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if parts:
            try:
                class_ids.append(int(parts[0]))
            except ValueError:
                continue

    return class_ids


def build_rdd_sharegpt(split_name: str) -> list[dict]:
    """
    Convert RDD images + YOLO labels into ShareGPT format.

    For each image:
      - Parse the label file to get class IDs
      - Build a structured training response
      - Use absolute image path (will be adjusted for LLaMA-Factory)
    """
    # Map split names: our convention -> RDD folder names
    rdd_split_map = {
        "train": "train",
        "val": "val",
        "test": "test",
    }
    rdd_split = rdd_split_map.get(split_name, split_name)

    images_dir = RDD_RAW_ROOT / rdd_split / "images"
    labels_dir = RDD_RAW_ROOT / rdd_split / "labels"

    if not images_dir.exists():
        print(f"  WARNING: RDD images dir not found: {images_dir}")
        return []

    image_files = sorted(images_dir.glob("*.jpg"))
    print(f"  Found {len(image_files)} images in RDD/{rdd_split}")

    entries = []
    class_counter = Counter()

    for img_path in tqdm(image_files, desc=f"  RDD {split_name}"):
        label_path = labels_dir / (img_path.stem + ".txt")
        class_ids = parse_yolo_labels(label_path)

        # Track class distribution
        if class_ids:
            for cid in set(class_ids):
                class_counter[cid] += 1
        else:
            class_counter["normal"] += 1

        # Build response
        response = build_stage2_training_response(class_ids)

        # Use absolute path — avoids all relative-path ambiguity
        abs_path = str(img_path).replace("\\", "/")

        entry = {
            "conversations": [
                {
                    "from": "human",
                    "value": STAGE2_USER_PROMPT,
                },
                {
                    "from": "gpt",
                    "value": response,
                },
            ],
            "system": STAGE2_SYSTEM_PROMPT,
            "images": [abs_path],
        }
        entries.append(entry)

    # Print class distribution
    print(f"  Class distribution for {split_name}:")
    for key in sorted(class_counter.keys(), key=lambda x: str(x)):
        label = RDD_LABEL_MAP.get(key, str(key))
        print(f"    {label}: {class_counter[key]} images")

    return entries


# ============================================================
# Dataset Info for LLaMA-Factory
# ============================================================

def write_dataset_info():
    """Generate the dataset_info.json that LLaMA-Factory requires."""
    info = {
        "gaps_train": {
            "file_name": "gaps_train.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "system": "system",
                "images": "images",
            },
        },
        "gaps_valid": {
            "file_name": "gaps_valid.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "system": "system",
                "images": "images",
            },
        },
        "rdd_train": {
            "file_name": "rdd_train.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "system": "system",
                "images": "images",
            },
        },
        "rdd_valid": {
            "file_name": "rdd_valid.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "system": "system",
                "images": "images",
            },
        },
        "combined_train": {
            "file_name": "combined_train.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "system": "system",
                "images": "images",
            },
        },
        "combined_valid": {
            "file_name": "combined_valid.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "system": "system",
                "images": "images",
            },
        },
    }

    out_path = DATA_DIR / "dataset_info.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"\nDataset info written to: {out_path}")


# ============================================================
# Main
# ============================================================

def save_json(data: list[dict], filename: str) -> None:
    """Save a list of entries as JSON."""
    out_path = DATA_DIR / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved {out_path.name}: {len(data)} entries")


def main():
    print("=" * 60)
    print("Building LLaMA-Factory Training Data")
    print("=" * 60)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ---- GAPs (Stage 1: Binary Detection) ----
    print("\n--- GAPs Dataset (Stage 1: Binary Detection) ---")

    gaps_train = build_gaps_sharegpt("train")
    gaps_valid = build_gaps_sharegpt("valid")
    gaps_test = build_gaps_sharegpt("test")

    if gaps_train:
        random.shuffle(gaps_train)
        save_json(gaps_train, "gaps_train.json")
    if gaps_valid:
        save_json(gaps_valid, "gaps_valid.json")
    if gaps_test:
        save_json(gaps_test, "gaps_test.json")

    # ---- RDD (Stage 2: Distress Type Classification) ----
    print("\n--- RDD Dataset (Stage 2: Distress Type Classification) ---")

    rdd_train = build_rdd_sharegpt("train")
    rdd_valid = build_rdd_sharegpt("val")
    rdd_test = build_rdd_sharegpt("test")

    if rdd_train:
        random.shuffle(rdd_train)
        save_json(rdd_train, "rdd_train.json")
    if rdd_valid:
        save_json(rdd_valid, "rdd_valid.json")
    if rdd_test:
        save_json(rdd_test, "rdd_test.json")

    # ---- Combined (Both stages for single training run) ----
    print("\n--- Combined Dataset ---")

    combined_train = gaps_train + rdd_train
    random.shuffle(combined_train)
    save_json(combined_train, "combined_train.json")

    combined_valid = gaps_valid + rdd_valid
    save_json(combined_valid, "combined_valid.json")

    print(f"\n  Combined train: {len(gaps_train)} GAPs + {len(rdd_train)} RDD = {len(combined_train)} total")
    print(f"  Combined valid: {len(gaps_valid)} GAPs + {len(rdd_valid)} RDD = {len(combined_valid)} total")

    # ---- Dataset Info ----
    write_dataset_info()

    print(f"\n{'='*60}")
    print("DONE! All training data generated.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
