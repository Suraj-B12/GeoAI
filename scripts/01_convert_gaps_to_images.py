"""
Step 1: Convert GAPs V2 .npy chunks into individual PNG images.

The GAPs V2 50k subset stores images as NumPy arrays:
  - X files: shape (N, 1, 160, 160), dtype float32
  - Y files: shape (N,), dtype int32  (0=Normal, 1=Distress)

This script:
  1. Iterates over each split (train, test, valid, valid-test)
  2. Loads each chunk with memory-mapping (no full RAM load)
  3. Rescales float32 pixel values to uint8 [0, 255]
  4. Saves each image as a grayscale PNG using parallel workers
  5. Writes a manifest CSV per split for traceability

Dynamically scales to available CPU cores and RAM.

Usage:
    python scripts/01_convert_gaps_to_images.py
"""

import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# Add project root to path so we can import utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils import (
    GAPS_RAW_ROOT,
    IMAGE_DIR,
    DATA_DIR,
    GAPS_LABEL_MAP,
    detect_system_resources,
    print_system_info,
)


SPLITS = ["train", "test", "valid", "valid-test"]


def rescale_to_uint8(img: np.ndarray) -> np.ndarray:
    """
    Rescale a float32 image to uint8 [0, 255].

    Handles both [-1, 1] and [0, 1] ranges by normalizing
    to [0, 1] first, then scaling to [0, 255].
    """
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        img_normalized = (img - img_min) / (img_max - img_min)
    else:
        img_normalized = np.zeros_like(img)
    return (img_normalized * 255).clip(0, 255).astype(np.uint8)


def _save_batch(args: tuple) -> list[dict]:
    """
    Worker function: convert a batch of images from an .npy chunk to PNGs.
    Each worker loads the chunk via mmap (shared OS page cache, no RAM duplication).
    """
    x_path, y_path, start_idx, end_idx, split_name, out_dir, global_offset = args

    # Memory-mapped read — shared across workers via OS page cache
    X = np.load(x_path, mmap_mode="r")
    Y = np.load(y_path, mmap_mode="r")

    batch_manifest = []
    for local_i in range(start_idx, end_idx):
        global_i = global_offset + local_i
        img = np.array(X[local_i, 0, :, :])  # Copy slice into RAM for processing
        img_uint8 = rescale_to_uint8(img)

        filename = f"{split_name}_{global_i:06d}.png"
        filepath = Path(out_dir) / filename
        Image.fromarray(img_uint8, mode="L").save(filepath)

        abs_path = str(filepath).replace("\\", "/")
        batch_manifest.append({
            "filename": filename,
            "absolute_path": abs_path,
            "label_id": int(Y[local_i]),
            "label_name": GAPS_LABEL_MAP[int(Y[local_i])],
        })

    return batch_manifest


def convert_split(split_name: str, num_workers: int, batch_size: int = 500) -> list[dict]:
    """
    Convert one split's .npy data to PNG images using parallel workers.

    Each worker loads the chunk via mmap and processes a batch of images.
    The OS page cache ensures the .npy data is shared across workers.
    """
    split_dir = GAPS_RAW_ROOT / split_name
    if not split_dir.exists():
        print(f"  WARNING: Split directory not found: {split_dir}")
        return []

    out_prefix = f"gaps_{split_name}"
    out_dir = IMAGE_DIR / out_prefix
    out_dir.mkdir(parents=True, exist_ok=True)

    x_files = sorted(split_dir.glob("*_chunk_*_x.npy"))
    y_files = sorted(split_dir.glob("*_chunk_*_y.npy"))

    if not x_files:
        print(f"  WARNING: No x chunk files found in {split_dir}")
        return []

    print(f"\n{'='*60}")
    print(f"Processing split: {split_name}")
    print(f"  Source: {split_dir}")
    print(f"  Output: {out_dir}")
    print(f"  Workers: {num_workers} | Batch size: {batch_size}")
    print(f"{'='*60}")

    # Build work items: (chunk_file, start, end, ...) for each batch
    work_items = []
    global_offset = 0

    for xf, yf in zip(x_files, y_files):
        # Peek at shape without loading into RAM
        X_peek = np.load(xf, mmap_mode="r")
        chunk_size = X_peek.shape[0]
        print(f"    {xf.name}: {chunk_size} images")

        for start in range(0, chunk_size, batch_size):
            end = min(start + batch_size, chunk_size)
            work_items.append((
                str(xf), str(yf), start, end,
                split_name, str(out_dir), global_offset,
            ))

        global_offset += chunk_size

    print(f"  Total: {global_offset} images in {len(work_items)} batches")

    # Class distribution (quick read from first y file via mmap)
    all_y = np.concatenate([np.load(yf, mmap_mode="r") for yf in y_files])
    unique, counts = np.unique(all_y, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    Label {u} ({GAPS_LABEL_MAP[u]}): {c} images")

    # Parallel conversion
    manifest = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_save_batch, item) for item in work_items]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"  {split_name}"):
            batch_result = future.result()
            manifest.extend(batch_result)

    # Sort manifest by filename to ensure consistent ordering
    manifest.sort(key=lambda x: x["filename"])
    return manifest


def write_manifest(split_name: str, manifest: list[dict]) -> None:
    """Write a CSV manifest for traceability."""
    csv_path = DATA_DIR / f"gaps_{split_name}_manifest.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "absolute_path", "label_id", "label_name"])
        writer.writeheader()
        writer.writerows(manifest)
    print(f"  Manifest saved: {csv_path} ({len(manifest)} entries)")


def main():
    print("=" * 60)
    print("GAPs V2 .npy -> PNG Converter (Parallel)")
    print("=" * 60)

    sysinfo = print_system_info()
    num_workers = sysinfo["parallel_processes"]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    total_converted = 0
    for split in SPLITS:
        manifest = convert_split(split, num_workers=num_workers)
        if manifest:
            write_manifest(split, manifest)
            total_converted += len(manifest)

    print(f"\n{'='*60}")
    print(f"DONE! Converted {total_converted} images total.")
    print(f"Images saved to: {IMAGE_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
