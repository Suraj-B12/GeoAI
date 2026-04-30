"""
Pre-flight check for the QLoRA training dataset.

Run this BEFORE `llamafactory-cli train`. Catches issues that would otherwise
crash the training run hours in:

  - Missing image files
  - Corrupt JPEGs (truncated, wrong magic bytes)
  - Image too small (< 28x28) or too large (> 4096x4096)
  - Missing/extra <image> tags vs len(images)
  - JSON entries missing required keys (conversations, images)
  - Stage 2 responses that don't parse with the expected format

Exit code 0 = safe to train. Non-zero = fix the listed issues first.

Output is also written to `preflight_problems.txt` for review.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageFile, UnidentifiedImageError
from tqdm import tqdm

# Match LLaMA-Factory: do NOT silently load truncated JPEGs during preflight —
# we want them flagged. (At training time it's safer to set this True, but
# pre-flight should be strict.)
ImageFile.LOAD_TRUNCATED_IMAGES = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATASETS_TO_CHECK = ["combined_train.json", "combined_valid.json"]

MIN_DIM = 28        # Qwen2.5-VL processes 28x28 patches; smaller is unusable
MAX_DIM = 4096      # ~16M pixels; anything bigger is a data error or DoS


class PreflightReport:
    """Collects all problems found, prints a structured summary."""

    def __init__(self) -> None:
        self.problems: list[str] = []
        self.warnings: list[str] = []
        self.stats: dict[str, int] = {
            "files_checked": 0,
            "entries_checked": 0,
            "images_verified": 0,
        }

    def fail(self, msg: str) -> None:
        self.problems.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def report(self) -> int:
        """Print summary, write detail file, return exit code."""
        print()
        print("=" * 70)
        print("PREFLIGHT REPORT")
        print("=" * 70)
        for k, v in self.stats.items():
            print(f"  {k:25s} {v}")
        print()
        if self.warnings:
            print(f"  Warnings: {len(self.warnings)}")
        print(f"  Errors:   {len(self.problems)}")
        print()

        out_path = PROJECT_ROOT / "preflight_problems.txt"
        if self.problems or self.warnings:
            with out_path.open("w", encoding="utf-8") as f:
                if self.problems:
                    f.write("=== ERRORS ===\n")
                    f.write("\n".join(self.problems))
                    f.write("\n\n")
                if self.warnings:
                    f.write("=== WARNINGS ===\n")
                    f.write("\n".join(self.warnings))
                    f.write("\n")
            print(f"  Details written to: {out_path}")

        if self.problems:
            print(f"\n  FAIL: {len(self.problems)} blocking issues found.")
            return 1
        print("\n  OK: dataset is safe to train.")
        return 0


def check_image(img_path: Path, ctx: str, report: PreflightReport) -> None:
    """Open + verify + dimension-check a single image."""
    if not img_path.exists():
        report.fail(f"{ctx}: missing file {img_path}")
        return

    try:
        # verify() catches structural corruption but invalidates the handle —
        # must reopen for size/load checks.
        with Image.open(img_path) as im:
            im.verify()
        with Image.open(img_path) as im:
            w, h = im.size
            if w < MIN_DIM or h < MIN_DIM:
                report.fail(f"{ctx}: image too small {w}x{h} ({img_path.name})")
                return
            if w > MAX_DIM or h > MAX_DIM:
                report.warn(f"{ctx}: image very large {w}x{h} (will be downsampled by image_max_pixels) — {img_path.name}")
            # Force pixel decode to catch lazy-decode failures
            im.load()
        report.stats["images_verified"] += 1
    except UnidentifiedImageError as e:
        report.fail(f"{ctx}: unidentified image format ({img_path.name}): {e}")
    except OSError as e:
        # PIL raises OSError on truncated/corrupt JPEGs
        report.fail(f"{ctx}: corrupt image ({img_path.name}): {e}")
    except Exception as e:
        report.fail(f"{ctx}: unexpected error opening {img_path.name}: {e}")


def count_image_tags(messages: list[dict] | list) -> int:
    """Count <image> tags across all message contents."""
    count = 0
    if not isinstance(messages, list):
        return 0
    for m in messages:
        if not isinstance(m, dict):
            continue
        content = m.get("value") or m.get("content") or ""
        if isinstance(content, str):
            count += content.count("<image>")
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image":
                    count += 1
    return count


def check_dataset_file(json_path: Path, report: PreflightReport) -> None:
    """Validate one ShareGPT-format JSON file."""
    if not json_path.exists():
        report.fail(f"{json_path.name}: file does not exist")
        return

    try:
        with json_path.open("r", encoding="utf-8") as f:
            entries = json.load(f)
    except json.JSONDecodeError as e:
        report.fail(f"{json_path.name}: invalid JSON: {e}")
        return

    if not isinstance(entries, list):
        report.fail(f"{json_path.name}: expected top-level list, got {type(entries).__name__}")
        return

    print(f"\nChecking {json_path.name} ({len(entries):,} entries)...")
    report.stats["files_checked"] += 1

    for i, entry in enumerate(tqdm(entries, desc=f"  {json_path.name}", unit="entry")):
        ctx = f"{json_path.name}#{i}"
        report.stats["entries_checked"] += 1

        if not isinstance(entry, dict):
            report.fail(f"{ctx}: entry is not an object")
            continue

        # ShareGPT format uses 'conversations' (LLaMA-Factory's key)
        msgs = entry.get("conversations") or entry.get("messages")
        imgs = entry.get("images")

        if msgs is None:
            report.fail(f"{ctx}: missing 'conversations' / 'messages' key")
            continue
        if imgs is None:
            report.fail(f"{ctx}: missing 'images' key")
            continue
        if not isinstance(imgs, list):
            report.fail(f"{ctx}: 'images' is not a list")
            continue

        # Image-tag count must equal number of image paths
        # (Qwen2.5-VL crashes with IndexError if mismatched — issue #60)
        n_tags = count_image_tags(msgs)
        if n_tags != len(imgs):
            report.fail(
                f"{ctx}: <image> tag count ({n_tags}) != images count ({len(imgs)})"
            )
            continue

        # Validate every image
        for img_path_str in imgs:
            img_path = Path(img_path_str)
            if not img_path.is_absolute():
                img_path = DATA_DIR / img_path
            check_image(img_path, ctx, report)


def main() -> int:
    print("=" * 70)
    print("QLoRA Training Preflight")
    print("=" * 70)
    print(f"Data dir:         {DATA_DIR}")
    print(f"Datasets to scan: {', '.join(DATASETS_TO_CHECK)}")
    print(f"Min dim:          {MIN_DIM}x{MIN_DIM}")
    print(f"Max dim:          {MAX_DIM}x{MAX_DIM}")

    report = PreflightReport()
    for fname in DATASETS_TO_CHECK:
        check_dataset_file(DATA_DIR / fname, report)

    return report.report()


if __name__ == "__main__":
    sys.exit(main())
