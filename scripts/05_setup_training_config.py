"""
Step 5 (helper): Auto-configure the LLaMA-Factory YAML with correct
absolute paths for the current machine.

Run this ONCE on the machine where you'll fine-tune (e.g., the A6000 at college)
AFTER running 01 + 02 data prep scripts.

Usage:
    python scripts/05_setup_training_config.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils import PROJECT_ROOT, DATA_DIR, OUTPUTS_DIR


def main():
    config_path = PROJECT_ROOT / "configs" / "qwen25vl_qlora_sft.yaml"

    if not config_path.exists():
        print(f"ERROR: Config not found at {config_path}")
        return

    content = config_path.read_text(encoding="utf-8")

    # Build absolute paths with forward slashes (works on both Windows and Linux)
    abs_data_dir = str(DATA_DIR).replace("\\", "/")
    abs_output_dir = str(OUTPUTS_DIR / "qwen25vl-qlora-gaps-rdd").replace("\\", "/")

    # Replace placeholders
    content = content.replace(
        "MUST_UPDATE_TO_ABSOLUTE_PATH/data",
        abs_data_dir,
    )
    content = content.replace(
        "MUST_UPDATE_TO_ABSOLUTE_PATH/outputs/qwen25vl-qlora-gaps-rdd",
        abs_output_dir,
    )

    config_path.write_text(content, encoding="utf-8")

    print(f"Config updated: {config_path}")
    print(f"  dataset_dir: {abs_data_dir}")
    print(f"  output_dir:  {abs_output_dir}")

    # Verify data files exist
    for f in ["combined_train.json", "combined_valid.json", "dataset_info.json"]:
        fp = DATA_DIR / f
        if fp.exists():
            print(f"  [OK] {f}")
        else:
            print(f"  [MISSING] {f} — run 01 + 02 scripts first!")


if __name__ == "__main__":
    main()
