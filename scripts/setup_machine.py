"""
Cross-platform setup script for the Pavement Distress Classification project.

Automates the full setup on Windows (RTX 4050) or Linux (A6000):
  1. Detects OS and GPU
  2. Validates Python and CUDA
  3. Checks that datasets exist
  4. Runs data prep if needed (01 + 02)
  5. Configures training YAML paths (05)
  6. Validates everything

Usage:
    python scripts/setup_machine.py
"""

import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils import PROJECT_ROOT, GAPS_RAW_ROOT, RDD_RAW_ROOT, DATA_DIR


def step(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def run(cmd, check=True):
    """Run a command and return stdout."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"  FAILED: {result.stderr.strip()}")
        return None
    return result.stdout.strip()


def main():
    print("Pavement Distress Classification — Machine Setup")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Python: {sys.version}")

    errors = []

    # ---- Step 1: Check GPU ----
    step("1. Checking GPU")
    gpu = run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", check=False)
    if gpu:
        print(f"  GPU: {gpu}")
    else:
        print("  WARNING: nvidia-smi not found or no GPU detected.")
        print("  You can still run data prep, but inference/training needs a GPU.")

    # ---- Step 2: Check CUDA via PyTorch ----
    step("2. Checking PyTorch + CUDA")
    cuda_check = run(f'{sys.executable} -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else \'none\')"', check=False)
    if cuda_check:
        print(f"  PyTorch CUDA: {cuda_check}")
    else:
        print("  WARNING: PyTorch not installed or CUDA not available.")
        print("  Install: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")

    # ---- Step 3: Check datasets ----
    step("3. Checking datasets")
    if GAPS_RAW_ROOT.exists():
        npy_count = len(list(GAPS_RAW_ROOT.glob("**/*.npy")))
        print(f"  GAPs V2: Found ({npy_count} .npy files)")
    else:
        errors.append(f"GAPs dataset not found at {GAPS_RAW_ROOT}")
        print(f"  ERROR: GAPs not found at {GAPS_RAW_ROOT}")

    if RDD_RAW_ROOT.exists():
        jpg_count = len(list(RDD_RAW_ROOT.glob("**/*.jpg")))
        print(f"  RDD: Found ({jpg_count} .jpg files)")
    else:
        errors.append(f"RDD dataset not found at {RDD_RAW_ROOT}")
        print(f"  ERROR: RDD not found at {RDD_RAW_ROOT}")

    # ---- Step 4: Run data prep if needed ----
    step("4. Data preparation")
    combined_json = DATA_DIR / "combined_train.json"
    if combined_json.exists():
        print(f"  Training data already exists: {combined_json}")
        print("  Skipping data prep (delete data/ folder to re-generate).")
    else:
        print("  Running 01_convert_gaps_to_images.py...")
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "01_convert_gaps_to_images.py")],
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            errors.append("01_convert_gaps_to_images.py failed")

        print("  Running 02_build_training_data.py...")
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "02_build_training_data.py")],
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            errors.append("02_build_training_data.py failed")

    # ---- Step 5: Configure training YAML ----
    step("5. Configuring training YAML")
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "05_setup_training_config.py")],
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        print("  WARNING: YAML configuration failed. Run 05_setup_training_config.py manually.")

    # ---- Step 6: Validate ----
    step("6. Validation")

    checks = {
        "data/combined_train.json": DATA_DIR / "combined_train.json",
        "data/combined_valid.json": DATA_DIR / "combined_valid.json",
        "data/dataset_info.json": DATA_DIR / "dataset_info.json",
        "data/gaps_test.json": DATA_DIR / "gaps_test.json",
        "data/rdd_test.json": DATA_DIR / "rdd_test.json",
    }

    for name, path in checks.items():
        if path.exists():
            print(f"  [OK] {name}")
        else:
            print(f"  [MISSING] {name}")
            errors.append(f"Missing: {name}")

    # Check YAML is configured
    yaml_path = PROJECT_ROOT / "configs" / "qwen25vl_qlora_sft.yaml"
    if yaml_path.exists():
        content = yaml_path.read_text()
        if "MUST_UPDATE" in content:
            print("  [WARN] Training YAML still has placeholder paths")
            errors.append("YAML not configured — run 05_setup_training_config.py")
        else:
            print("  [OK] Training YAML configured")

    # ---- Summary ----
    step("SUMMARY")
    if errors:
        print(f"  {len(errors)} issue(s) found:")
        for e in errors:
            print(f"    - {e}")
    else:
        print("  All checks passed! Ready to train and serve.")
        print()
        print("  Next steps:")
        print("    1. Baseline eval:  python scripts/03_baseline_eval.py --stage both --max-samples 100")
        print("    2. Fine-tune:      llamafactory-cli train configs/qwen25vl_qlora_sft.yaml")
        print("    3. Gradio UI:      python gradio_ui/demo.py")
        print("    4. API server:     uvicorn app.main:app --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()
