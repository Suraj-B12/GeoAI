"""
Comprehensive validation of all project interdependencies.
Run after any code changes to catch issues.

Usage:
    python scripts/validate_all.py
"""

import glob
import os
import sys
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

errors = []
passed = 0


def check(name, condition, msg=""):
    global passed
    if condition:
        print(f"[OK] {name}")
        passed += 1
    else:
        errors.append(f"{name}: {msg}")
        print(f"[FAIL] {name}: {msg}")


# ---- TEST 1: utils.py ----
try:
    from scripts.utils import (
        PROJECT_ROOT, STAGE2_SYSTEM_PROMPT, STAGE2_FEW_SHOT_EXAMPLES,
        build_stage2_training_response, parse_stage1_response,
        parse_stage2_response, map_stage2_to_rdd_ids,
        detect_system_resources, CONFIDENCE_THRESHOLD,
    )
    check("utils.py exports",
          "Example 1" in STAGE2_SYSTEM_PROMPT and "Example 3" in STAGE2_SYSTEM_PROMPT,
          "Few-shot examples missing from prompt")
    # Verify resource detection works
    sysinfo = detect_system_resources()
    check("utils.py resource detection",
          sysinfo["cpu_physical"] > 0 and sysinfo["ram_total_gb"] > 0 and sysinfo["num_workers"] >= 1,
          f"Bad resource detection: {sysinfo}")
except Exception as e:
    check("utils.py exports", False, str(e))

# ---- TEST 2: schemas.py ----
try:
    from app.schemas import (
        ClassificationResponse,
        ExpertCorrectionRequest, RetrainStatusResponse,
    )
    from scripts.utils import CONFIDENCE_THRESHOLD
    r = ClassificationResponse(
        is_distressed=True, stage1_label="Distress", stage1_confidence=0.92,
        stage2_confidence=0.85, needs_expert_review=False,
        processing_time_ms=1500, stage1_time_ms=500,
    )
    check("schemas.py", isinstance(r.stage1_confidence, float) and CONFIDENCE_THRESHOLD == 0.80,
          "confidence must be float, threshold 0.80")
except Exception as e:
    check("schemas.py", False, str(e))

# ---- TEST 3: model.py structure ----
try:
    code = open("app/model.py", encoding="utf-8").read()
    patterns = [
        "output_scores=True", "return_dict_in_generate=True",
        "_compute_stage1_confidence", "_compute_sequence_confidence",
        "needs_expert_review", "torch.softmax", "torch.log_softmax",
        "_normal_token_ids", "_distress_token_ids",
        "peft bug #2586",  # Must NOT merge on quantized model
        "flash_attention_2",  # Dynamic attention selection
        "PYTORCH_CUDA_ALLOC_CONF",  # Memory fragmentation prevention
    ]
    missing = [p for p in patterns if p not in code]
    check("model.py structure", len(missing) == 0, f"Missing: {missing}")
except Exception as e:
    check("model.py structure", False, str(e))

# ---- TEST 4: security.py ----
try:
    from app.security import verify_api_key, LimitUploadSizeMiddleware, validate_image, MAX_UPLOAD_BYTES
    check("security.py", MAX_UPLOAD_BYTES == 10 * 1024 * 1024, "Max upload should be 10MB")
except Exception as e:
    check("security.py", False, str(e))

# ---- TEST 5: main.py ----
try:
    code = open("app/main.py", encoding="utf-8").read()
    patterns = [
        "slowapi", "verify_api_key", "validate_image", "LimitUploadSizeMiddleware",
        "CORS_ORIGINS", "10/minute", "retrain/start", "retrain/status", "RetrainStatusResponse",
        "asyncio.to_thread",  # Non-blocking inference
    ]
    missing = [p for p in patterns if p not in code]
    check("main.py security", len(missing) == 0, f"Missing: {missing}")
except Exception as e:
    check("main.py security", False, str(e))

# ---- TEST 6: Training YAML ----
try:
    y = open("configs/qwen25vl_qlora_sft.yaml", encoding="utf-8").read()
    patterns = [
        "weight_decay: 0.05", "label_smoothing_factor: 0.1", "load_best_model_at_end: true",
        "neftune_noise_alpha: 5.0", "template: qwen2_5_vl",
        "flash_attn: fa2", "enable_liger_kernel: true", "gradient_checkpointing: true",
        "optim: adamw_8bit",
    ]
    missing = [p for p in patterns if p not in y]
    check("training YAML", len(missing) == 0, f"Missing: {missing}")
except Exception as e:
    check("training YAML", False, str(e))

# ---- TEST 7: DB schema ----
try:
    sql = open("db_schema.sql", encoding="utf-8").read()
    patterns = [
        "needs_expert_review", "expert_corrected_types", "custom_distress_types",
        "retrain_jobs", "stage1_confidence REAL", "stage2_confidence REAL", "ROW LEVEL SECURITY",
    ]
    missing = [p for p in patterns if p not in sql]
    check("db_schema.sql", len(missing) == 0, f"Missing: {missing}")
except Exception as e:
    check("db_schema.sql", False, str(e))

# ---- TEST 8: Expert UI ----
try:
    html = open("expert_ui/index.html", encoding="utf-8").read()
    check("expert_ui",
          "SK Modernist" in html and "grid" in html and "startRetrain" in html,
          "Missing font/grid/retrain in expert UI")
except Exception as e:
    check("expert_ui", False, str(e))

# ---- TEST 9: Round-trip parse ----
try:
    ok = True
    for cids in [[], [0], [1], [2], [3], [0, 3], [0, 1, 2, 3]]:
        resp = build_stage2_training_response(cids)
        parsed = parse_stage2_response(resp)
        recovered = map_stage2_to_rdd_ids(parsed["distress_types"])
        expected = sorted(set(c for c in cids if c <= 3))
        if recovered != expected:
            ok = False
            break
    check("round-trip parsing", ok, f"Failed for {cids}")
except Exception as e:
    check("round-trip parsing", False, str(e))

# ---- TEST 10: Gradio uses shared model ----
try:
    code = open("gradio_ui/demo.py", encoding="utf-8").read()
    check("gradio shared model",
          "from app.model import PavementClassifier" in code and "CONFIDENCE_THRESHOLD" in code,
          "Must import from app.model")
except Exception as e:
    check("gradio shared model", False, str(e))

# ---- TEST 11: Incremental retrain ----
try:
    code = open("scripts/06_incremental_retrain.py", encoding="utf-8").read()
    check("incremental retrain",
          "adapter_name_or_path" in code and "mix_ratio" in code,
          "Must load adapter and mix data")
except Exception as e:
    check("incremental retrain", False, str(e))

# ---- TEST 12: No hardcoded paths ----
try:
    bad_files = []
    # Exclude: dataset files we don't control, and this validation script itself
    exclude = {"validate_all.py", "display_images.py"}
    for f in glob.glob("**/*.py", recursive=True):
        if Path(f).name in exclude:
            continue
        content = open(f, encoding="utf-8", errors="ignore").read()
        if "C:\\Users\\suraj" in content or "C:/Users/suraj" in content:
            bad_files.append(f)
    check("no hardcoded paths", len(bad_files) == 0, f"Found in: {bad_files}")
except Exception as e:
    check("no hardcoded paths", False, str(e))

# ---- SUMMARY ----
total = passed + len(errors)
print(f"\n{'='*50}")
if errors:
    print(f"RESULT: {passed}/{total} passed, {len(errors)} failed")
    for e in errors:
        print(f"  FAIL: {e}")
    sys.exit(1)
else:
    print(f"ALL {total} TESTS PASSED")
