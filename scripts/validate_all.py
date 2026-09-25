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
# Stage 1 confidence maths stays inline here (it is a two-token softmax tied to
# the cached class token IDs). Stage 2 confidence moved to scripts/confidence.py
# so production and the eval scripts share one implementation - test 3c asserts
# that module still does the real logit work.
try:
    code = open("app/model.py", encoding="utf-8").read()
    patterns = [
        "output_scores=True", "return_dict_in_generate=True",
        "_compute_stage1_confidence", "_compute_sequence_confidence",
        "_compute_field_confidence",  # field-restricted Stage 2 confidence
        "_find_field_token_span",
        "needs_expert_review", "torch.softmax",
        "_normal_token_ids", "_distress_token_ids",
        "predict_is_pavement",  # Stage 0 pre-filter
        "scripts import confidence",  # Stage 2 maths is delegated, not copied
        "_stage2_confidences",  # every Stage 2 metric recorded on every call
    ]
    missing = [p for p in patterns if p not in code]
    check("model.py structure", len(missing) == 0, f"Missing: {missing}")
except Exception as e:
    check("model.py structure", False, str(e))

# ---- TEST 3c: confidence.py is the single source of the Stage 2 maths ----
# If this module ever stops computing from real logits, or app/model.py grows
# its own copy again, every confidence number in the paper becomes suspect.
try:
    code = open("scripts/confidence.py", encoding="utf-8").read()
    patterns = [
        "torch.log_softmax",        # real per-token log-probabilities
        "def token_log_probs",
        "def geomean",
        "def sequence_confidence",
        "def field_confidence",
        "def find_field_token_span",
        "def both_confidences",     # callers must be able to record both
        "def type_confidences",     # per-label probabilities (primary mode)
        "def primary_entry",        # aligns the gated label with the parser
    ]
    missing = [p for p in patterns if p not in code]
    model_code = open("app/model.py", encoding="utf-8").read()
    # A second implementation in model.py would silently diverge from this one.
    duplicated = "torch.log_softmax" in model_code
    check("confidence.py single source",
          len(missing) == 0 and not duplicated,
          f"Missing: {missing}" + ("; log_softmax duplicated in app/model.py"
                                   if duplicated else ""))
except Exception as e:
    check("confidence.py single source", False, str(e))

# ---- TEST 3b: model_loader.py guardrails ----
# These guardrails used to live inline in app/model.py. They now live in the
# shared loader that BOTH production and evaluation use — assert they survived
# the move, because silently losing any of them is a production hazard.
try:
    code = open("scripts/model_loader.py", encoding="utf-8").read()
    patterns = [
        "peft #2586",            # must never merge_and_unload on a quantized base
        "flash_attention_2",     # dynamic attention selection
        "PYTORCH_CUDA_ALLOC_CONF",  # fragmentation prevention
        "OutOfMemoryError",      # OOM fallback ladder
        "def self_test",         # load-time end-to-end verification
        "def preflight_vram",    # VRAM check before loading
        "def resolve_model_class",  # family-agnostic class resolution
    ]
    missing = [p for p in patterns if p not in code]
    check("model_loader.py guardrails", len(missing) == 0, f"Missing: {missing}")
except Exception as e:
    check("model_loader.py guardrails", False, str(e))

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
    # These assert the config as HARDENED during Phase 2a (commits ec4cffb ->
    # 5c539f1 + bb05df8). The previous version of this test still asserted the
    # pre-hardening values (label_smoothing 0.1, template qwen2_5_vl, liger on,
    # plain adamw_8bit) and had therefore been failing since 2026-05-01 while
    # the YAML was correct. Each value below is load-bearing:
    patterns = [
        "weight_decay: 0.05",
        "label_smoothing_factor: 0.0",   # 0.1 OOM'd at step 1 (5GB log_softmax)
        "load_best_model_at_end: false",  # would force save_steps % eval_steps
        "neftune_noise_alpha: 5.0",
        "template: qwen2_vl",            # qwen2_5_vl no longer exists in LF
        "flash_attn: fa2",
        "enable_liger_kernel: false",    # Liger is incompatible with QLoRA
        "gradient_checkpointing: true",
        "optim: paged_adamw_8bit",       # canonical QLoRA optimizer for <=24GB
        "per_device_train_batch_size: 1",
        "gradient_accumulation_steps: 32",
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

# ---- TEST 13: IRC:82 taxonomy corrections (2026-09-23) ----
try:
    import subprocess
    from scripts.irc82_taxonomy import (
        IRC82_CONDITION_INDICATORS, IRC82_DISTRESS_TAXONOMY, canonicalize_to_irc,
    )
    # 07_cross_dataset_eval imports torch, seaborn and pandas at module level.
    # After this suite has already loaded the app modules, that import dies
    # with a Windows access violation inside os.stat (DLL load order, not
    # our code), so the Attain parser is checked in a fresh interpreter.
    _probe_src = (
        "import importlib.util as u, sys\n"
        "sp = u.spec_from_file_location('_ae', r'%s')\n"
        "m = u.module_from_spec(sp); sp.loader.exec_module(m)\n"
        "ok = (m.parse_attain_class('Patch and utility cut- Low') == ('Patch and utility cut', 'Low')\n"
        "      and m.parse_attain_class('Alligator crack - High') == ('Alligator crack', 'High'))\n"
        "sys.exit(0 if ok else 3)\n"
    ) % str(Path(__file__).resolve().parent / "07_cross_dataset_eval.py")
    _rc = subprocess.run([sys.executable, "-c", _probe_src], capture_output=True,
                         timeout=300).returncode
    check("IRC taxonomy corrections",
          len(IRC82_DISTRESS_TAXONOMY) == 18
          and "Patching" in IRC82_CONDITION_INDICATORS
          and "Patching" not in IRC82_DISTRESS_TAXONOMY
          # IRC:82 §7.3.5.1: block cracking is a form of transverse cracking
          and canonicalize_to_irc("Block Crack (D43)") == "Transverse Cracking"
          and canonicalize_to_irc("patch") == "Patching"
          # Attain spells one class without a space before the dash
          and _rc == 0,
          f"block->Transverse, patch->Patching indicator, Attain severity split (parser rc={_rc})")
except Exception as e:
    check("IRC taxonomy corrections", False, str(e))

# ---- TEST 14: Stage 2 probe rule is one implementation, and it fails safe ----
try:
    import json as _json
    from scripts import stage2_probe_rules as _rules
    from scripts.stage2_probe import all_probe_types
    _keys = {t.key for t in all_probe_types()}
    model_code = open("app/model.py", encoding="utf-8").read()
    report_code = open("scripts/stage2_probe_report.py", encoding="utf-8").read()
    _cfg = {"schema_version": 1, "variant": {"system_style": "min", "question_style": "def"},
            "verify_threshold": 0.5,
            "groups": {"Pothole": {"kind": "distress", "keys": ["Potholes"],
                                   "threshold": 0.5, "platt": None}}}
    _rules.validate_config(_cfg, _keys)
    _p = {k: 0.0 for k in _keys}
    _p.update({"Potholes": 0.9, "Bleeding": 0.2})
    _out = _rules.combine(["Bleeding"], _p, _cfg)
    _bad_rejected = False
    try:
        _rules.validate_config(dict(_cfg, schema_version=99), _keys)
    except ValueError:
        _bad_rejected = True
    _live = Path("configs/stage2_probe.json")
    _live_ok = True
    if _live.exists():
        _rules.validate_config(_json.loads(_live.read_text(encoding="utf-8")), _keys)
    check("stage2 probe rule",
          "stage2_probe_rules as rules" in model_code and "rules.combine(" in model_code
          and "combine(" in report_code
          and "_init_probe" in model_code and "parity_check()" in model_code
          and "self._prober = None" in model_code          # dropped before a reload
          and _out["types"] == ["Potholes"] and _out["removed_by_probe"] == ["Bleeding"]
          and _bad_rejected and _live_ok,
          "combine() must be shared by production and the report; configs must validate")
except Exception as e:
    check("stage2 probe rule", False, str(e))

# ---- TEST 15: multi-label metrics are prevalence-honest ----
try:
    from scripts import multilabel_metrics as _M
    _y = [True] * 8 + [False] * 2
    check("multilabel metrics",
          _M.binary_metrics(_y, [True] * 10)["mcc"] == 0.0            # constant predictor
          and abs(_M.binary_metrics(_y, [True] * 10)["f1"] - 16 / 18) < 1e-9   # ...still scores F1 0.89
          and _M.roc_auc([0.9, 0.8, 0.1], [True, True, False]) == 1.0
          and _M.roc_auc([0.5, 0.5], [True, False]) == 0.5
          and abs(_M.mcnemar_exact(9, 1) - 0.021484375) < 1e-12,
          "MCC must be 0 for a constant predictor")
except Exception as e:
    check("multilabel metrics", False, str(e))

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
