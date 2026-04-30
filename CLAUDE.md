# Pavement Distress Classification — Capstone Project Context

## What This Project Is

A two-stage AI pipeline that classifies pavement distress from phone photos, built by students in Bengaluru. Users snap a photo of a road, the AI detects if there's damage, classifies the type, estimates severity, and flags low-confidence results for expert human review. The long-term goal is a geospatial heatmap of road distress across Bengaluru. This is a college-level capstone project that must produce a publishable paper.

## Architecture

```
Phone Photo → FastAPI /classify → Stage 1 (Normal/Distress?) → Stage 2 (What type?) → JSON Response
                                         ↓                              ↓
                                   confidence < 80%?            Parse: types, severity
                                         ↓                              ↓
                                   needs_expert_review=true      Store in Supabase
                                                                        ↓
                                                              Expert UI reviews + corrects
                                                                        ↓
                                                         Few-shot prompt injection (immediate)
                                                         + LoRA incremental retrain (permanent)
                                                                        ↓
                                                              Versioned adapter saved to adapters/vN/
```

- **Model:** Qwen2.5-VL-7B-Instruct, fine-tuned with QLoRA via LLaMA-Factory
- **Stage 1:** Binary detection (Normal/Distress) — trained on GAPs V2 50k dataset (160x160 grayscale .npy)
- **Stage 2:** Distress type classification — trained on RDD dataset (YOLO format, D00/D10/D20/D40)
- **API:** FastAPI with rate limiting, API key auth, CORS, async inference, SSE streaming
- **DB:** Supabase (PostgreSQL) — tables: assessments, custom_distress_types, retrain_jobs
- **Expert UI:** Single-page HTML/JS dashboard with Supabase Auth, tab-based pending/reviewed views, adapter version management
- **Test Website:** Standalone HTML file for uploading images and getting live AI analysis via SSE
- **Confidence:** Real logit-based extraction (softmax over class token IDs), NOT heuristic strings
- **Remote Access:** Cloudflare Tunnel (`cloudflared tunnel --url http://localhost:8000`)

## Hardware

- **Training GPU:** NVIDIA RTX A5000 (24GB VRAM) at college lab
- **Dev/inference GPU:** RTX 4050 laptop (6GB VRAM)
- **Project storage:** External SSD — project folder is transferred between laptop and college machine via SSD
- Code auto-detects GPU, CPU cores, RAM and scales dynamically
- **Package manager:** Use `uv pip install` instead of `pip` for speed

## Datasets

### GAPs V2 (Stage 1)
- Location: `GAPs v2/v2/NORMvsDISTRESS_50k_160/`
- Format: .npy chunks, shape (N, 1, 160, 160) float32 + int32 labels
- Splits: train (50k), test (10k), valid (10k), valid-test (10k)
- Labels: 0=Normal (60%), 1=Distress (40%)
- Script 01 converts these to PNGs with parallel workers + mmap

### RDD (Stage 2 — Training)
- Location: `RDD/RDD_SPLIT/{train,val,test}/{images,labels}/`
- Format: JPEG images + YOLO .txt label files
- Classes: D00 (longitudinal crack), D10 (transverse), D20 (alligator), D40 (pothole)
- Test split: 5,758 images (used for ALL evaluations — baseline and fine-tuned)
- The model also recognizes types beyond RDD labels via taxonomy injection in the system prompt

### Attain (Stage 2 — Cross-Dataset Evaluation Only, NOT Training)
- Location: `Attain/` (user will download and place here)
- Source: Mendeley Data (doi:10.17632/nykrzdm74f/1), CC BY 4.0
- Images: 2,293 images, 19,761 annotated distress instances
- Classes: 10 types — alligator crack, block crack, longitudinal crack, transverse crack, faded marking, lane/shoulder drop-off, patch/utility cut, pothole, manhole, weathering/raveling
- Severity: Low / Medium / High per instance (RDD doesn't have this)
- Collection: Smartphone-mounted on vehicles, 20-70 km/h, New Zealand roads
- **Purpose:** Zero-shot generalization testing. Model is trained on RDD (4 classes) and evaluated on Attain (10 classes). Classes not in RDD (block crack, raveling, weathering, faded marking, drop-off, manhole) are zero-shot evaluation targets — the model has never trained on them but can recognize them via taxonomy injection in the system prompt.
- **Phase 3 role:** Misclassifications on Attain's unseen classes become expert correction candidates → few-shot injection → LoRA retrain → measure improvement trajectory
- Script: `scripts/07_cross_dataset_eval.py` (needs building — loads Attain, runs inference, computes per-class metrics)
- **IMPORTANT — Annotation format unknown until downloaded:** After downloading Attain, the first task is to inspect the folder structure and annotation format (could be YOLO, VOC XML, COCO JSON, or custom). The `07_cross_dataset_eval.py` script must be written AFTER inspecting the actual data. Run `ls -R Attain/ | head -50` to see structure, then adapt.
- **Attain class → our taxonomy mapping:**
  ```
  Attain class              → ALL_DISTRESS_TYPES match         → In RDD training?
  ──────────────────────────────────────────────────────────────────────────────
  Alligator crack           → Alligator Crack (D20)            → YES (trained)
  Longitudinal crack        → Longitudinal Crack (D00)         → YES (trained)
  Transverse crack          → Transverse Crack (D10)           → YES (trained)
  Pothole                   → Pothole (D40)                    → YES (trained)
  Block crack               → Block Crack (D43)                → NO (zero-shot)
  Patch/utility cut         → Inlaid Patch (D44)/Utility Cut   → NO (zero-shot)
  Weathering                → Weathering/Oxidation             → NO (zero-shot)
  Raveling                  → Raveling                         → NO (zero-shot)
  Faded marking             → (not pavement distress)          → EXCLUDE from eval
  Lane/shoulder drop-off    → (not pavement distress)          → EXCLUDE from eval
  Manhole                   → (not pavement distress)          → EXCLUDE from eval
  ```
  Only 7 of 10 Attain classes are actual pavement distress. Exclude faded marking, drop-off, and manhole from evaluation — they're road features, not damage.

## Critical Technical Decisions (DO NOT change without understanding why)

1. **Template is `qwen2_5_vl`** (NOT `qwen2_vl`) — wrong template = broken training
2. **Never call `merge_and_unload()` on a quantized model** — peft bug #2586, produces broken weights. Keep as PeftModel for inference.
3. **`CONFIDENCE_THRESHOLD = 0.80`** is defined ONCE in `scripts/utils.py` — everything else imports from there
4. **Max 2 training epochs** — research shows 50k datasets overfit at 3+ epochs
5. **`lora_dropout: 0.1`** — acts as sparsity regularizer, critical for preventing overfitting
6. **Flash Attention 2** is auto-detected at runtime (checks GPU compute capability + flash-attn install). Falls back to SDPA if unavailable.
7. **`asyncio.to_thread`** wraps model inference in FastAPI so the event loop stays responsive
8. **Image format validation** checks both PIL `.format` AND `.mode` to prevent bypass when format is unset
9. **`predict()` is split into `predict_stage1()` + `predict_stage2()`** — enables SSE streaming per stage; `predict()` calls both internally (no logic duplication)
10. **SSE uses `fetch()` + `ReadableStream`** on the client, NOT `EventSource` — because `EventSource` only supports GET, and image upload requires POST with multipart/form-data
11. **Windows uses `active_version.txt`** instead of symlinks for adapter versioning — symlinks require admin privileges on Windows
12. **Unparseable Stage 1 responses default to Distress** (not Normal) with 0% confidence — errs on the side of caution and flags for expert review, never silently classifies damaged roads as normal
13. **`image.load()` is called after `Image.open()`** in the SSE endpoint — forces full pixel decode so the BytesIO buffer can be garbage collected, preventing stale references during async streaming
14. **Client disconnect check** before Stage 2 inference in SSE — avoids wasting GPU time if the client closed the connection during Stage 1
15. **No hardcoded file paths** — all paths derive from `PROJECT_ROOT` (auto-detected in `scripts/utils.py`). `validate_all.py` test 12 scans for `C:\Users\suraj` in all .py files and fails if found
16. **No `capture="environment"`** on mobile file input — this forces rear camera and blocks gallery selection. `accept="image/*"` alone lets mobile users choose camera OR gallery

## 3-Phase Implementation Plan

### Phase 1: Raw Model Baseline + Test Website + SSE Streaming (CODE DONE)

**Goal:** Run raw Qwen2.5-VL-7B-Instruct (no fine-tuning), establish baseline accuracy on ALL RDD test images, expose API with SSE streaming, create test website for remote demo.

**What was built:**
- `app/model.py` — split `predict()` into `predict_stage1()` and `predict_stage2()` methods (each thread-safe with `self._lock`), `predict()` refactored to call them internally
- `app/main.py` — added `POST /classify/stream` SSE endpoint using `sse-starlette`, streams progress events per stage (`stage1_start → stage1_complete → stage2_start → stage2_complete → result`)
- `test_website/index.html` — standalone test website matching expert_ui design (same CSS vars, SK Modernist font, colors, button styles), with API URL input, drag-drop image upload (mobile-friendly via `accept="image/*"`), live SSE progress with spinners/checkmarks, confidence bars with 80% threshold line, session history, error recovery on stream failure
- `requirements.txt` — added `sse-starlette>=1.6.0`

**SSE Event Format:**
```
event: progress → {stage: "stage1", status: "running", message: "Detecting if pavement shows distress..."}
event: progress → {stage: "stage1", status: "complete", result: {is_distressed, stage1_label, stage1_confidence, ...}}
event: progress → {stage: "stage2", status: "running", message: "Classifying distress type and severity..."}
event: progress → {stage: "stage2", status: "complete", result: {distress_types, severity, stage2_confidence, ...}}
event: result   → {full ClassificationResponse JSON}
event: error    → {error: "message"} (if something fails)
```

**What needs EXECUTING on college machine:**
```bash
uv pip install -r requirements.txt
python scripts/01_convert_gaps_to_images.py
python scripts/02_build_training_data.py
python scripts/03_baseline_eval.py --stage 1    # GAPs binary (10k images)
python scripts/03_baseline_eval.py --stage 2    # RDD type classification (5,758 images)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
cloudflared tunnel --url http://localhost:8000   # Expose to internet
# Open test_website/index.html in browser, paste tunnel URL, test
```

### Phase 2: Fine-Tuned Model + Comparison + Cross-Dataset Evaluation (CODE NOT YET WRITTEN)

**Goal:** Fine-tune Qwen with QLoRA on RDD2022, re-run same test images, compare accuracy vs baseline. Then evaluate on Attain for zero-shot generalization. Auto-detect and load adapter on API startup.

**What needs to be built:**
- `app/model.py` — add `_find_latest_adapter()` that checks `adapters/active_version.txt` then `outputs/qwen25vl-qlora-gaps-rdd/adapter_config.json`; modify `get_classifier()` to auto-detect if `ADAPTER_PATH` env var not set
- `scripts/07_cross_dataset_eval.py` — loads Attain images+labels, runs predict_stage2() on each, maps Attain's 10 classes to our taxonomy, computes per-class accuracy/F1/confusion matrix, saves to eval_results/cross_dataset_results.json

**PREREQUISITE:** Phase 1 baseline eval MUST be completed first (scripts 01, 02, 03 executed on A5000 with 7B model, baseline_results.json saved). Phase 2 compares AGAINST those baseline numbers.

**What needs EXECUTING (in this exact order):**
```bash
# Step 1: Inspect Attain dataset structure (FIRST — needed to write 07 script)
ls -R Attain/ | head -50

# Step 2: Build cross-dataset eval script based on Attain's actual format
# (07_cross_dataset_eval.py — Claude writes this after seeing folder structure)

# Step 3: Run cross-dataset eval on BASELINE model (zero-shot numbers on Attain)
python scripts/07_cross_dataset_eval.py  # No adapter = baseline model

# Step 4: Fix training config paths for this machine
python scripts/05_setup_training_config.py

# Step 5: Fine-tune (12-24 hours on A5000)
llamafactory-cli train configs/qwen25vl_qlora_sft.yaml

# Step 6: Evaluate fine-tuned model on RDD test set (in-distribution)
python scripts/04_post_finetune_eval.py --adapter-path outputs/qwen25vl-qlora-gaps-rdd/

# Step 7: Evaluate fine-tuned model on Attain (cross-dataset, zero-shot transfer)
python scripts/07_cross_dataset_eval.py --adapter-path outputs/qwen25vl-qlora-gaps-rdd/

# Step 8: Compare all results → three comparison tables for paper:
#   Table 1: Baseline vs Fine-tuned on RDD (in-distribution improvement)
#   Table 2: Baseline vs Fine-tuned on Attain (cross-dataset generalization)
#   Table 3: Known classes (trained) vs Unknown classes (zero-shot) on Attain
```

**Three-tier evaluation (core of the paper):**
1. **In-distribution:** baseline vs fine-tuned on RDD2022 test set (5,758 images, 4 classes)
2. **Cross-dataset zero-shot:** fine-tuned model on Attain (2,293 images, 10 classes) — measures generalization to unseen geographies + unseen distress types
3. **Phase 3 improvement:** expert corrections on Attain misclassifications → few-shot + LoRA → before/after comparison

**Zero-shot evaluation strategy:**
- The system prompt (`STAGE2_SYSTEM_PROMPT` in `utils.py`) includes `ALL_DISTRESS_TYPES` — a comprehensive taxonomy of 20+ distress types with descriptions
- This taxonomy injection enables "open-world recognition": the VLM's pre-trained visual-semantic alignment matches image features against textual descriptions of distress types it was never trained on
- Attain classes NOT in RDD training (block crack, raveling, weathering, faded marking, drop-off, manhole) are the zero-shot evaluation targets
- Published research supports this: DamageQwen (2025, same model family, few-shot improves 18% over zero-shot), Xu et al. (2025, zero-shot LLM beats experts on PSCI), Yong et al. (2023, prompt engineering beats supervised baselines)
- Even 20-30% accuracy on unseen classes is a publishable zero-shot transfer result

### Phase 3: Expert-in-the-Loop + Incremental Learning (CODE NOT YET WRITTEN)

**Goal:** For images with <80% confidence, experts correct predictions. Corrections improve the model TWO ways: (1) immediate few-shot prompt injection, (2) permanent LoRA incremental retrain. Adapters are versioned for rollback.

**Approach C — Both few-shot + LoRA retrain (chosen for paper-worthiness):**
- **Immediate improvement:** Expert corrections are stored and injected as few-shot examples in the Stage 2 system prompt (capped at 5 most recent). Model improves instantly without retraining.
- **Permanent improvement:** Accumulated corrections trigger LoRA incremental retrain (existing `06_incremental_retrain.py`). New adapter saved to `adapters/vN/` with metadata.

**What needs to be built:**
- `scripts/utils.py` — add `build_dynamic_stage2_prompt(corrections: list[dict])` that appends corrections as few-shot examples to `STAGE2_SYSTEM_PROMPT`
- `app/model.py` — add `self._few_shot_corrections` list, `add_correction()` method, `reload_adapter(version)` method; load `data/expert_corrections.json` on startup
- `app/main.py` — add `POST /corrections`, `GET /adapters`, `POST /adapters/switch`, `GET /adapters/active` endpoints
- `scripts/06_incremental_retrain.py` — add versioned output to `adapters/vN/` with `metadata.json`, update `active_version.txt`
- `expert_ui/index.html` — add adapter version dropdown, switch button, current version display; after submitting correction also `POST /corrections` to FastAPI

**Versioned Adapter Structure:**
```
adapters/
├── v1/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── metadata.json  → {version, created_at, corrections_count, base_adapter, metrics}
├── v2/ ...
└── active_version.txt → "v2"
```

**Corrections Flow:**
1. Expert corrects prediction in expert_ui
2. Correction saved to Supabase (authoritative record) AND `POST /corrections` to API (few-shot injection)
3. Correction persisted to `data/expert_corrections.json` (survives API restarts)
4. Next prediction for similar images benefits from few-shot example immediately
5. When enough corrections accumulate, expert clicks "Retrain" → `06_incremental_retrain.py` runs
6. New adapter saved as `adapters/vN/` → `active_version.txt` updated → API reloads adapter

## Project Structure

```
Capstone/
├── GAPs v2/                           # Raw GAPs dataset (.npy)
├── RDD/                               # Raw RDD dataset (YOLO) — Stage 2 TRAINING
├── Attain/                            # Attain dataset — Stage 2 EVALUATION ONLY (10 classes + severity, zero-shot testing)
├── data/                              # Generated: PNGs + training JSONs + expert_corrections.json
├── configs/
│   └── qwen25vl_qlora_sft.yaml       # LLaMA-Factory training config
├── scripts/
│   ├── utils.py                       # Single source of truth: paths, labels, prompts, parsing, system detection, CONFIDENCE_THRESHOLD
│   ├── 01_convert_gaps_to_images.py   # .npy → PNG (parallel, mmap, auto-scales to CPU cores)
│   ├── 02_build_training_data.py      # Build ShareGPT JSONs for LLaMA-Factory
│   ├── 03_baseline_eval.py            # Evaluate base model (before fine-tune)
│   ├── 04_post_finetune_eval.py       # Evaluate fine-tuned model + comparison table
│   ├── 05_setup_training_config.py    # Fix YAML paths for current machine
│   ├── 06_incremental_retrain.py      # Retrain on expert corrections (will output versioned adapters)
│   ├── 07_cross_dataset_eval.py      # (NEEDS BUILDING) Evaluate model on Attain dataset for cross-dataset zero-shot generalization
│   └── validate_all.py               # 13-test validation suite
├── app/
│   ├── main.py                        # FastAPI: /classify, /classify/stream, /classify/base64, /health, /retrain/*, /corrections, /adapters/*
│   ├── model.py                       # PavementClassifier: predict_stage1(), predict_stage2(), predict(), confidence extraction
│   ├── schemas.py                     # Pydantic models for API request/response
│   └── security.py                    # API key auth, rate limiting, upload size, image validation
├── gradio_ui/
│   └── demo.py                        # Interactive testing UI
├── expert_ui/
│   └── index.html                     # Expert review dashboard (Supabase Auth, tabs, retrain trigger, adapter management)
├── test_website/
│   └── index.html                     # Public test website: upload image → SSE progress → AI results
├── adapters/                          # Versioned LoRA adapters (v1/, v2/, ..., active_version.txt)
├── eval_results/                      # baseline_results.json, finetuned_results.json, confusion matrices
├── db_schema.sql                      # Supabase PostgreSQL schema with RLS
├── requirements.txt                   # Python dependencies (includes sse-starlette)
├── SETUP.md                           # Comprehensive setup + operations + variable reference guide
└── venv/                              # Virtual environment (not portable, recreate per machine)
```

## Key Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | `Qwen/Qwen2.5-VL-7B-Instruct` | Model to load |
| `ADAPTER_PATH` | None (auto-detects from adapters/ or outputs/) | LoRA adapter directory |
| `QUANTIZATION_BITS` | `4` | 4-bit or 8-bit quantization |
| `API_KEYS` | Empty (auth disabled) | Comma-separated API keys |
| `CORS_ORIGINS` | `*` | Allowed CORS origins |
| `PROJECT_ROOT` | Auto-detected | Project root path |

## How to Run (Quick Reference)

```bash
# Data prep (once per machine):
python scripts/01_convert_gaps_to_images.py
python scripts/02_build_training_data.py
python scripts/05_setup_training_config.py

# Test interactively:
python gradio_ui/demo.py

# Evaluate baseline (ALL test images, no mercy):
python scripts/03_baseline_eval.py --stage 1
python scripts/03_baseline_eval.py --stage 2

# Fine-tune (on A5000):
llamafactory-cli train configs/qwen25vl_qlora_sft.yaml

# Evaluate after fine-tuning:
python scripts/04_post_finetune_eval.py --adapter-path outputs/qwen25vl-qlora-gaps-rdd/

# Start API server:
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1

# Expose to internet via Cloudflare:
cloudflared tunnel --url http://localhost:8000

# Open test website:
# Open test_website/index.html in any browser, paste the Cloudflare tunnel URL

# Validate everything:
python scripts/validate_all.py
```

## Training Config Highlights

- **QLoRA:** 4-bit quantization, LoRA rank 64, alpha 128, dropout 0.1
- **Anti-overfitting:** weight_decay 0.05, label_smoothing 0.1, NEFTune noise 5.0, max 2 epochs, early stopping on eval_loss
- **Acceleration:** Flash Attention 2, Liger kernel (Linux only — requires triton, no Windows wheels), gradient checkpointing, 8-bit AdamW optimizer
- **Effective batch size:** 4 * 8 = 32

## API Contract

```
POST /classify
  Headers: X-API-Key: <key> (optional if API_KEYS not set)
  Body: multipart/form-data, field "file" = image (max 10MB, max 4096x4096)
  Response: {is_distressed, stage1_label, stage1_confidence, distress_types, severity, description, stage2_confidence, needs_expert_review, processing_time_ms, stage1_time_ms, stage2_time_ms, stage1_raw, stage2_raw}

POST /classify/stream
  Headers: X-API-Key: <key> (optional)
  Body: multipart/form-data, field "file" = image
  Response: SSE stream (text/event-stream) with events: progress, result, error
  Client must use fetch() + ReadableStream (NOT EventSource, which only supports GET)

POST /classify/base64
  Body: {"image": "<base64-string>"}
  Response: same as /classify

GET /health → {status, model_loaded, model_name, device, adapter_loaded}

POST /retrain/start → {is_running, progress_pct, ...}
GET /retrain/status → {is_running, progress_pct, ...}

# Phase 3 endpoints (not yet built):
POST /corrections → accepts ExpertCorrectionRequest, adds to few-shot list + persists to JSON
GET /adapters → list all versioned adapters with metadata
POST /adapters/switch → {version: "v1"} → reloads adapter
GET /adapters/active → current active version info
```

## Design System (Shared Across All UIs)

Both `expert_ui/index.html` and `test_website/index.html` use the same design language:
- **Font:** SK Modernist (CDN-loaded woff2)
- **Colors:** `--bg: #f5f5f4`, `--surface: #ffffff`, `--text: #1c1917`, `--accent: #1c1917`, `--red: #dc2626`, `--amber: #d97706`, `--green: #16a34a`
- **Borders:** `--radius: 12px`, `--radius-lg: 16px`, `--border: #e7e5e4`
- **Shadows:** `--shadow-sm: 0 1px 2px rgba(0,0,0,0.04)`, `--shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04)`
- **Button styles:** `.btn` (outlined), `.btn-primary` (filled dark)
- **Badge styles:** `.status-badge.normal` (green), `.status-badge.distress` (red), `.status-badge.review` (amber)
- **Confidence visualization:** Horizontal bars with 80% threshold line marked in red
- **No frameworks**, no build tools, no gradients — vanilla HTML/CSS/JS, self-contained single files
- **Mobile-responsive:** single column on small screens

## Model Pipeline Technical Details

### predict_stage1(image) → dict
- Runs binary classification with STAGE1_SYSTEM_PROMPT/STAGE1_USER_PROMPT
- Extracts confidence via softmax over "Normal" vs "Distress" first-token logits
- Returns: `{is_distressed, stage1_label, stage1_confidence, needs_expert_review, stage1_time_ms, stage1_raw}`

### predict_stage2(image) → dict
- Runs type classification with STAGE2_SYSTEM_PROMPT/STAGE2_USER_PROMPT
- Extracts confidence via geometric mean of per-token log-probabilities
- Parses structured output (DISTRESS_TYPES/SEVERITY/DESCRIPTION fields)
- Returns: `{distress_types, severity, description, stage2_confidence, stage2_time_ms, stage2_raw}`

### predict(image) → dict
- Calls `predict_stage1()`, then `predict_stage2()` only if distressed
- Combines into full ClassificationResponse dict
- Flags `needs_expert_review` if either stage < 80% confidence

### Confidence Extraction (Real, NOT Heuristic)
- **Stage 1:** Softmax probability over "Normal" vs "Distress" first-sub-token IDs only (cached at model load)
- **Stage 2:** Geometric mean of per-token probabilities = `exp(mean(log(p_i)))` across all generated tokens
- No string matching, no inflation, no manipulation — pure model probability

## User Preferences

- Wants robust, error-free code — "keep it robust, do it gracefully, make no errors"
- Values planning before implementation
- The model must NOT guess when unsure — flag for expert review instead
- No faking, no overfitting, all true and accurate
- Comprehensive documentation matters
- Full SETUP.md guide exists with every variable, constant, CLI arg, and DB column documented
- Prefers clean, minimal UI — no gradients, no overwork, similar to expert_ui styling
- This is a college project for a paper — features should be paper-worthy but not over-engineered to enterprise level
- "If you suggest that these things make a huge impact that will open eyes and drop mouths, let's surely implement them"
- Doesn't want to play with code for every operation — needs UIs for adapter management, expert review, etc.
- Approach C (few-shot + LoRA retrain) was chosen because it gives two methodology sections in the paper and a comparison table showing immediate vs permanent improvement

## What's Done vs What Needs Doing

### Phase 1 — CODE COMPLETE + TESTED ON LAPTOP (3B), NEEDS FULL EVAL ON A5000 (7B)

**Code completed:**
- [x] `predict_stage1()` and `predict_stage2()` split in `app/model.py`
- [x] `POST /classify/stream` SSE endpoint in `app/main.py` with `sse-starlette`
- [x] `test_website/index.html` — standalone HTML, SSE progress, confidence bars, session history
- [x] `sse-starlette>=1.6.0` added to `requirements.txt`
- [x] Unparseable Stage 1 responses default to Distress (0% confidence) — errs on caution
- [x] Client disconnect check before Stage 2 in SSE — avoids wasting GPU on closed connections
- [x] `image.load()` after `Image.open()` in SSE endpoint — forces pixel decode, frees BytesIO buffer
- [x] SSE `\r\n` normalization in JS client — Windows servers send `\r\n`, browsers expect `\n`
- [x] Multi-line `data:` field concatenation in SSE parser — handles split data across lines

**Tested on laptop (RTX 4050, 6GB VRAM):**
- [x] Data prep: 80k GAPs PNGs + 32k RDD training JSONs generated
- [x] 7B model does NOT fit on 6GB VRAM — `OutOfMemoryError` even with 4-bit quantization
- [x] Switched to Qwen2.5-VL-3B-Instruct for laptop testing — works in 4-bit quantization
- [x] Baseline eval (3B, 50 samples): Stage 1 = 76% accuracy, Stage 2 = 14% accuracy (expected for base model, no fine-tuning)
- [x] API server runs on localhost:8000 with 3B model
- [x] Test website connects, uploads images, shows live SSE progress, displays results
- [x] SSE streaming verified end-to-end (was broken initially due to `\r\n` line endings, fixed)

**Needs executing on college A5000 (7B, fp16, no quantization):**
- [ ] Run full baseline eval — ALL 10k GAPs + ALL 5,758 RDD test images with 7B model
- [ ] Save results to `eval_results/baseline_results.json` + confusion matrix PNGs
- [ ] Start API server with 7B model + Cloudflare tunnel
- [ ] Test website remotely from phone/laptop via tunnel URL

**Dependency pinning (critical — discovered during laptop testing):**
- `transformers==4.57.6` + `bitsandbytes==0.44.1` + `accelerate==0.34.2` + `peft==0.14.0`
- Latest `transformers` (4.52+) has `Params4bit` compatibility bug with `bitsandbytes`
- These pinned versions are confirmed working on both laptop and should work on A5000

### College Machine Setup Commands

**First time (one-time setup):**
```bash
cd Capstone
python -m venv venv
source venv/Scripts/activate

# Install deps
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv pip install transformers==4.57.6 accelerate==0.34.2 bitsandbytes==0.44.1 peft==0.14.0
uv pip install qwen-vl-utils numpy Pillow tqdm psutil scikit-learn matplotlib seaborn
uv pip install fastapi "uvicorn[standard]" python-multipart slowapi python-dotenv sse-starlette

# Download cloudflared (one-time, no admin needed)
curl -Lo cloudflared.exe https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe

# Run full baseline eval (7B, fp16, downloads model ~15GB on first run)
QUANTIZATION_BITS=0 python scripts/03_baseline_eval.py --stage 1
QUANTIZATION_BITS=0 python scripts/03_baseline_eval.py --stage 2

# Start API server (7B, fp16)
QUANTIZATION_BITS=0 uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1

# Separate terminal — expose to internet
cd Capstone
./cloudflared.exe tunnel --url http://localhost:8000
```

**Every time after:**
```bash
cd Capstone
source venv/Scripts/activate
QUANTIZATION_BITS=0 uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1

# Separate terminal:
cd Capstone
./cloudflared.exe tunnel --url http://localhost:8000
```

**Remote testing:** Open `test_website/index.html` on any device (phone, laptop), paste the Cloudflare tunnel URL (e.g., `https://random-words.trycloudflare.com`), click Connect, upload road photo, get AI results live.

### Phase 2 — CODE PARTIALLY DONE, NEEDS EXECUTING
**Prerequisite: Phase 1 baseline eval must be complete (baseline_results.json exists in eval_results/)**
- [ ] Download Attain dataset to `Attain/` folder from Mendeley (doi:10.17632/nykrzdm74f/1)
- [ ] Inspect Attain folder structure + annotation format (`ls -R Attain/ | head -50`)
- [ ] Build `scripts/07_cross_dataset_eval.py` — Attain cross-dataset evaluation script (format depends on actual Attain structure)
- [ ] Run `07_cross_dataset_eval.py` on BASELINE model (no adapter) — establishes zero-shot baseline on Attain
- [ ] Add `_find_latest_adapter()` auto-detection to `app/model.py`
- [ ] Run `05_setup_training_config.py` to fix paths for college machine
- [ ] Run LLaMA-Factory training on RTX A5000 (24GB VRAM) — ~12-24 hours
- [ ] Run `04_post_finetune_eval.py --adapter-path outputs/qwen25vl-qlora-gaps-rdd/` — in-distribution eval
- [ ] Run `07_cross_dataset_eval.py --adapter-path outputs/qwen25vl-qlora-gaps-rdd/` — cross-dataset eval
- [ ] Generate three comparison tables for paper: (1) baseline vs fine-tuned on RDD, (2) baseline vs fine-tuned on Attain, (3) known vs unknown classes on Attain
- [ ] Save all results to eval_results/ + confusion matrix PNGs

### Phase 3 — CODE NOT YET WRITTEN
- [ ] `build_dynamic_stage2_prompt()` in `scripts/utils.py`
- [ ] Few-shot correction tracking in `app/model.py`
- [ ] `POST /corrections` endpoint in `app/main.py`
- [ ] Versioned adapter output in `scripts/06_incremental_retrain.py`
- [ ] Adapter management endpoints (`GET /adapters`, `POST /adapters/switch`, `GET /adapters/active`)
- [ ] `reload_adapter()` method in `app/model.py`
- [ ] Expert UI enhancements (adapter dropdown, version display)
- [ ] End-to-end test: correct → few-shot → retrain → version → switch → verify

## Model Size vs VRAM Reference

| Model | fp16 (no quant) | 4-bit quantized | Fits A5000 (24GB)? | Fits RTX 4050 (6GB)? |
|-------|-----------------|-----------------|--------------------|-----------------------|
| Qwen2.5-VL-3B | ~6GB | ~2GB | Yes | Yes (4-bit only) |
| Qwen2.5-VL-7B | ~14GB | ~4-5GB | Yes (fp16) | No (OOM even 4-bit) |
| Qwen2.5-VL-13B | ~26GB | ~7-8GB | No | No |
| Qwen2.5-VL-72B | ~144GB | ~40GB | No | No |

**Decision:** Use 7B fp16 (no quantization) on A5000 for best accuracy. Use 3B 4-bit on laptop for dev testing only.

## Future Plans (Beyond Phase 3)

- Geospatial heatmap dashboard (Leaflet.js) showing distress hotspots in Bengaluru
- Repair cost estimation per distress type + severity (using BBMP/IRC standard rates)
- Field validation on real Bengaluru roads (50-100 photos, compare model vs manual inspection)
- Temporal deterioration tracking (same road over time)
- Benchmarking against YOLOv8 for comparison

## Bugs Found and Fixed (Session Log)

1. **7B model OOM on RTX 4050** — even 4-bit quantized, Qwen2.5-VL-7B needs more than 6GB VRAM. Solution: use 3B for laptop dev, 7B on A5000.
2. **`Params4bit` compatibility bug** — `transformers>=4.52` breaks with `bitsandbytes`. Solution: pin `transformers==4.57.6`, `bitsandbytes==0.44.1`.
3. **CPU offload not supported with bitsandbytes** — `device_map="auto"` with CPU offload fails on quantized models. Solution: don't use CPU offload, just use smaller model.
4. **SSE `\r\n` line endings** — Windows servers send `\r\n` but JS SSE parser splits on `\n\n`. Solution: normalize `\r\n` to `\n` in JS before parsing.
5. **Multi-line SSE `data:` fields** — some SSE events have data split across multiple `data:` lines. Solution: concatenate all `data:` lines before JSON parsing.
6. **`capture="environment"` blocks gallery** — on mobile, `capture="environment"` forces rear camera and prevents gallery selection. Solution: use `accept="image/*"` alone.
7. **`history` variable name collision** — JS `history` conflicts with `window.history`. Solution: renamed to `sessionHistory`.
