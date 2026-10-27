# Pavement Distress Classification — Capstone Project Context

## START HERE — Current State (updated 2026-10-27)

**Pipeline version in production** (code: `main`, last pipeline-code commit `a59eec1`, 2026-09-26):

```
RoadSide app → Cloudinary image + Supabase `photos` row → trigger → `assessments` (pending)
  → PipelineWorker (app/worker.py, inside uvicorn, FIFO claim, batch 5)
     Stage 0  pavement pre-filter          only "no" rejects → rejected_non_pavement
     Stage 1  Normal / Distress            softmax over the two first tokens
              + Stage 1 safety net         confident "Normal" but probe sees distress → expert_review
     Stage 2  free-form DISTRESS_TYPES     first label shown as main, others beside it
              + per-type probe (SHADOW)    20 Yes/No P(yes) + tile scores stored, labels unchanged
     Gate     field confidence ≥ 0.80 on both stages → classified, else expert_review
```

| Setting (`.env`) | Value |
|---|---|
| Model | `Qwen/Qwen2.5-VL-7B-Instruct` @ `cc594898…` base, `DISABLE_ADAPTER=true` |
| `QUANTIZATION_BITS` | `4` (NF4, bf16 compute) — measured at this setting only |
| `MAX_IMAGE_PIXELS` | `1048576` (1024²) |
| `PROMPTS_VERSION` | `v2` (Stage 2 prompt id `9da1601cfec54f76`) |
| `STAGE2_CONFIDENCE_MODE` | `field` |
| `STAGE2_MODE` / `STAGE2_PROBE_CONFIG` | `shadow` / `configs/stage2_probe.json` (views: decide on full photo, record tiles) |
| `STAGE1_SAFETY_NET` | `true` |

**The production version that must be up is P4** (the paper's label; P1–P3 are historical, see
"Project History" below). A correctly started server reports, on `GET /health`:
`model_name=Qwen/Qwen2.5-VL-7B-Instruct`, `adapter_loaded=false`, `quantization_bits=4`,
`max_image_pixels=1048576`, `prompts_version=v2`, `taxonomy=IRC:82-2015`, `stage2_confidence_mode=field`,
`stage2_mode=shadow`, `stage2_mode_effective=shadow`, `stage1_safety_net=true`,
`stage2_probe.parity.ok=true`. If `stage2_mode_effective` is `generate`, the probe failed to load —
`stage2_probe.reason` says why; the pipeline still works but stops collecting calibration data.

**Is it running?** No. The server last ran on 2026-10-22 (log `eval_results/production_final.log`);
the machine was restarted since. No upload has arrived since 2026-09-25. To start:
`.\start.ps1 -AutoStart -Watchdog` (logs to console; redirect with `*> eval_results\production_final.log`),
then check `GET /health` and `GET /operator/status`. Stop: `.\stop.ps1`. See "Restarting production"
below before restarting a running server.

**Production data (2026-10-27):** 361 uploads (`photos` = `assessments`, 2026-04-27 → 2026-09-25):
131 classified, 213 expert_review, 17 rejected_non_pavement. All re-classified on 2026-09-26 under the
version above. 310 rows carry probe scores. Only **1** row is expert-reviewed — this is the blocker
for everything below.

**Key metrics at a glance** (every number traceable to `eval_results/` and the paper section shown):

| What | Result | Data | Paper |
|---|---|---|---|
| Stage 1 baseline (P1, v1 prompts) | 76.53% accuracy | GAPs test, 10k | §6.1 |
| QLoRA adapter on RDD (P2) | S1 76.53% → 88.39%; D20 recall 2.57% → 35.22% (in-distribution only) | RDD test, 5,758 | §6.2 |
| Adapter vs prompts on Attain | trained classes 26.9% (B) vs 31.1% (C); zero-shot 3.8% vs 0% (corrected) | Attain, 769 | §6.3 |
| Stage 2 free-form list (P4) vs per-type probe | macro MCC 0.111 → 0.223 (test); 0.125 → 0.241 (CV, Δ +0.116 [+0.064, +0.157]) | Attain, 847 | §6.8 |
| Alligator cracking named by the list | on 8% of frames that have it; probe AUROC 0.70 test / 0.76 CV | Attain | §6.8 |
| Production Stage 1 recall on distress frames | 51.8% (probe max-score: 93.3%) | Attain | §6.8 |
| Field-confidence gate AUC | 0.744 (no-FP rule); sequence 0.373 (inverted) | Attain, 407 | §6.5 |
| Probe confidence AUC on its own errors | 0.78 exact set, 0.87 no-FP (OOF) | Attain | §6.8 |
| Tiles vs full photo (probe) | test AUROC +0.020 (CI incl. 0); CV MCC 0.248 → 0.308 | Attain | §6.9 |
| Worker time per upload | ~11 s at 1024² (272 s uncapped), + ~1.5 s shadow probe, + ~6.5 s recorded tiles | A5000 | §6.6, T7 |
| Production review load | 213 of 361 uploads in expert_review | Bengaluru | T7.2 |

**What is decided (with evidence — don't re-litigate without new data):**
| Decision | Evidence | Paper |
|---|---|---|
| Base model, no LoRA adapter | adapter lost on Attain (zero on pothole/block/ravelling) | §6.3, `attain_3way_comparison.md` |
| 1024² cap | 272 s → 11 s per photo, no accuracy change down to 512² | §6.6, Fig 6.6 |
| Field confidence gate at 0.80 | AUC 0.744 (no-FP rule); sequence inverted 0.373 | §6.5, Fig 6.5 |
| NOT gating on first label | AUC 0.468; no review-load gain on 68 uploads | §6.7, Fig 6.7 |
| NOT "most prominent first" prompt | first label unchanged, false positives up (p = 0.021) | §6.7 |
| Probe in shadow, not live | doubles Attain macro MCC, but Attain thresholds add Ravelling to 98% of Bengaluru uploads | §6.8 |
| Full photo, tiles recorded only | tiles +0.020 AUROC (CI incl. 0); find-then-zoom worse | §6.9, Fig 6.9 |

**Next steps, in order:**
1. Get expert labels: experts review uploads in `expert_ui/` (only 1 so far).
2. `scripts/calibrate_probe_from_expert_labels.py` (exits 2 until ≥15 positives per type) → promote its
   config to `configs/stage2_probe.json` and set `STAGE2_MODE=probe` only if it beats the free-form list
   on held-out expert labels.
3. Re-run `scripts/calibrate_confidence.py` on labelled Bengaluru photos before moving the 0.80 threshold.
4. Paper: §3.3–3.4, §6.1–6.3, §7.3–7.4 still describe the pre-IRC, adapter-era pipeline — revise before submission.
5. Open requests: external VLM hand-off (`vlm_handoff/`, 11 `[CONFIRM]` items for the user; folder-batch
   runner + Stage-1 bypass promised but unbuilt); drive-video ingestion (researched, not built).

**Where things are:**
- Research paper: `paper/baseline_evaluation.md` (§T7 = deployment record, test suites, incidents).
  Figures: `scripts/generate_paper_figures.py` → `eval_results/plots/paper/` (no inference needed).
- Results: `eval_results/` — `*_report.md` / `*.json` per study; `*.log` are run logs (git-ignored),
  `*_checkpoint.json` resumable state (git-ignored), `snapshots/assessments_*` / `pre_reclassify_*`
  are DB dumps with citizen addresses (git-ignored, keep local).
- Tests: `scripts/validate_all.py` (18 checks) + `scripts/tests/*` — all passing 2026-10-27; table in paper §T7.4.
- Architecture reference: `PROJECT_UNDERSTANDING.md`. Setup / every variable: `SETUP.md`.
- Commit rule: **no `Co-Authored-By` lines** in commits (user preference).
- Secrets live only in `.env` (git-ignored). Never print them. The Cloudinary API secret was shown once in
  a 2026-09-23 session transcript — rotation was recommended.

## What This Project Is

A two-stage AI pipeline that classifies pavement distress from phone photos, built by students in Bengaluru. Users snap a photo of a road, the AI detects if there's damage, classifies the type, estimates severity, and flags low-confidence results for expert human review. The long-term goal is a geospatial heatmap of road distress across Bengaluru. This is a college-level capstone project that must produce a publishable paper.

## Architecture

Production (P4) — built and deployed:
```
RoadSide app → Cloudinary + Supabase `photos` → trigger → `assessments` (pending)
   → PipelineWorker → Stage 0 pre-filter → Stage 1 (+ safety net) → Stage 2 (+ shadow probe)
   → field gate 0.80 → classified | expert_review → Supabase → operator dashboard / WebGIS
Same model also serves POST /classify, /classify/stream (SSE), /classify/base64.
```
Phase 3 (designed, NOT built — blocked on expert labels):
```
expert_review rows → Expert UI corrections → calibrate_probe_from_expert_labels.py (built)
   → STAGE2_MODE=probe if it wins → later: few-shot injection + incremental LoRA (not built)
```

- **Production model:** Qwen2.5-VL-7B-Instruct **BASE** (no adapter) with v2 Improved Baseline prompts. The RDD-fine-tuned LoRA adapter is built and verified but NOT used in production — empirically beaten by the Improved Baseline on multi-class detection (52.5% vs 16.9%), zero-shot classes (Block crack F1 0.30 vs 0.00), and cross-geography Pothole. See `eval_results/attain_3way_comparison.md`.
- **Taxonomy:** IRC:82-2015 compliant (Indian Roads Congress Code of Practice). 18 photo-visible distress types across 4 IRC categories (Surface Defects, Cracks, Deformation, Disintegration). Severity follows IRC quantitative criteria (mm thresholds). See `scripts/irc82_taxonomy.py` (single source of truth) and `eval_results/irc82_verification_report.md` (5-round verification, 100% IRC compliance on 100 real images).
- **Stage 1:** Binary detection (Normal/Distress) — pure prompt-based, deep persona + 5-step inspection protocol
- **Stage 2:** IRC:82 distress type + severity — pure prompt-based, deep persona + 6-step IRC protocol + full IRC taxonomy + quantitative severity criteria
- **Stage 2 per-type probe (shadow, since 2026-09-24):** one Yes/No question per IRC type, P(yes) from logits, ~1.4 s for all 20; stored on every row, labels unchanged until Bengaluru thresholds exist. Also powers the **Stage 1 safety net** (confident "Normal" + probe sees distress ⇒ expert review). See "Stage 2 per-type probe" below.
- **Condition indicator:** `Patching` (IRC:82 Tables 5.1–5.3) lives in `IRC82_CONDITION_INDICATORS`, never in `distress_types`.
- **API:** FastAPI with rate limiting, API key auth, CORS, async inference, SSE streaming
- **DB:** Supabase (PostgreSQL) — `photos` (written by the RoadSide app), `assessments` (pipeline; trigger-linked), `worker_state`, `custom_distress_types`, `retrain_jobs`; migrations 001–004 in `migrations/`
- **Expert UI:** Single-page HTML/JS (`expert_ui/index.html`), anon-key + RLS scoped to review rows (migration 003), reviewer name instead of login; pending/reviewed tabs
- **Operator UIs:** `/operator` (worker start/stop, live metrics, model precision) and `/dashboard` (image browser, delete-everywhere, re-classify)
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
- Location: `Attain/Attain/Attain/Attain_SMP_{OS_V1.0,WS_V1.0,WS_V2.0}/` — every evaluation uses **WS_V2.0** (847 images, Pascal-VOC XML, frames of 1479×508, 1920×1080 and 640×640 interleaved)
- Source: Mendeley Data (doi:10.17632/nykrzdm74f/1), CC BY 4.0
- Images: 2,293 images across the three subsets, 19,761 annotated distress instances
- Classes (WS_V2.0): Alligator crack, Block crack, Linear crack (longitudinal and transverse are NOT split), Pothole, Patch and utility cut, Raveling, Weathering, plus Faded marking and Lane shoulder drop-off (excluded). Severity High / Low only.
- **Class-name trap:** Attain spells one class `Patch and utility cut- Low` (no space before the dash). Before 2026-09-23 the parser split only on `" - "`, so every patch instance became an unmapped pseudo-class and silently left the zero-shot tally: the paper's Tier-2 figures were overstated (Improved Baseline 5.88% → 3.81%, plain 3.27% → 2.11%; ranking unchanged). Fixed in `parse_attain_class()`.
- Collection: smartphones mounted on a vehicle's windshields. **Created by Amirkabir University of Technology, Tehran** (Mendeley record; the data YAML links attain.aut.ac.ir). The record does not name the collection site; the roads look Iranian. Earlier docs said "New Zealand" — that was never supported, do not repeat it.
- 78 of 847 WS_V2.0 images carry no distress annotation (only faded marking / drop-off, or nothing). Earlier evaluations dropped them; the probe study (`stage2_probe_experiment.py`) keeps them as the only all-class negatives.
- **Purpose:** Zero-shot generalization testing. Model is trained on RDD (4 classes) and evaluated on Attain (10 classes). Classes not in RDD (block crack, raveling, weathering, faded marking, drop-off, manhole) are zero-shot evaluation targets — the model has never trained on them but can recognize them via taxonomy injection in the system prompt.
- **Phase 3 role:** Misclassifications on Attain's unseen classes become expert correction candidates → few-shot injection → LoRA retrain → measure improvement trajectory
- Scripts: `scripts/07_cross_dataset_eval.py` (built; Attain loading/scoring), `scripts/stage2_probe_experiment.py` and `stage2_views_experiment.py` (all 847 frames, incl. negatives)
- **Attain class → our taxonomy mapping:**
  ```
  Attain class              → ALL_DISTRESS_TYPES match         → In RDD training?
  ──────────────────────────────────────────────────────────────────────────────
  Alligator crack           → Alligator Crack (D20)            → YES (trained)
  Longitudinal crack        → Longitudinal Crack (D00)         → YES (trained)
  Transverse crack          → Transverse Crack (D10)           → YES (trained)
  Pothole                   → Pothole (D40)                    → YES (trained)
  Block crack               → Transverse Cracking (IRC §7.3.5.1, "large blocks") → NO (zero-shot)
  Patch/utility cut         → Patching (IRC condition indicator, Tables 5.1-5.3) → NO (zero-shot)
  Weathering                → Hungry Surface (IRC §7.2.4)      → NO (zero-shot)
  Raveling                  → Ravelling (IRC §7.5.2)           → NO (zero-shot)
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

## Project History — Pipeline Versions and Phases

| Version | When | Prompts / taxonomy | Adapter | Status | Evidence |
|---|---|---|---|---|---|
| P1 | 2026-03 | v1 plain, RDD codes D00–D40 | none | historical baseline | `eval_results/baseline_results.json`, paper §6.1 |
| P2 | 2026-05-07 | v1 plain, RDD codes | QLoRA `adapters/v2-rdd-2epochs-20260507` | retired 2026-05-16 (lost on Attain) | `finetuned_results.json`, `ab_comparison.json`, §6.2 |
| P3 | 2026-05-16 | v2 Improved Baseline, pre-IRC names | none | superseded same day | `attain_3way_comparison.md`, §6.3 |
| **P4** | **2026-05-16 → now** | **v2, IRC:82-2015 (18 types)** | **none** | **production**; since 2026-09: field gate, 1024² cap, shadow probe + tiles, Stage 1 safety net | §6.5–6.9, T7 |

**Phase 1 — done (2026-03-29).** Baseline on all 10k GAPs + 5,758 RDD test images; API with SSE
streaming; test website. Design rules that came out of it are in "Critical Technical Decisions".

**SSE Event Format:**
```
event: progress → {stage: "stage1", status: "running", message: "Detecting if pavement shows distress..."}
event: progress → {stage: "stage1", status: "complete", result: {is_distressed, stage1_label, stage1_confidence, ...}}
event: progress → {stage: "stage2", status: "running", message: "Classifying distress type and severity..."}
event: progress → {stage: "stage2", status: "complete", result: {distress_types, severity, stage2_confidence, ...}}
event: result   → {full ClassificationResponse JSON}
event: error    → {error: "message"} (if something fails)
```

**Phase 2 — done (2026-05).** QLoRA fine-tune (43 h, early-stopped; checkpoint-1000), post-fine-tune
eval, A/B promotion gate passed, Attain cross-dataset study (prompts vs adapter). The adapter won
in-distribution and lost cross-dataset, so production went to P4. Training workflow and its 7 OOM
fixes: paper §3.6, §7.5; `scripts/train_qlora.py`, `configs/qwen25vl_qlora_sft.yaml`.

**Since 2026-09 (all on P4):** confidence calibration (§6.5), resolution cap (§6.6), first-label gate
test (§6.7), per-type probe + safety net (§6.8), views study (§6.9), Supabase load fix (T7.3),
re-classification of all 341 rows under P4 (2026-09-26, labels reproduced exactly).

**Phase 3 — not started; blocked on expert labels.** Design (unchanged): corrections from the
expert UI → (1) immediate few-shot injection into the Stage 2 prompt (≤5 most recent), (2) periodic
incremental LoRA → versioned `adapters/vN/` + `active_version.txt` → hot reload. What exists today:
the expert UI, stored per-type probe scores on every row, and `calibrate_probe_from_expert_labels.py`.
Not built: `build_dynamic_stage2_prompt()`, `POST /corrections`, versioned retrain output, adapter
switch endpoints. Do the probe calibration first (START HERE → Next steps); a new fine-tune should use
IRC:82 labels from Bengaluru uploads, not RDD's four classes.

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
│   ├── 07_cross_dataset_eval.py      # Attain cross-dataset eval (--prompts-version, --model, crash-resumable; checkpoints are model-scoped)
│   ├── irc82_taxonomy.py             # IRC:82-2015 taxonomy — 18 vision-only types, single source of truth
│   ├── utils_v2_prompts.py           # v2 "Improved Baseline" prompts (production)
│   ├── model_loader.py               # Family-agnostic VLM loader + VRAM/OOM/offload guardrails + load-time self-test
│   ├── smoke_test_model.py           # Pre-flight gate for a model swap — must pass before trusting a new checkpoint
│   ├── compare_model_ab.py           # Model A/B comparison (refuses to compare runs that differ in more than the model)
│   ├── stage2_probe.py               # Per-type Yes/No probe: P(yes) per IRC type from logits, shared cached image prefix, parity self-test
│   ├── stage2_probe_rules.py         # combine() / safety_net(): the ONE implementation production and evaluation both call
│   ├── stage2_probe_experiment.py    # GPU sandbox: production Stage 1+2 vs probe variants on all 847 Attain frames (checkpointed)
│   ├── stage2_probe_report.py        # Pre-registered dev/test scoring (MCC, block bootstrap) -> eval_results/stage2_probe_report.*
│   ├── stage2_probe_cv.py            # Grouped 5-fold CV + eligibility + production config -> eval_results/stage2_probe_config_final.json
│   ├── probe_production_compare.py   # Read-only: probe vs stored production output on real Bengaluru uploads
│   ├── calibrate_probe_from_expert_labels.py  # Fits Bengaluru thresholds from expert-reviewed rows (candidate only; exits 2 if too few labels)
│   ├── multilabel_metrics.py         # MCC/AUROC/AP/cluster bootstrap/McNemar/Platt — checked against scikit-learn
│   ├── stage2_views.py               # Tiles / find-then-zoom crops for the probe, aggregate(), record-only views, DamageLocator
│   ├── stage2_views_experiment.py    # Sandbox: full vs tiles vs zoom vs oracle per-view scores (Attain dev/test, Bengaluru)
│   ├── stage2_views_report.py        # dev choice / test+CV comparison / --stage emit config for a views variant
│   ├── generate_paper_figures.py     # Paper Figures 6.5-6.9 from stored results (no inference)
│   ├── primary_confidence_report.py  # §6.7 first-label confidence evaluation
│   ├── production_gate_comparison.py # §6.7 gate comparison on production rows (read-only)
│   ├── cleanup_orphan_photos.py      # photos rows without an assessment (dry run by default)
│   ├── reset_for_reclassify.py       # re-queue rows (backs up first; --processed-before)
│   ├── tests/                        # see paper §T7.4
│   └── validate_all.py               # 18-check static validation suite
├── app/
│   ├── main.py                        # FastAPI: /classify*, /health, /operator/*, /dashboard/*, /retrain/*
│   ├── worker.py                      # PipelineWorker: claims pending rows, runs stages 0-2, writes results
│   ├── supabase_client.py             # PostgREST calls + _ReadCache for dashboard reads
│   ├── deletion.py                    # Cloudinary-first delete of image + photo + assessment
│   ├── model.py                       # PavementClassifier: predict_stage1(), predict_stage2(), predict(), confidence extraction
│   ├── schemas.py                     # Pydantic models for API request/response
│   └── security.py                    # API key auth, rate limiting, upload size, image validation
├── gradio_ui/
│   └── demo.py                        # Interactive testing UI
├── expert_ui/
│   └── index.html                     # Expert review dashboard (anon RLS + reviewer name, pending/reviewed tabs)
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
| `MODEL_PATH` | `Qwen/Qwen2.5-VL-7B-Instruct` | Model to load. Family-agnostic since `scripts/model_loader.py` — `Qwen/Qwen3-VL-8B-Instruct` works with no code change. |
| `DISABLE_ADAPTER` | `true` in production .env | Skip LoRA loading entirely. Production runs pure base model. |
| `PROMPTS_VERSION` | `v2` (default in app/model.py) | `v2` = Improved Baseline (production). `v2_primary_first` = v2 plus a "most prominent distress first" rule: tested 2026-09-23 and NOT adopted (first label unchanged, false positives up 85.3% → 83.3% correct, p = 0.021; `eval_results/primary_confidence.md`). `v1` = Plain Baseline (paper experiments only). |
| `ADAPTER_PATH` | Unset in production | LoRA adapter directory. Used only when `DISABLE_ADAPTER=false` for A/B experiments. |
| `QUANTIZATION_BITS` | `4` | Weight storage format at startup: `4` = NF4 + double quant (~4.3 GB), `8` = int8 (~8.6 GB), `0` = bf16, no quantization (~16.6 GB). Compute is bf16 in all three — only storage changes. Changeable at runtime from the operator dashboard (Model Precision panel) without a restart; this env var only sets the boot default. Production `.env` sets `4` explicitly (since 2026-09-23): the 1024² cap was measured at 4-bit, and bf16 adds ~12 GB that very likely pushes the peak past 24 GB. |
| `STAGE2_CONFIDENCE_MODE` | `field` (since 2026-09-17) | `field` = geomean over the DISTRESS_TYPES tokens only. `sequence` = geomean over all generated tokens, the pre-2026-09-17 default. On 407 labelled Attain images, under the no-false-positives rule: field AUC **0.744**, sequence **0.373** (inverted). Under Jaccard ≥ 0.5 field is 0.473 — it detects over-prediction, not correctness in general. Always quote the rule with the AUC. `primary` = joint probability of the FIRST label only (2026-09-23): **tested, not adopted** — AUC 0.468 for first-label correctness vs field 0.635 on the same 407 images, and 91% review load at 0.80 (`eval_results/primary_confidence.md`, paper §6.7). All three values are recorded on every row (`raw_response.stage2_confidence_primary`, `stage2_type_confidences`). |
| `STAGE2_MODE` | `shadow` in production `.env` (code default `generate`) | `generate` = free-form DISTRESS_TYPES list only. `shadow` = the per-type probe runs on every image and every type's P(yes) is stored in `raw_response.stage2_probe` (`applied: false`); labels and gate unchanged. `probe` = the probe decides calibrated types. **Do not switch to `probe` on the Attain config**: it doubled macro MCC on Attain (0.125→0.241, grouped CV) but adds Ravelling to 98% of Bengaluru uploads — thresholds don't transfer. Switch only after `calibrate_probe_from_expert_labels.py` produces a Bengaluru config that wins on held-out expert labels. |
| `STAGE1_SAFETY_NET` | `true` in production `.env` | When Stage 1 says Normal, the probe checks headline types; if one reaches the configured threshold the image goes to expert review. Never changes a label. Stage 1 recall on Attain distress is only 51.8%; the net catches 313/371 misses (22/78 clean frames flagged). Recorded in `raw_response.stage1_safety_net`. |
| `STAGE2_PROBE_CONFIG` | `configs/stage2_probe.json` | Probe thresholds/Platt/eligibility (schema 1, validated at load). Copied from `eval_results/stage2_probe_config_final.json`. Missing/invalid config or failed parity self-test ⇒ probe disabled, reason on `/health`, pipeline unchanged. |
| `PROBE_BATCH_BUDGET_GB` | `1.5` | GPU memory the probe's suffix batch may use for copies of the prefix KV cache. Unbounded batches paged to RAM on the shared 24 GB card (1.1 s → 23–111 s/image). |
| `MAX_IMAGE_PIXELS` | `1048576` (1024²) in production `.env`; code default 4,840,000 | Input resolution cap. Full-res 1736² photos need a 47.8 GB peak on the 24 GB card and page to system RAM (~272 s/image); at 1024² the peak is 19.8 GB and the whole worker takes ~11 s. No measured accuracy loss on 200 labelled Attain images down to 512². Measured at 4-bit only. See paper §6.6. |
| `STAGE1_MAX_NEW_TOKENS` | `12` | Stage 1 answers with one word. Was 200, which cost ~4x the time for identical output. |
| `SKIP_MODEL_SELFTEST` | Unset (self-test ON) | Skips the 2s load-time synthetic-image check. Leave ON in production. |
| `API_KEYS` | Empty (auth disabled) | Comma-separated API keys |
| `CORS_ORIGINS` | `*` | Allowed CORS origins |
| `PROJECT_ROOT` | Auto-detected | Project root path |

### Model loading (scripts/model_loader.py — single source of truth)

Production (`app/model.py`) and evaluation (`scripts/03_baseline_eval.py`, which
`07_cross_dataset_eval.py` imports) load models through the SAME function. If they diverged,
an A/B comparison between two models could be measuring a loader difference rather than a
model difference. Guardrails, all of which fire automatically:

1. **Family resolution** from `config.model_type` via `AutoModelForImageTextToText`, falling
   back to the explicit Qwen2.5-VL class. Untested families load with a loud warning.
2. **VRAM pre-flight** — parameter count pulled from HF Hub metadata (no download), compared
   against free VRAM; auto-reduces `max_pixels` when tight.
3. **OOM fallback ladder** — bf16 → 4-bit → clear error, instead of dying mid-batch.
4. **Processor kwarg compatibility** — `min_pixels`/`max_pixels` are Qwen-specific; falls back
   to a `size` dict, then to defaults.
5. **Load-time self-test** — a synthetic image through the full template → processor → generate
   path. Catches a broken model/template pairing at startup (2s) instead of writing `failed`
   rows for real citizen photos.
6. **CPU/disk offload detection** — `device_map="auto"` silently offloads when it thinks VRAM
   is short, which is invisible in accuracy metrics and 10-30x slower. Now reported and warned.
7. **Allocator cleanup** — see the performance note below.

### Measured performance findings (RTX A5000 24GB, 2026-09-15)

| Finding | Evidence |
|---|---|
| Sharded loading leaves ~16 GB of cached allocator blocks; on Windows WDDM the driver spills the excess into system RAM instead of OOMing, silently. `empty_cache()` after load: 16.65 GB tensors / **32.35 GB reserved** / 0.00 GB free → 16.84 GB reserved / 7.54 GB free. | Stage 1 inference **23.5s → 4.7s** on a fixed 512x512 fixture, measured 5x in one process |
| Stage 1 used `max_new_tokens=200` to produce a 2-token answer | **4.73s → 1.20s** for identical output |
| **The v2 IRC prompts cost ~8x throughput** vs the pre-IRC prompts. This predates any 2026-09 change — it is the price of the full 18-type taxonomy + severity criteria + 6-step protocol + few-shot block in every prompt. | `attain_baseline_v2_run.log` (pre-IRC): **4.54 s/image** over 769 images. `irc_audit_100.log` (IRC prompts, same machine/model/quantization): **37.94 s/image** over 100 images. |
| flash-attn is NOT installed and has no prebuilt wheel for Windows + cp312 + torch 2.5.1. SDPA is the fallback and is correct, just slower. | `pip download flash-attn --only-binary=:all:` → "no matching distribution" |

Throughput planning (re-measured 2026-09-23, whole worker, 4-bit, cache cleared between
passes): **~11 s per phone photo at the production 1024² cap**. Mixing image SHAPES also fills
the allocator: on Attain, Stage 2 took 8 / 24 / 84 s on 1479×508 / 640×640 / 1920×1080 frames
without clearing the cache between images, and ~8 s for all three with it. The worker and
`calibrate_confidence.py` now release the cache after every image. Uncapped 1736² photos took
~272 s because they overflow the 24 GB card. Earlier figures in this table (including the
"8x IRC prompt cost") were measured without clearing the allocator between images and may be
inflated by the same overflow; re-measure before relying on them.

### Stage 2 per-type probe (2026-09-24) — read before touching Stage 2

- **Finding that motivated it:** on 847 Attain frames the free-form list was exactly `Longitudinal Cracking, Transverse Cracking` on 61% of test frames; alligator recall 8%; ravelling/weathering/patch never. Its 0.95 linear-crack F1 equals a constant predictor's. Report **MCC** (0 for constant predictors), always beside a constant baseline, with block (20-frame) bootstrap CIs — see `eval_results/stage2_probe_report.md`, `stage2_probe_cv.md`, paper §5.3/§6.8/§7.10.
- **Protocol records:** `eval_results/stage2_probe_preregistration.json` (code hashes + adoption rules, addendum for CV). Don't re-tune on the test split.
- **Production state:** shadow + safety net + field gate (unchanged review load). Probe adds ~1.4–2 s per Stage-2 image.
- **Next step (Phase 3 link):** experts review uploads in `expert_ui/` → run `scripts/calibrate_probe_from_expert_labels.py` → only if its held-out per-type MCC beats the free-form list, copy its candidate to `configs/stage2_probe.json` and set `STAGE2_MODE=probe`.
- **Views (2026-09-26, paper §6.9):** `scripts/stage2_views.py` lets the probe score tiles or model-located zoom crops. Test/CV vs full photo: tiles (upscaled, max) AUROC +0.020 [−0.016,+0.051], CV macro MCC 0.248→0.308 (p=0.047), ~6.5 s/upload; find-then-zoom −0.009 AUROC (its boxes miss ~56% of damage, none usable on 32% of uploads); oracle crops only +0.008. Not adopted under the pre-registered rule. `configs/stage2_probe.json` has `views: {mode: full, record: tile}` — decisions on the full photo, per-tile scores stored in `raw_response.stage2_probe.views.recorded` (+~6.5 s per image; delete `record` to save it). `calibrate_probe_from_expert_labels.py` compares full vs tiled on expert labels and picks the winner (verified end-to-end by `scripts/tests/test_calibrate_from_expert_labels.py`). Aggregation must not be naive max over many views for common texture classes — it inflates false positives (see `stage2_views.aggregate`).
- **Restarting production:** it runs under `start.ps1 -Watchdog` (launched from an earlier background session; log = `eval_results/production_final.log`). Killing uvicorn makes the watchdog relaunch it with the current `.env` — do NOT start a second server (the watchdog's `Stop-StalePython` kills it). After any restart, `POST /operator/start`: auto-start fires only once per `start.ps1` run.

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

# Start API server (production — Improved Baseline pipeline):
# .env must contain DISABLE_ADAPTER=true and PROMPTS_VERSION=v2
uvicorn app.main:app --env-file .env --host 0.0.0.0 --port 8000 --workers 1

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
  Response: {is_distressed, stage1_label, stage1_confidence, distress_types, primary_distress_type,
            secondary_distress_types, severity, description, stage2_confidence,
            stage2_confidence_primary, needs_expert_review, processing_time_ms, stage1_time_ms,
            stage2_time_ms, stage1_raw, stage2_raw}
  distress_types keeps the model's order: [0] is shown as the main label, the rest beside
  it. [0] is the model's listing order, not a measured "most prominent" type.

POST /classify/stream
  Headers: X-API-Key: <key> (optional)
  Body: multipart/form-data, field "file" = image
  Response: SSE stream (text/event-stream) with events: progress, result, error
  Client must use fetch() + ReadableStream (NOT EventSource, which only supports GET)

POST /classify/base64
  Body: {"image": "<base64-string>"}
  Response: same as /classify

GET /health → {status, model_loaded, model_name, device, adapter_loaded,
               adapter_disabled_in_config, prompts_version, taxonomy,
               pavement_filter_enabled, model_family, model_class,
               quantization_bits, oom_fallback_used, max_image_pixels,
               stage2_confidence_mode, stage2_mode, stage2_mode_effective,
               stage1_safety_net, stage2_probe {enabled, reason, parity, variant,
               views, provenance}, vram_used_gb}

/classify and /classify/stream results also carry: stage2_mode, stage2_confidence_probe,
condition_indicators, stage1_safety_net (when Stage 1 said Normal). Stage 2 SSE progress
events carry the full Stage 2 dict incl. stage2_probe (P(yes) per type, views, recorded tiles).

# Runtime precision control (operator dashboard "Model Precision" panel)
GET /operator/runtime → {model_path, adapter_path, quantization_bits,
                         quantization_label, quantization_effective,
                         oom_fallback_used, model_family, model_class,
                         max_image_pixels, stage2_confidence_mode,
                         prompts_version, device, is_loaded,
                         vram: {total_gb, free_gb, used_gb, gpu_name},
                         options: [{bits, label, weights_gb, note}],
                         worker_running, reload_in_progress}

POST /operator/runtime/quantization
  Body: {"quantization_bits": 0 | 4 | 8}     # 0 = bf16 (no quantization)
  Rate limited to 4/minute — a reload is ~30-60s of GPU time.
  Orchestration (all of it server-side, nothing for the operator to sequence):
    1. serialise against any other in-flight reload (409 if one is running)
    2. drain + stop the worker if it is running (in-flight image completes
       on the OLD precision and is written normally)
    3. tear the model down and rebuild at the new precision; on failure the
       classifier restores the previous configuration so the server never
       comes back with a dead model
    4. restart the worker if and only if it was running before — on every
       path including failures, so a bad request cannot leave the pipeline
       silently stopped
  Response: the runtime descriptor above, plus {reloaded, previous,
            loaded_at, worker_restarted, worker_note}

POST /retrain/start → {is_running, progress_pct, ...}
GET /retrain/status → {is_running, progress_pct, ...}

DELETE /dashboard/delete/{id}?confirm=true   (operator dashboard)
  Removes the upload everywhere: Cloudinary image (signed destroy, invalidate=true) FIRST,
  then the RoadSide photos row (cascades to the assessment), then verifies the assessment is
  gone. Deletes nothing if the image cannot be confirmed gone from Cloudinary. app/deletion.py.
  The FK cascades photos -> assessments only; deleting just the assessment (the pre-2026-09-23
  behaviour) leaves the photo visible in the app. scripts/cleanup_orphan_photos.py (dry run by
  default) removes rows the old version left behind.

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
- Extracts confidence via geometric mean of per-token log-probabilities over the DISTRESS_TYPES value tokens (`STAGE2_CONFIDENCE_MODE=field`); falls back to the whole sequence when the field span cannot be located
- Parses structured output (DISTRESS_TYPES/SEVERITY/DESCRIPTION fields)
- Returns: `{distress_types, primary_distress_type, secondary_distress_types, severity, description, stage2_confidence, stage2_confidence_primary, stage2_confidence_field, stage2_confidence_sequence, stage2_type_confidences, stage2_time_ms, stage2_raw, ...}`

### predict(image) → dict
- Calls `predict_stage1()`, then `predict_stage2()` only if distressed
- Combines into full ClassificationResponse dict
- Flags `needs_expert_review` if either stage < 80% confidence

### Confidence Extraction (Real, NOT Heuristic)
- **Stage 1:** Softmax probability over "Normal" vs "Distress" first-sub-token IDs only (cached at model load). Unchanged since Phase 1.
- **Stage 2:** Geometric mean of per-token probabilities `exp(mean(log(p_i)))`, restricted to the
  tokens of the `DISTRESS_TYPES:` value (`STAGE2_CONFIDENCE_MODE=field`, default since 2026-09-17).
  The whole-sequence variant is still computed and stored, but is **not** what the gate reads.
- **Why the restriction:** the whole-sequence geomean averaged over the free-text DESCRIPTION.
  Measured on 37 labelled images it scored AUC **0.231** — below 0.5, i.e. it ranked *wrong*
  predictions *above* right ones, because confidently-worded prose reads as high probability.
  Field-restricted scores AUC **0.613**. See `eval_results/confidence_calibration_qwen25vl7b.json`.
- Both numbers are written to `assessments.raw_response` on every row
  (`stage2_confidence_field` / `stage2_confidence_sequence` / `stage2_confidence_mode`), so the
  0.80 threshold can be re-calibrated from real expert corrections with zero re-inference.
- **Per-label confidence** (since 2026-09-23): `stage2_type_confidences` holds each listed label's
  joint probability; `stage2_confidence_primary` is the first label's. The UIs show the first
  label as the main distress with the others beside it. Gating on the first label alone was
  tested and rejected: on Attain the first label is Longitudinal Cracking 96% of the time (the
  prompt's inspection order), its confidence does not predict whether it is right (AUC 0.468),
  and a "list the most prominent first" prompt rule moved the second label instead of the first
  while adding false positives (p = 0.021). Paper §6.7 / §7.9.
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

## Setup on a New Machine

Status lives in START HERE; history in "Project History". This section is only how to get running.

```bash
cd Capstone
python -m venv venv
source venv/Scripts/activate            # Windows Git Bash; PowerShell: venv\Scripts\Activate.ps1

uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv pip install -r requirements.txt      # pins transformers 4.57.6 + bitsandbytes 0.44.1 (hard pins),
                                        # accelerate 1.11.0, peft 0.18.1 (what produced all results)

# .env (git-ignored) needs SUPABASE_URL, SUPABASE_SERVICE_KEY, CLOUDINARY_* and the P4 settings in START HERE.
python scripts/validate_all.py          # 18 checks, CPU only
.\start.ps1 -AutoStart -Watchdog        # production: loads the model (~1 min), starts the worker
# check: GET http://localhost:8000/health  (values listed in START HERE)

# Optional remote access
curl -Lo cloudflared.exe https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe
./cloudflared.exe tunnel --url http://localhost:8000
```

- The model (~15 GB) downloads on first load. 7B needs the A5000; on a 6 GB laptop use
  `MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct` for development only (never for reported numbers).
- Remote testing: open `test_website/index.html`, paste the tunnel URL, upload a road photo.
- GPU tests (`scripts/tests/test_stage2_probe_integration.py`) load a second model copy (~6 GB);
  fine next to an idle production server, not during a re-classification run.

## Model Size vs VRAM Reference

| Model | fp16 (no quant) | 4-bit quantized | Fits A5000 (24GB)? | Fits RTX 4050 (6GB)? |
|-------|-----------------|-----------------|--------------------|-----------------------|
| Qwen2.5-VL-3B | ~6GB | ~2GB | Yes | Yes (4-bit only) |
| Qwen2.5-VL-7B | ~14GB | ~4-5GB | Yes (fp16) | No (OOM even 4-bit) |
| Qwen2.5-VL-13B | ~26GB | ~7-8GB | No | No |
| Qwen2.5-VL-72B | ~144GB | ~40GB | No | No |

**Decision (original):** 7B fp16 on the A5000. **Actual production:** 7B 4-bit NF4 with the 1024² cap — every production and evaluation number since 2026-09 is at 4-bit. Use 3B 4-bit on the laptop for dev testing only.

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
