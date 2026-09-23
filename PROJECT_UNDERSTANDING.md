# GeoAI — Complete System Understanding

> Reverse-engineered reference for the repo at https://github.com/Suraj-B12/GeoAI
> (local `C:\Users\PRO-LAB-3\Documents\Capstone`, branch `main`, HEAD `75a2317`, in sync with
> `origin/main`, working tree clean, 30 commits, 124 tracked files).
> Written 2026-09-15 from source, git history, every project `.md`, and Claude memory.

---

## 1. What this project actually is

**GeoAI** is a crowdsourced pavement-distress detection system for Bengaluru roads, built as a
college capstone that must produce a publishable paper. Authors: Suraj, Richik Chaudhuri,
Sushant Deo.

A citizen photographs a damaged road with the **RoadSide** mobile app. The image goes to
**Cloudinary**, the metadata (GPS, address, URL) to **Supabase**. A GPU worker on the college
**RTX A5000** pulls pending rows, runs a **vision-language model (Qwen2.5-VL-7B-Instruct)**
through a staged inspection pipeline, writes back an IRC:82-2015-compliant distress
classification, severity and description, and flags anything under 80 % confidence for a human
expert. Finished rows feed a **WebGIS** map for civic authorities.

The AI never guesses silently: unparseable or low-confidence results are routed to expert
review, not committed as fact. That is the project's central design value ("no faking, no
overfitting, all true and accurate").

---

## 2. End-to-end data flow

```
Citizen phone (RoadSide app)
        |  JPEG -> Cloudinary (cloud: dnxpt5gea)
        |  metadata -> Supabase table `photos`
        v
Postgres trigger trg_sync_photo_to_assessment   (migration 002)
        |  AFTER INSERT ON photos -> creates `assessments` row, status='pending'
        v
PipelineWorker (app/worker.py, asyncio task inside the uvicorn process)
        |  claim_pending_assessments() RPC - FOR UPDATE SKIP LOCKED, batch of 5
        |  download image from Cloudinary (httpx)
        +- Stage 0  pavement pre-filter  -> 'no'     => status='rejected_non_pavement' (stop)
        +- Stage 1  Normal vs Distress   -> 'Normal' => skip Stage 2
        +- Stage 2  IRC:82 type + severity + description
        |  confidence gate: both stages >= 0.80 => 'classified', else 'expert_review'
        v
Supabase `assessments` (results + audit trail + raw VLM text)
        +-- operator_ui/index.html     (/operator)  live pipeline control + metrics
        +-- operator_ui/dashboard.html (/dashboard) image browser, delete, re-send to AI
        +-- expert_ui/index.html                    human correction of flagged rows
        +-- WebGIS (separate work stream, paper/WEBGIS_HANDOFF_BRIEF.md)
```

The two-table split is deliberate: the mobile-app team writes only to `photos` and never needs
to know `assessments` exists; the pipeline reads/writes only `assessments`.

---

## 3. The model pipeline (app/model.py)

Singleton `PavementClassifier`, thread-safe via `self._lock`, loaded once at FastAPI startup.

| Stage | Method | Prompt | max_new_tokens | Output |
|---|---|---|---|---|
| 0 | `predict_is_pavement()` | `PAVEMENT_FILTER_*` | 8 | `yes` / `no` / `unsure` |
| 1 | `predict_stage1()` | `STAGE1_*` | 200 | `Normal` / `Distress` + confidence |
| 2 | `predict_stage2()` | `STAGE2_*` | 200 | IRC types, severity, description + confidence |

`predict()` chains 1 -> 2 for the synchronous `/classify` path. The worker calls the stage
methods separately so the dashboard can render live per-stage progress.

### Confidence is real, not heuristic
- **Stage 1** — softmax over the *first sub-token* logits of "Normal" vs "Distress" only (token
  IDs cached at load). Falls back to full-vocab max-prob if the model emits something else first.
- **Stage 2** — geometric mean of per-token probabilities, `exp(mean(log p_i))`, restricted since
  2026-09-17 to the tokens of the `DISTRESS_TYPES:` value (`STAGE2_CONFIDENCE_MODE=field`).
  The old whole-sequence variant spanned the free-text DESCRIPTION, which capped long answers
  around ~0.75 and — worse — was measured *anti-correlated* with correctness (AUC 0.231 vs 0.613
  field-restricted, n=37). It is still computed and stored on every row, but no longer gates.
  When the field span cannot be located the whole-sequence value is returned and
  `stage2_field_span_found=false` records that it happened.
- **Per-label confidence** (2026-09-23) — each listed label's joint probability is stored
  (`stage2_type_confidences`, `stage2_confidence_primary` for the first). The UIs show the first
  label as the main distress, the rest beside it. Gating on the first label alone was tested on
  407 labelled Attain images and rejected (AUC 0.468 vs 0.635 for the whole field; paper §6.7),
  as was a prompt rule asking for the most prominent distress first (`v2_primary_first`).
- `CONFIDENCE_THRESHOLD = 0.80` is defined exactly once, in `scripts/utils.py`.

### Runtime precision switching (2026-09-18)

Quantization cannot be changed in place - the weight storage format is fixed when
the tensors are built - so switching means a full teardown and reload (~30-60s).
`PavementClassifier.reload(quantization_bits, adapter_path)` owns that, validating
every argument before the working model is dropped and restoring the previous
configuration if the new one fails to load. `reload_adapter()` is now a thin
wrapper over it.

`POST /operator/runtime/quantization` wraps the reload in the orchestration the
operator would otherwise have to do by hand: serialise against concurrent
reloads, drain and stop the worker, reload, restart the worker - on every path
including failures, so a rejected request cannot leave the pipeline stopped.
The operator dashboard exposes this as a "Model Precision" panel.

Compute dtype stays bf16 in every mode (`bnb_4bit_compute_dtype=torch.bfloat16`),
so activations, attention, the KV cache and the logits that confidence is read
from are unaffected by the choice. Only weight storage changes.

Orchestration is covered GPU-free by `scripts/tests/test_operator_runtime.py`
(30 checks against a fake classifier and worker).

### Safety defaults baked in
- Unparseable Stage 1 -> treated as **Distress** with confidence 0.0 (never silently "Normal").
- Unparseable Stage 0 -> **unsure** -> image is kept (false rejects lose real data; false accepts
  only cost one extra inference).
- `merge_and_unload()` is never called on the quantized base (peft #2586 silently drops the LoRA
  delta).
- Processor pixel caps `max_pixels=2200^2`, `min_pixels=256*28` — added after a 3456x3456
  Cloudinary photo OOM'd a 24 GB A5000 through O(n^2) attention.

### Adapter-cascade (built, currently inert)
`predict_stage2()` runs a second pass with `PeftModel.disable_adapter()` when the first pass
returns Unknown/empty/Other, to recover the base model's zero-shot taxonomy. In production
`DISABLE_ADAPTER=true`, so there is no adapter to disable and the cascade never fires — the code
is unchanged, just dormant. Telemetry fields `stage2_used_fallback` / `stage2_primary_raw` are
still written into `raw_response`.

### Hot-swap
`reload_adapter(path|None)` tears the model down (frees VRAM first, so no 2x spike) and rebuilds
— safer than peft's `set_adapter()` on a 4-bit base. ~30 s. Driven from
`POST /operator/adapters/switch`; the entire adapter section of the operator UI is hidden when
`DISABLE_ADAPTER=true` so production cannot be broken from the browser.

---

## 4. Production configuration — and why the fine-tuned adapter is shelved

`.env` production values: `DISABLE_ADAPTER=true`, `PROMPTS_VERSION=v2`, `QUANTIZATION_BITS=0`
(fp16 on the A5000). `ADAPTER_PATH` is commented out.

So **production = base Qwen2.5-VL-7B-Instruct + v2 "Improved Baseline" IRC prompts. No LoRA.**

The adapter (`adapters/v2-rdd-2epochs-20260507/`, checkpoint-1000, eval_loss 0.0485) *passed* its
A/B promotion gate on RDD and was deployed for a week. The 3-way Attain study
(`eval_results/attain_3way_comparison.md`) then showed it destroys capabilities that matter for
real Bengaluru photos:

| Metric (Attain WS_V2.0, 769 imgs) | A: plain baseline (v1) | B: Improved Baseline (v2) | C: fine-tuned (v1) |
|---|---|---|---|
| Tier-1 in-distribution accuracy | 17.51 % | 26.93 % | **31.11 %** |
| Tier-2 zero-shot accuracy | 3.27 % | **5.88 %** | 0.00 % |
| Severity accuracy | 8.94 % | 7.23 % | **25.76 %** |
| Multi-class emission rate | 6.63 % | **52.54 %** | 16.91 % |
| Pothole F1 | 0.196 | **0.265** | **0.000** (0/163 GT — refuses NZ wide-angle) |
| Block crack F1 (zero-shot) | 0.215 | **0.304** | 0.000 |
| Weathering F1 (zero-shot) | 0.024 | **0.063** | 0.000 |
| Linear crack F1 | 0.479 | 0.637 | **0.677** |
| Alligator crack F1 | 0.018 | 0.055 | **0.217** |

The adapter wins aggregate Tier 1 (+4.18 pp) and severity (+18.53 pp) but zeroes out Pothole,
Block crack, Raveling and Weathering, and collapses multi-class output. IRC §7.1 explicitly
expects multi-class reporting, and potholes dominate Indian urban roads — so the trade is
unacceptable in production even though it looks better on the RDD test set.

Honest framing kept in the repo: the commit message for the 3-way test records that the
hypothesis "baseline + v2 prompts >= fine-tuned" is **NOT supported as stated** (head-to-head,
fine-tuned wins 118 images vs 73). The production decision rests on *class-level* behaviour and
IRC compliance, not on the aggregate score. Decomposition finding worth citing in the paper:
**v2 prompts alone deliver ~69 % of the total Tier-1 improvement** (9.42 of 13.6 pp), and close
80 % of the linear-crack F1 gap.

---

## 5. IRC:82-2015 taxonomy — the domain backbone

`scripts/irc82_taxonomy.py` (591 lines) is the single source of truth, verified against the
Indian Roads Congress PDF (`IRC 82/irc.gov.in.082.2015.pdf`, Section 7). Scope is
"Option B — pure vision-only": **18 types across 4 IRC categories**.

| IRC § | Category | Type | Severity tiers |
|---|---|---|---|
| 7.2.1 | Surface Defects | Bleeding | Low/Medium/High |
| 7.2.3 | Surface Defects | Streaking | N/A |
| 7.2.4 | Surface Defects | Hungry Surface | N/A |
| 7.3.2 | Cracks | Hairline Cracks | N/A |
| 7.3.3 | Cracks | Alligator Cracking | L/M/H (1-3 / 3-6 / >6 mm) |
| 7.3.4 | Cracks | Longitudinal Cracking | L/M/H (1-3 / 3-6 / >6 mm) |
| 7.3.5 | Cracks | Transverse Cracking | L/M/H (1-3 / 3-6 / >6 mm) |
| 7.3.6 | Cracks | Edge Cracking | L/M/H (10 % break-up) |
| 7.4.1 | Deformation | Slippage | Low/High |
| 7.4.2 | Deformation | Rutting | Low/High (4-10 / >10 mm) |
| 7.4.3 | Deformation | Corrugation | N/A |
| 7.4.4 | Deformation | Shoving | N/A |
| 7.4.5 | Deformation | Shallow Depression | N/A |
| 7.4.6 | Deformation | Settlement | N/A |
| 7.5.1 | Disintegration | Stripping | N/A |
| 7.5.2 | Disintegration | Ravelling | Low/Medium/High |
| 7.5.3 | Disintegration | Potholes | **Small/Medium/Large** (25/200/500 mm) |
| 7.5.4 | Disintegration | Edge Breaking | N/A |

Deliberately excluded: §7.2.2 Smooth Surface (severity needs a skid-number measurement) and
§7.3.7 Reflection Cracking (needs knowledge of the underlying layer). Both exclusions are
documented and machine-checked.

- `canonicalize_to_irc(label)` maps legacy RDD codes (D00/D10/D20/D40/D43/D44/D50), aliases
  (block crack -> Alligator Cracking, weathering/oxidation -> Hungry Surface, patch -> Skin Patch)
  and free text to canonical names; returns `None` for Normal/Unknown/Other so they are dropped.
- `render_taxonomy_for_prompt()` / `render_severity_criteria_for_prompt()` generate the prompt
  blocks from the same data — prompts can never drift from the taxonomy.

### Verification: 5 adversarial rounds, all PASS (`eval_results/irc82_verification_report.md`)
1. **PDF cross-reference** — 20/20 §7 subsections accounted for (18 mapped, 2 excluded).
2. **Parser canonicalization** — 32/32 synthetic cases (legacy codes, case, whitespace,
   hallucinated D-codes, Other/Unknown filtering, keyword fallback, empty input).
3. **Severity thresholds** — 16/16 encoded mm strings literally found in the PDF text.
4. **Prompt audit** — v1 and v2 both name all 18 types, cite §-numbers, forbid non-IRC labels.
5. **Empirical, 100 real Attain images / 145 emissions** — 100 % IRC label compliance,
   **0 % hallucination**, 0 % legacy slip-through, 100 % §-citation rate, 100 % severity-format
   compliance, 69 % multi-class rate.

Re-runnable: `python scripts/test_irc82_robustness.py` (rounds 1-4, ~1 s).

---

## 6. Prompts: v1 "Plain" vs v2 "Improved Baseline"

Selected at import time by `PROMPTS_VERSION` in `app/model.py`.

- **v1** (`scripts/utils.py`) — professional inspector persona, 5-step Stage 1 checklist, 6-step
  IRC Stage 2 protocol, full taxonomy + quantitative severity, 3 few-shot examples, explicit
  forbidden labels ("Other", "Unknown", "Various"). Kept for paper A/B runs only.
- **v2** (`scripts/utils_v2_prompts.py`) — the production set. Same IRC vocabulary, maximally
  engineered framing: named persona ("Dr. Rajesh Kumar", BBMP Senior Pavement Inspection
  Engineer, IIT Bombay PhD, 25 yrs), concrete stakes (₹15,000 crore budget, wrongful-death
  litigation, careers), "ONE CHANCE / NO REVIEW" urgency, numbered protocol, explicit exclusion
  list (shadows, lane markings, wet patches, oil, manhole covers), strict output format. Built
  from documented prompt-engineering literature (DamageQwen 2025; Bsharat et al. 2024
  principles 5, 9, 17, 24).
- Stage 0's pre-filter prompt is **identical in both** (re-exported), so the screening decision
  never changes with the prompt switch.

Stage 2 output contract:

```
DISTRESS_TYPES: <comma-separated canonical IRC names>
SEVERITY: <Low|Medium|High  |  Small|Medium|Large for potholes  |  N/A>
DESCRIPTION: <one sentence citing an IRC:82 section>
```

---

## 7. Database

`db_schema.sql` plus four migrations, all applied by pasting into the Supabase SQL editor.

| Migration | Adds |
|---|---|
| `001_operator_pipeline.sql` | `status` CHECK enum, `claimed_at`/`claimed_by`/`retry_count`/`error_message`/`processed_at`, partial indexes, RPCs `claim_pending_assessments` (FOR UPDATE SKIP LOCKED) and `reset_stale_processing_assessments`, `worker_state` table, `pipeline_metrics` + `pipeline_class_distribution` views |
| `002_photos_integration.sql` | `photo_id` FK + unique index, `sync_photo_to_assessment()` + `trg_sync_photo_to_assessment` trigger, backfill, RLS on `photos` |
| `003_expert_ui_anon_access.sql` | `reviewer_name TEXT`, anon RLS policies scoped only to `needs_expert_review=TRUE` / `expert_reviewed=TRUE` rows |
| `004_pavement_filter_and_dashboard.sql` | `rejected_non_pavement` status, `pavement_filter_decision`/`_raw`/`_at`, rejected-rows index |

Status state machine:
`pending -> processing -> (classified | expert_review | rejected_non_pavement | failed)`,
with `done` reserved for post-review completion.

**The constraint that bit hardest:** `severity CHECK IN ('None','Low','Medium','High','Unknown')`.
The IRC prompts legitimately emit `Small` / `Large` (pothole tiers, §7.5.3.4) and `N/A`
(severity-not-applicable types — 8 of the 18). Every such row failed its PATCH with a 400 and
went to `status='failed'` — 24 rows in production, and structurally ~30 % of all distress images
at scale. Fixed at HEAD (`75a2317`) by `normalize_severity()` in `scripts/utils.py`:
Small->Low, Large->High, N/A->None, minor/moderate/severe->Low/Medium/High, extensive->High,
anything unrecognized->Unknown (which routes to expert review instead of crashing the write).
Applied in two places — inside `parse_stage2_response()` and again at the worker's write —
defense in depth. 24-case smoke test passes. **`utils.py` is module-cached, so the worker must
be restarted for the fix to take effect.**

---

## 8. API surface (app/main.py, 1047 lines)

**Classification**
- `POST /classify` — multipart image -> full `ClassificationResponse`
- `POST /classify/base64` — same via JSON
- `POST /classify/stream` — SSE (`sse-starlette`):
  `stage1_start -> stage1_complete -> stage2_start -> stage2_complete -> result`.
  Client must use `fetch()` + `ReadableStream` (EventSource is GET-only). Checks for client
  disconnect before Stage 2 so a closed tab doesn't burn GPU time.
- `GET /health` — model status **plus the live pipeline truth**: `adapter_disabled_in_config`,
  `prompts_version`, `taxonomy` (`IRC:82-2015`), `pavement_filter_enabled`, `device`.

**Operator pipeline**
- `GET /operator` (UI), `POST /operator/start|stop`,
  `GET /operator/status|metrics|recent|inflight`
- `GET /operator/metrics/stream` — SSE every 2 s
- `GET /operator/adapters`, `/adapters/active`, `POST /operator/adapters/switch`
- `GET /operator/training/status`

**Image dashboard**
- `GET /dashboard` (UI), `/dashboard/summary`, `/dashboard/list` (status + date filters, paged)
- `DELETE /dashboard/delete/{id}?confirm=true` — removes the upload everywhere
  (`app/deletion.py`, since 2026-09-23): signed Cloudinary `destroy` with `invalidate=true`
  first, then the RoadSide `photos` row (cascades to the assessment), then confirms the
  assessment is gone. All-or-nothing on the Cloudinary side: missing creds, a foreign
  Cloudinary account, an API error, or a "not found" while the image is still served abort
  before any row is touched. The previous version deleted only the assessment. The FK
  cascades photos → assessments, not the reverse, so 16 `photos` rows were left behind, still
  visible in the app with dead images. `scripts/cleanup_orphan_photos.py` (dry run by
  default) lists and removes them.
- `POST /dashboard/reclassify/{id}` — reset one row to pending (false-reject recovery)

**Retraining**: `POST /retrain/start`, `GET /retrain/status`.

Security (`app/security.py`): optional `X-API-Key` auth (`API_KEYS` env), slowapi rate limits,
10 MB upload middleware, image validation on **both** PIL `.format` and `.mode`, 4096x4096 max.
CORS configurable. `/test_fixtures` static mount only when `ENABLE_TEST_FIXTURES=1`.

---

## 9. Worker internals (app/worker.py, 671 lines)

- Runs as an asyncio task inside the same uvicorn process, sharing the classifier singleton;
  inference is pushed through `run_in_executor` so the event loop stays responsive.
- **Not auto-started** — the operator must POST `/operator/start` (prevents accidental autonomous
  processing in dev). `start.ps1 -AutoStart` does it for you.
- Heartbeat to `worker_state` every 30 s; stale-claim recovery every 120 s.
- Retries transient errors with exponential backoff; non-transient errors fail fast to
  `status='failed'` with a 500-char error message.
- Graceful stop drains the current batch and releases unprocessed claims back to `pending`.
- `try/finally` always clears `current_image_id` + `current_stage`, so the dashboard never shows
  a phantom in-flight image.
- Metrics: rolling 50-image averages for **Stage 0, 1 and 2 separately**, counters for processed
  / classified / flagged / **rejected_non_pavement** / errors, throughput.
- Every row gets an audit trail in `raw_response`: stage 0/1/2 raw VLM text, per-stage ms,
  cascade telemetry.
- Robustness suite `scripts/tests/test_pipeline_robustness.py` — 404 image, corrupt JPEG, network
  timeout, mid-process stop, concurrent claim — 5/5 pass.

---

## 10. User interfaces (all vanilla single-file HTML, shared design system)

Design tokens: SK Modernist font, `--bg #f5f5f4`, `--surface #fff`, `--text #1c1917`,
`--red #dc2626`, `--amber #d97706`, `--green #16a34a`, 12/16 px radii. No frameworks, no build
step, no gradients. Mobile-responsive.

| UI | Route | Purpose |
|---|---|---|
| `operator_ui/index.html` (1951 L) | `/operator` | Start/Stop pipeline, SSE live metrics, **Pipeline panel** fed by `/health` (model, adapter, prompts, taxonomy, pre-filter, device), Stage 0-3 progress rows with live timers, adapter management (hidden in production) |
| `operator_ui/dashboard.html` (880 L) | `/dashboard` | Image browser: collapsible date groups (collapsed by default), click-to-load thumbnails, optional autoload toggle (localStorage), endless scroll 50/page, status tabs with badges, delete with preview modal, "Send to AI" re-classify, optimistic local updates |
| `expert_ui/index.html` (1321 L) | file:// or served | Human correction of flagged rows. No auth — anon RLS + `reviewer_name` in localStorage. Pending/Reviewed tabs, editable reviewed cards, lightbox with cursor-anchored scroll-zoom + pan, 30 s response cache mirrored to sessionStorage, Cloudinary transform URLs (`f_auto,q_auto,w_900,c_limit`), lazy images, "Retrain Model" button |
| `test_website/index.html` (978 L) | standalone | Public demo: paste tunnel URL, drag-drop photo, live SSE progress, confidence bars with the 80 % line marked, session history |
| `gradio_ui/demo.py` | local | Quick interactive testing |

Both operator UIs are **zero-touch**: API base defaults to `window.location.origin`
(`http://localhost:8000` under `file://`), auto-connect on load, credentials hidden behind a
connection pill that opens an override drawer. They cross-link to each other.

---

## 11. Training (Phase 2) — what it took

`configs/qwen25vl_qlora_sft.yaml` via LLaMA-Factory, launched by `scripts/train_qlora.py`
(imports `run_exp` in-process so a Supabase heartbeat callback can attach), optionally wrapped as
a Windows service by `scripts/setup_training_service.ps1` (NSSM -> survives RDP disconnect,
auto-restarts on crash).

Final config: QLoRA 4-bit NF4, **lora_rank 32 / alpha 64 / dropout 0.1**, `lora_target: all`,
frozen vision tower, `per_device_train_batch_size 1` x `grad_accum 32` (effective 32),
`cutoff_len 1024`, `image_max_pixels 100352` (128 vision tokens), lr 1e-4 cosine + 10 % warmup,
2 epochs, bf16, `max_grad_norm 0.3`, `weight_decay 0.05`, NEFTune 5.0,
**`label_smoothing_factor 0.0`** (0.1 materialised a ~5 GB `-log_softmax` tensor and OOM'd at
step 1), `paged_adamw_8bit`, `torch_empty_cache_steps 4`, `save_steps 200`,
`enable_liger_kernel: false` (Liger is incompatible with QLoRA), `template: qwen2_vl`, seed 42.

Six commits of OOM/thrashing hardening are preserved in history (`ec4cffb` -> `5c539f1`) and are
themselves a paper section: batch 4->2->1, grad_accum 8->16->32, rank 64->32,
adamw->paged_adamw_8bit, cutoff 2048->1024, pixels 200704->100352, empty-cache every 4 steps
against allocator fragmentation (step times had ballooned 22 s -> 330 s/it).

Result: **43 h 27 m wall clock on the A5000**, early-stopped at step 2500/4806 (eval loss diverged
3 evals running). Best *surviving* checkpoint = **checkpoint-1000, eval_loss 0.0485** (step-1500
was lower at 0.0478 but `save_total_limit=3` pruned it). The final step-2500 adapter is overfit
and must not be deployed.

`scripts/preflight_dataset.py` must pass before training — it catches broken JPEGs, dimension
violations, JSON shape errors, and `<image>`-tag/image-count mismatches that crash Qwen2.5-VL
mid-epoch.

---

## 12. Every verified number

### Phase 1 baseline — base 7B fp16 (`eval_results/baseline_results.json`)
- **Stage 1**, 10,000 GAPs test images: accuracy **76.53 %**, F1-macro 71.44 %, Normal recall
  98.97 % / precision 72.21 %, **Distress recall 42.88 %** (precision 96.51 %), 0 unparseable,
  2.08 s/img. -> strong Normal bias.
- **Stage 2**, 5,758 RDD test images: primary accuracy **48.66 %**, F1-macro 20.71 %, exact match
  34.40 %, per-class recall D00 72.5 % / D10 0 % / D20 2.57 % / D40 1.94 %, 34.24 % unparseable,
  5.32 s/img.

### Phase 2b fine-tuned, same test sets (`finetuned_results.json`, `ab_comparison_table.md`)
| Metric | Baseline | Fine-tuned | Delta |
|---|---|---|---|
| Stage 1 accuracy | 76.53 % | **88.39 %** | +11.86 pp |
| Stage 1 F1-macro | 71.44 % | 87.25 % | +15.82 pp |
| Stage 1 Distress recall | 42.88 % | **73.15 %** | +30.27 pp |
| Stage 2 accuracy | 48.66 % | **66.06 %** | +17.40 pp |
| Stage 2 F1-macro | 20.71 % | **40.32 %** | +19.61 pp |
| Stage 2 exact match | 34.40 % | 52.76 % | +18.36 pp |
| D20 Alligator recall | 2.57 % | **35.22 %** | +32.65 pp |
| D40 Pothole recall | 1.94 % | **34.56 %** | +32.61 pp |
| D10 Transverse recall | 0.00 % | 1.72 % | +1.72 pp (still unsolved) |

### Phase 2c A/B promotion gate — **PASSED**
Stage 1 accuracy delta +11.86 pp (>= +3), Stage 2 macro-F1 delta +19.61 pp (>= +10), Normal-class
precision delta +12.42 pp (>= -5). Adapter promoted to `adapters/v2-rdd-2epochs-20260507/`.

### Phase 2d Bengaluru re-classification (76 real photos)
classified 29 -> 32, expert_review 47 -> 45; 7 review flags resolved, 6 newly flagged; Stage 1
Normal->Distress x5, Distress->Normal x2; 22 type changes, 15 severity changes; the cascade
fallback fired on 51 % of rows (95 % of Stage 2 inferences).

### Phase 2e/2f Attain cross-dataset (769 images, WS_V2.0) — table in §4
Pothole-bias finding worth its own paper paragraph: the adapter emits Pothole on **0 of 769**
Attain images (163 GT instances) yet *over*-predicts pothole on close-up Bengaluru smartphone
shots. The bias is **dataset-conditional, not unconditional** — it learned RDD's close-up pothole
appearance and refuses NZ wide-angle vehicle-mounted framing.

### Artefacts
22 plots at 300 DPI in `eval_results/plots/{rdd,bengaluru,attain}/` (indexed in
`plots/INDEX.md`), 3 paper-ready comparison docs, snapshots of the pre-IRC v1 results in
`eval_results/snapshots/v1_2026-05-08/`, and `paper/GeoAI_Technical_Handbook.pdf` generated by
`scripts/generate_handbook_pdf.py` (1805 lines of reportlab + matplotlib).

---

## 13. Commit history — the actual story (30 commits)

**Phase 1 — foundation (Mar 24 – Apr 30)**
- `a6aa423` Two-stage Qwen2.5-VL pipeline: scripts 01-06, FastAPI + SSE, expert UI, test website.
- `b1d5b74` Phase 1 baseline complete + operator pipeline + expert review UI (migrations 001-003,
  worker, Supabase client, dashboards).

**Phase 2a — training the adapter (May 1 – May 3)** — eight commits, six of them OOM firefighting:
`96919fe` infra -> `d9c9d06` early stopping via callback -> `bb05df8` template fix -> `ec4cffb`,
`c6398f3`, `35ba024`, `021b1d1`, `5c539f1` the VRAM ratchet -> `b3c0a47` training complete, early
stop at 2500, use checkpoint-1000.

**Phase 2b-2d — evaluate, gate, promote (May 4 – May 8)**
`80aff92` checkpoint `.bak` fallback (Windows atomic rename is unreliable), `038ef73`
gc + empty_cache every 50 samples in Stage 2 eval, `033d575` / `2074d04` post-finetune eval +
A/B gate PASSED + adapter promoted, `320522a` adapter-cascade fallback + Bengaluru re-classify +
first Attain eval, `c84b31b` baseline-vs-finetuned on Attain + 18 plots.

**Phase 2f + IRC alignment — the pivot (May 16, ten commits in one day)**
- `4e20546` 3-way hypothesis test (A/B/C on Attain) -> exposes the adapter's zero-shot collapse.
- `b6849b2` rename "Codex" -> "Improved Baseline" across code, plots and docs.
- `73e5863` IRC:82-2015 taxonomy module; prompts regenerated from it.
- `07a62b8` 5-round adversarial IRC verification, all pass.
- `38bb277` **wire Improved Baseline into production** (`DISABLE_ADAPTER=true`,
  `PROMPTS_VERSION=v2`, `.env.example` committed).
- `3119c29` Stage 0 pavement pre-filter + migration 004 + image dashboard + reset-for-reclassify.
- `7d31766` zero-touch UIs + migration-004 probe so the dashboard works un-migrated.
- `670389c` Pipeline panel from `/health`, Stage 0 as a first-class metric, cross-linked UIs.
- `2cd1d47` dashboard rebuild (collapsed date groups, click-to-load, endless scroll) plus
  `start.ps1` / `stop.ps1` / `start.sh` / `health_check.ps1`.
- `0236d8f` reset payload column-name fix (`retry_count` / `claimed_by`, not `attempt_count` /
  `worker_id`).
- `2cabeb1` operator UI stage-progress fixes (missing Stage 0 row; stale closure-captured worker
  in `setInterval` showing "191.9 s download"; per-image row reset).
- `75a2317` **HEAD** — `normalize_severity()`, root cause of the 24 failed rows.

Commit messages in this repo are unusually rich: root-cause analysis, smoke-test output,
before/after byte counts, even JS brace-balance checks. They are a legitimate source for the
paper's engineering-story section.

---

## 14. Operations

```bash
# Production (Windows / college A5000)
.\start.ps1 -AutoStart -Tunnel -Watchdog   # pre-flight, uvicorn, worker, cloudflared, auto-restart
.\stop.ps1                                 # graceful drain, optional force-kill
.\scripts\health_check.ps1                 # 6-check diagnostic, cron-friendly
```

```bash
# Linux variant
./start.sh
```

```bash
# Manual
uvicorn app.main:app --env-file .env --host 0.0.0.0 --port 8000 --workers 1
```

```bash
./cloudflared.exe tunnel --url http://localhost:8000
```

`start.ps1` pre-flight: venv present, `.env` present, production config sane, port free, stale
`uvicorn app.main` processes killed, cloudflared present. Then prints the URL map (`/operator`,
`/dashboard`, `/health`, `/docs`).

Validation: `python scripts/validate_all.py` — 13 tests covering utils exports, schemas, model
structure, security constants, training YAML, DB schema, expert UI, round-trip parsing, and a
scan for hardcoded `C:\Users\suraj` paths.

Deployment order after pulling this HEAD:
1. Apply `migrations/004_pavement_filter_and_dashboard.sql` in the Supabase SQL editor.
2. Restart the API server (**required** — module caching means the severity fix only lands on
   restart).
3. `python scripts/reset_for_reclassify.py --apply` (dry-run by default; refuses to touch
   in-flight `processing` rows).
4. `/operator` -> Start. `/dashboard` for browsing and cleanup.

---

## 15. Hard-won gotchas (do not relearn these)

| # | Gotcha |
|---|---|
| 1 | `transformers==4.57.6`, `bitsandbytes==0.44.1`, `accelerate==0.34.2`, `peft==0.14.0` — pinned. transformers >= 4.52 has a `Params4bit` bug that breaks loading. |
| 2 | Never `merge_and_unload()` a quantized base (peft #2586) — LoRA delta silently dropped, no error. |
| 3 | Liger Kernel + QLoRA = broken. `enable_liger_kernel: false`. |
| 4 | LLaMA-Factory now unifies Qwen2-VL / 2.5-VL under `template: qwen2_vl`; `qwen2_5_vl` no longer exists. (CLAUDE.md still claims the opposite — the YAML is right.) |
| 5 | `label_smoothing_factor > 0` OOMs a 7B VLM at step 1 (~5 GB `-log_softmax`). |
| 6 | Phone photos OOM without processor pixel caps — cap at the processor, not with PIL. |
| 7 | `<image>` tag count must equal `len(images)` or Qwen2.5-VL raises IndexError mid-epoch. |
| 8 | Windows: no symlinks without admin -> `active_version.txt`; atomic rename unreliable -> `.bak` fallback in checkpoint saves. |
| 9 | SSE from a Windows server sends `\r\n`; normalize in JS before splitting on `\n\n`, and concatenate multi-line `data:` fields. |
| 10 | `capture="environment"` on mobile file inputs blocks gallery selection — use `accept="image/*"` alone. |
| 11 | JS `history` shadows `window.history` — renamed `sessionHistory`. |
| 12 | PostgREST can't `UPDATE ... LIMIT` — the atomic claim must be an RPC with `FOR UPDATE SKIP LOCKED`. |
| 13 | Live columns are `retry_count` / `claimed_by` (migration 001), not `attempt_count` / `worker_id`. |
| 14 | A service-role JWT cannot run DDL on hosted Supabase — migrations go through the SQL editor by hand. |
| 15 | Long eval runs need checkpoint resumption (every 25-50 images, `.tmp` + rename) — the lab machine reboots. |
| 16 | Anon RLS only exposes rows matching the review predicates; "0 rows but the data is there" is almost always a policy mismatch. |
| 17 | Closure-captured objects in `setInterval` go stale — keep a module-level ref updated on each poll (the "191.9 s download" bug). |

---

## 16. Current state and what's left

**Working and verified**
- Full ingest -> Stage 0/1/2 -> Supabase -> operator / dashboard / expert UIs, running on the base
  7B with IRC v2 prompts.
- IRC:82 compliance verified five ways, including empirically on 100 real images.
- Complete baseline / fine-tuned / cross-dataset evaluation suite with 22 plots and 3 comparison
  documents.
- One-command startup with watchdog, health checks, graceful stop.

**Immediately pending**
- Restart the worker so `normalize_severity()` takes effect, then re-process the pending queue
  (61 rows were reset; the 24 previously-failed rows were manually PATCHed back to `pending`).
- Confirm migration 004 is applied on the live DB (the code probes and degrades gracefully if not).

**Phase 3 — expert-in-the-loop (designed, not built)**
`build_dynamic_stage2_prompt(corrections)` in `utils.py`; `add_correction()` few-shot list in
`model.py`; `POST /corrections`; versioned adapter output `adapters/vN/` + `metadata.json` +
`active_version.txt` in `06_incremental_retrain.py`; expert-UI version dropdown. Approach C
(few-shot *and* LoRA retrain) was chosen deliberately — it gives the paper two methodology
sections plus an immediate-vs-permanent improvement table. The UI is ready; no real corrections
collected yet.

**Phase 4+** — WebGIS (brief handed off in `paper/WEBGIS_HANDOFF_BRIEF.md`), repair-cost
estimation from BBMP/IRC rates, field validation on 50-100 Bengaluru roads, temporal
deterioration tracking, YOLOv8 benchmark.

**Doc drift to be aware of:** `CLAUDE.md` is authoritative on the production model and the IRC
taxonomy, but its phase checklists still describe Phase 2 as "needs executing" and Phase 3 code as
unwritten, and its Project Structure omits `app/worker.py`, `app/supabase_client.py`,
`migrations/`, `operator_ui/` and the Stage 0 pipeline. `paper/baseline_evaluation.md` still
carries placeholder columns for Phase 2/3 results that the evaluation JSONs can now fill.

---

## 17. Model-upgrade work (2026-09-15/16)

### What changed

| File | Change |
|---|---|
| `scripts/model_loader.py` **(new)** | Family-agnostic loader shared by production and evaluation. Class resolution from config, VRAM pre-flight, OOM ladder (bf16→4-bit), processor-kwarg fallback, load-time self-test, CPU/disk-offload detection, allocator cleanup. |
| `app/model.py` | Uses the shared loader. Adds field-restricted Stage 2 confidence (`_find_field_token_span`, `_compute_field_confidence`) alongside the legacy sequence metric — both always reported. `STAGE1_MAX_NEW_TOKENS` (200→12). `load_info` / `confidence_mode` properties. |
| `scripts/03_baseline_eval.py` | Loads via the shared loader; records `LAST_LOAD_INFO`. |
| `scripts/07_cross_dataset_eval.py` | Records model provenance in the summary. **Checkpoint identity guard**: checkpoint filenames now include the model, and a resume is refused if the stored model/prompts differ — previously an interrupted run of model A could be silently resumed into model B's results file. |
| `app/worker.py` | Writes both confidence metrics into `raw_response`. |
| `app/main.py`, `app/schemas.py` | `/health` reports what actually loaded (family, class, real quantization, OOM fallback, effective pixel cap, confidence mode, VRAM). |
| `scripts/validate_all.py` | 13→14 tests. Fixed two stale assertions: the model.py guardrail strings moved to the loader, and the training-YAML test still asserted the pre-2026-05-01 config (it had been failing for months while the YAML was correct). |
| `scripts/smoke_test_model.py` **(new)** | Pre-flight gate for a model swap: load, all three stages, parse, IRC label validity, DB-writable severity, both confidences, on the committed fixtures. Exits non-zero on any hard failure. |
| `scripts/compare_model_ab.py` **(new)** | A/B comparison that **refuses** to compare runs differing in more than the model. |

### Verification performed

- `validate_all.py`: **14/14 pass**.
- Field-confidence span finder: 8/8 synthetic cases; verified directional (confident labels →
  field > sequence; confident prose → field < sequence); degenerate inputs safe.
- `smoke_test_model.py`: Qwen2.5-VL-7B **37 pass / 4 warn / 0 hard fail**; Qwen3-VL-8B
  **39 pass / 2 warn / 0 hard fail**.
- **Refactor proven output-neutral:** the first 25 Attain images re-run under the refactored code
  match the 2026-05-16 pre-refactor run on **25/25** Stage 1 labels, distress types and
  severities. This is what makes `attain_irc_audit_100.json` usable as the A/B control.

### Qwen3-VL-8B viability

Drop-in on the existing pinned stack — no dependency change:
`transformers 4.57.6` already ships `Qwen3VLForConditionalGeneration` / `Qwen3VLProcessor`, and
the vendored LLaMA-Factory already has `qwen3_vl` / `qwen3_vl_nothink` templates, so the training
path is available too. 8.77B params vs 8.29B for the "7B" (both include the vision tower), so
bf16 is ~17.5 GB vs ~16.6 GB — about +1 GB, not the +2.4 GB first estimated.

### Open item

`empty_cache_after_load` defaults to True. Its 5x win is **measured on repeated same-size inputs**
(the API path). For large variable-size batch images the controlled test did not complete — one
config hit a CUDA "unspecified launch failure" (collateral from force-killing an earlier run) and
the other was abandoned to free the GPU. Historical batch runs without the cleanup managed
37.9 s/image versus ~48 s/image observed with it, so a batch-path penalty cannot be ruled out.
Toggle via the parameter and measure before drawing a conclusion.
