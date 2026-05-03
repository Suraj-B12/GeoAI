# Continuation Prompt — Paste this whole file as the first message in a new Claude Code chat

> Read this entire file before doing anything. It is the handoff brief from the previous session.

---

## Project: GeoAI — Pavement Distress Detection

End-to-end system: phone app (RoadSide) → Cloudinary + Supabase → AI pipeline (Qwen2.5-VL-7B + LoRA) → WebGIS for civic authorities. Capstone project, Bengaluru. Paper authors: Suraj, Richik Chaudhuri, Sushant Deo. Project root: `C:\Users\PRO-LAB-3\Documents\Capstone`. GitHub: `https://github.com/Suraj-B12/GeoAI`.

**Read these first** — they have the full project state in your auto-memory:
- `C:\Users\PRO-LAB-3\.claude\projects\C--Users-PRO-LAB-3-Documents-Capstone\memory\MEMORY.md` (index)
- `C:\Users\PRO-LAB-3\Documents\Capstone\CLAUDE.md` (project-wide context)
- `C:\Users\PRO-LAB-3\Documents\Capstone\SETUP.md` (operations guide, full schema, all env vars)

---

## Where we are in the work

| Phase | Status |
|---|---|
| Phase 1: Baseline eval (Qwen2.5-VL-7B raw) on RDD test set | DONE — `eval_results/baseline_results.json` |
| Phase 2a: QLoRA fine-tuning | **DONE** — early-stopped at step 2500/4806, best checkpoint at `outputs/qwen25vl-qlora-gaps-rdd/checkpoint-1000/` (eval_loss 0.0485) |
| Phase 2b: Post-finetune eval | **RUNNING** as of handoff (started 2026-05-03, ~10h estimated) |
| Phase 2c: A/B promotion gate | PENDING (auto-trigger after 2b) |
| Phase 2d: Hot-swap adapter into operator pipeline | PENDING (UI ready) |
| Phase 3: Expert-in-the-loop retrain | NOT STARTED |
| Phase 4: WebGIS layer | HANDED OFF — `paper/WEBGIS_HANDOFF_BRIEF.md` |

### Phase 2a finished (training is over)

- Halted at step 2500 / 4806 by `EarlyStoppingCallback` (eval loss had risen for 3 consecutive evals)
- Best eval loss = 0.04775 at step 1500, but `checkpoint-1500` was pruned by `save_total_limit: 3`
- Closest surviving best = **`checkpoint-1000` (eval loss 0.0485)** — what we deploy
- Total training wall-clock: ~43h 27m
- Final adapter on disk (`outputs/qwen25vl-qlora-gaps-rdd/adapter_model.safetensors`) corresponds to step 2500 and is OVERFIT (eval 0.0655) — **don't deploy that**, use checkpoint-1000

Eval-loss curve (from `trainer_state.json`):
```
step  500: 0.0603   step 1000: 0.0485   step 1500: 0.0478   step 2000: 0.0633   step 2500: 0.0655
```

---

## How to check training progress (do this first)

The training writes a heartbeat to `outputs/training_run.json` every ~20 steps. Read it:

```powershell
cd C:\Users\PRO-LAB-3\Documents\Capstone
type outputs\training_run.json
```

Key fields:
- `status` — `running`, `completed`, `failed`, or `crashed`
- `current_step` / `total_steps` (4806) — progress
- `train_loss`, `eval_loss`, `best_eval_loss`
- `gpu_temp_c`, `vram_used_gb` — hardware health
- `eta_seconds` — remaining time

There's also a full step-by-step history at `outputs/training_run_history.jsonl` (one JSON per heartbeat).

**Last known state at handoff:** step 500/4806 (10.4%), train_loss 0.0555, first eval round running, 2 checkpoints saved (`outputs/qwen25vl-qlora-gaps-rdd/checkpoint-200`, `checkpoint-400`).

If `status == "completed"` → proceed to Phase 2b below.

If `status == "running"` → leave it alone. Re-check in an hour.

If `status == "failed"` or `"crashed"`, OR if `last_heartbeat_at` is more than ~10 minutes old:
1. Read the error: `tail -100 outputs/stderr.log` (if NSSM was used) or whatever shell logs exist
2. Most likely cause = OOM or fragmentation regression. We've seen this 4 times during this run. The fixes already applied:
   - `per_device_train_batch_size: 1` + `gradient_accumulation_steps: 32`
   - `lora_rank: 32` (not 64)
   - `optim: paged_adamw_8bit`
   - `torch_empty_cache_steps: 4`
   - `image_max_pixels: 100352` (128 vision tokens cap)
   - `cutoff_len: 1024`
   - `label_smoothing_factor: 0.0`
3. To resume from the latest checkpoint:
   ```powershell
   $latest = Get-ChildItem outputs\qwen25vl-qlora-gaps-rdd\checkpoint-* | Sort-Object LastWriteTime -Descending | Select-Object -First 1
   python scripts/train_qlora.py --resume $latest.FullName
   ```

---

## Phase 2b — Post-finetune evaluation (RUNNING)

This was started by the previous session. **It is currently running in the background** as of handoff.

```powershell
# Already running:
python scripts/04_post_finetune_eval.py --adapter-path outputs/qwen25vl-qlora-gaps-rdd/checkpoint-1000
```

Note the `checkpoint-1000` path — this is the best surviving checkpoint (see Phase 2a section). **Do not** point at the parent dir (`outputs/qwen25vl-qlora-gaps-rdd/`) because that loads the OVERFIT step-2500 adapter.

Estimated runtime: ~10 hours total (Stage 1 ~1.5h on 10k GAPs images, Stage 2 ~8.5h on 5,758 RDD test images). Has its own checkpoint resumption baked in (saves every 50 images to `eval_results/.stage*_checkpoint_finetuned.json`).

**To check progress:** look for the latest progress lines in the launching shell, or check `eval_results/`:
- During Stage 1: `eval_results/.stage1_checkpoint_finetuned.json` updates every 50 images
- After Stage 1 completes: `eval_results/finetuned_results.json` has Stage 1 numbers; Stage 2 runs next
- After both: full `eval_results/finetuned_results.json` + `confusion_matrices/finetuned_*.png`

If the system crashes during eval, just re-run the same command — it picks up from the checkpoint.

---

## Phase 2c — A/B promotion gate

```powershell
python scripts/ab_compare_adapters.py
```

Outputs:
- `eval_results/ab_comparison.json` (full numeric breakdown)
- `eval_results/ab_comparison_table.md` (paste-ready Markdown for the paper)
- Exit code 0 = passes promotion criteria, 1 = does not

Promotion criteria (in `scripts/ab_compare_adapters.py`):
- Stage 1 accuracy must improve by ≥ 3 percentage points
- Stage 2 macro F1 must improve by ≥ 10 percentage points
- Normal-class precision must NOT regress by more than 5 pp

---

## Phase 2d — Promote and hot-swap

Only do this if 2c passes.

```powershell
$ts = Get-Date -Format yyyyMMdd
mkdir adapters\v2-rdd-2epochs-$ts
Copy-Item outputs\qwen25vl-qlora-gaps-rdd\* adapters\v2-rdd-2epochs-$ts\ -Recurse

# Optional: write a metadata.json next to adapter_config.json so the
# operator UI shows context when listing adapters
```

Then either:
- Edit `.env` → `ADAPTER_PATH=adapters/v2-rdd-2epochs-YYYYMMDD` and restart the FastAPI server, OR
- Hot-swap via the operator dashboard: open `http://localhost:8000/operator` → Model Version card → select the new adapter from dropdown → Switch (~30s reload).

---

## What's installed and configured

| Item | State |
|---|---|
| venv at `venv/` | Python 3.12.10, all deps installed including pinned `transformers==4.57.6, accelerate==0.34.2, bitsandbytes==0.44.1, peft==0.14.0`, plus `torchaudio==2.5.1+cu121` (had to downgrade from 2.11.0) |
| LLaMA-Factory | Cloned at `LLaMA-Factory-src/` (gitignored), installed editable. Module at `llamafactory.train.tuner.run_exp` |
| GPU | NVIDIA RTX A5000, 24 GB VRAM, CUDA 12.1 |
| Disk | ~270 GB free at start of training |
| `.env` | Has `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`. **Never paste contents into chat or commits.** Already in `.gitignore`. |
| Supabase | Project `vtlkitpoffudiefuoijb`. Migrations 001, 002, 003 applied. Migration 004 was created then deleted (heartbeat moved to local file). |
| GitHub | `Suraj-B12/GeoAI`, all commits up to `5c539f1` pushed. Force-push happened earlier to remove a Co-Authored-By line; future pushes are normal. |

---

## What NOT to change without thinking

These are landmines documented during the previous session — touching them broke training for hours:

1. **Never call `merge_and_unload()` on a 4-bit base model** (peft #2586 silently drops the adapter). Already enforced in `app/model.py`.
2. **`enable_liger_kernel: false`** in YAML. Liger is incompatible with QLoRA — will silently slow down or crash.
3. **`template: qwen2_vl`** (NOT `qwen2_5_vl`). Current LLaMA-Factory unified them.
4. **`load_best_model_at_end: false`** because save_steps (200) is not a multiple of eval_steps (500). Best checkpoint is picked manually post-hoc from `trainer_state.json`'s `log_history`.
5. **`per_device_train_batch_size: 1`** was reached after 4 OOM iterations. Don't raise it.
6. **`label_smoothing_factor: 0.0`** — was 0.05, removed because it materializes `-log_softmax(logits)` at fp32 (5 GB allocation, OOM trigger).
7. **`optim: paged_adamw_8bit`** (not `adamw_8bit`) — pages optimizer state to CPU.
8. **`torch_empty_cache_steps: 4`** — without this, allocator fragmentation balloons step times from 22s to 400+s by step 25.

If you change any of these, expect to debug the same OOM/slowdown issues we already solved.

---

## File layout you should know

```
Capstone/
├── app/
│   ├── main.py                 — FastAPI server
│   ├── model.py                — PavementClassifier (with reload_adapter() for hot-swap)
│   ├── worker.py               — operator pipeline asyncio worker
│   └── supabase_client.py      — REST + RPC wrapper
├── configs/
│   └── qwen25vl_qlora_sft.yaml — hardened QLoRA config (do not edit naively)
├── scripts/
│   ├── 03_baseline_eval.py     — produces baseline_results.json
│   ├── 04_post_finetune_eval.py — produces finetuned_results.json (USE NEXT)
│   ├── ab_compare_adapters.py  — promotion gate (USE NEXT after 04)
│   ├── train_qlora.py          — training launcher (currently running)
│   ├── preflight_dataset.py    — pre-train sanity check (PASSED)
│   └── setup_training_service.ps1 — NSSM service wrapper (unused so far)
├── eval_results/
│   ├── baseline_results.json   — Phase 1 numbers (Stage 1 76.53%, Stage 2 48.66%)
│   ├── finetuned_results.json  — TO BE CREATED by Phase 2b
│   ├── ab_comparison.json      — TO BE CREATED by Phase 2c
│   └── confusion_matrices/     — PNGs from baseline + (will get) fine-tuned
├── outputs/
│   ├── qwen25vl-qlora-gaps-rdd/ — training output (checkpoint-XXX/, trainer_state.json, adapter_*.safetensors)
│   ├── training_run.json       — current heartbeat snapshot
│   └── training_run_history.jsonl — full step-by-step history
├── operator_ui/index.html      — operator dashboard with adapter management
├── expert_ui/index.html        — expert review (anon RLS, reviewer name)
├── paper/
│   ├── baseline_evaluation.md  — research doc with baseline numbers
│   ├── PAPER_INGREDIENTS.md    — verifiable metrics for the paper (READ THIS)
│   ├── CONTINUATION_PROMPT.md  — this file
│   └── WEBGIS_HANDOFF_BRIEF.md — for the WebGIS team
└── migrations/
    ├── 001_operator_pipeline.sql
    ├── 002_photos_integration.sql
    └── 003_expert_ui_anon_access.sql
```

---

## Operating the FastAPI server

It's NOT running right now — only training is. To start it (after training finishes):

```powershell
cd C:\Users\PRO-LAB-3\Documents\Capstone
venv\Scripts\activate
python -m uvicorn app.main:app --env-file .env --host 0.0.0.0 --port 8000 --workers 1
```

(Or just run `.\start.ps1`.)

Dashboard at `http://localhost:8000/operator`. Expert UI is the static HTML at `expert_ui/index.html` (open via file:// or any static server).

---

## Memory files to read

These are auto-loaded but worth reading explicitly:

- `project_geoai_overview.md` — full system arch
- `project_phase_status.md` — phase tracking + baseline numbers
- `project_operator_pipeline.md` — worker, dashboard, robustness tests
- `project_expert_ui.md` — anon RLS pattern
- `project_qlora_training.md` — training workflow + gotchas
- `feedback_version_pinning.md` — never upgrade transformers/bnb/accelerate/peft
- `feedback_checkpoint_resumption.md` — long jobs need checkpoint saves

---

## Things the previous session deferred for the new chat

1. **Manually pick the best checkpoint after training** by reading `outputs/qwen25vl-qlora-gaps-rdd/trainer_state.json` (look at `log_history` for the step with lowest `eval_loss`). With `load_best_model_at_end=false`, the final adapter on disk is the LAST checkpoint, not necessarily the best.
2. **Run Phase 2b, 2c, 2d** as documented above.
3. **Update `paper/PAPER_INGREDIENTS.md`** — fill in the post-finetune sections (currently omitted because we don't bluff).
4. **Possibly retrain at higher rank** if the gate fails. Rank 32 was a memory compromise — rank 64 has more capacity. Worth trying again with a different memory profile if results disappoint.
5. **Re-classify the 76 existing Bengaluru photos** with the new adapter for the WebGIS demo.

---

## How to talk to the user

- They prefer terse, action-oriented updates. No fluff.
- They're juggling time and don't want to play with code for every operation. Build UIs / scripts that hide complexity.
- They want everything robust ("keep it robust, do it gracefully, make no errors"). Edge cases matter.
- Don't add Co-Authored-By to commits. They had me remove it once.
- "ask questions if you've any doubts" — they explicitly invite questions when context matters. But pick your moments — don't over-ask.

That's the handoff. Good luck.
