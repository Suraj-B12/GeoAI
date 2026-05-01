"""
QLoRA training launcher with local-file heartbeat + crash-safe wrapping.

This is the entry point for the actual fine-tuning run. It:
  1. Validates env (CUDA, VRAM, disk space, dataset preflight already passed)
  2. Sets PYTORCH_CUDA_ALLOC_CONF and other safety env vars
  3. Registers a TrainerCallback that writes a heartbeat JSON to disk every
     ~50 steps so progress is observable from any tool tailing the file
  4. Invokes LLaMA-Factory's training entry point in-process
  5. On exit (success or crash), updates the heartbeat file with final status

Heartbeat file location: outputs/training_run.json
  - Written atomically (.tmp + rename) so a crash mid-write can't corrupt it
  - Same schema as the (deprecated) Supabase training_runs table — easy to
    migrate later if remote monitoring becomes valuable
  - Tail with: Get-Content outputs\\training_run.json -Wait

Usage:
    python scripts/train_qlora.py
    python scripts/train_qlora.py --resume outputs/qwen25vl-qlora-gaps-rdd/checkpoint-2400
    python scripts/train_qlora.py --run-id manual-test-2026-04-28
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ============================================================
# Paths + constants
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "qwen25vl_qlora_sft.yaml"
HEARTBEAT_PATH = PROJECT_ROOT / "outputs" / "training_run.json"
HEARTBEAT_HISTORY = PROJECT_ROOT / "outputs" / "training_run_history.jsonl"
DISK_FREE_MIN_GB = 80


# ============================================================
# Pre-flight validation (env, disk, GPU)
# ============================================================

def preflight_environment() -> None:
    """Hard-fail before training if any prereq is broken."""
    print("=" * 70)
    print("Pre-flight environment checks")
    print("=" * 70)

    # Set CUDA allocator hints if not already
    cuda_alloc = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    desired = "expandable_segments:True,max_split_size_mb:512"
    if "expandable_segments" not in cuda_alloc:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = desired
        print(f"  Set PYTORCH_CUDA_ALLOC_CONF={desired}")

    # Disable HF tokenizer fork warning spam
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Check disk
    free_bytes = shutil.disk_usage(PROJECT_ROOT).free
    free_gb = free_bytes / (1024 ** 3)
    print(f"  Disk free:        {free_gb:.1f} GB")
    if free_gb < DISK_FREE_MIN_GB:
        print(f"  ERROR: need at least {DISK_FREE_MIN_GB} GB free for checkpoints + cache")
        sys.exit(1)

    # Check CUDA
    try:
        import torch
        if not torch.cuda.is_available():
            print("  ERROR: CUDA not available")
            sys.exit(1)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        vram_free = torch.cuda.mem_get_info(0)[0] / (1024 ** 3)
        print(f"  GPU:              {torch.cuda.get_device_name(0)}")
        print(f"  VRAM total:       {vram_total:.1f} GB")
        print(f"  VRAM free:        {vram_free:.1f} GB")
        if vram_free < 10:
            print("  WARN: < 10 GB VRAM free — close other GPU users")
    except ImportError:
        print("  ERROR: PyTorch not installed in this venv")
        sys.exit(1)

    # Check dataset preflight has been run recently
    preflight_log = PROJECT_ROOT / "preflight_problems.txt"
    if preflight_log.exists() and preflight_log.read_text(encoding="utf-8").strip():
        if "ERRORS" in preflight_log.read_text(encoding="utf-8"):
            print("  ERROR: preflight_problems.txt has unresolved errors")
            print(f"         run: python scripts/preflight_dataset.py")
            sys.exit(1)

    print("  All checks passed.")
    print()


# ============================================================
# Heartbeat (local file — atomic writes, append-only history)
# ============================================================

class HeartbeatClient:
    """Atomic local-file heartbeat for training progress.

    Writes the LATEST snapshot to `outputs/training_run.json` (single object,
    overwritten every heartbeat) AND appends every heartbeat to
    `outputs/training_run_history.jsonl` so we have a full step-by-step record
    for the paper.

    Atomic write strategy: write to .tmp then rename. Rename is atomic on
    Windows for files on the same volume, so a crash mid-write can't corrupt
    the current snapshot.

    Failures are non-fatal — a heartbeat error must NEVER kill training.
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.snapshot_path = HEARTBEAT_PATH
        self.history_path = HEARTBEAT_HISTORY
        self.enabled = True
        # Ensure outputs/ exists
        try:
            self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"  [hb] could not create outputs dir: {e}")
            self.enabled = False
        # Persistent state across heartbeats so partial updates don't lose
        # fields that this particular call didn't recompute (e.g. config_path
        # is set once at startup, eta_seconds only after we have steps_per_sec)
        self._latest: dict = {"run_id": run_id}

    def upsert(self, fields: dict) -> None:
        """Merge `fields` into the latest snapshot, write it atomically."""
        if not self.enabled:
            return
        # Merge: callers pass only the fields that changed; we keep accumulated state
        self._latest.update(fields)
        self._latest["updated_at"] = datetime.now(timezone.utc).isoformat()

        # 1. Atomic snapshot write
        try:
            tmp = self.snapshot_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._latest, indent=2), encoding="utf-8")
            os.replace(tmp, self.snapshot_path)
        except Exception as e:
            print(f"  [hb] snapshot write failed (non-fatal): {e}")

        # 2. Append-only history line — never overwrites; safe to lose one line
        try:
            with self.history_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(self._latest) + "\n")
        except Exception as e:
            print(f"  [hb] history append failed (non-fatal): {e}")

    def close(self) -> None:
        # No resources to release for the file-based heartbeat
        pass


# ============================================================
# GPU sampler (used by callback for hardware metrics)
# ============================================================

def sample_gpu() -> dict:
    """Sample current GPU temperature, utilization, VRAM."""
    try:
        import torch
        from pynvml_smi import safe_query
        return safe_query()
    except Exception:
        # Fallback: just torch numbers (no temp/util)
        try:
            import torch
            free, total = torch.cuda.mem_get_info(0)
            return {
                "vram_used_gb": (total - free) / (1024 ** 3),
                "vram_total_gb": total / (1024 ** 3),
                "gpu_temp_c": None,
                "gpu_util_pct": None,
            }
        except Exception:
            return {}


def _query_nvidia_smi() -> dict:
    """Tiny inline nvidia-smi parser — avoids adding pynvml dep."""
    import subprocess
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            text=True,
            timeout=5,
        ).strip()
        temp, util, used, total = [float(x.strip()) for x in out.split(",")]
        return {
            "gpu_temp_c": temp,
            "gpu_util_pct": util,
            "vram_used_gb": used / 1024,    # nvidia-smi reports MiB
            "vram_total_gb": total / 1024,
        }
    except Exception as e:
        return {"hw_sample_error": str(e)[:100]}


# Replace the placeholder above
sample_gpu = _query_nvidia_smi


# ============================================================
# TrainerCallback that publishes heartbeats
# ============================================================

def make_heartbeat_callback(hb: HeartbeatClient, total_epochs: int):
    """Build a transformers TrainerCallback that updates training_runs every N steps."""
    from transformers import TrainerCallback

    class HeartbeatCallback(TrainerCallback):
        def __init__(self):
            self.start_time = time.monotonic()
            self.last_step = 0
            self.last_step_time = self.start_time
            self.best_eval_loss: Optional[float] = None

        def on_train_begin(self, args, state, control, **kwargs):
            hb.upsert({
                "status": "running",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "last_heartbeat_at": datetime.now(timezone.utc).isoformat(),
                "config_path": str(CONFIG_PATH),
                "output_dir": args.output_dir,
                "total_steps": int(state.max_steps) if state.max_steps else 0,
                "total_epochs": total_epochs,
            })

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            now = time.monotonic()

            # steps/sec across the last logging interval
            step_delta = state.global_step - self.last_step
            time_delta = now - self.last_step_time
            steps_per_sec = (step_delta / time_delta) if time_delta > 0 and step_delta > 0 else 0
            self.last_step = state.global_step
            self.last_step_time = now

            # ETA in seconds
            remaining_steps = max(0, (state.max_steps or 0) - state.global_step)
            eta = int(remaining_steps / steps_per_sec) if steps_per_sec > 0 else None

            train_loss = logs.get("loss")
            eval_loss = logs.get("eval_loss")
            grad_norm = logs.get("grad_norm")
            lr = logs.get("learning_rate")

            if eval_loss is not None:
                if self.best_eval_loss is None or eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss

            payload = {
                "status": "running",
                "last_heartbeat_at": datetime.now(timezone.utc).isoformat(),
                "current_step": int(state.global_step),
                "current_epoch": float(state.epoch or 0),
                "steps_per_sec": float(steps_per_sec),
                "eta_seconds": eta,
            }
            if train_loss is not None: payload["train_loss"] = float(train_loss)
            if eval_loss is not None: payload["eval_loss"] = float(eval_loss)
            if self.best_eval_loss is not None: payload["best_eval_loss"] = float(self.best_eval_loss)
            if grad_norm is not None: payload["grad_norm"] = float(grad_norm)
            if lr is not None: payload["learning_rate"] = float(lr)

            payload.update(sample_gpu())
            hb.upsert(payload)

        def on_train_end(self, args, state, control, **kwargs):
            hb.upsert({
                "status": "completed",
                "last_heartbeat_at": datetime.now(timezone.utc).isoformat(),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "current_step": int(state.global_step),
            })

    return HeartbeatCallback()


# ============================================================
# Main
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="QLoRA training launcher (local-file heartbeat)")
    parser.add_argument("--config", default=str(CONFIG_PATH), help="YAML config path")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint dir")
    parser.add_argument("--run-id", default=None,
                        help="Override run_id (default: hostname-timestamp). "
                             "Reuse the same id when resuming so the heartbeat row continues.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run env preflight only — exit before training starts. "
                             "Useful for verifying setup without burning GPU hours.")
    args = parser.parse_args()

    preflight_environment()

    # Stable run_id — survives restarts so resumed runs update the same heartbeat
    if args.run_id:
        run_id = args.run_id
    else:
        host = socket.gethostname()
        run_id = f"qwen25vl-qlora-{host}-{datetime.now().strftime('%Y%m%d-%H%M')}"

    print(f"  Run ID:           {run_id}")
    print(f"  Config:           {args.config}")
    if args.resume:
        print(f"  Resuming from:    {args.resume}")
    print()

    hb = HeartbeatClient(run_id)
    print(f"  Heartbeat:        {HEARTBEAT_PATH} (snapshot)")
    print(f"                    {HEARTBEAT_HISTORY} (history)")
    print()

    # Initial snapshot so any monitor sees this run starting
    hb.upsert({
        "status": "starting",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "last_heartbeat_at": datetime.now(timezone.utc).isoformat(),
        "config_path": args.config,
        "resume_from": args.resume,
    })

    if args.dry_run:
        print("  --dry-run: env preflight passed. Exiting without starting training.")
        hb.upsert({
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "last_error": "dry-run only",
        })
        return 0

    try:
        # We import LLaMA-Factory in-process (rather than subprocess) so we
        # can register our TrainerCallback. This requires LF to be installed.
        try:
            from llamafactory.train.tuner import run_exp
        except ImportError as e:
            print("ERROR: LLaMA-Factory not installed. Install it with:")
            print("  git clone https://github.com/hiyouga/LLaMA-Factory.git")
            print("  cd LLaMA-Factory && python -m pip install -e \".[torch,metrics]\"")
            hb.upsert({"status": "failed", "last_error": f"LLaMA-Factory not installed: {e}",
                       "completed_at": datetime.now(timezone.utc).isoformat()})
            return 1

        # Build the args list as if calling `llamafactory-cli train <yaml>`
        sys.argv = ["train", args.config]
        if args.resume:
            sys.argv += ["--resume_from_checkpoint", args.resume]

        # Read total_epochs from YAML for the heartbeat metadata
        try:
            import yaml
            with open(args.config, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            total_epochs = int(cfg.get("num_train_epochs", 2))
        except Exception:
            total_epochs = 2

        # Inject our callbacks by patching the Trainer init. LLaMA-Factory's
        # run_exp() builds the trainer internally — patching __init__ is the
        # cleanest way to attach extras without forking LF.
        from transformers import Trainer as HFTrainer
        from transformers import EarlyStoppingCallback
        original_init = HFTrainer.__init__
        heartbeat_cb = make_heartbeat_callback(hb, total_epochs)
        # Early-stopping: stop when eval_loss hasn't improved by >= 0.005 for
        # 3 consecutive evals (eval_strategy='steps', eval_steps=500 in YAML
        # → ~1500 steps grace window). Doesn't require load_best_model_at_end.
        early_stop_cb = EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.005,
        )

        def patched_init(self, *a, **kw):
            original_init(self, *a, **kw)
            try:
                self.add_callback(heartbeat_cb)
                self.add_callback(early_stop_cb)
                print("  [hb] Heartbeat + EarlyStopping callbacks attached")
            except Exception as e:
                print(f"  [hb] Failed to attach callbacks: {e}")
        HFTrainer.__init__ = patched_init

        # Hand off to LLaMA-Factory
        run_exp()
        return 0

    except KeyboardInterrupt:
        hb.upsert({
            "status": "crashed",
            "last_error": "KeyboardInterrupt",
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
        print("\nInterrupted by user.")
        return 130

    except Exception as e:
        tb = traceback.format_exc()
        print(f"\nTRAINING FAILED:\n{tb}")
        hb.upsert({
            "status": "failed",
            "last_error": f"{type(e).__name__}: {str(e)[:500]}",
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
        return 1

    finally:
        hb.close()


if __name__ == "__main__":
    sys.exit(main())
