"""
Operator Pipeline Worker.

Runs as an asyncio background task inside the FastAPI process.
Polls Supabase for pending assessments, downloads images from Cloudinary,
runs the two-stage classification, and writes results back.

Design principles:
  - GRACEFUL: stop_event.set() lets in-flight image finish, then exits
  - ROBUST: exponential-backoff retry on transient errors, atomic claim
            via FOR UPDATE SKIP LOCKED, stale-claim recovery on startup
  - OBSERVABLE: heartbeat every 30s, in-memory metrics for dashboard SSE
  - SINGLE INSTANCE: assumes one worker per A5000 GPU (model is singleton)

Lifecycle:
  1. PipelineWorker(client, image_client, classifier, worker_id)
  2. await worker.start()       -> begins asyncio task
  3. dashboard reads worker.metrics_snapshot()
  4. await worker.stop()        -> graceful drain + final heartbeat
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import socket
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import httpx
from PIL import Image

from app.model import PavementClassifier
from app.supabase_client import (
    SupabaseError,
    claim_pending_batch,
    download_image_bytes,
    get_pipeline_metrics,
    reset_stale_processing,
    update_assessment,
    upsert_worker_state,
)
from scripts.utils import CONFIDENCE_THRESHOLD


log = logging.getLogger("pipeline_worker")


# ============================================================
# Tunables (overridable via env vars)
# ============================================================

POLL_INTERVAL_SECONDS = float(os.environ.get("WORKER_POLL_INTERVAL", "10"))
BATCH_SIZE = int(os.environ.get("WORKER_BATCH_SIZE", "5"))
MAX_RETRIES = int(os.environ.get("WORKER_MAX_RETRIES", "3"))
RETRY_BASE_DELAY = float(os.environ.get("WORKER_RETRY_BASE_DELAY", "1.0"))
HEARTBEAT_INTERVAL_SECONDS = float(os.environ.get("WORKER_HEARTBEAT_INTERVAL", "30"))
STALE_CLAIM_THRESHOLD_SECONDS = int(os.environ.get("WORKER_STALE_THRESHOLD", "300"))
STALE_RECOVERY_INTERVAL_SECONDS = float(os.environ.get("WORKER_STALE_RECOVERY_INTERVAL", "120"))


# ============================================================
# Errors classified by retryability
# ============================================================

# Network / transient: retry with backoff
TRANSIENT_EXCEPTIONS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.RemoteProtocolError,
    httpx.PoolTimeout,
    asyncio.TimeoutError,
    ConnectionError,
)


# ============================================================
# Metrics (in-memory, snapshot-able for dashboard SSE)
# ============================================================

@dataclass
class WorkerMetrics:
    """In-process counters and rolling stats. Reset only on worker restart."""
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None

    # Cumulative since start
    images_processed: int = 0
    images_classified: int = 0          # confidence OK on both stages
    images_flagged_review: int = 0      # routed to expert_review
    images_failed: int = 0              # exceeded MAX_RETRIES

    # Latest single-image timings (ms)
    last_stage1_time_ms: float = 0.0
    last_stage2_time_ms: float = 0.0
    last_total_time_ms: float = 0.0

    # Rolling averages (last 50 images)
    _recent_stage0_ms: deque = field(default_factory=lambda: deque(maxlen=50))
    _recent_stage1_ms: deque = field(default_factory=lambda: deque(maxlen=50))
    _recent_stage2_ms: deque = field(default_factory=lambda: deque(maxlen=50))
    _recent_total_ms: deque = field(default_factory=lambda: deque(maxlen=50))

    # Pre-filter rejection count (Stage 0 said NO)
    images_rejected_non_pavement: int = 0

    last_processed_at: Optional[datetime] = None
    last_processed_id: Optional[str] = None
    last_error: Optional[str] = None
    last_heartbeat_at: Optional[datetime] = None
    current_image_id: Optional[str] = None

    # Per-image live progress (cleared between images)
    current_stage: Optional[str] = None
    # 'downloading' | 'pavement_filter' | 'stage1' | 'stage2' | None
    current_stage_started_at: Optional[datetime] = None
    current_stage0_time_ms: float = 0.0
    current_stage1_time_ms: float = 0.0
    current_stage2_time_ms: float = 0.0
    current_skipped_stage2: bool = False
    last_stage0_time_ms: float = 0.0

    def record_timing(self, s0_ms: float, s1_ms: float, s2_ms: float):
        self.last_stage0_time_ms = s0_ms
        self.last_stage1_time_ms = s1_ms
        self.last_stage2_time_ms = s2_ms
        self.last_total_time_ms = s0_ms + s1_ms + s2_ms
        if s0_ms > 0:
            self._recent_stage0_ms.append(s0_ms)
        if s1_ms > 0:
            self._recent_stage1_ms.append(s1_ms)
        if s2_ms > 0:
            self._recent_stage2_ms.append(s2_ms)
        self._recent_total_ms.append(s0_ms + s1_ms + s2_ms)

    @property
    def avg_stage0_ms(self) -> float:
        return sum(self._recent_stage0_ms) / len(self._recent_stage0_ms) if self._recent_stage0_ms else 0.0

    @property
    def avg_stage1_ms(self) -> float:
        return sum(self._recent_stage1_ms) / len(self._recent_stage1_ms) if self._recent_stage1_ms else 0.0

    @property
    def avg_stage2_ms(self) -> float:
        return sum(self._recent_stage2_ms) / len(self._recent_stage2_ms) if self._recent_stage2_ms else 0.0

    @property
    def avg_total_ms(self) -> float:
        return sum(self._recent_total_ms) / len(self._recent_total_ms) if self._recent_total_ms else 0.0

    def to_dict(self) -> dict:
        return {
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "stopped_at": self.stopped_at.isoformat() if self.stopped_at else None,
            "last_heartbeat_at": self.last_heartbeat_at.isoformat() if self.last_heartbeat_at else None,
            "last_processed_at": self.last_processed_at.isoformat() if self.last_processed_at else None,
            "current_image_id": self.current_image_id,
            "current_stage": self.current_stage,
            "current_stage_started_at": (
                self.current_stage_started_at.isoformat()
                if self.current_stage_started_at else None
            ),
            "current_stage0_time_ms": round(self.current_stage0_time_ms, 1),
            "current_stage1_time_ms": round(self.current_stage1_time_ms, 1),
            "current_stage2_time_ms": round(self.current_stage2_time_ms, 1),
            "current_skipped_stage2": self.current_skipped_stage2,
            "last_processed_id": self.last_processed_id,
            "last_error": self.last_error,
            "images_processed": self.images_processed,
            "images_classified": self.images_classified,
            "images_flagged_review": self.images_flagged_review,
            "images_failed": self.images_failed,
            "images_rejected_non_pavement": self.images_rejected_non_pavement,
            "last_stage0_time_ms": round(self.last_stage0_time_ms, 1),
            "last_stage1_time_ms": round(self.last_stage1_time_ms, 1),
            "last_stage2_time_ms": round(self.last_stage2_time_ms, 1),
            "last_total_time_ms": round(self.last_total_time_ms, 1),
            "avg_stage0_time_ms": round(self.avg_stage0_ms, 1),
            "avg_stage1_time_ms": round(self.avg_stage1_ms, 1),
            "avg_stage2_time_ms": round(self.avg_stage2_ms, 1),
            "avg_total_time_ms": round(self.avg_total_ms, 1),
        }


# ============================================================
# Worker
# ============================================================

class PipelineWorker:
    """
    Async worker that processes pending assessments from Supabase.

    Runs as a single asyncio task. Operator dashboard controls it via
    start() / stop() and reads metrics_snapshot() for live updates.
    """

    def __init__(
        self,
        supabase_client: httpx.AsyncClient,
        image_client: httpx.AsyncClient,
        classifier: PavementClassifier,
        worker_id: Optional[str] = None,
    ):
        self.supabase = supabase_client
        self.image_client = image_client
        self.classifier = classifier
        self.worker_id = worker_id or self._generate_worker_id()

        self.metrics = WorkerMetrics()
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._is_running = False

    @staticmethod
    def _generate_worker_id() -> str:
        """Stable ID for this worker instance: hostname + short uuid."""
        host = socket.gethostname().replace(".", "-")
        suffix = uuid.uuid4().hex[:8]
        return f"{host}-{suffix}"

    @property
    def is_running(self) -> bool:
        return self._is_running

    # -----------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------

    async def start(self) -> dict:
        """Begin the worker loop. Returns initial state dict."""
        if self._is_running:
            return {"status": "already_running", "worker_id": self.worker_id}

        # Recover any stale claims from previous crashed runs
        try:
            reset_count = await reset_stale_processing(
                self.supabase,
                stale_threshold_seconds=STALE_CLAIM_THRESHOLD_SECONDS,
            )
            if reset_count:
                log.warning("Reset %d stale 'processing' rows on startup", reset_count)
        except Exception as e:
            log.error("Failed to reset stale claims on startup: %s", e)

        self._stop_event.clear()
        self._is_running = True
        self.metrics.started_at = datetime.now(timezone.utc)
        self.metrics.stopped_at = None

        await self._write_heartbeat(initial=True)

        self._task = asyncio.create_task(self._run_loop(), name="pipeline_worker")
        log.info("Pipeline worker '%s' started", self.worker_id)
        return {"status": "started", "worker_id": self.worker_id}

    async def stop(self, drain_timeout_seconds: float = 60.0) -> dict:
        """
        Signal the loop to stop. Waits up to `drain_timeout_seconds` for the
        current image to finish before forcefully cancelling.
        """
        if not self._is_running:
            return {"status": "already_stopped", "worker_id": self.worker_id}

        log.info("Stop requested for worker '%s' — draining...", self.worker_id)
        self._stop_event.set()

        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=drain_timeout_seconds)
            except asyncio.TimeoutError:
                log.warning("Drain timeout exceeded — cancelling worker task")
                self._task.cancel()
                try:
                    await self._task
                except (asyncio.CancelledError, Exception):
                    pass

        self._is_running = False
        self.metrics.stopped_at = datetime.now(timezone.utc)

        # Final heartbeat to mark stopped state
        try:
            await self._write_heartbeat(stopped=True)
        except Exception as e:
            log.error("Failed to write final heartbeat: %s", e)

        log.info("Pipeline worker '%s' stopped", self.worker_id)
        return {"status": "stopped", "worker_id": self.worker_id}

    def metrics_snapshot(self) -> dict:
        """Read-only view of current metrics. Safe to call from any thread."""
        return {
            "worker_id": self.worker_id,
            "is_running": self._is_running,
            **self.metrics.to_dict(),
            "config": {
                "poll_interval_s": POLL_INTERVAL_SECONDS,
                "batch_size": BATCH_SIZE,
                "max_retries": MAX_RETRIES,
                "confidence_threshold": CONFIDENCE_THRESHOLD,
            },
        }

    # -----------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------

    async def _run_loop(self):
        """
        Poll-claim-process loop. Runs until _stop_event is set.

        Uses asyncio.wait_for(_stop_event.wait(), timeout=poll_interval) instead
        of asyncio.sleep so a stop request is responsive.
        """
        last_heartbeat = time.monotonic()
        last_stale_recovery = time.monotonic()

        try:
            while not self._stop_event.is_set():
                # Periodic background tasks
                now = time.monotonic()
                if now - last_heartbeat >= HEARTBEAT_INTERVAL_SECONDS:
                    await self._write_heartbeat()
                    last_heartbeat = now
                if now - last_stale_recovery >= STALE_RECOVERY_INTERVAL_SECONDS:
                    try:
                        n = await reset_stale_processing(
                            self.supabase,
                            stale_threshold_seconds=STALE_CLAIM_THRESHOLD_SECONDS,
                        )
                        if n:
                            log.warning("Recovered %d stale claims mid-loop", n)
                    except Exception as e:
                        log.error("Stale recovery failed: %s", e)
                    last_stale_recovery = now

                # Try to claim a batch
                try:
                    batch = await claim_pending_batch(
                        self.supabase, BATCH_SIZE, self.worker_id
                    )
                except Exception as e:
                    log.error("Failed to claim batch: %s", e)
                    self.metrics.last_error = f"claim_failed: {e}"
                    batch = []

                if not batch:
                    # Empty queue — wait poll interval (responsive to stop)
                    await self._sleep_or_stop(POLL_INTERVAL_SECONDS)
                    continue

                # Process each claimed row
                for row in batch:
                    if self._stop_event.is_set():
                        # Stop signal during batch — release remaining claims back to pending
                        await self._release_claim(row["id"])
                        continue
                    await self._process_one(row)
                    if self._stop_event.is_set():
                        break

        except asyncio.CancelledError:
            log.info("Worker loop cancelled")
            raise
        except Exception as e:
            log.exception("Worker loop crashed: %s", e)
            self.metrics.last_error = f"loop_crashed: {e}"
            raise

    async def _sleep_or_stop(self, seconds: float):
        """Sleep N seconds or wake immediately if stop_event is set."""
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass

    # -----------------------------------------------------------
    # Per-image processing
    # -----------------------------------------------------------

    async def _process_one(self, row: dict):
        """
        Process a single claimed assessment with retries.

        Updates Supabase with the result (status='classified' / 'expert_review' / 'failed').
        Always clears current_image_id + current_stage on exit, even on errors,
        so the dashboard never gets stuck showing a phantom in-flight image.
        """
        image_id = row["id"]
        image_url = row["image_url"]
        self.metrics.current_image_id = image_id

        try:
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    await self._do_inference(row)
                    return
                except TRANSIENT_EXCEPTIONS as e:
                    # Retry transient
                    if attempt < MAX_RETRIES:
                        delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                        log.warning(
                            "Transient error processing %s (attempt %d/%d): %s — retrying in %.1fs",
                            image_id, attempt, MAX_RETRIES, e, delay,
                        )
                        await self._sleep_or_stop(delay)
                        if self._stop_event.is_set():
                            await self._release_claim(image_id)
                            return
                    else:
                        await self._mark_failed(image_id, str(e), attempt)
                        return
                except Exception as e:
                    # Non-transient (image decode, model error, parse error) — fail fast
                    log.exception("Permanent error processing %s: %s", image_id, e)
                    await self._mark_failed(image_id, str(e), attempt)
                    return
        finally:
            # Always clear in-flight state so the dashboard returns to idle.
            # Don't reset current_stage1/2_time_ms here — UI freezes them as
            # the "last finished" timings until the next image arrives.
            self.metrics.current_image_id = None
            self._enter_stage(None)

    def _enter_stage(self, stage: Optional[str]) -> None:
        """
        Set the live-stage indicator the dashboard reads. Pass None when the
        image is finished. Always update via this method so the timestamp
        stays in sync with the stage name.
        """
        self.metrics.current_stage = stage
        self.metrics.current_stage_started_at = (
            datetime.now(timezone.utc) if stage else None
        )

    async def _do_inference(self, row: dict):
        """
        Download image, run classifier in two distinct stages, write results
        back to Supabase. Stages are executed separately (not via predict())
        so the dashboard can show live per-stage progress via current_stage.

        Stage 2 runs only if Stage 1 says the pavement is distressed
        (consistent with the synchronous predict() pipeline).

        Raises on any failure (caller handles retry classification).
        """
        image_id = row["id"]
        image_url = row["image_url"]
        loop = asyncio.get_event_loop()

        # Reset per-image state
        self.metrics.current_stage0_time_ms = 0.0
        self.metrics.current_stage1_time_ms = 0.0
        self.metrics.current_stage2_time_ms = 0.0
        self.metrics.current_skipped_stage2 = False

        # 1. Download image
        self._enter_stage("downloading")
        img_bytes = await download_image_bytes(self.image_client, image_url)
        img = Image.open(io.BytesIO(img_bytes))
        img.load()  # force decode so BytesIO can be released
        img_w, img_h = img.size

        # 2. Stage 0 — pavement pre-filter (conservative; rejects only clear non-pavement)
        self._enter_stage("pavement_filter")
        pavement = await loop.run_in_executor(None, self.classifier.predict_is_pavement, img)
        s0_time = pavement.get("time_ms", 0.0)
        self.metrics.current_stage0_time_ms = s0_time
        if not pavement["is_pavement"]:
            # Clear NO decision — reject WITHOUT running Stage 1/2.
            # Operator can re-classify from dashboard if this was a false reject.
            self._enter_stage(None)
            # Record Stage 0 timing in the rolling average so the dashboard
            # shows accurate pre-filter cost on rejects too.
            self.metrics.record_timing(s0_time, 0.0, 0.0)
            await update_assessment(self.supabase, image_id, {
                "status": "rejected_non_pavement",
                "pavement_filter_decision": pavement["decision"],
                "pavement_filter_raw": pavement["raw"],
                "pavement_filter_at": datetime.now(timezone.utc).isoformat(),
                "image_width": img_w,
                "image_height": img_h,
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "needs_expert_review": False,
                "error_message": None,
            })
            # Local metrics — counts as "processed" but routes through a separate path
            self.metrics.images_processed += 1
            self.metrics.images_rejected_non_pavement += 1
            print(f"[Worker] {image_id[:8]} rejected_non_pavement "
                  f"(decision={pavement['decision']!r}, raw={pavement['raw']!r})")
            return

        # 3. Stage 1 — binary detection (Normal vs Distress)
        self._enter_stage("stage1")
        s1 = await loop.run_in_executor(None, self.classifier.predict_stage1, img)
        self.metrics.current_stage1_time_ms = s1.get("stage1_time_ms", 0.0)
        s1_conf = s1.get("stage1_confidence", 0.0)
        is_distressed = s1.get("is_distressed", False)

        # 3. Stage 2 — only if Stage 1 says distressed
        s2: dict = {}
        if is_distressed:
            self._enter_stage("stage2")
            s2 = await loop.run_in_executor(None, self.classifier.predict_stage2, img)
            self.metrics.current_stage2_time_ms = s2.get("stage2_time_ms", 0.0)
        else:
            self.metrics.current_skipped_stage2 = True

        # Inference complete — clear stage indicator before the DB round trip
        # so the dashboard shows "writing results" implicitly via idle state.
        self._enter_stage(None)

        s2_conf = s2.get("stage2_confidence", 0.0)

        # 4. Determine final status
        if not is_distressed:
            # Normal pavement — classified directly, no Stage 2
            final_status = "classified" if s1_conf >= CONFIDENCE_THRESHOLD else "expert_review"
        else:
            # Both stages must clear threshold
            if s1_conf >= CONFIDENCE_THRESHOLD and s2_conf >= CONFIDENCE_THRESHOLD:
                final_status = "classified"
            else:
                final_status = "expert_review"

        s1_time = s1.get("stage1_time_ms", 0.0)
        s2_time = s2.get("stage2_time_ms", 0.0)

        # 5. Build update payload
        update_fields = {
            "status": final_status,
            "stage1_label": s1["stage1_label"],
            "stage1_confidence": s1_conf,
            "is_distressed": is_distressed,
            "distress_types": s2.get("distress_types") or [],
            "severity": s2.get("severity") or "None",
            "description": s2.get("description") or "",
            "stage2_confidence": s2_conf,
            "needs_expert_review": final_status == "expert_review",
            "processing_time_ms": s0_time + s1_time + s2_time,
            "pavement_filter_decision": pavement["decision"],
            "pavement_filter_raw": pavement["raw"],
            "pavement_filter_at": datetime.now(timezone.utc).isoformat(),
            "raw_response": {
                "stage0_pavement_raw": pavement.get("raw", ""),
                "stage1_raw": s1.get("stage1_raw", ""),
                "stage2_raw": s2.get("stage2_raw", ""),
                "stage0_time_ms": s0_time,
                "stage1_time_ms": s1_time,
                "stage2_time_ms": s2_time,
                # Cascade telemetry: did Stage 2's first pass (with adapter)
                # fall back to running with the adapter disabled to recover
                # broader-taxonomy types? See app/model.py predict_stage2.
                "stage2_used_fallback": s2.get("stage2_used_fallback", False),
                "stage2_primary_raw": s2.get("stage2_primary_raw"),
            },
            "image_width": img_w,
            "image_height": img_h,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "error_message": None,  # clear any prior error
        }

        # 6. Persist to Supabase
        await update_assessment(self.supabase, image_id, update_fields)

        # 7. Update local metrics
        self.metrics.images_processed += 1
        if final_status == "classified":
            self.metrics.images_classified += 1
        elif final_status == "expert_review":
            self.metrics.images_flagged_review += 1
        self.metrics.record_timing(s0_time, s1_time, s2_time)
        self.metrics.last_processed_at = datetime.now(timezone.utc)
        self.metrics.last_processed_id = image_id
        self.metrics.last_error = None

    async def _mark_failed(self, image_id: str, error: str, attempts: int):
        """Mark an image as failed after exhausting retries."""
        try:
            await update_assessment(self.supabase, image_id, {
                "status": "failed",
                "error_message": error[:500],  # truncate to fit
                "retry_count": attempts,
                "processed_at": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as e:
            log.error("Failed to mark %s as failed (DB update failed): %s", image_id, e)

        self.metrics.images_failed += 1
        self.metrics.last_error = error
        self.metrics.current_image_id = None

    async def _release_claim(self, image_id: str):
        """Release a claim back to pending (for graceful stop mid-batch)."""
        try:
            await update_assessment(self.supabase, image_id, {
                "status": "pending",
                "claimed_at": None,
                "claimed_by": None,
            })
            log.info("Released claim on %s due to stop signal", image_id)
        except Exception as e:
            log.error("Failed to release claim on %s: %s", image_id, e)

    # -----------------------------------------------------------
    # Heartbeat
    # -----------------------------------------------------------

    async def _write_heartbeat(self, initial: bool = False, stopped: bool = False):
        """Write current state to the worker_state table."""
        now = datetime.now(timezone.utc)
        fields = {
            "is_running": self._is_running and not stopped,
            "last_heartbeat": now.isoformat(),
            "current_image_id": self.metrics.current_image_id,
            "images_processed": self.metrics.images_processed,
            "images_failed": self.metrics.images_failed,
            "images_flagged_review": self.metrics.images_flagged_review,
            "avg_stage1_time_ms": self.metrics.avg_stage1_ms,
            "avg_stage2_time_ms": self.metrics.avg_stage2_ms,
            "last_error": (self.metrics.last_error or "")[:500] if self.metrics.last_error else None,
            "last_processed_at": self.metrics.last_processed_at.isoformat() if self.metrics.last_processed_at else None,
            "updated_at": now.isoformat(),
        }
        if initial:
            fields["started_at"] = now.isoformat()
            fields["stopped_at"] = None
        if stopped:
            fields["stopped_at"] = now.isoformat()

        try:
            await upsert_worker_state(self.supabase, self.worker_id, fields)
            self.metrics.last_heartbeat_at = now
        except Exception as e:
            log.error("Heartbeat upsert failed: %s", e)


# ============================================================
# Module-level singleton (managed by FastAPI lifespan)
# ============================================================

_worker: Optional[PipelineWorker] = None


def get_worker() -> Optional[PipelineWorker]:
    """Return the singleton worker if it's been built (None if startup hasn't run)."""
    return _worker


def set_worker(worker: PipelineWorker) -> None:
    """Used by the FastAPI lifespan to register the singleton."""
    global _worker
    _worker = worker


def clear_worker() -> None:
    """Used by the FastAPI lifespan on shutdown."""
    global _worker
    _worker = None
