"""
Video pipeline service: drive-video session packages -> local store, with no manual step.

The GeoAI_Video capture app (Galaxy S21 on a vehicle) produces signed "session packages"
(video + frames.csv + gnss.csv + imu.csv + session.json + manifest.json). With the operator
toggle "video_pipeline" on, this service runs the whole chain for every new package dropped
into the inbox (GeoAI_Video docs/HANDOFF_PLAN.md section 8):

    queued       package found in VIDEO_INBOX, copy settled, validated with
                 GeoAI_Video tools/validate_session.py (a FAIL refuses it; the tool's output is kept)
    extracting   python -m geoai process <pkg> --out <runs_root> --mode segments
                 (best full frame per 5 m of road + distress check; DINOv2 on the GPU)
    classifying  python -m geoai vlm-export <run> --batch <batch>
                 pipeline/tools/run_vlm_batch.py <batch> --url <this server>   (resumable runner)
                 python -m geoai vlm-import <batch> <run>;  python -m geoai analyze <batch>
    storing      app.video_store.store_session() + verify_session()   (local Postgres + image store)
    stored       rows = records, every image path exists with its checksum
    cleaned      unused intermediates deleted; package, records, results and stored frames kept
    failed       with the error (and, for a refused package, the validator's output)

GPU sharing rule (the VLM team's, GeoAI_Video CLAUDE.md): nothing else may run on the GPU while
the VLM classifies. Extraction and classification of a session therefore run inside a "GPU
section": the PipelineWorker (citizen uploads) is put on hold, drained and stopped first - the
same drain the quantization reload in app/main.py does - and restarted afterwards on every path,
failures included. A start requested while the hold is on (an operator click, start.ps1
-AutoStart) is refused and remembered, and honoured when the hold is released. One session at a
time; the rest wait in FIFO order.

Disabling the toggle stops after the current item: the stage that is running finishes (for
classification, the current chunk of VLM_CLASSIFY_CHUNK images), the session keeps its state,
and the next enable resumes it from that stage. A server shutdown kills the running
subprocess instead; the same resume applies (extraction is redone only if its run folder is
incomplete; the runner is resumable by design).

Logging (HANDOFF_PLAN section 9): one JSON-lines file per session and day,
<VIDEO_LOG_ROOT>/<YYYY-MM-DD>/<session_id>.jsonl (stage, start/end, counts, errors with stack
traces), the full output of every external command beside it, plus the usual human log lines.

Configuration (environment; every path has a default):
    VIDEO_INBOX             C:\\GeoAI\\inbox     session packages (folders with manifest.json)
    VIDEO_RUNS_ROOT         C:\\GeoAI\\runs      run folders <runs_root>/<session_id>/
    VIDEO_BATCH_ROOT        C:\\GeoAI\\batches   temporary VLM batches (deleted after storing)
    VIDEO_LOG_ROOT          C:\\GeoAI\\logs
    GEOAI_VIDEO_REPO        <PROJECT_ROOT>/../GeoAI_Video
    GEOAI_VIDEO_PYTHON      <repo>/pipeline/.venv/Scripts/python.exe
    VIDEO_CLASSIFY_URL      http://127.0.0.1:<PORT or 8000>   this server, for the runner
    VIDEO_PIPELINE_ENABLED  false   toggle default; the operator's choice is persisted in
                                    <inbox>/.video_pipeline/state.json and wins over it
    VIDEO_EXTRACT_ARGS      extra arguments for `geoai process` (e.g. --allow-test)
    FFPROBE_DIR             newest C:\\GeoAI\\ffmpeg\\*\\bin holding ffprobe   prepended to the tools' PATH
                                    (validate_session.py needs ffprobe)
    VIDEO_KEEP_RUN_FRAMES   false   true keeps the run folder's copy of each stored best frame
    VIDEO_POLL_S 15, VIDEO_SETTLE_S 30, VIDEO_CLASSIFY_CHUNK 10
Best frames are stored by app/video_store.py in its image store (LOCAL_IMAGE_ROOT, default
C:\\GeoAI\\images), the root the dashboards are allowed to read.
The runner's API key is the first key of API_KEYS (if set); it is never logged.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import hashlib
import importlib
import json
import logging
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

log = logging.getLogger("video_pipeline")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Session states. The first seven are video_store's states; the local queue uses the same names.
STATES = ("queued", "extracting", "classifying", "storing", "stored", "cleaned", "failed")
RESUMABLE = ("extracting", "classifying", "storing", "stored")   # interrupted mid-way: resume first
STATE_FILE_SCHEMA = "capstone.video_pipeline_state.v1"

SENT_STATUSES = ("send", "audit")          # segment statuses with a best frame sent to the VLM
SEGMENT_STATUSES = ("send", "audit", "skip", "not_covered")

# Run-folder intermediates removed after a verified store (GeoAI_Video docs/pipeline/13 section 5;
# crop mode = documents 05-07). Everything else in the run folder is a record or a result.
INTERMEDIATE_DIRS = ("segment_views", "crops", "not_sent", "cache", ".cache", "__pycache__")
INTERMEDIATE_SUFFIXES = (".tmp", ".partial")

# Copied out of the batch into <run>/vlm/ before the batch is deleted: the VLM's answers and the
# analysis are results. images/ and records/ duplicate the run folder and are not copied.
BATCH_KEEP_FILES = ("batch.json", "requests.jsonl", "responses.jsonl", "vlm_run.json",
                    "import_summary.json", "vlm_results.csv", "vlm_results.geojson")
BATCH_KEEP_DIRS = ("analysis",)

# A 4xx answer other than these will not change on a retry (e.g. a rejected image): such requests
# are recorded as unanswered instead of blocking the session forever.
_RETRYABLE_4XX = (401, 403, 408, 429)

VALIDATION_OUTPUT_MAX = 64_000      # characters of validator output kept in the state file
ERROR_TAIL_CHARS = 4_000            # tail of a failed command's output quoted in the error
STALL_PASSES = 3                    # runner passes without progress before a session fails
DRAIN_TIMEOUT_S = 90.0              # same drain budget as the quantization reload in app/main.py

# Generous ceilings: they only catch a hung tool. A 4K drive session decodes at ~85 fps.
TIMEOUT_S = {"validate": 1800.0, "extract": 12 * 3600.0, "export": 3600.0,
             "classify": 4 * 3600.0, "import": 3600.0, "analyze": 3600.0}

_SECRET_ENV = re.compile(r"KEY|SECRET|TOKEN|PASSWORD|PASSWD|DSN|DATABASE_URL|SUPABASE|CLOUDINARY", re.I)


class StageError(RuntimeError):
    """A stage failed; the message is what the operator sees."""


class Interrupted(RuntimeError):
    """The server is shutting down: leave the session in its stage, resume on next start."""


# ============================================================
# Configuration
# ============================================================

def _env_path(name: str, default: Path | str | None) -> Optional[Path]:
    raw = os.environ.get(name, "").strip()
    if raw:
        return Path(raw)
    return Path(default) if default is not None else None


def _env_num(name: str, default: float, *, minimum: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        log.warning("%s=%r is not a number; using %s", name, raw, default)
        return default
    return max(minimum, value)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _default_ffprobe_dir() -> Optional[Path]:
    """FFPROBE_DIR, else the newest C:\\GeoAI\\ffmpeg\\*\\bin that holds ffprobe (None: rely on PATH)."""
    raw = os.environ.get("FFPROBE_DIR", "").strip()
    if raw:
        return Path(raw)
    exe = "ffprobe.exe" if os.name == "nt" else "ffprobe"
    found = sorted(p.parent for p in Path(r"C:\GeoAI\ffmpeg").glob(f"*/bin/{exe}"))
    return found[-1] if found else None


def _default_python(repo: Path) -> Path:
    venv = repo / "pipeline" / ".venv"
    return venv / "Scripts" / "python.exe" if os.name == "nt" else venv / "bin" / "python"


@dataclass
class VideoPipelineConfig:
    inbox: Path
    runs_root: Path
    batch_root: Path
    log_root: Path
    geoai_repo: Path
    python: Path
    classify_url: str
    image_root: Optional[Path] = None
    api_key: Optional[str] = field(default=None, repr=False)
    enabled_default: bool = False
    keep_run_frames: bool = False
    ffprobe_dir: Optional[Path] = None
    extract_args: list[str] = field(default_factory=list)
    poll_s: float = 15.0
    settle_s: float = 30.0
    classify_chunk: int = 10
    drain_timeout_s: float = DRAIN_TIMEOUT_S

    @classmethod
    def from_env(cls) -> "VideoPipelineConfig":
        repo = _env_path("GEOAI_VIDEO_REPO", PROJECT_ROOT.parent / "GeoAI_Video")
        port = os.environ.get("PORT", "").strip() or "8000"
        keys = [k.strip() for k in os.environ.get("API_KEYS", "").split(",") if k.strip()]
        raw_args = os.environ.get("VIDEO_EXTRACT_ARGS", "").strip()
        return cls(
            inbox=_env_path("VIDEO_INBOX", r"C:\GeoAI\inbox"),
            runs_root=_env_path("VIDEO_RUNS_ROOT", r"C:\GeoAI\runs"),
            batch_root=_env_path("VIDEO_BATCH_ROOT", r"C:\GeoAI\batches"),
            log_root=_env_path("VIDEO_LOG_ROOT", r"C:\GeoAI\logs"),
            image_root=None,   # video_store's own root (LOCAL_IMAGE_ROOT), which the dashboards read
            geoai_repo=repo,
            python=_env_path("GEOAI_VIDEO_PYTHON", _default_python(repo)),
            classify_url=(os.environ.get("VIDEO_CLASSIFY_URL", "").strip()
                          or f"http://127.0.0.1:{port}").rstrip("/"),
            api_key=keys[0] if keys else None,
            enabled_default=_env_bool("VIDEO_PIPELINE_ENABLED", False),
            keep_run_frames=_env_bool("VIDEO_KEEP_RUN_FRAMES", False),
            ffprobe_dir=_default_ffprobe_dir(),
            extract_args=shlex.split(raw_args, posix=os.name != "nt") if raw_args else [],
            poll_s=_env_num("VIDEO_POLL_S", 15.0, minimum=1.0),
            settle_s=_env_num("VIDEO_SETTLE_S", 30.0, minimum=0.0),
            classify_chunk=int(_env_num("VIDEO_CLASSIFY_CHUNK", 10, minimum=1)),
        )

    @property
    def state_file(self) -> Path:
        return self.inbox / ".video_pipeline" / "state.json"

    def public(self) -> dict:
        """Paths and tunables for the operator UI (no API key)."""
        return {"inbox": str(self.inbox), "runs_root": str(self.runs_root), "batch_root": str(self.batch_root),
                "log_root": str(self.log_root), "image_root": str(self.image_root) if self.image_root else None,
                "geoai_repo": str(self.geoai_repo), "python": str(self.python), "classify_url": self.classify_url,
                "api_key_set": bool(self.api_key), "keep_run_frames": self.keep_run_frames,
                "ffprobe_dir": str(self.ffprobe_dir) if self.ffprobe_dir else None, "extract_args": self.extract_args, "poll_s": self.poll_s,
                "settle_s": self.settle_s, "classify_chunk": self.classify_chunk}


# ============================================================
# Small helpers (pure, unit-tested)
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def safe_name(text: str) -> str:
    """A file-name-safe form of a session id (ids are already safe; this guards odd folder names)."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", text) or "_"


def percentile(values: list[float], q: float) -> Optional[float]:
    """q-th percentile with linear interpolation between order statistics (numpy's default)."""
    xs = sorted(values)
    if not xs:
        return None
    pos = (len(xs) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def dir_bytes(path: Path) -> int:
    """Bytes of all files under path (0 if it does not exist). Symlinks are not followed."""
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    stack = [path]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(Path(e.path))
                        elif e.is_file(follow_symlinks=False):
                            total += e.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total


def fingerprint(path: Path) -> list:
    """[files, bytes, newest mtime ns] of a package: changes while a copy is still running."""
    n = size = newest = 0
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in files:
            try:
                st = os.stat(os.path.join(root, name))
            except OSError:
                continue
            n += 1
            size += st.st_size
            newest = max(newest, st.st_mtime_ns)
    return [n, size, newest]


def read_jsonl(path: Path) -> list[dict]:
    """Parsed lines of a JSON-lines file; unreadable lines (a cut-off last line) are skipped."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def request_ids(batch_dir: Path) -> list[str]:
    return [r["request_id"] for r in read_jsonl(batch_dir / "requests.jsonl") if r.get("request_id")]


def answers_from_responses(lines: list[dict]) -> dict[str, dict]:
    """{request_id: /classify JSON} from responses.jsonl lines without an error (the last such line wins)."""
    out: dict[str, dict] = {}
    for r in lines:
        rid = r.get("request_id")
        if rid and not r.get("error") and isinstance(r.get("response"), dict):
            out[rid] = r["response"]
    return out


def permanently_rejected(lines: list[dict], answered: set[str]) -> set[str]:
    """Unanswered requests whose last line is a 4xx a retry will not change (e.g. a rejected image)."""
    last: dict[str, dict] = {}
    for r in lines:
        if r.get("request_id"):
            last[r["request_id"]] = r
    out = set()
    for rid, r in last.items():
        code = r.get("http_status")
        if rid not in answered and r.get("error") and isinstance(code, int) \
                and 400 <= code < 500 and code not in _RETRYABLE_4XX:
            out.add(rid)
    return out


def vlm_metrics(lines: list[dict], ids: list[str]) -> dict:
    """VLM time per image and error counts from responses.jsonl (the runner's own records).

    seconds = the successful request's round trip per image. retries = every HTTP attempt beyond
    the first per request (runner retries inside a line, plus lines re-sent by later passes).
    """
    want = set(ids)
    lines = [r for r in lines if r.get("request_id") in want]
    ok_seconds: dict[str, float] = {}
    for r in lines:
        if not r.get("error") and isinstance(r.get("response"), dict) and isinstance(r.get("seconds"), (int, float)):
            ok_seconds[r["request_id"]] = float(r["seconds"])
    attempts = sum(max(1, int(r.get("attempts") or 1)) for r in lines)
    with_lines = len({r["request_id"] for r in lines})
    secs = list(ok_seconds.values())
    answered = set(answers_from_responses(lines))
    return {
        "vlm_requests": len(want),
        "vlm_answered": len(answered),
        "vlm_unanswered": len(want - answered),
        "vlm_error_lines": sum(1 for r in lines if r.get("error")),
        "vlm_http_attempts": attempts,
        "vlm_retries": attempts - with_lines,
        "vlm_s_median": _round(percentile(secs, 50), 3),
        "vlm_s_p95": _round(percentile(secs, 95), 3),
        "vlm_s_mean": _round(sum(secs) / len(secs), 3) if secs else None,
        "vlm_s_total": _round(sum(secs), 3),
    }


# ============================================================
# Live view of the session in progress (operator page, like the citizen-upload card)
# ============================================================
PROGRESS_TAIL_BYTES = 64_000          # the extraction's output tail searched for its last progress line
_PROGRESS_LINE = re.compile(r"progress (\{.*\})\s*$")
_SEGMENT_ID = re.compile(r"^(?P<sid>.+)_seg(?P<k>\d{5})$")


def extract_progress(output_file: Optional[Path]) -> Optional[dict]:
    """The last `progress {json}` line `geoai process` logged (GeoAI_Video frames._Progress), or None.

    Phases: plan, bank, bank_finalize, decode, outputs; `done`/`total` count bank frames (total an
    estimate) or segments. Only the file's tail is read: a long run's log is megabytes.
    """
    if not output_file:
        return None
    p = Path(output_file)
    try:
        with open(p, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - PROGRESS_TAIL_BYTES))
            text = f.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    for line in reversed(text.splitlines()):
        m = _PROGRESS_LINE.search(line)
        if m:
            try:
                d = json.loads(m.group(1))
            except ValueError:
                continue
            if isinstance(d, dict) and d.get("phase"):
                return d
    return None


def latest_frame_id(run_dir: Path) -> Optional[str]:
    """Segment id of the newest best frame written to <run>/frames (the extraction's latest sent segment)."""
    best, newest = None, -1
    try:
        it = os.scandir(Path(run_dir) / "frames")
    except OSError:
        return None
    with it:
        for e in it:
            if e.name.endswith(".png") and e.is_file():
                try:
                    t = e.stat().st_mtime_ns
                except OSError:
                    continue
                if t > newest or (t == newest and e.name > best):
                    best, newest = e.name, t
    return best[:-4] if best else None


def _record_summary(batch_dir: Path, rid: str) -> Optional[dict]:
    """Where a segment is (from the batch's copy of its geoai.segment.v1 record), for the live card."""
    rec = _read_json(Path(batch_dir) / "records" / f"{rid}.json")
    if not rec:
        return None
    seg, gate = rec.get("segment") or {}, rec.get("gate") or {}
    centre = seg.get("centre") or {}
    frame = rec.get("frame") or {}
    return {"segment_id": rid, "k": seg.get("k"), "s0_m": seg.get("s0_m"), "length_m": seg.get("length_m"),
            "lat": centre.get("lat"), "lon": centre.get("lon"), "chainage_start_km": seg.get("chainage_start_km"),
            "utc_iso": frame.get("utc_iso"), "status": rec.get("status"),
            "gate_decision_if_gated": gate.get("decision_if_gated"), "gate_reasons": gate.get("reasons")}


def _answer_summary(line: dict) -> dict:
    resp = line.get("response") or {}
    types = resp.get("distress_types") or []
    labels = [str(t) for t in types] if isinstance(types, list) else [str(types)]
    if not labels and str(resp.get("stage1_label") or "").lower() == "normal":
        labels = ["Normal"]  # Stage 1 Normal: Stage 2 never ran (GeoAI_Video vlm._labels reads it the same way)
    return {"segment_id": line.get("request_id"), "labels": labels, "stage1_label": resp.get("stage1_label"),
            "stage1_confidence": resp.get("stage1_confidence"), "severity": resp.get("severity"),
            "stage2_confidence": resp.get("stage2_confidence"), "needs_expert_review": resp.get("needs_expert_review"),
            "seconds": line.get("seconds"), "started_utc": line.get("started_utc")}


def classify_live(batch_dir: Path) -> Optional[dict]:
    """The classification in progress: counts, the segment the runner is sending now, the last answer.

    The runner sends requests in requests.jsonl order and appends one line per image, so the segment
    in flight is the first one neither answered nor permanently rejected; it has been in flight since
    responses.jsonl was last written (or since the batch was exported, before the first answer).
    s/image and the ETA come from the answered images' own round trips.
    """
    b = Path(batch_dir)
    ids = request_ids(b)
    if not ids:
        return None
    lines = read_jsonl(b / "responses.jsonl")
    answered = set(answers_from_responses(lines)) & set(ids)
    rejected = permanently_rejected(lines, answered)
    pending = [rid for rid in ids if rid not in answered and rid not in rejected]
    secs = [float(r["seconds"]) for r in lines
            if r.get("request_id") in answered and not r.get("error") and isinstance(r.get("seconds"), (int, float))]
    mean = sum(secs) / len(secs) if secs else None
    last_ok = next((r for r in reversed(lines) if not r.get("error") and isinstance(r.get("response"), dict)), None)
    last_err = lines[-1] if lines and lines[-1].get("error") else None
    resp_file = b / "responses.jsonl"
    try:
        since = datetime.fromtimestamp((resp_file if resp_file.exists() else b / "requests.jsonl").stat().st_mtime,
                                       timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    except OSError:
        since = None
    in_flight = pending[0] if pending else None
    return {"requests": len(ids), "answered": len(answered), "rejected": len(rejected), "remaining": len(pending),
            "s_per_image_mean": _round(mean, 2), "s_per_image_median": _round(percentile(secs, 50), 2),
            "eta_s": _round(mean * len(pending), 0) if mean is not None else None,
            "in_flight": _record_summary(b, in_flight) if in_flight else None, "in_flight_since_utc": since,
            "last_answer": _answer_summary(last_ok) if last_ok else None,
            "last_error": {"segment_id": last_err.get("request_id"), "error": str(last_err.get("error"))[:300]}
            if last_err else None}


def _round(x: Optional[float], nd: int) -> Optional[float]:
    return None if x is None else round(x, nd)


def load_sidecars(run_dir: Path) -> list[dict]:
    recs = []
    for p in sorted((run_dir / "sidecars").glob("*.json")):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise StageError(f"unreadable segment record {p.name}: {e}") from e
        if not isinstance(rec, dict):
            raise StageError(f"segment record {p.name} is not a JSON object")
        recs.append(rec)
    return recs


def run_status(run_dir: Path) -> tuple[bool, str]:
    """(complete, why not) of a segment run folder.

    `geoai process` deletes the run folder when it starts and writes report.json last, so a
    report.json with mode "segments", records in sidecars/ and the frame of every send/audit
    record on disk mean the extraction finished.
    """
    rep = run_dir / "report.json"
    if not rep.exists():
        return False, "no report.json"
    try:
        report = json.loads(rep.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return False, f"report.json unreadable: {e}"
    if not isinstance(report, dict) or report.get("mode") != "segments":
        return False, f"report.json mode is {report.get('mode') if isinstance(report, dict) else None!r}, not 'segments'"
    try:
        recs = load_sidecars(run_dir)
    except StageError as e:
        return False, str(e)
    if not recs:
        return False, "no segment records in sidecars/"
    missing = [r.get("id") for r in recs if r.get("status") in SENT_STATUSES
               and not (run_dir / str((r.get("frame") or {}).get("file") or "")).is_file()]
    if missing:
        return False, f"{len(missing)} send/audit frames missing, e.g. {missing[0]}"
    return True, ""


def batch_complete(batch_dir: Path) -> bool:
    """vlm-export writes SHA256SUMS after everything else; all three present = a finished export."""
    return all((batch_dir / n).is_file() for n in ("batch.json", "requests.jsonl", "SHA256SUMS"))


def segment_metrics(run_dir: Path) -> dict:
    """Route and coverage metres and status counts, from the segment records (all statuses)."""
    recs = load_sidecars(run_dir)
    counts = {s: 0 for s in SEGMENT_STATUSES}
    route = covered = 0.0
    low_light = 0
    for r in recs:
        status = r.get("status")
        if status not in counts:
            raise StageError(f"segment record {r.get('id')} has unknown status {status!r}")
        try:
            length = float((r.get("segment") or {}).get("length_m"))
        except (TypeError, ValueError) as e:
            raise StageError(f"segment record {r.get('id')} has no segment.length_m") from e
        counts[status] += 1
        route += length
        if status != "not_covered":
            covered += length
            flags = r.get("flags") or []  # contract: list of flag names; tolerate a dict too
            flagged = flags.get("low_light") if isinstance(flags, dict) else "low_light" in flags
            low_light += bool(flagged or (r.get("capture") or {}).get("low_light"))
    n_cov = len(recs) - counts["not_covered"]
    return {"segments": len(recs), "route_m": round(route, 2), "covered_m": round(covered, 2),
            "not_covered_m": round(route - covered, 2),
            "coverage_pct": round(100.0 * covered / route, 2) if route > 0 else None,
            "send": counts["send"], "audit": counts["audit"], "skip": counts["skip"],
            "not_covered": counts["not_covered"],
            "low_light_share": round(low_light / n_cov, 4) if n_cov else None}


def _json_safe(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=str))


def _count(v: Any) -> int:
    if v is None:
        return 0
    if isinstance(v, (list, tuple, set, dict)):
        return len(v)
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def rmtree(path: Path) -> None:
    """shutil.rmtree that also removes read-only files (Windows marks some copied files read-only)."""
    import stat

    def clear_and_retry(func, p, _exc):
        os.chmod(p, stat.S_IWRITE)
        func(p)

    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=clear_and_retry)
    else:
        shutil.rmtree(path, onerror=clear_and_retry)


def _tail(path: Path, chars: int = ERROR_TAIL_CHARS) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-chars:].strip()


def child_env(extra_path: Optional[list] = None) -> dict:
    """The environment for GeoAI_Video subprocesses: this server's secrets are not passed on;
    `extra_path` folders (ffprobe) go in front of PATH."""
    env = {k: v for k, v in os.environ.items() if not _SECRET_ENV.search(k)}
    if extra_path:
        key = next((k for k in env if k.upper() == "PATH"), "PATH")
        env[key] = os.pathsep.join([str(p) for p in extra_path] + ([env[key]] if env.get(key) else []))
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def redact(argv: list[str]) -> list[str]:
    out, hide = [], False
    for a in argv:
        out.append("***" if hide else a)
        hide = a == "--api-key"
    return out


# ============================================================
# External commands
# ============================================================

# run_command(step, argv, cwd, out_file, timeout_s, cancel) -> exit code. Blocking; the service
# calls it through asyncio.to_thread. Tests substitute a fake with the same signature.
RunCommand = Callable[[str, list, Path, Path, Optional[float], threading.Event], int]


def run_process(step: str, argv: list, cwd: Path, out_file: Path, timeout_s: Optional[float],
                cancel: threading.Event, *, extra_path: Optional[list] = None) -> int:
    """Run one command, stdout+stderr to out_file; kill it when `cancel` is set (server shutdown).

    Returns the exit code. Raises Interrupted when cancelled and StageError on a timeout.
    """
    if cancel.is_set():
        raise Interrupted(f"{step} not started: server shutting down")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    with open(out_file, "ab") as f:
        f.write(f"# {utc_now()} {step}: {subprocess.list2cmdline(redact([str(a) for a in argv]))}\n"
                f"# cwd {cwd}\n".encode("utf-8"))
        f.flush()
        proc = subprocess.Popen([str(a) for a in argv], cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, env=child_env(extra_path), creationflags=flags)
        t0 = time.monotonic()
        while True:
            try:
                return proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass
            reason = None
            if cancel.is_set():
                reason = "cancelled"
            elif timeout_s is not None and time.monotonic() - t0 > timeout_s:
                reason = "timeout"
            if reason:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=15)
                f.write(f"\n# {utc_now()} {step}: killed ({reason})\n".encode("utf-8"))
                if reason == "cancelled":
                    raise Interrupted(f"{step} stopped: server shutting down")
                raise StageError(f"{step} did not finish within {timeout_s / 3600:.1f} h; killed")


# ============================================================
# Per-session JSON-lines log
# ============================================================

class SessionLog:
    """<root>/<YYYY-MM-DD>/<session>.jsonl, one event per line; command output files beside it."""

    def __init__(self, root: Path, session_id: str):
        self.root = Path(root)
        self.name = safe_name(session_id)
        self.session_id = session_id
        self._lock = threading.Lock()

    def _day_dir(self) -> Path:
        return self.root / datetime.now().strftime("%Y-%m-%d")

    def path(self) -> Path:
        return self._day_dir() / f"{self.name}.jsonl"

    def output_file(self, step: str) -> Path:
        return self._day_dir() / f"{self.name}.{step}.txt"

    def write(self, stage: str, event: str, **fields: Any) -> None:
        line = {"ts": utc_now(), "session_id": self.session_id, "stage": stage, "event": event, **fields}
        try:
            p = self.path()
            p.parent.mkdir(parents=True, exist_ok=True)
            data = json.dumps(_json_safe(line), ensure_ascii=False) + "\n"
            with self._lock, open(p, "a", encoding="utf-8", newline="\n") as f:
                f.write(data)
        except OSError as e:  # a log that cannot be written must not stop the pipeline
            log.warning("cannot write session log for %s: %s", self.session_id, e)

    @staticmethod
    def read(root: Path, session_id: str, limit: int = 500) -> list[dict]:
        """Events of a session across all days, oldest first (the last `limit`)."""
        name = safe_name(session_id)
        files = sorted(Path(root).glob(f"*/{name}.jsonl")) if Path(root).exists() else []
        events: list[dict] = []
        for f in files:
            events.extend(read_jsonl(f))
        return events[-limit:]


def _say(msg: str, level: int = logging.INFO) -> None:
    """Human log line: through logging, or printed like app/main.py's [API] lines when logging
    has no handler (uvicorn configures only its own loggers)."""
    if log.hasHandlers():
        log.log(level, msg)
    else:
        print(f"[VIDEO] {msg}", flush=True)


# ============================================================
# The service
# ============================================================

class VideoPipelineService:
    """Inbox watcher + one-session-at-a-time state machine. All state lives on the event loop;
    only filesystem work and subprocesses run in threads."""

    def __init__(self, cfg: Optional[VideoPipelineConfig] = None, *, store: Any = None,
                 run_command: Optional[RunCommand] = None,
                 get_worker: Optional[Callable[[], Any]] = None):
        self.cfg = cfg or VideoPipelineConfig.from_env()
        self._store = store
        self._run_command = run_command or functools.partial(
            run_process, extra_path=[self.cfg.ffprobe_dir] if self.cfg.ffprobe_dir else None)
        if get_worker is None:
            from app.worker import get_worker as _gw
            get_worker = _gw
        self._get_worker = get_worker
        self.cancel = threading.Event()             # set on shutdown: kills the running command
        self._wake: Optional[asyncio.Event] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._task: Optional[asyncio.Task] = None
        self._shutting_down = False
        self._current: Optional[str] = None         # key of the session being processed
        self._gpu_section_for: Optional[str] = None
        self._last_error: Optional[str] = None
        self._persist_error: Optional[str] = None
        self._sessions: dict[str, dict] = {}
        self._enabled = self.cfg.enabled_default
        self._enabled_source = "env VIDEO_PIPELINE_ENABLED"
        self._load_state()

    # ----------------------------------------------------------- persistence
    def _load_state(self) -> None:
        p = self.cfg.state_file
        if not p.exists():
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            _say(f"state file {p} unreadable ({e}); starting with an empty queue", logging.WARNING)
            return
        if isinstance(data.get("enabled"), bool):
            self._enabled = data["enabled"]
            self._enabled_source = str(p)
        sessions = data.get("sessions")
        if isinstance(sessions, dict):
            self._sessions = {k: v for k, v in sessions.items() if isinstance(v, dict) and v.get("state") in STATES}

    def _save(self) -> None:
        p = self.cfg.state_file
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(p.name + ".tmp")
            tmp.write_text(json.dumps({"schema": STATE_FILE_SCHEMA, "enabled": self._enabled,
                                       "updated_utc": utc_now(), "sessions": _json_safe(self._sessions)},
                                      indent=1), encoding="utf-8")
            os.replace(tmp, p)
            self._persist_error = None
        except OSError as e:
            if self._persist_error != str(e):
                _say(f"cannot write {p}: {e} (state kept in memory only)", logging.WARNING)
            self._persist_error = str(e)

    # ----------------------------------------------------------- control
    async def start(self) -> None:
        """Start the background loop (idle while the toggle is off)."""
        if self._task and not self._task.done():
            return
        self._loop = asyncio.get_running_loop()
        self._wake = asyncio.Event()
        self._shutting_down = False
        self.cancel.clear()
        self._task = asyncio.create_task(self._run_loop(), name="video_pipeline")
        _say(f"video pipeline service started (enabled={self._enabled}, inbox={self.cfg.inbox})")

    async def stop(self, timeout_s: float = 30.0) -> None:
        """Server shutdown: kill the running command, leave the session in its stage, save state."""
        self._shutting_down = True
        self.cancel.set()
        self._wakeup()
        if self._task:
            try:
                await asyncio.wait_for(asyncio.shield(self._task), timeout=timeout_s)
            except asyncio.TimeoutError:
                self._task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await self._task
            except Exception:
                pass
        self._save()
        _say("video pipeline service stopped")

    def enable(self) -> dict:
        self._enabled = True
        self._enabled_source = str(self.cfg.state_file)
        self._save()
        self._wakeup()
        _say("video_pipeline enabled")
        return self.status_summary()

    def disable(self) -> dict:
        """Stop after the current item; the session in progress keeps its stage and resumes on enable."""
        self._enabled = False
        self._enabled_source = str(self.cfg.state_file)
        self._save()
        self._wakeup()
        _say("video_pipeline disabled" + (f"; stopping after the current stage of {self._current}"
                                          if self._current else ""))
        return self.status_summary()

    def retry(self, key_or_id: str) -> dict:
        """Put a failed session back in the queue; it resumes from what is already on disk."""
        key = self._find(key_or_id)
        if key is None:
            raise KeyError(key_or_id)
        s = self._sessions[key]
        if s["state"] != "failed":
            raise ValueError(f"session {key_or_id} is {s['state']}, only failed sessions can be retried")
        resume = s.get("failed_from") if s.get("failed_from") in RESUMABLE else "queued"
        if s.get("error_stage") == "validate":
            resume = "queued"
            s.pop("validation", None)
        self._transition(s, resume, detail="retry requested", error=None)
        self._wakeup()
        return dict(s)

    def _wakeup(self) -> None:
        if self._wake is None or self._loop is None:
            return
        try:
            self._loop.call_soon_threadsafe(self._wake.set)
        except RuntimeError:
            pass

    def _pause_requested(self) -> bool:
        return self._shutting_down or not self._enabled

    # ----------------------------------------------------------- loop
    async def _run_loop(self) -> None:
        backoff = self.cfg.poll_s
        while not self._shutting_down:
            try:
                worked = await self.tick()
                backoff = self.cfg.poll_s
                if worked:
                    continue
            except Exception as e:  # the loop must survive anything; the error is shown on the UI
                self._last_error = f"{type(e).__name__}: {e}"
                _say(f"video pipeline loop error: {self._last_error}\n{traceback.format_exc()}", logging.ERROR)
                backoff = min(300.0, backoff * 2)
            if self._shutting_down:
                break
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=backoff)
            except asyncio.TimeoutError:
                pass
            self._wake.clear()

    async def tick(self) -> bool:
        """One scheduling step: scan the inbox, then process the next session (to completion, or
        until disabled or shut down). Returns True if a session was worked on."""
        if self._pause_requested():
            return False
        found = await asyncio.to_thread(self._scan_inbox)
        self._merge_scan(found)
        key = self._next_session()
        if key is None:
            return False
        problem = self._tools_problem()
        if problem:
            # Not the package's fault: keep it queued and say why on the UI.
            if self._last_error != problem:
                _say(problem, logging.ERROR)
            self._last_error = problem
            return False
        self._current = key
        try:
            await self._process(key)
        finally:
            self._current = None
            self._save()
        return True

    # ----------------------------------------------------------- inbox
    def _scan_inbox(self) -> list[dict]:
        inbox = self.cfg.inbox
        if not inbox.is_dir():
            return []
        out = []
        for d in sorted(inbox.iterdir(), key=lambda p: p.name):
            if not d.is_dir() or d.name.startswith((".", "_")) or not (d / "manifest.json").is_file():
                continue
            sid = d.name
            try:
                meta = json.loads((d / "session.json").read_text(encoding="utf-8"))
                if isinstance(meta, dict) and meta.get("session_id"):
                    sid = str(meta["session_id"])
            except (OSError, ValueError):
                pass   # the validator reports an unreadable session.json
            out.append({"key": d.name, "path": str(d), "session_id": sid, "fingerprint": fingerprint(d)})
        return out

    def _merge_scan(self, found: list[dict]) -> None:
        now = time.time()
        changed = False
        seen = set()
        for f in found:
            key = f["key"]
            seen.add(key)
            s = self._sessions.get(key)
            if s is None:
                self._sessions[key] = s = {
                    "key": key, "session_id": f["session_id"], "package_dir": f["path"], "state": "queued",
                    "detail": "waiting for the copy to settle", "error": None, "fingerprint": f["fingerprint"],
                    "stable_since": now, "first_seen_utc": utc_now(), "updated_utc": utc_now(),
                    "history": [{"state": "queued", "utc": utc_now()}], "metrics": {}}
                _say(f"new session package {key} ({f['session_id']})")
                changed = True
                continue
            if f["fingerprint"] != s.get("fingerprint"):
                s["fingerprint"] = f["fingerprint"]
                s["stable_since"] = now
                changed = True
                if s["state"] == "failed" and s.get("error_stage") == "validate":
                    # A refused package was replaced or completed: validate it again.
                    s["session_id"] = f["session_id"]
                    s.pop("validation", None)
                    self._transition(s, "queued", detail="package changed since it was refused", error=None)
        for key, s in self._sessions.items():
            if key not in seen and s["state"] in ("queued",) + RESUMABLE[:1] and not Path(s["package_dir"]).exists():
                self._fail(s, "inbox", "the session package is no longer in the inbox")
                changed = True
        if changed:
            self._save()

    def _next_session(self) -> Optional[str]:
        def order(k: str):
            s = self._sessions[k]
            return (s.get("first_seen_utc") or "", k)

        resumable = sorted((k for k, s in self._sessions.items() if s["state"] in RESUMABLE), key=order)
        if resumable:
            return resumable[0]
        now = time.time()
        ready = sorted((k for k, s in self._sessions.items() if s["state"] == "queued"
                        and now - float(s.get("stable_since") or 0) >= self.cfg.settle_s), key=order)
        return ready[0] if ready else None

    def _find(self, key_or_id: str) -> Optional[str]:
        if key_or_id in self._sessions:
            return key_or_id
        for k, s in self._sessions.items():
            if s.get("session_id") == key_or_id:
                return k
        return None

    # ----------------------------------------------------------- state
    def _transition(self, s: dict, state: str, *, detail: Optional[str] = None, error: Any = "keep") -> None:
        if s["state"] != state:
            s.setdefault("history", []).append({"state": state, "utc": utc_now()})
            _say(f"{s['session_id']}: {s['state']} -> {state}" + (f" ({detail})" if detail else ""))
        s["state"] = state
        s["detail"] = detail
        if error != "keep":
            s["error"] = error
            if error is None:
                s.pop("error_stage", None)
                s.pop("failed_from", None)
        s["updated_utc"] = utc_now()
        self._save()

    def _fail(self, s: dict, stage: str, error: str) -> None:
        if s["state"] in RESUMABLE:
            s["failed_from"] = s["state"]
        s["error_stage"] = stage
        self._transition(s, "failed", detail=f"failed in {stage}", error=error)

    async def _db_state(self, s: dict, state: str, **kw: Any) -> None:
        """Mirror the state into video_store (best effort: the local queue is authoritative)."""
        try:
            store = self._store_module()
            sid = s["session_id"]

            def call():
                conn = store.connect()
                try:
                    store.set_session_state(conn, sid, state, **kw)
                finally:
                    _close(conn)

            await asyncio.to_thread(call)
            s.pop("db_warning", None)
        except Exception as e:
            s["db_warning"] = f"session state not written to the database: {type(e).__name__}: {e}"
            _say(f"{s['session_id']}: {s['db_warning']}", logging.WARNING)

    async def _set_state(self, s: dict, state: str, detail: Optional[str] = None, **db: Any) -> None:
        self._transition(s, state, detail=detail)
        await self._db_state(s, state, metrics=s.get("metrics") or None, **db)

    def _store_module(self):
        if self._store is None:
            try:
                self._store = importlib.import_module("app.video_store")
            except Exception as e:
                raise StageError(f"app/video_store.py is not available: {type(e).__name__}: {e}") from e
        return self._store

    # ----------------------------------------------------------- processing
    async def _process(self, key: str) -> None:
        s = self._sessions[key]
        sid = s["session_id"]
        run = self.cfg.runs_root / sid
        batch = self.cfg.batch_root / f"batch_{safe_name(sid)}"
        s["run_dir"], s["batch_dir"] = str(run), str(batch)
        s.setdefault("metrics", {})
        s["started_utc"] = s.get("started_utc") or utc_now()
        jl = SessionLog(self.cfg.log_root, sid)
        stage = s["state"]
        jl.write("session", "resume" if stage in RESUMABLE else "pick", state=stage, package=s["package_dir"])
        try:
            if s["state"] == "queued":
                stage = "validate"
                if not await self._validate(s, jl):
                    return
                if self._pause_requested():
                    self._transition(s, "queued", detail="validated; waiting for enable")
                    return
            if s["state"] in ("queued", "extracting", "classifying"):
                need_extract = s["state"] != "classifying" and not (await asyncio.to_thread(run_status, run))[0]
                # A new extraction invalidates any earlier batch (its frames' checksums change).
                need_classify = need_extract or not await asyncio.to_thread(self._classify_done, batch)
                gpu = self._gpu_section(s, jl) if need_extract or need_classify else contextlib.nullcontext()
                async with gpu:
                    if s["state"] in ("queued", "extracting"):
                        stage = "extract"
                        await self._set_state(s, "extracting")
                        await self._extract(s, jl, run, batch, skip=not need_extract)
                        if self._pause_requested():
                            self._transition(s, "extracting", detail="paused after extraction")
                            return
                    stage = "classify"
                    if s["state"] != "classifying":
                        await self._set_state(s, "classifying")
                    if need_classify and not await self._classify(s, jl, run, batch):
                        self._transition(s, "classifying", detail="paused; resumes on enable")
                        return
                stage = "import"
                await self._import_and_analyze(s, jl, run, batch)
                if self._pause_requested():
                    self._transition(s, "classifying", detail="paused after import and analysis")
                    return
            if s["state"] in ("classifying", "storing"):
                stage = "store"
                await self._set_state(s, "storing")
                await self._store_and_verify(s, jl, run)
                if self._pause_requested():
                    return
            if s["state"] == "stored":
                stage = "cleanup"
                await self._cleanup(s, jl, run, batch)
            s["finished_utc"] = utc_now()
            jl.write("session", "done", state=s["state"], metrics=s.get("metrics"))
        except Interrupted as e:
            s["detail"] = f"interrupted during {stage}; resumes on restart"
            jl.write(stage, "interrupted", error=str(e))
            _say(f"{sid}: interrupted during {stage}")
        except Exception as e:
            msg = str(e) if isinstance(e, StageError) else f"{type(e).__name__}: {e}"
            jl.write(stage, "failed", error=msg, traceback=traceback.format_exc())
            self._fail(s, stage, msg)
            self._last_error = f"{sid}: {msg}"
            await self._db_state(s, "failed", error=msg, metrics=s.get("metrics") or None)
            _say(f"{sid}: FAILED in {stage}: {msg}", logging.ERROR)

    @contextlib.asynccontextmanager
    async def _stage(self, s: dict, jl: SessionLog, stage: str, **start_fields: Any):
        """Time a stage, log start/end (or the error with its stack trace) to the JSON-lines log."""
        t0, start = time.monotonic(), utc_now()
        jl.write(stage, "start", **start_fields)
        s["stage"], s["stage_started_utc"] = stage, start   # the operator page's live stage timer
        counts: dict = {}
        try:
            yield counts
        except Interrupted:
            raise
        except Exception as e:
            jl.write(stage, "error", start_utc=start, end_utc=utc_now(), seconds=round(time.monotonic() - t0, 3),
                     error=str(e), traceback=traceback.format_exc(), counts=counts)
            raise
        finally:
            dt = time.monotonic() - t0
            timings = s["metrics"].setdefault("timings_s", {})
            timings[stage] = round(timings.get(stage, 0.0) + dt, 3)
            s["stage"] = None
        jl.write(stage, "end", start_utc=start, end_utc=utc_now(), seconds=round(dt, 3), counts=counts)

    async def _cmd(self, s: dict, jl: SessionLog, step: str, argv: list, cwd: Path) -> tuple[int, Path]:
        out = jl.output_file(step)
        s["live_output"] = {"step": step, "file": str(out)}   # tailed by status() for extraction progress
        timeout = TIMEOUT_S.get(step.split("_")[0])
        code = await asyncio.to_thread(self._run_command, step, argv, cwd, out, timeout, self.cancel)
        if self._shutting_down:
            raise Interrupted(f"{step} stopped: server shutting down")
        return code, out

    @property
    def _pipeline_dir(self) -> Path:
        return self.cfg.geoai_repo / "pipeline"

    def _geoai(self, *args: str) -> list[str]:
        return [str(self.cfg.python), "-m", "geoai", *args]

    def _tools_problem(self) -> Optional[str]:
        if not self.cfg.geoai_repo.is_dir():
            return f"GeoAI_Video repository not found at {self.cfg.geoai_repo} (set GEOAI_VIDEO_REPO)"
        if not self.cfg.python.is_file():
            return f"pipeline Python not found at {self.cfg.python} (set GEOAI_VIDEO_PYTHON)"
        return None

    def _check_tools(self) -> None:
        problem = self._tools_problem()
        if problem:
            raise StageError(problem)

    # --- validate
    async def _validate(self, s: dict, jl: SessionLog) -> bool:
        pkg = Path(s["package_dir"])
        dup = [o for k, o in self._sessions.items() if k != s["key"] and o.get("session_id") == s["session_id"]
               and o["state"] != "failed"]
        if dup:
            raise StageError(f"another package ({dup[0]['key']}) has the same session_id {s['session_id']}")
        v = s.get("validation") or {}
        if v.get("exit_code") == 0 and v.get("fingerprint") == s.get("fingerprint"):
            jl.write("validate", "skip", reason="already validated, package unchanged")
            return True
        self._transition(s, "queued", detail="validating")
        async with self._stage(s, jl, "validate", package=str(pkg)) as counts:
            self._check_tools()
            code, out = await self._cmd(s, jl, "validate",
                                        [str(self.cfg.python), str(self.cfg.geoai_repo / "tools" / "validate_session.py"),
                                         str(pkg)], self.cfg.geoai_repo)
            text = _tail(out, VALIDATION_OUTPUT_MAX)
            fails = [ln.strip() for ln in text.splitlines() if ln.startswith("FAIL")]
            counts.update(exit_code=code, fail_lines=len(fails))
            s["validation"] = {"exit_code": code, "utc": utc_now(), "fingerprint": s.get("fingerprint"),
                               "output": text, "output_file": str(out)}
        if code == 0:
            self._transition(s, "queued", detail="validated")
            await self._db_state(s, "queued")
            return True
        why ="; ".join(fails[:5]) if fails else (text.splitlines()[-1] if text else f"exit code {code}")
        what = "cannot read the session" if code == 2 else "validation failed"
        msg = f"refused: {what} (validate_session.py exit {code}): {why}"
        jl.write("validate", "refused", error=msg, exit_code=code)
        self._fail(s, "validate", msg)
        await self._db_state(s, "failed", error=msg)
        _say(f"{s['session_id']}: {msg}", logging.WARNING)
        return False

    # --- GPU section
    @contextlib.asynccontextmanager
    async def _gpu_section(self, s: dict, jl: SessionLog):
        """Hold, drain and stop the PipelineWorker for extraction/classification; restart it after.

        Mirrors the quantization reload in app/main.py: drain rather than cancel (the in-flight
        citizen photo finishes and is written normally), and restart on every path including
        failures, so the citizen queue is never left silently stopped.
        """
        worker = self._get_worker()
        held = was_running = False
        try:
            if worker is not None:
                if callable(getattr(worker, "hold", None)):
                    worker.hold(f"video pipeline: session {s['session_id']} (extraction/classification)")
                    held = True
                was_running = bool(worker.is_running)
                if was_running:
                    jl.write("gpu", "worker_stop", reason="GPU sharing rule")
                    _say(f"{s['session_id']}: draining and stopping the PipelineWorker for the GPU")
                    await worker.stop(drain_timeout_seconds=self.cfg.drain_timeout_s)
                    if worker.is_running:
                        raise StageError("the PipelineWorker did not stop; not starting GPU work beside it")
            self._gpu_section_for = s["key"]
            yield
        finally:
            self._gpu_section_for = None
            if worker is not None:
                requested = bool(worker.release_hold()) if held else False
                if (was_running or requested) and not self._shutting_down:
                    try:
                        await worker.start()
                        note = "PipelineWorker restarted after the GPU section"
                    except Exception as e:
                        note = f"PipelineWorker did NOT restart cleanly: {type(e).__name__}: {e}"
                        _say(f"{s['session_id']}: {note}", logging.ERROR)
                    s["worker_note"] = note
                    jl.write("gpu", "worker_start", note=note, was_running=was_running, requested_during_hold=requested)

    # --- extract
    async def _extract(self, s: dict, jl: SessionLog, run: Path, batch: Path, *, skip: bool) -> None:
        if skip:
            jl.write("extract", "skip", reason="run folder already complete", run_dir=str(run))
        else:
            async with self._stage(s, jl, "extract", run_dir=str(run)) as counts:
                self._check_tools()
                if batch.exists():
                    if not _is_within(batch, self.cfg.batch_root):
                        raise StageError(f"refusing to delete {batch}: not under {self.cfg.batch_root}")
                    await asyncio.to_thread(rmtree, batch)
                    counts["removed_stale_batch"] = True
                argv = self._geoai("process", str(s["package_dir"]), "--out", str(self.cfg.runs_root),
                                   "--mode", "segments", *self.cfg.extract_args)
                code, out = await self._cmd(s, jl, "extract", argv, self._pipeline_dir)
                counts["exit_code"] = code
                if code != 0:
                    raise StageError(f"extraction failed (exit {code}): {_tail(out, 600)}")
                ok, why = await asyncio.to_thread(run_status, run)
                if not ok:
                    raise StageError(f"extraction finished but the run folder is incomplete: {why}")
                counts.update(await asyncio.to_thread(segment_metrics, run))
        seg = await asyncio.to_thread(segment_metrics, run)
        s["metrics"].update(seg)
        report = await asyncio.to_thread(_read_json, run / "report.json")
        # geoai process writes its wall-clock timings as report.json "timings_s" (frames.py)
        if isinstance(report.get("timings_s"), dict):
            s["metrics"]["extract_report_timings"] = report["timings_s"]

    # --- classify
    def _classify_done(self, batch: Path) -> bool:
        if not batch_complete(batch):
            return False
        ids = request_ids(batch)
        lines = read_jsonl(batch / "responses.jsonl")
        answered = set(answers_from_responses(lines))
        return not (set(ids) - answered - permanently_rejected(lines, answered))

    async def _classify(self, s: dict, jl: SessionLog, run: Path, batch: Path) -> bool:
        """Export the batch (unless a finished export exists) and run the runner in chunks.
        Returns False when paused between chunks (disable), True when every request is done."""
        if not batch_complete(batch):
            async with self._stage(s, jl, "export", batch_dir=str(batch)) as counts:
                self._check_tools()
                if batch.exists():
                    if not _is_within(batch, self.cfg.batch_root):
                        raise StageError(f"refusing to delete {batch}: not under {self.cfg.batch_root}")
                    await asyncio.to_thread(rmtree, batch)
                    counts["removed_partial_export"] = True
                batch.parent.mkdir(parents=True, exist_ok=True)
                code, out = await self._cmd(s, jl, "export", self._geoai("vlm-export", str(run), "--batch", str(batch)),
                                            self._pipeline_dir)
                counts["exit_code"] = code
                if code != 0:
                    raise StageError(f"vlm-export failed (exit {code}): {_tail(out, 600)}")
                if not batch_complete(batch):
                    raise StageError("vlm-export finished but the batch is incomplete (no SHA256SUMS)")
                counts["requests"] = len(request_ids(batch))
        else:
            jl.write("export", "skip", reason="batch already exported", batch_dir=str(batch))

        ids = await asyncio.to_thread(request_ids, batch)
        runner = self.cfg.geoai_repo / "pipeline" / "tools" / "run_vlm_batch.py"
        stalls = 0
        # The runner re-hashes the whole batch against SHA256SUMS on every call (~5 GB for an 800-image
        # batch, 0.24-0.31 GB/s here). The first call of this pass verifies it; later chunks skip that
        # (--skip-batch-check), and every image is still hashed as it is uploaded.
        batch_verified = False
        async with self._stage(s, jl, "classify", requests=len(ids)) as counts:
            while True:
                lines = await asyncio.to_thread(read_jsonl, batch / "responses.jsonl")
                answered = set(answers_from_responses(lines)) & set(ids)
                rejected = permanently_rejected(lines, answered)
                remaining = set(ids) - answered - rejected
                s["progress"] = {"answered": len(answered), "requests": len(ids), "rejected": len(rejected)}
                counts.update(s["progress"])
                if not remaining:
                    break
                if self._pause_requested():
                    jl.write("classify", "paused", **s["progress"])
                    return False
                if stalls >= STALL_PASSES:
                    last = {r.get("request_id"): r.get("error") for r in lines if r.get("request_id") in remaining}
                    eg = next((f"{k}: {v}" for k, v in last.items() if v), "no answer recorded")
                    raise StageError(f"the VLM did not answer {len(remaining)} of {len(ids)} images after "
                                     f"{STALL_PASSES} runner passes without progress (e.g. {eg})")
                self._check_tools()
                limit = self.cfg.classify_chunk + len(rejected)
                argv = [str(self.cfg.python), str(runner), str(batch), "--url", self.cfg.classify_url,
                        "--limit", str(limit), "--profile", "video"]
                if batch_verified:
                    argv.append("--skip-batch-check")
                if self.cfg.api_key:
                    argv += ["--api-key", self.cfg.api_key]
                self._transition(s, "classifying", detail=f"VLM {len(answered)}/{len(ids)} images")
                code, out = await self._cmd(s, jl, "classify", argv, self.cfg.geoai_repo / "pipeline" / "tools")
                if code == 2:
                    raise StageError(f"VLM runner stopped (exit 2: batch checksums, /health or API key): {_tail(out, 600)}")
                if code not in (0, 1):
                    raise StageError(f"VLM runner exited with {code}: {_tail(out, 600)}")
                batch_verified = True   # exit 0/1 only after SHA256SUMS matched (or was already verified)
                after = set(answers_from_responses(await asyncio.to_thread(read_jsonl, batch / "responses.jsonl")))
                stalls = 0 if len(after & set(ids)) > len(answered) else stalls + 1
            if rejected:
                jl.write("classify", "warning", rejected=sorted(rejected),
                         note="the server rejected these images (4xx); stored without a VLM answer")
        s["metrics"].update(vlm_metrics(await asyncio.to_thread(read_jsonl, batch / "responses.jsonl"), ids))
        return True

    # --- import + analyze
    async def _import_and_analyze(self, s: dict, jl: SessionLog, run: Path, batch: Path) -> None:
        if not batch_complete(batch):
            # Resumed after the batch was already copied out (crash between copy and state save).
            if (run / "vlm" / "responses.jsonl").is_file():
                jl.write("import", "skip", reason="batch gone; results already in run/vlm")
                return
            raise StageError(f"batch {batch} is missing and no results were copied out; retry re-exports it")
        self._check_tools()
        self._transition(s, "classifying", detail="importing VLM answers")
        async with self._stage(s, jl, "import") as counts:
            code, out = await self._cmd(s, jl, "import", self._geoai("vlm-import", str(batch), str(run)),
                                        self._pipeline_dir)
            counts["exit_code"] = code
            if code != 0:
                raise StageError(f"vlm-import failed (exit {code}): {_tail(out, 600)}")
        self._transition(s, "classifying", detail="analysing")
        async with self._stage(s, jl, "analyze") as counts:
            code, out = await self._cmd(s, jl, "analyze", self._geoai("analyze", str(batch)), self._pipeline_dir)
            counts["exit_code"] = code
            if code != 0:
                raise StageError(f"analyze failed (exit {code}): {_tail(out, 600)}")
        async with self._stage(s, jl, "collect") as counts:
            counts["copied"] = await asyncio.to_thread(copy_batch_results, batch, run)
        ids = await asyncio.to_thread(request_ids, batch)
        s["metrics"].update(vlm_metrics(await asyncio.to_thread(read_jsonl, batch / "responses.jsonl"), ids))

    # --- store + verify
    async def _store_and_verify(self, s: dict, jl: SessionLog, run: Path) -> None:
        store = self._store_module()
        responses = run / "vlm" / "responses.jsonl"
        if not responses.is_file():
            raise StageError(f"{responses} missing: the VLM answers were not collected")
        answers = answers_from_responses(await asyncio.to_thread(read_jsonl, responses))
        sid = s["session_id"]
        image_root = self.cfg.image_root

        def work() -> tuple[dict, dict]:
            conn = store.connect()
            try:
                result = store.store_session(conn, run, answers, image_root=image_root)
                ver = store.verify_session(conn, sid, run)
                return result, ver
            finally:
                _close(conn)

        async with self._stage(s, jl, "store", answers=len(answers)) as counts:
            result, ver = await asyncio.to_thread(work)
            counts["result"] = _json_safe(result)
            counts["verify"] = _verify_summary(ver)
        s["metrics"]["store"] = _json_safe(result)
        s["metrics"]["verify"] = _verify_summary(ver)
        if not (isinstance(ver, dict) and ver.get("ok")):
            raise StageError(f"stored but verification failed: {_verify_text(ver)}")
        report = await asyncio.to_thread(_read_json, run / "report.json")
        await self._set_state(s, "stored", report=report or None)

    # --- cleanup
    async def _cleanup(self, s: dict, jl: SessionLog, run: Path, batch: Path) -> None:
        store = self._store_module()
        sid = s["session_id"]
        pkg = Path(s["package_dir"])
        root = self.cfg.image_root or getattr(store, "IMAGE_ROOT", None)
        image_dir = (Path(root) / sid) if root else None

        def verify() -> dict:
            conn = store.connect()
            try:
                return store.verify_session(conn, sid, run)
            finally:
                _close(conn)

        async with self._stage(s, jl, "cleanup") as counts:
            ver = await asyncio.to_thread(verify)
            if not (isinstance(ver, dict) and ver.get("ok")):
                raise StageError(f"not cleaning up: verification failed: {_verify_text(ver)}")
            if not (run / "vlm" / "responses.jsonl").is_file():
                raise StageError("not cleaning up: the VLM answers were not copied into the run folder")
            if not _is_within(batch, self.cfg.batch_root) or _is_within(batch, pkg) or _is_within(run, pkg):
                raise StageError("not cleaning up: batch or run folder lies outside its root or inside the package")
            before = await asyncio.to_thread(storage_parts, pkg, run, batch, image_dir)
            dupes = set() if self.cfg.keep_run_frames else await asyncio.to_thread(stored_duplicates, run, image_dir)
            deleted = await asyncio.to_thread(delete_intermediates, run, batch, dupes)
            counts["run_frames_deleted_as_stored"] = len(dupes)
            after = await asyncio.to_thread(storage_parts, pkg, run, batch, image_dir)
            ver_after = await asyncio.to_thread(verify)
            if not (isinstance(ver_after, dict) and ver_after.get("ok")):
                raise StageError(f"verification failed AFTER cleanup: {_verify_text(ver_after)}")
            storage = {"before_gb": round(sum(before.values()) / 1e9, 6), "after_gb": round(sum(after.values()) / 1e9, 6),
                       "freed_gb": round((sum(before.values()) - sum(after.values())) / 1e9, 6),
                       "before_bytes": before, "after_bytes": after, "unit": "GB = 1e9 bytes",
                       "deleted_files": deleted["files"], "deleted_paths": deleted["paths"][:200]}
            counts.update({k: v for k, v in storage.items() if k != "deleted_paths"})
        s["metrics"]["storage"] = storage
        await self._set_state(s, "cleaned")

    # ----------------------------------------------------------- views
    def status_summary(self) -> dict:
        counts = {st: 0 for st in STATES}
        for sess in self._sessions.values():
            counts[sess["state"]] = counts.get(sess["state"], 0) + 1
        return {"enabled": self._enabled, "enabled_source": self._enabled_source,
                "running": bool(self._task and not self._task.done()), "busy": self._current is not None,
                "stopping": (not self._enabled) and self._current is not None, "counts": counts}

    def status(self) -> dict:
        """Full status for GET /operator/video/status (reads a little from disk: call in a thread)."""
        out = self.status_summary()
        cur = self._sessions.get(self._current) if self._current else None
        if cur:
            progress = dict(cur.get("progress") or {})
            if cur["state"] == "classifying" and cur.get("batch_dir"):
                b = Path(cur["batch_dir"])
                ids = request_ids(b)
                if ids:
                    answered = set(answers_from_responses(read_jsonl(b / "responses.jsonl"))) & set(ids)
                    progress.update(answered=len(answered), requests=len(ids))
            out["current"] = {"key": cur["key"], "session_id": cur["session_id"], "state": cur["state"],
                              "detail": cur.get("detail"), "progress": progress, "live": self._live(cur)}
        worker = self._get_worker()
        out["worker"] = {"present": worker is not None,
                         "running": bool(worker and worker.is_running),
                         "hold_reason": getattr(worker, "hold_reason", None) if worker else None}
        out["gpu_section"] = self._gpu_section_for
        out["config"] = self.cfg.public()
        exe = "ffprobe.exe" if os.name == "nt" else "ffprobe"
        out["checks"] = {"inbox_exists": self.cfg.inbox.is_dir(), "geoai_repo_exists": self.cfg.geoai_repo.is_dir(),
                         "ffprobe_found": bool((self.cfg.ffprobe_dir and (self.cfg.ffprobe_dir / exe).is_file())
                                               or shutil.which("ffprobe")),
                         "python_exists": self.cfg.python.is_file(),
                         "runner_exists": (self.cfg.geoai_repo / "pipeline" / "tools" / "run_vlm_batch.py").is_file(),
                         "video_store": _module_available("app.video_store") if self._store is None else "ok",
                         "state_file_error": self._persist_error}
        out["last_error"] = self._last_error
        return out

    def _live(self, cur: dict) -> dict:
        """What the session in progress is doing right now, for the operator page's live card.

        stage / stage_started_utc: the running stage (validate, extract, export, classify, import,
        analyze, store, cleanup) and when it began; timings_s: seconds of each finished stage.
        extract: the last progress line of `geoai process` (phase, done/total, last segment) plus the
        newest best frame written. classify: counts, the segment in flight and the last answer
        (classify_live). frame: the segment whose image to show (GET /video-frame).
        """
        live: dict = {"stage": cur.get("stage"), "stage_started_utc": cur.get("stage_started_utc"),
                      "timings_s": dict((cur.get("metrics") or {}).get("timings_s") or {}),
                      "history": list(cur.get("history") or [])[-12:], "frame": None}
        run = Path(cur["run_dir"]) if cur.get("run_dir") else None
        lo = cur.get("live_output") or {}
        if cur["state"] == "extracting" and run is not None:
            prog = extract_progress(Path(lo["file"])) if lo.get("step") == "extract" and lo.get("file") else None
            frame = latest_frame_id(run)
            live["extract"] = {"progress": prog, "latest_frame": frame}
            live["frame"] = frame
        elif cur["state"] == "classifying" and cur.get("batch_dir"):
            c = classify_live(Path(cur["batch_dir"]))
            live["classify"] = c
            if c:
                live["frame"] = (c.get("in_flight") or {}).get("segment_id") or \
                    (c.get("last_answer") or {}).get("segment_id")
        return live

    def frame_file(self, key_or_id: str, segment_id: str) -> Optional[Path]:
        """The PNG of one of a session's segments, wherever it is now: the run's frames/ (extraction),
        the batch's images/ (classification), or the local image store (after cleanup). Only that
        session's own `<session_id>_segNNNNN` ids resolve, so no other file can be named."""
        key = self._find(key_or_id)
        if key is None:
            return None
        s = self._sessions[key]
        m = _SEGMENT_ID.match(segment_id or "")
        if not m or m.group("sid") != s["session_id"]:
            return None
        name = f"{segment_id}.png"
        dirs = []
        if s.get("run_dir"):
            dirs.append(Path(s["run_dir"]) / "frames")
        if s.get("batch_dir"):
            dirs.append(Path(s["batch_dir"]) / "images")
        root = getattr(self.cfg, "image_root", None)
        if root is None:
            try:
                from app.local_images import image_root
                root = image_root()
            except Exception:
                root = None
        if root is not None:
            dirs.append(Path(root) / s["session_id"])
        for d in dirs:
            p = d / name
            if p.is_file():
                return p
        return None

    def sessions(self, limit: int = 100) -> list[dict]:
        """Sessions, newest first, without the bulky validator output."""
        rows = []
        for sess in self._sessions.values():
            row = {k: v for k, v in sess.items() if k not in ("validation", "stable_since")}
            v = sess.get("validation")
            if v:
                row["validation"] = {"exit_code": v.get("exit_code"), "utc": v.get("utc"),
                                     "output_file": v.get("output_file"),
                                     "output": v.get("output") if v.get("exit_code") != 0 else None}
            row["current"] = sess["key"] == self._current
            rows.append(row)
        rows.sort(key=lambda r: (r.get("first_seen_utc") or "", r["key"]), reverse=True)
        return rows[:limit]

    def session_log(self, key_or_id: str, limit: int = 500) -> dict:
        key = self._find(key_or_id)
        sid = self._sessions[key]["session_id"] if key else key_or_id
        return {"session_id": sid, "known": key is not None,
                "events": SessionLog.read(self.cfg.log_root, sid, limit=limit)}


# ============================================================
# Filesystem work for collect / cleanup (pure functions, unit-tested)
# ============================================================

def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _close(conn: Any) -> None:
    close = getattr(conn, "close", None)
    if callable(close):
        with contextlib.suppress(Exception):
            close()


def _module_available(name: str) -> str:
    try:
        importlib.import_module(name)
        return "ok"
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def _verify_summary(ver: Any) -> dict:
    if not isinstance(ver, dict):
        return {"ok": False, "raw": str(ver)[:500]}
    return {"ok": bool(ver.get("ok")), "records": ver.get("records"), "rows": ver.get("rows"),
            "answered_rows": ver.get("answered_rows"), "missing_images": _count(ver.get("missing_images")),
            "bad_checksums": _count(ver.get("bad_checksums")), "problems": _count(ver.get("problems"))}


def _verify_text(ver: Any) -> str:
    v = _verify_summary(ver)
    text = (f"records={v.get('records')} rows={v.get('rows')} answered_rows={v.get('answered_rows')} "
            f"missing_images={v.get('missing_images')} bad_checksums={v.get('bad_checksums')}")
    if isinstance(ver, dict):
        for k in ("problems", "missing_images", "bad_checksums"):
            if isinstance(ver.get(k), (list, tuple)) and ver[k]:
                text += f"; {k} e.g. {ver[k][0]}"
    return text


def copy_batch_results(batch: Path, run: Path) -> list[str]:
    """Copy the VLM answers and analysis out of the batch into <run>/vlm/ (atomically per file)."""
    dst = run / "vlm"
    dst.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in BATCH_KEEP_FILES:
        src = batch / name
        if src.is_file():
            tmp = dst / (name + ".tmp")
            shutil.copy2(src, tmp)
            os.replace(tmp, dst / name)
            copied.append(name)
    for name in BATCH_KEEP_DIRS:
        src = batch / name
        if src.is_dir():
            target = dst / name
            tmp = dst / (name + ".tmp")
            if tmp.exists():
                rmtree(tmp)
            shutil.copytree(src, tmp)
            if target.exists():
                rmtree(target)
            os.replace(tmp, target)
            copied.append(name + "/")
    return copied


def storage_parts(pkg: Path, run: Path, batch: Path, image_dir: Optional[Path]) -> dict:
    parts = {"package": dir_bytes(pkg), "run": dir_bytes(run), "batch": dir_bytes(batch)}
    if image_dir is not None:
        parts["image_store"] = dir_bytes(image_dir)
    return parts


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stored_duplicates(run: Path, image_dir: Optional[Path]) -> set[Path]:
    """Run-folder best frames whose image-store copy exists with the record's SHA-256.

    video_store copies every send/audit frame to <image root>/<session>/<id>.png. Once that copy
    hashes to the record's frame.sha256, the run's frames/<id>.png is a second copy of the same
    bytes (about 5 MB per 4K frame, ~1 GB per km). A frame without such a copy is never returned.
    """
    if image_dir is None or not image_dir.is_dir() or _is_within(image_dir, run):
        return set()
    out = set()
    for rec in load_sidecars(run):
        fr = rec.get("frame") or {}
        if rec.get("status") not in SENT_STATUSES or not fr.get("file") or not fr.get("sha256"):
            continue
        src = (run / fr["file"]).resolve()
        dst = (image_dir / f"{rec['id']}.png").resolve()
        if src != dst and src.is_file() and dst.is_file() and sha256_file(dst) == fr["sha256"]:
            out.add(src)
    return out


def delete_intermediates(run: Path, batch: Path, stored_frames: set[Path] = frozenset()) -> dict:
    """Delete what a stored session no longer needs; return what was deleted.

    Deleted: frames/ PNGs not referenced by a send/audit record (skip frames, if any were
    written), the run's copies of best frames already verified in the image store
    (`stored_frames`, see stored_duplicates), segment_views/, crop-mode outputs (crops/,
    not_sent/), caches and *.tmp files in the run folder, and the whole temporary batch.
    Kept: everything else in the run folder - the records (sidecars/, segments.csv/.geojson,
    report.json, contact sheet) and the VLM results (vlm_results_*, vlm/ with the analysis) -
    and the session package, which this function never touches.
    """
    paths: list[str] = []
    files = 0

    def rm_file(p: Path) -> None:
        nonlocal files
        p.unlink()
        files += 1
        paths.append(str(p))

    def rm_tree(p: Path) -> None:
        nonlocal files
        files += sum(1 for q in p.rglob("*") if q.is_file())
        rmtree(p)
        paths.append(str(p) + os.sep)

    keep_frames = set()
    for rec in load_sidecars(run):
        f = (rec.get("frame") or {}).get("file")
        if rec.get("status") in SENT_STATUSES and f and (run / f).resolve() not in stored_frames:
            keep_frames.add((run / f).resolve())
    frames = run / "frames"
    if frames.is_dir():
        for p in sorted(frames.iterdir()):
            if p.is_file() and p.resolve() not in keep_frames:
                rm_file(p)
    for name in INTERMEDIATE_DIRS:
        for p in sorted(run.rglob(name)):
            if p.is_dir() and _is_within(p, run):
                rm_tree(p)
    for p in sorted(run.rglob("*")):
        if p.is_file() and p.suffix in INTERMEDIATE_SUFFIXES:
            rm_file(p)
    if batch.exists():
        rm_tree(batch)
    return {"files": files, "paths": paths}


# ============================================================
# Singleton (started by app/video_routes.install)
# ============================================================

_service: Optional[VideoPipelineService] = None


def get_service() -> Optional[VideoPipelineService]:
    return _service


def set_service(service: Optional[VideoPipelineService]) -> None:
    global _service
    _service = service
