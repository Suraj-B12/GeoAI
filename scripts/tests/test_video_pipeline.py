"""
Tests for the drive-video pipeline service (app/video_pipeline.py), its routes and the
PipelineWorker GPU hold (GPU-free, no network, no database).

The service chains four external tools (GeoAI_Video's validator, `geoai process`, the VLM batch
runner, `geoai vlm-import/analyze`) and a local store (app/video_store.py, written separately),
and must keep the citizen-upload worker off the GPU while it extracts or classifies. Each of
those is replaced by a fake that does what the real tool does to the filesystem, so the
orchestration can be checked end to end:

  1. a session goes queued -> extracting -> classifying -> storing -> stored -> cleaned
  2. a package that fails validation is refused, with the validator's reason kept
  3. a failure mid-way marks the session failed with the error, and the worker is restarted
  4. disabling the toggle stops after the current item; the session resumes from its stage
  5. the worker is never running while extraction or classification runs
  6. cleanup deletes exactly the intermediates and keeps the listed files; storage measured
  7. the JSON-lines log is written (start/end per stage, errors with stack traces)
  8. a server shutdown kills the running command and the session resumes on the next start
  9. helpers (percentile, VLM metrics, run completeness), the real subprocess runner
     (redaction, cancel), the real PipelineWorker hold, the HTTP routes and the main.py include

Expected numbers come from hand calculation on the fake data, not from the code under test.

Usage:
    venv/Scripts/python.exe scripts/tests/test_video_pipeline.py
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("SKIP_MODEL_SELFTEST", "1")

from app import video_pipeline as vp  # noqa: E402
from app.video_pipeline import Interrupted, VideoPipelineConfig, VideoPipelineService  # noqa: E402

FAILURES: list[str] = []
CHECKS = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if cond:
        print(f"  [OK]   {name}")
    else:
        print(f"  [FAIL] {name}" + (f" - {detail}" if detail else ""))
        FAILURES.append(name)


# ============================================================
# Fake data
# ============================================================

# (k, status, length_m) of the fake run: route 22.5 m, covered 17.5 m, not covered 5.0 m;
# send 2, audit 1, skip 1, not_covered 1.
SEGMENTS = [(0, "send", 5.0), (1, "skip", 5.0), (2, "audit", 5.0), (3, "not_covered", 5.0), (4, "send", 2.5)]
SENT_K = [0, 2, 4]
# Seconds of the successful request per segment: median 14.0, p95 = 14 + 0.9 * (20 - 14) = 19.4.
SECONDS = {0: 10.0, 2: 14.0, 4: 20.0}
# Intermediate bytes the fake extraction writes, all to be deleted: skip frame 111, segment view
# 222, crop 50, cache 77, tmp 13.
INTERMEDIATE_BYTES = 111 + 222 + 50 + 77 + 13
FRAME_BYTES = 300
API_KEY = "SECRET-KEY-123"


def seg_id(sid: str, k: int) -> str:
    return f"{sid}_seg{k:05d}"


def write(p: Path, data: bytes | str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        p.write_text(data, encoding="utf-8")
    else:
        p.write_bytes(data)


def arg_after(argv: list, flag: str) -> str:
    return str(argv[argv.index(flag) + 1])


class FakeWorker:
    """PipelineWorker surface used by the service, with the same hold semantics."""

    def __init__(self, running: bool = True):
        self._running = running
        self.hold_reason = None
        self._requested = False
        self.events: list[str] = []

    @property
    def is_running(self) -> bool:
        return self._running

    def hold(self, reason: str) -> None:
        self.hold_reason = reason
        self._requested = False
        self.events.append("hold")

    def release_hold(self) -> bool:
        r = self._requested
        self.hold_reason = None
        self._requested = False
        self.events.append("release")
        return r

    async def start(self) -> dict:
        if self.hold_reason:
            self._requested = True
            self.events.append("start_refused")
            return {"status": "held"}
        self._running = True
        self.events.append("start")
        return {"status": "started"}

    async def stop(self, drain_timeout_seconds: float = 60.0) -> dict:
        self._running = False
        self.events.append("stop")
        return {"status": "stopped"}


class FakeConn:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeStore:
    """app/video_store interface: connect / store_session / verify_session / set_session_state / list_sessions."""

    def __init__(self, world: "World"):
        self.world = world
        self.states: list[tuple[str, str, str | None]] = []
        self.stored: dict[str, dict] = {}
        self.store_calls: list[dict] = []
        self.conns: list[FakeConn] = []

    def connect(self, dsn=None):
        c = FakeConn()
        self.conns.append(c)
        return c

    def store_session(self, conn, run_dir, answers, *, image_root=None):
        recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((Path(run_dir) / "sidecars").glob("*.json"))]
        sid = recs[0]["session_id"]
        images = 0
        for r in recs:
            if r["status"] in ("send", "audit"):
                write(Path(image_root) / sid / f"{r['id']}.png", (Path(run_dir) / r["frame"]["file"]).read_bytes())
                images += 1
        batch = self.world.batches / f"batch_{sid}"
        self.world.batch_bytes_at_store[sid] = sum(f.stat().st_size for f in batch.rglob("*") if f.is_file())
        self.store_calls.append({"run_dir": Path(run_dir), "answers": answers, "image_root": image_root})
        self.stored[sid] = {"records": len(recs), "answered": len(answers), "images": images}
        return {"session_id": sid, "rows": len(recs), "images": images}

    def verify_session(self, conn, session_id, run_dir):
        st = self.stored.get(session_id)
        if st is None:
            return {"ok": False, "records": 0, "rows": 0, "answered_rows": 0, "missing_images": [], "bad_checksums": []}
        missing = [str(p) for k in SENT_K
                   if not (p := self.world.images / session_id / f"{seg_id(session_id, k)}.png").exists()]
        return {"ok": not missing and not self.world.verify_fails, "records": st["records"], "rows": st["records"],
                "answered_rows": st["answered"], "missing_images": missing, "bad_checksums": []}

    def set_session_state(self, conn, session_id, state, *, error=None, metrics=None, report=None):
        self.states.append((session_id, state, error))

    def list_sessions(self, conn, limit=50):
        return [{"session_id": s} for s in self.stored][:limit]

    def db_states(self, sid: str) -> list[str]:
        out = []
        for s, st, _ in self.states:
            if s == sid and (not out or out[-1] != st):
                out.append(st)
        return out


class World:
    """Temp folders + fakes for the GeoAI_Video tools, the worker and the store."""

    def __init__(self, root: Path, *, worker_running: bool = True):
        self.root = root
        self.inbox, self.runs, self.batches = root / "inbox", root / "runs", root / "batches"
        self.logs, self.images = root / "logs", root / "images"
        self.repo = root / "GeoAI_Video"
        self.python = self.repo / "pipeline" / ".venv" / "Scripts" / "python.exe"
        write(self.python, b"")
        write(self.repo / "tools" / "validate_session.py", "")
        write(self.repo / "pipeline" / "tools" / "run_vlm_batch.py", "")
        self.inbox.mkdir(parents=True)
        self.worker = FakeWorker(worker_running)
        self.store = FakeStore(self)
        self.calls: list[tuple[str, str]] = []
        self.classify_argvs: list[list[str]] = []
        self.violations: list[str] = []
        self.fail_at: dict[str, tuple[int, str]] = {}
        self.on_step: dict = {}
        self.batch_bytes_at_store: dict[str, int] = {}
        self.errored_once: set[str] = set()
        self.verify_fails = False
        self.block_extract = threading.Event()   # set: extraction waits for cancel (shutdown test)
        self.in_extract = threading.Event()

    def cfg(self, **kw) -> VideoPipelineConfig:
        base = dict(inbox=self.inbox, runs_root=self.runs, batch_root=self.batches, log_root=self.logs,
                    geoai_repo=self.repo, python=self.python, classify_url="http://127.0.0.1:8000",
                    image_root=self.images, api_key=API_KEY, settle_s=0.0, poll_s=0.05, classify_chunk=2)
        base.update(kw)
        return VideoPipelineConfig(**base)

    def service(self, **kw) -> VideoPipelineService:
        return VideoPipelineService(self.cfg(**kw), store=self.store, run_command=self.run_command,
                                    get_worker=lambda: self.worker)

    def add_package(self, name: str, sid: str, *, bad: bool = False) -> Path:
        d = self.inbox / name
        write(d / "manifest.json", json.dumps({"files": ["video.mp4"]}))
        write(d / "session.json", json.dumps({"session_id": sid, "spec_version": "1.1"}))
        write(d / "video.mp4", b"v" * 1000)
        write(d / "frames.csv", "t_ns\n1\n")
        if bad:
            write(d / "BAD", "corrupt")
        return d

    def count(self, step: str) -> int:
        return sum(1 for s, _ in self.calls if s == step)

    # -------------------------------------------------- the fake tools
    def run_command(self, step, argv, cwd, out_file, timeout_s, cancel):
        out_file.parent.mkdir(parents=True, exist_ok=True)
        if step in ("extract", "classify"):
            if self.worker.is_running:
                self.violations.append(f"worker running during {step}")
            if not self.worker.hold_reason:
                self.violations.append(f"worker not held during {step}")
        hook = self.on_step.pop(step, None)
        if hook:
            hook()
        if step in self.fail_at:
            code, msg = self.fail_at.pop(step)
            self.calls.append((step, "failed"))
            with open(out_file, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
            return code
        return getattr(self, "_" + step)(argv, out_file, cancel)

    def _validate(self, argv, out_file, cancel):
        pkg = Path(argv[-1])
        self.calls.append(("validate", pkg.name))
        if (pkg / "BAD").exists():
            write(out_file, "PASS  spec_version             1.1\nFAIL  manifest checksums  video.mp4 differs from manifest\n")
            return 1
        write(out_file, "PASS  spec_version  1.1\nPASS  manifest checksums  4 files\n")
        return 0

    def _extract(self, argv, out_file, cancel):
        pkg = Path(arg_after(argv, "process"))
        sid = json.loads((pkg / "session.json").read_text(encoding="utf-8"))["session_id"]
        self.calls.append(("extract", sid))
        if arg_after(argv, "--mode") != "segments" or Path(arg_after(argv, "--out")) != self.runs:
            self.violations.append(f"bad extract argv {argv}")
        run = self.runs / sid
        if run.exists():
            shutil.rmtree(run)
        if self.block_extract.is_set():
            write(run / "sidecars" / "partial.json", "{")   # an interrupted run is incomplete
            self.in_extract.set()
            cancel.wait(10)
            raise Interrupted("extract stopped: server shutting down")
        for k, status, length in SEGMENTS:
            rid = seg_id(sid, k)
            rec = {"id": rid, "kind": "segment", "schema": "geoai.segment.v1", "status": status, "session_id": sid,
                   "segment": {"k": k, "length_m": length}, "flags": ["geometry_nominal"]}
            if status in ("send", "audit"):
                write(run / "frames" / f"{rid}.png", bytes([k]) * FRAME_BYTES)
                rec["frame"] = {"file": f"frames/{rid}.png", "width": 3840, "height": 2160,
                                "sha256": hashlib.sha256(bytes([k]) * FRAME_BYTES).hexdigest()}
            write(run / "sidecars" / f"{rid}.json", json.dumps(rec))
        write(run / "frames" / f"{seg_id(sid, 1)}.png", b"s" * 111)          # a skip frame
        write(run / "segment_views" / f"{seg_id(sid, 0)}.png", b"w" * 222)
        write(run / "crops" / "c.png", b"c" * 50)
        write(run / "cache" / "bank.npy", b"b" * 77)
        write(run / "sidecars.tmp", b"t" * 13)
        write(run / "segments.csv", "id,status\n")
        write(run / "segments.geojson", '{"type": "FeatureCollection", "features": []}')
        write(run / "contact_sheet.jpg", b"j" * 40)
        write(run / "report.json", json.dumps({"mode": "segments", "session_id": sid, "timings_s": {"total": 1.5, "decode": 0.9}}))
        write(out_file, f"{sid}: {len(SEGMENTS)} segments\n")
        return 0

    def _export(self, argv, out_file, cancel):
        run = Path(arg_after(argv, "vlm-export"))
        batch = Path(arg_after(argv, "--batch"))
        sid = run.name
        self.calls.append(("export", sid))
        reqs = []
        for p in sorted((run / "sidecars").glob("*.json")):
            rec = json.loads(p.read_text(encoding="utf-8"))
            write(batch / "records" / p.name, json.dumps(rec))
            if rec["status"] in ("send", "audit"):
                write(batch / "images" / f"{rec['id']}.png", (run / rec["frame"]["file"]).read_bytes())
                reqs.append({"request_id": rec["id"], "kind": "segment", "image": f"images/{rec['id']}.png"})
        write(batch / "runs" / sid / "report.json", (run / "report.json").read_text(encoding="utf-8"))
        write(batch / "requests.jsonl", "".join(json.dumps(r) + "\n" for r in reqs))
        write(batch / "batch.json", json.dumps({"mode": "segments", "sessions": [sid]}))
        write(batch / "SHA256SUMS", "sums\n")
        return 0

    def _classify(self, argv, out_file, cancel):
        batch = Path(argv[2])
        self.classify_argvs.append([str(a) for a in argv])
        if arg_after(argv, "--url") != "http://127.0.0.1:8000" or arg_after(argv, "--api-key") != API_KEY:
            self.violations.append("runner argv lacks url or api key")
        limit = int(arg_after(argv, "--limit"))
        reqs = [json.loads(x)["request_id"] for x in (batch / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
        resp = batch / "responses.jsonl"
        done = {r["request_id"] for r in vp.read_jsonl(resp) if not r.get("error")}
        todo = [r for r in reqs if r not in done][:limit]
        self.calls.append(("classify", ",".join(t[-1] for t in todo)))
        with open(resp, "a", encoding="utf-8") as f:
            for rid in todo:
                k = int(rid[-5:])
                if k == 4 and rid not in self.errored_once:
                    # First pass of segment 4: server busy, the runner gives up after 4 retries.
                    self.errored_once.add(rid)
                    line = {"request_id": rid, "http_status": 503, "seconds": 1.0, "attempts": 5,
                            "response": {"detail": "busy"}, "error": "HTTP 503"}
                else:
                    line = {"request_id": rid, "http_status": 200, "seconds": SECONDS[k],
                            "attempts": 2 if k == 0 else 1,
                            "response": {"stage1_label": "Distress", "distress_types": ["Pothole"]}, "error": None}
                    done.add(rid)
                f.write(json.dumps(line) + "\n")
        return 0 if all(r in done for r in reqs) else 1

    def _import(self, argv, out_file, cancel):
        batch, run = Path(arg_after(argv, "vlm-import")), Path(argv[argv.index("vlm-import") + 2])
        self.calls.append(("import", run.name))
        write(batch / "import_summary.json", json.dumps({"answered": 3}))
        write(batch / "vlm_results.csv", "request_id\n")
        write(batch / "vlm_results.geojson", '{"type": "FeatureCollection", "features": []}')
        write(run / f"vlm_results_{batch.name}.csv", "request_id\n")
        return 0

    def _analyze(self, argv, out_file, cancel):
        batch = Path(arg_after(argv, "analyze"))
        self.calls.append(("analyze", batch.name))
        write(batch / "analysis" / "REPORT.md", "# report\n")
        write(batch / "analysis" / "label_summary.csv", "label\n")
        return 0


async def drain(svc: VideoPipelineService, max_ticks: int = 20) -> None:
    for _ in range(max_ticks):
        if not await svc.tick():
            return


def history(svc: VideoPipelineService, key: str) -> list[str]:
    return [h["state"] for h in svc._sessions[key]["history"]]


def log_events(w: World, sid: str) -> list[dict]:
    logf = w.logs / datetime.now().strftime("%Y-%m-%d") / f"{sid}.jsonl"
    return [json.loads(x) for x in logf.read_text(encoding="utf-8").splitlines()] if logf.exists() else []


# ============================================================
# 1, 5, 6, 7: the happy path
# ============================================================

def test_happy_path() -> None:
    print("\n[1] queued -> extracting -> classifying -> storing -> stored -> cleaned")
    with tempfile.TemporaryDirectory() as tmp:
        w = World(Path(tmp))
        pkg = w.add_package("pkgA", "S1")
        pkg_files_before = sorted((p.relative_to(pkg).as_posix(), p.stat().st_size) for p in pkg.rglob("*") if p.is_file())
        svc = w.service()
        svc.enable()
        asyncio.run(drain(svc))
        s = svc._sessions["pkgA"]
        check("session cleaned", s["state"] == "cleaned", f"{s['state']}: {s.get('error')}")
        check("local state sequence", history(svc, "pkgA") ==
              ["queued", "extracting", "classifying", "storing", "stored", "cleaned"], str(history(svc, "pkgA")))
        check("database state sequence", w.store.db_states("S1") ==
              ["queued", "extracting", "classifying", "storing", "stored", "cleaned"], str(w.store.db_states("S1")))
        check("every DB connection closed", all(c.closed for c in w.store.conns))
        steps = [st for st, _ in w.calls]
        check("tools called in order", steps == ["validate", "extract", "export", "classify", "classify", "classify",
                                                 "import", "analyze"], str(steps))
        # 5: GPU rule
        check("worker never running/unheld during extract or classify", not w.violations, str(w.violations))
        check("worker held, drained once, restarted once", w.worker.events == ["hold", "stop", "release", "start"],
              str(w.worker.events))
        check("worker running again at the end", w.worker.is_running)
        # store got the answers
        call = w.store.store_calls[0]
        check("store_session got answers keyed by segment id",
              set(call["answers"]) == {seg_id("S1", k) for k in SENT_K}, str(sorted(call["answers"])))
        check("answers are the raw /classify JSON", call["answers"][seg_id("S1", 0)].get("distress_types") == ["Pothole"])
        check("store_session got the image root", Path(call["image_root"]) == w.images)
        # metrics, against hand calculation
        m = s["metrics"]
        check("route / covered / not covered m", (m["route_m"], m["covered_m"], m["not_covered_m"]) == (22.5, 17.5, 5.0),
              str((m.get("route_m"), m.get("covered_m"), m.get("not_covered_m"))))
        check("segments and send/audit/skip counts",
              (m["segments"], m["send"], m["audit"], m["skip"], m["not_covered"]) == (5, 2, 1, 1, 1))
        check("VLM s/image median 14.0, p95 19.4", (m["vlm_s_median"], m["vlm_s_p95"]) == (14.0, 19.4),
              str((m.get("vlm_s_median"), m.get("vlm_s_p95"))))
        # lines: seg0 ok (2 attempts), seg2 ok (1), seg4 503 (5), seg4 ok (1): 9 attempts, 3 ids -> 6 retries
        check("VLM errors and retries", (m["vlm_error_lines"], m["vlm_retries"], m["vlm_unanswered"]) == (1, 6, 0),
              str((m.get("vlm_error_lines"), m.get("vlm_retries"), m.get("vlm_unanswered"))))
        check("timings for every stage", set(m["timings_s"]) >= {"validate", "extract", "export", "classify", "import",
                                                                 "analyze", "store", "cleanup"}, str(m["timings_s"]))
        check("extraction report timings carried (report.json 'timings_s')",
              m.get("extract_report_timings") == {"total": 1.5, "decode": 0.9}, str(m.get("extract_report_timings")))
        # Runner argv: every chunk sends profile=video; only the first chunk re-hashes the batch.
        cav = w.classify_argvs
        check("runner gets --profile video on every chunk",
              len(cav) == 3 and all(arg_after(a, "--profile") == "video" for a in cav), str(cav))
        check("first chunk verifies SHA256SUMS, later chunks skip it",
              ["--skip-batch-check" in a for a in cav] == [False, True, True], str(cav))
        # 6: cleanup
        run = w.runs / "S1"
        gone = [run / "frames" / f"{seg_id('S1', 1)}.png", run / "segment_views", run / "crops", run / "cache",
                run / "sidecars.tmp", w.batches / "batch_S1",
                *[run / "frames" / f"{seg_id('S1', k)}.png" for k in SENT_K]]   # stored copies verified
        check("intermediates deleted", not any(p.exists() for p in gone), str([str(p) for p in gone if p.exists()]))
        kept = [run / "report.json", run / "segments.csv", run / "segments.geojson", run / "contact_sheet.jpg",
                run / "vlm_results_batch_S1.csv", run / "vlm" / "responses.jsonl", run / "vlm" / "requests.jsonl",
                run / "vlm" / "batch.json", run / "vlm" / "import_summary.json", run / "vlm" / "vlm_results.csv",
                run / "vlm" / "vlm_results.geojson", run / "vlm" / "analysis" / "REPORT.md",
                *[w.images / "S1" / f"{seg_id('S1', k)}.png" for k in SENT_K]]
        check("records, results and best frames kept", all(p.exists() for p in kept),
              str([str(p) for p in kept if not p.exists()]))
        check("all 5 sidecars kept", len(list((run / "sidecars").glob("*.json"))) == 5)
        pkg_files_after = sorted((p.relative_to(pkg).as_posix(), p.stat().st_size) for p in pkg.rglob("*") if p.is_file())
        check("session package untouched", pkg_files_before == pkg_files_after)
        st = m["storage"]
        freed = sum(st["before_bytes"].values()) - sum(st["after_bytes"].values())
        expected = INTERMEDIATE_BYTES + len(SENT_K) * FRAME_BYTES + w.batch_bytes_at_store["S1"]
        check("freed bytes = intermediates + batch", freed == expected, f"{freed} vs {expected}")
        check("storage in GB matches the bytes", st["unit"] == "GB = 1e9 bytes"
              and st["before_gb"] == round(sum(st["before_bytes"].values()) / 1e9, 6)
              and st["freed_gb"] == round(expected / 1e9, 6), str(st))
        check("package bytes the same before and after", st["before_bytes"]["package"] == st["after_bytes"]["package"])
        # 7: JSON-lines log
        lines = log_events(w, "S1")
        check("JSON-lines log exists and parses", bool(lines))
        for stage in ("validate", "extract", "export", "classify", "import", "analyze", "store", "cleanup"):
            evs = {e["event"] for e in lines if e["stage"] == stage}
            check(f"log has start and end for {stage}", {"start", "end"} <= evs, str(evs))
        ends = [e for e in lines if e["event"] == "end"]
        check("end events carry start/end time, seconds and counts",
              all({"start_utc", "end_utc", "seconds", "counts"} <= set(e) for e in ends))
        logdir = w.logs / datetime.now().strftime("%Y-%m-%d")
        check("command output kept beside the log", (logdir / "S1.extract.txt").exists())
        check("API key never in the log folder",
              not any(API_KEY in p.read_text(encoding="utf-8", errors="ignore") for p in w.logs.rglob("*") if p.is_file()))
        state = json.loads(w.cfg().state_file.read_text(encoding="utf-8"))
        check("state file persisted (toggle on, session cleaned)",
              state["enabled"] is True and state["sessions"]["pkgA"]["state"] == "cleaned")
        check("nothing processed twice on the next tick", asyncio.run(svc.tick()) is False and w.count("extract") == 1)


# ============================================================
# 2: refused package
# ============================================================

def test_refused_validation() -> None:
    print("\n[2] a package that fails validation is refused with the reason")
    with tempfile.TemporaryDirectory() as tmp:
        w = World(Path(tmp))
        pkg = w.add_package("pkgBad", "S2", bad=True)
        svc = w.service(keep_run_frames=True)
        svc.enable()
        asyncio.run(drain(svc))
        s = svc._sessions["pkgBad"]
        check("refused session is failed", s["state"] == "failed", s["state"])
        check("failed in validate", s.get("error_stage") == "validate", str(s.get("error_stage")))
        check("error quotes the validator's FAIL line", "manifest checksums" in (s.get("error") or ""), str(s.get("error")))
        check("validator output kept", "FAIL  manifest checksums" in s["validation"]["output"])
        check("exit code kept", s["validation"]["exit_code"] == 1)
        check("nothing extracted", w.count("extract") == 0)
        check("worker untouched", w.worker.events == [] and w.worker.is_running)
        check("DB told failed with the reason", any(st == "failed" and "manifest" in (err or "")
                                                    for _, st, err in w.store.states))
        rows = svc.sessions()
        check("sessions view shows the validator output for a refused package",
              "FAIL" in ((rows[0].get("validation") or {}).get("output") or ""))
        asyncio.run(drain(svc))
        check("a refused package is not retried while unchanged", w.count("validate") == 1)
        (pkg / "BAD").unlink()
        asyncio.run(drain(svc))
        check("a changed package is validated again and processed", svc._sessions["pkgBad"]["state"] == "cleaned",
              svc._sessions["pkgBad"]["state"])
        check("VIDEO_KEEP_RUN_FRAMES keeps the run's best frames",
              all((w.runs / "S2" / "frames" / f"{seg_id('S2', k)}.png").exists() for k in SENT_K)
              and not (w.runs / "S2" / "frames" / f"{seg_id('S2', 1)}.png").exists())


# ============================================================
# 3: failure mid-way
# ============================================================

def test_failure_midway() -> None:
    print("\n[3] a failure mid-way marks failed with the error and restarts the worker")
    with tempfile.TemporaryDirectory() as tmp:
        w = World(Path(tmp))
        w.add_package("pkgF", "S3")
        w.fail_at["classify"] = (2, "STOP: cannot reach http://127.0.0.1:8000/health (connection refused)")
        svc = w.service()
        svc.enable()
        asyncio.run(drain(svc))
        s = svc._sessions["pkgF"]
        check("session failed", s["state"] == "failed", s["state"])
        check("error names the cause", "cannot reach" in (s.get("error") or ""), str(s.get("error")))
        check("failed in classify", s.get("error_stage") == "classify", str(s.get("error_stage")))
        check("worker restarted after the failure", w.worker.is_running and w.worker.events[-1] == "start",
              str(w.worker.events))
        check("hold released", w.worker.hold_reason is None)
        check("worker rule held", not w.violations, str(w.violations))
        failed = [e for e in log_events(w, "S3") if e["event"] == "failed"]
        check("log has the failure with a stack trace", bool(failed) and "Traceback" in failed[-1].get("traceback", ""))
        check("DB told failed", "failed" in w.store.db_states("S3"))
        # retry resumes from classifying without re-extracting
        svc.retry("S3")
        check("retry resumes from its stage", svc._sessions["pkgF"]["state"] == "classifying",
              svc._sessions["pkgF"]["state"])
        asyncio.run(drain(svc))
        check("retried session cleaned", svc._sessions["pkgF"]["state"] == "cleaned", str(svc._sessions["pkgF"].get("error")))
        check("extraction not repeated on retry", w.count("extract") == 1)

    with tempfile.TemporaryDirectory() as tmp:
        w = World(Path(tmp))
        w.add_package("pkgE", "S4")
        w.fail_at["extract"] = (1, "S4: FAILED: no GNSS fixes")
        svc = w.service()
        svc.enable()
        asyncio.run(drain(svc))
        s = svc._sessions["pkgE"]
        check("extraction failure marks failed", s["state"] == "failed" and "no GNSS fixes" in (s.get("error") or ""),
              str(s.get("error")))
        check("worker restarted after an extraction failure", w.worker.is_running and w.worker.events[-1] == "start")

    with tempfile.TemporaryDirectory() as tmp:
        w = World(Path(tmp), worker_running=False)
        w.add_package("pkgN", "S5")
        w.fail_at["extract"] = (1, "boom")
        svc = w.service()
        svc.enable()
        asyncio.run(drain(svc))
        check("a worker that was stopped stays stopped", not w.worker.is_running and "start" not in w.worker.events,
              str(w.worker.events))

    with tempfile.TemporaryDirectory() as tmp:
        w = World(Path(tmp))
        w.add_package("pkgV", "S6")
        w.verify_fails = True
        svc = w.service()
        svc.enable()
        asyncio.run(drain(svc))
        s = svc._sessions["pkgV"]
        check("failed verification is not cleaned up", s["state"] == "failed" and (w.batches / "batch_S6").exists(),
              s["state"])


# ============================================================
# 4: disable stops after the current item; resume
# ============================================================

def test_disable_and_resume() -> None:
    print("\n[4] disabling stops after the current item; resumes from its stage")
    with tempfile.TemporaryDirectory() as tmp:
        w = World(Path(tmp))
        w.add_package("pkgA", "SA")
        w.add_package("pkgB", "SB")
        svc = w.service()
        svc.enable()
        w.on_step["extract"] = svc.disable          # operator turns it off during extraction of A
        asyncio.run(drain(svc))
        a, b = svc._sessions["pkgA"], svc._sessions["pkgB"]
        check("current extraction finished, then stopped", a["state"] == "extracting" and w.count("extract") == 1,
              a["state"])
        check("nothing classified after disable", w.count("classify") == 0 and w.count("export") == 0)
        check("next session untouched", b["state"] == "queued" and w.count("validate") == 1)
        check("worker restarted while paused", w.worker.is_running)
        check("disabled service does no work", asyncio.run(svc.tick()) is False)
        check("toggle persisted off", json.loads(w.cfg().state_file.read_text(encoding="utf-8"))["enabled"] is False)

        # A new service instance (restart) keeps the queue and the toggle.
        svc2 = w.service()
        check("restart keeps the paused session", svc2._sessions["pkgA"]["state"] == "extracting")
        w.on_step["classify"] = svc2.disable         # off again, during the first runner chunk
        svc2.enable()
        asyncio.run(drain(svc2))
        a = svc2._sessions["pkgA"]
        check("resumed without re-extracting", w.count("extract") == 1)
        check("stopped after the current chunk", a["state"] == "classifying" and w.count("classify") == 1, a["state"])
        check("progress kept", (a.get("progress") or {}).get("answered") == 2, str(a.get("progress")))
        n_before = len(w.classify_argvs)
        svc2.enable()
        asyncio.run(drain(svc2))
        resumed = [a for a in w.classify_argvs[n_before:] if Path(a[2]).name == "batch_SA"]
        check("a resumed classify pass re-verifies the batch on its first chunk",
              bool(resumed) and "--skip-batch-check" not in resumed[0], str(resumed[:1]))
        check("first session cleaned after resume", svc2._sessions["pkgA"]["state"] == "cleaned",
              str(svc2._sessions["pkgA"].get("error")))
        check("second session cleaned after it", svc2._sessions["pkgB"]["state"] == "cleaned")
        resp = w.runs / "SA" / "vlm" / "responses.jsonl"
        ok = [r["request_id"] for r in vp.read_jsonl(resp) if not r.get("error")]
        check("runner resumed: each image answered exactly once", sorted(ok) == sorted(seg_id("SA", k) for k in SENT_K),
              str(ok))
        check("worker rule held throughout", not w.violations, str(w.violations))


# ============================================================
# 8: shutdown mid-extraction, resume on the next start
# ============================================================

def test_shutdown_and_resume() -> None:
    print("\n[8] shutdown kills the running command; the session resumes on restart")
    with tempfile.TemporaryDirectory() as tmp:
        w = World(Path(tmp))
        w.add_package("pkgS", "SS")
        w.block_extract.set()

        async def run_and_stop():
            svc = w.service()
            svc.enable()
            await svc.start()
            for _ in range(250):
                if w.in_extract.is_set():
                    break
                await asyncio.sleep(0.02)
            t0 = time.monotonic()
            await svc.stop()
            return svc, time.monotonic() - t0

        svc, took = asyncio.run(run_and_stop())
        s = svc._sessions["pkgS"]
        check("extraction was running", w.in_extract.is_set())
        check("stop returned promptly", took < 5.0, f"{took:.1f} s")
        check("session keeps its stage", s["state"] == "extracting", s["state"])
        check("hold released, worker not restarted during shutdown",
              w.worker.hold_reason is None and not w.worker.is_running, str(w.worker.events))
        w.block_extract.clear()
        svc2 = w.service()
        check("toggle on after restart", svc2.status_summary()["enabled"] is True)
        asyncio.run(drain(svc2))
        check("incomplete run re-extracted and session cleaned",
              svc2._sessions["pkgS"]["state"] == "cleaned" and w.count("extract") == 2,
              f"{svc2._sessions['pkgS']['state']}, extracts {w.count('extract')}")


# ============================================================
# 9: helpers, real runner, real worker hold, routes, main.py
# ============================================================

def test_helpers() -> None:
    print("\n[9a] helpers")
    check("percentile p50 of 3", vp.percentile([20.0, 10.0, 14.0], 50) == 14.0)
    check("percentile p95 of 3 (linear)", abs(vp.percentile([10.0, 14.0, 20.0], 95) - 19.4) < 1e-9)
    check("percentile p95 of 1..100 = 95.05", abs(vp.percentile([float(i) for i in range(1, 101)], 95) - 95.05) < 1e-9)
    check("percentile of nothing", vp.percentile([], 50) is None)
    lines = [{"request_id": "a", "error": "HTTP 400", "http_status": 400},
             {"request_id": "b", "error": "HTTP 429", "http_status": 429},
             {"request_id": "c", "error": "HTTP 400", "http_status": 400},
             {"request_id": "c", "error": None, "http_status": 200, "response": {}}]
    ans = set(vp.answers_from_responses(lines))
    check("answers skip error lines; a later success wins", ans == {"c"}, str(ans))
    check("only non-retryable 4xx count as rejected", vp.permanently_rejected(lines, ans) == {"a"})
    check("redact hides the API key",
          vp.redact(["x", "--api-key", "k", "--limit", "3"]) == ["x", "--api-key", "***", "--limit", "3"])
    os.environ["SOME_API_KEY"] = "zzz"
    os.environ["SUPABASE_SERVICE_KEY_TEST"] = "zzz"
    env = vp.child_env()
    check("secrets not passed to subprocesses", "SOME_API_KEY" not in env and "SUPABASE_SERVICE_KEY_TEST" not in env
          and "PATH" in {k.upper() for k in env})
    del os.environ["SOME_API_KEY"], os.environ["SUPABASE_SERVICE_KEY_TEST"]
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "run"
        check("no report.json = incomplete", vp.run_status(run)[0] is False)
        write(run / "report.json", json.dumps({"mode": "crops"}))
        check("crop-mode report = incomplete", vp.run_status(run)[0] is False)
        write(run / "report.json", json.dumps({"mode": "segments"}))
        write(run / "sidecars" / "x.json", json.dumps({"id": "x", "status": "send", "frame": {"file": "frames/x.png"}}))
        ok, why = vp.run_status(run)
        check("missing send frame = incomplete", ok is False and "missing" in why, why)
        write(run / "frames" / "x.png", b"p")
        check("complete run", vp.run_status(run) == (True, ""))
    # toggle default and persistence
    old = os.environ.pop("VIDEO_PIPELINE_ENABLED", None)
    check("toggle defaults to off", VideoPipelineConfig.from_env().enabled_default is False)
    os.environ["VIDEO_PIPELINE_ENABLED"] = "true"
    check("env VIDEO_PIPELINE_ENABLED=true turns it on", VideoPipelineConfig.from_env().enabled_default is True)
    os.environ.pop("VIDEO_PIPELINE_ENABLED")
    old_ff = os.environ.get("FFPROBE_DIR")
    os.environ["FFPROBE_DIR"] = r"D:\tools\ffmpeg\bin"
    check("FFPROBE_DIR configures the ffprobe folder", VideoPipelineConfig.from_env().ffprobe_dir == Path(r"D:\tools\ffmpeg\bin"))
    if old_ff is None:
        os.environ.pop("FFPROBE_DIR")
    else:
        os.environ["FFPROBE_DIR"] = old_ff
    if old is not None:
        os.environ["VIDEO_PIPELINE_ENABLED"] = old
    with tempfile.TemporaryDirectory() as tmp:
        w = World(Path(tmp))
        w.service(enabled_default=False).enable()
        check("persisted toggle wins over the env default", w.service(enabled_default=False).status_summary()["enabled"])
        repo_missing = w.service(geoai_repo=Path(tmp) / "nope")
        w.add_package("pkgR", "SR")
        asyncio.run(drain(repo_missing))
        s = repo_missing._sessions["pkgR"]
        check("missing GeoAI_Video repo keeps the package queued (not refused)",
              s["state"] == "queued" and "GeoAI_Video" in (repo_missing._last_error or ""), s["state"])


def test_real_runner() -> None:
    print("\n[9b] real subprocess runner")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.txt"
        cancel = threading.Event()
        code = vp.run_process("validate", [sys.executable, "-c", "print('hello'); raise SystemExit(3)",
                                           "--api-key", API_KEY], Path(tmp), out, 60.0, cancel)
        text = out.read_text(encoding="utf-8")
        check("exit code returned", code == 3, str(code))
        check("output captured", "hello" in text)
        check("API key redacted in the command line", API_KEY not in text and "***" in text)
        out3 = Path(tmp) / "out3.txt"
        extra = Path(tmp) / "ffbin"
        vp.run_process("validate", [sys.executable, "-c", "import os; print(os.environ['PATH'])"], Path(tmp), out3,
                       60.0, cancel, extra_path=[extra])
        first = [ln for ln in out3.read_text(encoding="utf-8").splitlines() if not ln.startswith("#")][0]
        check("ffprobe folder first on the tools' PATH", first.split(os.pathsep)[0] == str(extra), first[:120])
        out2 = Path(tmp) / "out2.txt"
        threading.Timer(0.5, cancel.set).start()
        t0 = time.monotonic()
        try:
            vp.run_process("extract", [sys.executable, "-c", "import time; time.sleep(60)"], Path(tmp), out2, None, cancel)
            check("cancel raises Interrupted", False, "returned normally")
        except Interrupted:
            check("cancel raises Interrupted", True)
        check("cancel kills the process within seconds", time.monotonic() - t0 < 20)
        cancel.clear()
        try:
            vp.run_process("extract", [sys.executable, "-c", "import time; time.sleep(60)"], Path(tmp), out2, 0.5, cancel)
            check("timeout raises StageError", False)
        except vp.StageError as e:
            check("timeout raises StageError", "did not finish" in str(e))


def test_real_worker_hold() -> None:
    print("\n[9c] PipelineWorker hold (real class)")
    from app.worker import PipelineWorker

    w = PipelineWorker(supabase_client=None, image_client=None, classifier=None, worker_id="test")
    w.hold("video pipeline: test")
    r = asyncio.run(w.start())
    check("start during a hold is refused", r["status"] == "held" and not w.is_running, str(r))
    check("hold shown in the metrics snapshot", w.metrics_snapshot()["hold_reason"] == "video pipeline: test")
    check("release reports the refused start", w.release_hold() is True)
    check("hold cleared", w.hold_reason is None and w.release_hold() is False)


def test_routes() -> None:
    print("\n[9d] /operator/video routes and the main.py include")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app import video_routes

    with tempfile.TemporaryDirectory() as tmp:
        w = World(Path(tmp))
        w.add_package("pkgA", "S1")
        w.add_package("pkgBad", "S2", bad=True)
        svc = w.service()
        app = FastAPI()
        app.include_router(video_routes.router)
        client = TestClient(app)
        vp.set_service(None)
        check("503 before the service exists", client.get("/operator/video/status").status_code == 503)
        vp.set_service(svc)
        try:
            r = client.post("/operator/video/enable")
            check("enable", r.status_code == 200 and r.json()["enabled"] is True, r.text[:200])
            asyncio.run(drain(svc))
            st = client.get("/operator/video/status").json()
            check("status has counts, checks and paths",
                  st["counts"]["cleaned"] == 1 and st["counts"]["failed"] == 1 and st["config"]["inbox"] == str(w.inbox),
                  json.dumps(st)[:300])
            check("status never exposes the API key", API_KEY not in json.dumps(st))
            items = client.get("/operator/video/sessions").json()["items"]
            by = {i["session_id"]: i for i in items}
            check("sessions list states and metrics",
                  by["S1"]["state"] == "cleaned" and by["S1"]["metrics"]["route_m"] == 22.5)
            check("failed session carries its error", "manifest" in (by["S2"].get("error") or ""))
            log = client.get("/operator/video/sessions/S1/log")
            check("session log", log.status_code == 200 and any(e["event"] == "end" for e in log.json()["events"]))
            check("unknown session log 404", client.get("/operator/video/sessions/nope/log").status_code == 404)
            check("retry of a non-failed session 409", client.post("/operator/video/sessions/S1/retry").status_code == 409)
            r = client.post("/operator/video/sessions/S2/retry")
            check("retry of a failed session", r.status_code == 200 and r.json()["state"] == "queued", r.text[:200])
            r = client.post("/operator/video/disable")
            check("disable", r.status_code == 200 and r.json()["enabled"] is False)
        finally:
            vp.set_service(None)

    import app.main as main

    paths = {getattr(r, "path", "") for r in main.app.routes}
    check("main.py includes the video router", {"/operator/video/status", "/operator/video/enable",
                                                "/operator/video/disable", "/operator/video/sessions",
                                                "/operator/video/sessions/{session_id}/log"} <= paths)
    check("main.py lifespan wrapped by install()",
          getattr(main.app.router.lifespan_context, "__name__", "") == "lifespan_with_video_pipeline")
    html = (PROJECT_ROOT / "operator_ui" / "index.html").read_text(encoding="utf-8")
    check("operator UI has the video_pipeline toggle and session table",
          all(x in html for x in ('id="vp-toggle"', 'id="vp-rows"', "/operator/video/status", "/operator/video/")))


def test_local_images_route() -> None:
    print("\n[9e] /local-images serves stored images only from LOCAL_IMAGE_ROOT")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app import video_routes

    with tempfile.TemporaryDirectory() as tmp:
        root, outside = Path(tmp) / "images", Path(tmp) / "elsewhere"
        png = b"\x89PNG\r\n\x1a\nfake"
        write(root / "S1" / "S1_seg00001.png", png)
        write(root / "S1" / "notes.txt", "not an image")
        write(outside / "x.png", png)
        old = os.environ.get("LOCAL_IMAGE_ROOT")
        os.environ["LOCAL_IMAGE_ROOT"] = str(root)
        try:
            app = FastAPI()
            app.include_router(video_routes.image_router)
            client = TestClient(app)

            def get(path: str):
                return client.get("/local-images", params={"path": path})

            r = get(str(root / "S1" / "S1_seg00001.png"))
            check("a stored PNG is served as image/png", r.status_code == 200 and r.content == png
                  and r.headers["content-type"] == "image/png", f"{r.status_code} {r.headers.get('content-type')}")
            check("a file outside the root is 404", get(str(outside / "x.png")).status_code == 404)
            check("'..' out of the root is 404", get(str(root / ".." / "elsewhere" / "x.png")).status_code == 404)
            check("a non-image inside the root is 404", get(str(root / "S1" / "notes.txt")).status_code == 404)
            check("a missing image is 404", get(str(root / "S1" / "gone.png")).status_code == 404)
            check("a URL is not a local path: 404", get("https://res.cloudinary.com/x/a.png").status_code == 404)
            check("a relative path is 404", get("S1/S1_seg00001.png").status_code == 404)
        finally:
            if old is None:
                os.environ.pop("LOCAL_IMAGE_ROOT", None)
            else:
                os.environ["LOCAL_IMAGE_ROOT"] = old


def test_live_view() -> None:
    print("\n[9f] live view of the session in progress (operator page's live card)")
    import io as _io

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from PIL import Image

    from app import video_routes

    def png(w: int, h: int, color) -> bytes:
        buf = _io.BytesIO()
        Image.new("RGB", (w, h), color).save(buf, format="PNG")
        return buf.getvalue()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sid = "20261027T050257Z_SM-G991B-F0D1_d47c"
        # -- extract_progress: the LAST progress line wins; noise, other lines and a cut-off line are ignored
        out = root / "extract.txt"
        write(out, "\n".join([
            '2026-10-27 12:00:00,000 INFO geoai: plan: 528 segments of 5.0 m',
            '2026-10-27 12:00:00,001 INFO geoai: progress {"done": 0, "phase": "plan", "route_m": 2637.5, "total": 528}',
            '2026-10-27 12:00:05,000 INFO geoai: progress {"done": 40, "phase": "bank", "total": 175}',
            '2026-10-27 12:03:00,000 INFO geoai: progress {"done": 120, "k": 119, "phase": "decode", "segment": "'
            + sid + '_seg00119", "send": 100, "skip": 20, "status": "send", "total": 528}',
            'some other line', '2026-10-27 12:03:01,000 INFO geoai: progress {"done": 121, "pha']))
        p = vp.extract_progress(out)
        check("extract_progress reads the last complete progress line",
              p is not None and p["phase"] == "decode" and (p["done"], p["total"], p["send"]) == (120, 528, 100), str(p))
        check("extract_progress: missing file -> None", vp.extract_progress(root / "nope.txt") is None)
        check("extract_progress: no file given -> None", vp.extract_progress(None) is None)

        # -- latest_frame_id: newest PNG by mtime
        run = root / "runs" / sid
        for k, t in ((3, 1_000), (7, 3_000), (5, 2_000)):
            f = run / "frames" / f"{sid}_seg{k:05d}.png"
            write(f, png(64, 36, (k * 30, 0, 0)))
            os.utime(f, ns=(t * 1_000_000_000, t * 1_000_000_000))
        write(run / "frames" / "notes.txt", "x")
        check("latest_frame_id = newest frame", vp.latest_frame_id(run) == f"{sid}_seg00007")
        check("latest_frame_id: no frames dir -> None", vp.latest_frame_id(root / "empty") is None)

        # -- classify_live: 4 requests; 0 answered (14 s), 1 Stage-1 Normal (10 s), 2 errored once, 3 pending
        batch = root / "batches" / f"batch_{sid}"
        ids = [f"{sid}_seg{k:05d}" for k in range(4)]
        write(batch / "requests.jsonl", "".join(json.dumps({"request_id": i, "kind": "segment"}) + "\n" for i in ids))
        for k, i in enumerate(ids):
            write(batch / "records" / f"{i}.json", json.dumps({
                "id": i, "status": "send", "segment": {"k": k, "s0_m": 100.0 + 5 * k, "length_m": 5.0,
                                                       "centre": {"lat": 12.9 + k * 1e-5, "lon": 77.6},
                                                       "chainage_start_km": None},
                "frame": {"utc_iso": "2026-10-27T05:03:00.000Z"}, "gate": {"decision_if_gated": "skip", "reasons": []}}))
            write(batch / "images" / f"{i}.png", png(64, 36, (0, k * 50, 0)))
        lines = [
            {"request_id": ids[0], "seconds": 14.0, "response": {"stage1_label": "Distress", "stage1_confidence": 0.91,
                                                                  "distress_types": ["Potholes"], "severity": "Medium"}},
            {"request_id": ids[1], "seconds": 10.0, "response": {"stage1_label": "Normal", "stage1_confidence": 0.88,
                                                                  "distress_types": [], "severity": "None"}},
            {"request_id": ids[2], "seconds": 3.0, "http_status": 500, "error": "HTTP 500", "response": {"detail": "x"}},
        ]
        write(batch / "responses.jsonl", "".join(json.dumps(x) + "\n" for x in lines))
        c = vp.classify_live(batch)
        check("classify_live counts", c is not None and (c["requests"], c["answered"], c["rejected"], c["remaining"])
              == (4, 2, 0, 2), str(c)[:300])
        # By hand: mean of 14 and 10 = 12 s; 2 remaining -> ETA 24 s; median of two values = 12.
        check("classify_live speed and ETA", (c["s_per_image_mean"], c["s_per_image_median"], c["eta_s"]) == (12.0, 12.0, 24.0),
              str((c["s_per_image_mean"], c["s_per_image_median"], c["eta_s"])))
        check("in flight = first unanswered request, with its record",
              c["in_flight"]["segment_id"] == ids[2] and c["in_flight"]["k"] == 2 and c["in_flight"]["s0_m"] == 110.0
              and c["in_flight"]["gate_decision_if_gated"] == "skip", str(c["in_flight"]))
        check("last answer is the Stage-1 Normal one, read as Normal",
              c["last_answer"]["segment_id"] == ids[1] and c["last_answer"]["labels"] == ["Normal"]
              and c["last_answer"]["seconds"] == 10.0, str(c["last_answer"]))
        check("last error reported (it is the last line)",
              c["last_error"] and c["last_error"]["segment_id"] == ids[2] and "500" in c["last_error"]["error"])
        check("classify_live: no requests -> None", vp.classify_live(root / "nobatch") is None)

        # -- service: _live per state, frame_file, /video-frame
        w = World(root / "w")
        svc = w.service(image_root=root / "images")
        key = "pkgL"
        svc._sessions[key] = {"key": key, "session_id": sid, "state": "extracting", "run_dir": str(run),
                              "batch_dir": str(batch), "metrics": {"timings_s": {"validate": 1.5}},
                              "stage": "extract", "stage_started_utc": "2026-10-27T12:00:00Z",
                              "live_output": {"step": "extract", "file": str(out)}, "history": []}
        live = svc._live(svc._sessions[key])
        check("live while extracting: stage, progress, newest frame",
              live["stage"] == "extract" and live["extract"]["progress"]["done"] == 120
              and live["frame"] == f"{sid}_seg00007" and live["timings_s"] == {"validate": 1.5}, str(live)[:300])
        svc._sessions[key].update(state="classifying", stage="classify")
        live = svc._live(svc._sessions[key])
        check("live while classifying: the frame shown is the segment in flight",
              live["frame"] == ids[2] and live["classify"]["answered"] == 2, str(live)[:300])
        svc._sessions[key].update(state="storing", stage="store")
        live = svc._live(svc._sessions[key])
        check("live while storing: no extract/classify block", "extract" not in live and "classify" not in live)

        write(root / "images" / sid / f"{sid}_seg00009.png", png(32, 18, (9, 9, 9)))
        check("frame_file: run frames/", svc.frame_file(key, f"{sid}_seg00005") == run / "frames" / f"{sid}_seg00005.png")
        check("frame_file: batch images/ (not in the run's frames)",
              svc.frame_file(key, ids[0]) == batch / "images" / f"{ids[0]}.png")
        check("frame_file: image store after cleanup", svc.frame_file(key, f"{sid}_seg00009") == root / "images" / sid / f"{sid}_seg00009.png")
        check("frame_file: another session's id -> None", svc.frame_file(key, "OTHER_seg00001") is None)
        check("frame_file: a path -> None", svc.frame_file(key, "..\\..\\x") is None)
        check("frame_file: unknown session -> None", svc.frame_file("nope", f"{sid}_seg00005") is None)

        app = FastAPI()
        app.include_router(video_routes.image_router)
        client = TestClient(app)
        old = vp.get_service()
        vp.set_service(svc)
        try:
            r = client.get("/video-frame", params={"key": key, "segment": f"{sid}_seg00007", "w": 32})
            im = Image.open(_io.BytesIO(r.content)) if r.status_code == 200 else None
            # w=32 is clamped to the 160 px minimum, wider than the 64 px frame: never enlarged.
            check("/video-frame: JPEG, a small frame is not enlarged",
                  im is not None and r.headers["content-type"] == "image/jpeg" and im.format == "JPEG"
                  and im.size == (64, 36), f"{r.status_code} {im.size if im else None}")
            write(run / "frames" / f"{sid}_seg00011.png", png(2000, 1000, (1, 2, 3)))
            r = client.get("/video-frame", params={"key": key, "segment": f"{sid}_seg00011", "w": 640})
            im = Image.open(_io.BytesIO(r.content)) if r.status_code == 200 else None
            check("/video-frame: a large frame is scaled to w, aspect kept (2000x1000 -> 640x320)",
                  im is not None and im.size == (640, 320), f"{r.status_code} {im.size if im else None}")
            r2 = client.get("/video-frame", params={"key": key, "segment": "OTHER_seg00001"})
            check("/video-frame: unknown segment -> 404", r2.status_code == 404)
            r3 = client.get("/video-frame", params={"key": key, "segment": f"{sid}_seg00042"})
            check("/video-frame: missing frame -> 404", r3.status_code == 404)
        finally:
            vp.set_service(old)


def main() -> int:
    test_helpers()
    test_happy_path()
    test_refused_validation()
    test_failure_midway()
    test_disable_and_resume()
    test_shutdown_and_resume()
    test_real_runner()
    test_real_worker_hold()
    test_routes()
    test_local_images_route()
    test_live_view()
    print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} checks passed")
    if FAILURES:
        print("FAILED:", *FAILURES, sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
