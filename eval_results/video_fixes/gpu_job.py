"""Run one GPU job under the shared lock C:\\GeoAI\\gpu.lock, safely enough to survive a dead agent.

    python gpu_job.py --purpose "what and who" --log C:/path/job.log [--status C:/path/job.status.json] -- cmd args...

- The lock is the directory C:\\GeoAI\\gpu.lock (mkdir is atomic). owner.txt inside names this process's PID,
  the purpose and the UTC start; it is written to a temp file and renamed, so it is never half-written.
- A lock whose owner.txt names a PID ('pid: N' or 'pid=N') that is no longer running is stale and is taken
  over (logged); if the dead owner recorded its child (child_pid + child_created), that exact process tree is
  killed first, because an orphaned child may still be using the GPU. A lock without a readable PID is never
  taken over: we wait.
- The command runs as a child with stdout+stderr appended to --log. The lock is removed in `finally`, but only
  if owner.txt still names this PID, so a lock that someone else took over is never deleted by us. The
  launcher also tries to put itself in a kill-on-close Job Object; on this machine children can escape it
  (silent breakaway from an outer job), hence the child record above.
- --status gets an atomically written JSON {state, pid, child_pid, purpose, cmd, log, started_utc, ended_utc,
  exit_code} for resume notes (STATUS.md points at it).
Launch it detached so it outlives the agent that started it (PowerShell Start-Process -WindowStyle Hidden).
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

LOCK = Path(os.environ.get("GPU_JOB_LOCK", r"C:\GeoAI\gpu.lock"))  # override only for self-tests


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def pid_alive(pid: int) -> bool:
    """True if a process with this PID exists and has not exited (Windows: OpenProcess + GetExitCodeProcess).
    Never uses os.kill: on Windows os.kill(pid, 0) TERMINATES the process."""
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return ctypes.GetLastError() == 5    # access denied: exists but not ours
    try:
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    finally:
        k32.CloseHandle(h)


def kill_children_with_me() -> bool:
    """Windows: put this process in a Job Object with KILL_ON_JOB_CLOSE. Children inherit the job, so if this
    launcher dies (even TerminateProcess) the OS closes the only job handle and kills the whole tree: no orphan
    keeps using the GPU after the lock goes stale."""
    if os.name != "nt":
        return False
    from ctypes import wintypes

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in ("r", "w", "o", "rb", "wb", "ob")]

    class BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class EXTENDED(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    k32 = ctypes.windll.kernel32
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    job = k32.CreateJobObjectW(None, None)
    if not job:
        return False
    info = EXTENDED()
    info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not k32.SetInformationJobObject(wintypes.HANDLE(job), 9, ctypes.byref(info), ctypes.sizeof(info)):
        return False
    ok = bool(k32.AssignProcessToJobObject(wintypes.HANDLE(job), wintypes.HANDLE(k32.GetCurrentProcess())))
    globals()["_JOB_HANDLE"] = job  # keep the handle open for the life of this process
    return ok


def owner_fields() -> dict:
    try:
        return dict(ln.split(":", 1) for ln in (LOCK / "owner.txt").read_text(encoding="utf-8").splitlines()
                    if ":" in ln)
    except OSError:
        return {}


def owner_pid() -> int | None:
    """PID of the lock owner. Reads our 'pid: N' lines and the other agents' one-line 'pid=N; ...' format."""
    try:
        m = re.search(r"\bpid\s*[:=]\s*(\d+)", (LOCK / "owner.txt").read_text(encoding="utf-8"))
    except OSError:
        return None
    return int(m.group(1)) if m else None


def creation_time(pid: int) -> int | None:
    """Process creation time (FILETIME as int) or None; used to tell a process from a later one with the same PID."""
    if os.name != "nt":
        return None
    from ctypes import wintypes
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x1000, False, pid)
    if not h:
        return None
    try:
        t = [wintypes.FILETIME() for _ in range(4)]
        if not k32.GetProcessTimes(h, *[ctypes.byref(x) for x in t]):
            return None
        return (t[0].dwHighDateTime << 32) | t[0].dwLowDateTime
    finally:
        k32.CloseHandle(h)


def kill_stale_child(fh) -> None:
    """The dead owner's child (recorded with its creation time) may still be running and using the GPU:
    the Job Object does not always hold (a parent job with silent breakaway lets children escape). Kill that
    exact process tree; the creation time guards against a reused PID."""
    f = owner_fields()
    try:
        cpid, ctime = int(f["child_pid"].strip()), int(f["child_created"].strip())
    except (KeyError, ValueError):
        return
    if pid_alive(cpid) and creation_time(cpid) == ctime:
        log(fh, f"stale owner's child {cpid} still running: killing its process tree")
        subprocess.call(["taskkill", "/PID", str(cpid), "/T", "/F"], stdout=fh, stderr=subprocess.STDOUT)


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def log(fh, msg: str) -> None:
    fh.write(f"[gpu_job {utc()}] {msg}\n")
    fh.flush()


def acquire(fh, purpose: str, poll_s: float) -> None:
    waited = False
    while True:
        try:
            LOCK.mkdir()
            write_atomic(LOCK / "owner.txt", f"pid: {os.getpid()}\npurpose: {purpose}\nstart: {utc()}\n")
            log(fh, f"lock acquired (pid {os.getpid()})")
            return
        except FileExistsError:
            pid = owner_pid()
            if pid is not None and pid != os.getpid() and not pid_alive(pid):
                log(fh, f"stale lock: owner pid {pid} is not running; taking it over")
                kill_stale_child(fh)
                shutil.rmtree(LOCK, ignore_errors=True)
                continue
            if not waited:
                try:
                    who = (LOCK / "owner.txt").read_text(encoding="utf-8").replace("\n", " | ")
                except OSError:
                    who = "?"
                log(fh, f"waiting for gpu.lock: {who}")
                waited = True
            time.sleep(poll_s)


def release(fh) -> None:
    if owner_pid() == os.getpid():
        shutil.rmtree(LOCK, ignore_errors=True)
        log(fh, "lock released")
    else:
        log(fh, "lock not ours any more; left alone")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--purpose", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--status")
    ap.add_argument("--poll-s", type=float, default=30.0)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        ap.error("no command")
    st = {"state": "waiting", "pid": os.getpid(), "child_pid": None, "purpose": a.purpose, "cmd": cmd,
          "log": a.log, "started_utc": utc(), "ended_utc": None, "exit_code": None}

    def status(**kw):
        st.update(kw)
        if a.status:
            write_atomic(Path(a.status), json.dumps(st, indent=1))

    Path(a.log).parent.mkdir(parents=True, exist_ok=True)
    with open(a.log, "a", encoding="utf-8") as fh:
        log(fh, f"start pid {os.getpid()}; children die with this process: {kill_children_with_me()}")
        status()
        acquire(fh, a.purpose, a.poll_s)
        code = None
        try:
            log(fh, "run: " + " ".join(cmd))
            p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
            f = owner_fields()
            write_atomic(LOCK / "owner.txt",
                         f"pid: {os.getpid()}\npurpose: {a.purpose}\nstart: {f.get('start', '').strip()}\n"
                         f"child_pid: {p.pid}\nchild_created: {creation_time(p.pid)}\nlog: {a.log}\n")
            status(state="running", child_pid=p.pid)
            code = p.wait()
            log(fh, f"exit {code}")
        finally:
            release(fh)
            status(state="done" if code == 0 else "failed", ended_utc=utc(), exit_code=code)
    return 0 if code == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
