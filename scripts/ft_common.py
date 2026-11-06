"""
Shared pieces of the post-training study (PostTrainBench-style, paper §6.11).

  * GPU mutual exclusion through C:\\GeoAI\\gpu.lock (the convention every GPU job
    on this machine uses): mkdir is atomic; owner.txt says who holds it.
  * Atomic writes (temp file + os.replace) so an interrupted job never leaves a
    half-written JSON or checkpoint.
  * Training data: Attain WS_V1.0 and OS_V1.0 with labels mapped through the SAME
    tables the evaluation uses (07_cross_dataset_eval.ATTAIN_CLASS_MAP and
    stage2_probe_report.PROBE_KEYS), and the WS_V2.0 dev/test split from
    stage2_probe_report.split().

Nothing here touches the model; CPU tests import it freely.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FT_DIR = PROJECT_ROOT / "eval_results" / "finetune"
ART_DIR = Path(os.environ.get("FT_ARTIFACT_DIR", r"C:\GeoAI\finetune"))
LOCK_DIR = Path(os.environ.get("GPU_LOCK_DIR", r"C:\GeoAI\gpu.lock"))
ATTAIN_ROOT = PROJECT_ROOT / "Attain" / "Attain" / "Attain"

_spec = importlib.util.spec_from_file_location(
    "attain_eval_ft", PROJECT_ROOT / "scripts" / "07_cross_dataset_eval.py")
_ae = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ae)

# The evaluation's own class tables (imported, not copied).
from scripts.stage2_probe_report import (  # noqa: E402
    ALL_CLASSES, BLOCK, HEADLINE, PATCH, PROBE_KEYS, split as v2_split)

SPLIT_B_SEED = 20261106     # dev blocks -> arm B train (2/3) / calibration (1/3)


# ============================================================
# Atomic writes
# ============================================================

def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj, indent=1) -> None:
    atomic_write_text(path, json.dumps(obj, indent=indent, default=str))


def read_json(path: Path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ============================================================
# GPU lock
# ============================================================

def _pid_alive(pid: int) -> bool:
    """psutil (OpenProcess) - never os.kill, which TERMINATES a process on Windows."""
    try:
        import psutil
        return psutil.pid_exists(pid)
    except Exception:
        return True   # unknown -> assume alive (never steal a lock on a guess)


def _owner_text() -> str:
    try:
        return (LOCK_DIR / "owner.txt").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _owner_pid() -> int | None:
    """Owner PID in either format used on this machine ('pid: N' lines or 'pid=N; ...')."""
    import re
    m = re.search(r"(?<![a-z_])pid\s*[:=]\s*(\d+)", _owner_text())
    return int(m.group(1)) if m else None


def _owner_field(name: str):
    import re
    m = re.search(rf"\b{name}\s*[:=]\s*(\d+)", _owner_text())
    return int(m.group(1)) if m else None


def _kill_stale_child() -> None:
    """A dead owner's recorded child (child_pid + child_created, Windows FILETIME)
    may still be using the GPU: kill exactly that process tree (as gpu_job.py)."""
    cpid, created = _owner_field("child_pid"), _owner_field("child_created")
    if not cpid or not created or not _pid_alive(cpid):
        return
    try:
        import psutil
        ft = int((psutil.Process(cpid).create_time() + 11644473600) * 1e7)
        if abs(ft - created) < 20000:      # same process (2 ms tolerance on the float conversion)
            import subprocess
            print(f"[gpu.lock] stale owner's child {cpid} still running: killing its process tree", flush=True)
            subprocess.call(["taskkill", "/PID", str(cpid), "/T", "/F"])
    except Exception as e:
        print(f"[gpu.lock] could not check stale child {cpid}: {e}", flush=True)


class GpuLock:
    """Hold C:\\GeoAI\\gpu.lock for one GPU job (or one pass).

    Same convention as C:\\GeoAI\\analysis\\fixes\\gpu_job.py: mkdir is the atomic
    acquire; owner.txt ('pid: N', 'purpose: ...', 'start: ...') names us; a lock
    whose owner PID is no longer running is taken over (its recorded child is
    killed first); a lock without a readable PID is never taken over. On exit
    the lock is removed only if owner.txt still names this PID.
    """

    def __init__(self, purpose: str, poll_s: float = 30.0, max_wait_s: float = 24 * 3600,
                 quiet: bool = False):
        self.purpose = purpose.replace("\n", " ")
        self.poll_s = poll_s
        self.max_wait_s = max_wait_s
        self.quiet = quiet
        self.held = False

    def __enter__(self):
        t0 = time.time()
        announced = False
        while True:
            try:
                LOCK_DIR.mkdir(parents=False, exist_ok=False)
                atomic_write_text(LOCK_DIR / "owner.txt",
                                  f"pid: {os.getpid()}\npurpose: {self.purpose}\nstart: {utc_now()}\n")
                self.held = True
                if not self.quiet:
                    print(f"[gpu.lock] acquired for {self.purpose} (pid {os.getpid()})", flush=True)
                return self
            except FileExistsError:
                pid = _owner_pid()
                if pid is not None and pid != os.getpid() and not _pid_alive(pid):
                    print(f"[gpu.lock] stale lock: owner pid {pid} is not running; taking it over", flush=True)
                    _kill_stale_child()
                    shutil.rmtree(LOCK_DIR, ignore_errors=True)
                    continue
                if not announced:
                    print(f"[gpu.lock] held by: {_owner_text().strip()[:200] or '(no owner.txt)'} - waiting",
                          flush=True)
                    announced = True
                if time.time() - t0 > self.max_wait_s:
                    raise TimeoutError(f"gpu.lock still held after {self.max_wait_s} s")
                time.sleep(self.poll_s)

    def __exit__(self, *exc):
        if self.held:
            if _owner_pid() == os.getpid():
                shutil.rmtree(LOCK_DIR, ignore_errors=True)
                if not self.quiet:
                    print("[gpu.lock] released", flush=True)
            else:
                print("[gpu.lock] lock not ours any more; left alone", flush=True)
            self.held = False
        return False


# ============================================================
# Labels: one canonical Attain class per evaluation class
# ============================================================

def canonical_attain_class(raw_class: str):
    """Map a subset's class name to the evaluation's class name, or None if the
    class is excluded (not pavement distress). Two names are the same class
    when ATTAIN_CLASS_MAP gives them the same pipeline label: WS_V1.0 says
    'Patch', WS_V2.0 says 'Patch and utility cut' - both 'Inlaid Patch (D44)'."""
    m = _ae.ATTAIN_CLASS_MAP.get(raw_class)
    if m is None:
        raise ValueError(f"unmapped Attain class {raw_class!r}")
    if m["label"] == "EXCLUDE":
        return None
    for c in ALL_CLASSES:
        if _ae.ATTAIN_CLASS_MAP[c]["label"] == m["label"]:
            return c
    raise ValueError(f"class {raw_class!r} has no evaluation counterpart")


def subset_classes(subset: str) -> dict:
    """{'class_names': raw names from the YAML, 'annotated': evaluation classes
    this subset annotates (absence of one of these is a known negative)}."""
    info = _ae.get_subset_info(subset)
    ann = set()
    for n in info["class_names"]:
        cls, _sev = _ae.parse_attain_class(n)
        c = canonical_attain_class(cls)
        if c is not None:
            ann.add(c)
    return {"class_names": info["class_names"], "annotated": sorted(ann), "info": info}


def load_train_subset(subset: str) -> list[dict]:
    """Every image of a YOLO-labelled subset with its evaluation classes."""
    sc = subset_classes(subset)
    info = sc["info"]
    rows = []
    for img in sorted(info["images_dir"].glob("*.jpg")):
        lab = info["labels_dir"] / (img.stem + ".txt")
        objs = _ae.parse_yolo_labels(lab, info["class_names"]) if lab.exists() else []
        if lab.exists():
            # parse_yolo_labels silently drops ids outside the class list; we refuse to
            for line in lab.read_text(encoding="utf-8").splitlines():
                p = line.split()
                if p and not (0 <= int(p[0]) < len(info["class_names"])):
                    raise ValueError(f"{lab.name}: class id {p[0]} outside the YAML list")
        classes = set()
        for cls, _sev in objs:
            c = canonical_attain_class(cls)
            if c is not None:
                classes.add(c)
        rows.append({"subset": subset, "image": img.name, "image_path": str(img),
                     "gt": sorted(classes), "annotated": sc["annotated"],
                     "has_label_file": lab.exists()})
    return rows


def load_v2_rows() -> list[dict]:
    """WS_V2.0 rows with gt and the pre-registered dev/test split."""
    from scripts.stage2_probe_experiment import load_all_attain   # noqa: E402 (sets prod env vars)
    rows = load_all_attain("WS_V2.0")
    dev, test = v2_split(rows)
    return rows


def arm_b_dev_split(dev_rows: list[dict]) -> tuple[list, list]:
    """Dev blocks -> (train part 2/3, calibration part 1/3), by block, seeded."""
    blocks = sorted({r["block"] for r in dev_rows})
    rng = random.Random(SPLIT_B_SEED)
    rng.shuffle(blocks)
    n_cal = round(len(blocks) / 3)
    cal = set(blocks[:n_cal])
    return ([r for r in dev_rows if r["block"] not in cal],
            [r for r in dev_rows if r["block"] in cal])


# ============================================================
# Training targets
# ============================================================

def targets_for(row: dict) -> list[dict]:
    """Supervised Yes/No targets for one training image.

    One target per evaluation class the image's subset annotates:
      {"cls", "keys" (probe questions), "y" (1/0)}; a class scored by several
      questions (Linear crack = max(Longitudinal, Transverse)) is trained on the
      max, exactly how the evaluation scores it.
    Masked (no target): Linear crack NEGATIVE on an image that has Block crack -
    IRC:82 §7.3.5.1 files block cracking under Transverse Cracking, so "no
    linear crack" there is not a clean No for the Transverse question.
    """
    gt = set(row["gt"])
    out = []
    for c in row["annotated"]:
        y = 1 if c in gt else 0
        if c == "Linear crack" and y == 0 and BLOCK in gt:
            continue
        out.append({"cls": c, "keys": list(PROBE_KEYS[c]), "y": y})
    return out


def build_manifest(rows: list[dict], name: str) -> dict:
    items = []
    for r in rows:
        t = targets_for(r)
        if not t:
            continue
        items.append({"subset": r.get("subset", "WS_V2.0"), "image": r["image"],
                      "image_path": r["image_path"], "sha256": sha256_file(Path(r["image_path"])),
                      "gt": r["gt"], "targets": t})
    counts = {}
    for it in items:
        for t in it["targets"]:
            k = counts.setdefault(t["cls"], {"pos": 0, "neg": 0})
            k["pos" if t["y"] else "neg"] += 1
    return {"name": name, "n_images": len(items), "target_counts": counts, "items": items}


def probe_keys_needed(manifest: dict) -> list[str]:
    ks = []
    for it in manifest["items"]:
        for t in it["targets"]:
            for k in t["keys"]:
                if k not in ks:
                    ks.append(k)
    return ks
