"""VLM-view A/B, step 2b (GPU; run it through ..\\gpu_job.py, which holds C:\\GeoAI\\gpu.lock).

Re-extract session d47c with the current GeoAI_Video HEAD, twice, --mode segments --send-all:
  control  default config (segments.choice = sharpest)        -> field/runs_eval/sharpest_head
  nearest  nearest_config.json (segments.choice = nearest_sharp) -> field/runs_eval/nearest
The control checks that HEAD still picks the frames batch_001 answered (field/runs_segments_sendall, 27 Oct),
so any frame change in the nearest arm comes from the choice rule alone (a04 compares them).
Each arm is written to <out>.partial and renamed only after the CLI exits 0, so a killed run never leaves a
folder that looks complete. An arm whose final folder already exists is skipped (resumable).
Usage: python a02b_reextract.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(r"C:/Users/PRO-LAB-3/Documents/GeoAI_Video")
PY = REPO / "pipeline/.venv/Scripts/python.exe"
SID = "20261027T050257Z_SM-G991B-F0D1_d47c"
EV = REPO / "field/runs_eval"
ARMS = [("sharpest_head", []),
        ("nearest", ["--config", r"C:/GeoAI/analysis/fixes/vlm_view/nearest_config.json"])]

for name, extra in ARMS:
    final, part = EV / name, EV / (name + ".partial")
    if (final / SID / "report.json").exists():
        print(f"{name}: already done, skipped", flush=True)
        continue
    shutil.rmtree(part, ignore_errors=True)
    cmd = [str(PY), "-m", "geoai.cli", *extra, "process", str(REPO / "field/sessions" / SID),
           "--mode", "segments", "--send-all", "--out", str(part)]
    print(f"== {name}: {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd, cwd=REPO / "pipeline")
    if rc != 0 or not (part / SID / "report.json").exists():
        print(f"{name}: FAILED exit {rc}", flush=True)
        sys.exit(1)
    shutil.rmtree(final, ignore_errors=True)
    part.rename(final)
    print(f"{name}: done", flush=True)
print("all arms done", flush=True)
