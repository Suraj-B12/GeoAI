"""Time run_vlm_batch.py end to end on one of the 30-image copies (wall clock incl. checksums, /health and pacing).

Usage: python time_runner.py runner_before -- [runner args]
"""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNER = r"C:\Users\PRO-LAB-3\Documents\GeoAI_Video\pipeline\tools\run_vlm_batch.py"

name = sys.argv[1]
args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
d = HERE / name
cmd = [sys.executable, RUNNER, str(d), "--url", "http://127.0.0.1:8000", *args]
t0 = time.perf_counter()
with open(d / "runner_stdout.log", "w", encoding="utf-8") as f:
    code = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
wall = time.perf_counter() - t0
(d / "runner_timing.json").write_text(json.dumps({"args": args, "exit_code": code, "wall_s": round(wall, 2)}, indent=2),
                                      encoding="utf-8")
print(name, args, "exit", code, f"{wall:.1f} s")
