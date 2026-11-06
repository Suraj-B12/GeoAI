"""
Run a queue of dev evaluations (one pass each) against the forward-pass server.

The queue is a JSON list of {name, edits, purpose, dose_group?, c?, control?}.
Runs already recorded in eval_results/finetune/wsp_dev_runs.json (by name) are
skipped, so re-running the same command resumes. Waits for the server to come up.

    venv/Scripts/python.exe scripts/wsp_queue.py C:/GeoAI/wsp/queue_<n>.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import wsp_analysis as A  # noqa: E402
from scripts import wsp_client as C  # noqa: E402
from scripts.ft_common import FT_DIR, read_json  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("queue")
    a = ap.parse_args()
    q = json.loads(Path(a.queue).read_text(encoding="utf-8"))
    while True:
        try:
            C.status()
            break
        except Exception:
            print("[queue] waiting for the server", flush=True)
            time.sleep(30)
    for job in q:
        done = {r["name"] for r in read_json(FT_DIR / "wsp_dev_runs.json", {"runs": []})["runs"]}
        if job["name"] in done:
            print(f"[queue] {job['name']} already done", flush=True)
            continue
        ns = argparse.Namespace(name=job["name"], edits=",".join(job.get("edits", [])),
                                purpose=job["purpose"], dose_group=job.get("dose_group"),
                                c=job.get("c"), control=bool(job.get("control")))
        A.cmd_devrun(ns)
    print("[queue] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
