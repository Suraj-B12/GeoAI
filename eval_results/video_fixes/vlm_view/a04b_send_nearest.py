"""VLM-view A/B, step 4b (GPU through the running server; launch it through ..\\gpu_job.py so it holds the lock).

1. If field/batches/ab_nearest_001 does not exist: export it (CPU) from field/runs_eval/nearest with exactly the
   segment ids in changed_ids.txt (written by `a04_nearest_vs_sharpest.py geometry`), image = whole frame,
   the batch export's own checks (SHA-256, records). Written to ab_nearest_001.partial, renamed when complete.
2. Send it with the batch's own run_vlm_batch.py to http://127.0.0.1:8000 with --profile video (identical
   answers to the default profile, 60/60 in vlm_speed; ~3.8 s per Normal answer). The operator API key is read
   from C:\\GeoAI\\operator_api_key.txt and passed to the runner in-process (never printed or logged).
   The runner is resumable: already answered requests are skipped, so rerunning this script is safe.
Exit code = the runner's (0 = every request answered).
"""
import os
import runpy
import shutil
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # this process only exports and sends; the server does the GPU work
REPO = Path(r"C:/Users/PRO-LAB-3/Documents/GeoAI_Video")
OUT = Path(r"C:/GeoAI/analysis/fixes/vlm_view")
DST = REPO / "field/batches/ab_nearest_001"
SID = "20261027T050257Z_SM-G991B-F0D1_d47c"

if not DST.exists():
    sys.path.insert(0, str(REPO / "pipeline"))
    from geoai.config import Config  # noqa: E402
    from geoai.vlm import export_batch  # noqa: E402
    ids = [x for x in (OUT / "changed_ids.txt").read_text(encoding="utf-8").split() if x]
    if not ids:
        print("changed_ids.txt is empty: nothing to send")
        sys.exit(0)
    part = DST.with_name(DST.name + ".partial")
    shutil.rmtree(part, ignore_errors=True)
    cfg = Config()
    cfg.segments.choice = "nearest_sharp"
    b = export_batch([REPO / "field/runs_eval/nearest" / SID], part, cfg, ids=ids)
    print("exported", b["n_requests"], f"{b['image_bytes_total'] / 1e9:.2f} GB", b.get("not_exported"), flush=True)
    part.rename(DST)

key = Path(r"C:/GeoAI/operator_api_key.txt").read_text(encoding="utf-8").strip()
sys.argv = [str(DST / "run_vlm_batch.py"), str(DST), "--url", "http://127.0.0.1:8000", "--api-key", key,
            "--profile", "video"]
try:
    runpy.run_path(str(DST / "run_vlm_batch.py"), run_name="__main__")
    code = 0
except SystemExit as e:
    code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
sys.exit(code)
