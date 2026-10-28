"""Time the runner's full SHA256SUMS check on batch_001 (read-only), as each toggle chunk did before --skip-batch-check."""
import importlib.util, json, sys, time
from pathlib import Path
B = Path(r"C:\Users\PRO-LAB-3\Documents\GeoAI_Video\field\batches\batch_001")
spec = importlib.util.spec_from_file_location("r", r"C:\Users\PRO-LAB-3\Documents\GeoAI_Video\pipeline\tools\run_vlm_batch.py")
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
entries = [l.split("  ", 1) for l in (B / "SHA256SUMS").read_text(encoding="utf-8").splitlines() if l.strip()]
size = sum((B / rel).stat().st_size for _, rel in entries)
out = []
for k in range(2):
    t = time.perf_counter(); bad = r.check_sums(B); dt = time.perf_counter() - t
    out.append({"pass": k + 1, "seconds": round(dt, 2), "bad": len(bad)})
    print(out[-1], flush=True)
json.dump({"files": len(entries), "bytes": size, "passes": out, "gb_per_s": [round(size / 1e9 / p["seconds"], 3) for p in out]},
          open(Path(__file__).with_name("hash_time.json"), "w"), indent=2)
