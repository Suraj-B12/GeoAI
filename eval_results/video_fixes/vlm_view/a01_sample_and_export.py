"""VLM-view A/B, step 1 (CPU only): the stratified sample and the two batches (frame arm, band arm).

Sample: every VLM-positive segment of batch_001 (vlm_class distress) and as many VLM-Normal ones, drawn
with a fixed seed. Both arms send exactly these segments, from the send-all run folders (the frames
batch_001 sent: SHA-256 checked against batch_001's records). The frame arm is a re-run of the
whole frames so both arms run on the same server, profile and day.
Writes sample_ids.txt, sample.json here and field/batches/ab_frame_001, ab_band_001.
"""
import os
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "run with CUDA_VISIBLE_DEVICES=-1"
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
assert not torch.cuda.is_available()

REPO = Path(r"C:/Users/PRO-LAB-3/Documents/GeoAI_Video")
sys.path.insert(0, str(REPO / "pipeline"))
from geoai.config import Config  # noqa: E402
from geoai.vlm import export_batch  # noqa: E402

OUT = Path(r"C:/GeoAI/analysis/fixes/vlm_view")
B1 = REPO / "field/batches/batch_001"
RUNS = sorted(p for p in (REPO / "field/runs_segments_sendall").iterdir() if p.name != "logs")
SEED = 20261028

rows = list(csv.DictReader(open(B1 / "vlm_results.csv", encoding="utf-8-sig")))
pos = [r["request_id"] for r in rows if r["vlm_class"] == "distress"]
neg = [r["request_id"] for r in rows if r["vlm_class"] == "normal"]
rng = np.random.default_rng(SEED)
neg_s = sorted(rng.choice(neg, size=len(pos), replace=False).tolist())
ids = sorted(pos + neg_s)
# The run frames are the ones batch_001 sent.
for rid in ids:
    b = json.loads((B1 / "records" / f"{rid}.json").read_text(encoding="utf-8"))
    run = next(r for r in RUNS if rid.startswith(r.name))
    s = json.loads((run / "sidecars" / f"{rid}.json").read_text(encoding="utf-8"))
    assert b["frame"]["sha256"] == s["frame"]["sha256"], rid
(OUT / "sample_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
json.dump({"seed": SEED, "n_positive": len(pos), "n_normal_pool": len(neg), "n_normal_drawn": len(neg_s),
           "n_unknown_excluded": sum(1 for r in rows if r["vlm_class"] not in ("distress", "normal")),
           "positives": sorted(pos), "normals": neg_s}, open(OUT / "sample.json", "w"), indent=1)
print(f"sample: {len(pos)} positive + {len(neg_s)} normal of {len(neg)}")

for image, name in (("frame", "ab_frame_001"), ("band", "ab_band_001")):
    dst = REPO / "field/batches" / name
    if dst.exists():
        print(f"{dst} exists, kept")
        continue
    cfg = Config()
    cfg.vlm.image = image
    b = export_batch(RUNS, dst, cfg, ids=ids)
    print(name, b["n_requests"], f"{b['image_bytes_total'] / 1e9:.2f} GB", b["not_exported"], b["reencoded"])
