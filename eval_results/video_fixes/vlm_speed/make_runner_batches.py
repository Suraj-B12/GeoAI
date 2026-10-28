"""Two identical 30-image batches (export layout) copied from batch_001, for timing run_vlm_batch.py end to end:
runner_before/ (default request, --min-interval 6.5 as the toggle ran it) and runner_after/ (--profile video,
loopback default interval 0). 24 Stage-1-Normal + 6 Distress frames (the batch's 80/20 mix), fixed seed, not in
the 60-frame A/B sample. batch_001 is only read; files are copied byte-for-byte and re-hashed."""
from __future__ import annotations

import hashlib
import json
import random
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
BATCH = Path(r"C:\Users\PRO-LAB-3\Documents\GeoAI_Video\field\batches\batch_001")
SEED = 7


def main() -> None:
    ab = {f["request_id"] for f in json.loads((HERE / "ab_sample.json").read_text(encoding="utf-8"))}
    reqs = [json.loads(x) for x in (BATCH / "requests.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    s1 = {json.loads(x)["request_id"]: json.loads(x)["response"]["stage1_label"]
          for x in (BATCH / "responses.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()}
    rng = random.Random(SEED)
    pool = {lab: sorted(r["request_id"] for r in reqs if s1.get(r["request_id"]) == lab and r["request_id"] not in ab)
            for lab in ("Normal", "Distress")}
    chosen = set(rng.sample(pool["Normal"], 24)) | set(rng.sample(pool["Distress"], 6))
    sub = [r for r in reqs if r["request_id"] in chosen]
    batch_json = json.loads((BATCH / "batch.json").read_text(encoding="utf-8"))
    for name in ("runner_before", "runner_after"):
        d = HERE / name
        if d.exists():
            shutil.rmtree(d)
        (d / "images").mkdir(parents=True)
        sums = []
        for r in sub:
            dst = d / r["image"]
            shutil.copyfile(BATCH / r["image"], dst)
            h = hashlib.sha256(dst.read_bytes()).hexdigest()
            assert h == r["image_sha256"], r["request_id"]
            sums.append(f"{h}  {r['image']}")
        bj = dict(batch_json, batch_id=f"vlm_speed_{name}", n_requests=len(sub), note="30-frame copy of batch_001 for timing")
        (d / "batch.json").write_text(json.dumps(bj, indent=2), encoding="utf-8")
        (d / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sub), encoding="utf-8")
        sums.insert(0, f"{hashlib.sha256((d / 'batch.json').read_bytes()).hexdigest()}  batch.json")
        sums.insert(1, f"{hashlib.sha256((d / 'requests.jsonl').read_bytes()).hexdigest()}  requests.jsonl")
        (d / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(len(sub), "requests;", sum(s1[r["request_id"]] == "Normal" for r in sub), "Normal in batch_001")


if __name__ == "__main__":
    main()
