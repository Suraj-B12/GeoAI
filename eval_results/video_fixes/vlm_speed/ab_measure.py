"""A/B on the real VLM server: the same batch_001 PNG sent with the default request and with profile=video.

Sample: 60 frames, fixed seed, stratified by the Stage 1 answer recorded in batch_001/responses.jsonl
(40 Normal, 20 Distress) and spread over the 3 sessions. Each frame is POSTed twice, alternating which
profile goes first (even index: default first; odd: video first) so warm-up and drift cancel.
Resumable: one JSON line per (frame, profile) in ab_raw.jsonl; finished pairs are skipped.

batch_001 is only read. Usage: python ab_measure.py [--url http://127.0.0.1:8000]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BATCH = Path(r"C:\Users\PRO-LAB-3\Documents\GeoAI_Video\field\batches\batch_001")
RUNNER = Path(r"C:\Users\PRO-LAB-3\Documents\GeoAI_Video\pipeline\tools\run_vlm_batch.py")
SEED = 20261028
# (session suffix, Stage 1 answer) -> frames drawn; proportional to the batch (426/189/34 Normal,
# 102/41/21 Distress) with at least 4 per cell so the smallest session is represented.
ALLOC = {("d47c", "Normal"): 24, ("f428", "Normal"): 12, ("293d", "Normal"): 4,
         ("d47c", "Distress"): 10, ("f428", "Distress"): 6, ("293d", "Distress"): 4}


def load_runner():
    spec = importlib.util.spec_from_file_location("runner", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sample() -> list[dict]:
    reqs = {json.loads(x)["request_id"]: json.loads(x) for x in (BATCH / "requests.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()}
    resp = [json.loads(x) for x in (BATCH / "responses.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    cells: dict = {k: [] for k in ALLOC}
    for ln in sorted(resp, key=lambda r: r["request_id"]):
        if ln.get("error"):
            continue
        key = (ln["request_id"].split("_seg")[0][-4:], ln["response"]["stage1_label"])
        if key in cells:
            cells[key].append(ln)
    rng = random.Random(SEED)
    out = []
    for key, n in ALLOC.items():
        for ln in rng.sample(cells[key], n):
            out.append({"request_id": ln["request_id"], "session": key[0], "batch001_stage1": key[1],
                        "image": reqs[ln["request_id"]]["image"], "image_sha256": reqs[ln["request_id"]]["image_sha256"],
                        "batch001_seconds": ln["seconds"], "batch001_response": ln["response"]})
    rng.shuffle(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    a = ap.parse_args()
    runner = load_runner()
    frames = sample()
    (HERE / "ab_sample.json").write_text(json.dumps([{k: v for k, v in f.items() if k != "batch001_response"} for f in frames], indent=1), encoding="utf-8")
    raw = HERE / "ab_raw.jsonl"
    done = set()
    if raw.exists():
        for x in raw.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(x)
            except ValueError:
                continue
            if r.get("http_status") == 200:
                done.add((r["request_id"], r["profile"]))
    with open(raw, "a", encoding="utf-8") as out:
        for i, f in enumerate(frames):
            order = ["default", "video"] if i % 2 == 0 else ["video", "default"]
            for prof in order:
                if (f["request_id"], prof) in done:
                    continue
                fields = {"profile": "video"} if prof == "video" else None
                t0 = time.perf_counter()
                status, resp, text, _, body = runner.post_file(a.url + "/classify", BATCH / f["image"], {}, 600.0, fields=fields)
                dt = time.perf_counter() - t0
                assert body.complete and body.hexdigest() == f["image_sha256"], "image differs from requests.jsonl"
                line = {"i": i, "request_id": f["request_id"], "session": f["session"], "batch001_stage1": f["batch001_stage1"],
                        "profile": prof, "position": order.index(prof), "http_status": status, "seconds": round(dt, 3),
                        "response": resp if resp is not None else {"text": text}}
                out.write(json.dumps(line) + "\n")
                out.flush()
                print(f"[{i + 1}/60] {f['request_id'][-15:]} {prof:7s} {status} {dt:6.2f} s "
                      f"{(resp or {}).get('stage1_label')} {(resp or {}).get('distress_types')}", flush=True)
                if status != 200:
                    print("  error:", text[:300], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
