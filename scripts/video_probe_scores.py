"""
Per-type probe scores for every stored drive-video segment image -> research.video_probe_scores (migration 010).

Why: the A->B condition report (app/pci_core.py) can rate only cracking and potholes from the free-form
Stage 2 list; on the 7,050 video segments that list never names Ravelling, Patching or Settlement, so the
IRC:82 rating stays a range. The per-type Yes/No probe (scripts/stage2_probe.py, the production variant
from configs/stage2_probe.json, full photo, 1024^2 cap) gives P(yes) for every type. These are RAW scores;
they become labels only through a calibration file fitted on a survey/expert split
(scripts/pci_param_calibration.py), never on the Attain thresholds (those put Ravelling on 98% of
Bengaluru photos).

Resumable: one INSERT per image is the checkpoint; rows already present for this probe_fp are skipped.
GPU: run it under C:\\GeoAI\\analysis\\fixes\\gpu_job.py (shared lock). Loads its own 4-bit model copy
(~6 GB) next to the idle production server.

Usage:
    venv/Scripts/python.exe scripts/video_probe_scores.py [--limit N] [--database geoai]
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MAX_IMAGE_PIXELS", "1048576")

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


def load_env():
    env = PROJECT_ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


BAND_MARGIN = 0.10      # --view band: extra rows above/below the segment's strip, as a share of its height


def probe_fingerprint(cfg: dict, keys: list[str], max_pixels: int, view: str = "full") -> str:
    blob = {"model": MODEL, "quant": 4, "variant": cfg["variant"], "keys": keys, "max_pixels": max_pixels,
            "view": "full"}
    if view != "full":           # the full-view fingerprint stays what it was (rows already stored)
        blob["view"] = {"name": view, "margin": BAND_MARGIN}
    return hashlib.sha256(json.dumps(blob, sort_keys=True).encode()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--database", default=None, help="override the database name in DATABASE_URL")
    ap.add_argument("--view", choices=["full", "band"], default="full",
                    help="band: only the image rows of the segment's own 5 m strip (record.frame.rows_of_segment), "
                         "full width, +10 %% margin")
    args = ap.parse_args()
    load_env()
    if args.database:
        import re
        os.environ["DATABASE_URL"] = re.sub(r"/[^/?]+(\?|$)", f"/{args.database}\\1",
                                            os.environ["DATABASE_URL"], count=1)

    import torch
    from PIL import Image
    from app import video_store
    from app.local_images import local_image_path
    from scripts import stage2_probe_rules as rules
    from scripts.model_loader import load_vlm
    from scripts.stage2_probe import TypeProber, all_probe_types

    cfg = rules.load_config(PROJECT_ROOT / "configs" / "stage2_probe.json")
    types = all_probe_types()
    keys = [t.key for t in types]
    max_pixels = int(os.environ["MAX_IMAGE_PIXELS"])
    fp = probe_fingerprint(cfg, keys, max_pixels, args.view)

    conn = video_store.connect()
    sql = (PROJECT_ROOT / "migrations" / "010_video_probe_scores.sql").read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute(sql)
        cur.execute("""SELECT g.segment_id, g.image_path, g.image_sha256,
                              g.record->'frame'->'rows_of_segment' AS rows FROM video_segments g
                       WHERE g.image_path IS NOT NULL
                         AND NOT EXISTS (SELECT 1 FROM research.video_probe_scores s
                                         WHERE s.segment_id = g.segment_id AND s.probe_fp = %s)
                       ORDER BY g.segment_id""", (fp,))
        todo = cur.fetchall()
    if args.limit:
        todo = todo[: args.limit]
    print(f"[data] {len(todo)} segment images to score; probe_fp {fp}; {len(keys)} types", flush=True)
    if not todo:
        return 0

    model, proc, _ = load_vlm(MODEL, quantization_bits=4, min_pixels=256 * 28, max_pixels=max_pixels,
                              run_self_test=True)
    prober = TypeProber(model, proc, types, cfg["variant"]["system_style"], cfg["variant"]["question_style"])
    par = prober.parity_check()
    print(f"[parity] {par}", flush=True)
    if not par.get("ok"):
        sys.exit("probe parity failed")

    t_run = time.time()
    n_err = 0
    for i, row in enumerate(todo, 1):
        seg, path, sha, rows = row[0], row[1], row[2], row[3]
        scores, err, dt = None, None, None
        try:
            p = local_image_path(path)
            if p is None or not p.exists():
                raise FileNotFoundError(f"image missing: {path}")
            with Image.open(p) as im:
                img = im.convert("RGB")
            if args.view == "band":
                if not (isinstance(rows, list) and len(rows) == 2):
                    raise ValueError("no rows_of_segment in the segment record")
                r0, r1 = sorted(float(v) for v in rows)
                m = BAND_MARGIN * (r1 - r0)
                top, bot = max(0, int(r0 - m)), min(img.height, int(r1 + m + 0.5))
                if bot - top < 32:
                    raise ValueError(f"strip too thin ({bot - top} rows)")
                img = img.crop((0, top, img.width, bot))
            t0 = time.time()
            scores = {r["key"]: round(float(r["p_yes"]), 6) for r in prober.probe(img)}
            dt = time.time() - t0
        except Exception as e:                               # recorded, never fatal
            err = f"{type(e).__name__}: {e}"[:500]
            n_err += 1
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        with conn.cursor() as cur:
            cur.execute("""INSERT INTO research.video_probe_scores
                           (segment_id, probe_fp, image_sha256, scores, error, time_s)
                           VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING""",
                        (seg, fp, sha, json.dumps(scores) if scores is not None else None, err, dt))
        if i % 25 == 0 or i == len(todo):
            el = (time.time() - t_run) / i
            print(f"[{i}/{len(todo)}] {el:.2f} s/image, eta {el * (len(todo) - i) / 60:.0f} min, errors {n_err}",
                  flush=True)
    print(f"[done] {len(todo)} images, {n_err} errors", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
