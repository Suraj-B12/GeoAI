"""
Sandbox: does the probe do better on tiles or zoomed crops than on the whole
photo? Records every view's per-type P(yes); scoring lives in
stage2_views_report.py so aggregation rules can be changed without the GPU.

Per image:
  full        the whole photo (the current probe)
  tile_native ~4 overlapping square-ish tiles, no upscaling
  tile_up     the same tiles, each upscaled up to 2x within the pixel budget
  zoom        crops around the damage Qwen2.5-VL boxes itself (up to 4,
              margin 20%, boxes over half the frame dropped), upscaled up to 2x
  oracle      (Attain only) the same crop procedure on the ANNOTATED boxes:
              the most any locator could deliver. A diagnostic ceiling, never
              a deployable method.

Datasets:
  attain     Attain WS_V2.0 frames; --split dev | test | all (the same
             20-frame-block split as stage2_probe_report.py)
  bengaluru  production uploads since --since, downloaded read-only from
             Cloudinary; nothing is written to the database

Usage:
    venv/Scripts/python.exe scripts/stage2_views_experiment.py --dataset attain --split dev
    venv/Scripts/python.exe scripts/stage2_views_experiment.py --dataset bengaluru
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MAX_IMAGE_PIXELS", "1048576")

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from scripts import stage2_probe_report as R  # noqa: E402
from scripts import stage2_views as V  # noqa: E402
from scripts.model_loader import load_vlm  # noqa: E402
from scripts.stage2_probe import TypeProber, all_probe_types  # noqa: E402
from scripts import stage2_probe_rules as rules  # noqa: E402

EVAL_DIR = PROJECT_ROOT / "eval_results"
EXCLUDED = ("Faded marking", "Lane shoulder drop-off", "Manhole")


def release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def gt_boxes(image_path: str) -> list:
    lab = Path(image_path.replace("Images", "Labels")).with_suffix(".xml")
    out = []
    for o in ET.parse(lab).findall(".//object"):
        if (o.findtext("name") or "").startswith(EXCLUDED):
            continue
        b = o.find("bndbox")
        out.append([float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")])
    return out


def load_attain(split: str) -> list:
    d = json.loads((EVAL_DIR / "stage2_probe_raw_attain.json").read_text(encoding="utf-8"))
    rows = [r for r in d["rows"] if "error" not in r]
    dev, test = R.split(rows)
    pick = {"dev": dev, "test": test, "all": rows}[split]
    return [{"key": r["image"], "image_path": r["image_path"], "gt": r["gt"]} for r in pick]


def load_bengaluru(since: str) -> list:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import probe_production_compare as P
    rows = P.fetch_rows(since)
    return [{"key": r["id"], "url": r["image_url"], "kind": r["_kind"],
             "v0_types": r.get("distress_types") or []} for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["attain", "bengaluru"], default="attain")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    ap.add_argument("--since", default="2026-09-23T08:27:00Z")
    ap.add_argument("--families", default="tile_native,tile_up,zoom,oracle")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    fams = [f for f in args.families.split(",") if f]
    if args.dataset == "bengaluru":
        fams = [f for f in fams if f != "oracle"]
    out_name = args.out or (f"stage2_views_{args.dataset}"
                            + (f"_{args.split}" if args.dataset == "attain" else "") + ".json")

    cfg = rules.load_config(PROJECT_ROOT / "configs" / "stage2_probe.json")
    types = all_probe_types()
    setup = {"variant": cfg["variant"], "families": fams, "grid": {"target": 4, "overlap": 0.1},
             "upscale_limit": 2.0, "zoom": {"max_boxes": 4, "max_area_frac": 0.5, "margin": 0.2,
                                            "min_side_frac": 0.25, "nms_iou": 0.3},
             "ground_prompt_sha": hashlib.sha256(V.GROUND_PROMPT.encode()).hexdigest()[:12],
             "max_pixels": int(os.environ["MAX_IMAGE_PIXELS"])}
    fp = hashlib.sha256(json.dumps(setup, sort_keys=True).encode()).hexdigest()[:16]

    items = load_attain(args.split) if args.dataset == "attain" else load_bengaluru(args.since)
    if args.limit:
        items = items[: args.limit]
    print(f"[data] {len(items)} {args.dataset} images; families {fams}; config {fp}")

    ckpt = EVAL_DIR / (Path(out_name).stem + "_checkpoint.json")
    done = {}
    if ckpt.exists():
        blob = json.loads(ckpt.read_text(encoding="utf-8"))
        if blob.get("_fp") == fp:
            done = blob["rows"]
            print(f"[resume] {len(done)} done")
        else:
            sys.exit(f"{ckpt.name} was made under another configuration; delete it or use --out")

    model, proc, _info = load_vlm("Qwen/Qwen2.5-VL-7B-Instruct", quantization_bits=4,
                                  min_pixels=256 * 28, max_pixels=setup["max_pixels"],
                                  run_self_test=True)
    prober = TypeProber(model, proc, types, cfg["variant"]["system_style"],
                        cfg["variant"]["question_style"])
    par = prober.parity_check()
    print(f"[parity] {par}")
    if not par["ok"]:
        sys.exit("probe parity failed")
    locator = V.DamageLocator(model, proc)

    def probe(img):
        res = prober.probe(img)
        release()
        return {r["key"]: round(r["p_yes"], 6) for r in res}

    client = None
    if args.dataset == "bengaluru":
        import httpx
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        import probe_production_compare as P
        client = httpx.Client(timeout=60, follow_redirects=True)

    t_run = time.time()
    for i, it in enumerate(items, 1):
        if it["key"] in done:
            continue
        rec = {k: v for k, v in it.items() if k not in ("url",)}
        try:
            if client is not None:
                img = Image.open(io.BytesIO(P.get_with_retry(client, it["url"]).content))
                img.load()
                img = img.convert("RGB")
            else:
                img = Image.open(it["image_path"]).convert("RGB")
            w, h = img.size
            rec["size"] = [w, h]
            t = {}
            t0 = time.time()
            rec["full"] = probe(img)
            t["full"] = time.time() - t0
            tiles = V.grid_boxes(w, h, **setup["grid"])
            rec["tile_boxes"] = [[round(v, 1) for v in b] for b in tiles]
            for fam, lim in (("tile_native", 1.0), ("tile_up", setup["upscale_limit"])):
                if fam in fams:
                    t0 = time.time()
                    rec[fam] = [probe(V.crop_view(img, b, lim, setup["max_pixels"])) for b in tiles]
                    t[fam] = time.time() - t0
            if "zoom" in fams:
                t0 = time.time()
                loc = locator.locate(img)
                release()
                sel = V.select_zoom_boxes(loc["boxes"], w, h, **setup["zoom"])
                rec["zoom_raw_boxes"] = [[round(v, 1) for v in b] for b in loc["boxes"]]
                rec["zoom_boxes"] = [[round(v, 1) for v in b] for b in sel]
                rec["zoom_raw"] = loc["raw"][:1500]
                rec["zoom"] = [probe(V.crop_view(img, b, setup["upscale_limit"], setup["max_pixels"]))
                               for b in sel]
                t["zoom"] = time.time() - t0
            if "oracle" in fams:
                t0 = time.time()
                sel = V.select_zoom_boxes(gt_boxes(it["image_path"]), w, h, **setup["zoom"])
                rec["oracle_boxes"] = [[round(v, 1) for v in b] for b in sel]
                rec["oracle"] = [probe(V.crop_view(img, b, setup["upscale_limit"], setup["max_pixels"]))
                                 for b in sel]
                t["oracle"] = time.time() - t0
            rec["time_s"] = {k: round(v, 2) for k, v in t.items()}
            img.close()
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
        done[it["key"]] = rec
        ckpt.write_text(json.dumps({"_fp": fp, "_setup": setup, "rows": done}), encoding="utf-8")
        release()
        el = (time.time() - t_run) / max(1, i)
        if "error" in rec:
            print(f"[{i}/{len(items)}] ERROR {rec['error']}", flush=True)
        else:
            print(f"[{i}/{len(items)}] {str(it['key'])[-12:]} {rec['size']} tiles={len(rec['tile_boxes'])} "
                  f"zoom={len(rec.get('zoom_boxes', []))} oracle={len(rec.get('oracle_boxes', []))} "
                  f"t={rec['time_s']}", flush=True)
    if client is not None:
        client.close()
    out = {"setup": setup, "fingerprint": fp, "dataset": args.dataset, "split": args.split,
           "parity": par, "rows": [done[it["key"]] for it in items if it["key"] in done]}
    (EVAL_DIR / out_name).write_text(json.dumps(out), encoding="utf-8")
    ckpt.unlink(missing_ok=True)
    print(f"[done] {len(out['rows'])} rows -> {out_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
