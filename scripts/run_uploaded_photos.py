"""
Run the production pipeline over the REAL photos uploaded by the RoadSide
mobile app, and measure what the Stage 2 confidence switch actually changes.

Why this script exists
----------------------
`scripts/calibrate_confidence.py` compared the two Stage 2 confidence metrics
on Attain — New Zealand dashcam footage. Production is Bengaluru handheld
phone photos. Before trusting the switch to 'field' in production, the same
comparison has to run on the data the system actually sees.

It does NOT write to Supabase. The worker owns that column. This script only
measures, so it is safe to run against live data at any time.

Source of images, in order of preference:
  1. The live `assessments` table (every row with an image_url)
  2. eval_results/bengaluru_baseline_snapshot.json  (offline fallback — used
     when the Supabase project is paused/unreachable)

Images are fetched from their Cloudinary URLs. Rows whose image was
hard-deleted from Cloudinary by the operator dashboard return 404 and are
recorded as skipped rather than failing the run.

Per image it runs the SAME three stages the worker runs — Stage 0 pavement
pre-filter, Stage 1 binary, Stage 2 IRC type+severity — and records both
confidence metrics, so the 0.80 gate can be replayed under either one.

Crash-resumable: every image is checkpointed. Re-running picks up where it
stopped (the college machine reboots mid-run).

Usage:
    venv/Scripts/python.exe scripts/run_uploaded_photos.py
    venv/Scripts/python.exe scripts/run_uploaded_photos.py --limit 20
    venv/Scripts/python.exe scripts/run_uploaded_photos.py --fresh
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image  # noqa: E402

SNAPSHOT = PROJECT_ROOT / "eval_results" / "bengaluru_baseline_snapshot.json"


# ============================================================
# .env loading (this script runs standalone, not under uvicorn --env-file)
# ============================================================

def load_dotenv() -> None:
    env = PROJECT_ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# ============================================================
# Row sources
# ============================================================

def fetch_live_rows(timeout: float = 15.0):
    """
    Pull every assessment that has an image. Returns None (not []) when the
    project is unreachable, so the caller can distinguish 'down' from 'empty'.
    """
    url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        return None
    req = urllib.request.Request(
        f"{url.rstrip('/')}/rest/v1/assessments"
        "?select=id,photo_id,image_url,status,stage1_label,stage1_confidence,"
        "is_distressed,distress_types,severity,stage2_confidence,"
        "needs_expert_review,raw_response,created_at"
        "&image_url=not.is.null&order=created_at.asc",
        headers={"apikey": key, "Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f"[source] live Supabase unreachable ({type(e).__name__}) "
              f"— falling back to the offline snapshot")
        return None


def load_snapshot_rows() -> list:
    if not SNAPSHOT.exists():
        sys.exit(f"ERROR: no live DB and no snapshot at {SNAPSHOT}")
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))["rows"]


def fetch_image(url: str, timeout: float = 30.0):
    """Download from Cloudinary. None means gone (404) or unreachable."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        print(f"      image unavailable (HTTP {e.code})")
        return None
    except Exception as e:
        print(f"      image fetch failed ({type(e).__name__})")
        return None


# ============================================================
# Reporting
# ============================================================

def gate(stage1_conf: float, stage2_conf: float, is_distressed: bool,
         threshold: float) -> str:
    """Replay the worker's routing decision (app/worker.py _do_inference)."""
    if not is_distressed:
        return "classified" if stage1_conf >= threshold else "expert_review"
    if stage1_conf >= threshold and stage2_conf >= threshold:
        return "classified"
    return "expert_review"


def summarize(rows: list, threshold: float) -> dict:
    done = [r for r in rows if r.get("ok")]
    s2 = [r for r in done if r.get("ran_stage2")]

    def stats(key: str) -> dict:
        xs = [r[key] for r in s2]
        if not xs:
            return {"n": 0}
        xs_sorted = sorted(xs)
        return {
            "n": len(xs),
            "mean": round(sum(xs) / len(xs), 4),
            "min": round(xs_sorted[0], 4),
            "median": round(xs_sorted[len(xs) // 2], 4),
            "max": round(xs_sorted[-1], 4),
            "at_or_above_threshold": sum(1 for x in xs if x >= threshold),
        }

    routing = {}
    for mode in ("field", "sequence"):
        c = Counter(
            gate(r.get("stage1_confidence", 0.0),
                 r[mode] if r.get("ran_stage2") else 0.0,
                 r.get("is_distressed", False), threshold)
            for r in done
        )
        routing[mode] = {"classified": c["classified"],
                         "expert_review": c["expert_review"]}

    return {
        "n_rows_attempted": len(rows),
        "n_processed": len(done),
        "n_image_unavailable": sum(1 for r in rows if r.get("skipped") == "image_unavailable"),
        "n_errors": sum(1 for r in rows if r.get("error")),
        "stage0_rejected_non_pavement": sum(1 for r in done if not r.get("is_pavement")),
        "stage1_distress": sum(1 for r in done if r.get("is_distressed")),
        "stage1_normal": sum(1 for r in done
                             if r.get("is_pavement") and not r.get("is_distressed")),
        "n_with_stage2": len(s2),
        "field_span_found": sum(1 for r in s2 if r.get("field_span_found")),
        "confidence": {"field": stats("field"), "sequence": stats("sequence")},
        "routing_at_threshold": routing,
        "threshold": threshold,
    }


def print_report(report: dict) -> None:
    s = report["summary"]
    c = s["confidence"]
    r = s["routing_at_threshold"]
    th = s["threshold"]
    line = "=" * 70
    print("\n" + line)
    print(f"UPLOADED PHOTOS - STAGE 2 CONFIDENCE - {report['model']}")
    print(f"source: {report['source']}   threshold: {th}")
    print(line)
    print(f"rows attempted            {s['n_rows_attempted']}")
    print(f"  processed               {s['n_processed']}")
    print(f"  image gone (404)        {s['n_image_unavailable']}")
    print(f"  errors                  {s['n_errors']}")
    print(f"Stage 0 rejected          {s['stage0_rejected_non_pavement']}")
    print(f"Stage 1 Distress/Normal   {s['stage1_distress']} / {s['stage1_normal']}")
    print(f"Stage 2 ran on            {s['n_with_stage2']}  "
          f"(DISTRESS_TYPES span located on {s['field_span_found']})")
    print(line)
    print(f"{'':<28}{'field':>14}{'sequence':>14}")
    for label, k in [("mean", "mean"), ("median", "median"),
                     ("min", "min"), ("max", "max"),
                     (f"n >= {th}", "at_or_above_threshold")]:
        print(f"{label:<28}{str(c['field'].get(k)):>14}{str(c['sequence'].get(k)):>14}")
    print(line)
    print("Routing of ALL processed images at the 0.80 gate:")
    print(f"{'':<28}{'field':>14}{'sequence':>14}")
    print(f"{'auto-classified':<28}{r['field']['classified']:>14}"
          f"{r['sequence']['classified']:>14}")
    print(f"{'sent to expert review':<28}{r['field']['expert_review']:>14}"
          f"{r['sequence']['expert_review']:>14}")
    print(line)


# ============================================================
# Main
# ============================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows")
    ap.add_argument("--model", default=os.environ.get("MODEL_PATH",
                                                      "Qwen/Qwen2.5-VL-7B-Instruct"))
    ap.add_argument("--quantization-bits", type=int, default=0, choices=[0, 4, 8])
    ap.add_argument("--out", default="uploaded_photos_confidence.json")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any existing checkpoint and start over")
    args = ap.parse_args()

    load_dotenv()
    os.environ.setdefault("PROMPTS_VERSION", "v2")
    os.environ["QUANTIZATION_BITS"] = str(args.quantization_bits)
    os.environ["DISABLE_ADAPTER"] = "true"

    from app.model import PavementClassifier
    from scripts.utils import CONFIDENCE_THRESHOLD, EVAL_DIR

    live = fetch_live_rows()
    if live is None:
        source = "offline snapshot (bengaluru_baseline_snapshot.json)"
        src_rows = load_snapshot_rows()
    else:
        source = "live Supabase assessments table"
        src_rows = live
    if args.limit:
        src_rows = src_rows[: args.limit]
    print(f"[source] {source} - {len(src_rows)} rows with an image")

    ckpt = EVAL_DIR / (Path(args.out).stem + "_checkpoint.json")
    results: dict = {}
    if ckpt.exists() and not args.fresh:
        results = json.loads(ckpt.read_text(encoding="utf-8"))
        # A checkpoint from a different model would silently mix two models
        # into one result set.
        if results.get("_model") not in (None, args.model):
            sys.exit(f"ERROR: checkpoint {ckpt.name} was written by "
                     f"{results['_model']!r}, not {args.model!r}. "
                     f"Use --fresh or a different --out.")
        results.pop("_model", None)
        print(f"[resume] {len(results)} images already done in {ckpt.name}")

    clf = PavementClassifier(model_path=args.model, adapter_path=None,
                             quantization_bits=args.quantization_bits)

    def save_ckpt():
        ckpt.write_text(json.dumps({**results, "_model": args.model}, indent=2),
                        encoding="utf-8")

    t_start = time.time()
    for i, row in enumerate(src_rows, 1):
        rid = str(row["id"])
        if rid in results:
            continue
        print(f"[{i}/{len(src_rows)}] {rid[:8]}")
        rec = {"id": rid, "photo_id": row.get("photo_id"),
               "image_url": row.get("image_url"),
               "prev_status": row.get("status"),
               "prev_stage2_confidence": row.get("stage2_confidence")}

        raw = fetch_image(row["image_url"])
        if raw is None:
            rec["skipped"] = "image_unavailable"
            results[rid] = rec
            save_ckpt()
            continue

        try:
            img = Image.open(io.BytesIO(raw))
            img.load()
            img = img.convert("RGB")

            s0 = clf.predict_is_pavement(img)
            rec["is_pavement"] = s0["is_pavement"]
            rec["pavement_decision"] = s0["decision"]
            if not s0["is_pavement"]:
                # Worker rejects here without running Stage 1/2.
                rec.update(ok=True, ran_stage2=False, is_distressed=False,
                           stage1_confidence=0.0, field=0.0, sequence=0.0)
                print("      Stage 0: rejected_non_pavement")
            else:
                s1 = clf.predict_stage1(img)
                rec["stage1_label"] = s1["stage1_label"]
                rec["stage1_confidence"] = s1["stage1_confidence"]
                rec["is_distressed"] = s1["is_distressed"]
                if s1["is_distressed"]:
                    s2 = clf.predict_stage2(img)
                    rec.update(
                        ok=True, ran_stage2=True,
                        distress_types=s2["distress_types"],
                        severity=s2["severity"],
                        field=s2["stage2_confidence_field"],
                        sequence=s2["stage2_confidence_sequence"],
                        field_span_found=s2["stage2_field_span_found"],
                        active_mode=s2["stage2_confidence_mode"],
                        active_confidence=s2["stage2_confidence"],
                    )
                    print(f"      Distress {s2['distress_types']} / {s2['severity']}  "
                          f"field={s2['stage2_confidence_field']:.3f} "
                          f"seq={s2['stage2_confidence_sequence']:.3f}")
                else:
                    rec.update(ok=True, ran_stage2=False, field=0.0, sequence=0.0)
                    print(f"      Normal (s1={s1['stage1_confidence']:.3f})")
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            print(f"      ERROR {rec['error']}")

        results[rid] = rec
        save_ckpt()

    rows_out = list(results.values())
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "quantization_bits": args.quantization_bits,
        "prompts_version": os.environ.get("PROMPTS_VERSION"),
        "source": source,
        "seconds_total": round(time.time() - t_start, 1),
        "summary": summarize(rows_out, CONFIDENCE_THRESHOLD),
        "per_image": rows_out,
    }

    out = EVAL_DIR / args.out
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_report(report)
    print(f"Written: {out}")
    print(f"Checkpoint kept at: {ckpt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
