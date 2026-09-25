"""
What would the per-type probe change on REAL production photos? (read-only)

Bengaluru uploads have no expert labels (one correction exists), so this
cannot measure accuracy. It measures what can be measured without labels:

  * how often the probe-verified answer differs from the stored production
    answer, and in which direction (types added / removed);
  * the resulting label distribution;
  * the review load under the production gate (field confidence >= 0.80)
    and under the probe gate at the same threshold;
  * probe time and Yes/No health on handheld phone photos.

Rows are taken from `assessments` only if they were classified under the
CURRENT production configuration (4-bit, 1024x1024 cap, v2 prompts, field
gate) - i.e. processed after --since - and ran Stage 2. The stored
distress_types of such a row is exactly the free-form list the probe rule
starts from, so production's generation does not need re-running.

Nothing is written to the database. Images are downloaded from their public
Cloudinary URLs with retries (the college network drops TLS handshakes).

Usage:
    venv/Scripts/python.exe scripts/probe_production_compare.py \
        --config eval_results/stage2_probe_config_candidate.json
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MAX_IMAGE_PIXELS", "1048576")

import httpx  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from scripts import stage2_probe_rules as rules  # noqa: E402
from scripts.model_loader import load_vlm  # noqa: E402
from scripts.stage2_probe import TypeProber, all_probe_types  # noqa: E402
from scripts.utils import CONFIDENCE_THRESHOLD  # noqa: E402

EVAL_DIR = PROJECT_ROOT / "eval_results"


def env_from_dotenv():
    env = {}
    p = PROJECT_ROOT / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def get_with_retry(client, url, tries=6, **kw):
    last = None
    for k in range(tries):
        try:
            r = client.get(url, **kw)
            r.raise_for_status()
            return r
        except Exception as e:
            last = e
            time.sleep(2 * (k + 1))
    raise RuntimeError(f"GET failed after {tries} tries: {last}")


def fetch_rows(since: str) -> list:
    env = env_from_dotenv()
    url, key = env.get("SUPABASE_URL"), env.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        sys.exit("SUPABASE_URL / SUPABASE_SERVICE_KEY missing from .env")
    q = (f"{url}/rest/v1/assessments?select=id,image_url,status,distress_types,"
         f"stage1_label,stage1_confidence,stage2_confidence,processed_at,raw_response"
         f"&status=in.(classified,expert_review)&processed_at=gte.{since}"
         f"&order=processed_at.asc&limit=2000")
    with httpx.Client(timeout=60) as c:
        r = get_with_retry(c, q, headers={"apikey": key, "Authorization": f"Bearer {key}"})
    rows = r.json()
    keep = []
    for row in rows:
        rr = row.get("raw_response") or {}
        if row.get("stage1_label") == "Normal":
            row["_kind"] = "stage1_normal"      # safety-net candidates
            keep.append(row)
        elif rr.get("stage2_confidence_field") is not None and rr.get("stage2_raw"):
            row["_kind"] = "stage2"
            keep.append(row)
        # else: pre-field-metric row or no Stage 2 output - not comparable
    return keep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="eval_results/stage2_probe_config_final.json")
    ap.add_argument("--since", default="2026-09-23T08:27:00Z",
                    help="production server start under the current configuration")
    ap.add_argument("--out", default="probe_production_compare.json")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    cfg = rules.load_config(cfg_path)
    types = all_probe_types()
    rules.validate_config(cfg, {t.key for t in types})
    labels = {t.key: t.output_label for t in types}

    rows = fetch_rows(args.since)
    if args.limit:
        rows = rows[: args.limit]
    print(f"[data] {len(rows)} production rows that ran Stage 2 since {args.since}")

    ckpt = EVAL_DIR / (Path(args.out).stem + "_checkpoint.json")
    done = {}
    if ckpt.exists():
        try:
            blob = json.loads(ckpt.read_text(encoding="utf-8"))
            if blob.get("config") == cfg:
                done = blob.get("rows", {})
                print(f"[resume] {len(done)} done")
        except Exception:
            pass

    model, proc, info = load_vlm("Qwen/Qwen2.5-VL-7B-Instruct", quantization_bits=4,
                                 min_pixels=256 * 28,
                                 max_pixels=int(os.environ["MAX_IMAGE_PIXELS"]),
                                 run_self_test=True)
    pr = TypeProber(model, proc, types, cfg["variant"]["system_style"],
                    cfg["variant"]["question_style"])
    parity = pr.parity_check()
    print(f"[parity] {parity}")
    if not parity["ok"]:
        sys.exit("fast/slow probe paths disagree - not running")

    client = httpx.Client(timeout=60, follow_redirects=True)
    for i, row in enumerate(rows, 1):
        if row["id"] in done:
            continue
        rec = {"id": row["id"], "processed_at": row["processed_at"], "status": row["status"],
               "kind": row["_kind"],
               "v0_types": row.get("distress_types") or [],
               "stage1_confidence": row.get("stage1_confidence"),
               "field_confidence": (row.get("raw_response") or {}).get("stage2_confidence_field")}
        try:
            img = Image.open(io.BytesIO(get_with_retry(client, row["image_url"]).content))
            img.load()
            img = img.convert("RGB")
            rec["size"] = list(img.size)
            t0 = time.time()
            res = pr.probe(img)
            rec["probe_ms"] = round((time.time() - t0) * 1000, 1)
            p = {r["key"]: r["p_yes"] for r in res}
            rec["p"] = {k: round(v, 5) for k, v in p.items()}
            rec["min_yes_no_mass"] = round(min(r["yes_no_mass"] for r in res), 5)
            if rec["kind"] == "stage2":
                rec["probe"] = rules.combine(rec["v0_types"], p, cfg, labels)
            else:
                rec["safety_net"] = rules.safety_net(p, cfg)
            img.close()
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
        done[row["id"]] = rec
        ckpt.write_text(json.dumps({"config": cfg, "rows": done}), encoding="utf-8")
        gc.collect()
        torch.cuda.empty_cache()
        if "error" in rec:
            print(f"[{i}/{len(rows)}] ERROR {rec['error']}")
        elif rec["kind"] == "stage2":
            print(f"[{i}/{len(rows)}] v0={rec['v0_types']} -> probe={rec['probe']['types']} "
                  f"ind={rec['probe']['indicators']} ({rec['probe_ms']:.0f} ms)", flush=True)
        else:
            print(f"[{i}/{len(rows)}] Stage 1 Normal -> safety net {rec['safety_net']}", flush=True)
    client.close()

    ok_all = [r for r in done.values() if "error" not in r]
    ok = [r for r in ok_all if r["kind"] == "stage2"]
    nrm = [r for r in ok_all if r["kind"] == "stage1_normal"]
    th = CONFIDENCE_THRESHOLD

    def s1_pass(r):
        return (r.get("stage1_confidence") or 0) >= th

    def dist(key):
        c = Counter()
        for r in ok:
            for t in (r["v0_types"] if key == "v0" else r["probe"]["types"]):
                c[t] += 1
        return dict(c.most_common())

    summary = {
        "n": len(ok),
        "errors": sum(1 for r in done.values() if "error" in r),
        "since": args.since,
        "config_provenance": cfg.get("provenance"),
        "parity": parity,
        "identical_type_sets": sum(1 for r in ok if set(r["v0_types"]) == set(r["probe"]["types"])),
        "label_counts_v0": dist("v0"),
        "label_counts_probe": dist("probe"),
        "added_by_probe": dict(Counter(t for r in ok for t in r["probe"]["added_by_probe"]).most_common()),
        "removed_by_probe": dict(Counter(t for r in ok for t in r["probe"]["removed_by_probe"]).most_common()),
        "indicators": dict(Counter(t for r in ok for t in r["probe"]["indicators"]).most_common()),
        "no_type_after_probe": sum(1 for r in ok if not r["probe"]["types"]),
        "review_load": {
            "production_field_gate": sum(1 for r in ok if not (s1_pass(r) and (r["field_confidence"] or 0) >= th)) / max(len(ok), 1),
            "probe_gate": sum(1 for r in ok if not (s1_pass(r) and r["probe"]["types"]
                                                    and r["probe"]["confidence"] >= th)) / max(len(ok), 1),
            "threshold": th,
            "note": "share of Stage-2 rows that would go to expert review (Stage 1 "
                    "confidence also has to clear the threshold, as in the worker)",
        },
        "stage1_safety_net": {
            "stage1_normal_rows": len(nrm),
            "auto_classified_normal_today": sum(1 for r in nrm if (r.get("stage1_confidence") or 0) >= th),
            "flagged_by_probe": sum(1 for r in nrm if (r.get("safety_net") or {}).get("flagged")),
            "flagged_among_auto_classified": sum(1 for r in nrm if (r.get("stage1_confidence") or 0) >= th
                                                 and (r.get("safety_net") or {}).get("flagged")),
            "strongest_types": dict(Counter((r.get("safety_net") or {}).get("strongest_type")
                                            for r in nrm if (r.get("safety_net") or {}).get("flagged")).most_common()),
        },
        "rows_by_kind": dict(Counter(r["kind"] for r in ok_all)),
        "probe_ms_median": sorted(r["probe_ms"] for r in ok_all)[len(ok_all) // 2] if ok_all else None,
        "min_yes_no_mass": min((r["min_yes_no_mass"] for r in ok_all), default=None),
    }
    out = {"summary": summary, "rows": list(done.values())}
    (EVAL_DIR / args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    try:
        ckpt.unlink()
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
