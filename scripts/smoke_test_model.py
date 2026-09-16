"""
Pre-flight gate for a VLM swap. Run this BEFORE trusting a new model.

Verifies that a checkpoint works end-to-end with the existing pipeline —
loading, all three inference stages, output parsing, IRC:82 label compliance,
confidence sanity — on the committed test fixtures. Every check is pass/fail
with a printed reason, and the script exits non-zero if any hard check fails.

This exists because a model swap can fail in ways that look like success: a
wrong chat template produces fluent text that parses to zero distress types; a
processor mismatch silently ignores the image and the model answers from the
prompt alone. Both would show up as "working" in a casual test and as garbage
in the database a thousand images later.

Usage:
    venv/Scripts/python.exe scripts/smoke_test_model.py
    venv/Scripts/python.exe scripts/smoke_test_model.py --model Qwen/Qwen3-VL-8B-Instruct
    venv/Scripts/python.exe scripts/smoke_test_model.py --model Qwen/Qwen3-VL-8B-Instruct \
        --quantization-bits 4 --json eval_results/smoke_qwen3vl.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image  # noqa: E402

RESULTS: list[dict] = []


def check(name: str, ok: bool, detail: str = "", hard: bool = True) -> bool:
    """Record and print a check. hard=False marks it informational."""
    RESULTS.append({"name": name, "ok": bool(ok), "detail": detail, "hard": hard})
    tag = "PASS" if ok else ("FAIL" if hard else "WARN")
    print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MODEL_PATH",
                                                      "Qwen/Qwen2.5-VL-7B-Instruct"))
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--quantization-bits", type=int, default=None, choices=[0, 4, 8])
    ap.add_argument("--prompts-version", default=os.environ.get("PROMPTS_VERSION", "v2"),
                    choices=["v1", "v2"])
    ap.add_argument("--json", default=None, help="write the full report here")
    args = ap.parse_args()

    # Prompt selection happens at import time in app.model, so set it first.
    os.environ["PROMPTS_VERSION"] = args.prompts_version
    if args.quantization_bits is not None:
        os.environ["QUANTIZATION_BITS"] = str(args.quantization_bits)

    print("=" * 70)
    print(f"MODEL SMOKE TEST — {args.model}")
    print(f"prompts={args.prompts_version} adapter={args.adapter_path or 'none'} "
          f"quant={os.environ.get('QUANTIZATION_BITS', '4')}-bit")
    print("=" * 70)

    from app.model import PavementClassifier, STAGE2_CONFIDENCE_MODE
    from scripts.irc82_taxonomy import IRC82_DISTRESS_TAXONOMY, PATCH_LABEL_TREATED_AS_VALID
    from scripts.utils import CONFIDENCE_THRESHOLD

    fixtures_dir = PROJECT_ROOT / "test_fixtures"
    fixtures = sorted(fixtures_dir.glob("*.jpg")) + sorted(fixtures_dir.glob("*.png"))
    if not fixtures:
        print(f"FATAL: no test fixtures in {fixtures_dir}")
        return 2
    print(f"\nFixtures: {[f.name for f in fixtures]}\n")

    # ---- 1. Load -----------------------------------------------------------
    print("[1/5] Loading model (includes loader self-test)...")
    t0 = time.time()
    try:
        clf = PavementClassifier(
            model_path=args.model,
            adapter_path=args.adapter_path,
            quantization_bits=int(os.environ.get("QUANTIZATION_BITS", "4")),
        )
    except Exception as e:
        check("model loads", False, f"{type(e).__name__}: {e}")
        return 1
    load_s = time.time() - t0
    info = clf.load_info
    check("model loads", clf.is_loaded, f"{load_s:.0f}s on {clf.device}")
    check("loader self-test", info.get("self_test", {}).get("ok", False),
          str(info.get("self_test", {}).get("error") or "synthetic image generated a reply"))
    check("model family resolved", info.get("model_type") not in (None, "unknown"),
          f"{info.get('model_type')} via {info.get('model_class')}")
    qb = info.get("quantization_bits")
    check("no OOM fallback", not info.get("oom_fallback_used", False),
          ("running bf16 (no quantization)" if qb == 0 else f"running {qb}-bit")
          + (" (DOWNGRADED from request)" if info.get("oom_fallback_used") else ""),
          hard=False)
    check("model fully on GPU", info.get("fully_on_gpu", True),
          f"placement={info.get('hf_device_map_summary')} — CPU/disk offload "
          f"makes inference 10-30x slower and invalidates timing comparisons")
    if info.get("vram_allocated_gb") is not None:
        print(f"  [INFO] VRAM: {info['vram_allocated_gb']} GB tensors / "
              f"{info.get('vram_reserved_gb')} GB reserved / "
              f"{info.get('vram_total_gb')} GB total")
    check("stage 1 token ids cached",
          bool(clf._normal_token_ids) and bool(clf._distress_token_ids),
          f"Normal={clf._normal_token_ids[:2]} Distress={clf._distress_token_ids[:2]}")

    # 'Normal' and 'Unknown' are legitimate parser outputs, not label errors:
    # parse_stage2_response() emits ["Normal"] when the model reports no
    # distress, and ["Unknown"] when nothing could be canonicalized. Both are
    # handled downstream (Unknown routes to expert review). Only labels
    # outside the IRC taxonomy AND outside those two are real failures.
    valid_labels = (set(IRC82_DISTRESS_TAXONOMY)
                    | {PATCH_LABEL_TREATED_AS_VALID, "Normal", "Unknown"})
    per_image = []

    # ---- 2-4. Run all three stages on every fixture ------------------------
    for fp in fixtures:
        # test_fixtures/ deliberately contains a corrupt file for the pipeline
        # robustness suite. Skip anything PIL cannot decode instead of dying
        # partway through the report.
        try:
            img = Image.open(fp)
            img.load()
        except Exception as e:
            check(f"{fp.name}: decodable", False,
                  f"skipped — {type(e).__name__} (expected for corrupt fixtures)",
                  hard=False)
            continue

        print(f"\n[2/5] {fp.name} — Stage 0 (pavement pre-filter)")
        s0 = clf.predict_is_pavement(img)
        check(f"{fp.name}: stage 0 decision valid",
              s0["decision"] in ("yes", "no", "unsure"),
              f"{s0['decision']!r} in {s0['time_ms']:.0f}ms — raw={s0['raw']!r}")
        check(f"{fp.name}: stage 0 accepts real pavement", s0["is_pavement"],
              "fixtures are real RDD road photos; a reject here means the "
              "pre-filter prompt is broken for this model")

        print(f"[3/5] {fp.name} — Stage 1 (Normal/Distress)")
        s1 = clf.predict_stage1(img)
        check(f"{fp.name}: stage 1 label valid",
              s1["stage1_label"] in ("Normal", "Distress"),
              f"{s1['stage1_label']} conf={s1['stage1_confidence']:.3f} "
              f"({s1['stage1_time_ms']:.0f}ms)")
        check(f"{fp.name}: stage 1 confidence in range",
              0.0 <= s1["stage1_confidence"] <= 1.0,
              f"{s1['stage1_confidence']}")
        check(f"{fp.name}: stage 1 detects damage", s1["is_distressed"],
              "fixtures are known-damaged RDD images (D00/D20/D40)", hard=False)

        print(f"[4/5] {fp.name} — Stage 2 (IRC:82 classification)")
        s2 = clf.predict_stage2(img)
        types = s2["distress_types"]
        check(f"{fp.name}: stage 2 returned types", bool(types), f"{types}")
        unknown = [t for t in types if t not in valid_labels]
        check(f"{fp.name}: all labels are IRC:82 canonical", not unknown,
              f"non-IRC labels: {unknown}" if unknown else f"{types}")
        check(f"{fp.name}: severity is DB-writable",
              s2["severity"] in ("None", "Low", "Medium", "High", "Unknown"),
              f"{s2['severity']!r} (DB CHECK allows exactly these five)")
        check(f"{fp.name}: description non-empty", bool(s2["description"].strip()),
              s2["description"][:80])
        check(f"{fp.name}: both confidences computed",
              s2.get("stage2_confidence_field") is not None
              and s2.get("stage2_confidence_sequence") is not None,
              f"field={s2.get('stage2_confidence_field')} "
              f"sequence={s2.get('stage2_confidence_sequence')} "
              f"(active={s2.get('stage2_confidence_mode')})")
        check(f"{fp.name}: field span located", s2.get("stage2_field_span_found", False),
              "falls back to whole-sequence confidence when not found", hard=False)

        per_image.append({
            "fixture": fp.name,
            "stage0": {k: s0[k] for k in ("decision", "is_pavement", "time_ms")},
            "stage1": {k: s1[k] for k in ("stage1_label", "stage1_confidence",
                                          "stage1_time_ms")},
            "stage2": {
                "distress_types": types,
                "severity": s2["severity"],
                "description": s2["description"],
                "confidence_active": s2["stage2_confidence"],
                "confidence_field": s2.get("stage2_confidence_field"),
                "confidence_sequence": s2.get("stage2_confidence_sequence"),
                "field_span_found": s2.get("stage2_field_span_found"),
                "time_ms": s2["stage2_time_ms"],
            },
        })

    # ---- 5. Confidence-metric comparison -----------------------------------
    print("\n[5/5] Confidence metric comparison")
    fields = [p["stage2"]["confidence_field"] for p in per_image
              if p["stage2"]["confidence_field"] is not None]
    seqs = [p["stage2"]["confidence_sequence"] for p in per_image
            if p["stage2"]["confidence_sequence"] is not None]
    summary_conf = {}
    if fields and seqs:
        avg_f, avg_s = sum(fields) / len(fields), sum(seqs) / len(seqs)
        pass_f = sum(1 for v in fields if v >= CONFIDENCE_THRESHOLD)
        pass_s = sum(1 for v in seqs if v >= CONFIDENCE_THRESHOLD)
        summary_conf = {
            "avg_field": round(avg_f, 4),
            "avg_sequence": round(avg_s, 4),
            "delta": round(avg_f - avg_s, 4),
            "n": len(fields),
            "clear_threshold_field": pass_f,
            "clear_threshold_sequence": pass_s,
            "threshold": CONFIDENCE_THRESHOLD,
        }
        print(f"  avg field-restricted : {avg_f:.4f}  "
              f"({pass_f}/{len(fields)} clear the {CONFIDENCE_THRESHOLD} threshold)")
        print(f"  avg whole-sequence   : {avg_s:.4f}  "
              f"({pass_s}/{len(seqs)} clear the {CONFIDENCE_THRESHOLD} threshold)")
        print(f"  delta                : {avg_f - avg_s:+.4f}")
        print("  NOTE: a large positive delta means switching "
              "STAGE2_CONFIDENCE_MODE=field would auto-accept images that are "
              "currently routed to expert review. Calibrate on real data first.")
        check("field confidence >= sequence confidence", avg_f >= avg_s,
              "expected: restricting to label tokens removes high-entropy prose",
              hard=False)

    # ---- Report ------------------------------------------------------------
    hard_fails = [r for r in RESULTS if r["hard"] and not r["ok"]]
    warns = [r for r in RESULTS if not r["hard"] and not r["ok"]]
    print("\n" + "=" * 70)
    print(f"RESULT: {len(RESULTS) - len(hard_fails) - len(warns)} passed, "
          f"{len(warns)} warnings, {len(hard_fails)} hard failures")
    for r in hard_fails:
        print(f"  FAILED: {r['name']} — {r['detail']}")
    print("=" * 70)

    report = {
        "model": args.model,
        "adapter": args.adapter_path,
        "prompts_version": args.prompts_version,
        "confidence_mode": STAGE2_CONFIDENCE_MODE,
        "load_info": {k: v for k, v in info.items() if k != "preflight"},
        "preflight": info.get("preflight"),
        "load_seconds": round(load_s, 1),
        "checks": RESULTS,
        "per_image": per_image,
        "confidence_comparison": summary_conf,
        "passed": not hard_fails,
    }
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report written to {out}")

    return 0 if not hard_fails else 1


if __name__ == "__main__":
    sys.exit(main())
