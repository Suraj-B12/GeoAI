"""
Cross-dataset evaluation on Attain.

Tests how well the fine-tuned Qwen2.5-VL adapter (trained on RDD) generalises
to a geographically distinct dataset (Attain_SMP_WS_V2.0 — New Zealand roads,
847 images, 17 classes covering both type + severity).

Three-tier reporting:
  1. In-distribution classes (also in RDD training): D00 longitudinal /
     D10 transverse / D20 alligator / D40 pothole. Should perform well.
  2. Zero-shot classes (NOT in RDD training): block crack, raveling, weathering,
     patch/utility cut. Recognition only via taxonomy injection in the system
     prompt — measures open-world VLM capability.
  3. Severity classification. Attain provides High/Low per instance.

Output: eval_results/attain_cross_dataset_results.json + per-class table.

Usage:
    python scripts/07_cross_dataset_eval.py
    python scripts/07_cross_dataset_eval.py --max-samples 100
    python scripts/07_cross_dataset_eval.py --adapter-path adapters/v2-rdd-2epochs-20260507/
    python scripts/07_cross_dataset_eval.py --subset OS_V1.0   # smaller subset (637 imgs, 2 classes)
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils import (
    EVAL_DIR,
    STAGE1_SYSTEM_PROMPT,
    STAGE1_USER_PROMPT,
    STAGE2_SYSTEM_PROMPT,
    STAGE2_USER_PROMPT,
    map_stage2_to_rdd_ids,
    parse_stage1_response,
    parse_stage2_response,
)

# Import the model loader from the existing eval script
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "baseline_eval", PROJECT_ROOT / "scripts" / "03_baseline_eval.py"
)
baseline_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline_eval)


# ============================================================
# Attain class mapping
# ============================================================
ATTAIN_ROOT = PROJECT_ROOT / "Attain" / "Attain" / "Attain"

# Map Attain class names (with severity suffix stripped) to:
#   - our pipeline label
#   - whether it was in RDD training (in_distribution) or zero-shot
ATTAIN_CLASS_MAP = {
    # In-distribution (trained on RDD)
    "Alligator crack":          {"label": "Alligator Crack (D20)",     "in_dist": True,  "rdd_id": 2},
    "Linear crack":             {"label": "Linear Crack",              "in_dist": True,  "rdd_id": -2},  # Maps to D00 OR D10 (Attain doesn't split)
    "Pothole":                  {"label": "Pothole (D40)",             "in_dist": True,  "rdd_id": 3},
    # Zero-shot (NOT in RDD training, only known via taxonomy injection)
    "Block crack":              {"label": "Block Crack (D43)",         "in_dist": False, "rdd_id": -3},
    "Patch and utility cut":    {"label": "Inlaid Patch (D44)",        "in_dist": False, "rdd_id": -3},
    "Patch":                    {"label": "Inlaid Patch (D44)",        "in_dist": False, "rdd_id": -3},
    "Raveling":                 {"label": "Raveling",                  "in_dist": False, "rdd_id": -3},
    "Weathering":               {"label": "Weathering/Oxidation",      "in_dist": False, "rdd_id": -3},
    # Excluded (not pavement distress)
    "Faded marking":            {"label": "EXCLUDE",                   "in_dist": None,  "rdd_id": None},
    "Lane shoulder drop-off":   {"label": "EXCLUDE",                   "in_dist": None,  "rdd_id": None},
    "Manhole":                  {"label": "EXCLUDE",                   "in_dist": None,  "rdd_id": None},
}


def parse_attain_class(raw: str) -> tuple[str, str]:
    """Split 'Alligator crack - High' into ('Alligator crack', 'High')."""
    if " - " not in raw:
        return raw.strip(), "Unknown"
    name, sev = raw.rsplit(" - ", 1)
    sev = sev.strip().lower()
    if sev in ("high", "medium", "low"):
        return name.strip(), sev.capitalize()
    return raw.strip(), "Unknown"


# ============================================================
# Label parsers (per-subset format)
# ============================================================

def parse_xml_labels(label_path: Path) -> list[tuple[str, str]]:
    """Parse Pascal-VOC-style XML (WS_V2.0). Returns list of (class, severity)."""
    out = []
    try:
        tree = ET.parse(label_path)
        for obj in tree.findall(".//object"):
            name_el = obj.find("name")
            if name_el is not None and name_el.text:
                cls, sev = parse_attain_class(name_el.text)
                out.append((cls, sev))
    except ET.ParseError:
        pass
    return out


def parse_yolo_labels(label_path: Path, class_names: list[str]) -> list[tuple[str, str]]:
    """Parse YOLO-format text (OS_V1.0, WS_V1.0). Each line is `class_id x1 y1 x2 y2 ...`."""
    out = []
    try:
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            try:
                cid = int(parts[0])
                if 0 <= cid < len(class_names):
                    cls, sev = parse_attain_class(class_names[cid])
                    out.append((cls, sev))
            except (ValueError, IndexError):
                continue
    except OSError:
        pass
    return out


def get_subset_info(subset: str) -> dict:
    """Return paths and metadata for a given subset name (e.g. 'WS_V2.0')."""
    subset_dir = ATTAIN_ROOT / f"Attain_SMP_{subset}"
    yaml_path = subset_dir / f"Attain_SMP_{subset}_data.yaml"
    images_dir = subset_dir / "Images"
    labels_dir = subset_dir / "Labels"

    # Parse class names from YAML (rough; YAML has comments, so do it manually)
    class_names = []
    if yaml_path.exists():
        in_names = False
        for line in yaml_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s.startswith("names:"):
                in_names = True
                continue
            if in_names:
                if s.startswith("]"):
                    break
                # Lines like:    'Alligator crack - High',
                s = s.strip(",").strip().strip("'").strip('"')
                if s and not s.startswith("#") and not s.startswith("nc:"):
                    class_names.append(s)

    # Detect label format
    xml_count = len(list(labels_dir.glob("*.xml"))) if labels_dir.exists() else 0
    txt_count = len(list(labels_dir.glob("*.txt"))) if labels_dir.exists() else 0
    label_format = "xml" if xml_count > txt_count else "yolo"

    return {
        "subset": subset,
        "subset_dir": subset_dir,
        "images_dir": images_dir,
        "labels_dir": labels_dir,
        "class_names": class_names,
        "label_format": label_format,
        "xml_count": xml_count,
        "txt_count": txt_count,
    }


def load_ground_truth(info: dict) -> list[dict]:
    """Build list of {image_path, gt_classes (set), gt_severities (set)}."""
    gt = []
    images = sorted(info["images_dir"].glob("*.jpg")) + sorted(info["images_dir"].glob("*.png"))
    for img_path in images:
        label_path = info["labels_dir"] / (img_path.stem + (".xml" if info["label_format"] == "xml" else ".txt"))
        if not label_path.exists():
            continue
        if info["label_format"] == "xml":
            objs = parse_xml_labels(label_path)
        else:
            objs = parse_yolo_labels(label_path, info["class_names"])

        # Aggregate per-image: unique classes, max severity per class
        sev_rank = {"Low": 1, "Medium": 2, "High": 3, "Unknown": 0}
        cls_to_sev: dict[str, str] = {}
        for cls, sev in objs:
            cur = cls_to_sev.get(cls)
            if cur is None or sev_rank.get(sev, 0) > sev_rank.get(cur, 0):
                cls_to_sev[cls] = sev

        # Filter excluded classes
        kept = {c: s for c, s in cls_to_sev.items()
                if ATTAIN_CLASS_MAP.get(c, {}).get("label") != "EXCLUDE"}
        if not kept:
            continue  # image only had excluded classes, skip

        gt.append({
            "image_path": str(img_path),
            "gt_attain_classes": list(kept.keys()),
            "gt_severities": kept,
            "gt_pipeline_labels": [
                ATTAIN_CLASS_MAP[c]["label"] for c in kept.keys()
                if c in ATTAIN_CLASS_MAP
            ],
            "any_in_distribution": any(
                ATTAIN_CLASS_MAP.get(c, {}).get("in_dist") is True for c in kept.keys()
            ),
            "any_zero_shot": any(
                ATTAIN_CLASS_MAP.get(c, {}).get("in_dist") is False for c in kept.keys()
            ),
        })
    return gt


# ============================================================
# Inference + matching
# ============================================================

def _norm(s: str) -> str:
    """Lowercase, strip parenthetical codes, drop punctuation. For fuzzy matching."""
    import re
    s = re.sub(r"\(.*?\)", "", s).lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())


PIPELINE_TO_ATTAIN = {
    "Longitudinal Crack (D00)":  ["Linear crack"],
    "Transverse Crack (D10)":    ["Linear crack"],
    "Alligator Crack (D20)":     ["Alligator crack"],
    "Pothole (D40)":             ["Pothole"],
    "Block Crack (D43)":         ["Block crack"],
    "Inlaid Patch (D44)":        ["Patch and utility cut", "Patch"],
    "Raveling":                  ["Raveling"],
    "Weathering/Oxidation":      ["Weathering"],
}


def map_pipeline_to_attain_classes(pipeline_types: list[str]) -> set[str]:
    """Convert model output (pipeline taxonomy) back to Attain class names for comparison."""
    out: set[str] = set()
    for t in pipeline_types:
        # Direct lookup
        if t in PIPELINE_TO_ATTAIN:
            out.update(PIPELINE_TO_ATTAIN[t])
            continue
        # Fuzzy: search by normalised name
        n = _norm(t)
        for pipe_label, attain_classes in PIPELINE_TO_ATTAIN.items():
            if _norm(pipe_label) in n or n in _norm(pipe_label):
                out.update(attain_classes)
                break
    return out


def evaluate(args):
    """Main eval loop."""
    info = get_subset_info(args.subset)
    if not info["images_dir"].exists():
        print(f"ERROR: Attain subset not found at {info['subset_dir']}")
        sys.exit(1)

    print(f"Subset:        {args.subset}")
    print(f"Path:          {info['subset_dir']}")
    print(f"Label format:  {info['label_format']} ({info['xml_count']} xml, {info['txt_count']} yolo)")
    print(f"Classes ({len(info['class_names'])}):")
    for c in info["class_names"]:
        cls, _ = parse_attain_class(c)
        m = ATTAIN_CLASS_MAP.get(cls, {})
        flag = "EXCLUDE" if m.get("label") == "EXCLUDE" else (
            "in-dist" if m.get("in_dist") else ("zero-shot" if m.get("in_dist") is False else "unmapped")
        )
        print(f"  - {c}  [{flag}]")

    print()
    print("Loading ground truth...")
    gt = load_ground_truth(info)
    print(f"  {len(gt)} usable images (after EXCLUDE filter)")

    if args.max_samples > 0:
        gt = gt[: args.max_samples]
        print(f"  Limited to {len(gt)} samples")

    # Load model
    model, processor = baseline_eval.load_model(
        args.model, args.adapter_path, args.quantization_bits
    )

    # Choose prompt set based on --prompts-version
    if args.prompts_version == "v2":
        from scripts.utils_v2_prompts import (
            STAGE1_SYSTEM_PROMPT_V2 as S1_SYS,
            STAGE1_USER_PROMPT_V2 as S1_USR,
            STAGE2_SYSTEM_PROMPT_V2 as S2_SYS,
            STAGE2_USER_PROMPT_V2 as S2_USR,
        )
        print(f"Prompts: V2 ('insane' — deep persona + high-stakes negative-sentiment)")
    else:
        S1_SYS, S1_USR = STAGE1_SYSTEM_PROMPT, STAGE1_USER_PROMPT
        S2_SYS, S2_USR = STAGE2_SYSTEM_PROMPT, STAGE2_USER_PROMPT
        print(f"Prompts: V1 (current scripts/utils.py)")

    # Checkpoint config — crash-resume support
    ckpt_name = f"attain_{args.subset}_{args.prompts_version}_"
    ckpt_name += "adapter" if args.adapter_path else "baseline"
    ckpt_path = EVAL_DIR / f".{ckpt_name}_checkpoint.json"
    ckpt_tmp = EVAL_DIR / f".{ckpt_name}_checkpoint.tmp"

    # Resume if checkpoint exists
    results = []
    in_dist_correct = in_dist_total = 0
    zero_shot_correct = zero_shot_total = 0
    sev_correct = sev_total = 0
    per_class_tp: dict[str, int] = defaultdict(int)
    per_class_fp: dict[str, int] = defaultdict(int)
    per_class_fn: dict[str, int] = defaultdict(int)
    processed_image_names: set[str] = set()

    if ckpt_path.exists():
        try:
            with ckpt_path.open(encoding="utf-8") as f:
                state = json.load(f)
            results = state.get("results", [])
            in_dist_correct = state.get("in_dist_correct", 0)
            in_dist_total = state.get("in_dist_total", 0)
            zero_shot_correct = state.get("zero_shot_correct", 0)
            zero_shot_total = state.get("zero_shot_total", 0)
            sev_correct = state.get("sev_correct", 0)
            sev_total = state.get("sev_total", 0)
            per_class_tp = defaultdict(int, state.get("per_class_tp", {}))
            per_class_fp = defaultdict(int, state.get("per_class_fp", {}))
            per_class_fn = defaultdict(int, state.get("per_class_fn", {}))
            processed_image_names = {r["image"] for r in results}
            print(f"Resumed from checkpoint: {len(results)} images already processed")
        except Exception as e:
            print(f"WARNING: checkpoint {ckpt_path} unreadable ({e}), starting fresh")
            results = []

    def save_checkpoint():
        """Atomic-ish checkpoint via .tmp + replace, with .bak fallback for Windows."""
        state = {
            "results": results,
            "in_dist_correct": in_dist_correct,
            "in_dist_total": in_dist_total,
            "zero_shot_correct": zero_shot_correct,
            "zero_shot_total": zero_shot_total,
            "sev_correct": sev_correct,
            "sev_total": sev_total,
            "per_class_tp": dict(per_class_tp),
            "per_class_fp": dict(per_class_fp),
            "per_class_fn": dict(per_class_fn),
        }
        try:
            with ckpt_tmp.open("w", encoding="utf-8") as f:
                json.dump(state, f)
            # On Windows, os.replace works if dest doesn't exist or is unlocked
            if ckpt_path.exists():
                bak = ckpt_path.with_suffix(".json.bak")
                if bak.exists(): bak.unlink()
                ckpt_path.rename(bak)
            ckpt_tmp.rename(ckpt_path)
        except Exception as e:
            print(f"WARNING: checkpoint save failed: {e}")

    # Inference loop
    for entry in tqdm(gt, desc="Attain eval"):
        img_path = Path(entry["image_path"])
        if img_path.name in processed_image_names:
            continue  # skip already-processed (resume)
        try:
            image = Image.open(img_path).convert("RGB")

            # Stage 1
            s1_resp = baseline_eval.run_inference(
                model, processor, image,
                S1_SYS, S1_USR, max_new_tokens=20
            )
            s1_id = parse_stage1_response(s1_resp)
            is_distressed = s1_id == 1

            # Stage 2 only if Stage 1 says distressed (matches production pipeline)
            if is_distressed:
                s2_resp = baseline_eval.run_inference(
                    model, processor, image,
                    S2_SYS, S2_USR, max_new_tokens=200
                )
                parsed = parse_stage2_response(s2_resp)
                pred_pipeline = parsed["distress_types"]
                pred_severity = parsed["severity"]
            else:
                pred_pipeline = []
                pred_severity = "None"

            pred_attain = map_pipeline_to_attain_classes(pred_pipeline)
            gt_attain = set(entry["gt_attain_classes"])

            # Per-class TP/FP/FN
            tracked = set(ATTAIN_CLASS_MAP.keys()) - {c for c, m in ATTAIN_CLASS_MAP.items()
                                                       if m.get("label") == "EXCLUDE"}
            for cls in tracked:
                in_pred = cls in pred_attain
                in_gt   = cls in gt_attain
                if in_pred and in_gt:
                    per_class_tp[cls] += 1
                elif in_pred and not in_gt:
                    per_class_fp[cls] += 1
                elif not in_pred and in_gt:
                    per_class_fn[cls] += 1

            # Tier 1 / Tier 2 accuracy
            for cls in entry["gt_attain_classes"]:
                in_dist = ATTAIN_CLASS_MAP.get(cls, {}).get("in_dist")
                if in_dist is True:
                    in_dist_total += 1
                    if cls in pred_attain:
                        in_dist_correct += 1
                elif in_dist is False:
                    zero_shot_total += 1
                    if cls in pred_attain:
                        zero_shot_correct += 1

            # Severity check (use model's overall severity vs the highest severity in GT)
            sev_rank = {"Low": 1, "Medium": 2, "High": 3, "Unknown": 0, "None": 0}
            gt_top_sev = max(entry["gt_severities"].values(), key=lambda s: sev_rank.get(s, 0)) \
                          if entry["gt_severities"] else "Unknown"
            pred_sev = pred_severity if pred_severity in ("Low", "Medium", "High") else "Unknown"
            if gt_top_sev != "Unknown":
                sev_total += 1
                if pred_sev == gt_top_sev:
                    sev_correct += 1

            results.append({
                "image": img_path.name,
                "gt_attain": list(gt_attain),
                "gt_severity_top": gt_top_sev,
                "pred_pipeline": pred_pipeline,
                "pred_attain": list(pred_attain),
                "pred_severity": pred_sev,
                "is_distressed_pred": is_distressed,
                "stage1_raw": s1_resp,
                "stage2_raw": s2_resp if is_distressed else None,
            })

            image.close()
            del image
            if len(results) % 25 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                save_checkpoint()

        except Exception as e:
            print(f"\n  ERROR on {img_path.name}: {e}")

    # Final checkpoint save before computing metrics
    save_checkpoint()

    # Compute aggregate metrics
    def _f1(tp, fp, fn):
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec  = tp / (tp + fn) if (tp + fn) else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        return prec, rec, f1

    per_class_metrics = {}
    for cls in (set(per_class_tp) | set(per_class_fp) | set(per_class_fn)):
        m = ATTAIN_CLASS_MAP.get(cls, {})
        prec, rec, f1 = _f1(per_class_tp[cls], per_class_fp[cls], per_class_fn[cls])
        per_class_metrics[cls] = {
            "tier": "in-dist" if m.get("in_dist") else "zero-shot",
            "tp": per_class_tp[cls],
            "fp": per_class_fp[cls],
            "fn": per_class_fn[cls],
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }

    summary = {
        "subset": args.subset,
        "n_images": len(gt),
        "n_processed": len(results),
        "tier1_in_distribution": {
            "instances": in_dist_total,
            "correct": in_dist_correct,
            "accuracy": round(in_dist_correct / max(in_dist_total, 1), 4),
        },
        "tier2_zero_shot": {
            "instances": zero_shot_total,
            "correct": zero_shot_correct,
            "accuracy": round(zero_shot_correct / max(zero_shot_total, 1), 4),
        },
        "severity": {
            "instances": sev_total,
            "correct": sev_correct,
            "accuracy": round(sev_correct / max(sev_total, 1), 4),
        },
        "per_class": per_class_metrics,
        "adapter_path": args.adapter_path,
    }

    # Print summary
    print()
    print("=" * 70)
    print("ATTAIN CROSS-DATASET EVALUATION RESULTS")
    print("=" * 70)
    print(f"Subset: {args.subset}  ({summary['n_processed']}/{summary['n_images']} images processed)")
    print()
    print("Tier 1 — In-distribution classes (also in RDD training):")
    t1 = summary["tier1_in_distribution"]
    print(f"  {t1['correct']:4d} / {t1['instances']:4d}  ({t1['accuracy']:.2%})")
    print("Tier 2 — Zero-shot classes (only via taxonomy injection):")
    t2 = summary["tier2_zero_shot"]
    print(f"  {t2['correct']:4d} / {t2['instances']:4d}  ({t2['accuracy']:.2%})")
    print("Severity classification (when annotated):")
    sv = summary["severity"]
    print(f"  {sv['correct']:4d} / {sv['instances']:4d}  ({sv['accuracy']:.2%})")
    print()
    print("Per-class:")
    print(f"  {'Class':28s} {'tier':10s} {'TP':>4} {'FP':>4} {'FN':>4} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    for cls, m in sorted(per_class_metrics.items(), key=lambda x: (x[1]["tier"], x[0])):
        print(f"  {cls:28s} {m['tier']:10s} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4}"
              f" {m['precision']:>7.3f} {m['recall']:>7.3f} {m['f1']:>7.3f}")

    # Save
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_name = args.output_name or f"attain_{args.prompts_version}_"
    if not args.output_name:
        out_name += "adapter" if args.adapter_path else "baseline"
        out_name += "_results.json"
    out = EVAL_DIR / out_name
    summary["prompts_version"] = args.prompts_version
    with out.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_image_results": results}, f, indent=2)
    print(f"\nResults saved to: {out}")

    # Clear checkpoint on success
    try:
        if ckpt_path.exists():
            ckpt_path.unlink()
        bak = ckpt_path.with_suffix(".json.bak")
        if bak.exists():
            bak.unlink()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Cross-dataset eval on Attain")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--adapter-path", default=None,
                        help="LoRA adapter path. None = baseline (no fine-tuning).")
    parser.add_argument("--subset", default="WS_V2.0", choices=["OS_V1.0", "WS_V1.0", "WS_V2.0"])
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Limit number of test images (0 = all)")
    parser.add_argument("--quantization-bits", type=int, default=None, choices=[0, 4, 8])
    parser.add_argument("--prompts-version", default="v1", choices=["v1", "v2"],
                        help="v1 = current scripts/utils.py prompts. "
                             "v2 = scripts/utils_v2_prompts.py (insane persona + stakes).")
    parser.add_argument("--output-name", default=None,
                        help="Custom output filename. Default: attain_<prompts>_<mode>_results.json")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
