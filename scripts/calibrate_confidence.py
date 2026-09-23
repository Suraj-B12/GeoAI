"""
Does the Stage 2 confidence score actually predict whether the model is right?

A confidence score is only useful if it SEPARATES correct predictions from
wrong ones. "Higher numbers" is not an improvement: a metric returning 0.99
for everything is worse than useless, because the expert-review gate would
auto-accept every mistake.

This runs the PRODUCTION path (PavementClassifier, the same code the worker
uses) over labelled images, records both confidence metrics for every image,
and reports AUC, separation, and what each candidate threshold would cost.

Datasets
--------
  attain     Attain SMP (New Zealand, vehicle-mounted). 10 annotated classes,
             7 of which are real pavement distress.
  rdd_india  The India subset of RDD2022 (Indian roads, the target geography).
             4 annotated classes: D00/D10/D20/D40.
  mixed      Both, sampled independently and pooled. Each row keeps its
             `dataset` field so results can be split back apart.

Three things this script is careful about
-----------------------------------------
1. COMPARISON HAPPENS IN EACH DATASET'S OWN LABEL SPACE.
   RDD annotates 4 classes; Attain annotates 7. If the model says
   "Potholes + Bleeding" on an RDD image whose ground truth is "Potholes",
   Bleeding is NOT a false positive - RDD never annotates bleeding, so the
   dataset can neither confirm nor deny it. Scoring it as an error would
   manufacture failures out of annotation coverage. Every predicted label is
   therefore intersected with the dataset's annotated label space before being
   compared, and the labels dropped that way are recorded per row
   (`pred_unevaluable`) so the size of the effect is visible rather than
   assumed.

2. SAMPLING IS RANDOM, NOT THE FIRST N.
   Both datasets are stored in sorted filename order, which correlates with
   capture session, road and therefore distress type. Taking the first N
   samples one stretch of road. Sampling is seeded so a run is reproducible.
   It is deliberately NOT stratified by class: AUC has to be measured on the
   distribution the gate will actually see, and balancing classes would
   distort the prevalence that the threshold decision depends on.

3. CORRECTNESS IS AMBIGUOUS IN MULTI-LABEL, SO ALL DEFINITIONS ARE STORED.
   The previous version of this script scored only `any_overlap`, under which
   all 37 images came out correct, leaving zero negatives and an undefined
   AUC - while the reported headline AUC came from a stricter rule that the
   script did not implement. Every definition below is now computed and
   stored per row, and every AUC states which rule produced it.

     any_overlap        at least one predicted label is in ground truth
     no_false_positives everything predicted is in ground truth (PRIMARY:
                        a false positive is what sends a repair crew to the
                        wrong defect)
     exact_match        predicted set equals ground truth set
     jaccard_50         intersection over union >= 0.5

Usage:
    venv/Scripts/python.exe scripts/calibrate_confidence.py --dataset mixed --n 400
    venv/Scripts/python.exe scripts/calibrate_confidence.py --dataset rdd_india --n 60 \
        --out pilot_rdd.json
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "attain_eval", PROJECT_ROOT / "scripts" / "07_cross_dataset_eval.py"
)
_ae = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ae)

from scripts.irc82_taxonomy import canonicalize_to_irc  # noqa: E402
from scripts.utils import RDD_LABEL_MAP  # noqa: E402

RDD_TEST_IMAGES = PROJECT_ROOT / "RDD" / "RDD_SPLIT" / "test" / "images"
RDD_TEST_LABELS = PROJECT_ROOT / "RDD" / "RDD_SPLIT" / "test" / "labels"

# RDD2022 class 4 is the benchmark's noise/other bucket, not a distress type.
RDD_VALID_CLASS_IDS = (0, 1, 2, 3)


# ============================================================
# Label spaces
# ============================================================

def _irc(labels) -> set:
    """Canonicalise any label spelling into the IRC:82 name the model emits."""
    out = set()
    for raw in labels:
        c = canonicalize_to_irc(raw)
        if c:
            out.add(c)
    return out


# Each dataset is scored in ITS OWN annotation vocabulary, never in a shared
# "IRC space". The two taxonomies disagree on granularity in a way that a
# shared space silently destroys:
#
#   Attain annotates one merged "Linear crack" class.
#   RDD annotates Longitudinal (D00) and Transverse (D10) separately.
#   The model emits the IRC names "Longitudinal Cracking" / "Transverse
#   Cracking".
#
# canonicalize_to_irc("Linear crack") returns "Linear crack" UNCHANGED - it is
# not an IRC type, it just passes through the alias table. So an IRC-space
# intersection drops every longitudinal and transverse prediction on Attain as
# "unevaluable", which would erase most of the Attain signal without raising
# anything. That is the same failure mode as the PIPELINE_TO_ATTAIN bug that
# silently zeroed the Ravelling and Hungry Surface scores.
#
# So: Attain is scored through map_pipeline_to_attain_classes (already written
# and already fixed for that bug), and RDD through the explicit table below.

def rdd_label_space() -> set:
    """The four classes RDD2022 annotates, in RDD vocabulary."""
    return {RDD_LABEL_MAP[i] for i in RDD_VALID_CLASS_IDS}


def attain_label_space() -> set:
    """The Attain classes that are real distress, in Attain vocabulary."""
    return {c for c, m in _ae.ATTAIN_CLASS_MAP.items()
            if m.get("label") != "EXCLUDE"}


# Pipeline IRC output -> RDD2022 class. Only these four have an RDD
# counterpart; anything else the model names (Bleeding, Edge Cracking,
# Rutting, ...) is outside what RDD annotates and cannot be adjudicated
# against it in either direction.
IRC_TO_RDD = {
    "Longitudinal Cracking": "Longitudinal Crack (D00)",
    "Transverse Cracking": "Transverse Crack (D10)",
    "Alligator Cracking": "Alligator Crack (D20)",
    "Potholes": "Pothole (D40)",
}


def map_pred_to_dataset_space(pred_raw: list, dataset: str) -> set:
    """Predicted labels expressed in the dataset own vocabulary."""
    if dataset == "attain":
        return _ae.map_pipeline_to_attain_classes(pred_raw)
    out = set()
    for raw in pred_raw:
        irc = canonicalize_to_irc(raw)
        if irc in IRC_TO_RDD:
            out.add(IRC_TO_RDD[irc])
    return out


# ============================================================
# Ground-truth loaders -> a single common row shape
# ============================================================

def load_attain(n: int, seed: int, subset: str) -> list[dict]:
    info = _ae.get_subset_info(subset)
    gt = _ae.load_ground_truth(info)
    rng = random.Random(seed)
    rng.shuffle(gt)
    rows = []
    for e in gt[:n]:
        gt_native = set(e["gt_attain_classes"])
        if not gt_native:
            continue
        rows.append({
            "dataset": "attain",
            "image_path": e["image_path"],
            "image": Path(e["image_path"]).name,
            "gt_native": sorted(gt_native),
            # IRC rendering is informational only - never used for scoring.
            "gt_irc": sorted(_irc(e["gt_pipeline_labels"])),
        })
    return rows


def load_attain_from_run(run_file: str, n: int, seed: int) -> list[dict]:
    """Attain images that a previous full run already established pass Stage 1.

    Stage 1 is deterministic under greedy decoding, so re-running it over the
    same images would spend roughly four fifths of the compute re-deriving a
    verdict already stored on disk. This reads that verdict instead and runs
    Stage 2 only.

    The stored `pred_attain` is carried through as `stored_pred_attain` so the
    new run can be checked against it: if the predictions do not reproduce,
    something about the model or prompts has changed since, and the stored
    accuracy numbers cannot be pooled with the new confidence numbers.
    """
    path = Path(run_file)
    if not path.is_absolute():
        path = PROJECT_ROOT / "eval_results" / run_file
    data = json.loads(path.read_text(encoding="utf-8"))
    stored = [r for r in data["per_image_results"] if r.get("is_distressed_pred")]

    info = _ae.get_subset_info("WS_V2.0")
    by_name = {Path(e["image_path"]).name: e for e in _ae.load_ground_truth(info)}

    rows = []
    for r in stored:
        e = by_name.get(r["image"])
        if e is None:
            continue
        rows.append({
            "dataset": "attain",
            "image_path": e["image_path"],
            "image": r["image"],
            "gt_native": sorted(set(r.get("gt_attain") or [])),
            "gt_irc": sorted(_irc(e["gt_pipeline_labels"])),
            "stored_pred_attain": sorted(set(r.get("pred_attain") or [])),
            "stage1_from_stored_run": True,
        })
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n] if n else rows


def attain_dominant_class(image_path: str) -> dict:
    """Which annotated Attain class covers the most area in the image.

    "Dominant" is measured as the summed bounding-box area per class, in
    Attain vocabulary, excluding the non-distress classes. Box area is a
    proxy: it overstates thin diagonal cracks, whose boxes are mostly
    pavement, and double counts overlapping boxes. It is the only
    per-instance extent the annotations carry. Returns {} when there are no
    usable boxes.
    """
    import xml.etree.ElementTree as ET
    info = _ae.get_subset_info("WS_V2.0")
    label = info["labels_dir"] / (Path(image_path).stem + ".xml")
    if not label.exists():
        return {}
    areas: dict = {}
    try:
        for obj in ET.parse(label).findall(".//object"):
            name = obj.findtext("name")
            box = obj.find("bndbox")
            if not name or box is None:
                continue
            cls, _sev = _ae.parse_attain_class(name)
            if _ae.ATTAIN_CLASS_MAP.get(cls, {}).get("label") == "EXCLUDE":
                continue
            w = float(box.findtext("xmax")) - float(box.findtext("xmin"))
            h = float(box.findtext("ymax")) - float(box.findtext("ymin"))
            if w > 0 and h > 0:
                areas[cls] = areas.get(cls, 0.0) + w * h
    except (ET.ParseError, TypeError, ValueError):
        return {}
    if not areas:
        return {}
    total = sum(areas.values())
    top = max(areas, key=areas.get)
    return {"gt_dominant": top,
            "gt_dominant_share": round(areas[top] / total, 4),
            "gt_area_share": {k: round(v / total, 4) for k, v in areas.items()}}


def load_rdd_india(n: int, seed: int) -> list[dict]:
    """India subset of the RDD2022 test split, read from YOLO label files."""
    if not RDD_TEST_IMAGES.is_dir():
        sys.exit(f"ERROR: RDD test images not found at {RDD_TEST_IMAGES}")
    images = sorted(p for p in RDD_TEST_IMAGES.glob("India_*.jpg"))
    rng = random.Random(seed)
    rng.shuffle(images)

    rows = []
    for img in images:
        if len(rows) >= n:
            break
        label = RDD_TEST_LABELS / (img.stem + ".txt")
        if not label.exists():
            continue
        ids = set()
        for line in label.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if not parts:
                continue
            try:
                cid = int(float(parts[0]))
            except ValueError:
                continue
            if cid in RDD_VALID_CLASS_IDS:
                ids.add(cid)
        if not ids:
            # Either an empty label file or class 4 (noise) only. Neither can
            # support a correctness judgement, so it is not a usable sample.
            continue
        rows.append({
            "dataset": "rdd_india",
            "image_path": str(img),
            "image": img.name,
            "gt_native": sorted(RDD_LABEL_MAP[i] for i in ids),
            "gt_irc": sorted(_irc(RDD_LABEL_MAP[i] for i in ids)),
        })
    return rows


# ============================================================
# Correctness definitions
# ============================================================

def correctness(pred_eval: set, gt: set) -> dict:
    """Every definition at once, on labels already in the dataset vocabulary.

    An empty prediction is wrong under all definitions - the model was shown a
    photo with annotated distress and named none of it.
    """
    if not pred_eval:
        return {"any_overlap": False, "no_false_positives": False,
                "exact_match": False, "jaccard_50": False, "jaccard": 0.0}
    inter = pred_eval & gt
    union = pred_eval | gt
    j = len(inter) / len(union) if union else 0.0
    return {
        "any_overlap": bool(inter),
        "no_false_positives": pred_eval <= gt,
        "exact_match": pred_eval == gt,
        "jaccard_50": j >= 0.5,
        "jaccard": round(j, 4),
    }


DEFINITIONS = ("no_false_positives", "any_overlap", "jaccard_50", "exact_match")
PRIMARY = "no_false_positives"


# ============================================================
# Crash-safe checkpointing
# ============================================================
# A bare write_text() truncates the file if the process dies mid-write, which
# loses the ENTIRE run rather than the current image - the worst possible
# failure for something that takes hours. Path.replace() is not a fix here:
# atomic rename is unreliable on Windows whenever any other process (Search
# Indexer, Defender, an open Explorer window) holds a read handle, which
# 03_baseline_eval.py established empirically in commit 80aff92.
#
# So: rotate the existing file to .bak (a rename needs no write access on the
# target), then write fresh. A crash mid-write leaves the previous good save
# in .bak, and the loader falls back to it.

def save_checkpoint(path: Path, data: dict) -> None:
    """Write the checkpoint, keeping the previous save as .bak.

    Save failures are non-fatal. A transient file lock must not kill a
    multi-hour evaluation: the state is still in memory and the next image
    will retry.
    """
    bak = path.with_suffix(".json.bak")
    try:
        if path.exists():
            try:
                if bak.exists():
                    bak.unlink()
                path.rename(bak)
            except OSError:
                pass  # cannot rotate - overwrite directly rather than stall
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except OSError as e:
        print(f"  [warn] checkpoint save failed: {e} - continuing in memory only")


def load_checkpoint(path: Path):
    """Load the checkpoint, falling back to .bak if the primary is corrupt."""
    for cand in (path, path.with_suffix(".json.bak")):
        if not cand.exists():
            continue
        try:
            blob = json.loads(cand.read_text(encoding="utf-8"))
            if cand != path:
                print(f"  [resume] primary checkpoint was unreadable; "
                      f"recovered {len(blob.get('rows', {}))} rows from {cand.name}")
            return blob
        except Exception as e:
            print(f"  [warn] {cand.name} unreadable ({type(e).__name__}); "
                  f"trying fallback")
    return None


# ============================================================
# Metrics
# ============================================================

def auc(pos: list, neg: list) -> float:
    """P(random correct scores above random incorrect). Ties count half."""
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


def auc_stderr(a: float, n_pos: int, n_neg: int) -> float:
    """Hanley-McNeil standard error - the honest width on a small sample."""
    if not (n_pos and n_neg) or a != a:
        return float("nan")
    q1 = a / (2 - a)
    q2 = 2 * a * a / (1 + a)
    v = (a * (1 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg)
    return v ** 0.5 if v > 0 else float("nan")


def summarize(rows: list, metric: str, definition: str, thresholds) -> dict:
    pos = [r[metric] for r in rows if r["correct"][definition]]
    neg = [r[metric] for r in rows if not r["correct"][definition]]
    a = auc(pos, neg)
    out = {
        "definition": definition,
        "n_correct": len(pos),
        "n_wrong": len(neg),
        "mean_correct": round(sum(pos) / len(pos), 4) if pos else None,
        "mean_wrong": round(sum(neg) / len(neg), 4) if neg else None,
        "separation": (round(sum(pos) / len(pos) - sum(neg) / len(neg), 4)
                       if pos and neg else None),
        "auc": round(a, 4) if a == a else None,
        "auc_stderr": (round(auc_stderr(a, len(pos), len(neg)), 4)
                       if a == a else None),
        "thresholds": {},
    }
    for th in thresholds:
        acc = [r for r in rows if r[metric] >= th]
        bad = [r for r in acc if not r["correct"][definition]]
        caught = [r for r in rows if r[metric] < th and not r["correct"][definition]]
        wasted = [r for r in rows if r[metric] < th and r["correct"][definition]]
        out["thresholds"][f"{th:.2f}"] = {
            "auto_accepted": len(acc),
            "auto_accepted_pct": round(100 * len(acc) / len(rows), 1) if rows else 0,
            "wrong_waved_through": len(bad),
            "auto_accept_error_rate_pct": (round(100 * len(bad) / len(acc), 1)
                                           if acc else None),
            "wrong_caught": len(caught),
            "wrong_caught_pct": (round(100 * len(caught) / len(neg), 1) if neg else None),
            "correct_sent_to_expert": len(wasted),
            "review_load_pct": round(100 * (len(rows) - len(acc)) / len(rows), 1) if rows else 0,
        }
    return out


# ============================================================
# Main
# ============================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mixed",
                    choices=["attain", "rdd_india", "mixed"])
    ap.add_argument("--n", type=int, default=400,
                    help="total images; for 'mixed' it is split evenly")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subset", default="WS_V2.0", help="Attain subset name")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--quantization-bits", type=int, default=4, choices=[0, 4, 8],
                    help="4 matches production; 0 is bf16")
    ap.add_argument("--out", default="confidence_calibration.json")
    ap.add_argument("--fresh", action="store_true", help="ignore any checkpoint")
    ap.add_argument("--thresholds", default="0.70,0.72,0.75,0.78,0.80,0.82,0.85,0.90")
    ap.add_argument("--skip-stage1", action="store_true",
                    help="run Stage 2 directly without the Stage 1 gate. Needed "
                         "for RDD-India, where Stage 1 rejects essentially every "
                         "image because they are wide dashcam traffic scenes "
                         "rather than the close-up pavement photos the prompts "
                         "were written for. The resulting population is NOT the "
                         "one production sees; the report records the flag.")
    ap.add_argument("--attain-from-run", default=None,
                    help="reuse the Stage 1 verdicts in a stored eval result "
                         "(e.g. attain_baseline_v2_results.json) instead of "
                         "re-deriving them, and run Stage 2 only")
    args = ap.parse_args()

    thresholds = [float(t) for t in args.thresholds.split(",")]

    os.environ.setdefault("PROMPTS_VERSION", "v2")
    os.environ["QUANTIZATION_BITS"] = str(args.quantization_bits)
    os.environ["DISABLE_ADAPTER"] = "true"

    from app.model import PavementClassifier
    from scripts.utils import CONFIDENCE_THRESHOLD, EVAL_DIR

    spaces = {"attain": attain_label_space(), "rdd_india": rdd_label_space()}
    print(f"[label space] attain    ({len(spaces['attain'])}): "
          f"{sorted(spaces['attain'])}")
    print(f"[label space] rdd_india ({len(spaces['rdd_india'])}): "
          f"{sorted(spaces['rdd_india'])}")

    def _attain(k):
        if args.attain_from_run:
            return load_attain_from_run(args.attain_from_run, k, args.seed)
        return load_attain(k, args.seed, args.subset)

    if args.dataset == "attain":
        rows_in = _attain(args.n)
    elif args.dataset == "rdd_india":
        rows_in = load_rdd_india(args.n, args.seed)
    else:
        half = args.n // 2
        rows_in = _attain(half) + load_rdd_india(args.n - half, args.seed)
    print(f"[sample] {len(rows_in)} labelled images "
          f"(seed {args.seed}, random order, not stratified)")
    for ds in ("attain", "rdd_india"):
        k = sum(1 for r in rows_in if r["dataset"] == ds)
        if k:
            print(f"           {ds}: {k}")

    ckpt = EVAL_DIR / (Path(args.out).stem + "_checkpoint.json")
    done: dict = {}
    if args.fresh:
        # Explicitly discarding prior work: say how much, so a --fresh left in
        # a restart command cannot silently throw away hours of GPU time.
        prior = load_checkpoint(ckpt)
        n_prior = len(prior.get("rows", {})) if prior else 0
        if n_prior:
            print(f"[fresh] DISCARDING {n_prior} completed images in "
                  f"{ckpt.name} because --fresh was passed")
    else:
        blob = load_checkpoint(ckpt)
        if blob:
            if blob.get("_model") not in (None, args.model) or \
               blob.get("_quant") not in (None, args.quantization_bits):
                sys.exit(f"ERROR: {ckpt.name} was written by model "
                         f"{blob.get('_model')!r} at {blob.get('_quant')}-bit. "
                         f"Use --fresh or a different --out.")
            done = blob.get("rows", {})
            print(f"[resume] {len(done)} images already done - "
                  f"re-running only what is missing")

    clf = PavementClassifier(model_path=args.model, adapter_path=None,
                             quantization_bits=args.quantization_bits)

    def save():
        save_checkpoint(ckpt, {"_model": args.model,
                               "_quant": args.quantization_bits,
                               "rows": done})

    t0 = time.time()
    for i, e in enumerate(rows_in, 1):
        key = f"{e['dataset']}::{e['image']}"
        if key in done:
            continue
        try:
            img = Image.open(e["image_path"]).convert("RGB")
            rec = dict(e)
            if args.skip_stage1 or e.get("stage1_from_stored_run"):
                # Stage 1 either already ran (verdict on disk) or is being
                # bypassed deliberately. Either way, go straight to Stage 2.
                s1 = {"is_distressed": True, "stage1_confidence": None,
                      "stage1_label": "Distress (assumed)"}
                rec["stage1_skipped"] = True
            else:
                s1 = clf.predict_stage1(img)
                rec["stage1_skipped"] = False
            rec["stage1_confidence"] = s1["stage1_confidence"]
            rec["stage1_label"] = s1["stage1_label"]
            if not s1["is_distressed"]:
                # Stage 2 never runs, so there is no Stage 2 confidence to
                # calibrate. Recorded for the Stage 1 miss rate, excluded from
                # the AUC population.
                rec["ran_stage2"] = False
                done[key] = rec
                print(f"[{i}/{len(rows_in)}] {e['dataset'][:4]} {e['image'][:28]:<30} "
                      f"Stage 1 said Normal (gt={e['gt_irc']})")
                save()
                continue

            s2 = clf.predict_stage2(img)
            ds = e["dataset"]
            pred_raw = s2["distress_types"]
            pred_irc = _irc(pred_raw)
            pred_native = map_pred_to_dataset_space(pred_raw, ds)
            gt_native = set(e["gt_native"])

            # A prediction that names ONLY types this dataset does not
            # annotate cannot be judged right or wrong by it. Scoring it either
            # way would invent a result, so it is recorded and excluded from
            # the AUC population. Predicting nothing at all is different: that
            # is a real miss and stays in.
            adjudicable = bool(pred_native) or not pred_irc

            # Primary label: the first one, which the prompt asks to be the
            # dominant distress. Scored on its own, in the dataset vocabulary.
            primary = s2.get("primary_distress_type")
            primary_native = (map_pred_to_dataset_space([primary], ds)
                              if primary else set())
            dom = attain_dominant_class(e["image_path"]) if ds == "attain" else {}
            rec.update({
                "primary": s2.get("stage2_confidence_primary"),
                "primary_span_found": s2.get("stage2_primary_span_found"),
                "primary_label": primary,
                "primary_native": sorted(primary_native),
                # None = the dataset does not annotate that type, so it can be
                # neither right nor wrong (same rule as the multi-label score).
                "primary_correct": (bool(primary_native & gt_native)
                                    if primary_native else None),
                "type_confidences": s2.get("stage2_type_confidences"),
                **dom,
                "primary_is_dominant": (dom["gt_dominant"] in primary_native
                                        if dom and primary_native else None),
            })
            rec.update({
                "ran_stage2": True,
                "pred_raw": pred_raw,
                "pred_irc": sorted(pred_irc),
                "pred_native": sorted(pred_native),
                # Types the model named that this dataset does not annotate.
                "pred_unadjudicable": sorted(
                    x for x in pred_raw
                    if not map_pred_to_dataset_space([x], ds)),
                "adjudicable": adjudicable,
                "severity": s2["severity"],
                "field": s2["stage2_confidence_field"],
                "sequence": s2["stage2_confidence_sequence"],
                "field_span_found": s2["stage2_field_span_found"],
                "correct": correctness(pred_native, gt_native),
                "stage2_time_ms": s2["stage2_time_ms"],
            })
            # Greedy decoding is deterministic, so a prediction that differs
            # from the stored run means the model, prompts or processor changed
            # and the two runs must not be pooled.
            if e.get("stored_pred_attain") is not None:
                rec["reproduced_stored_pred"] = (
                    sorted(pred_native) == sorted(e["stored_pred_attain"]))

            done[key] = rec
            mark = ("--" if not adjudicable
                    else ("OK " if rec["correct"][PRIMARY] else "ERR"))
            print(f"[{i}/{len(rows_in)}] {ds[:4]} {e['image'][:26]:<28} "
                  f"{mark} f={rec['field']:.3f} s={rec['sequence']:.3f} "
                  f"pred={sorted(pred_native)} gt={sorted(gt_native)}")
        except Exception as ex:
            done[key] = {**e, "error": f"{type(ex).__name__}: {ex}"}
            print(f"[{i}/{len(rows_in)}] ERROR {ex}")
        save()
        # Release cached GPU blocks between images. Attain mixes three frame
        # shapes; without this the allocator's reserve grew to the full 24 GB
        # and Windows paged the excess to system RAM: 640x640 frames took
        # ~22 s and 1920x1080 ~84 s against ~8 s for 1479x508, with identical
        # outputs. Timing only - predictions are unaffected.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    stage2_rows = [r for r in done.values() if r.get("ran_stage2")]
    # Rows the dataset cannot adjudicate carry no correctness signal and must
    # not be counted as either right or wrong.
    rows = [r for r in stage2_rows if r.get("adjudicable")]
    if not rows:
        print("No Stage 2 predictions collected.")
        return 1

    report = {
        "model": args.model,
        "quantization_bits": args.quantization_bits,
        "prompts_version": os.environ.get("PROMPTS_VERSION"),
        "dataset": args.dataset,
        "seed": args.seed,
        "sampling": "uniform random without replacement, not stratified "
                    "(AUC must reflect the operating distribution)",
        "scoring_note": "predictions are intersected with each dataset's "
                        "annotated label space before comparison; labels "
                        "outside it are recorded as pred_unevaluable and are "
                        "neither credited nor penalised",
        "primary_definition": PRIMARY,
        "live_threshold": CONFIDENCE_THRESHOLD,
        "n_sampled": len(rows_in),
        "n_attempted": len(done),
        "n_with_stage2": len(stage2_rows),
        "n_adjudicable": len(rows),
        "n_not_adjudicable": len(stage2_rows) - len(rows),
        "n_stage1_normal": sum(1 for r in done.values()
                               if r.get("ran_stage2") is False),
        "n_errors": sum(1 for r in done.values() if r.get("error")),
        "stage1_skipped": bool(args.skip_stage1),
        "attain_from_run": args.attain_from_run,
        "reproduced_stored_pred": {
            "checked": sum(1 for r in stage2_rows
                           if r.get("reproduced_stored_pred") is not None),
            "matched": sum(1 for r in stage2_rows
                           if r.get("reproduced_stored_pred") is True),
        },
        "field_span_found_rate": round(
            sum(1 for r in stage2_rows if r.get("field_span_found"))
            / max(len(stage2_rows), 1), 4),
        "seconds_total": round(elapsed, 1),
        "seconds_per_image": round(elapsed / max(len(rows), 1), 1),
        "metrics": {},
        "per_image": list(done.values()),
    }

    for scope, subset_rows in [("all", rows)] + [
            (ds, [r for r in rows if r["dataset"] == ds])
            for ds in ("attain", "rdd_india")
            if any(r["dataset"] == ds for r in rows)]:
        report["metrics"][scope] = {
            defn: {m: summarize(subset_rows, m, defn, thresholds)
                   for m in ("field", "sequence")}
            for defn in DEFINITIONS
        }
        # The primary-label confidence judged against the primary label only.
        # Rows whose primary type the dataset does not annotate are excluded.
        prim_rows = [dict(r, correct={**r["correct"],
                                      "primary_correct": r["primary_correct"]})
                     for r in subset_rows
                     if r.get("primary_correct") is not None
                     and r.get("primary") is not None]
        if prim_rows:
            report["metrics"][scope]["primary_correct"] = {
                m: summarize(prim_rows, m, "primary_correct", thresholds)
                for m in ("primary", "field")}

    out = EVAL_DIR / args.out
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # ---- console report ----
    L = "=" * 78
    print("\n" + L)
    print(f"CONFIDENCE CALIBRATION  {args.model}  {args.quantization_bits}-bit")
    print(f"{report['n_with_stage2']} Stage 2 predictions, "
          f"{report['n_adjudicable']} adjudicable "
          f"({report['n_not_adjudicable']} named only out-of-vocabulary types, "
          f"{report['n_stage1_normal']} filtered by Stage 1, "
          f"{report['n_errors']} errors) in {elapsed/3600:.2f} h")
    print(L)
    for defn in DEFINITIONS:
        m = report["metrics"]["all"][defn]["field"]
        s = report["metrics"]["all"][defn]["sequence"]
        star = "  <-- PRIMARY" if defn == PRIMARY else ""
        print(f"\n{defn}: {m['n_correct']} correct / {m['n_wrong']} wrong{star}")
        if m["n_wrong"] == 0 or m["n_correct"] == 0:
            print("   AUC undefined - only one class present")
            continue
        print(f"   {'':<10}{'AUC':>8}{'+/-':>8}{'mean ok':>10}{'mean bad':>10}{'sep':>9}")
        for name, x in (("field", m), ("sequence", s)):
            print(f"   {name:<10}{x['auc']:>8.4f}{x['auc_stderr']:>8.4f}"
                  f"{x['mean_correct']:>10.4f}{x['mean_wrong']:>10.4f}"
                  f"{x['separation']:>+9.4f}")
    print("\n" + L)
    print(f"THRESHOLD TABLE - field metric, {PRIMARY}, all datasets")
    print(L)
    pm = report["metrics"]["all"][PRIMARY]["field"]
    print(f"{'thresh':>7}{'accepted':>11}{'wrong through':>15}{'err rate':>10}"
          f"{'wrong caught':>14}{'review load':>13}")
    for th in thresholds:
        t = pm["thresholds"][f"{th:.2f}"]
        print(f"{th:>7.2f}{t['auto_accepted']:>7} /{len(rows):<3}"
              f"{t['wrong_waved_through']:>15}"
              f"{(str(t['auto_accept_error_rate_pct']) + '%'):>10}"
              f"{(str(t['wrong_caught']) + '/' + str(pm['n_wrong'])):>14}"
              f"{(str(t['review_load_pct']) + '%'):>13}")
    print(L)
    print(f"Written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
