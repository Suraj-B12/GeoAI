"""
How far can input resolution drop before the pipeline's answers change?

What actually costs time
------------------------
JPEG quality does not matter to the worker: the image is decoded back to
pixels before the model sees it, so a heavily compressed JPEG costs exactly as
much inference as a pristine one. What matters is RESOLUTION. Qwen2.5-VL turns
every 28x28 pixel block into one image token, and attention cost grows with the
square of the sequence length. A 1728x1728 phone photo is ~3,800 image tokens;
capped at 1024x1024 it is ~1,340.

The production knob is MAX_IMAGE_PIXELS. This script reproduces its effect
exactly: every downscaled view is produced with the processor's own
`smart_resize` (same factor-28 rounding, same pixel budget) and PIL bicubic,
the filter the Qwen processor uses. Because the result is already a multiple of
28 and inside the processor's cap, the processor passes it through unchanged.
One model load serves every resolution.

Two datasets, two different questions
-------------------------------------
  bengaluru  The real RoadSide uploads (1512-1728px handheld close-ups). No
             expert labels exist, so this measures CHANGE: does the answer at a
             lower resolution match the answer at full resolution?
  attain     Labelled. Measures LOSS: when the answer changes, does it get
             worse? Attain frames come in three sizes - 1484x504, 644x644 and
             1932x1092 - so the 1280 and 1024 caps change only the 16 largest
             of the 200 sampled. Attain's evidence is strongest at 768 and
             below, where most or all frames are downscaled.
             Its full-resolution arm is reused from calib_attain_407.json
             (same code, 4-bit, same prompts) instead of being paid for again,
             with a small re-run to prove it still reproduces.

Pass criteria - fixed BEFORE the run, so the conclusion cannot be fitted to it
--------------------------------------------------------------------------
A resolution cap is acceptable only if, compared with full resolution:

  1. SAFETY  (bengaluru) Distress -> Normal flips at Stage 1 on at most 2% of
             images. A downscaled crack that disappears is a damaged road
             classified as fine - the one error the pipeline must not make.
  2. STAGE 1 (bengaluru) Stage 1 label agreement >= 95%.
  3. STAGE 2 (bengaluru) mean Jaccard of the distress-type set vs full
             resolution >= 0.80, over images full resolution calls Distress.
  4. ACCURACY (attain, where the cap changes the image) no statistically
             significant drop in accuracy (paired McNemar p >= 0.05) under
             BOTH no_false_positives and jaccard_50, and a point-estimate drop
             of at most 2 percentage points.

The recommended cap is the smallest one that passes every criterion that
applies to it.

Control arms (added after the first four images, and why)
---------------------------------------------------------
Criterion 3 treats the full-resolution Stage 2 answer as a stable reference.
The first images showed the second distress label changing even at mild caps
(e.g. "Edge Breaking + Potholes" at full resolution, "Bleeding + Potholes" at
1280). That is either lost detail, or a second label so low-confidence that ANY
perturbation flips it - and the two call for opposite conclusions. The only
way to tell them apart is a perturbation that removes no information:

  1024_q95  the 1024 view re-encoded as JPEG quality 95. Same resolution,
            visually identical. Disagreement with plain 1024 is the model's
            own instability - the noise floor every cap has to be read
            against.
  1024_q75  the same at quality 75: the realistic upload/bandwidth
            compression, so JPEG compression gets measured rather than
            assumed.

These arms are compared with the plain 1024 answer, not with full resolution.
Criterion 3 is NOT relaxed after the fact: it is reported as pre-registered,
next to the measured noise floor, so the reader can see both.

Usage:
    venv/Scripts/python.exe scripts/resolution_ab.py
    venv/Scripts/python.exe scripts/resolution_ab.py --analyse-only
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time
import urllib.request
import gc
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
from PIL import Image  # noqa: E402


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Reused rather than re-implemented: crash-safe checkpointing, the four
# correctness definitions, and per-dataset label-space mapping.
cc = _load("calib", "scripts/calibrate_confidence.py")

from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize  # noqa: E402

EVAL_DIR = PROJECT_ROOT / "eval_results"
CACHE = PROJECT_ROOT / "data" / "images" / "resolution_cache"   # gitignored

FACTOR = 28
PROD_MIN_PIXELS = int(os.environ.get("MIN_IMAGE_PIXELS", str(256 * 28)))
PROD_MAX_PIXELS = int(os.environ.get("MAX_IMAGE_PIXELS", str(2200 * 2200)))

LEVELS = [("native", None), ("1280", 1280 * 1280), ("1024", 1024 * 1024),
          ("768", 768 * 768), ("640", 640 * 640), ("512", 512 * 512)]

# (name, pixel cap, JPEG quality, compared against)
CONTROL_ARMS = [("1024_q95", 1024 * 1024, 95, "1024"),
                ("1024_q75", 1024 * 1024, 75, "1024")]

CRIT_FLIP_PCT = 2.0
CRIT_S1_AGREE = 95.0
CRIT_S2_JACCARD = 0.80
CRIT_ACC_DROP_PP = 2.0
CRIT_MCNEMAR_P = 0.05


# ============================================================
# Resolution
# ============================================================

def target_dims(w: int, h: int, cap) -> tuple:
    """(width, height) the model sees under a MAX_IMAGE_PIXELS of `cap`."""
    budget = PROD_MAX_PIXELS if cap is None else min(cap, PROD_MAX_PIXELS)
    th, tw = smart_resize(h, w, factor=FACTOR,
                          min_pixels=PROD_MIN_PIXELS, max_pixels=budget)
    return tw, th


def render(img: Image.Image, cap, jpeg_quality=None):
    """The view the model receives at this cap. Full resolution is returned
    untouched so the processor applies production's own resize, exactly as
    the worker does. With jpeg_quality, the view is additionally round-tripped
    through a JPEG encode at that quality - the only change a control arm
    makes."""
    view = img if cap is None else img.resize(target_dims(*img.size, cap), Image.BICUBIC)
    if jpeg_quality is not None:
        import io
        buf = io.BytesIO()
        view.save(buf, format="JPEG", quality=jpeg_quality)
        buf.seek(0)
        view = Image.open(buf)
        view.load()
        view = view.convert("RGB")
    return view


# ============================================================
# Sources
# ============================================================

def load_bengaluru(n: int, seed: int) -> list:
    """Uploads that passed Stage 0 in the full-resolution production run."""
    d = json.loads((EVAL_DIR / "uploaded_photos_confidence.json").read_text(encoding="utf-8"))
    rows = [r for r in d["per_image"] if r.get("ok") and r.get("is_pavement")]
    random.Random(seed).shuffle(rows)
    rows = rows[:n] if n else rows
    return [{"dataset": "bengaluru", "image": r["id"], "image_url": r["image_url"]}
            for r in rows]


def load_attain(n: int, seed: int) -> list:
    """Attain images with ground truth and a stored full-resolution answer."""
    d = json.loads((EVAL_DIR / "calib_attain_407.json").read_text(encoding="utf-8"))
    rows = [r for r in d["per_image"] if r.get("ran_stage2") and r.get("adjudicable")]
    random.Random(seed).shuffle(rows)
    rows = rows[:n] if n else rows
    return [{"dataset": "attain", "image": r["image"], "image_path": r["image_path"],
             "gt_native": r["gt_native"], "stored_native": r} for r in rows]


def get_image(entry: dict) -> Image.Image:
    if entry["dataset"] == "attain":
        img = Image.open(entry["image_path"])
    else:
        CACHE.mkdir(parents=True, exist_ok=True)
        path = CACHE / f"{entry['image']}.jpg"
        if not path.exists():
            with urllib.request.urlopen(entry["image_url"], timeout=60) as r:
                path.write_bytes(r.read())
        img = Image.open(path)
    img.load()
    return img.convert("RGB")


# ============================================================
# Statistics
# ============================================================

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar test on discordant pairs. b = right at full
    resolution and wrong downscaled; c = the reverse."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


# ============================================================
# Analysis
# ============================================================

def _irc_set(rec: dict) -> set:
    return cc._irc(rec.get("distress_types") or [])


def analyse(done: dict, levels: list) -> dict:
    by = {}
    for rec in done.values():
        if rec.get("error"):
            continue
        by.setdefault((rec["dataset"], rec["image"]), {})[rec["level"]] = rec

    def resolve(recs: dict, level: str):
        r = recs.get(level)
        if r and r.get("duplicate_of"):
            return recs.get(r["duplicate_of"])
        return r

    out = {"bengaluru": {}, "attain": {}}

    # ---------- bengaluru: change vs full resolution ----------
    b_imgs = {k[1]: v for k, v in by.items() if k[0] == "bengaluru" and "native" in v}
    for level, _cap in levels:
        pairs = [(v["native"], resolve(v, level)) for v in b_imgs.values()]
        pairs = [(n_, l_) for n_, l_ in pairs if n_ and l_]
        if not pairs:
            continue
        s1_agree = sum(1 for n_, l_ in pairs if n_["stage1_label"] == l_["stage1_label"])
        d2n = sum(1 for n_, l_ in pairs if n_["is_distressed"] and not l_["is_distressed"])
        n2d = sum(1 for n_, l_ in pairs if not n_["is_distressed"] and l_["is_distressed"])
        dist = [(n_, l_) for n_, l_ in pairs if n_["is_distressed"]]
        jac = [jaccard(_irc_set(n_), _irc_set(l_)) for n_, l_ in dist]
        exact = sum(1 for n_, l_ in dist if _irc_set(n_) == _irc_set(l_))
        sev = sum(1 for n_, l_ in dist if n_.get("severity") == l_.get("severity"))
        t_nat = sum(n_["stage1_time_ms"] + n_["stage2_time_ms"] for n_, _ in pairs)
        t_lvl = sum(l_["stage1_time_ms"] + l_["stage2_time_ms"] for _, l_ in pairs)
        n = len(pairs)
        out["bengaluru"][level] = {
            "n": n,
            "mean_vision_tokens": round(sum(l_["vision_tokens"] for _, l_ in pairs) / n),
            "mean_seconds": round(t_lvl / n / 1000, 1),
            "speedup": round(t_nat / t_lvl, 2) if t_lvl else None,
            "stage1_agree_pct": round(100 * s1_agree / n, 1),
            "distress_to_normal": d2n,
            "distress_to_normal_pct": round(100 * d2n / n, 1),
            "normal_to_distress": n2d,
            "stage2_n": len(dist),
            "stage2_exact_pct": round(100 * exact / len(dist), 1) if dist else None,
            "stage2_mean_jaccard": round(sum(jac) / len(jac), 3) if jac else None,
            "severity_agree_pct": round(100 * sev / len(dist), 1) if dist else None,
            "mean_field_conf": round(sum(l_["field"] for _, l_ in dist) / len(dist), 4) if dist else None,
            "mean_field_conf_native": round(sum(n_["field"] for n_, _ in dist) / len(dist), 4) if dist else None,
        }

    # ---------- attain: loss vs ground truth, paired ----------
    a_imgs = {k[1]: v for k, v in by.items() if k[0] == "attain" and "native" in v}
    for level, _cap in levels:
        pairs = [(v["native"], resolve(v, level)) for v in a_imgs.values()]
        pairs = [(n_, l_) for n_, l_ in pairs
                 if n_ and l_ and n_.get("adjudicable") and l_.get("adjudicable")]
        if not pairs:
            continue
        changed = any(l_.get("level") != "native" and not l_.get("duplicate_of")
                      for _, l_ in pairs)
        res = {"n": len(pairs), "image_changed": changed,
               "mean_vision_tokens": round(sum(l_["vision_tokens"] for _, l_ in pairs) / len(pairs)),
               "set_agree_pct": round(100 * sum(1 for n_, l_ in pairs
                                               if set(n_["pred_native"]) == set(l_["pred_native"]))
                                      / len(pairs), 1)}
        for defn in ("no_false_positives", "jaccard_50", "any_overlap"):
            nat_ok = sum(1 for n_, _ in pairs if n_["correct"][defn])
            lvl_ok = sum(1 for _, l_ in pairs if l_["correct"][defn])
            b = sum(1 for n_, l_ in pairs if n_["correct"][defn] and not l_["correct"][defn])
            c = sum(1 for n_, l_ in pairs if not n_["correct"][defn] and l_["correct"][defn])
            res[defn] = {
                "acc_native_pct": round(100 * nat_ok / len(pairs), 1),
                "acc_level_pct": round(100 * lvl_ok / len(pairs), 1),
                "drop_pp": round(100 * (nat_ok - lvl_ok) / len(pairs), 1),
                "right_to_wrong": b, "wrong_to_right": c,
                "mcnemar_p": round(mcnemar_p(b, c), 4),
            }
        out["attain"][level] = res

    # ---------- verdicts against the pre-registered criteria ----------
    verdicts = {}
    for level, _cap in levels:
        if level == "native":
            continue
        reasons, applies = [], False
        b = out["bengaluru"].get(level)
        if b:
            applies = True
            if b["distress_to_normal_pct"] > CRIT_FLIP_PCT:
                reasons.append(f"Distress->Normal flips {b['distress_to_normal_pct']}% > {CRIT_FLIP_PCT}%")
            if b["stage1_agree_pct"] < CRIT_S1_AGREE:
                reasons.append(f"Stage 1 agreement {b['stage1_agree_pct']}% < {CRIT_S1_AGREE}%")
            if b["stage2_mean_jaccard"] is not None and b["stage2_mean_jaccard"] < CRIT_S2_JACCARD:
                reasons.append(f"Stage 2 Jaccard {b['stage2_mean_jaccard']} < {CRIT_S2_JACCARD}")
        a = out["attain"].get(level)
        if a and a["image_changed"]:
            applies = True
            for defn in ("no_false_positives", "jaccard_50"):
                x = a[defn]
                if x["mcnemar_p"] < CRIT_MCNEMAR_P and x["drop_pp"] > 0:
                    reasons.append(f"Attain {defn} significant drop "
                                   f"{x['drop_pp']}pp (p={x['mcnemar_p']})")
                elif x["drop_pp"] > CRIT_ACC_DROP_PP:
                    reasons.append(f"Attain {defn} drop {x['drop_pp']}pp > {CRIT_ACC_DROP_PP}pp")
        verdicts[level] = {"evaluated": applies, "pass": applies and not reasons,
                           "reasons": reasons}
    out["verdicts"] = verdicts
    passing = [lv for lv, _ in levels if lv != "native" and verdicts[lv]["pass"]]
    out["recommended_cap"] = passing[-1] if passing else None
    return out


def analyse_arms(done: dict) -> dict:
    """Compare each control arm with its base arm, same resolution."""
    by = {}
    for rec in done.values():
        if rec.get("error"):
            continue
        by.setdefault((rec["dataset"], rec["image"]), {})[rec["level"]] = rec

    def resolve(recs, level):
        r = recs.get(level)
        if r and r.get("duplicate_of"):
            return recs.get(r["duplicate_of"])
        return r

    out = {}
    for name, _cap, q, base in CONTROL_ARMS:
        res = {"jpeg_quality": q, "compared_with": base}
        # bengaluru: change relative to the same-resolution base
        pairs = [(resolve(v, base), v.get(name)) for k, v in by.items()
                 if k[0] == "bengaluru"]
        pairs = [(b_, a_) for b_, a_ in pairs if b_ and a_]
        if pairs:
            n = len(pairs)
            dist = [(b_, a_) for b_, a_ in pairs if b_["is_distressed"]]
            jac = [jaccard(_irc_set(b_), _irc_set(a_)) for b_, a_ in dist]
            res["bengaluru"] = {
                "n": n,
                "stage1_agree_pct": round(100 * sum(1 for b_, a_ in pairs
                                                    if b_["stage1_label"] == a_["stage1_label"]) / n, 1),
                "distress_to_normal": sum(1 for b_, a_ in pairs
                                          if b_["is_distressed"] and not a_["is_distressed"]),
                "stage2_n": len(dist),
                "stage2_exact_pct": round(100 * sum(1 for b_, a_ in dist
                                                    if _irc_set(b_) == _irc_set(a_)) / len(dist), 1) if dist else None,
                "stage2_mean_jaccard": round(sum(jac) / len(jac), 3) if jac else None,
            }
        # attain: paired accuracy against the same-resolution base
        pairs = [(resolve(v, base), v.get(name)) for k, v in by.items()
                 if k[0] == "attain"]
        pairs = [(b_, a_) for b_, a_ in pairs
                 if b_ and a_ and b_.get("adjudicable") and a_.get("adjudicable")]
        if pairs:
            att = {"n": len(pairs)}
            for defn in ("no_false_positives", "jaccard_50"):
                bb = sum(1 for b_, a_ in pairs if b_["correct"][defn] and not a_["correct"][defn])
                cc_ = sum(1 for b_, a_ in pairs if not b_["correct"][defn] and a_["correct"][defn])
                att[defn] = {
                    "acc_base_pct": round(100 * sum(1 for b_, _ in pairs if b_["correct"][defn]) / len(pairs), 1),
                    "acc_arm_pct": round(100 * sum(1 for _, a_ in pairs if a_["correct"][defn]) / len(pairs), 1),
                    "right_to_wrong": bb, "wrong_to_right": cc_,
                    "mcnemar_p": round(mcnemar_p(bb, cc_), 4),
                }
            res["attain"] = att
        out[name] = res
    return out


def analyse_detail(done: dict, levels: list) -> dict:
    """Two numbers the aggregate tables hide, both needed for the paper.

    per_class_recall  Attain, against ground truth: for each annotated class,
                      how often the model still names it at each cap. This is
                      the direct test of "does downscaling lose fine detail" -
                      hairline cracks are the first thing it would remove.
    stage1_flips      Bengaluru: every image whose Stage 1 label changes at
                      some cap, with its full-resolution confidence. A flip on
                      an image the model was already unsure of is not the same
                      as a flip on a confident one.
    """
    gt = {r["image"]: set(r["gt_native"]) for r in json.loads(
        (EVAL_DIR / "calib_attain_407.json").read_text(encoding="utf-8"))["per_image"]
        if r.get("ran_stage2")}
    by = {}
    for rec in done.values():
        if rec.get("error") or rec.get("jpeg_quality"):
            continue
        by.setdefault((rec["dataset"], rec["image"]), {})[rec["level"]] = rec

    def resolve(recs, level):
        r = recs.get(level)
        if r and r.get("duplicate_of"):
            return recs.get(r["duplicate_of"])
        return r

    names = [lv for lv, _ in levels]
    att = {k[1]: v for k, v in by.items() if k[0] == "attain" and k[1] in gt}
    counts = {}
    for img in att:
        for c in gt[img]:
            counts[c] = counts.get(c, 0) + 1
    recall = {}
    for cls, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        row = {"n": n}
        for lv in names:
            hit = tot = 0
            for img, v in att.items():
                r = resolve(v, lv)
                if r and cls in gt[img]:
                    tot += 1
                    hit += cls in (r.get("pred_native") or [])
            row[lv] = round(100 * hit / tot, 1) if tot else None
        recall[cls] = row

    flips = []
    for (ds, img), v in by.items():
        if ds != "bengaluru" or not v.get("native"):
            continue
        n_ = v["native"]
        for lv in names[1:]:
            l_ = resolve(v, lv)
            if l_ and l_ is not n_ and l_["is_distressed"] != n_["is_distressed"]:
                flips.append({"image": img, "cap": lv,
                              "direction": "D->N" if n_["is_distressed"] else "N->D",
                              "native_stage1_conf": n_["stage1_confidence"],
                              "capped_stage1_conf": l_["stage1_confidence"]})
    max_conf = max((f["native_stage1_conf"] for f in flips), default=None)
    return {
        "per_class_recall": recall,
        "stage1_flips": flips,
        "stage1_flip_summary": {
            "n_flips": len(flips),
            "n_distress_to_normal": sum(1 for f in flips if f["direction"] == "D->N"),
            "images_involved": len({f["image"] for f in flips}),
            "max_native_stage1_conf": max_conf,
            # Every flipped image below the review threshold at full resolution
            # means capping only moves answers that already went to a human.
            "all_below_threshold": (max_conf is not None and max_conf < 0.80),
        },
    }


def print_report(a: dict) -> None:
    L = "=" * 96
    print("\n" + L + "\nBENGALURU - change vs full resolution (57 real uploads, no labels)\n" + L)
    print(f"{'cap':>7}{'tokens':>8}{'s/img':>7}{'speedup':>9}{'S1 agree':>10}"
          f"{'D->N':>7}{'N->D':>6}{'S2 exact':>10}{'S2 Jacc':>9}{'sev agree':>11}{'field':>8}")
    for lv, b in a["bengaluru"].items():
        print(f"{lv:>7}{b['mean_vision_tokens']:>8}{b['mean_seconds']:>7}"
              f"{str(b['speedup']) + 'x':>9}{str(b['stage1_agree_pct']) + '%':>10}"
              f"{b['distress_to_normal']:>7}{b['normal_to_distress']:>6}"
              f"{str(b['stage2_exact_pct']) + '%':>10}{str(b['stage2_mean_jaccard']):>9}"
              f"{str(b['severity_agree_pct']) + '%':>11}{str(b['mean_field_conf']):>8}")
    print("\n" + L + "\nATTAIN - accuracy vs ground truth, paired with full resolution\n" + L)
    print(f"{'cap':>7}{'n':>5}{'tokens':>8}{'set agree':>11}   "
          f"{'no_false_positives':>30}   {'jaccard_50':>30}")
    for lv, x in a["attain"].items():
        cols = []
        for defn in ("no_false_positives", "jaccard_50"):
            d = x[defn]
            cols.append(f"{d['acc_native_pct']}%->{d['acc_level_pct']}% "
                        f"(-{d['right_to_wrong']}/+{d['wrong_to_right']}, p={d['mcnemar_p']})")
        tag = "" if x["image_changed"] else "  [cap does not touch Attain]"
        print(f"{lv:>7}{x['n']:>5}{x['mean_vision_tokens']:>8}{str(x['set_agree_pct']) + '%':>11}"
              f"   {cols[0]:>30}   {cols[1]:>30}{tag}")
    print("\n" + L + "\nVERDICT against the pre-registered criteria\n" + L)
    for lv, v in a["verdicts"].items():
        state = "PASS" if v["pass"] else ("not evaluated" if not v["evaluated"] else "FAIL")
        print(f"{lv:>7}  {state:<14}{'; '.join(v['reasons'])}")
    arms = a.get("control_arms") or {}
    if arms:
        print("\n" + L + "\nCONTROL ARMS - same resolution, JPEG round-trip only (noise floor)\n" + L)
        for name, r in arms.items():
            b = r.get("bengaluru")
            if b:
                print(f"{name:>9} vs {r['compared_with']:<5} bengaluru n={b['n']}: "
                      f"S1 agree {b['stage1_agree_pct']}%  D->N {b['distress_to_normal']}  "
                      f"S2 exact {b['stage2_exact_pct']}%  S2 Jaccard {b['stage2_mean_jaccard']}")
            t = r.get("attain")
            if t:
                x, y = t["no_false_positives"], t["jaccard_50"]
                print(f"{'':>9}    {'':<5} attain    n={t['n']}: "
                      f"nfp {x['acc_base_pct']}%->{x['acc_arm_pct']}% (p={x['mcnemar_p']})  "
                      f"j50 {y['acc_base_pct']}%->{y['acc_arm_pct']}% (p={y['mcnemar_p']})")
    det = a.get("detail") or {}
    if det.get("per_class_recall"):
        print("\n" + L + "\nATTAIN per-class recall vs ground truth (does fine detail survive?)\n" + L)
        lvls = [k for k in next(iter(det["per_class_recall"].values())) if k != "n"]
        print(f"{'class':<28}{'n':>5}" + "".join(f"{lv:>8}" for lv in lvls))
        for cls, row in det["per_class_recall"].items():
            print(f"{cls:<28}{row['n']:>5}" + "".join(
                f"{(str(row[lv]) + '%') if row[lv] is not None else '-':>8}" for lv in lvls))
    fs = det.get("stage1_flip_summary")
    if fs:
        print(f"\nStage 1 flips: {fs['n_flips']} across {fs['images_involved']} images, "
              f"{fs['n_distress_to_normal']} Distress->Normal; highest full-resolution "
              f"confidence among them {fs['max_native_stage1_conf']} "
              f"(all below 0.80: {fs['all_below_threshold']})")
    print(f"\nRecommended MAX_IMAGE_PIXELS cap: {a['recommended_cap'] or 'none - keep full resolution'}")


# ============================================================
# Main
# ============================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="bengaluru,attain")
    ap.add_argument("--n-bengaluru", type=int, default=0, help="0 = all")
    ap.add_argument("--n-attain", type=int, default=200)
    ap.add_argument("--recheck-native", type=int, default=10,
                    help="re-run this many reused Attain full-resolution rows to "
                         "prove the stored answers still reproduce")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quantization-bits", type=int, default=4, choices=[0, 4, 8])
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--out", default="resolution_ab.json")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--control-arms", action="store_true",
                    help="also run the same-resolution JPEG control arms "
                         "(noise floor + realistic upload compression)")
    args = ap.parse_args()

    ckpt = EVAL_DIR / (Path(args.out).stem + "_checkpoint.json")
    done: dict = {}
    if not args.fresh:
        blob = cc.load_checkpoint(ckpt)
        if blob:
            if blob.get("_quant") not in (None, args.quantization_bits):
                sys.exit(f"ERROR: {ckpt.name} was written at {blob.get('_quant')}-bit")
            done = blob.get("rows", {})
            print(f"[resume] {len(done)} rows already done")

    wanted = {d.strip() for d in args.datasets.split(",")}
    entries = []
    if "bengaluru" in wanted:
        entries += load_bengaluru(args.n_bengaluru, args.seed)
    if "attain" in wanted:
        entries += load_attain(args.n_attain, args.seed)
    print(f"[sample] {sum(e['dataset'] == 'bengaluru' for e in entries)} bengaluru, "
          f"{sum(e['dataset'] == 'attain' for e in entries)} attain")

    def save():
        cc.save_checkpoint(ckpt, {"_model": args.model,
                                  "_quant": args.quantization_bits, "rows": done})

    if not args.analyse_only:
        os.environ.setdefault("PROMPTS_VERSION", "v2")
        os.environ["QUANTIZATION_BITS"] = str(args.quantization_bits)
        os.environ["DISABLE_ADAPTER"] = "true"
        from app.model import PavementClassifier
        clf = PavementClassifier(model_path=args.model, adapter_path=None,
                                 quantization_bits=args.quantization_bits)

        recheck = {e["image"] for e in entries if e["dataset"] == "attain"}
        recheck = set(sorted(recheck)[: args.recheck_native])

        t0 = time.time()
        for i, e in enumerate(entries, 1):
            ds, name = e["dataset"], e["image"]
            img = None
            native_dims = None
            arms = [(lv, cap, None) for lv, cap in LEVELS]
            if args.control_arms:
                arms += [(nm, cap, q) for nm, cap, q, _base in CONTROL_ARMS]
            for level, cap, jpeg_q in arms:
                key = f"{ds}::{name}::{level}"
                if key in done:
                    continue
                try:
                    if img is None:
                        img = get_image(e)
                        native_dims = target_dims(*img.size, None)
                    dims = target_dims(*img.size, cap)
                    base = {"dataset": ds, "image": name, "level": level,
                            "width": dims[0], "height": dims[1],
                            "vision_tokens": (dims[0] // FACTOR) * (dims[1] // FACTOR)}

                    if jpeg_q is None and level != "native" and dims == native_dims:
                        # The cap does not bind for this image: the model would
                        # see exactly the full-resolution input. Not re-run.
                        done[key] = {**base, "duplicate_of": "native"}
                        save()
                        continue

                    if ds == "attain" and level == "native" and name not in recheck:
                        s = e["stored_native"]
                        done[key] = {**base, "reused_from": "calib_attain_407.json",
                                     "distress_types": s["pred_raw"],
                                     "pred_native": s["pred_native"],
                                     "field": s["field"], "sequence": s["sequence"],
                                     "correct": s["correct"],
                                     "adjudicable": s["adjudicable"]}
                        save()
                        continue

                    view = render(img, cap, jpeg_q)
                    rec = dict(base)
                    if jpeg_q is not None:
                        rec["jpeg_quality"] = jpeg_q
                    if ds == "bengaluru":
                        s1 = clf.predict_stage1(view)
                        rec.update(stage1_label=s1["stage1_label"],
                                   stage1_confidence=s1["stage1_confidence"],
                                   is_distressed=s1["is_distressed"],
                                   stage1_time_ms=s1["stage1_time_ms"])
                    # Stage 2 runs at every level regardless of Stage 1, so a
                    # resolution that flips Stage 1 still yields a comparable
                    # Stage 2 answer. Production gating is applied in analysis.
                    s2 = clf.predict_stage2(view)
                    rec.update(distress_types=s2["distress_types"],
                               severity=s2["severity"],
                               field=s2["stage2_confidence_field"],
                               sequence=s2["stage2_confidence_sequence"],
                               field_span_found=s2["stage2_field_span_found"],
                               stage2_time_ms=s2["stage2_time_ms"])
                    if ds == "attain":
                        pred_irc = cc._irc(s2["distress_types"])
                        pred_native = cc.map_pred_to_dataset_space(s2["distress_types"], "attain")
                        rec.update(pred_native=sorted(pred_native),
                                   correct=cc.correctness(pred_native, set(e["gt_native"])),
                                   adjudicable=bool(pred_native) or not pred_irc)
                        if level == "native":
                            rec["reproduced_stored"] = (
                                sorted(pred_native) == sorted(e["stored_native"]["pred_native"]))
                    done[key] = rec
                    s1txt = (f"S1={rec['stage1_label'][:4]} " if ds == "bengaluru" else "")
                    print(f"[{i}/{len(entries)}] {ds[:4]} {str(name)[:14]:<15}{level:>7} "
                          f"{dims[0]}x{dims[1]} tok={rec['vision_tokens']:<5}"
                          f"{s1txt}S2={sorted(cc._irc(s2['distress_types']))} "
                          f"f={rec['field']:.3f} "
                          f"t={(rec.get('stage1_time_ms', 0) + rec['stage2_time_ms']) / 1000:.1f}s",
                          flush=True)
                except Exception as ex:
                    done[key] = {"dataset": ds, "image": name, "level": level,
                                 "error": f"{type(ex).__name__}: {ex}"}
                    print(f"[{i}/{len(entries)}] ERROR {ds} {name} {level}: {ex}", flush=True)
                save()
                # Release cached allocator blocks between caps. The caching
                # allocator otherwise keeps the full-resolution peak (~24 GB)
                # reserved, and on Windows WDDM that footprint can sit partly
                # in system RAM - so every smaller cap would be timed against
                # the full-resolution memory state and the speed comparison
                # would be contaminated. Accuracy is unaffected either way.
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        print(f"\ninference done in {(time.time() - t0) / 3600:.2f} h")

    analysis = analyse(done, LEVELS)
    analysis["control_arms"] = analyse_arms(done)
    analysis["detail"] = analyse_detail(done, LEVELS)
    rechecked = [r for r in done.values() if "reproduced_stored" in r]
    analysis["native_recheck"] = {
        "checked": len(rechecked),
        "reproduced": sum(1 for r in rechecked if r["reproduced_stored"]),
    }
    analysis["criteria"] = {
        "distress_to_normal_max_pct": CRIT_FLIP_PCT,
        "stage1_agree_min_pct": CRIT_S1_AGREE,
        "stage2_jaccard_min": CRIT_S2_JACCARD,
        "attain_drop_max_pp": CRIT_ACC_DROP_PP,
        "attain_mcnemar_alpha": CRIT_MCNEMAR_P,
    }
    report = {"model": args.model, "quantization_bits": args.quantization_bits,
              "production_max_pixels": PROD_MAX_PIXELS, "levels": LEVELS,
              "analysis": analysis, "per_row": list(done.values())}
    (EVAL_DIR / args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_report(analysis)
    rc = analysis["native_recheck"]
    if rc["checked"]:
        print(f"\nReused-baseline check: {rc['reproduced']}/{rc['checked']} "
              f"Attain full-resolution answers reproduced exactly")
    print(f"Written: {EVAL_DIR / args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
