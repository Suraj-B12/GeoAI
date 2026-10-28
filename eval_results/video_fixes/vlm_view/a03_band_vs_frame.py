"""VLM-view A/B, step 3 (CPU only): band vs frame on the same chosen frames.

Arms (same 328 segments, sample.json):
  F0  batch_001: whole frames, default profile, 27 Oct (the original answers);
  F1  ab_frame_001: whole frames again, same server/profile/day as the band arm (control: F1 vs F0 =
      run-to-run reproducibility);
  B   ab_band_001: bands.
Primary comparison: B vs F1. Writes band_vs_frame.json and flips.csv here, contact sheets in sheets/.

Clustering with a stratified sample: every F0-positive segment is in the sample, F0-Normal segments with
probability 164/649. Each segment gets weight 1/pi (Horvitz-Thompson), a pair of consecutive sampled
segments the product. Checked against the full batch for F0 (all 813 known). 95 % CIs: block bootstrap
(blocks of 10 consecutive segments per session, 2000 replicates, seed fixed).
"""
import os
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "run with CUDA_VISIBLE_DEVICES=-1"
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
assert not torch.cuda.is_available()

REPO = Path(r"C:/Users/PRO-LAB-3/Documents/GeoAI_Video")
sys.path.insert(0, str(REPO / "pipeline"))
from geoai.config import Config  # noqa: E402
from geoai.vlm import answer_class, read_answers  # noqa: E402

OUT = Path(r"C:/GeoAI/analysis/fixes/vlm_view")
BATCHES = REPO / "field/batches"
cfg = Config()
ALLOWED = set(cfg.vlm.labels)
sample = json.load(open(OUT / "sample.json"))
IDS = sorted(sample["positives"] + sample["normals"])
POS0 = set(sample["positives"])
W_NORMAL = sample["n_normal_pool"] / sample["n_normal_drawn"]
rng = np.random.default_rng(7)


def arm(name):
    a = read_answers(BATCHES / name, cfg)
    out = {}
    for rid, res in a.results.items():
        cls, dist = answer_class(res, ALLOWED)
        raw = res.get("raw") or {}
        out[rid] = {"cls": cls, "labels": dist, "stage1": res.get("stage1_label"), "s1conf": res.get("stage1_confidence"),
                    "seconds": res.get("seconds"), "severity": res.get("severity"), "desc": res.get("description"),
                    "safety_net": (raw.get("stage1_safety_net") or {}).get("flagged")}
    return a, out


A0, F0 = arm("batch_001")
A1, F1 = arm("ab_frame_001")
AB, B = arm("ab_band_001")
reqB = AB.requests
missing = [i for i in IDS if i not in F1 or i not in B]
assert not missing, f"unanswered: {len(missing)}"


def key(rid):
    sid, k = rid.rsplit("_seg", 1)
    return sid, int(k)


def pos(arm_, rid):
    return arm_[rid]["cls"] == "distress"


def table(a, b):
    """2x2 counts of (a positive, b positive) over IDS, plus unknown counts."""
    t = {"pp": 0, "pn": 0, "np": 0, "nn": 0, "unknown": 0}
    for rid in IDS:
        ca, cb = a[rid]["cls"], b[rid]["cls"]
        if "unknown" in (ca, cb):
            t["unknown"] += 1
            continue
        t[("p" if ca == "distress" else "n") + ("p" if cb == "distress" else "n")] += 1
    n = t["pp"] + t["pn"] + t["np"] + t["nn"]
    po = (t["pp"] + t["nn"]) / n
    pa, pb = (t["pp"] + t["pn"]) / n, (t["pp"] + t["np"]) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    t.update(n=n, agreement=po, kappa=(po - pe) / (1 - pe) if pe < 1 else None,
             label_set_equal=sum(1 for rid in IDS if a[rid]["labels"] == b[rid]["labels"] and a[rid]["cls"] == b[rid]["cls"]) / len(IDS))
    return t


def wilson(k, n, z=1.959964):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


# ---- weighted statistics: positive rate, clustering -------------------------------------------------
def weight(rid):
    return 1.0 if rid in POS0 else W_NORMAL


def stats(arm_, ids):
    """Weighted positive rate, P(pos | previous pos) and lag-1 correlation over consecutive sampled pairs."""
    idset = set(ids)
    w = {r: weight(r) for r in ids}
    p = {r: float(pos(arm_, r)) for r in ids}
    W = sum(w.values())
    base = sum(w[r] * p[r] for r in ids) / W
    num = den = 0.0
    xy = []
    for r in ids:
        sid, k = key(r)
        prev = f"{sid}_seg{k - 1:05d}"
        if prev in idset:
            wp = w[r] * w[prev]
            den += wp * p[prev]
            num += wp * p[prev] * p[r]
            xy.append((wp, p[prev], p[r]))
    cond = num / den if den else None
    if xy:
        ww = np.array([x[0] for x in xy]); a = np.array([x[1] for x in xy]); b = np.array([x[2] for x in xy])
        ma, mb = np.average(a, weights=ww), np.average(b, weights=ww)
        cov = np.average((a - ma) * (b - mb), weights=ww)
        sa, sb = math.sqrt(np.average((a - ma) ** 2, weights=ww)), math.sqrt(np.average((b - mb) ** 2, weights=ww))
        r1 = cov / (sa * sb) if sa > 0 and sb > 0 else None
    else:
        r1 = None
    return {"base": base, "p_pos_given_prev_pos": cond, "lift": cond / base if cond is not None and base else None,
            "lag1_r": r1}


blocks = {}
for r in IDS:
    sid, k = key(r)
    blocks.setdefault((sid, k // 10), []).append(r)
blist = list(blocks.values())


def stats_blocks(arm_, blks):
    """stats() over a bootstrap resample of blocks (each copy of a block is its own island)."""
    w_num = w_den = w_all = w_pos = 0.0
    for blk in blks:
        bs = set(blk)
        for r in blk:
            wr, pr = weight(r), float(pos(arm_, r))
            w_all += wr
            w_pos += wr * pr
            sid, k = key(r)
            prev = f"{sid}_seg{k - 1:05d}"
            if prev in bs:
                wp = wr * weight(prev)
                pp = float(pos(arm_, prev))
                w_den += wp * pp
                w_num += wp * pp * pr
    return w_pos / w_all, (w_num / w_den if w_den else float("nan"))


def boot_stats(arms_):
    """Bootstrap CIs of base, P(pos|prev pos) per arm and their differences (B - F1)."""
    res = {name: {"base": [], "cond": []} for name in arms_}
    diffs = {"base": [], "cond": []}
    for _ in range(2000):
        pick = rng.integers(0, len(blist), len(blist))
        blks = [blist[i] for i in pick]
        cur = {}
        for name, a in arms_.items():
            b_, c_ = stats_blocks(a, blks)
            res[name]["base"].append(b_)
            res[name]["cond"].append(c_)
            cur[name] = (b_, c_)
        diffs["base"].append(cur["band"][0] - cur["frame_rerun"][0])
        diffs["cond"].append(cur["band"][1] - cur["frame_rerun"][1])

    def ci(v):
        v = np.array(v)
        v = v[np.isfinite(v)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return ({n: {k: ci(v) for k, v in d.items()} for n, d in res.items()}, {k: ci(v) for k, v in diffs.items()})


# Full-batch reference for F0 (all 813 known): validates the weighting.
all0 = {}
for rid, res in A0.results.items():
    all0[rid] = answer_class(res, ALLOWED)[0] == "distress"
full = {}
for sid in sorted({key(r)[0] for r in all0}):
    ks = sorted(key(r)[1] for r in all0 if key(r)[0] == sid)
    seq = [all0[f"{sid}_seg{k:05d}"] for k in ks]
    pairs = [(seq[i - 1], seq[i]) for i in range(1, len(seq)) if ks[i] == ks[i - 1] + 1]
    full[sid] = (sum(seq), len(seq), sum(1 for a, b in pairs if a and b), sum(1 for a, b in pairs if a))
tot = [sum(v[i] for v in full.values()) for i in range(4)]
full_ref = {"base": tot[0] / tot[1], "p_pos_given_prev_pos": tot[2] / tot[3]}

arms = {"frame_b001": F0, "frame_rerun": F1, "band": B}
st = {n: stats(a, IDS) for n, a in arms.items()}
ci_arm, ci_diff = boot_stats({"frame_rerun": F1, "band": B, "frame_b001": F0})

# ---- strata --------------------------------------------------------------------------------------
strata = {}
for name, a in arms.items():
    kp = sum(1 for r in sample["positives"] if pos(a, r))
    kn = sum(1 for r in sample["normals"] if pos(a, r))
    strata[name] = {"pos_stratum_positive": kp, "pos_stratum_n": len(sample["positives"]),
                    "pos_stratum_share": kp / len(sample["positives"]), "pos_stratum_ci": wilson(kp, len(sample["positives"])),
                    "normal_stratum_positive": kn, "normal_stratum_n": len(sample["normals"]),
                    "normal_stratum_share": kn / len(sample["normals"]), "normal_stratum_ci": wilson(kn, len(sample["normals"]))}

# ---- tokens and seconds --------------------------------------------------------------------------
def timing(a_answers, arm_):
    sec = {"normal": [], "distress": []}
    for rid in IDS:
        s = arm_[rid]["seconds"]
        if s is not None:
            sec["distress" if arm_[rid]["stage1"] == "Distress" else "normal"].append(float(s))
    allsec = sec["normal"] + sec["distress"]
    tok = [(r["model_input"][0] // 28) * (r["model_input"][1] // 28) for r in a_answers.requests.values() if r.get("model_input")]
    return {"tokens_median": float(np.median(tok)) if tok else None, "tokens_range": [min(tok), max(tok)] if tok else None,
            "seconds_median": float(np.median(allsec)), "seconds_mean": float(np.mean(allsec)),
            "seconds_median_stage1_normal": float(np.median(sec["normal"])) if sec["normal"] else None,
            "seconds_median_stage1_distress": float(np.median(sec["distress"])) if sec["distress"] else None,
            "n_stage1_distress": len(sec["distress"]),
            "profile": [r.get("profile") for r in a_answers.run_info.get("runs", [])]}


timings = {"frame_b001": timing(A0, F0), "frame_rerun": timing(A1, F1), "band": timing(AB, B)}
# frame_b001 tokens: the whole frame -> 1344 x 756
timings["frame_b001"]["tokens_median"] = 48 * 27

# ---- flips ---------------------------------------------------------------------------------------
flips = []
for rid in IDS:
    cf, cb = F1[rid]["cls"], B[rid]["cls"]
    if cf != cb:
        flips.append({"id": rid, "stratum": "positive" if rid in POS0 else "normal", "frame_rerun": cf, "band": cb,
                      "frame_labels": "+".join(F1[rid]["labels"]), "band_labels": "+".join(B[rid]["labels"]),
                      "frame_b001": F0[rid]["cls"], "frame_b001_labels": "+".join(F0[rid]["labels"]),
                      "frame_desc": F1[rid]["desc"], "band_desc": B[rid]["desc"]})
with open(OUT / "flips.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(flips[0]) if flips else ["id"])
    w.writeheader()
    w.writerows(flips)

# ---- contact sheets (JPEG, for people only) ------------------------------------------------------
SHEETS = OUT / "sheets"
SHEETS.mkdir(exist_ok=True)


def sheet(fl):
    rid = fl["id"]
    rec = json.loads((BATCHES / "ab_band_001" / "records" / f"{rid}.json").read_text(encoding="utf-8"))
    frame = cv2.imread(str(BATCHES / "ab_frame_001" / "images" / f"{rid}.png"), cv2.IMREAD_COLOR)
    band = cv2.imread(str(BATCHES / "ab_band_001" / "images" / f"{rid}.png"), cv2.IMREAD_COLOR)
    W = 1600
    s = W / frame.shape[1]
    th = cv2.resize(frame, (W, round(frame.shape[0] * s)), interpolation=cv2.INTER_AREA)
    vt, vb = rec["frame"]["rows_of_segment"]
    y0, y1 = rec["vlm_input"]["box"][1], rec["vlm_input"]["box"][3]
    for v, col in ((vt, (0, 200, 0)), (min(vb, frame.shape[0] - 1), (0, 200, 0))):
        cv2.line(th, (0, round(v * s)), (W - 1, round(v * s)), col, 2)
    cv2.rectangle(th, (1, round(y0 * s)), (W - 2, round(y1 * s) - 1), (0, 140, 255), 2)
    bt = cv2.resize(band, (W, round(band.shape[0] * W / band.shape[1])), interpolation=cv2.INTER_AREA)
    sv = W / band.shape[1]
    rv = rec["vlm_input"]["rows_of_segment"]
    cv2.line(bt, (0, round(rv[0] * sv)), (W - 1, round(rv[0] * sv)), (0, 200, 0), 2)
    cap = 64
    out = np.full((cap + th.shape[0] + 8 + bt.shape[0], W, 3), 255, np.uint8)
    t1 = f"{rid[-15:]}  stratum {fl['stratum']}  FRAME: {fl['frame_rerun']} {fl['frame_labels']}  (27 Oct: {fl['frame_b001']} {fl['frame_b001_labels']})"
    t2 = f"BAND: {fl['band']} {fl['band_labels']}   green = segment far/near edge, orange = band"
    cv2.putText(out, t1, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(out, t2, (6, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 160), 2, cv2.LINE_AA)
    out[cap:cap + th.shape[0]] = th
    out[cap + th.shape[0] + 8:] = bt
    p = SHEETS / f"flip_{rid}.jpg"
    cv2.imwrite(str(p), out, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return p


for fl in flips:
    sheet(fl)

result = {
    "sample": {k: sample[k] for k in ("seed", "n_positive", "n_normal_pool", "n_normal_drawn", "n_unknown_excluded")},
    "agreement": {"band_vs_frame_rerun": table(F1, B), "frame_rerun_vs_frame_b001": table(F0, F1),
                  "band_vs_frame_b001": table(F0, B)},
    "strata": strata,
    "weighted": st, "weighted_ci95": ci_arm, "band_minus_frame_rerun_ci95": ci_diff,
    "full_batch_frame_b001_reference": full_ref,
    "timing": timings,
    "n_flips": len(flips),
    "flips_by_direction": {"frame_pos_band_normal": sum(1 for f in flips if f["frame_rerun"] == "distress" and f["band"] == "normal"),
                           "frame_normal_band_pos": sum(1 for f in flips if f["frame_rerun"] == "normal" and f["band"] == "distress"),
                           "involving_unknown": sum(1 for f in flips if "unknown" in (f["frame_rerun"], f["band"]))},
}
json.dump(result, open(OUT / "band_vs_frame.json", "w"), indent=1)
print(json.dumps({k: result[k] for k in ("agreement", "strata", "weighted", "band_minus_frame_rerun_ci95",
                                         "full_batch_frame_b001_reference", "timing", "flips_by_direction")}, indent=1))
