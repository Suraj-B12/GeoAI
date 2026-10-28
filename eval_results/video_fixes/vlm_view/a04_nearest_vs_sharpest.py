"""VLM-view A/B, step 4 (CPU only): segments.choice nearest_sharp vs sharpest on session d47c.

Usage: a04_nearest_vs_sharpest.py geometry   -> nearest_geometry.json, changed_ids.txt
       a04_nearest_vs_sharpest.py answers    -> nearest_vs_sharpest.json (after the VLM answered ab_nearest_001)

Sharpest arm: the send-all run of 27 Oct (field/runs_segments_sendall, default choice), answers from batch_001.
Nearest arm: the same session re-extracted with segments.choice = nearest_sharp (field/runs_eval/nearest),
otherwise the default config with --send-all. For every segment covered in both: which frame, how far ahead,
the segment's rows and along-track resolution at the VLM's input (whole frame -> 1344 x 756, x 0.35), the
predicted blur, and the fair sharpness of both chosen frames on the same ground strip (segments.FairSharpness:
the segment's first metre on a 4 mm grid). Also: how much of the discriminator's fair strip (|x| <= 1.5 m,
4 mm, no anti-aliasing) lay outside the image at each window position (its constant fill adds edges).
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
from geoai.geometry import CameraModel  # noqa: E402
from geoai.segments import FairSharpness, SegmentGeometry  # noqa: E402
from geoai.vlm import answer_class, read_answers  # noqa: E402

OUT = Path(r"C:/GeoAI/analysis/fixes/vlm_view")
SID = "20261027T050257Z_SM-G991B-F0D1_d47c"
RUN_S = REPO / "field/runs_segments_sendall" / SID
RUN_N = REPO / "field/runs_eval/nearest" / SID
VLM_SCALE = 1344 / 3840
cfg = Config()
rng = np.random.default_rng(11)


def load(run):
    return {json.loads(p.read_text(encoding="utf-8"))["segment"]["k"]: json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((run / "sidecars").glob("*.json"))}


def model_of(run):
    c = json.loads((run / "report.json").read_text(encoding="utf-8"))["geometry"]["3840x2160"]["camera"]
    return CameraModel(c["width"], c["height"], np.array(c["K"]), np.array(c["dist"]), np.array(c["R_v_from_c"]),
                       np.array(c["C_v"]), c["source"])


def med_ci(x, n=2000):
    x = np.asarray(x, float)
    b = [np.median(rng.choice(x, len(x))) for _ in range(n)]
    return {"median": float(np.median(x)), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
            "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90)), "n": int(len(x))}


def geometry():
    S, N = load(RUN_S), load(RUN_N)
    m = model_of(RUN_N)
    g = SegmentGeometry.build(m, cfg)
    rep_n = json.loads((RUN_N / "report.json").read_text(encoding="utf-8"))
    near = rep_n["plan"]["near_m"]
    fs = FairSharpness(g, near, cfg.segments)
    rows, changed = [], []
    for k in sorted(set(S) & set(N)):
        a, b = S[k], N[k]
        if a["frame"] is None or b["frame"] is None:
            continue
        fa, fb = a["frame"], b["frame"]
        row = {"k": k, "changed": fa["frame_global"] != fb["frame_global"]}
        for tag, r, run in (("s", a, RUN_S), ("n", b, RUN_N)):
            f = r["frame"]
            vt, vb = f["rows_of_segment"]
            r0, r1 = g.row_slice(vt, vb)
            row[f"{tag}_near_ahead_m"] = f["near_edge_ahead_m"]
            row[f"{tag}_vlm_rows"] = (r1 - r0) * VLM_SCALE
            row[f"{tag}_gsd_centre_vlm_mm"] = float(g.gsd_along_at((r0 + r1 - 1) / 2)) / VLM_SCALE
            row[f"{tag}_blur_px"] = r["quality"]["blur_px"]
            row[f"{tag}_reason"] = r["quality"]["reason"]
        if row["changed"]:
            imgs = {}
            for tag, r, run in (("s", a, RUN_S), ("n", b, RUN_N)):
                bgr = cv2.imread(str(run / r["frame"]["file"]), cv2.IMREAD_COLOR)
                imgs[tag] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                row[f"{tag}_fair"] = fs(imgs[tag], r["frame"]["near_edge_ahead_m"])
            row["fair_ratio_n_over_s"] = row["n_fair"] / row["s_fair"]
            changed.append(a["id"])
        rows.append(row)
    ch = [r for r in rows if r["changed"]]
    # The discriminator's strip: x in [-1.5, 1.5] m, [d0, d0 + 1.5] m at 4 mm, remapped with a zero border.
    xs = np.arange(-1.5, 1.5, 0.004) + 0.002
    border = {}
    for off in (0.0, 0.25, 0.5, 0.75, 1.0):
        d0 = near + off
        X, Y = np.meshgrid(xs, d0 + np.arange(0, 1.5, 0.004))
        u, v, ok = m.ground_to_pixel(X, Y)
        inside = ok & (u >= 0) & (u <= m.width - 1) & (v >= 0) & (v <= m.height - 1)
        border[f"{off:.2f}"] = float(1 - inside.mean())
    out = {
        "session": SID, "segments_compared": len(rows), "changed": len(ch), "changed_share": len(ch) / len(rows),
        "near_report_quality": rep_n["quality"].get("fair_sharpness_rel"),
        "chosen_is_raw_sharpest_share": rep_n["quality"].get("chosen_is_raw_sharpest_share"),
        "reasons_nearest": rep_n["quality"]["chosen_reasons"],
        "over_blur_share": {"sharpest": float(np.mean([r["s_reason"].endswith("over_blur_limit") for r in rows])),
                            "nearest": float(np.mean([r["n_reason"].endswith("over_blur_limit") for r in rows]))},
        "all_segments": {f"{t}_{q}": med_ci([r[f"{t}_{q}"] for r in rows])
                         for t in ("s", "n") for q in ("near_ahead_m", "vlm_rows", "gsd_centre_vlm_mm", "blur_px")},
        "changed_segments": {
            **{f"{t}_{q}": med_ci([r[f"{t}_{q}"] for r in ch]) for t in ("s", "n")
               for q in ("near_ahead_m", "vlm_rows", "gsd_centre_vlm_mm", "blur_px", "fair")},
            "vlm_rows_gain": med_ci([r["n_vlm_rows"] / r["s_vlm_rows"] for r in ch]),
            "gsd_gain_s_over_n": med_ci([r["s_gsd_centre_vlm_mm"] / r["n_gsd_centre_vlm_mm"] for r in ch]),
            "fair_ratio_n_over_s": med_ci([r["fair_ratio_n_over_s"] for r in ch]),
            "share_fair_ratio_ge_0.9": float(np.mean([r["fair_ratio_n_over_s"] >= 0.9 for r in ch])),
            "share_fair_ratio_ge_1": float(np.mean([r["fair_ratio_n_over_s"] >= 1.0 for r in ch]))},
        "discriminator_strip_outside_image_share_by_window_offset_m": border,
        "timings_s": {"sharpest": json.loads((RUN_S / "report.json").read_text())["timings_s"].get("total"),
                      "nearest": rep_n["timings_s"].get("total")},
    }
    with open(OUT / "nearest_segments.csv", "w", newline="", encoding="utf-8") as f:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    (OUT / "changed_ids.txt").write_text("\n".join(changed) + "\n", encoding="utf-8")
    json.dump(out, open(OUT / "nearest_geometry.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


def answers():
    allowed = set(cfg.vlm.labels)
    a0 = read_answers(REPO / "field/batches/batch_001", cfg)
    an = read_answers(REPO / "field/batches/ab_nearest_001", cfg)
    S = load(RUN_S)
    cls_s = {r["segment"]["k"]: answer_class(a0.results.get(r["id"]), allowed)[0] for r in S.values() if r["frame"]}
    cls_n = dict(cls_s)
    changed = [x for x in (OUT / "changed_ids.txt").read_text().split() if x]
    for rid in changed:
        cls_n[int(rid.rsplit("_seg", 1)[1])] = answer_class(an.results.get(rid), allowed)[0]
    t = {"pp": 0, "pn": 0, "np": 0, "nn": 0, "unknown": 0}
    flips = []
    for rid in changed:
        k = int(rid.rsplit("_seg", 1)[1])
        a, b = cls_s[k], cls_n[k]
        if "unknown" in (a, b):
            t["unknown"] += 1
        else:
            t[("p" if a == "distress" else "n") + ("p" if b == "distress" else "n")] += 1
        if a != b:
            flips.append({"id": rid, "sharpest": a, "nearest": b,
                          "sharpest_labels": "+".join(answer_class(a0.results.get(rid), allowed)[1]),
                          "nearest_labels": "+".join(answer_class(an.results.get(rid), allowed)[1])})

    def cluster(cls):
        ks = sorted(cls)
        p = [cls[k] == "distress" for k in ks]
        pairs = [(p[i - 1], p[i]) for i in range(1, len(ks)) if ks[i] == ks[i - 1] + 1]
        base = sum(p) / len(p)
        cond = sum(1 for a, b in pairs if a and b) / max(1, sum(1 for a, b in pairs if a))
        return {"n": len(p), "positives": sum(p), "base": base, "p_pos_given_prev_pos": cond, "lift": cond / base}

    sec = [float(an.results[r]["seconds"]) for r in changed if r in an.results and an.results[r].get("seconds")]
    out = {"changed_answered": sum(1 for r in changed if r in an.results), "table_sharpest_vs_nearest": t,
           "agreement": (t["pp"] + t["nn"]) / max(1, t["pp"] + t["pn"] + t["np"] + t["nn"]),
           "clustering_full_session": {"sharpest": cluster(cls_s), "nearest": cluster(cls_n)},
           "seconds_median_nearest_arm": float(np.median(sec)) if sec else None,
           "profile": [r.get("profile") for r in an.run_info.get("runs", [])], "flips": flips}
    json.dump(out, open(OUT / "nearest_vs_sharpest.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "flips"}, indent=1), len(flips), "flips")


if __name__ == "__main__":
    {"geometry": geometry, "answers": answers}[sys.argv[1]]()
