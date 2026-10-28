"""Discriminator analysis 3 (CPU only): why the crack filter fires everywhere, and per-step CPU costs.

Uses geoai's own CrackDetector, View (gate.plan_segment_view) and write_png on real best frames of
the send-all run. Background statistics from 20 d47c frames spread over the route (the gate uses its
first 20 bank keyframes: same road, same kind of statistic). Writes out/crack_timing.json and
out/crack_*.png overlays.
"""
import os
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "run with CUDA_VISIBLE_DEVICES=-1"
import json
import sys
import time
import hashlib
from pathlib import Path

import numpy as np
import torch
assert not torch.cuda.is_available()
torch.set_num_threads(8)
import cv2

sys.path.insert(0, r"C:/Users/PRO-LAB-3/Documents/GeoAI_Video/pipeline")
from geoai.config import Config  # noqa: E402
from geoai.geometry import CameraModel  # noqa: E402
from geoai.gate import plan_segment_view, gate_far_m  # noqa: E402
from geoai.cracks import CrackDetector, normalised_gray, ridge_strength  # noqa: E402
from geoai.records import write_png  # noqa: E402

ROOT = Path(r"C:/Users/PRO-LAB-3/Documents/GeoAI_Video/field/runs_segments_sendall")
OUT = Path(r"C:/GeoAI/analysis/discriminator/out")
SESS = sys.argv[1] if len(sys.argv) > 1 else "20261027T050257Z_SM-G991B-F0D1_d47c"
cfg = Config()
rep = json.loads((ROOT / SESS / "report.json").read_text())
cam = rep["geometry"]["3840x2160"]["camera"]
model = CameraModel(cam["width"], cam["height"], np.array(cam["K"]), np.array(cam["dist"]), np.array(cam["R_v_from_c"]),
                    np.array(cam["C_v"]), cam["source"], cam["notes"])
near = rep["plan"]["near_m"]
view = plan_segment_view(model, near, gate_far_m(near, cfg), cfg, False)
print("view", view.summary()["analysis_size"], view.v0, view.v1, view.sx)
side = sorted((ROOT / SESS / "sidecars").glob("*.json"))
recs = [json.loads(p.read_text()) for p in side]
recs = [r for r in recs if r["status"] == "send"]
rng = np.random.default_rng(0)
bank_idx = np.linspace(0, len(recs) - 1, 20).astype(int)
test_idx = np.linspace(5, len(recs) - 6, 12).astype(int)


def load(r):
    t = time.perf_counter()
    bgr = cv2.imread(str(ROOT / SESS / r["frame"]["file"]), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb, time.perf_counter() - t


det = CrackDetector(cfg.cracks.sigmas_px, cfg.cracks.z_threshold, cfg.cracks.min_length_mm)
t_png_read = []
for i in bank_idx:
    rgb, dt = load(recs[i])
    t_png_read.append(dt)
    det.add_background(view.analysis_image(rgb), rng)
det.finalize()
print("background med, scale", det.med, det.scale)

res = {"session": SESS[-4:], "analysis_size": [view.ana_w, view.ana_h], "bg_median": det.med, "bg_scale": det.scale,
       "frames": []}
tim = {k: [] for k in ("png_read", "analysis_resize", "normalised_gray", "ridge", "detect_total", "sharpness_band",
                       "png_write_l1", "sha256", "png_write_l0")}
tim["png_read"] = t_png_read
# GSD across-track vs along-track per analysis row (centre column)
rr = np.arange(view.ana_h, dtype=float)
along, across = view.model_ana.gsd_mm(np.full_like(rr, (view.ana_w - 1) / 2), rr)
res["gsd_ratio_along_over_across_p10_50_90"] = np.nanpercentile(along / across, [10, 50, 90]).round(2).tolist()

for j, i in enumerate(test_idx):
    r = recs[i]
    rgb, _ = load(r)
    t = time.perf_counter(); ana = view.analysis_image(rgb); tim["analysis_resize"].append(time.perf_counter() - t)
    t = time.perf_counter(); g = normalised_gray(ana); tim["normalised_gray"].append(time.perf_counter() - t)
    t = time.perf_counter(); rs = ridge_strength(g, det.sigmas); tim["ridge"].append(time.perf_counter() - t)
    t = time.perf_counter(); mask, comps = det.detect(ana, view.gsd_rows_mm); tim["detect_total"].append(time.perf_counter() - t)
    z = (rs - det.med) / det.scale
    # segment rows in analysis coords
    vt, vb = r["frame"]["rows_of_segment"]
    a0 = int(max(0, (vt - view.v0) * view.sy)); a1 = int(min(view.ana_h, (vb - view.v0 + 1) * view.sy))
    seg_mask = mask[a0:a1]
    # length estimates: bbox diagonal x along-GSD (the code) vs skeleton pixels x local GSD (mean of along/across)
    lens_code, lens_skel, big = [], [], 0
    n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    skel = cv2.ximgproc.thinning(mask.astype(np.uint8) * 255) if hasattr(cv2, "ximgproc") else None
    for c in comps:
        x0, y0, x1, y1 = c.bbox
        cy = (y0 + y1) // 2
        if not (a0 <= cy < a1):
            continue
        lens_code.append(c.length_mm)
        if skel is not None:
            sub = skel[y0:y1, x0:x1] > 0
            rows = np.nonzero(sub)[0] + y0
            gm = np.sqrt(along[rows] * across[rows]) if rows.size else np.array([0.0])
            lens_skel.append(float(gm.sum()))
        if (x1 - x0) > 0.25 * view.ana_w or (y1 - y0) > 0.25 * (a1 - a0):
            big += 1
    fr = {"id": r["id"][-12:], "seg_rows_ana": [a0, a1], "pix_frac_z_gt_6": float((z[a0:a1] > 6).mean()),
          "pix_frac_z_gt_10": float((z[a0:a1] > 10).mean()), "pix_frac_z_gt_15": float((z[a0:a1] > 15).mean()),
          "mask_frac_segment": float(seg_mask.mean()), "n_components_kept_in_segment": len(lens_code),
          "crack_mm_code": float(np.sum(lens_code)), "crack_mm_skeleton": float(np.sum(lens_skel)) if lens_skel else None,
          "n_big_components": big, "record_crack_mm": r["gate"]["scores"]["crack_mm"],
          "comp_len_p50_p90_max": np.percentile(lens_code, [50, 90, 100]).round(0).tolist() if lens_code else None}
    # how many components survive stricter settings (same image)
    for zt in (10, 15, 20):
        d2 = CrackDetector(det.sigmas, zt, cfg.cracks.min_length_mm); d2.med, d2.scale = det.med, det.scale
        _, c2 = d2.detect(ana, view.gsd_rows_mm)
        fr[f"crack_mm_z{zt}"] = float(sum(c.length_mm for c in c2 if a0 <= (c.bbox[1] + c.bbox[3]) // 2 < a1))
    res["frames"].append(fr)
    print(fr)
    if j < 3:
        ov = ana.copy()
        ov[mask] = (255, 0, 0)
        cv2.line(ov, (0, a0), (ov.shape[1] - 1, a0), (0, 255, 0), 3)
        cv2.line(ov, (0, a1 - 1), (ov.shape[1] - 1, a1 - 1), (0, 255, 0), 3)
        cv2.imwrite(str(OUT / f"crack_{SESS[-4:]}_{r['id'][-5:]}.jpg"),
                    cv2.cvtColor(cv2.resize(ov, (ov.shape[1] // 2, ov.shape[0] // 2), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    # sharpness on the segment band (code's function)
    from geoai.segments import sharpness
    t = time.perf_counter(); sharpness(rgb, (int(max(0, vt)), int(min(2160, vb))), cols=(0.2, 0.8), downscale=0.5)
    tim["sharpness_band"].append(time.perf_counter() - t)
    if j < 4:
        tmp = OUT / "tmp_write.png"
        t = time.perf_counter(); write_png(tmp, rgb, None, compress_level=1); tim["png_write_l1"].append(time.perf_counter() - t)
        sz1 = tmp.stat().st_size
        t = time.perf_counter(); hashlib.sha256(tmp.read_bytes()).hexdigest(); tim["sha256"].append(time.perf_counter() - t)
        t = time.perf_counter(); write_png(tmp, rgb, None, compress_level=0); tim["png_write_l0"].append(time.perf_counter() - t)
        res.setdefault("png_bytes_l1", []).append(sz1); res.setdefault("png_bytes_l0", []).append(tmp.stat().st_size)
        tmp.unlink()
res["timing_s_median"] = {k: float(np.median(v)) for k, v in tim.items() if v}
(OUT / f"crack_timing_{SESS[-4:]}.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res["timing_s_median"], indent=1), res["gsd_ratio_along_over_across_p10_50_90"])
