"""Discriminator analysis 5 (CPU only): every candidate of every segment in the first N frames of a session.

Rebuilds the plan exactly as frames.py (Session, trajectory, nominal model, SegmentGeometry, plan_segments,
BestFrameSelector), decodes the first N frames, and for each candidate of each segment records:
  win_pos (0 = nearest frame, 1 = farthest), the code's sharpness, the segment's VLM-input rows and the
  along-track GSD at the segment centre after the VLM's resize (x 1/0.35), predicted blur, and a
  scale-fair sharpness: Tenengrad of the segment's NEAR 1.5 m strip, |x| <= 1.5 m, rectified to a
  common 4 mm ground grid (so all candidates are compared at the same ground resolution).
Writes out/bestframe_<sess>.csv.  Usage: a05_bestframe.py SESSION_ID N_FRAMES
"""
import os
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "run with CUDA_VISIBLE_DEVICES=-1"
import sys
import time
from pathlib import Path

import numpy as np
import torch
assert not torch.cuda.is_available()
torch.set_num_threads(4)
import cv2
import pandas as pd

sys.path.insert(0, r"C:/Users/PRO-LAB-3/Documents/GeoAI_Video/pipeline")
from geoai.config import Config  # noqa: E402
from geoai.session import Session  # noqa: E402
from geoai.trajectory import build_trajectory, clean_fixes  # noqa: E402
from geoai.extract import _median_gravity, _RowClock, _in_intervals  # noqa: E402
from geoai.geometry import nominal_model, cam_from_device  # noqa: E402
from geoai.segments import (SegmentGeometry, plan_segments, plan_offset_m, BestFrameSelector, FrameInfo,  # noqa: E402
                            GyroImageRates, sharpness, predicted_blur)
from geoai.video import iter_segment, segment_info  # noqa: E402

SID = sys.argv[1]
NMAX = int(sys.argv[2])
sdir = Path(r"C:/Users/PRO-LAB-3/Documents/GeoAI_Video/field/sessions") / SID
OUT = Path(r"C:/GeoAI/analysis/discriminator/out")
cfg = Config()
s = Session.load(sdir)
t0 = s.t0_ns
traj = build_trajectory(clean_fixes(s.gnss, t0, cfg.gnss), s.imu, t0, cfg.gnss)
files = s.video_files()
seg1 = sorted(files)[0]
info = segment_info(files[seg1])
W, H = info["width"], info["height"]
model = nominal_model(s.meta, W, H, _median_gravity(s), cfg.geometry, rotate180=False)
geom = SegmentGeometry.build(model, cfg)
frames = s.frames.sort_values("t_ns", kind="stable")
clock = _RowClock(t0, H, False)
first, last = frames.iloc[0].to_dict(), frames.iloc[-1].to_dict()
segments = plan_segments(traj, cfg, t_span=(clock(first, (H - 1) / 2), clock(last, (H - 1) / 2)),
                         offset_m=plan_offset_m([geom], cfg))
sel = BestFrameSelector(segments, [geom], cfg)
gyro = GyroImageRates(s.imu, cam_from_device(90), t0)
black = s.black_intervals_ns()
rows = s.frames[s.frames["segment"] == seg1].sort_values("frame_in_segment")
recs = rows.to_dict("records")[:NMAX]
t = np.array([clock(r, (H - 1) / 2) for r in recs])
p = traj.sample(t)
low = s.low_light_at(np.array([int(r["t_ns"]) for r in recs]))
om = gyro(t)
infos = {int(r["frame_in_segment"]): FrameInfo(float(t[i]), float(p["s"][i]), float(p["speed"][i]), bool(p["valid"][i]),
                                               bool(p["reversing"][i]), _in_intervals(int(r["t_ns"]), black), bool(low[i]),
                                               r.get("exposure_ns"), float(om[i]), r) for i, r in enumerate(recs)}
VLM_SCALE = 1344 / 3840
GRID_MM = 4.0
STRIP_M = 1.5
X_HALF = 1.5
xs = np.arange(-X_HALF, X_HALF, GRID_MM / 1000) + GRID_MM / 2000


def fair_sharp(rgb, d0):
    """Tenengrad of ground strip [d0, d0 + STRIP_M] x [-X_HALF, X_HALF], rectified to GRID_MM."""
    ys = d0 + np.arange(0, STRIP_M, GRID_MM / 1000)[::-1]
    X, Y = np.meshgrid(xs, ys)
    u, v, ok = model.ground_to_pixel(X, Y)
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # area-average to ~ the grid scale first (prevents aliasing), then sample
    img = cv2.remap(grey, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)[1:-1, 1:-1]
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)[1:-1, 1:-1]
    m = float(img[1:-1, 1:-1].mean())
    return float(np.mean(gx * gx + gy * gy)) / (m * m) if m > 0 else 0.0


out = []
tic = time.time()
n = 0
for fr in iter_segment(files[seg1], rows, segment=seg1, want=lambda i, r: i in infos and sel.wants(infos[i]),
                       allow_extra_rows=True):
    if fr.index not in infos:
        break
    inf = infos[fr.index]
    n += 1
    if fr.rgb is not None:
        for k in sel._in_windows(inf.c):
            sg = segments[k]
            d0, d1 = sg.s0 - inf.c, sg.s1 - inf.c
            vt, vb = geom.rows_for(d0, d1)
            r0, r1 = geom.row_slice(vt, vb)
            sh = sharpness(fr.rgb, (r0, r1), cols=cfg.segments.sharp_cols, downscale=cfg.segments.sharp_downscale)
            gsd_c = float(geom.gsd_along_at((r0 + r1 - 1) / 2))
            gsd_far = float(geom.gsd_along_at(max(r0, 0)))
            bl = predicted_blur(inf.speed_mps, inf.exposure_ns, gsd_c, inf.omega_perp_rad_s, geom.f_px)
            out.append({"k": k, "frame": fr.index, "c": inf.c, "speed": inf.speed_mps, "win_pos": d0 - sel.near_m,
                        "sharp": sh, "fair_sharp": fair_sharp(fr.rgb, max(d0, sel.near_m)), "rows": r1 - r0,
                        "vlm_rows": (r1 - r0) * VLM_SCALE, "gsd_c_full": gsd_c, "gsd_c_vlm": gsd_c / VLM_SCALE,
                        "gsd_far_vlm": gsd_far / VLM_SCALE, "blur_px": bl["blur_px"], "blur_rot": bl["blur_rotation_px"]})
    sel.offer(inf, fr.rgb)
    sel.pop_ready(inf.c)
    if n >= NMAX - 1:
        break
df = pd.DataFrame(out)
df.to_csv(OUT / f"bestframe_{SID[-4:]}.csv", index=False)
print("decoded", n, "frames in", round(time.time() - tic, 1), "s; candidates", len(df), "segments", df.k.nunique())
