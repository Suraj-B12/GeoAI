"""Discriminator analysis 6 (CPU only): where the decode pass spends its time, per frame.

Decodes the first N frames of a session's first video file with geoai's own settings (PyAV, thread_type
AUTO) and times: decode only; the BT.709 RGB conversion (geoai.video.to_rgb); the sharpness of a
typical segment band (geoai.segments.sharpness, rows 473..2159 as the nearest segment, lane columns)
on the RGB frame; the same Tenengrad from the decoded luma plane (no RGB); and luma-plane extraction.
Writes out/decode_timing.json.  Usage: a06_decode_timing.py SESSION N
"""
import os
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
import json
import sys
import time
from pathlib import Path

import av
import cv2
import numpy as np
from av.video.reformatter import VideoReformatter

sys.path.insert(0, r"C:/Users/PRO-LAB-3/Documents/GeoAI_Video/pipeline")
import torch  # noqa: E402
assert not torch.cuda.is_available()
from geoai.video import to_rgb  # noqa: E402
from geoai.segments import sharpness  # noqa: E402

SID, N = sys.argv[1], int(sys.argv[2])
vid = sorted((Path(r"C:/Users/PRO-LAB-3/Documents/GeoAI_Video/field/sessions") / SID / "video").glob("*.mp4"))[0]
rows = (473, 2159)


def sharp_luma(y, rows, cols=(0.2, 0.8), downscale=0.5):
    H, W = y.shape
    c0, c1 = int(round(cols[0] * W)), int(round(cols[1] * W))
    g = y[rows[0]:rows[1], c0:c1]
    g = cv2.resize(g, (int(round(g.shape[1] * downscale)), int(round(g.shape[0] * downscale))), interpolation=cv2.INTER_AREA)
    g = g.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)[1:-1, 1:-1]
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)[1:-1, 1:-1]
    m = float(g[1:-1, 1:-1].mean())
    return float(np.mean(gx * gx + gy * gy)) / (m * m)


res = {}
# 1. decode only
with av.open(str(vid)) as c:
    st = c.streams.video[0]; st.thread_type = "AUTO"
    t = time.perf_counter(); n = 0
    for f in c.decode(st):
        n += 1
        if n >= N:
            break
    res["decode_only_fps"] = n / (time.perf_counter() - t)
# 2. decode + per-op timing on every 5th frame
rf = VideoReformatter()
tr, ts, tl, tsl, corr = [], [], [], [], []
with av.open(str(vid)) as c:
    st = c.streams.video[0]; st.thread_type = "AUTO"
    n = 0
    for f in c.decode(st):
        n += 1
        if n % 5 == 0:
            t = time.perf_counter(); rgb = to_rgb(f, rf); tr.append(time.perf_counter() - t)
            t = time.perf_counter(); s1 = sharpness(rgb, rows, cols=(0.2, 0.8), downscale=0.5); ts.append(time.perf_counter() - t)
            t = time.perf_counter(); y = f.to_ndarray(format="gray"); tl.append(time.perf_counter() - t)
            t = time.perf_counter(); s2 = sharp_luma(y, rows); tsl.append(time.perf_counter() - t)
            corr.append((s1, s2))
        if n >= N:
            break
corr = np.array(corr)
from scipy.stats import spearmanr  # noqa: E402
res.update({"frames": N, "to_rgb_s": float(np.median(tr)), "sharpness_rgb_s": float(np.median(ts)),
            "luma_plane_s": float(np.median(tl)), "sharpness_luma_s": float(np.median(tsl)),
            "spearman_sharp_rgb_vs_luma": float(spearmanr(corr[:, 0], corr[:, 1])[0]), "n_pairs": len(corr)})
(Path(r"C:/GeoAI/analysis/discriminator/out") / f"decode_timing_{SID[-4:]}.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
