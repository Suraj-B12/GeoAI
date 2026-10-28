"""Discriminator analysis 4 (CPU only): gate signals recomputed with candidate fixes, on every VLM-answered frame.

For each answered send-all frame (session f428 so far) it recomputes, with geoai's own code:
  - Attain-probe logits per patch (PatchFeatures on CPU + DistressProbe, paint-masked as AnomalyDetector.maps)
  - the Hessian crack components (CrackDetector, background from 20 frames of the same session)
and scores the segment rows under variants:
  probe_frac / probe_max over all columns (= the record) vs road-only columns (|ground x| <= X m);
  crack_mm as coded (bbox diagonal x along-track GSD, all columns) vs road-only with the length from
  the box extents in their own GSDs (hypot(w x across, h x along)), at z 6 and 10.
Writes out/gate_variants.csv. Usage: a04_gate_variants.py [limit]
"""
import os
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "run with CUDA_VISIBLE_DEVICES=-1"
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
assert not torch.cuda.is_available()
torch.set_num_threads(int(os.environ.get("NTHREADS", "24")))
import cv2
import pandas as pd

sys.path.insert(0, r"C:/Users/PRO-LAB-3/Documents/GeoAI_Video/pipeline")
from geoai.config import Config  # noqa: E402
from geoai.geometry import CameraModel  # noqa: E402
from geoai.gate import plan_segment_view, gate_far_m  # noqa: E402
from geoai.cracks import CrackDetector  # noqa: E402
from geoai.dinov2 import PatchFeatures, PATCH  # noqa: E402
from geoai.probe import DistressProbe  # noqa: E402
from geoai.anomaly import normalise_illumination, paint_fraction  # noqa: E402

ROOT = Path(r"C:/Users/PRO-LAB-3/Documents/GeoAI_Video/field/runs_segments_sendall")
OUT = Path(r"C:/GeoAI/analysis/discriminator/out")
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 10**9
cfg = Config()
an = cfg.anomaly
J = pd.read_csv(OUT / "joined.csv")
ids = list(J["id"])[:LIMIT]
sess = ids[0].rsplit("_seg", 1)[0]
rep = json.loads((ROOT / sess / "report.json").read_text())
cam = rep["geometry"]["3840x2160"]["camera"]
model = CameraModel(cam["width"], cam["height"], np.array(cam["K"]), np.array(cam["dist"]), np.array(cam["R_v_from_c"]),
                    np.array(cam["C_v"]), cam["source"], cam["notes"])
near = rep["plan"]["near_m"]
view = plan_segment_view(model, near, gate_far_m(near, cfg), cfg, False)
gh, gw = view.ana_h // PATCH, view.ana_w // PATCH
feats_fn = PatchFeatures(Path(an.weights), "cpu", an.tile_patches, an.tile_overlap_patches, an.batch_tiles,
                         an.weights_sha256, an.layers)
probe = DistressProbe.load(Path(an.probe))
seed = probe.seed_logit
# ground x of every analysis pixel row/col (centre of patches and pixels)
ma = view.model_ana
pr = (np.arange(gh) + 0.5) * PATCH - 0.5
pc = (np.arange(gw) + 0.5) * PATCH - 0.5
PU, PV = np.meshgrid(pc, pr)
px_x, _, px_ok = ma.pixel_to_ground(PU, PV)
rr = np.arange(view.ana_h, dtype=float)
along, across = ma.gsd_mm(np.full_like(rr, (view.ana_w - 1) / 2), rr)
patch_row_v = view.v0 + ((np.arange(gh) * PATCH) + (np.arange(1, gh + 1) * PATCH)) / (2 * view.sy) - 0.5


def rgb_of(rid):
    rec = json.loads((ROOT / sess / "sidecars" / f"{rid}.json").read_text())
    bgr = cv2.imread(str(ROOT / sess / rec["frame"]["file"]), cv2.IMREAD_COLOR)
    return rec, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# crack background from 20 frames spread over the session (all send-all records of it)
allrec = sorted((ROOT / sess / "sidecars").glob("*.json"))
det = CrackDetector(cfg.cracks.sigmas_px, cfg.cracks.z_threshold, cfg.cracks.min_length_mm)
rng = np.random.default_rng(0)
for p in [allrec[i] for i in np.linspace(0, len(allrec) - 1, 20).astype(int)]:
    _, rgb = rgb_of(p.stem)
    det.add_background(view.analysis_image(rgb), rng)
det.finalize()
det10 = CrackDetector(det.sigmas, 10.0, cfg.cracks.min_length_mm); det10.med, det10.scale = det.med, det.scale

rows = []
t_feat = []
for n, rid in enumerate(ids):
    rec, rgb = rgb_of(rid)
    ana = view.analysis_image(rgb)
    t = time.perf_counter()
    feats = feats_fn(normalise_illumination(ana, an.illum_sigma_px))
    t_feat.append(time.perf_counter() - t)
    logit = probe.logits(feats)
    paint = paint_fraction(ana, gh, gw, an.paint_ratio) > an.paint_patch_frac
    logit = np.where(paint, -1e3, logit)
    vt, vb = rec["frame"]["rows_of_segment"]
    seg_rows = (patch_row_v > vt) & (patch_row_v <= vb)
    out = {"id": rid, "rec_probe_frac": rec["gate"]["scores"]["probe_frac"],
           "rec_probe_max": rec["gate"]["scores"]["probe_max_logit"], "rec_crack_mm": rec["gate"]["scores"]["crack_mm"]}
    for X in (None, 2.5, 1.75):
        m = seg_rows[:, None] & np.ones((1, gw), bool) if X is None else (seg_rows[:, None] & px_ok & (np.abs(px_x) <= X))
        lg = logit[m]
        tag = "all" if X is None else f"x{X}"
        out[f"probe_frac_{tag}"] = float((lg >= seed).mean()) if lg.size else 0.0
        out[f"probe_max_{tag}"] = float(lg.max()) if lg.size else -1e3
        out[f"probe_p99_{tag}"] = float(np.percentile(lg[lg > -1e2], 99)) if (lg > -1e2).any() else -1e3
        out[f"n_patch_{tag}"] = int(lg.size)
    # crack variants
    a0 = (vt - view.v0) * view.sy
    a1 = (vb - view.v0 + 1) * view.sy
    for dname, d in (("z6", det), ("z10", det10)):
        _, comps = d.detect(ana, view.gsd_rows_mm)
        tot_code = tot_fix_all = tot_fix_road = 0.0
        for c in comps:
            x0, y0, x1, y1 = c.bbox
            cy = (y0 + y1) / 2
            if not (a0 < cy <= a1):
                continue
            tot_code += c.length_mm
            ri = int(min(view.ana_h - 1, cy))
            L = math.hypot((x1 - x0) * across[ri], (y1 - y0) * along[ri])
            if L < cfg.cracks.min_length_mm:
                continue
            tot_fix_all += L
            gx, _, ok = ma.pixel_to_ground(np.array([(x0 + x1) / 2]), np.array([cy]))
            if ok[0] and abs(gx[0]) <= 1.75:
                tot_fix_road += L
        out[f"crack_code_{dname}"] = tot_code
        out[f"crack_fixlen_{dname}"] = tot_fix_all
        out[f"crack_fixlen_road_{dname}"] = tot_fix_road
    rows.append(out)
    if n % 10 == 0:
        print(n, rid[-8:], round(np.median(t_feat), 2), {k: round(v, 3) for k, v in out.items() if k.startswith(("rec_probe", "probe_frac_all", "probe_max_all"))}, flush=True)
pd.DataFrame(rows).to_csv(OUT / "gate_variants.csv", index=False)
print("dinov2 cpu s/frame", np.median(t_feat))
