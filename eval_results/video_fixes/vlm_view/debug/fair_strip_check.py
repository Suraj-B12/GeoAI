"""Which ground points of FairSharpness's strip leave the image, per drive (CPU, no frames decoded)."""
import json, math, os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from pathlib import Path
import numpy as np
REPO = Path(r"C:/Users/PRO-LAB-3/Documents/GeoAI_Video")
sys.path.insert(0, str(REPO / "pipeline"))
from geoai.config import Config
from geoai.geometry import CameraModel
from geoai.segments import FairSharpness, SegmentGeometry

cfg = Config(); cfg.segments.choice = "nearest_sharp"
c = cfg.segments
for run in sorted((REPO / "field/runs_segments_sendall").glob("2026*")):
    cam = json.loads((run / "report.json").read_text(encoding="utf-8"))["geometry"]["3840x2160"]["camera"]
    m = CameraModel(cam["width"], cam["height"], np.array(cam["K"]), np.array(cam["dist"]), np.array(cam["R_v_from_c"]),
                    np.array(cam["C_v"]), cam["source"])
    g = SegmentGeometry.build(m, cfg)
    R = np.array(cam["R_v_from_c"])
    # camera optical axis and right axis in the vehicle frame -> yaw / roll indication
    z = R @ np.array([0, 0, 1.0]); x = R @ np.array([1.0, 0, 0])
    yaw = math.degrees(math.atan2(z[0], z[1])) if abs(z[1]) > 1e-9 else float("nan")
    roll = math.degrees(math.asin(max(-1, min(1, x[2]))))
    try:
        FairSharpness(g, g.near_m, cfg); ok = "ok"
    except ValueError as e:
        ok = f"FAIL: {e}"
    # replicate the strip to see where it leaves
    k = int(c.fair_supersample); step = c.fair_grid_mm / 1000 / k
    n_strip = k * max(3, int(round(c.fair_strip_m / (c.fair_grid_mm / 1000))))
    n_y = n_strip + int(math.ceil(c.window_m / step)) + 1
    xb, yb, okb = m.pixel_to_ground(np.array([0.0, m.width - 1.0]), np.array([float(g.v_usable)] * 2))
    hw = min(float(c.fair_half_width_m), float(-xb[0]), float(xb[1])) - c.fair_grid_mm / 1000
    n_x = int(math.floor(2 * hw / (c.fair_grid_mm / 1000))) * k
    xs = (np.arange(n_x) - (n_x - 1) / 2) * step; ys = g.near_m + (np.arange(n_y) + 0.5) * step
    X, Y = np.meshgrid(xs, ys); u, v, okp = m.ground_to_pixel(X, Y)
    out = ~(okp & (u >= 0) & (u <= m.width - 1) & (v >= 0) & (v <= g.v_usable))
    print(run.name[-4:], ok, f"| yaw {yaw:.2f} deg roll {roll:.2f} deg | near_m {g.near_m:.3f} v_usable {g.v_usable}",
          f"| bottom corners x {xb[0]:.3f} {xb[1]:.3f} y {yb[0]:.3f} {yb[1]:.3f} | hw {hw:.3f}",
          f"| outside {int(out.sum())}/{out.size}", end="")
    if out.any():
        r, cidx = np.nonzero(out)
        print(f" rows y {ys[r.min()]:.3f}..{ys[r.max()]:.3f} cols x {xs[cidx.min()]:.3f}..{xs[cidx.max()]:.3f}",
              f"v max {np.nanmax(v[out]):.1f} u {np.nanmin(u[out]):.1f}..{np.nanmax(u[out]):.1f}")
    else:
        print()
