"""Proof for the 8-bit luma sharpness (CPU only): does it choose the same frame as the float-RGB version?

For each real session, one decode pass feeds two BestFrameSelectors with the same frames: A scores with the
previous float-RGB sharpness (copied verbatim below from git HEAD 3c72086, geoai/segments.py), B with the new
8-bit luma one (geoai.segments.sharpness of the working tree). The selection logic is geoai's own (the module
global `sharpness` is switched around each offer). Plan, geometry and frame descriptions follow
geoai.frames._process step by step. Also checks A against the chosen frames of the existing send-all run.

Writes out/sharpness_identity.json.  Usage (CPU): CUDA_VISIBLE_DEVICES=-1 python g02_sharpness_identity.py
"""
import os
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, r"C:/GeoAI/tmp/gate_speed_after/pipeline")  # git HEAD + this study's changes only
import torch  # noqa: E402
assert not torch.cuda.is_available()
import geoai  # noqa: E402
assert Path(geoai.__file__).as_posix().startswith("C:/GeoAI/tmp/gate_speed_after/"), geoai.__file__
import geoai.segments as segmod  # noqa: E402
from geoai.config import Config  # noqa: E402
from geoai.extract import _in_intervals, _median_gravity, _RowClock  # noqa: E402
from geoai.geometry import cam_from_device, nominal_model  # noqa: E402
from geoai.segment_records import load_segment_records  # noqa: E402
from geoai.segments import (BestFrameSelector, FrameInfo, GyroImageRates, SegmentGeometry, plan_offset_m,  # noqa: E402
                            plan_segments)
from geoai.session import Session  # noqa: E402
from geoai.trajectory import build_trajectory, clean_fixes  # noqa: E402
from geoai.video import iter_segment, segment_info  # noqa: E402

FIELD = Path(r"C:/Users/PRO-LAB-3/Documents/GeoAI_Video/field")
OUT = Path(__file__).resolve().parent / "out"
NEW = segmod.sharpness


def OLD(rgb, rows, *, cols, downscale):
    """geoai.segments.sharpness at git HEAD 3c72086 (float RGB), verbatim."""
    c_lo, c_hi = cols
    if not 0.0 <= c_lo < c_hi <= 1.0:
        raise ValueError(f"sharpness columns must satisfy 0 <= lo < hi <= 1, got {cols}")
    if not 0.0 < downscale <= 1.0:
        raise ValueError(f"sharpness downscale must be in (0, 1], got {downscale}")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"sharpness needs an H x W x 3 RGB image, got {rgb.shape}")
    H, W = rgb.shape[:2]
    r0, r1 = max(0, int(rows[0])), min(H, int(rows[1]))
    c0, c1 = int(round(c_lo * W)), int(round(c_hi * W))
    if r1 - r0 < 3 or c1 - c0 < 3:
        raise ValueError(f"sharpness region too small: rows {r0}..{r1}, columns {c0}..{c1}")
    grey = cv2.cvtColor(rgb[r0:r1, c0:c1].astype(np.float32), cv2.COLOR_RGB2GRAY)
    if downscale < 1.0:
        h, w = grey.shape
        size = (max(1, int(round(w * downscale))), max(1, int(round(h * downscale))))
        grey = cv2.resize(grey, size, interpolation=cv2.INTER_AREA)
    if grey.shape[0] < 3 or grey.shape[1] < 3:
        raise ValueError(f"sharpness region too small ({grey.shape[1]} x {grey.shape[0]} px after downscaling)")
    gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)[1:-1, 1:-1].astype(np.float64)
    gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)[1:-1, 1:-1].astype(np.float64)
    mean_i = float(grey[1:-1, 1:-1].mean(dtype=np.float64))
    if mean_i <= 0.0:
        return 0.0
    return float(np.mean(gx * gx + gy * gy)) / (mean_i * mean_i)


class Timed:
    def __init__(self, fn):
        self.fn, self.s, self.n, self.vals = fn, 0.0, 0, []

    def __call__(self, *a, **k):
        t = time.perf_counter()
        v = self.fn(*a, **k)
        self.s += time.perf_counter() - t
        self.n += 1
        self.vals.append(v)
        return v


def run(session_dir: Path, cfg: Config) -> dict:
    s = Session.load(session_dir)
    rotate180 = s.orientation_verdict == "UPSIDE_DOWN"
    t0 = s.t0_ns
    traj = build_trajectory(clean_fixes(s.gnss, t0, cfg.gnss), s.imu, t0, cfg.gnss)
    gravity = _median_gravity(s)
    seg_files = s.video_files()
    seg_res = {}
    for seg, path in seg_files.items():
        info = segment_info(path)
        seg_res[seg] = (info["width"], info["height"])
    models = {res: nominal_model(s.meta, res[0], res[1], gravity, cfg.geometry, rotate180=rotate180)
              for res in sorted(set(seg_res.values()))}
    geoms = {res: SegmentGeometry.build(m, cfg) for res, m in models.items()}
    frames = s.frames.sort_values("t_ns", kind="stable")
    first, last = frames.iloc[0].to_dict(), frames.iloc[-1].to_dict()

    def centre_time(row):
        H = seg_res[int(row["segment"])][1]
        return _RowClock(t0, H, rotate180)(row, (H - 1) / 2)

    segments = plan_segments(traj, cfg, t_span=(centre_time(first), centre_time(last)),
                             offset_m=plan_offset_m(list(geoms.values()), cfg))
    A = BestFrameSelector(segments, list(geoms.values()), cfg)
    B = BestFrameSelector(segments, list(geoms.values()), cfg)
    cam = s.meta.get("camera", {}) or {}
    so = int(cam.get("sensor_orientation_deg") or (s.meta.get("orientation") or {}).get("sensor_orientation_deg") or 90)
    gyro = GyroImageRates(s.imu, cam_from_device(so), t0)
    black = s.black_intervals_ns()
    old, new = Timed(OLD), Timed(NEW)
    ra, rb = [], []
    t_start = time.time()
    for seg, path in seg_files.items():
        res = seg_res[seg]
        rows = s.frames[s.frames["segment"] == seg]
        recs = rows.sort_values("frame_in_segment").to_dict("records")
        clock = _RowClock(t0, res[1], rotate180)
        t = np.array([clock(r, (res[1] - 1) / 2) for r in recs], dtype=np.float64)
        t_ns = np.array([int(r["t_ns"]) for r in recs], dtype=np.int64)
        p = traj.sample(t)
        low = s.low_light_at(t_ns)
        omega = gyro(t)
        infos = {int(r["frame_in_segment"]): FrameInfo(
            t=float(t[i]), c=float(p["s"][i]), speed_mps=float(p["speed"][i]), valid=bool(p["valid"][i]),
            reversing=bool(p["reversing"][i]), black=_in_intervals(int(t_ns[i]), black), low_light=bool(low[i]),
            exposure_ns=r.get("exposure_ns"), omega_perp_rad_s=float(omega[i]), ref=r) for i, r in enumerate(recs)}

        def want(idx, row):
            info = infos.get(idx)
            wa = info is not None and A.wants(info)
            assert wa == (info is not None and B.wants(info))
            return wa

        for fr in iter_segment(path, rows, segment=seg, rotate180=rotate180, want=want, allow_extra_rows=True):
            info = infos[fr.index]
            segmod.sharpness = old
            A.offer(info, fr.rgb)
            segmod.sharpness = new
            B.offer(info, fr.rgb)
            ra += A.pop_ready(info.c)
            rb += B.pop_ready(info.c)
    ra += A.flush()
    rb += B.flush()
    segmod.sharpness = NEW
    assert [r.segment.k for r in ra] == [r.segment.k for r in rb] == list(range(len(segments)))

    def key(r):
        return None if r.best is None else (int(r.best.info.ref["segment"]), int(r.best.info.ref["frame_in_segment"]))

    diffs = []
    for a, b in zip(ra, rb):
        if (key(a), a.sharpness_rank, a.n_candidates, a.reason) != (key(b), b.sharpness_rank, b.n_candidates, b.reason):
            diffs.append({"k": a.segment.k, "old": [key(a), a.sharpness_rank, a.n_candidates, a.reason],
                          "new": [key(b), b.sharpness_rank, b.n_candidates, b.reason]})
    ov, nv = np.array(old.vals), np.array(new.vals)
    rel = np.abs(nv - ov) / np.maximum(ov, 1e-12)
    # The existing send-all run (older code, same selector): does A reproduce its chosen frames?
    run = FIELD / "runs_segments_sendall" / s.session_id / "sidecars"
    prev = {r["segment"]["k"]: (r["frame"]["segment"], r["frame"]["frame_in_segment"]) if r.get("frame") else None
            for r in load_segment_records(run)}
    prev_diff = [a.segment.k for a in ra if prev.get(a.segment.k) != key(a)]
    from scipy.stats import spearmanr
    return {"session": s.session_id, "segments": len(segments), "covered": sum(r.best is not None for r in ra),
            "candidates_scored": old.n, "chosen_frame_differences": diffs, "n_differences": len(diffs),
            "old_vs_existing_sendall_run_differences": prev_diff,
            "sharpness_rel_diff": {"median": float(np.median(rel)), "p99": float(np.percentile(rel, 99)), "max": float(rel.max())},
            "spearman_old_new": float(spearmanr(ov, nv)[0]),
            "ms_per_call": {"old_float_rgb": 1000 * old.s / old.n, "new_u8_luma": 1000 * new.s / new.n},
            "elapsed_s": time.time() - t_start}


if __name__ == "__main__":
    cfg = Config()
    out = {}
    for d in sorted(p for p in (FIELD / "sessions").iterdir() if (p / "session.json").exists()):
        r = run(d, cfg)
        out[d.name[-4:]] = r
        print(json.dumps({k: v for k, v in r.items() if k != "chosen_frame_differences"}), flush=True)
    (OUT / "sharpness_identity.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
