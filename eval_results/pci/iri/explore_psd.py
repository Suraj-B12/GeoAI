"""Exploratory (2026-11-16): vertical-acceleration spectra of all drives by speed bin (ESTIMATE.md update).
Usage: explore_psd.py OUT.npz ; then explore_psd_look.py OUT.npz"""
import os, sys, glob
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, r"C:\Users\PRO-LAB-3\Documents\Capstone"); sys.path.insert(0, r"C:\Users\PRO-LAB-3\Documents\Capstone\scripts")
import numpy as np
from scipy import signal

from app import pci_iri as P
import iri_from_imu as I
out = sys.argv[1]
FS, N = 200.0, 1024
W, S, V, G = [], [], [], []
for si, d in enumerate(sorted(glob.glob(r"C:\GeoAI\inbox\*"))):
    d = Path(d)
    if not (d / "imu.csv").exists() or not (d / "gnss_fixes.csv").exists(): continue
    try:
        imu = P.read_imu_csv(d / "imu.csv"); gn = I.read_gnss(d / "gnss_fixes.csv")
        acc = imu["acc"]; t0 = int(acc["t_ns"][0]); ta = (acc["t_ns"] - t0) / 1e9; a = acc["xyz"]
        grav = imu.get("gravity")
        g = P._interp_vec(ta, (grav["t_ns"] - t0) / 1e9, grav["xyz"]) if grav is not None and len(grav["t_ns"]) > 10 else P._lowpass_gravity(ta, a)
        gh = P._unit(g); av = np.einsum("ij,ij->i", a, gh)
        grid = np.arange(0, ta[-1], 1 / FS); x = np.interp(grid, ta, av)
        j = np.clip(np.searchsorted(ta, grid), 1, len(ta) - 1); valid = (ta[j] - ta[j - 1]) <= 0.05
        tg = (np.asarray(gn["t_ns"]) - t0) / 1e9; sp = np.interp(grid, tg, np.asarray(gn["speed_mps"], float))
        # pitch-ish rate: gyro perpendicular to gravity, magnitude
        gy = imu.get("gyro"); rot = None
        if gy is not None:
            w = P._interp_vec(grid, (gy["t_ns"] - t0) / 1e9, gy["xyz"]); ghg = P._unit(P._interp_vec(grid, ta, gh))
            wp = w - np.einsum("ij,ij->i", w, ghg)[:, None] * ghg
        for s in range(0, len(grid) - N, N // 2):
            if not valid[s:s + N].all(): continue
            v = sp[s:s + N]
            if v.min() < 2.0 or v.std() > 0.15 * v.mean(): continue
            f, p = signal.periodogram(x[s:s + N] - x[s:s + N].mean(), FS, window="hann")
            W.append(p); V.append(v.mean()); S.append(si)
            if gy is not None:
                pg = sum(signal.periodogram(wp[s:s + N, k] - wp[s:s + N, k].mean(), FS, window="hann")[1] for k in range(3)); G.append(pg)
        print(d.name, len(W), flush=True)
    except Exception as e:
        print(d.name, "ERR", e, flush=True)
np.savez_compressed(out, f=f, psd=np.array(W, np.float32), v=np.array(V), s=np.array(S), gyro=np.array(G, np.float32))
