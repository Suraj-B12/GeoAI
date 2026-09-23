"""
Why is full resolution so slow? Timing + peak GPU memory per resolution cap,
each measured in isolation.

resolution_ab.py measured accuracy. Its timings were taken with the caps run
back-to-back after a full-resolution pass, so a cap that follows an
out-of-memory-sized pass could inherit its cost. This script times each cap on
its own, with the allocator cleared and peak-memory counters reset before every
pass, and records how much GPU memory each pass actually needed.

The hypothesis under test: full-resolution phone photos need more GPU memory
than the 24 GB card has. On Windows (WDDM) that does not raise an OOM error -
the driver quietly pages the overflow to system RAM over PCIe, and the pass
runs an order of magnitude slower. If true, peak memory at full resolution sits
at or above physical VRAM while every fast cap sits well below it.

It times all three worker stages (Stage 0 pavement filter, Stage 1, Stage 2),
so the result is the per-image cost of the actual production worker, not just
of the two classification stages.

Usage:
    venv/Scripts/python.exe scripts/resolution_timing.py
"""

from __future__ import annotations

import gc
import glob
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "ra", PROJECT_ROOT / "scripts" / "resolution_ab.py")
ra = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ra)

CAPS = [("1280", 1280 * 1280), ("1024", 1024 * 1024), ("768", 768 * 768)]
N_PHOTOS = 6          # timing is stable; accuracy is resolution_ab's job
N_NATIVE = 2          # full resolution is slow - two is enough to confirm


def clean():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def timed(clf, img) -> dict:
    """All three worker stages on one view, with peak memory."""
    clean()
    t0 = time.time()
    s0 = clf.predict_is_pavement(img)
    s1 = clf.predict_stage1(img)
    s2 = clf.predict_stage2(img)
    torch.cuda.synchronize()
    return {
        "stage0_s": round(s0["time_ms"] / 1000, 2),
        "stage1_s": round(s1["stage1_time_ms"] / 1000, 2),
        "stage2_s": round(s2["stage2_time_ms"] / 1000, 2),
        "total_s": round(time.time() - t0, 2),
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2),
    }


def main() -> int:
    os.environ.setdefault("PROMPTS_VERSION", "v2")
    os.environ["QUANTIZATION_BITS"] = "4"
    os.environ["DISABLE_ADAPTER"] = "true"
    from app.model import PavementClassifier
    clf = PavementClassifier(adapter_path=None, quantization_bits=4)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3

    # The large production photos are the case that matters.
    photos = []
    for f in sorted(glob.glob(str(ra.CACHE / "*.jpg"))):
        im = Image.open(f)
        if min(im.size) >= 1700:
            photos.append(f)
    photos = photos[:N_PHOTOS]
    print(f"{len(photos)} full-size photos (>=1700px), GPU {total_gb:.1f} GB\n")

    rows = []
    for i, f in enumerate(photos):
        img = Image.open(f).convert("RGB")
        # Rotate cap order per photo so no cap always runs first or last.
        order = CAPS[i % len(CAPS):] + CAPS[:i % len(CAPS)]
        for name, cap in order:
            r = timed(clf, ra.render(img, cap))
            r.update(photo=Path(f).stem[:8], cap=name,
                     tokens=(lambda d: (d[0] // 28) * (d[1] // 28))(ra.target_dims(*img.size, cap)))
            rows.append(r)
            print(f"  {r['photo']} {name:>6} tok={r['tokens']:<5} total={r['total_s']:>6.1f}s "
                  f"(S0 {r['stage0_s']:.1f} / S1 {r['stage1_s']:.1f} / S2 {r['stage2_s']:.1f})  "
                  f"peak alloc {r['peak_allocated_gb']:.1f} GB", flush=True)
        if i < N_NATIVE:
            r = timed(clf, img)
            r.update(photo=Path(f).stem[:8], cap="native",
                     tokens=(lambda d: (d[0] // 28) * (d[1] // 28))(ra.target_dims(*img.size, None)))
            rows.append(r)
            print(f"  {r['photo']} {'native':>6} tok={r['tokens']:<5} total={r['total_s']:>6.1f}s "
                  f"(S0 {r['stage0_s']:.1f} / S1 {r['stage1_s']:.1f} / S2 {r['stage2_s']:.1f})  "
                  f"peak alloc {r['peak_allocated_gb']:.1f} GB", flush=True)

    summary = {}
    print(f"\n{'cap':>7}{'n':>4}{'tokens':>8}{'median total':>14}{'median peak GB':>16}{'fits in VRAM':>14}")
    for name in ["native"] + [c for c, _ in CAPS]:
        rs = sorted((r for r in rows if r["cap"] == name), key=lambda r: r["total_s"])
        if not rs:
            continue
        med = rs[len(rs) // 2]
        peak = sorted(r["peak_reserved_gb"] for r in rs)[len(rs) // 2]
        summary[name] = {"n": len(rs), "tokens": med["tokens"],
                         "median_total_s": med["total_s"],
                         "median_peak_reserved_gb": peak,
                         "fits_in_vram": peak < total_gb * 0.95}
        print(f"{name:>7}{len(rs):>4}{med['tokens']:>8}{med['total_s']:>13.1f}s"
              f"{peak:>15.1f}{'yes' if summary[name]['fits_in_vram'] else 'NO':>14}")

    out = ra.EVAL_DIR / "resolution_timing.json"
    out.write_text(json.dumps({"gpu_total_gb": round(total_gb, 2), "quantization_bits": 4,
                               "summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nWritten: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
