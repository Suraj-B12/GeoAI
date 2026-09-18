#!/usr/bin/env bash
# Resume-safe calibration chain.
#
# Deliberately NO --fresh: re-running this script after a crash, a reboot or a
# kill picks up exactly where it stopped. Adding --fresh would discard every
# completed image, which is the one mistake that costs hours here.
#
# Each run checkpoints after every image, rotating the previous save to .bak,
# so a process killed mid-write loses one image rather than the whole run.
set -u
cd "$(dirname "$0")"
PY=venv/Scripts/python.exe

echo "=== Attain (Stage 2 only, 407 images) $(date +%H:%M:%S) ==="
$PY -u scripts/calibrate_confidence.py --dataset attain \
    --attain-from-run attain_baseline_v2_results.json --n 0 \
    --quantization-bits 4 --out calib_attain_407.json \
    >> eval_results/calib_attain_407.log 2>&1
echo "attain exit=$? $(date +%H:%M:%S)"

echo "=== RDD-India (Stage 1 bypassed, 250 images) $(date +%H:%M:%S) ==="
$PY -u scripts/calibrate_confidence.py --dataset rdd_india \
    --skip-stage1 --n 250 --quantization-bits 4 \
    --out calib_rdd_india_250.json \
    >> eval_results/calib_rdd_india_250.log 2>&1
echo "rdd exit=$? $(date +%H:%M:%S)"
