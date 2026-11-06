# Distress gate held out by drive, extraction time, and 8-bit sharpness

**Result.**
- **Gate threads:** extracting the drive-video segments got 1.30× faster with byte-identical outputs: 231 → 178 s per km of road over 4.06 km. This is now in GeoAI_Video (`ce82205`).
- **Probe-only gate:** it would make extraction 2.6× faster (88 s/km) and save about 13 min of VLM time per km. Held out by drive, though, it keeps only 87.8 % [78.5, 95.2] of the segments the VLM calls distressed, below the 90 % target. It stays opt-in and the default thresholds stay.
- **8-bit luma sharpness:** about 2.9× faster per call, but it picks a different frame on 2 of 813 segments, which breaks the pre-registered identity rule. Not adopted.

Pre-registered protocol: `PREREGISTRATION.md` (SHA-256 recorded in `results.json`), written before any held-out number was computed. Code under test:
- **before:** GeoAI_Video `3c72086`;
- **after:** `3c72086` + this study's changes. The same changes were committed in `ce82205`, with every new behaviour opt-in.

## 1. Gate held out by drive (leave-one-session-out)

**Data and reference**
- **Segments:** the 813 send-all segments of batch_001 (f428 230, 293d 55, d47c 528).
- **Reference:** the VLM's answer (distress = any valid non-Normal label). This is not ground truth.

**Method**
- **Calibration:** on two drives with the shipping exact solver at target recall 0.90, applied to the third.
- **Uncertainty:** 95 % CIs from a 20-segment block bootstrap with re-calibration, B = 2000.

| Rule (pooled over the 3 held-out drives) | Recall [95 % CI] | Pass rate [95 % CI] |
|---|---|---|
| V0 default thresholds (not calibrated) | 0.921 [0.874, 0.962] | 0.909 [0.888, 0.929] |
| V1 probe_frac alone (pre-registered default candidate) | 0.878 [0.785, 0.952] | 0.594 [0.469, 0.716] |
| V2 probe_max_logit alone | 0.902 [0.820, 0.962] | 0.633 [0.547, 0.736] |
| V3 probe_frac OR probe_max_logit | 0.854 [0.703, 0.930] | 0.594 [0.414, 0.688] |
| V4 any-of probe_frac / evidence / crack_mm | 0.878 [0.678, 0.944] | 0.594 [0.433, 0.706] |

**Per drive**

| Drive | V0 default (recall / pass) | V1 calibrated (recall / pass) |
|---|---|---|
| f428 | 1.000 / 0.957 | 0.878 / 0.504 |
| 293d | 1.000 / 0.945 | 1.000 / 0.673 |
| d47c | 0.873 / 0.884 | 0.853 / 0.625 |

**Notes**
- **Signal quality:** probe_frac and probe_max_logit both reach AUROC ≈ 0.80 on f428. The bank evidence and the crack length are near chance (0.57 and 0.55).
- **Pre-registered choice rule:** no candidate (V2–V4) replaces V1. The rule needs a pass rate at least 3 points lower, with a CI wholly below 0, at equal recall.
- **V1 itself:** it misses the 0.90 recall target held out. The in-sample calibration on all 813 segments (recall 0.902 at pass 0.614) does not generalise cleanly to a new drive.
- **Decision:** the default thresholds stay. The calibrated gate ships only as the opt-in config `calibration/segments_probe_gate.json` until more drives exist. Figure: `fig1_gate_heldout.png`.

## 2. Extraction time (RTX A5000, 2026-10-28 15:14–16:17 UTC)

**Run conditions**
- **Lock:** all runs were made by `g03_gpu_runs.sh` under the GPU lock, in interleaved order (before, after, before, after, after-default).
- **Disk cache:** the session videos were pre-read into the cache first.
- **VLM server:** resident and idle (6.95 GB, 0 % at the start). The VLM A/B batches ended at 10:35 UTC, and no classification was written to the database that day.
- **Repetitions:** the two repetitions of each arm agree within 0.5 s.

| Variant | Code | Gate | 4.06 km total | s per km | Outputs |
|---|---|---|---|---|---|
| before (×2) | `3c72086` | default (bank + crack filter, serial) | 937.1 / 937.6 s | 230.8 | reference |
| after-default (×1) | + this study | default, gate on its own thread | 721.5 s | 177.7 | byte-identical to before (every record, PNG and sidecar; only report.json's config and timing keys differ) |
| after (×2) | + this study | probe-only calibrated (no bank, no crack filter) | 357.0 / 357.6 s | 88.0 | different decisions by design (pass 50–67 %) |

**Where the time goes, per drive (before → after):**
- f428: 285 → 99 s
- 293d: 85 → 44 s
- d47c: 567 → 214 s

The bank pass (275 s over the three drives) disappears when no enabled gate signal needs it. Figure: `fig2_extraction_time.png`.

## 3. VLM time a gate would save

These numbers replay batch_001's measured per-image VLM times (before `profile=video`: Normal 13.0 s median, Distress 21.7 s):
- **Held-out probe gate:** 13.1 min per km less than the default gate, and 17.4 min per km less than send-all.
- **Recall cost:** that gate keeps 146 of 164 VLM positives, against 151 of 164 for the default.

With `profile=video` (Normal 3.8 s) the saving shrinks roughly in proportion.

## 4. Sharpness on 8-bit luma (`g02_sharpness_identity.py`)

Tenengrad on uint8 BT.601 luma instead of float RGB, on all 813 segments:
- **Speed:** 31–36 → 10–13 ms per call.
- **Agreement:** Spearman 0.99997.
- **Differences:** a different chosen frame on 2 segments, both in d47c, where two candidates were within rounding of each other.

The pre-registered rule was "adopt only if the chosen frame is identical in every segment", so it is not adopted; HEAD's `segments.sharpness` docstring records why. The saving would have been about 60 s of CPU per 2.6 km drive, against about 1.4 h of VLM time for the same drive.

## Tests

- **After tree on the GPU (2026-10-28):** 235 passed, 2 skipped, 1 failed. The failure is `test_probe.py::test_flags_held_out_distress_frames_more_than_clean_ones`: KeyError `08_no_distress_annotated`, because that sample folder is missing on this machine. This is a fixture gap, not a code fault.
- **GeoAI_Video HEAD on CPU (2026-11-06):** 256 passed, 12 skipped.

## Files

- **Scripts:** `g01_loso.py` … `g07_results.py`, `PREREGISTRATION.md`.
- **Results:** `out/loso.json`, `out/sharpness_identity.json`, `out/compare.json`, `out/vlm_time.json` and `results.json` (everything assembled).
- **Figures:** `fig1_gate_heldout.png` (relabelled 2026-11-06: legends instead of overlapping labels; "probe_frac (ships)" corrected), `fig2_extraction_time.png`.
- **Run log:** `out/gpu_runs.log`.

Reproduce with `CUDA_VISIBLE_DEVICES=-1`, in this order:
- `g01`, `g02`, `g04`, `g05` and `g07` with the GeoAI_Video venv;
- `g06` with the Capstone venv, which has matplotlib.

`g03` needs the GPU lock and takes about 65 minutes. Result files are written atomically (`../atomic_io.py`).
