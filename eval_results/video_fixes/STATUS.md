# Drive-video improvement studies — status (saved 2026-10-28, 20:10 IST)

Copies of the study folders in `C:\GeoAI\analysis\` (scripts, results JSON/CSV, figures; large raw
batches and image sheets stay there only). The weekly usage limit stopped the work; this file is where
to resume. Paper §8.6 and GeoAI_Video `docs/research/METRICS.md` §9 are still placeholders for these numbers.

| Study | Folder | State | Result so far |
|---|---|---|---|
| Discriminator (flaw finder) | `discriminator/` | done | `FINDINGS.json`: speed, gate, best-frame and crop flaws that the studies below test |
| VLM speed (`profile=video`) | `vlm_speed/` | **done, committed** (Capstone e87eb83, GeoAI_Video ce82205) | Normal 13.72→3.76 s, Distress 24.41→14.43 s, 60/60 answers identical; runner 14.4→5.2 s/image. `REPORT.md` |
| Band crop vs whole frame | `vlm_view/` | analysis done, report not written | 328 segments (164 VLM-distress + 164 Normal, seed 20261028). Agreement 84.1 %, κ 0.68. Band drops 42/164 frame-distress, adds 10/164 normals. One-look visual judgement of the 52 flips (not ground truth): band right 21, frame 17, unclear 14 (band share 0.55, CI 0.40–0.70) → **no evidence band is better; keep `image=frame`**. Band gave 2 footpath false positives; 5 flips were damage outside the segment. Frame re-run reproduced batch_001 exactly (328/328) |
| Nearest-sharp frame choice | `vlm_view/` (`a04_nearest_vs_sharpest.py`) | re-extraction done in `GeoAI_Video\field\runs_eval\nearest`; **analysis not run** | — |
| Gate held-out (LOSO) | `gate_speed/out/loso.json` | done, report not written | Pooled: default gate recall 92.1 % at 90.9 % pass. Probe-only (V1) held-out: recall 87.8 % [78.5, 95.2] at 59.4 % pass — below the 0.90 recall target, so the in-sample 90.2 %/61.4 % does **not** hold out cleanly. Pre-registered decision: no candidate (V2–V4) beats V1. Per session d47c is the weak one (default recall 87.3 %) |
| Luma sharpness speed-up | `gate_speed/out/sharpness_identity.json` | done | Tenengrad on uint8 luma vs float RGB: 31–36 → 10–13 ms per call (~2.9×); same chosen frame on 811/813 segments (2 differ in d47c), Spearman 0.99997 |
| Extraction timing on GPU | `gate_speed/g03_gpu_runs.sh` | **stuck** | The script waits on `C:\GeoAI\gpu.lock`, which the same agent took (owner.txt "gate_speed study … 14:12:32Z"). Fix: when nothing else uses the GPU, `rmdir /s C:\GeoAI\gpu.lock`, rerun `g03_gpu_runs.sh`, then `g04_compare.py` … `g07_results.py` |

GeoAI_Video full suite on GPU (in `gate_speed/out/gpu_runs.log`): 235 passed, 2 skipped, 1 failed —
`tests/test_probe.py::test_flags_held_out_distress_frames_more_than_clean_ones` (KeyError
`08_no_distress_annotated`: that sample folder is missing on this machine; a fixture gap, not a code bug).

## To resume
1. Release the stale GPU lock (above) only after checking `nvidia-smi` shows nothing but the VLM server idle.
2. Run the nearest-sharp analysis (`a04`) and the extraction timing (`g03`…`g07`).
3. Write `REPORT.md` for `vlm_view/` and `gate_speed/` from the numbers above, copy figures to
   `eval_results/plots/paper/`, fill paper §8.6 and METRICS.md §9, commit both repos (no Co-Authored-By).
4. Decide adoption: keep `image=frame` (band not better); `nearest_sharp` pending; luma sharpness is
   safe to adopt (same frames, ~2.9× faster); gate stays at defaults until more sessions exist.
