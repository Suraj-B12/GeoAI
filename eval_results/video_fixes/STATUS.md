# Drive-video improvement studies — status: DONE (2026-11-06, 11:10 UTC)

Study folders: `C:\GeoAI\analysis\fixes\` (scripts, results, figures, logs, judgement sheets). Small outputs are
copied here (`eval_results/video_fixes/<study>/`). Nothing is running; the GPU lock is not held by these studies.

## Rules for anything that runs (since 2026-11-06)
- **GPU jobs** run detached through `gpu_job.py` (copy here; original `C:\GeoAI\analysis\fixes\gpu_job.py`).
  - It takes `C:\GeoAI\gpu.lock` itself. owner.txt holds its PID, purpose and UTC start, then child_pid and child_created.
  - It releases the lock in `finally`.
  - It takes over a lock whose owner PID is dead, after killing that owner's recorded child tree. On this machine, children can escape a kill-on-close Job Object.
  - Never `rmdir` the lock while its owner PID is alive.
- **Result files** are written atomically (`atomic_io.py`: temp file + rename). Run folders are written as `<name>.partial`, then renamed.
- **Production server:** the VLM server (port 8000, start.ps1 watchdog) stays up; studies never restart it.

## Results

| Study | Folder | Result | Decision |
|---|---|---|---|
| VLM speed (`profile=video`) | `vlm_speed/` | Normal 13.72→3.76 s, Distress 24.41→14.43 s, 60/60 identical | adopted (earlier) |
| Gate on its own thread | `gate_speed/` | 937→722 s per 4.06 km (231→178 s/km), byte-identical outputs | adopted (GeoAI_Video `ce82205`) |
| Probe-only gate (LOSO) | `gate_speed/` | held-out recall 87.8 % [78.5, 95.2] at 59.4 % pass (< 0.90); 88 s/km; ~13 min/km less VLM | not adopted (defaults kept) |
| uint8 luma sharpness | `gate_speed/` | ~2.9× per call, 2/813 different frames | not adopted (pre-registered identity rule) |
| Band crop vs frame | `vlm_view/` | 84.1 % agreement, κ 0.68; flips band 21 / frame 17 / unclear 14 | not adopted (`image=frame`) |
| nearest_sharp | `vlm_view/` | failed on every drive (camera roll) → fixed in GeoAI_Video `8f18fa2`. d47c: 225/528 changed; 2.10→1.63 m, 9.7→7.1 mm/px; agreement 200/225 (κ 0.64); flips nearest 12 / sharpest 4 / unclear 9, 0.75 [0.505, 0.898] | rule met with **zero** margin (non-blind look) → default unchanged; blind expert check first |

## Corrections to the 28 Oct notes
- **"g03 stuck":** wrong. The GPU runs had completed on 10-28 at 15:14–16:17Z, under clean conditions. They were not re-run.
- **"nearest re-extraction done":** wrong. `runs_eval/nearest` was empty, because the option crashed on every drive (camera roll).
- **"luma sharpness safe to adopt":** wrong under the study's own pre-registered rule.

## Commits
- GeoAI_Video `8f18fa2`: roll fix + tests + METRICS §9 timing/gate/luma rows.
- GeoAI_Video `d016de7`: METRICS/PROGRESS nearest result. Both pushed. CPU suite: 256 passed, 12 skipped.
- Capstone: uncommitted (the main session commits). Changes:
  - `eval_results/video_fixes/` (this file, `gpu_job.py`, `atomic_io.py`, `gate_speed/`, `vlm_view/`);
  - `eval_results/plots/paper/fig_v9b_gate_heldout.png`, `fig_v9c_extraction_time.png`, `fig_v9d_view_resolution.png`;
  - `paper/baseline_evaluation.md`: §8.6, Figures 8.9a–d, and the 2026-11-06 revision note.

## Open (owner / main session)
1. Save `gate_speed/REPORT.md` and `vlm_view/REPORT.md`, both here and in `C:\GeoAI\analysis\fixes\<study>\`. The texts are in the video agent's final report; the harness blocks subagents from creating report files.
2. Blind expert check of the 25 nearest-vs-sharpest flips. The sheets are in `C:\GeoAI\analysis\fixes\vlm_view\judge_nearest\` and name the arms; for a blind check, rebuild them with the arms shuffled and the answers hidden. Then apply `vlm_view/nearest_decision_rule.json` to the expert verdicts.
3. Test fixture gap: `tests/test_probe.py::test_flags_held_out_distress_frames_more_than_clean_ones` needs the sample folder `08_no_distress_annotated`, which is missing on this machine. It is a GPU-only test, so the CPU run skips it.

## Re-running any step (idempotent)
- **Environment:** CPU steps use `CUDA_VISIBLE_DEVICES=-1` and the GeoAI_Video venv (`pipeline\.venv`). Figures (`g06`, `a05`) use the Capstone venv, which has matplotlib.
- **gate_speed:** `g04_compare.py` → `g05_vlm_time.py` → `g06_figures.py` → `g07_results.py`.
- **vlm_view, GPU steps through `gpu_job.py`:** `a02b_reextract.py` (GPU) → `a04_nearest_vs_sharpest.py geometry` → `a04b_send_nearest.py` (GPU, server `/classify`) → `a04 … answers` → `a06b_judge_sheets_nearest.py` → `a05_figures.py` → `a07_results.py`.
