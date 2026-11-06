# Weight-space programming study (paper §6.11) — STATUS

Updated: 2026-11-06 ~15:30 UTC — **COMPLETE.** Nothing is running. Nothing is committed (the main session commits).

## Result (eval_results/finetune/wsp_report.md / .json)
- Verdict under the pre-registered rule: **not better.** Production is unchanged.
- Final edit `e5_h26.17_ro_c0.75`: remove 3/4 of attention head 26.17's write along the readout direction
  (128 columns of layers.26.self_attn.o_proj, 458,743 entries), baked into bf16.
- Held-out test (547 frames, 23 clean):
  | Model | Recall | Specificity | MCC | AUROC |
  |---|---:|---:|---:|---:|
  | base | 51.7% | 23/23 | 0.208 | 0.957 |
  | recalibrated base (+1.56) | 77.1% | 23/23 | 0.352 | 0.957 |
  | edited | 74.8% | 23/23 | 0.333 | 0.955 |
  | random control | 52.1% | 23/23 | 0.209 | 0.956 |
- Edited − recalibrated MCC −0.019 [−0.036, −0.006]. Bengaluru Distress share 19.7% → 36.7%.

## Counts (for honesty)
- Forward passes used while editing: 32 of 50 (`wsp_ledger.jsonl`, counted rows).
- Test evaluations: 1 (WSP run `final_20261106T191810_bfc0`); 0 for the abandoned QLoRA plan (ABANDONED.md).

## Where things are
- Code: `scripts/wsp_*.py` (core, server, client, analysis, edits, queue, freeze, apply_edit, final_eval, report, figures),
  `scripts/ft_common.py`, `scripts/ft_contamination_check.py`, `scripts/ft_prepare.py`;
  tests `scripts/tests/test_wsp_core.py` (6 pass).
- Records: `wsp_preregistration.json` + addenda 1–3, `wsp_work_notes.md` (steps 0–8), `wsp_dev_runs.json`,
  `wsp_ledger.jsonl`, `wsp_report.*`, `wsp_final_edit/` (frozen manifest + edit definitions), contamination files.
- Large artifacts (not in git): `C:\GeoAI\wsp\` (captures, analysis JSON, edits, baked matrices in `edits\final`
  and `edits\random_baked`, final raw results `final\final_raw.json`).
- Figures: `eval_results/plots/paper/fig_6_11{a,b,c,d}_*.png` (`scripts/wsp_figures.py`, no inference).
- Paper: §6.11, §7.12, revision note 2026-11-06 (b).

## To redo the scoring or the figures (no GPU)
- `venv/Scripts/python.exe scripts/wsp_report.py`
- `venv/Scripts/python.exe scripts/wsp_figures.py`
