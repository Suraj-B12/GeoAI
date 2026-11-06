# QLoRA / PostTrainBench-style plan — abandoned 2026-11-06

The study was switched, at the user's request on 2026-11-06, from gradient post-training
(QLoRA, modelled on PostTrainBench, Rank et al. 2026, arXiv:2603.08640) to weight-space
programming (Takehi, Teoh, Zhang, Khattab, MIT CSAIL, blog post 2 Nov 2026), which is the
paper the user meant.

What had run before the switch (complete list):
- No pre-registration had been written for the QLoRA plan.
- GPU: one memory/speed smoke test of `scripts/ft_train.py` (run name `smoke`): 12 WS_V1.0
  training images (picked by "at least two distress classes", no evaluation-based choice),
  3 optimizer steps (accum 4, lr 1e-4 warm-up), to measure memory. Peak 19.9 GB reserved,
  9.4 s/image (paging). Its adapter (`C:\GeoAI\finetune\runs\smoke\`) was never evaluated on
  anything and is not used by any later step.
- No dev evaluation and no test evaluation of any trained weights. Test evaluations: 0.
- CPU work kept and reused by the new study: the contamination check
  (`scripts/ft_contamination_check.py`, ORB + SIFT), the manifests and splits
  (`scripts/ft_prepare.py`, `scripts/ft_common.py`), the WS_V2.0 dev/test split, and the
  GPU-lock / atomic-write helpers.
- Code that stays in the repo unused: `scripts/ft_train.py`, `scripts/ft_eval.py`.
