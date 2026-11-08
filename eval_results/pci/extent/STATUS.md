# STATUS — Attain -> IRC:82-2023 extent calibration (workstream "extent")

Owner files: app/pci_extent.py, scripts/pci_attain_extent_calibration.py, scripts/tests/test_pci_extent.py,
eval_results/pci/extent/*. Nothing else is edited.

## Steps
- [x] 0. Read context (IRC82_2023.md, pci_irc2023.py, pci_core 2023 section, probe report split, IRC text §5, §7, App-1)
- [x] 1. MAPPING.md
- [x] 2. Ground truth per frame (ROI choice + sensitivity)
- [x] 3. PREREG.md (before any test number)
- [x] 4. Dev fit + calibration JSON
- [x] 5. Test evaluation + grouped CV
- [x] 6. app/pci_extent.py estimate_extents + tests
- [x] 7. Transfer check (Bengaluru video_probe_scores) + 8 sample stretches
- [x] 8. REPORT.md

## Log
- 2026-11-09 started.
- gt done (attain_gt.json, roi_check/ viewed; 1920 trapezoid widened to x 0.05-0.92 at top BEFORE prereg). PREREG.md hashed 2026-11-08T18:38Z UTC. fit done (dev only): cracking bins5, ravelling bins5, rav+weath bins3, patching bins5 (nearly flat), potholes bins3 (nearly flat).
- NEXT: test stage (once), then cv, tests, transfer, report.
- TEST RUN DONE (once) -> test_results.json. Verdicts: cracking interval-only (C1 fail CI[-1.11,0.74]); ravelling FAILS (C4 cov 22/28=0.786<0.80; C1,C3 fail); patching interval-only; potholes interval-only; rav+weathering (sensitivity) calibrated (C1,C3,C4 pass) but bias -3.9 pp. TEST now spent.
- cv done (cv_results.json); tests 8/8 pass; transfer done (transfer.json; KS written into calibration.transfer; viewpoint_check.jpg for the scooter-vs-car note); REPORT.md could NOT be written (the harness blocks report files from subagents) - the full report was returned to the coordinator as text. Workstream COMPLETE 2026-11-09. Nothing committed; no other files edited.
