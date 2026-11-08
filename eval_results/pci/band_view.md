# Probe on the segment's own 5 m strip vs the full frame (2026-11-09)

Question: does scoring only the image rows of a segment's own 5 m strip (`record.frame.rows_of_segment`, full width,
+10 % margin; `scripts/video_probe_scores.py --view band`, probe_fp `84b543cd8ccc3c7c`) localise potholes better
than the full frame (probe_fp `7e47e0ab123dddb5`)? Both were run on all 7,050 segment images (GPU, 1.8 s/image,
0 errors). There are no pothole labels by experts, so the independent check is the scooter's accelerometer: a jolt
above the 90th percentile of all segments' jolt (peak vertical acceleration / median RMS of its drive).

| Pothole P(yes) from | AUC for "jolt at the same segment" | Precision of P ≥ 0.5 for a jolt | Segments with P ≥ 0.5 |
|---|---|---|---|
| Full frame | 0.711 | 0.325 | 609 |
| 5 m strip | 0.697 | 0.308 | 419 |
| Mean logit of both | 0.712 | 0.319 | 298 |
| Max of both | 0.712 | 0.306 | 710 |

AUC falls with distance for both (full 0.71 / 0.64 / 0.59 at offsets 0 / +1 / +2 segments; strip 0.70 / 0.61 /
0.56). **Conclusion: the strip view does not localise potholes better than the full frame, and combining the two adds
nothing (0.712 vs 0.711). Not adopted.** The scores stay stored (`research.video_probe_scores`) so the survey
calibration (`scripts/pci_survey_compare.py`) can test them against ground truth. 7,024 segments with both scores
and IMU data. Caveat: a jolt is an imperfect reference (riders steer around potholes; speed breakers jolt too).
