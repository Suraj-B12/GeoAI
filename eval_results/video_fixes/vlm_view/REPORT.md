# What the VLM is shown: band crop vs whole frame, and the nearest-sharp frame

**Result.**
- **Band crop:** cropping the frame to the segment's band is not better than the whole frame. The default stays `vlm.image=frame`.
- **Nearest-sharp frame:** `segments.choice=nearest_sharp` shows the VLM each segment nearer and at finer resolution. On the 25 segments where the answer flips, it looks right more often than the default (12 vs 4, 9 unclear). That meets the pre-registered bar with no flip to spare, on a non-blind look, so the default stays `sharpest` until an expert checks the flips blind.
- **Bug found and fixed:** nearest_sharp could not run on any real drive (camera roll); fixed in GeoAI_Video `8f18fa2`.

Neither study has ground truth. The VLM's answer is compared between two views of the same segment, and flips are judged by one look of the analysing agent (not an expert).

## 1. Band crop vs whole frame (28 Oct)

**Sample and setup**
- **Segments:** 328 segments of batch_001: all 164 VLM-distress and 164 VLM-Normal, seed 20261028.
- **Arms:** both sent to the same server on the same day.
- **Reproducibility:** the frame arm reproduced batch_001's answers exactly (328/328, κ 1.0).

**Agreement, band vs frame**
- **Overall:** 84.1 %, κ 0.68.
- **Flips:** the band drops 42/164 of the frame's distress answers and adds 10/164.
- **Clustering lift:** band 2.77 vs frame 2.46.
- **Input size:** visual tokens are equal (median 1,302 vs 1,296).

**The 52 flips, judged by eye** (whole frame with the segment edges and the band marked; not blind)
- **Tally:** band right 21, frame right 17, unclear 14. The band's share of decided flips is 0.55 [0.40, 0.70].
- **Out-of-segment damage:** 5 flips were damage outside the segment.
- **False positives:** the band gave 2 footpath false positives.

Decision: no evidence that the band is better, so `image=frame` stays.

## 2. Nearest-sharp frame vs raw-sharpest frame (6 Nov)

**Why.** Raw Tenengrad grows with distance: the same road in fewer, denser rows. So `sharpest` prefers far frames, where the segment has fewer rows at the VLM's input. `nearest_sharp` instead picks the nearest candidate whose fair sharpness is at least 0.9 × the best. Fair sharpness is measured on a common 4 mm ground grid.

**Bug found first**
- **Failure:** the option failed on all three S21 drives with "fair sharpness strip leaves the image".
- **Cause:** the phone is mounted with 1.2–4.4° of roll, so the bottom image row is slanted on the road. `near_m` is measured at the centre column, so the nearest candidate's strip started below the image on one side.
- **Fix** (`8f18fa2`): when the unshifted strip does not fit, it starts `offset_m` into the segment for every candidate: f428 0.078, 293d 0.111, d47c 0.034 m.
- **Fairness:** a level camera keeps offset 0. The far/near score ratio stays within 0.1 % of a level camera's (full resolution 0.9916–0.9924 vs 0.9924).
- **Tests:** `test_fair_sharpness_with_camera_roll`, `test_fair_sharpness_offset_is_zero_without_roll`.
- **Earlier note was wrong:** the 28 Oct note "re-extraction done" was not true; the folder was empty.

**Arms (drive d47c, 528 segments, 2.64 km)**
- **sharpest:** the 27 Oct send-all run, answered in batch_001 (default profile). A control re-extraction with today's code picks the same frame in 528/528 segments; 25/25 sampled PNGs are pixel-identical, and the PNG hashes differ only in metadata.
- **nearest:** the same drive re-extracted with `nearest_sharp` (`8f18fa2`). Only the 225 segments whose frame changes were sent (`ab_nearest_001`, `--profile video`, which leaves answers unchanged: 60/60 in `vlm_speed`). The rest keep batch_001's answers.
- **Decision rule:** written at 10:17:28 UTC, before any nearest answer existed (`nearest_decision_rule.json`): nearest becomes the default only if the Wilson 95 % CI of its share of decided flips lies wholly above 0.5.

**Geometry of the 225 changed segments** (median [95 % CI])

| | sharpest | nearest |
|---|---|---|
| segment near edge ahead of the camera | 2.10 m [2.05, 2.17] | 1.63 m [1.61, 1.68] |
| rows of the segment at the VLM input | 395 [383, 406] | 510 [495, 517] (×1.21 [1.18, 1.26]) |
| mm per pixel at the segment centre (VLM input) | 9.7 [9.4, 10.1] | 7.1 [6.9, 7.4] (×1.27 finer) |
| fair sharpness (same ground grid) | 0.070 | 0.090 (ratio ×1.19 [1.13, 1.26]; ≥ 1 on 79 %) |

On 31 of the 225, nearest picked a farther, coarser frame than raw sharpest did. Over all 528 segments, raw sharpest is still the choice in 56.8 %. The extraction-time difference (656 vs 576 s) is not reported as a result, because both runs overlapped CPU jobs.

**Answers on the 225**

| | nearest distress | nearest Normal |
|---|---|---|
| sharpest distress | 31 | 16 |
| sharpest Normal | 9 | 169 |

- **Agreement:** 200/225 = 88.9 %, κ 0.64.
- **Whole drive:** distressed segments 102 → 95 of 528. The clustering lift (run-on of positives into the next segment) is 2.20 → 2.01.
- **VLM time:** median 4.2 s per image with `--profile video`.

**The 25 flips, judged by eye** (`a06b_judge_sheets_nearest.py` → `judge_nearest/`, verdicts in `judgements_nearest.py`)

Method: both whole frames side by side with the segment edges marked, one look each. The question was whether the segment's own carriageway shows IRC:82 distress. The look was not blind: each arm and its answer are named on the sheet.

| Flips | nearest right | sharpest right | unclear | nearest share of decided [Wilson 95 %] |
|---|---|---|---|---|
| all 25 | 12 | 4 | 9 | 0.75 [0.505, 0.898] |
| sharpest Distress → nearest Normal (16) | 8 | 2 | 6 | 0.80 [0.49, 0.94] |
| sharpest Normal → nearest Distress (9) | 4 | 2 | 3 | 0.67 [0.30, 0.90] |

**What decided the verdicts**
- **Nearest right:** mostly sharpest's "Bleeding + Potholes" on clean, shaded asphalt, stains or manhole covers, which the nearer frame answers Normal. Also cracks the nearer frame resolves: 4 flips (#6, 7, 17, 21).
- **Sharpest right:** a longitudinal crack the nearer frame missed (#23). One clearly damaged segment whose nearer frame is badly defocused, because the S21 focus does not reach the nearest rows (#24). Two false "potholes" in the nearer frame (#8, #14).

**Decision**
- **Rule met, zero margin:** losing a single "nearest" verdict to "unclear" would fail the rule (`results.json` `nearest_flip_judgements.margin_flips`).
- **Why the default stays:** the judge was not blind, and one failure mode (defocus at the nearest distance) was seen. The default stays `segments.choice=sharpest`, and `nearest_sharp` stays opt-in.
- **Deviation:** this departs from the letter of the rule's "otherwise" clause. It is reported in the paper (§8.6).
- **Next step:** a blind expert check of the 25 flips (sheets rebuilt with the arms shuffled and the answers hidden), with the same rule applied.

## Files

- **Scripts:** `a01_sample_and_export.py`, `a02b_reextract.py` (GPU, through `../gpu_job.py`), `a03_band_vs_frame.py`, `a04_nearest_vs_sharpest.py` (`geometry` | `answers`), `a04b_send_nearest.py` (production `/classify` through `gpu_job.py`; the API key is read from the key file, never printed or logged), `a05_figures.py`, `a06_judge_sheets.py`, `a06b_judge_sheets_nearest.py`, `a07_results.py`, `judgements.py`, `judgements_nearest.py`, `debug/fair_strip_check.py`.
- **Results:** `band_vs_frame.json`, `flips.csv`, `judgements.csv`, `nearest_geometry.json`, `nearest_segments.csv`, `changed_ids.txt`, `nearest_vs_sharpest.json`, `judgements_nearest.csv`, `nearest_decision_rule.json` and `results.json` (everything assembled).
- **Figures:** `fig1_agreement_flips.png`, `fig2_clustering.png`, `fig3_resolution_time.png` (paper Figure 8.9d).
- **Kept in `C:\GeoAI\analysis\fixes\vlm_view\` only:** the judgement sheets (`judge/`, `judge_nearest/`) and the logs.
- **Batches in GeoAI_Video `field/batches/`:** `ab_frame_001`, `ab_band_001`, `ab_nearest_001`.
- **Runs:** `field/runs_eval/sharpest_head` and `field/runs_eval/nearest`.
