# Stage 2: free-form list (production) vs per-type probing

Source `stage2_probe_raw_attain.json`. 847 Attain WS_V2.0 images; DEV 300 (55 without distress), TEST 547 (23 without distress). Split by blocks of 20 consecutive frames, seed 20260923. Variant, thresholds and calibration chosen on DEV only; chosen variant: `min:name`. Intervals: cluster (block) bootstrap, B=2000.

## Headline (TEST, five classes both methods can express)

| metric | production v0 | probe | constant prior | delta probe - v0 [95% CI] | p |
|---|---:|---:|---:|---:|---:|
| macro MCC | 0.111 | 0.223 | 0.000 | +0.112 [+0.033, +0.183] | 0.002 |
| macro F1 | 0.264 | 0.545 | 0.346 | +0.281 [+0.219, +0.332] | 0.000 |
| macro balanced acc. | 0.552 | 0.590 | 0.500 | +0.037 [-0.000, +0.074] | 0.051 |
| mean Jaccard (per image) | 0.376 | 0.623 | 0.621 | +0.246 [+0.195, +0.297] | 0.000 |
| Hamming loss (lower better) | 0.294 | 0.254 | 0.209 | -0.041 [-0.081, +0.003] | 0.062 |
| exact label set | 9.7% | 25.2% | 27.2% | McNemar 118 vs 33 | 0.000 |

Probe macro AUROC (threshold-free): 0.644 [0.573, 0.707]. Constant prior predicts ['Alligator crack', 'Linear crack'] on every image.

## Per class (TEST)

| class | prevalence | v0 P / R | v0 MCC | probe P / R | probe MCC [95% CI] | probe AUROC | delta MCC [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|
| Linear crack | 83.9% | 0.89 / 0.82 | 0.257 | 0.87 / 0.97 | 0.358 [0.147, 0.529] | 0.672 | +0.101 [-0.044, +0.229] |
| Alligator crack | 69.1% | 0.89 / 0.08 | 0.114 | 0.74 / 0.97 | 0.336 [0.196, 0.456] | 0.703 | +0.222 [+0.069, +0.371] |
| Pothole | 18.5% | 0.43 / 0.25 | 0.219 | 0.37 / 0.40 | 0.234 [0.032, 0.412] | 0.650 | +0.016 [-0.067, +0.112] |
| Raveling | 12.8% | 0.00 / 0.00 | 0.000 | 0.09 / 0.14 | -0.063 [-0.164, 0.043] | 0.539 | -0.063 [-0.162, +0.053] |
| Weathering | 26.3% | 0.00 / 0.00 | -0.036 | 0.40 / 0.58 | 0.249 [-0.040, 0.472] | 0.659 | +0.285 [-0.021, +0.508] |
| Patch and utility cut | 19.4% | 0.00 / 0.00 | 0.000 | 0.40 / 0.32 | 0.224 [0.035, 0.442] | 0.616 | (not in headline) |
| Block crack | 3.3% | 0.00 / 0.00 | 0.000 | 0.07 / 0.72 | 0.138 [-0.052, 0.285] | 0.774 | (not in headline) |

## Variant ablation (TEST, each with its own DEV thresholds; selection was on DEV)

| variant | dev macro AUROC | test macro AUROC | test macro MCC | test macro F1 | Patch AUROC | Block AUROC |
|---|---:|---:|---:|---:|---:|---:|
| min:def | 0.683 | 0.643 | 0.227 | 0.563 | 0.609 | 0.786 |
| min:name (chosen) | 0.716 | 0.644 | 0.223 | 0.545 | 0.616 | 0.774 |
| tax:def | 0.653 | 0.672 | 0.168 | 0.500 | 0.565 | 0.837 |

## Output shape (TEST)

Distinct label sets: v0 8, probe 35.

v0 raw lists: `Longitudinal Cracking, Transverse Cracking` x332; `N/A` x71; `Normal` x29; `Longitudinal Cracking, Potholes` x26; `Potholes, Longitudinal Cracking` x19


## Stage 1 and end to end (TEST)

524 distress / 23 no-distress images. only the no-distress images are negatives; there are few of them, so specificity and AUROC here have wide uncertainty.

| detector | recall | specificity | MCC | AUROC |
|---|---:|---:|---:|---:|
| production_stage1 | 51.7% | 100.0% | 0.208 | 0.957 |
| probe_max_score | 93.3% | 87.0% | 0.536 | 0.966 |

With the production Stage 1 gate in front (271 of 547 pass): macro MCC v0 0.094 [0.044, 0.138], probe 0.159 [0.074, 0.238].


## Confidence gate (TEST)

Each method's confidence judged on its OWN errors. Error rate = share of wrong outputs among the auto-accepted.

| method | correctness event | right / wrong | AUROC [95% CI] | err @0% review | @30% | @50% | @70% |
|---|---|---:|---:|---:|---:|---:|---:|
| v0_production(field) | exact_headline_set | 53 / 494 | 0.346 [0.247, 0.471] | 90.3% | 91.9% | 94.9% | 98.2% |
| v0_production(field) | no_false_positive | 468 / 79 | 0.581 [0.478, 0.687] | 14.4% | 14.1% | 10.9% | 6.1% |
| v0_production(field) | jaccard_ge_0.5 | 237 / 310 | 0.568 [0.483, 0.659] | 56.7% | 51.2% | 55.8% | 53.7% |
| probe_min:name | exact_headline_set | 138 / 409 | 0.769 [0.676, 0.845] | 74.8% | 65.3% | 57.3% | 52.4% |
| probe_min:name | no_false_positive | 239 / 308 | 0.905 [0.848, 0.949] | 56.3% | 39.2% | 24.4% | 12.2% |
| probe_min:name | jaccard_ge_0.5 | 422 / 125 | 0.661 [0.561, 0.756] | 22.9% | 18.8% | 16.4% | 7.9% |

## At the live threshold (TEST; diagnostic added after pre-registration)

| method | review load | auto-accepted | exact set right among accepted | no false positive among accepted |
|---|---:|---:|---:|---:|
| v0_production(field) @ 0.8 | 26.0% | 405 | 7.6% | 85.7% |
| probe_min:name @ 0.8 | 100.0% | 0 | - | - |

Reliability (exact headline set):

| method | confidence bin | n | mean confidence | exact set right |
|---|---|---:|---:|---:|
| v0_production(field) | 0.0-0.5 | 9 | 0.461 | 0.0% |
| v0_production(field) | 0.5-0.8 | 133 | 0.675 | 16.5% |
| v0_production(field) | 0.8-0.9 | 353 | 0.861 | 8.5% |
| v0_production(field) | 0.9-1.0 | 52 | 0.912 | 1.9% |
| probe_min:name | 0.0-0.5 | 421 | 0.171 | 18.5% |
| probe_min:name | 0.5-0.8 | 126 | 0.579 | 47.6% |
| probe_min:name | 0.8-0.9 | 0 | - | - |
| probe_min:name | 0.9-1.0 | 0 | - | - |

## Cost

Median Stage 2 time: v0 generation 8177 ms; probe min:def 1641 ms, min:name 1084 ms, tax:def 2704 ms. Peak reserved GPU memory (whole process, all steps): 18.31 GB. Minimum probability mass on Yes/No tokens: {'min:def': 0.9993, 'min:name': 0.9996, 'tax:def': 0.9993}.

