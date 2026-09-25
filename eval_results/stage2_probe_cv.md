# Stage 2 probe - grouped 5-fold cross-validation (secondary analysis)

847 Attain WS_V2.0 images in 43 blocks of 20 frames; variant `min:name` (pre-registered DEV choice); thresholds and calibration fitted out-of-fold. Rules fixed in `eval_results/stage2_probe_preregistration.json addendum_1` before this ran.

## Which classes can the probe be trusted with? (pooled AUROC, 95% cluster CI)

| class | AUROC | eligible |
|---|---:|---|
| Linear crack | 0.766 [0.652, 0.854] | yes |
| Alligator crack | 0.758 [0.672, 0.837] | yes |
| Pothole | 0.620 [0.486, 0.739] | no - verify-only |
| Raveling | 0.624 [0.506, 0.741] | yes |
| Weathering | 0.655 [0.432, 0.802] | no - verify-only |
| Patch and utility cut | 0.579 [0.467, 0.737] | no - verify-only |
| Block crack | 0.804 [0.655, 0.917] | diagnostic |

## Out-of-fold, all 847 images

| method | macro MCC [95% CI] | macro F1 | macro bal. acc. | mean Jaccard | exact set |
|---|---:|---:|---:|---:|---:|
| v0_production | 0.125 [0.074, 0.170] | 0.249 | 0.560 | 0.410 | 14.1% |
| probe_all_classes | 0.248 [0.148, 0.335] | 0.508 | 0.614 | 0.585 | 23.6% |
| probe_eligible_only | 0.241 [0.159, 0.302] | 0.443 | 0.604 | 0.594 | 24.8% |
| constant_prior | 0.000 [0.000, 0.000] | 0.339 | 0.500 | 0.606 | 29.4% |

| paired vs production | macro MCC delta | macro F1 delta | mean Jaccard delta |
|---|---:|---:|---:|
| probe_all_classes | +0.123 [+0.046, +0.194] (p 0.002) | +0.259 [+0.200, +0.312] | +0.175 [+0.122, +0.225] |
| probe_eligible_only | +0.116 [+0.064, +0.157] (p 0.001) | +0.193 [+0.161, +0.226] | +0.183 [+0.128, +0.234] |

### Per class (out-of-fold)

| class | prevalence | v0 P / R | v0 MCC | probe (eligible rule) P / R | probe MCC [95% CI] | delta MCC [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| Linear crack | 81.1% | 0.90 / 0.82 | 0.385 | 0.88 / 0.96 | 0.478 [0.264, 0.617] | +0.093 [-0.013, +0.192] |
| Alligator crack | 66.2% | 0.88 / 0.08 | 0.113 | 0.76 / 0.93 | 0.414 [0.247, 0.554] | +0.302 [+0.151, +0.444] |
| Pothole | 19.2% | 0.41 / 0.18 | 0.166 | 0.50 / 0.12 | 0.169 [0.007, 0.306] | +0.003 [-0.057, +0.058] |
| Raveling | 12.6% | 0.00 / 0.00 | 0.000 | 0.17 / 0.76 | 0.143 [-0.023, 0.289] | +0.143 [-0.026, +0.288] |
| Weathering | 19.7% | 0.00 / 0.00 | -0.038 | 0.00 / 0.00 | 0.000 [0.000, 0.000] | +0.038 [+0.000, +0.060] |
| Patch and utility cut | 19.7% | 0.00 / 0.00 | 0.000 | 0.00 / 0.00 | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| Block crack | 3.8% | 0.00 / 0.00 | 0.000 | 0.00 / 0.00 | 0.000 [0.000, 0.000] | - |

## Which confidence should gate the probe's outputs? (out-of-fold AUROC for the probe output being right)

| correctness event | n right | probe confidence | field confidence |
|---|---:|---:|---:|
| exact_headline_set | 210 | 0.778 [0.686, 0.854] | 0.483 [0.376, 0.603] |
| jaccard_ge_0.5 | 607 | 0.629 [0.554, 0.701] | 0.662 [0.596, 0.723] |
| no_false_positive | 359 | 0.874 [0.804, 0.926] | 0.473 [0.382, 0.570] |

Probe confidence reliability (exact headline set, out-of-fold): 0.0-0.2: n=462, mean 0.098, right 7.6%; 0.2-0.4: n=111, mean 0.318, right 27.9%; 0.4-0.6: n=141, mean 0.504, right 53.9%; 0.6-0.8: n=127, mean 0.674, right 52.0%; 0.8-1.0: n=6, mean 0.813, right 33.3%. ECE 0.076.

## Stage 1 safety net (reported only)

Production Stage 1 recall on distress images: 51.8%. Of the 449 images Stage 1 calls Normal, 371 carry annotated distress; the probe (out-of-fold threshold) would flag 313 of them for review, at the cost of flagging 22 of the 78 without distress.

