# Expert-review threshold decision

Metric: `field` (Stage 2 confidence).  Live threshold: **0.8**.

Sources: calib_attain_407.json

## ALL POOLED (n=407)

### no_false_positives

347 right / 60 wrong. AUC **0.744** (95% CI 0.686-0.802) - informative.

| threshold | auto-accepted | wrong through | error rate (95% CI) | wrong caught | review load |
|---:|---:|---:|---:|---:|---:|
| 0.70 | 407/407 (100%) | 60 | 15% [12-19] | 0/60 (0%) | 0% |
| 0.72 | 407/407 (100%) | 60 | 15% [12-19] | 0/60 (0%) | 0% |
| 0.75 | 406/407 (100%) | 60 | 15% [12-19] | 0/60 (0%) | 0% |
| 0.78 | 405/407 (100%) | 60 | 15% [12-19] | 0/60 (0%) | 0% |
| 0.80  <- live | 401/407 (99%) | 56 | 14% [11-18] | 4/60 (7%) | 1% |
| 0.82 | 390/407 (96%) | 53 | 14% [11-17] | 7/60 (12%) | 4% |
| 0.85 | 338/407 (83%) | 36 | 11% [8-14] | 24/60 (40%) | 17% |

**0.80 vs 0.75.** 0.80 -> 0.75: auto-accepts +5 images, lets through +4 more wrong ones, catches -4 fewer errors, review load 1% -> 0%.  Error rates are NOT statistically distinguishable at this sample size.

### jaccard_50

156 right / 251 wrong. AUC **0.473** (95% CI 0.415-0.530) - carries no information (95% CI spans 0.5).

| threshold | auto-accepted | wrong through | error rate (95% CI) | wrong caught | review load |
|---:|---:|---:|---:|---:|---:|
| 0.70 | 407/407 (100%) | 251 | 62% [57-66] | 0/251 (0%) | 0% |
| 0.72 | 407/407 (100%) | 251 | 62% [57-66] | 0/251 (0%) | 0% |
| 0.75 | 406/407 (100%) | 251 | 62% [57-66] | 0/251 (0%) | 0% |
| 0.78 | 405/407 (100%) | 250 | 62% [57-66] | 1/251 (0%) | 0% |
| 0.80  <- live | 401/407 (99%) | 248 | 62% [57-66] | 3/251 (1%) | 1% |
| 0.82 | 390/407 (96%) | 244 | 63% [58-67] | 7/251 (3%) | 4% |
| 0.85 | 338/407 (83%) | 223 | 66% [61-71] | 28/251 (11%) | 17% |

**0.80 vs 0.75.** 0.80 -> 0.75: auto-accepts +5 images, lets through +3 more wrong ones, catches -3 fewer errors, review load 1% -> 0%.  Error rates are NOT statistically distinguishable at this sample size.

### any_overlap

377 right / 30 wrong. AUC **0.593** (95% CI 0.493-0.692) - carries no information (95% CI spans 0.5).

| threshold | auto-accepted | wrong through | error rate (95% CI) | wrong caught | review load |
|---:|---:|---:|---:|---:|---:|
| 0.70 | 407/407 (100%) | 30 | 7% [5-10] | 0/30 (0%) | 0% |
| 0.72 | 407/407 (100%) | 30 | 7% [5-10] | 0/30 (0%) | 0% |
| 0.75 | 406/407 (100%) | 30 | 7% [5-10] | 0/30 (0%) | 0% |
| 0.78 | 405/407 (100%) | 30 | 7% [5-10] | 0/30 (0%) | 0% |
| 0.80  <- live | 401/407 (99%) | 30 | 7% [5-10] | 0/30 (0%) | 1% |
| 0.82 | 390/407 (96%) | 30 | 8% [5-11] | 0/30 (0%) | 4% |
| 0.85 | 338/407 (83%) | 27 | 8% [6-11] | 3/30 (10%) | 17% |

**0.80 vs 0.75.** 0.80 -> 0.75: auto-accepts +5 images, lets through +0 more wrong ones, catches +0 fewer errors, review load 1% -> 0%.  Error rates are NOT statistically distinguishable at this sample size.

### exact_match

15 right / 392 wrong. AUC **0.414** (95% CI 0.276-0.553) - carries no information (95% CI spans 0.5).

| threshold | auto-accepted | wrong through | error rate (95% CI) | wrong caught | review load |
|---:|---:|---:|---:|---:|---:|
| 0.70 | 407/407 (100%) | 392 | 96% [94-98] | 0/392 (0%) | 0% |
| 0.72 | 407/407 (100%) | 392 | 96% [94-98] | 0/392 (0%) | 0% |
| 0.75 | 406/407 (100%) | 391 | 96% [94-98] | 1/392 (0%) | 0% |
| 0.78 | 405/407 (100%) | 390 | 96% [94-98] | 2/392 (1%) | 0% |
| 0.80  <- live | 401/407 (99%) | 386 | 96% [94-98] | 6/392 (2%) | 1% |
| 0.82 | 390/407 (96%) | 375 | 96% [94-98] | 17/392 (4%) | 4% |
| 0.85 | 338/407 (83%) | 327 | 97% [94-98] | 65/392 (17%) | 17% |

**0.80 vs 0.75.** 0.80 -> 0.75: auto-accepts +5 images, lets through +5 more wrong ones, catches -5 fewer errors, review load 1% -> 0%.  Error rates are NOT statistically distinguishable at this sample size.

## attain (n=407)

### no_false_positives

347 right / 60 wrong. AUC **0.744** (95% CI 0.686-0.802) - informative.

| threshold | auto-accepted | wrong through | error rate (95% CI) | wrong caught | review load |
|---:|---:|---:|---:|---:|---:|
| 0.70 | 407/407 (100%) | 60 | 15% [12-19] | 0/60 (0%) | 0% |
| 0.72 | 407/407 (100%) | 60 | 15% [12-19] | 0/60 (0%) | 0% |
| 0.75 | 406/407 (100%) | 60 | 15% [12-19] | 0/60 (0%) | 0% |
| 0.78 | 405/407 (100%) | 60 | 15% [12-19] | 0/60 (0%) | 0% |
| 0.80  <- live | 401/407 (99%) | 56 | 14% [11-18] | 4/60 (7%) | 1% |
| 0.82 | 390/407 (96%) | 53 | 14% [11-17] | 7/60 (12%) | 4% |
| 0.85 | 338/407 (83%) | 36 | 11% [8-14] | 24/60 (40%) | 17% |

**0.80 vs 0.75.** 0.80 -> 0.75: auto-accepts +5 images, lets through +4 more wrong ones, catches -4 fewer errors, review load 1% -> 0%.  Error rates are NOT statistically distinguishable at this sample size.

### jaccard_50

156 right / 251 wrong. AUC **0.473** (95% CI 0.415-0.530) - carries no information (95% CI spans 0.5).

| threshold | auto-accepted | wrong through | error rate (95% CI) | wrong caught | review load |
|---:|---:|---:|---:|---:|---:|
| 0.70 | 407/407 (100%) | 251 | 62% [57-66] | 0/251 (0%) | 0% |
| 0.72 | 407/407 (100%) | 251 | 62% [57-66] | 0/251 (0%) | 0% |
| 0.75 | 406/407 (100%) | 251 | 62% [57-66] | 0/251 (0%) | 0% |
| 0.78 | 405/407 (100%) | 250 | 62% [57-66] | 1/251 (0%) | 0% |
| 0.80  <- live | 401/407 (99%) | 248 | 62% [57-66] | 3/251 (1%) | 1% |
| 0.82 | 390/407 (96%) | 244 | 63% [58-67] | 7/251 (3%) | 4% |
| 0.85 | 338/407 (83%) | 223 | 66% [61-71] | 28/251 (11%) | 17% |

**0.80 vs 0.75.** 0.80 -> 0.75: auto-accepts +5 images, lets through +3 more wrong ones, catches -3 fewer errors, review load 1% -> 0%.  Error rates are NOT statistically distinguishable at this sample size.

### any_overlap

377 right / 30 wrong. AUC **0.593** (95% CI 0.493-0.692) - carries no information (95% CI spans 0.5).

| threshold | auto-accepted | wrong through | error rate (95% CI) | wrong caught | review load |
|---:|---:|---:|---:|---:|---:|
| 0.70 | 407/407 (100%) | 30 | 7% [5-10] | 0/30 (0%) | 0% |
| 0.72 | 407/407 (100%) | 30 | 7% [5-10] | 0/30 (0%) | 0% |
| 0.75 | 406/407 (100%) | 30 | 7% [5-10] | 0/30 (0%) | 0% |
| 0.78 | 405/407 (100%) | 30 | 7% [5-10] | 0/30 (0%) | 0% |
| 0.80  <- live | 401/407 (99%) | 30 | 7% [5-10] | 0/30 (0%) | 1% |
| 0.82 | 390/407 (96%) | 30 | 8% [5-11] | 0/30 (0%) | 4% |
| 0.85 | 338/407 (83%) | 27 | 8% [6-11] | 3/30 (10%) | 17% |

**0.80 vs 0.75.** 0.80 -> 0.75: auto-accepts +5 images, lets through +0 more wrong ones, catches +0 fewer errors, review load 1% -> 0%.  Error rates are NOT statistically distinguishable at this sample size.

### exact_match

15 right / 392 wrong. AUC **0.414** (95% CI 0.276-0.553) - carries no information (95% CI spans 0.5).

| threshold | auto-accepted | wrong through | error rate (95% CI) | wrong caught | review load |
|---:|---:|---:|---:|---:|---:|
| 0.70 | 407/407 (100%) | 392 | 96% [94-98] | 0/392 (0%) | 0% |
| 0.72 | 407/407 (100%) | 392 | 96% [94-98] | 0/392 (0%) | 0% |
| 0.75 | 406/407 (100%) | 391 | 96% [94-98] | 1/392 (0%) | 0% |
| 0.78 | 405/407 (100%) | 390 | 96% [94-98] | 2/392 (1%) | 0% |
| 0.80  <- live | 401/407 (99%) | 386 | 96% [94-98] | 6/392 (2%) | 1% |
| 0.82 | 390/407 (96%) | 375 | 96% [94-98] | 17/392 (4%) | 4% |
| 0.85 | 338/407 (83%) | 327 | 97% [94-98] | 65/392 (17%) | 17% |

**0.80 vs 0.75.** 0.80 -> 0.75: auto-accepts +5 images, lets through +5 more wrong ones, catches -5 fewer errors, review load 1% -> 0%.  Error rates are NOT statistically distinguishable at this sample size.

## Matched operating points (production: uploaded_photos_confidence.json, n=49)

### attain (n=407)

| production threshold | production review load | matched labelled threshold | definition | auto-accept error (95% CI) | errors caught |
|---:|---:|---:|---|---:|---:|
| 0.85 | 80% | 0.898 | no_false_positives | 3.6% [1-10] | 57/60 (95%) |
| 0.85 | 80% | 0.898 | any_overlap | 3.6% [1-10] | 27/30 (90%) |
| 0.80 | 76% | 0.896 | no_false_positives | 3.0% [1-8] | 57/60 (95%) |
| 0.80 | 76% | 0.896 | any_overlap | 3.0% [1-8] | 27/30 (90%) |
| 0.78 | 69% | 0.890 | no_false_positives | 5.6% [3-11] | 53/60 (88%) |
| 0.78 | 69% | 0.890 | any_overlap | 4.8% [2-10] | 24/30 (80%) |
| 0.75 | 63% | 0.885 | no_false_positives | 6.0% [3-11] | 51/60 (85%) |
| 0.75 | 63% | 0.885 | any_overlap | 4.6% [2-9] | 23/30 (77%) |
| 0.70 | 37% | 0.866 | no_false_positives | 7.0% [4-11] | 42/60 (70%) |
| 0.70 | 37% | 0.866 | any_overlap | 5.4% [3-9] | 16/30 (53%) |
