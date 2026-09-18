# Expert-review threshold decision

Metric: `field` (Stage 2 confidence).  Live threshold: **0.8**.

Sources: pilot_mixed.json

## ALL POOLED (n=14)

### no_false_positives

11 right / 3 wrong. AUC **1.000** (CI undefined, perfect separation) - separates perfectly on this sample - with this few negatives that is a sample-size artefact, not a result.

| threshold | auto-accepted | wrong through | error rate (95% CI) | wrong caught | review load |
|---:|---:|---:|---:|---:|---:|
| 0.70 | 13/14 (93%) | 2 | 15% [4-42] | 1/3 (33%) | 7% |
| 0.72 | 13/14 (93%) | 2 | 15% [4-42] | 1/3 (33%) | 7% |
| 0.75 | 13/14 (93%) | 2 | 15% [4-42] | 1/3 (33%) | 7% |
| 0.78 | 12/14 (86%) | 1 | 8% [1-35] | 2/3 (67%) | 14% |
| 0.80  <- live | 12/14 (86%) | 1 | 8% [1-35] | 2/3 (67%) | 14% |
| 0.82 | 10/14 (71%) | 0 | 0% [0-28] | 3/3 (100%) | 29% |
| 0.85 | 10/14 (71%) | 0 | 0% [0-28] | 3/3 (100%) | 29% |

**0.80 vs 0.75.** 0.80 -> 0.75: auto-accepts +1 images, lets through +1 more wrong ones, catches -1 fewer errors, review load 14% -> 7%.  Error rates are NOT statistically distinguishable at this sample size.

### jaccard_50

6 right / 8 wrong. AUC **0.500** (95% CI 0.184-0.816) - carries no information (95% CI spans 0.5).

| threshold | auto-accepted | wrong through | error rate (95% CI) | wrong caught | review load |
|---:|---:|---:|---:|---:|---:|
| 0.70 | 13/14 (93%) | 7 | 54% [29-77] | 1/8 (12%) | 7% |
| 0.72 | 13/14 (93%) | 7 | 54% [29-77] | 1/8 (12%) | 7% |
| 0.75 | 13/14 (93%) | 7 | 54% [29-77] | 1/8 (12%) | 7% |
| 0.78 | 12/14 (86%) | 7 | 58% [32-81] | 1/8 (12%) | 14% |
| 0.80  <- live | 12/14 (86%) | 7 | 58% [32-81] | 1/8 (12%) | 14% |
| 0.82 | 10/14 (71%) | 6 | 60% [31-83] | 2/8 (25%) | 29% |
| 0.85 | 10/14 (71%) | 6 | 60% [31-83] | 2/8 (25%) | 29% |

**0.80 vs 0.75.** 0.80 -> 0.75: auto-accepts +1 images, lets through +0 more wrong ones, catches +0 fewer errors, review load 14% -> 7%.  Error rates are NOT statistically distinguishable at this sample size.

### any_overlap

13 right / 1 wrong. AUC **1.000** (CI undefined, perfect separation) - separates perfectly on this sample - with this few negatives that is a sample-size artefact, not a result.

| threshold | auto-accepted | wrong through | error rate (95% CI) | wrong caught | review load |
|---:|---:|---:|---:|---:|---:|
| 0.70 | 13/14 (93%) | 0 | 0% [0-23] | 1/1 (100%) | 7% |
| 0.72 | 13/14 (93%) | 0 | 0% [0-23] | 1/1 (100%) | 7% |
| 0.75 | 13/14 (93%) | 0 | 0% [0-23] | 1/1 (100%) | 7% |
| 0.78 | 12/14 (86%) | 0 | 0% [0-24] | 1/1 (100%) | 14% |
| 0.80  <- live | 12/14 (86%) | 0 | 0% [0-24] | 1/1 (100%) | 14% |
| 0.82 | 10/14 (71%) | 0 | 0% [0-28] | 1/1 (100%) | 29% |
| 0.85 | 10/14 (71%) | 0 | 0% [0-28] | 1/1 (100%) | 29% |

**0.80 vs 0.75.** 0.80 -> 0.75: auto-accepts +1 images, lets through +0 more wrong ones, catches +0 fewer errors, review load 14% -> 7%.  Error rates are NOT statistically distinguishable at this sample size.

### exact_match

0 right / 14 wrong - AUC undefined (one class).

## attain (n=11)

### no_false_positives

11 right / 0 wrong - AUC undefined (one class).

### jaccard_50

5 right / 6 wrong. AUC **0.433** (95% CI 0.081-0.786) - carries no information (95% CI spans 0.5).

| threshold | auto-accepted | wrong through | error rate (95% CI) | wrong caught | review load |
|---:|---:|---:|---:|---:|---:|
| 0.70 | 11/11 (100%) | 6 | 55% [28-79] | 0/6 (0%) | 0% |
| 0.72 | 11/11 (100%) | 6 | 55% [28-79] | 0/6 (0%) | 0% |
| 0.75 | 11/11 (100%) | 6 | 55% [28-79] | 0/6 (0%) | 0% |
| 0.78 | 11/11 (100%) | 6 | 55% [28-79] | 0/6 (0%) | 0% |
| 0.80  <- live | 11/11 (100%) | 6 | 55% [28-79] | 0/6 (0%) | 0% |
| 0.82 | 10/11 (91%) | 6 | 60% [31-83] | 0/6 (0%) | 9% |
| 0.85 | 10/11 (91%) | 6 | 60% [31-83] | 0/6 (0%) | 9% |

**0.80 vs 0.75.** 0.80 -> 0.75: auto-accepts +0 images, lets through +0 more wrong ones, catches +0 fewer errors, review load 0% -> 0%.  Error rates are NOT statistically distinguishable at this sample size.

### any_overlap

11 right / 0 wrong - AUC undefined (one class).

### exact_match

0 right / 11 wrong - AUC undefined (one class).

## rdd_india (n=3)

### no_false_positives

0 right / 3 wrong - AUC undefined (one class).

### jaccard_50

1 right / 2 wrong. AUC **0.500** (95% CI 0.000-1.000) - carries no information (95% CI spans 0.5).

| threshold | auto-accepted | wrong through | error rate (95% CI) | wrong caught | review load |
|---:|---:|---:|---:|---:|---:|
| 0.70 | 2/3 (67%) | 1 | 50% [9-91] | 1/2 (50%) | 33% |
| 0.72 | 2/3 (67%) | 1 | 50% [9-91] | 1/2 (50%) | 33% |
| 0.75 | 2/3 (67%) | 1 | 50% [9-91] | 1/2 (50%) | 33% |
| 0.78 | 1/3 (33%) | 1 | 100% [21-100] | 1/2 (50%) | 67% |
| 0.80  <- live | 1/3 (33%) | 1 | 100% [21-100] | 1/2 (50%) | 67% |
| 0.82 | 0/3 (0%) | 0 | n/a | 2/2 (100%) | 100% |
| 0.85 | 0/3 (0%) | 0 | n/a | 2/2 (100%) | 100% |

**0.80 vs 0.75.** 0.80 -> 0.75: auto-accepts +1 images, lets through +0 more wrong ones, catches +0 fewer errors, review load 67% -> 33%.  Error rates are NOT statistically distinguishable at this sample size.

### any_overlap

2 right / 1 wrong. AUC **1.000** (CI undefined, perfect separation) - separates perfectly on this sample - with this few negatives that is a sample-size artefact, not a result.

| threshold | auto-accepted | wrong through | error rate (95% CI) | wrong caught | review load |
|---:|---:|---:|---:|---:|---:|
| 0.70 | 2/3 (67%) | 0 | 0% [0-66] | 1/1 (100%) | 33% |
| 0.72 | 2/3 (67%) | 0 | 0% [0-66] | 1/1 (100%) | 33% |
| 0.75 | 2/3 (67%) | 0 | 0% [0-66] | 1/1 (100%) | 33% |
| 0.78 | 1/3 (33%) | 0 | 0% [0-79] | 1/1 (100%) | 67% |
| 0.80  <- live | 1/3 (33%) | 0 | 0% [0-79] | 1/1 (100%) | 67% |
| 0.82 | 0/3 (0%) | 0 | n/a | 1/1 (100%) | 100% |
| 0.85 | 0/3 (0%) | 0 | n/a | 1/1 (100%) | 100% |

**0.80 vs 0.75.** 0.80 -> 0.75: auto-accepts +1 images, lets through +0 more wrong ones, catches +0 fewer errors, review load 67% -> 33%.  Error rates are NOT statistically distinguishable at this sample size.

### exact_match

0 right / 3 wrong - AUC undefined (one class).
