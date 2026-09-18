# Expert-review threshold decision

Metric: `field` (Stage 2 confidence).  Live threshold: **0.8**.

Sources: calib_attain_407.json, calib_rdd_india_250.json

## ALL POOLED (n=606)

### no_false_positives

352 right / 254 wrong. AUC **0.931** (95% CI 0.911-0.951) - informative.

| threshold | auto-accepted | wrong through | error rate (95% CI) | wrong caught | review load |
|---:|---:|---:|---:|---:|---:|
| 0.70 | 562/606 (93%) | 211 | 38% [34-42] | 43/254 (17%) | 7% |
| 0.72 | 543/606 (90%) | 192 | 35% [31-39] | 62/254 (24%) | 10% |
| 0.75 | 500/606 (83%) | 150 | 30% [26-34] | 104/254 (41%) | 17% |
| 0.78 | 457/606 (75%) | 109 | 24% [20-28] | 145/254 (57%) | 25% |
| 0.80  <- live | 432/606 (71%) | 84 | 19% [16-23] | 170/254 (67%) | 29% |
| 0.82 | 403/606 (67%) | 63 | 16% [12-20] | 191/254 (75%) | 33% |
| 0.85 | 343/606 (57%) | 38 | 11% [8-15] | 216/254 (85%) | 43% |

**0.80 vs 0.75.** 0.80 -> 0.75: auto-accepts +68 images, lets through +66 more wrong ones, catches -66 fewer errors, review load 29% -> 17%.  The error-rate difference is outside sampling noise.

### jaccard_50

192 right / 414 wrong. AUC **0.585** (95% CI 0.535-0.635) - informative.

| threshold | auto-accepted | wrong through | error rate (95% CI) | wrong caught | review load |
|---:|---:|---:|---:|---:|---:|
| 0.70 | 562/606 (93%) | 384 | 68% [64-72] | 30/414 (7%) | 7% |
| 0.72 | 543/606 (90%) | 371 | 68% [64-72] | 43/414 (10%) | 10% |
| 0.75 | 500/606 (83%) | 331 | 66% [62-70] | 83/414 (20%) | 17% |
| 0.78 | 457/606 (75%) | 292 | 64% [59-68] | 122/414 (29%) | 25% |
| 0.80  <- live | 432/606 (71%) | 271 | 63% [58-67] | 143/414 (35%) | 29% |
| 0.82 | 403/606 (67%) | 251 | 62% [57-67] | 163/414 (39%) | 33% |
| 0.85 | 343/606 (57%) | 225 | 66% [60-70] | 189/414 (46%) | 43% |

**0.80 vs 0.75.** 0.80 -> 0.75: auto-accepts +68 images, lets through +60 more wrong ones, catches -60 fewer errors, review load 29% -> 17%.  Error rates are NOT statistically distinguishable at this sample size.

### any_overlap

429 right / 177 wrong. AUC **0.857** (95% CI 0.828-0.886) - informative.

| threshold | auto-accepted | wrong through | error rate (95% CI) | wrong caught | review load |
|---:|---:|---:|---:|---:|---:|
| 0.70 | 562/606 (93%) | 153 | 27% [24-31] | 24/177 (14%) | 7% |
| 0.72 | 543/606 (90%) | 140 | 26% [22-30] | 37/177 (21%) | 10% |
| 0.75 | 500/606 (83%) | 103 | 21% [17-24] | 74/177 (42%) | 17% |
| 0.78 | 457/606 (75%) | 66 | 14% [12-18] | 111/177 (63%) | 25% |
| 0.80  <- live | 432/606 (71%) | 48 | 11% [8-14] | 129/177 (73%) | 29% |
| 0.82 | 403/606 (67%) | 36 | 9% [7-12] | 141/177 (80%) | 33% |
| 0.85 | 343/606 (57%) | 28 | 8% [6-12] | 149/177 (84%) | 43% |

**0.80 vs 0.75.** 0.80 -> 0.75: auto-accepts +68 images, lets through +55 more wrong ones, catches -55 fewer errors, review load 29% -> 17%.  The error-rate difference is outside sampling noise.

### exact_match

17 right / 589 wrong. AUC **0.583** (95% CI 0.439-0.727) - carries no information (95% CI spans 0.5).

| threshold | auto-accepted | wrong through | error rate (95% CI) | wrong caught | review load |
|---:|---:|---:|---:|---:|---:|
| 0.70 | 562/606 (93%) | 546 | 97% [95-98] | 43/589 (7%) | 7% |
| 0.72 | 543/606 (90%) | 527 | 97% [95-98] | 62/589 (11%) | 10% |
| 0.75 | 500/606 (83%) | 484 | 97% [95-98] | 105/589 (18%) | 17% |
| 0.78 | 457/606 (75%) | 441 | 96% [94-98] | 148/589 (25%) | 25% |
| 0.80  <- live | 432/606 (71%) | 416 | 96% [94-98] | 173/589 (29%) | 29% |
| 0.82 | 403/606 (67%) | 387 | 96% [94-98] | 202/589 (34%) | 33% |
| 0.85 | 343/606 (57%) | 331 | 97% [94-98] | 258/589 (44%) | 43% |

**0.80 vs 0.75.** 0.80 -> 0.75: auto-accepts +68 images, lets through +68 more wrong ones, catches -68 fewer errors, review load 29% -> 17%.  Error rates are NOT statistically distinguishable at this sample size.

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

## rdd_india (n=199)

### no_false_positives

5 right / 194 wrong. AUC **0.793** (95% CI 0.555-1.000) - informative.

| threshold | auto-accepted | wrong through | error rate (95% CI) | wrong caught | review load |
|---:|---:|---:|---:|---:|---:|
| 0.70 | 155/199 (78%) | 151 | 97% [94-99] | 43/194 (22%) | 22% |
| 0.72 | 136/199 (68%) | 132 | 97% [93-99] | 62/194 (32%) | 32% |
| 0.75 | 94/199 (47%) | 90 | 96% [90-98] | 104/194 (54%) | 53% |
| 0.78 | 52/199 (26%) | 49 | 94% [84-98] | 145/194 (75%) | 74% |
| 0.80  <- live | 31/199 (16%) | 28 | 90% [75-97] | 166/194 (86%) | 84% |
| 0.82 | 13/199 (7%) | 10 | 77% [50-92] | 184/194 (95%) | 93% |
| 0.85 | 5/199 (3%) | 2 | 40% [12-77] | 192/194 (99%) | 97% |

**0.80 vs 0.75.** 0.80 -> 0.75: auto-accepts +63 images, lets through +62 more wrong ones, catches -62 fewer errors, review load 84% -> 53%.  Error rates are NOT statistically distinguishable at this sample size.

### jaccard_50

36 right / 163 wrong. AUC **0.433** (95% CI 0.333-0.533) - carries no information (95% CI spans 0.5).

| threshold | auto-accepted | wrong through | error rate (95% CI) | wrong caught | review load |
|---:|---:|---:|---:|---:|---:|
| 0.70 | 155/199 (78%) | 133 | 86% [79-90] | 30/163 (18%) | 22% |
| 0.72 | 136/199 (68%) | 120 | 88% [82-93] | 43/163 (26%) | 32% |
| 0.75 | 94/199 (47%) | 80 | 85% [77-91] | 83/163 (51%) | 53% |
| 0.78 | 52/199 (26%) | 42 | 81% [68-89] | 121/163 (74%) | 74% |
| 0.80  <- live | 31/199 (16%) | 23 | 74% [57-86] | 140/163 (86%) | 84% |
| 0.82 | 13/199 (7%) | 7 | 54% [29-77] | 156/163 (96%) | 93% |
| 0.85 | 5/199 (3%) | 2 | 40% [12-77] | 161/163 (99%) | 97% |

**0.80 vs 0.75.** 0.80 -> 0.75: auto-accepts +63 images, lets through +57 more wrong ones, catches -57 fewer errors, review load 84% -> 53%.  Error rates are NOT statistically distinguishable at this sample size.

### any_overlap

52 right / 147 wrong. AUC **0.435** (95% CI 0.347-0.524) - carries no information (95% CI spans 0.5).

| threshold | auto-accepted | wrong through | error rate (95% CI) | wrong caught | review load |
|---:|---:|---:|---:|---:|---:|
| 0.70 | 155/199 (78%) | 123 | 79% [72-85] | 24/147 (16%) | 22% |
| 0.72 | 136/199 (68%) | 110 | 81% [73-87] | 37/147 (25%) | 32% |
| 0.75 | 94/199 (47%) | 73 | 78% [68-85] | 74/147 (50%) | 53% |
| 0.78 | 52/199 (26%) | 36 | 69% [56-80] | 111/147 (76%) | 74% |
| 0.80  <- live | 31/199 (16%) | 18 | 58% [41-74] | 129/147 (88%) | 84% |
| 0.82 | 13/199 (7%) | 6 | 46% [23-71] | 141/147 (96%) | 93% |
| 0.85 | 5/199 (3%) | 1 | 20% [4-62] | 146/147 (99%) | 97% |

**0.80 vs 0.75.** 0.80 -> 0.75: auto-accepts +63 images, lets through +55 more wrong ones, catches -55 fewer errors, review load 84% -> 53%.  Error rates are NOT statistically distinguishable at this sample size.

### exact_match

2 right / 197 wrong. AUC **0.607** (95% CI 0.188-1.000) - carries no information (95% CI spans 0.5).

| threshold | auto-accepted | wrong through | error rate (95% CI) | wrong caught | review load |
|---:|---:|---:|---:|---:|---:|
| 0.70 | 155/199 (78%) | 154 | 99% [96-100] | 43/197 (22%) | 22% |
| 0.72 | 136/199 (68%) | 135 | 99% [96-100] | 62/197 (31%) | 32% |
| 0.75 | 94/199 (47%) | 93 | 99% [94-100] | 104/197 (53%) | 53% |
| 0.78 | 52/199 (26%) | 51 | 98% [90-100] | 146/197 (74%) | 74% |
| 0.80  <- live | 31/199 (16%) | 30 | 97% [84-99] | 167/197 (85%) | 84% |
| 0.82 | 13/199 (7%) | 12 | 92% [67-99] | 185/197 (94%) | 93% |
| 0.85 | 5/199 (3%) | 4 | 80% [38-96] | 193/197 (98%) | 97% |

**0.80 vs 0.75.** 0.80 -> 0.75: auto-accepts +63 images, lets through +63 more wrong ones, catches -63 fewer errors, review load 84% -> 53%.  Error rates are NOT statistically distinguishable at this sample size.
