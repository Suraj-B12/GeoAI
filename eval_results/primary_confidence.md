# Primary-type confidence: evaluation

Paired labelled Attain images: **407** (old = `calib_attain_407.json`, unordered prompt; new = `calib_attain_407_primary.json`, primary-first prompt). New run: 4-bit, prompts `v2`.

## A. Did the prompt change hurt the multi-label answer?

| rule | old correct | new correct | old right, new wrong | old wrong, new right | McNemar p |
|---|---:|---:|---:|---:|---:|
| no_false_positives | 347 (85.3%) | 339 (83.3%) | 9 | 1 | 0.021 |
| any_overlap | 377 (92.6%) | 379 (93.1%) | 1 | 3 | 0.625 |
| jaccard_50 | 156 (38.3%) | 160 (39.3%) | 11 | 15 | 0.557 |
| exact_match | 15 (3.7%) | 19 (4.7%) | 5 | 9 | 0.424 |

Labels per image: old 2.01, new 2.01. Identical label set on 358/407 images (88%): the prompt change mostly reordered labels.

## B. Is the first label the dominant distress?

Dominant = the annotated class with the largest summed bounding-box area (Attain vocabulary). Undecidable when the first label or the dominant class has no counterpart in the other vocabulary.

| first label | old | new |
|---|---:|---:|
| Longitudinal Cracking | 380 | 390 |
| Alligator Cracking | 17 | 9 |
| Potholes | 10 | 8 |

First label = dominant class, on 407 decidable images: old **150** (36.9% [32-42]), new **148** (36.4% [32-41]); 2 images improved, 4 got worse, McNemar p = 0.688.

## C. Does the primary confidence separate right from wrong primary labels?

Scored on the primary label alone: right if the dataset annotates it on the image. 407 images have a primary label the dataset can judge (371 right, 36 wrong). Primary label located in the token stream on 407/407 images.

| metric | AUC ± SE | mean (right) | mean (wrong) |
|---|---:|---:|---:|
| primary (joint) | 0.468 ± 0.051 | 0.636 | 0.649 |
| primary (geomean) | 0.476 ± 0.051 | 0.909 | 0.907 |
| field (all labels) | 0.635 ± 0.044 | 0.878 | 0.863 |

### Operating points for `primary` (joint)

| threshold | auto-accepted | review load | auto-accept error (95% CI) | wrong caught |
|---:|---:|---:|---:|---:|
| 0.50 | 352/407 | 13.5% | 8.8% [6-12] | 5/36 |
| 0.60 | 252/407 | 38.1% | 9.9% [7-14] | 11/36 |
| 0.70 | 135/407 | 66.8% | 8.1% [5-14] | 25/36 |
| 0.75 | 86/407 | 78.9% | 9.3% [5-17] | 28/36 |
| 0.80 ← live | 35/407 | 91.4% | 17.1% [8-33] | 30/36 |
| 0.85 | 8/407 | 98.0% | 50.0% [22-78] | 32/36 |
| 0.90 | 0/407 | 100.0% | nan% [0-0] | 36/36 |
| 0.95 | 0/407 | 100.0% | nan% [0-0] | 36/36 |

Stage 2 review load at the live 0.80 threshold on these 407 images: old gate (field, old prompt) **1.5%**, new gate (primary, new prompt) **91.4%**.

