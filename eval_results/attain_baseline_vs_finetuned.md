# Attain WS_V2.0 — Baseline vs Fine-tuned Comparison

Subset: WS_V2.0, 769 images

## Headline accuracy

| Metric | Baseline | Fine-tuned | Δ |
|---|---|---|---|
| Tier 1 (RDD-trained classes) | 17.51% | 31.11% | **+13.60%** |
| Tier 2 (zero-shot, taxonomy-only) | 3.27% | 0.00% | **-3.27%** |
| Severity classification | 8.94% | 25.76% | **+16.82%** |

## Per-class precision / recall / F1

| Class | Tier | Base P | FT P | Base R | FT R | Base F1 | FT F1 | ΔF1 |
|---|---|---|---|---|---|---|---|---|
| Alligator crack | in-dist | 1.000 | 0.932 | 0.009 | 0.123 | 0.018 | 0.217 | **+0.200** |
| Block crack | zero-shot | 0.212 | 0.000 | 0.219 | 0.000 | 0.215 | 0.000 | **-0.215** |
| Linear crack | in-dist | 0.929 | 0.911 | 0.323 | 0.539 | 0.479 | 0.677 | **+0.198** |
| Pothole | in-dist | 0.488 | 0.000 | 0.123 | 0.000 | 0.196 | 0.000 | **-0.196** |
| Raveling | zero-shot | 0.143 | 0.000 | 0.009 | 0.000 | 0.018 | 0.000 | **-0.018** |
| Weathering | zero-shot | 1.000 | 0.000 | 0.012 | 0.000 | 0.024 | 0.000 | **-0.024** |

## Per-class TP / FP / FN

| Class | Base TP | FT TP | Base FP | FT FP | Base FN | FT FN |
|---|---|---|---|---|---|---|
| Alligator crack | 5 | 69 | 0 | 5 | 556 | 492 |
| Block crack | 7 | 0 | 26 | 0 | 25 | 32 |
| Linear crack | 222 | 370 | 17 | 36 | 465 | 317 |
| Pothole | 20 | 0 | 21 | 0 | 143 | 163 |
| Raveling | 1 | 0 | 6 | 0 | 106 | 107 |
| Weathering | 2 | 0 | 0 | 0 | 165 | 167 |

## Stage 1 detection rate

| Model | Total | Predicted Distress | Predicted Normal | Distress rate |
|---|---|---|---|---|
| baseline | 769 | 293 | 476 | 38.10% |
| finetuned | 769 | 527 | 242 | 68.53% |

## Pothole bias check

- Ground-truth pothole instances in dataset: **163**
- Baseline predictions containing 'Pothole (D40)': **41**
- Fine-tuned predictions containing 'Pothole (D40)': **0**

## Pipeline-label prediction frequency

How often each model emitted each pipeline label across all 769 images:

| Pipeline label | Baseline | Fine-tuned | Δ |
|---|---|---|---|
| Longitudinal Crack (D00) | 239 | 406 | +167 |
| Transverse Crack (D10) | 17 | 97 | +80 |
| Unknown | 0 | 80 | +80 |
| Alligator Crack (D20) | 5 | 74 | +69 |
| Pothole (D40) | 41 | 0 | -41 |
| Block Crack (D43) | 33 | 0 | -33 |
| Raveling | 7 | 0 | -7 |
| Weathering/Oxidation | 2 | 0 | -2 |

## Severity prediction distribution

| Severity | Baseline | Fine-tuned |
|---|---|---|
| High | 45 | 2 |
| Low | 56 | 396 |
| Medium | 192 | 129 |
| Unknown | 476 | 242 |

## Head-to-head per-image (whose predictions cover MORE of GT)

- Images compared: 769
- **Fine-tuned strictly better**: 214
- **Baseline strictly better**:   37
- Tied, both correct on at least one type: 191
- Tied, both wrong / partial:             327

### Examples where baseline beat fine-tuned (first 10)

| Image | Ground truth | Baseline preds | Fine-tuned preds |
|---|---|---|---|
| Attain_SMP_WS_v2_000019.jpg | Pothole, Linear crack, Weathering | Raveling, Weathering/Oxidation | Unknown |
| Attain_SMP_WS_v2_000374.jpg | Pothole, Alligator crack, Linear crack | Longitudinal Crack (D00) | Unknown |
| Attain_SMP_WS_v2_000378.jpg | Pothole, Alligator crack, Linear crack | Pothole (D40) | Unknown |
| Attain_SMP_WS_v2_000018.jpg | Pothole, Linear crack, Weathering | Pothole (D40) | Unknown |
| Attain_SMP_WS_v2_000021.jpg | Pothole, Linear crack, Weathering | Longitudinal Crack (D00), Raveling | Unknown |
| Attain_SMP_WS_v2_000578.jpg | Block crack, Linear crack, Alligator crack, Patch and utility cut- Low | Longitudinal Crack (D00), Block Crack (D43) | Alligator Crack (D20) |
| Attain_SMP_WS_v2_000391.jpg | Linear crack, Pothole, Patch and utility cut- Low, Patch and utility cut- High, Alligator crack | Longitudinal Crack (D00), Pothole (D40) | Longitudinal Crack (D00), Transverse Crack (D10) |
| Attain_SMP_WS_v2_000028.jpg | Pothole, Linear crack, Weathering | Pothole (D40) | Unknown |
| Attain_SMP_WS_v2_000020.jpg | Pothole, Linear crack, Weathering | Longitudinal Crack (D00), Block Crack (D43) | Alligator Crack (D20) |
| Attain_SMP_WS_v2_000022.jpg | Pothole, Linear crack, Weathering | Pothole (D40) | Unknown |

### Examples where fine-tuned beat baseline (first 10)

| Image | Ground truth | Baseline preds | Fine-tuned preds |
|---|---|---|---|
| Attain_SMP_WS_v2_000718.jpg | Alligator crack |  | Alligator Crack (D20) |
| Attain_SMP_WS_v2_000350.jpg | Alligator crack, Linear crack |  | Longitudinal Crack (D00) |
| Attain_SMP_WS_v2_000533.jpg | Alligator crack, Linear crack |  | Longitudinal Crack (D00) |
| Attain_SMP_WS_v2_000005.jpg | Block crack, Raveling, Linear crack, Weathering |  | Longitudinal Crack (D00) |
| Attain_SMP_WS_v2_000373.jpg | Pothole, Alligator crack, Linear crack |  | Longitudinal Crack (D00), Transverse Crack (D10) |
| Attain_SMP_WS_v2_000193.jpg | Alligator crack, Linear crack |  | Longitudinal Crack (D00) |
| Attain_SMP_WS_v2_000458.jpg | Alligator crack, Linear crack, Weathering |  | Longitudinal Crack (D00) |
| Attain_SMP_WS_v2_000624.jpg | Raveling, Linear crack, Alligator crack, Patch and utility cut- Low |  | Longitudinal Crack (D00) |
| Attain_SMP_WS_v2_000400.jpg | Pothole, Alligator crack, Linear crack | Block Crack (D43) | Longitudinal Crack (D00), Transverse Crack (D10) |
| Attain_SMP_WS_v2_000175.jpg | Alligator crack, Linear crack |  | Longitudinal Crack (D00) |
