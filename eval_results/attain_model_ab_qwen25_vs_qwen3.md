# Model A/B — Qwen2.5-VL-7B vs Qwen3-VL-8B

**Dataset:** Attain WS_V2.0, 100 images · **Prompts:** v2 · **Adapter:** none

Both runs used identical prompts, subset, sample count and adapter setting — the only difference is the model, so the deltas below are attributable to the model.

| | A | B |
|---|---|---|
| model_path | Qwen/Qwen2.5-VL-7B-Instruct | Qwen/Qwen3-VL-8B-Instruct |
| model_family | qwen2_5_vl | qwen3_vl |
| quantization_bits | 0 | 0 |
| attn_implementation | sdpa | sdpa |
| max_pixels | 4840000 | 4840000 |

## Headline metrics

| Metric | Qwen2.5-VL-7B | Qwen3-VL-8B | Δ (pp) |
|---|---|---|---|
| Tier 1 in-distribution accuracy | 47.26% | 43.84% | **-3.42** |
| Tier 2 zero-shot accuracy | 0.00% | 4.00% | **+4.00** |
| Severity accuracy | 2.00% | 29.00% | **+27.00** |
| Stage 1 distress prediction rate | 69.00% | 99.00% | **+30.00** |
| Multi-class prediction rate | 69.00% | 66.00% | **-3.00** |

## Per-class F1

| Class | Tier | Qwen2.5-VL-7B | Qwen3-VL-8B | Δ |
|---|---|---|---|---|
| Alligator crack | in-dist | 0.1364 | 0.5185 | +0.382 |
| Block crack | zero-shot | 0.0 | 0.0 | +0.000 |
| Linear crack | in-dist | 0.7389 | 0.5366 | -0.202 |
| Patch | zero-shot | 0.0 | 0.0 | +0.000 |
| Patch and utility cut | zero-shot | 0.0 | 0.0 | +0.000 |
| Pothole | in-dist | 0.3404 | 0.3922 | +0.052 |
| Raveling | zero-shot | 0.0 | 0.1277 | +0.128 |
| Weathering | zero-shot | 0.0 | 0.0235 | +0.024 |

## Per-class TP / FP / FN

| Class | Qwen2.5-VL-7B TP/FP/FN | Qwen3-VL-8B TP/FP/FN |
|---|---|---|
| Alligator crack | 3/5/33 | 21/24/15 |
| Block crack | 0/0/12 | 0/0/12 |
| Linear crack | 58/11/30 | 33/2/55 |
| Patch | 0/0/0 | 0/0/0 |
| Patch and utility cut | 0/0/0 | 0/0/0 |
| Pothole | 8/17/14 | 10/19/12 |
| Raveling | 0/0/4 | 3/40/1 |
| Weathering | 0/0/84 | 1/0/83 |

## Label emissions

| Pipeline label | Qwen2.5-VL-7B | Qwen3-VL-8B |
|---|---|---|
| Alligator Cracking | 8 | 45 |
| Bleeding | 1 | 0 |
| Hairline Cracks | 0 | 1 |
| Hungry Surface | 0 | 1 |
| Longitudinal Cracking | 69 | 35 |
| Normal | 0 | 10 |
| Potholes | 25 | 29 |
| Ravelling | 0 | 43 |
| Transverse Cracking | 42 | 1 |

## Head-to-head (per image)

- Images compared: **100**
- Qwen2.5-VL-7B strictly better: **24**
- Qwen3-VL-8B strictly better: **24**
- Tied, both correct on >=1 class: 28
- Tied, both wrong/partial: 24

### Sample disagreements

| Image | Winner | Ground truth | A predicted | B predicted |
|---|---|---|---|---|
| Attain_SMP_WS_v2_000002.jpg | A | Alligator crack, Linear crack, Weathering | Alligator crack, Linear crack | Linear crack, Raveling |
| Attain_SMP_WS_v2_000003.jpg | B | Block crack, Linear crack, Raveling, Weathering | Linear crack | Linear crack, Raveling |
| Attain_SMP_WS_v2_000005.jpg | B | Block crack, Linear crack, Raveling, Weathering | Linear crack | Linear crack, Raveling |
| Attain_SMP_WS_v2_000006.jpg | B | Block crack, Linear crack, Raveling, Weathering | Linear crack | Linear crack, Raveling |
| Attain_SMP_WS_v2_000007.jpg | B | Alligator crack, Block crack, Linear crack, Pothole, Weathering | Linear crack | Alligator crack, Pothole |
| Attain_SMP_WS_v2_000008.jpg | A | Alligator crack, Block crack, Linear crack, Pothole, Weathering | Linear crack, Pothole | Alligator crack, Raveling |
| Attain_SMP_WS_v2_000009.jpg | B | Alligator crack, Block crack, Linear crack, Pothole, Weathering | Linear crack | Alligator crack, Pothole |
| Attain_SMP_WS_v2_000010.jpg | B | Alligator crack, Block crack, Linear crack, Pothole, Weathering | Linear crack | Alligator crack, Linear crack |
| Attain_SMP_WS_v2_000017.jpg | A | Linear crack, Pothole, Weathering | Linear crack | Alligator crack, Raveling |
| Attain_SMP_WS_v2_000018.jpg | A | Linear crack, Pothole, Weathering | Linear crack, Pothole | Alligator crack, Raveling |
| Attain_SMP_WS_v2_000019.jpg | A | Linear crack, Pothole, Weathering | Linear crack | Alligator crack, Raveling |
| Attain_SMP_WS_v2_000020.jpg | A | Linear crack, Pothole, Weathering | Linear crack | Alligator crack, Raveling |