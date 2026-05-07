# Baseline vs Fine-Tuned Comparison

## Stage 1 — Binary Detection (10,000 GAPs test images)

| Metric | Baseline | Fine-Tuned | Delta |
|---|---|---|---|
| accuracy | 76.53% | 88.39% | +11.86pp |
| precision_macro | 84.36% | 90.87% | +6.51pp |
| recall_macro | 70.92% | 85.85% | +14.93pp |
| f1_macro | 71.44% | 87.25% | +15.82pp |

## Stage 2 — Distress Type Classification (5,758 RDD test images)

| Metric | Baseline | Fine-Tuned | Delta |
|---|---|---|---|
| primary_accuracy | 48.66% | 66.06% | +17.40pp |
| f1_macro | 20.71% | 40.32% | +19.61pp |
| exact_match_rate | 34.40% | 52.76% | +18.36pp |

### Stage 2 per-class recall

| Class | Baseline | Fine-Tuned | Delta |
|---|---|---|---|
| Alligator Crack (D20) | 2.57% | 35.22% | +32.65pp |
| Longitudinal Crack (D00) | 72.50% | 71.78% | -0.72pp |
| Other Distress | 0.00% | 0.00% | +0.00pp |
| Pothole (D40) | 1.94% | 34.56% | +32.61pp |
| Transverse Crack (D10) | 0.00% | 1.72% | +1.72pp |
| Unparseable/Normal | 64.12% | 94.73% | +30.61pp |

## Promotion Decision

- Stage 1 accuracy delta: +11.86pp (threshold: +3.0pp)
- Stage 2 macro F1 delta: +19.61pp (threshold: +10.0pp)
- Normal precision delta: +12.42pp (must be >= -5.0pp)

**Recommendation: PROMOTE**

Reasons:
- Stage 1 accuracy improvement 11.86pp >= 3.0pp
- Stage 2 macro F1 improvement 19.61pp >= 10.0pp
- Normal-class precision delta 12.42pp within tolerance