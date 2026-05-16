# Attain WS_V2.0 — 3-way comparison (Improved-Baseline vs Fine-tuned)

**Subset:** WS_V2.0, 769 images, ground-truth pavement distress only (EXCLUDE filter applied).

## The three configurations

| Config | Prompts | Adapter |
|---|---|---|
| **A** Plain Baseline    | v1 (current production prompts) | none |
| **B** Improved Baseline | v2 (deep persona + stakes + protocol + taxonomy) | none |
| **C** Fine-tuned        | v1 (current production prompts) | LoRA `adapters/v2-rdd-2epochs-20260507` |

Decomposition of total improvement A→C:
  - Prompt-engineering contribution: **A → B** (improved prompts vs plain prompts)
  - Adapter contribution: **B → C** (what the adapter adds *on top of* the improved prompts — sign matters: negative means the adapter actively hurts vs improved baseline)

## Headline accuracy

| Metric | A: Plain Baseline | B: Improved Baseline | C: Fine-tuned | Prompt Δ (A→B) | Adapter Δ (B→C) |
|---|---|---|---|---|---|
| Tier 1 in-distribution (Linear, Alligator, Pothole) | 17.51% | 26.93% | 31.11% | **+9.42%** | **+4.18%** |
| Tier 2 zero-shot (Block, Raveling, Weathering) | 3.27% | 5.88% | 0.00% | **+2.61%** | **-5.88%** |
| Severity classification | 8.94% | 7.23% | 25.76% | **-1.71%** | **+18.53%** |
| Stage 1 distress prediction rate | 38.10% | 52.93% | 68.53% | **+14.82%** | **+15.60%** |
| Multi-class prediction rate | 6.63% | 52.54% | 16.91% | **+45.90%** | **-35.63%** |

## Per-class F1 — all three configurations

| Class | Tier | A: Plain | B: Improved | C: Fine-tuned | Prompt Δ | Adapter Δ |
|---|---|---|---|---|---|---|
| Alligator crack | in-dist | 0.018 | 0.055 | 0.217 | +0.038 | +0.162 |
| Block crack | zero-shot | 0.215 | 0.304 | 0.000 | +0.088 | -0.304 |
| Linear crack | in-dist | 0.479 | 0.637 | 0.677 | +0.157 | +0.040 |
| Pothole | in-dist | 0.196 | 0.265 | 0.000 | +0.069 | -0.265 |
| Raveling | zero-shot | 0.018 | 0.000 | 0.000 | -0.018 | +0.000 |
| Weathering | zero-shot | 0.024 | 0.063 | 0.000 | +0.039 | -0.063 |

## Per-class TP / FP / FN

| Class | A: Plain TP/FP/FN | B: Improved TP/FP/FN | C: Fine-tuned TP/FP/FN |
|---|---|---|---|
| Alligator crack | 5/0/556 | 16/1/545 | 69/5/492 |
| Block crack | 7/26/25 | 12/35/20 | 0/0/32 |
| Linear crack | 222/17/465 | 335/30/352 | 370/36/317 |
| Pothole | 20/21/143 | 29/27/134 | 0/0/163 |
| Raveling | 1/6/106 | 0/2/107 | 0/0/107 |
| Weathering | 2/0/165 | 6/18/161 | 0/0/167 |

## Pipeline-label emission counts (out of 769 images)

Reveals which classes each configuration prefers to emit:

| Pipeline label | A: Plain Baseline | B: Improved Baseline | C: Fine-tuned |
|---|---|---|---|
| Longitudinal Crack (D00) | 239 | 364 | 406 |
| Transverse Crack (D10) | 17 | 209 | 97 |
| Unknown | 0 | 0 | 80 |
| Alligator Crack (D20) | 5 | 17 | 74 |
| Rutting (D30) | 0 | 63 | 0 |
| Pothole (D40) | 41 | 56 | 0 |
| Block Crack (D43) | 33 | 47 | 0 |
| Rutting | 0 | 30 | 0 |
| Weathering/Oxidation | 2 | 20 | 0 |
| Raveling | 7 | 2 | 0 |
| Weathering/Oxidation (D70) | 0 | 4 | 0 |
| Rutting (D41) | 0 | 2 | 0 |
| Polishing | 0 | 1 | 0 |

## Head-to-head per-image

Each pairwise comparison: who recovered MORE of the ground-truth classes on each image.

### **A vs B** — prompt-engineering effect (Plain Baseline vs Improved Baseline, both no adapter)

- A strictly better: **3** (0.4%)
- B strictly better: **141** (18.3%)
- Tied (both correct on ≥1 class): 222
- Tied (both wrong/partial): 403

### **B vs C** — the prompt-vs-adapter test (Improved Baseline vs Fine-tuned)

- A strictly better: **73** (9.5%)
- B strictly better: **118** (15.3%)
- Tied (both correct on ≥1 class): 271
- Tied (both wrong/partial): 307

### **A vs C** — total fine-tune-plus-prompts effect (Plain Baseline vs Fine-tuned)

- A strictly better: **37** (4.8%)
- B strictly better: **214** (27.8%)
- Tied (both correct on ≥1 class): 191
- Tied (both wrong/partial): 327

## Hypothesis — final verdict

**Hypothesis being tested:** Improved Baseline (no adapter) >= Fine-tuned (with adapter)

- Aggregate metrics where Improved Baseline ≥ Fine-tuned: **1 / 3**
  - Tier 1 in-distribution: Improved Baseline wins = `False`
  - Tier 2 zero-shot: Improved Baseline wins = `True`
  - Severity: Improved Baseline wins = `False`
- Head-to-head per-image: Improved Baseline wins **73** vs Fine-tuned wins **118**

**Hypothesis supported overall:** `False`

## How to read this

- **Prompt Δ positive** → improved prompts help on this metric, regardless of adapter.
- **Adapter Δ positive** → adapter still adds value *even with the strongest prompts*. The improvement is real and not just confounded by prompts.
- **Adapter Δ negative** → adapter actively hurts when improved prompts are already in play. On these classes, the Improved Baseline pathway is strictly better.
- **Class-level signal trumps aggregate signal for routing decisions.** Even if the adapter wins on aggregate, it might be the wrong choice for specific class subsets — which is the argument for a hybrid router architecture.
