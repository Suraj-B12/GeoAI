# Two-Stage Pavement Distress Classification Using Vision-Language Models with Expert-in-the-Loop Incremental Learning

## Paper-Ready Sections + Technical Notes

> **Revision 2026-09-23.** Added: field-restricted Stage 2 confidence (§3.4), threshold
> calibration on 407 labelled images (§3.5, §6.5), and an input-resolution study (§3.8,
> §6.6, §7.6–7.8, T2.2–T2.5). Every figure in these sections is regenerated from stored
> result files by the scripts named beside it, without re-running inference.
>
> **Revision 2026-09-23 (b).** Added: a test of gating on the confidence of the *first* label
> alone and of a prompt rule asking for the most prominent distress first (§3.4, §6.7, §7.9).
> Neither was adopted; both are reported with the evidence. §7.7 adds a second allocator
> effect (image shape).
>
> **Revision 2026-09-24.** Added per-type probing for Stage 2 (§3.4.1), a prevalence-honest evaluation protocol with block-level splits, a recorded protocol and cluster-bootstrap intervals (§5.3), results on all 847 Attain WS_V2.0 frames and on 229 Bengaluru uploads (§6.8), and a discussion of why the probe runs in shadow mode (§7.10). Corrected: Attain's origin (Amirkabir University of Technology, Tehran, not New Zealand; §4.3), the IRC:82 mapping of block cracking (Transverse, §7.3.5.1) and patches (condition indicator, Tables 5.1–5.3), and a class-name parsing bug that had removed 167 patch instances from the zero-shot tally (Tier-2 accuracies 5.88% → 3.81% and 3.27% → 2.11%; ranking unchanged; §4.3).
>
> **Not yet revised:** §3.3–3.4 (prompts), §6.1–6.3 and §7.3–7.4 describe the pre-IRC:82
> taxonomy and a production pipeline that used the QLoRA adapter. Production has since
> moved to the base model with IRC:82-aligned prompts and no adapter; the adapter is kept
> for comparison only. Those sections need their own revision before submission.

---

# PART I: ACADEMIC PAPER SECTIONS

---

## 1. Introduction

Urban road infrastructure maintenance in developing cities like Bengaluru requires scalable, automated pavement condition assessment. Traditional methods rely on manual visual inspection by trained engineers, which is expensive, subjective, and cannot scale to cover thousands of kilometers of urban road network. Recent advances in Vision-Language Models (VLMs) present an opportunity to automate this process using nothing more than smartphone photographs.

This work presents a two-stage AI pipeline for pavement distress classification that:
1. Detects whether pavement damage exists (binary classification)
2. Identifies the specific distress type and severity (multi-class classification)
3. Flags low-confidence predictions for expert human review
4. Improves incrementally through expert corrections via few-shot prompt injection and LoRA incremental retraining

The key contributions are:
- A two-stage pipeline architecture using Qwen2.5-VL-7B-Instruct fine-tuned with QLoRA
- Logit-based confidence extraction (not heuristic) for reliable uncertainty quantification
- A comprehensive taxonomy injection approach enabling zero-shot recognition of distress types beyond the training distribution
- Cross-dataset generalization evaluation using geographically distinct datasets (RDD2022 from multiple countries; Attain, created by Amirkabir University of Technology, Tehran)
- An expert-in-the-loop feedback mechanism combining immediate few-shot improvement with permanent LoRA incremental retraining
- Per-type probing: one Yes/No question per IRC:82 type scored from the model's own next-token probabilities, sharing one cached image prefix (about 1 s for 20 questions). On 847 labelled frames it doubles macro MCC over a free-form type list (0.125 → 0.241, grouped cross-validation), with an image-level confidence that predicts its own errors (AUROC 0.78–0.87)
- A prevalence-honest evaluation protocol for multi-label distress classification (MCC against a constant baseline, block-level splits and bootstrap, a recorded protocol), which also exposed three scoring and labelling errors in earlier results

---

## 2. Related Work

### 2.1 Traditional Pavement Distress Detection

Conventional approaches use specialized hardware (laser profilers, ground-penetrating radar) or classical computer vision (edge detection, texture analysis) to identify road damage. These methods require dedicated vehicles and equipment, limiting scalability.

### 2.2 Deep Learning Approaches

YOLO-based object detection models (YOLOv5, YOLOv8) have been applied to pavement distress detection with strong results on in-distribution data. However, these models require bounding box annotations and are limited to the exact classes they were trained on, with no ability to recognize novel distress types.

### 2.3 Vision-Language Models for Infrastructure

Recent work has demonstrated the potential of VLMs for infrastructure inspection:
- **DamageQwen (2025):** Uses the same Qwen2.5-VL model family for damage assessment, showing that few-shot prompting improves performance by 18% over zero-shot baselines.
- **Xu et al. (2025):** Demonstrated that zero-shot LLM-based pavement assessment can match or exceed expert-level performance on the Pavement Surface Condition Index (PSCI).
- **Yong et al. (2023):** Showed that prompt engineering with VLMs can outperform supervised baselines on infrastructure inspection tasks.

Our work builds on these findings by combining fine-tuning (QLoRA) with taxonomy injection for open-world recognition, evaluated rigorously across both in-distribution and cross-dataset benchmarks.

---

## 3. Methodology

### 3.1 Model Architecture

We use **Qwen2.5-VL-7B-Instruct** (Alibaba Group, 2024), a 7-billion parameter Vision-Language Model based on the Qwen2.5 architecture. The model processes visual inputs through a Vision Transformer (ViT) encoder and generates text outputs through an autoregressive language model decoder, connected via a cross-attention bridge.

**Model specifications:**
| Parameter | Value |
|---|---|
| Architecture | Qwen2.5-VL (ViT encoder + LLM decoder) |
| Total Parameters | ~7.6 billion |
| Vision Encoder | ViT with dynamic resolution support |
| Language Model | Qwen2.5-7B-Instruct (decoder-only transformer) |
| Context Window | 32,768 tokens |
| Image Resolution | Dynamic (min 7,168 pixels to max 4,840,000 pixels) |
| Pre-training Data | Multi-modal web data (images, text, video) |
| Instruction Tuning | Chat-optimized with RLHF |

**Why Qwen2.5-VL-7B over alternatives:**
- Native dynamic resolution: handles 160x160 GAPs images and 512x512 RDD images without resizing artifacts
- Strong visual grounding: pre-trained on diverse visual domains including infrastructure and outdoor scenes
- Efficient fine-tuning: compatible with QLoRA 4-bit quantization, fitting within 24GB VRAM budget
- Structured output: instruction-tuned to follow structured response formats reliably

### 3.2 Two-Stage Pipeline Architecture

```
Input Image --> Stage 1: Binary Detection --> Is Distressed?
                                                  |
                                     No: Return "Normal"
                                     Yes: Continue to Stage 2
                                                  |
                                              Stage 2: Type Classification
                                                  |
                                     Parse: distress types, severity, description
                                                  |
                                     Confidence < 80%? --> Flag for expert review
                                                  |
                                     Return structured classification result
```

**Rationale for two-stage design:**
- **Computational efficiency:** Stage 2 inference is skipped entirely for normal pavement (approximately 60% of real-world images), halving average inference cost.
- **Independent confidence tracking:** Each stage produces its own confidence score. A high-confidence "Distress" detection followed by a low-confidence type classification correctly triggers expert review for type uncertainty only.
- **Modular fine-tuning:** Each stage can be fine-tuned independently on its respective dataset (GAPs for Stage 1, RDD for Stage 2) without interference.

### 3.3 Stage 1: Binary Detection

**Task:** Classify a pavement image as "Normal" (no damage) or "Distress" (damage present).

**System Prompt:**
> You are an expert pavement condition inspector. Your task is to examine pavement images and determine whether the pavement shows any signs of distress or damage. Respond with exactly one word: Normal or Distress.

**User Prompt:**
> Examine this pavement image carefully. Is this pavement in normal condition, or does it show signs of distress? Respond with exactly one word: Normal or Distress.

**Confidence Extraction (Stage 1):**
The confidence score is computed from the model's raw logits, not from string matching or heuristic rules:
1. After generation, extract the logit vector for the **first generated token** (shape: `vocab_size`)
2. Identify the token IDs for "Normal" and "Distress" first sub-tokens (cached at model load time)
3. Apply softmax over only these two logits: `P(Normal)`, `P(Distress)`
4. Return the probability corresponding to whichever class the model actually generated

This binary softmax approach gives a calibrated probability in [0, 1] that directly represents the model's certainty between the two classes.

**Unparseable response handling:**
If the model generates a response that cannot be parsed as either "Normal" or "Distress", the pipeline defaults to **Distress with 0% confidence**. This design decision errs on the side of caution: a damaged road incorrectly classified as normal is a safety hazard, while a normal road sent for expert review wastes only human time. The 0% confidence guarantees the result is flagged for expert review.

### 3.4 Stage 2: Distress Type Classification

**Task:** Given a pavement image showing distress, identify the specific type(s) of damage, estimate severity, and provide a description.

**Taxonomy Injection:**
The Stage 2 system prompt includes a comprehensive taxonomy of 20+ pavement distress types organized into categories (Cracking, Surface Deformation, Surface Defects, Patches and Repairs, Joint Defects, Other). This taxonomy injection serves two purposes:
1. **Grounding:** Ensures the model outputs standardized distress type names that can be programmatically parsed
2. **Open-world recognition:** Enables the model to recognize distress types beyond the RDD training labels (e.g., block crack, raveling, weathering) by leveraging the VLM's pre-trained visual-semantic alignment with textual descriptions

**Few-Shot Examples in Prompt:**
Three text-only examples are included in the system prompt demonstrating the expected output format:
- Example 1: Single distress type (Longitudinal Crack, Low severity)
- Example 2: Multiple distress types (Alligator Crack + Pothole, High severity)
- Example 3: Mixed types (Transverse Crack + Inlaid Patch, Medium severity)

These examples establish the structured output format without using actual training images.

**Output Format:**
```
DISTRESS_TYPES: <comma-separated list>
SEVERITY: <Low / Medium / High>
DESCRIPTION: <one sentence description>
```

**Confidence Extraction (Stage 2):**
Stage 2 uses the **geometric mean of per-token probabilities, restricted to the tokens that carry the classification**:
1. For each generated token `t_i`, compute `log P(t_i | t_1, ..., t_{i-1})` from the logits
2. Locate the token span covering the value of the `DISTRESS_TYPES:` field
3. Average the log-probabilities over that span only and exponentiate: `confidence = exp(mean_log_prob)`

Unlike Stage 1's binary softmax, Stage 2 produces a multi-token structured response, so a sequence likelihood is the natural measure. What matters is which tokens enter it. Locating the span is not trivial: byte-level BPE splits multi-byte characters across token boundaries, so decoding token by token gives unreliable character offsets. We decode cumulative prefixes, which yields an exact character offset at every token boundary, and map the field's character range onto a token range. If the span cannot be located, or covers fewer than two tokens, the whole-sequence value is used and the fallback is recorded. Across 456 predictions (49 production, 407 calibration) the span was located every time.

**Why the span is restricted.** An earlier version averaged over the entire response, including the free-text `DESCRIPTION` field. That prose is high-entropy: a fluent but wrong description reads as high-probability text, and a hedged but correct one reads as low. Measured against ground truth (§6.5), the whole-sequence score ranked wrong predictions *above* correct ones (AUC 0.373, below the 0.5 of chance). Both variants are computed and stored for every prediction, so the comparison can be repeated as labelled data accumulates. Implementation: `scripts/confidence.py`, shared by the production classifier and the evaluation scripts.

**Per-label confidence.** The same character-to-token mapping locates each comma-separated label inside the `DISTRESS_TYPES` value. For each label we record its *joint* probability, `exp(Σ log p_i)` over its tokens: the probability the model gave to that exact label. The geometric mean over its tokens is recorded for comparison but is biased upward for long names, whose tokens after the first are close to certain. The first label's joint probability is unconditional. Every later label's probability is conditional on the labels already written, so it cannot be ranked against the first. The label the model lists first is shown to operators as the main distress, with the others beside it. Its confidence (the *primary* confidence) is stored on every prediction. It was evaluated as a replacement for the whole-field gate and not adopted (§6.7).

**Response Parsing:**
The parser extracts structured fields using prefix matching (`DISTRESS_TYPES:`, `SEVERITY:`, `DESCRIPTION:`). If structured parsing fails, a fallback keyword extraction scans for known distress type names in the raw text. If no distress types can be extracted, the result is labeled "Unknown" — which always falls below the 80% confidence threshold and triggers expert review.

#### 3.4.1 Per-Type Probing

The free-form list has a structural weakness: the model writes it in the order of the prompt's inspection protocol, and every label after the first is conditioned on the ones already written. On 407 labelled Attain images the list was exactly "Longitudinal Cracking, Transverse Cracking" for 318 images (78%), the first two items of the protocol (§6.8). A list produced this way reflects decoding order as much as the image.

**Probing.** Instead of asking the model to *write* the types, we *ask about* each type separately. For every IRC:82 type, and for the condition indicator Patching (§4.3), the model receives the same system prompt and image followed by one question — "Does this road photo show *Alligator Cracking* (IRC:82 §7.3.3)? Definition: … What it looks like: … Answer Yes or No." — and we read the next-token distribution. The type's score is

`P(yes) = (p(Yes) + p(yes)) / (p(Yes) + p(yes) + p(No) + p(no))`

at that single position (summed over the tokenizer's single-token spellings of each answer): the same logit read-out Stage 1 uses for Normal against Distress. The Yes/No logits are recomputed in float32 from the final hidden state, because bf16 rounding quantises P(yes) into visible steps and creates ties that an ROC curve cannot rank. Across the 847 evaluation images and every prompt variant, at least 99.93% of the next-token probability fell on the Yes/No answer tokens, so the model answers the questions as asked.

This gives every type its own probability, whether or not the model would have chosen to write it. Each type can then have its own ROC curve and threshold. No type's answer depends on another type's.

**Cost.** The system prompt and the image make up more than 90% of every question's tokens. They are encoded once. The resulting key/value cache is shared by all 20 questions, which then run as one batch of short suffixes. Qwen2.5-VL uses three-dimensional multimodal rotary positions, so suffix positions are taken from the model's own `get_rope_index` over the full sequence. The code asserts that they continue the prefix as plain text positions. The suffix batch is chunked to a 1.5 GB budget for its copies of the prefix cache: on a card shared with the production server, an unbounded batch pushed the process past physical memory, where Windows pages to system RAM instead of failing (1.1 s → 23–111 s per image). A self-test at load compares this path with one full forward pass per question. The largest difference in P(yes) was 0.014. With name-only questions all 20 take a median 1.08 s per Attain frame (1.43 s per 1024² Bengaluru upload), against a median 8.18 s for the free-form generation; with the full IRC definition in each question, 1.64 s.

**Decision rule (probe-verified list).** Thresholds need labelled data, which exists only for the types Attain annotates. The rule therefore treats types differently:
- *Calibrated types* (longitudinal and transverse cracking, alligator cracking, potholes, ravelling, hungry surface, and the Patching indicator): the probe decides. A type is reported when its P(yes) reaches the threshold tuned on the development split, whatever the free-form list said.
- *The other twelve IRC types* (bleeding, rutting, edge breaking, …): no threshold can be justified without labels. The probe may only remove them. A type the free-form list named is kept if the probe gives it P(yes) ≥ 0.5, and never added on the probe's word alone.
- *Condition indicators* never enter the distress list and are reported separately.

Types are ordered by calibrated probability, so the first label is the type the model is most confident is present. It is not a measured "most prominent" distress (§6.7). The free-form generation still runs, because severity and the one-sentence description come from it.

**Confidence.** Each calibrated type's P(yes) is Platt-scaled on the development split into q. The image-level confidence is the probability that every calibrated decision is right, treating decisions as independent: ∏ (q if reported, else 1 − q). If Stage 1 said "distressed" but no type survives, the confidence is set to 0 so the image goes to review.

**Deployment: shadow mode and a Stage 1 safety net.** The thresholds are tuned on Attain, whose vehicle-mounted road strips are a different kind of photograph from Bengaluru's handheld close-ups. The probe's probabilities move with the domain in sensible directions: the median P(yes) for potholes is 0.60 on Bengaluru uploads and 0.02 on Attain frames, and for longitudinal cracking 0.10 and 0.69. But the decision thresholds do not carry over. Applied to 204 production uploads, the Attain thresholds would add Ravelling to 197 of them (§6.8). Without labelled Bengaluru photographs there is no way to set thresholds there, so production runs the probe in **shadow mode**. Every upload's P(yes) for every type is stored with the row, while the reported types and the review gate stay those of the free-form list. When experts have reviewed enough uploads, `scripts/calibrate_probe_from_expert_labels.py` fits Bengaluru thresholds from the stored probabilities without re-running the model. Its candidate replaces the configuration only if it beats the free-form list on held-out expert labels.

One use of the probe does not depend on thresholds being right, because it can only add human review. Stage 1 missed 48% of the Attain images that carry annotated distress. With the **Stage 1 safety net**, an image Stage 1 calls Normal is checked by the probe. If any of the headline types reaches its threshold, the image goes to expert review instead of being auto-classified Normal. It never changes a label.

**Fail-safes.** A missing or invalid configuration, or a failed parity self-test, disables probing at load and the pipeline runs the free-form list. The reason is exposed on `/health`. An exception on one image keeps the free-form answer for that image and records the error in the row. Every row stores each type's P(yes), what the probe added and removed, and the free-form list it started from, so any row can be re-scored under a new threshold without re-inference. The rule is implemented once (`scripts/stage2_probe_rules.py`) and called by both production and the evaluation, so reported test numbers are what production computes.

### 3.5 Expert Review Threshold

The confidence threshold is set at **80%** (0.80), defined once in `scripts/utils.py` and imported by all components. Any prediction where either Stage 1 or Stage 2 confidence falls below this threshold is flagged with `needs_expert_review = true`.

**Calibration.** The value 0.80 was originally set a priori. It was then tested against ground truth on 407 Attain images (§6.5). Production confidences run lower than Attain's (mean 0.746 against 0.875), so an absolute threshold does not transfer between them; thresholds were instead compared at matched review load — the Attain threshold that sends the same fraction of images to review as the candidate does in production. At the review load implied by 0.80 in production (76%), the matched operating point auto-accepted wrong predictions at a rate of 3.0% (95% CI 1–8%) and caught 57 of 60 errors. Lowering to 0.75 (63% review load) doubled the auto-accept error rate to 6.0% (3–11%) and caught 51 of 60. The intervals overlap, so the difference is not statistically established at this sample size; every point estimate nevertheless favours 0.80, and it is retained.

### 3.6 Fine-Tuning Strategy (QLoRA)

Fine-tuning uses **QLoRA** (Quantized Low-Rank Adaptation) via the LLaMA-Factory framework, which applies low-rank adapter matrices to a 4-bit quantized base model. This enables training a 7B model on a single 24GB GPU.

**Training Configuration:**

| Hyperparameter | Value | Rationale |
|---|---|---|
| LoRA Rank | 64 | Higher rank captures more domain-specific patterns |
| LoRA Alpha | 128 | Alpha = 2 * rank, standard scaling |
| LoRA Dropout | 0.1 | Sparsity regularizer, critical for preventing overfitting |
| LoRA Target | all | Apply adapters to all linear layers |
| Quantization | 4-bit NF4 with double quantization | Minimum VRAM footprint |
| Compute Dtype | bfloat16 | Avoids fp16 overflow in attention |
| Learning Rate | 1e-4 | Standard for QLoRA with this model size |
| LR Scheduler | Cosine with 10% warmup | Smooth decay prevents catastrophic forgetting |
| Epochs | 2 (maximum) | Research shows 50k+ sample datasets overfit at 3+ epochs |
| Batch Size | 4 per device * 8 gradient accumulation = **32 effective** | Stable gradient estimates |
| Weight Decay | 0.05 | L2 regularization against overfitting |
| Label Smoothing | 0.1 | Prevents overconfident predictions |
| NEFTune Noise Alpha | 5.0 | Noisy embedding fine-tuning improves generalization |
| Optimizer | AdamW (8-bit) | Memory-efficient optimizer |
| Gradient Checkpointing | Enabled | Trades compute for VRAM savings |
| Vision Tower | Frozen | Standard for VLM LoRA — fine-tune language model only |
| Template | `qwen2_5_vl` | Must match model architecture exactly (NOT `qwen2_vl`) |
| Max Sequence Length | 2,048 tokens | Sufficient for image tokens + structured response |

**Anti-Overfitting Stack:**
Five independent regularization mechanisms work together:
1. **Weight decay (0.05)** — L2 penalty on parameter magnitudes
2. **Label smoothing (0.1)** — softens target distribution, prevents overconfident logits
3. **LoRA dropout (0.1)** — randomly zeros adapter activations during training
4. **NEFTune noise (5.0)** — adds calibrated noise to embedding vectors
5. **Early stopping on eval_loss** — monitors validation loss every 500 steps, loads best checkpoint

**Critical Technical Constraint:**
After QLoRA training, the adapter must **never** be merged into the base model via `merge_and_unload()`. This is a known bug in PEFT (issue #2586) where merging 4-bit quantized weights with LoRA adapters produces corrupted weights. The model is kept as a `PeftModel` for inference, with the adapter loaded on top of the frozen quantized base.

### 3.7 Acceleration and Hardware Optimization

| Technique | What It Does | Requirement |
|---|---|---|
| Flash Attention 2 | Fused attention kernel, O(N) memory instead of O(N^2) | Ampere+ GPU (compute >= 8.0) + flash-attn package |
| Liger Kernel | Fused MLP + RMS-norm kernels for faster forward/backward | Linux only (requires Triton) |
| 8-bit AdamW | Quantizes optimizer states to 8-bit | bitsandbytes >= 0.44.1 |
| Gradient Checkpointing | Recomputes activations instead of storing them | Built into transformers |
| SDPA Fallback | PyTorch native scaled dot-product attention | Universal (no extra deps) |

**Runtime auto-detection:** The system automatically detects GPU compute capability and installed packages at startup, selecting the best available attention implementation (Flash Attention 2 > SDPA > Eager) without manual configuration.

### 3.8 Input Resolution Budget

Qwen2.5-VL tokenises an image into one visual token per 28×28 pixel block, so a 1736×1736 smartphone photograph becomes 3,844 visual tokens before any text is added. The processor's `max_pixels` parameter caps this: an image above the budget is resized, preserving aspect ratio and rounding each side to a multiple of 28, until its area fits. JPEG compression, by contrast, does not change inference cost, because images are decoded to pixels before tokenisation.

We evaluate caps of 1280², 1024², 768², 640² and 512² pixels against full resolution. Each downscaled view is produced with the processor's own `smart_resize` routine and bicubic resampling, so it is identical to what the production processor would produce under that cap. Two datasets answer different questions:

- **Bengaluru uploads** — 57 citizen photographs that passed the pavement pre-filter, predominantly 1512 or 1728 px square. No labels exist, so this measures *change*: does the answer at a lower resolution match the answer at full resolution?
- **Attain** — 200 labelled images. This measures *loss*: when an answer changes, is it a worse one? Attain frames come in three sizes (1484×504, 644×644 and 1932×1092); the 1280² and 1024² caps change only the 16 largest of the 200 frames, 768² changes 136, and 640² and 512² change all 200. Attain's evidence is therefore strongest at 768² and below.

Acceptance criteria were fixed before the run: at most 2% of images flipping from Distress to Normal at Stage 1; Stage 1 label agreement ≥ 95%; mean Jaccard similarity of the Stage 2 distress-type set against full resolution ≥ 0.80; and on Attain, no significant accuracy drop (paired exact McNemar test, α = 0.05) and no point-estimate drop above 2 percentage points.

After the first four images showed the second distress label changing even at mild caps, two control arms were added to separate lost detail from model instability. The 1024² view was re-encoded as JPEG at quality 95 — visually identical, with a mean change under 2 grey levels on sampled photographs — and at quality 75, one step below the quality of about 80 at which uploads are already stored (estimated from the files' quantisation tables for 56 of 57 photographs). Disagreement between the quality-95 arm and the plain 1024² view measures how much the answer changes under a perturbation that removes no information. The acceptance criteria were not relaxed when these arms were added; both are reported in §6.6.

Timing was measured separately, with the allocator cache cleared and peak-memory counters reset before every pass, across all three worker stages (pavement pre-filter, detection and classification). Scripts: `scripts/resolution_ab.py` (accuracy and agreement), `scripts/resolution_timing.py` (timing and memory).

---

## 4. Datasets

### 4.1 GAPs V2 (Stage 1 Training + Evaluation)

| Property | Value |
|---|---|
| Source | German Asphalt Pavement Distress (GAPs) dataset, version 2 |
| Purpose | Stage 1 binary detection (Normal vs Distress) |
| Format | NumPy arrays (.npy), shape (N, 1, 160, 160), float32 grayscale |
| Image Size | 160 x 160 pixels (grayscale, converted to RGB for model input) |
| Total Images | 80,000 (with additional valid-test split) |
| Training Split | 50,000 images |
| Validation Split | 10,000 images |
| Test Split | 10,000 images |
| Class Distribution | ~60% Normal, ~40% Distress |
| Collection Method | Automated pavement imaging vehicle, German highways |
| Preprocessing | .npy chunks loaded via memory-mapping, converted to RGB PNGs with parallel workers auto-scaled to available CPU cores |

**Class imbalance note:** The 60/40 Normal/Distress split reflects real-world distribution where the majority of road surface is undamaged. No oversampling or class weighting is applied — the model must learn to handle this imbalance naturally.

### 4.2 RDD2022 (Stage 2 Training + In-Distribution Evaluation)

| Property | Value |
|---|---|
| Source | Road Damage Dataset 2022 (IEEE Big Data Cup Challenge) |
| Purpose | Stage 2 distress type classification |
| Format | JPEG images + YOLO-format label files (.txt) |
| Image Size | Primarily 512 x 512 pixels (color) |
| Training Split | 26,869 images |
| Validation Split | 5,758 images |
| Test Split | 5,758 images |
| Classes | 4: D00 (Longitudinal Crack), D10 (Transverse Crack), D20 (Alligator Crack), D40 (Pothole) |
| Collection | Smartphone-mounted on vehicles across Japan, India, Czech Republic, Norway, United States, China |
| Annotations | Bounding box annotations in YOLO format; for VLM training, converted to image-level multi-label classification |
| Multi-label | A single image may contain multiple distress types (e.g., D00 + D40) |

**Label conversion for VLM training:**
The original YOLO bounding box labels are converted to image-level classification labels. For each image, the unique set of class IDs present in the label file becomes the ground truth. This converts the object detection task into a multi-label classification task suitable for VLM fine-tuning:
```
YOLO labels: [D00 x1 y1 w1 h1, D00 x2 y2 w2 h2, D40 x3 y3 w3 h3]
  --> VLM label: "Longitudinal Crack (D00), Pothole (D40)" with heuristic severity
```

**Training data format (ShareGPT/LLaMA-Factory):**
Training conversations follow the ShareGPT format for multi-modal instruction tuning:
```json
{
  "images": ["path/to/image.jpg"],
  "conversations": [
    {"from": "human", "value": "<image>\nAnalyze this pavement image..."},
    {"from": "gpt", "value": "DISTRESS_TYPES: Longitudinal Crack (D00)\nSEVERITY: Low\nDESCRIPTION: ..."}
  ]
}
```

**Severity heuristic for training labels:**
Since RDD2022 does not include severity annotations, severity is derived heuristically:
- **High:** Pothole present OR >= 5 total annotations in image
- **Medium:** >= 2 distinct distress types OR >= 3 total annotations
- **Low:** Single type with 1-2 annotations

### 4.3 Attain Dataset (Stage 2 Cross-Dataset Evaluation Only)

| Property | Value |
|---|---|
| Source | Mendeley Data (doi:10.17632/nykrzdm74f/1) |
| License | CC BY 4.0 |
| Purpose | Cross-dataset zero-shot generalization evaluation (NOT used for training) |
| Images | 2,293 images with 19,761 annotated distress instances across three subsets; every evaluation here uses **WS_V2.0** (847 images) |
| Frames (WS_V2.0) | 403 at 1479×508, 275 at 1920×1080, 169 at 640×640, interleaved; consecutive frames from a moving vehicle, so neighbours are near-duplicates |
| Classes | 10 types (see mapping below); longitudinal and transverse cracks are one merged "Linear crack" class |
| Severity | High / Low per instance in WS_V2.0 (unlike RDD, which has no severity labels) |
| Collection | Smartphone cameras mounted on a vehicle's front and rear windshields |
| Creator | Amirkabir University of Technology, Tehran (Mendeley record; the dataset's data file links attain.aut.ac.ir). The record does not state the collection site. |

**Correction.** Earlier drafts of this paper described Attain as New Zealand footage. Nothing in the dataset record supports that. The creating institution is in Tehran, and the road furniture in the frames is consistent with Iran. The cross-geography framing still holds (Iran and India, both different from RDD's training mix); the country named was wrong.

**Attain-to-Pipeline Class Mapping:**

| Attain Class | Pipeline Taxonomy Match | In RDD Training? | Evaluation Role |
|---|---|---|---|
| Alligator Crack | Alligator Crack (D20) | YES | In-distribution transfer |
| Longitudinal Crack | Longitudinal Crack (D00) | YES | In-distribution transfer |
| Transverse Crack | Transverse Crack (D10) | YES | In-distribution transfer |
| Pothole | Pothole (D40) | YES | In-distribution transfer |
| Block Crack | Transverse Cracking (IRC §7.3.5.1) | NO | **Zero-shot evaluation** (not separately reportable, see below) |
| Patch/Utility Cut | Patching (IRC condition indicator, Tables 5.1–5.3) | NO | **Zero-shot evaluation** |
| Weathering | Hungry Surface (IRC §7.2.4) | NO | **Zero-shot evaluation** |
| Raveling | Ravelling (IRC §7.5.2) | NO | **Zero-shot evaluation** |
| Faded Marking | (not pavement distress) | - | EXCLUDED |
| Lane/Shoulder Drop-off | (not pavement distress) | - | EXCLUDED |
| Manhole | (not pavement distress) | - | EXCLUDED |

**Exclusion rationale:** Faded markings, lane/shoulder drop-offs, and manholes are road features, not pavement structural damage. Including them would inflate false positive rates and dilute the evaluation's focus on actual distress classification. The 78 WS_V2.0 images whose only annotations are such features (or none) were dropped by the earlier evaluations. The probing study (§6.8) keeps them: they are the only images that are negative for every distress class at once.

**Two mapping corrections (2026-09-23), both against the IRC:82 text.**
1. *Block cracking is transverse cracking in IRC:82.* §7.3.5.1 defines transverse cracks as cracks "in the transverse directions or as interconnected cracks forming series of large blocks perpendicular to the direction of the road". Alligator cracking (§7.3.3) is defined by *small* irregular blocks. The canonicaliser had sent "block crack" to Alligator Cracking as the "closest visible analog", which the standard contradicts; it now maps to Transverse Cracking. A consequence is that an IRC label cannot say "block" at all, so Attain's Block crack class is not reportable under the IRC taxonomy by either method. The per-type probe asks about the block pattern as a diagnostic (§6.8).
2. *Patches are rated, not ignored.* IRC:82 Tables 5.1–5.3 rate pavement condition from cracking, ravelling, potholes, shoving, settlement, rut depth — and patching (% of area). A patch is a repair, not a Section 7 distress, so it is added as a separate *condition indicator* rather than a 19th distress type: it is reported beside the distress list, never in it.

**A scoring bug, found and corrected.** Attain spells one class `Patch and utility cut- Low` (no space before the dash). The class parser split only on `" - "`, so every patch instance became an unmapped pseudo-class and silently left the zero-shot tally. Re-scoring the stored per-image predictions with the fixed parser adds 167 patch instances to the zero-shot denominator (306 → 473). The three-way comparison's zero-shot accuracies fall from 5.88% to 3.81% (Improved Baseline), 3.27% to 2.11% (plain prompts) and stay 0% (fine-tuned). The ranking is unchanged. No method predicted a patch in any of those runs, so no per-class F1 changes.

**Three-tier evaluation significance:**
1. **Known classes (4):** Trained on RDD, tested on Attain — measures cross-geography generalization
2. **Unknown classes (3):** Never trained, recognized only through taxonomy injection — measures zero-shot capability
3. **Comparison:** Accuracy gap between known and unknown classes quantifies the value of fine-tuning vs taxonomy injection alone

---

## 5. Experimental Setup

### 5.1 Hardware

| Component | Training (College Lab) | Development/Inference (Laptop) |
|---|---|---|
| GPU | NVIDIA RTX A5000 (24 GB VRAM) | NVIDIA RTX 4050 (6 GB VRAM) |
| CPU | Multi-core workstation | Laptop-class |
| RAM | Sufficient for 7B fp16 | Limited |
| Storage | External SSD (transferred between machines) | Same SSD |
| Model Config | 7B, fp16 (no quantization), ~14 GB VRAM | 3B, 4-bit quantization, ~2 GB VRAM |

**Hardware constraints drove key design decisions:**
- 7B model does NOT fit on 6GB VRAM even with 4-bit quantization (OOM error confirmed empirically)
- Development testing used Qwen2.5-VL-3B-Instruct as a functional proxy
- All reported baseline and fine-tuned metrics use the 7B model on RTX A5000
- QLoRA enables training 7B within the 24GB VRAM budget of the A5000

### 5.2 Software Environment

| Package | Version | Purpose |
|---|---|---|
| PyTorch | CUDA 12.1 build | GPU computation |
| transformers | 4.57.6 (pinned) | Model loading + inference |
| accelerate | 0.34.2 (pinned) | Multi-device model placement |
| bitsandbytes | 0.44.1 (pinned) | 4-bit/8-bit quantization |
| peft | 0.14.0 (pinned) | LoRA adapter management |
| LLaMA-Factory | Latest (git clone) | Fine-tuning orchestration |
| scikit-learn | Latest | Evaluation metrics |
| FastAPI | >= 0.110.0 | REST API server |
| sse-starlette | >= 1.6.0 | Server-Sent Events for streaming |

**Version pinning rationale:** `transformers >= 4.52` introduced a `Params4bit` compatibility bug with `bitsandbytes` that causes model loading to fail. The pinned combination (transformers 4.57.6 + bitsandbytes 0.44.1 + accelerate 0.34.2 + peft 0.14.0) is empirically verified on both laptop and A5000 hardware.

### 5.3 Evaluation Protocol

**Stage 1 Evaluation:**
- Dataset: GAPs V2 test split (10,000 images)
- Metric: Accuracy, Precision (macro), Recall (macro), F1-score (macro)
- Per-class: Normal precision/recall/F1, Distress precision/recall/F1
- Visualization: Confusion matrix heatmap

**Stage 2 Evaluation:**
- Dataset: RDD2022 test split (5,758 images)
- Primary metric: Top-1 accuracy on primary (first) distress type
- Secondary metric: Exact match rate (all predicted types match all ground truth types)
- Per-class: Per-distress-type precision/recall/F1
- Visualization: Confusion matrix heatmap
- Checkpoint resumption: Progress saved every 50 images to survive system crashes/reboots

**Cross-Dataset Evaluation (Phase 2):**
- Dataset: Attain (2,293 images, excluding non-distress classes)
- Same metrics as Stage 2
- Additional: Separate accuracy for known classes (trained) vs unknown classes (zero-shot)

**Inference settings (all evaluations):**
| Parameter | Value |
|---|---|
| do_sample | False (greedy decoding) |
| temperature | None |
| top_p | None |
| max_new_tokens (Stage 1) | 20 |
| max_new_tokens (Stage 2) | 150 |
| Image preprocessing | Convert to RGB, no resize (dynamic resolution) |

Greedy decoding ensures deterministic, reproducible results across runs.

**Stage 2 probing study (§6.8).** Every Attain WS_V2.0 frame (847, including the 78 with no distress annotation) is run once through production Stage 1, production Stage 2 and three probe variants, in one process on the same decoded image. The comparison is multi-label and the classes are very unequally common: 81% of frames carry Linear crack and 66% Alligator crack. A predictor that says "yes" to both on every frame scores F1 0.90 and 0.80 without looking at the image. The headline metric is therefore the **Matthews correlation coefficient** (MCC), which is 0 for any constant predictor whatever the prevalence, macro-averaged over the five classes both methods can express (Linear crack, Alligator crack, Pothole, Raveling, Weathering). F1, balanced accuracy, per-image Jaccard and exact-set agreement are reported beside it, always next to the constant predictor's score. Attain frames are consecutive shots from a moving vehicle, so neighbouring frames are near-duplicates. All splits and confidence intervals therefore work on **blocks of 20 consecutive frames**: the development/test split assigns whole blocks, and 95% intervals come from a cluster bootstrap that resamples blocks (B = 2,000; paired for differences). The protocol and a SHA-256 hash of the analysis code were recorded before any test number existed (`eval_results/stage2_probe_preregistration.json`). It fixed the variant choice (best development macro AUROC), the threshold rule (MCC-optimal per class on the development split) and the adoption rule: the probe is adopted if the paired macro-MCC difference on the test split is positive with a 95% interval excluding 0, and no class is significantly worse. A second, grouped 5-fold cross-validation over all 847 frames was added after the test results, with its rules recorded in an addendum before it ran.

---

## 6. Results

### 6.1 Baseline Results (Pre-Fine-Tuning)

All baseline results use the **raw Qwen2.5-VL-7B-Instruct** model with no fine-tuning, no adapter, evaluated in fp16 (no quantization) on RTX A5000.

#### 6.1.1 Stage 1: Binary Detection (GAPs V2 Test Set)

| Metric | Value |
|---|---|
| **Overall Accuracy** | **76.53%** |
| Precision (macro) | 84.36% |
| Recall (macro) | 70.92% |
| F1-Score (macro) | 71.44% |
| Unparseable Responses | 0 / 10,000 |
| Average Inference Time | 2.08 s/image |

**Per-Class Breakdown:**

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Normal | 72.21% | **98.97%** | 83.50% | 6,000 |
| Distress | **96.51%** | 42.88% | 59.37% | 4,000 |
| **Macro Avg** | **84.36%** | **70.92%** | **71.44%** | **10,000** |
| Weighted Avg | 81.93% | 76.53% | 73.85% | 10,000 |

**Confusion Matrix:**

|  | Predicted Normal | Predicted Distress |
|---|---|---|
| **Actual Normal** | 5,938 (TN) | 62 (FP) |
| **Actual Distress** | 2,285 (FN) | 1,715 (TP) |

**Analysis:**

The base model exhibits a strong **Normal bias**: it correctly identifies 98.97% of normal pavement but misses 57.12% of distressed pavement (2,285 false negatives). This is the worst possible failure mode for a safety-critical application: damaged roads silently classified as normal.

Key observations:
- **High Distress precision (96.51%):** When the model says "Distress," it is almost always correct. The problem is that it rarely says "Distress."
- **Low Distress recall (42.88%):** The model treats ambiguous or mild damage as normal. This is consistent with the pre-training distribution where "normal" surfaces dominate web imagery.
- **Zero unparseable responses:** The model consistently follows the "one word" instruction format, demonstrating strong instruction compliance.
- **2.08s/image inference speed:** 160x160 grayscale images are very small (25,600 pixels); inference is dominated by model loading overhead, not image processing.

**Implication for fine-tuning:** Fine-tuning should shift the decision boundary to increase Distress recall, potentially at the cost of some Normal precision. The goal is balanced detection, not normal-biased detection.

#### 6.1.2 Stage 2: Distress Type Classification (RDD2022 Test Set)

| Metric | Value |
|---|---|
| **Primary Accuracy** | **48.66%** |
| F1-Score (macro) | 20.71% |
| Exact Match Rate | 34.40% |
| Average Inference Time | 5.32 s/image |

**Per-Class Breakdown:**

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Unparseable/Normal (-1) | 58.19% | 64.12% | 61.01% | 1,973 |
| Longitudinal Crack (D00) | 45.16% | **72.50%** | 55.66% | 2,080 |
| Transverse Crack (D10) | 0.00% | 0.00% | 0.00% | 464 |
| Alligator Crack (D20) | 64.52% | 2.57% | 4.94% | 778 |
| Pothole (D40) | 4.21% | 1.94% | 2.66% | 463 |
| Other Distress | 0.00% | 0.00% | 0.00% | 0 |
| **Macro Avg** | **28.68%** | **23.52%** | **20.71%** | **5,758** |

**Confusion Matrix:**

|  | Pred: Unparse | Pred: D00 | Pred: D10 | Pred: D20 | Pred: D40 | Pred: Other |
|---|---|---|---|---|---|---|
| **GT: Unparseable** | 1,265 | 615 | 0 | 0 | 93 | 0 |
| **GT: D00 (Long.)** | 504 | 1,508 | 0 | 10 | 58 | 0 |
| **GT: D10 (Trans.)** | 111 | 344 | 0 | 1 | 8 | 0 |
| **GT: D20 (Allig.)** | 119 | 593 | 0 | 20 | 46 | 0 |
| **GT: D40 (Pothole)** | 175 | 279 | 0 | 0 | 9 | 0 |

**Analysis:**

The base model's Stage 2 performance reveals a fundamental limitation of zero-shot VLM classification on specialized domains:

1. **D00 dominance (72.50% recall):** The model recognizes "crack" patterns and defaults to "Longitudinal Crack" — the most generic crack type. D00 is the only class with meaningful recall.

2. **D10 complete failure (0.00%):** Transverse cracks are visually identical to longitudinal cracks except for orientation (perpendicular vs parallel to road). The base model cannot distinguish orientation without domain training. Every D10 image is classified as either D00 or Unparseable.

3. **D20 near-failure (2.57% recall):** Alligator cracking has a distinctive interconnected pattern, but the base model almost never identifies it. Of 778 alligator crack images, 593 were classified as D00 (longitudinal crack).

4. **D40 near-failure (1.94% recall):** Potholes are the most visually distinctive distress type, yet the base model rarely identifies them. This suggests the model lacks the domain vocabulary to associate bowl-shaped depressions with the specific term "Pothole."

5. **High unparseable rate (34.24%):** 1,973 images could not be parsed into any RDD class. The model often generates responses that don't follow the expected format or names distress types not in the RDD label set.

6. **48.66% overall accuracy is misleading:** This number is inflated by D00 (the majority class with 72.5% recall) and the unparseable category (64.12% recall). For the minority classes (D10, D20, D40), accuracy is near zero.

**Random baseline comparison:** With 4 classes + unparseable, uniform random guessing would yield ~20% accuracy. The 48.66% is above random (primarily due to D00), but the macro F1 of 20.71% reveals that most classes are not being learned.

### 6.2 Fine-Tuned Results (Post-QLoRA Training)

QLoRA fine-tuned `Qwen2.5-VL-7B-Instruct` with rank-32 LoRA adapter, 2-epoch schedule, paged 8-bit AdamW, on the combined GAPs+RDD training set. Training wall-clock 43h 27m on RTX A5000 24 GB. Best surviving checkpoint: `checkpoint-1000` (eval_loss 0.0485). All numbers verified in `eval_results/finetuned_results.json` and `eval_results/ab_comparison.json`.

#### 6.2.1 Stage 1: Binary Detection (Fine-Tuned)

| Metric | Baseline | Fine-Tuned | Delta |
|---|---|---|---|
| Overall Accuracy | 76.53% | **88.39%** | **+11.86 pp** |
| Precision (macro) | 84.36% | 90.87% | +6.51 pp |
| Recall (macro) | 70.92% | 85.85% | +14.93 pp |
| F1-Score (macro) | 71.44% | **87.25%** | **+15.82 pp** |
| Normal Recall | 98.97% | 98.55% | -0.42 pp (held) |
| **Distress Recall** | **42.88%** | **73.15%** | **+30.27 pp** |
| Distress Precision | 96.51% | 97.11% | +0.60 pp (held) |

The Normal-bias documented in §6.1.1 is fixed. Distress recall climbed from 42.88% (missed 57% of damaged roads) to 73.15% (catches ~3 in 4) without sacrificing Distress precision (96.51% → 97.11%, essentially unchanged). The model is no longer afraid to label damage as damage.

#### 6.2.2 Stage 2: In-Distribution Type Classification (Fine-Tuned, RDD Test Set)

| Metric | Baseline | Fine-Tuned | Delta |
|---|---|---|---|
| Primary Accuracy | 48.66% | **66.06%** | **+17.40 pp** |
| F1-Score (macro) | 20.71% | **40.32%** | **+19.61 pp** |
| Exact Match Rate | 34.40% | 52.76% | +18.36 pp |
| Avg inference time | 5.32 s/img | 15.70 s/img | longer (more complete responses, see §6.5) |

**Per-Class Comparison (recall):**

| Class | Baseline Recall | Fine-Tuned Recall | Delta |
|---|---|---|---|
| Unparseable / Normal | 64.12% | 94.73% | +30.61 pp |
| D00 Longitudinal | 72.50% | 71.78% | -0.72 pp (held) |
| D10 Transverse | 0.00% | 1.72% | +1.72 pp (still hard) |
| **D20 Alligator** | **2.57%** | **35.22%** | **+32.65 pp** |
| **D40 Pothole** | **1.94%** | **34.56%** | **+32.61 pp** |

**The headline finding:** D20 (alligator cracking) and D40 (potholes) — the two classes that were essentially unrecognized by the baseline — both gained more than 30 percentage points of recall. The model went from detecting 1 in 39 alligator cracks and 1 in 51 potholes to roughly 1 in 3 of each. This is the single most impactful change in the project and validates the central hypothesis that domain-specific fine-tuning fixes catastrophic class blindness without retraining the vision encoder.

D10 (transverse cracks) remains a failure mode. Hypothesis: discriminating longitudinal vs transverse from visual evidence requires orientation-specific reasoning that the model does not fully acquire from 26,869 RDD training examples. Future work could close this with explicit orientation augmentation or a dedicated D00↔D10 discrimination loss.

#### 6.2.3 Promotion Gate Decision

The fine-tuned adapter cleared all three production-promotion thresholds defined in `scripts/ab_compare_adapters.py`:

| Criterion | Threshold | Actual | Pass |
|---|---|---|---|
| Stage 1 accuracy improvement | ≥ +3.0 pp | +11.86 pp | ✅ |
| Stage 2 macro F1 improvement | ≥ +10.0 pp | +19.61 pp | ✅ |
| Normal-class precision regression | ≥ −5.0 pp | +12.42 pp | ✅ |

The adapter was promoted to `adapters/v2-rdd-2epochs-20260507/` and pointed at via `ADAPTER_PATH` in the production environment.

### 6.3 Cross-Dataset Generalization (Attain Dataset)

> **[TO BE FILLED — Phase 2]**
> After running 07_cross_dataset_eval.py on both baseline and fine-tuned models.

#### 6.3.1 Known Classes (Trained on RDD)

| Class | Baseline Accuracy | Fine-Tuned Accuracy | Delta |
|---|---|---|---|
| Alligator Crack | ___ | ___ | ___ |
| Longitudinal Crack | ___ | ___ | ___ |
| Transverse Crack | ___ | ___ | ___ |
| Pothole | ___ | ___ | ___ |

#### 6.3.2 Unknown Classes (Zero-Shot via Taxonomy Injection)

| Class | Baseline Accuracy | Fine-Tuned Accuracy | Delta |
|---|---|---|---|
| Block Crack | ___ | ___ | ___ |
| Patch/Utility Cut | ___ | ___ | ___ |
| Weathering | ___ | ___ | ___ |
| Raveling | ___ | ___ | ___ |

### 6.4 Expert-in-the-Loop Improvement (Phase 3)

> **[TO BE FILLED — Phase 3]**
> After implementing few-shot prompt injection + LoRA incremental retraining.

| Metric | Before Corrections | After Few-Shot | After LoRA Retrain |
|---|---|---|---|
| Stage 2 Accuracy (RDD) | ___ | ___ | ___ |
| Stage 2 Accuracy (Attain Known) | ___ | ___ | ___ |
| Stage 2 Accuracy (Attain Unknown) | ___ | ___ | ___ |

### 6.5 Stage 2 Confidence Calibration

We ran the production classifier (4-bit, IRC:82 prompts, no adapter) over 407 randomly sampled Attain images and recorded both confidence variants for each Stage 2 prediction. Multi-label correctness is ambiguous, so four definitions are reported. A confidence score's ranking quality is measured by AUC: the probability that a randomly chosen correct prediction scores above a randomly chosen incorrect one (0.5 means the score carries no information).

**Table 6.5a — Stage 2 confidence AUC by correctness definition (Attain, n = 407)**

| Correctness definition | Correct / wrong | Field-restricted | Whole-sequence |
|---|---:|---:|---:|
| No false positives (every predicted label is in ground truth) | 347 / 60 | **0.744 ± 0.030** | 0.373 ± 0.041 |
| Any overlap (at least one predicted label is in ground truth) | 377 / 30 | 0.593 ± 0.051 | 0.593 ± 0.051 |
| Jaccard ≥ 0.5 | 156 / 251 | 0.473 ± 0.029 | 0.453 ± 0.029 |
| Exact set match | 15 / 392 | 0.415 ± 0.071 | 0.662 ± 0.078 |

*± is the Hanley–McNeil standard error. Exact match has only 15 correct predictions and is not interpreted. Source: `eval_results/calib_attain_407.json`.*

Two findings follow. First, the whole-sequence score used before this revision is **inverted** under the no-false-positives rule: at 0.373 it sits more than three standard errors below chance, meaning it gave higher confidence to predictions that contained a spurious label. Second, the field-restricted score is informative under that rule (0.744, eight standard errors above chance) but uninformative under Jaccard ≥ 0.5 (0.473). The restricted score falls as the model names more distress types, so it detects over-prediction and is blind to under-prediction, which the Jaccard rule also counts as an error. The choice of correctness definition, not the choice of metric, determines which conclusion a single AUC would report, which is why all four are given.

**Production photographs.** On the 49 Bengaluru uploads that reached Stage 2, the restricted score separates committed predictions from hedged ones: single-label predictions averaged 0.961 (all 9 above 0.80), two-label predictions 0.698 (3 of 40 above 0.80). The gap is 0.264, against 0.041 for the whole-sequence score (0.809 against 0.768). The whole-sequence score occupied only the range 0.707–0.872 across all 49 predictions; the restricted score spans 0.545–0.990. These photographs have no expert labels, so this is a separation result, not an accuracy result. Source: `eval_results/uploaded_photos_confidence.json`.

**Threshold at matched operating points.** Mean field-restricted confidence differs by dataset — 0.875 on Attain, 0.743 on RDD2022-India, 0.746 on the Bengaluru uploads — so an absolute threshold does not transfer. Table 6.5b compares thresholds at matched review load.

**Table 6.5b — Error cost of each production threshold, read at the matched Attain operating point**

| Production threshold | Production review load | Auto-accept error (95% CI) | Errors caught |
|---:|---:|---:|---:|
| 0.85 | 80% | 3.6% (1–10) | 57 / 60 |
| **0.80** | **76%** | **3.0% (1–8)** | **57 / 60** |
| 0.78 | 69% | 5.6% (3–11) | 53 / 60 |
| 0.75 | 63% | 6.0% (3–11) | 51 / 60 |
| 0.70 | 37% | 7.0% (4–11) | 42 / 60 |

*No-false-positives definition, Attain n = 407; Wilson intervals. Source: `scripts/threshold_decision.py --in calib_attain_407.json --production uploaded_photos_confidence.json`.*

**RDD2022-India** (250 images) was evaluated but is not used for calibration. Its images are dash-camera traffic scenes in which annotated distress occupies a median 7.6% of the frame, and the Stage 1 detector — built for close-up photographs — passed only 3 of 15 images in a pilot. It was run with Stage 1 bypassed; 51 of its 250 predictions named only distress types RDD does not annotate and were excluded rather than scored as errors the dataset cannot adjudicate.

### 6.6 Input Resolution: Accuracy, Agreement and Cost

**Table 6.6a — Agreement with full resolution, Bengaluru uploads (n = 57, no labels)**

| Cap | Visual tokens (mean) | Stage 1 agreement | Distress → Normal | Stage 2 Jaccard vs full | Severity agreement |
|---|---:|---:|---:|---:|---:|
| Full | 3,208 | — | — | — | — |
| 1280² | 1,886 | 98.2% | 1 | 0.779 | 82.6% |
| 1024² | 1,254 | 91.2% | 1 | 0.732 | 78.3% |
| 768² | 719 | 91.2% | 1 | 0.645 | 82.6% |
| 640² | 481 | 96.5% | 0 | 0.703 | 73.9% |
| 512² | 324 | 89.5% | 0 | 0.670 | 67.4% |

*Stage 2 columns cover the 46 images that full resolution classifies as distressed. Source: `eval_results/resolution_ab.json`.*

**Table 6.6b — Accuracy against ground truth, Attain (n = 200, paired with full resolution)**

| Cap | Frames changed | No false positives: full → capped (worse / better, p) | Jaccard ≥ 0.5: full → capped (worse / better, p) |
|---|---:|---|---|
| 1024² | 16 | 88.0% → 87.5% (1 / 0, p = 1.00) | 38.5% → 38.0% (2 / 1, p = 1.00) |
| 768² | 136 | 88.0% → 88.0% (2 / 2, p = 1.00) | 38.5% → 38.5% (9 / 9, p = 1.00) |
| 640² | 200 | 88.0% → 88.0% (3 / 3, p = 1.00) | 38.5% → 40.0% (7 / 10, p = 0.63) |
| 512² | 200 | 87.9% → 86.4% (5 / 2, p = 0.45) | 38.2% → 38.7% (13 / 14, p = 1.00) |

*Exact two-sided McNemar test on discordant pairs. At 512², one frame named only types outside Attain's annotations and is excluded from that row (n = 199).*

**Table 6.6c — Per-class recall against ground truth, Attain**

| Class | n | Full | 1280² | 1024² | 768² | 640² | 512² |
|---|---:|---:|---:|---:|---:|---:|---:|
| Linear crack | 186 | 99.5% | 99.5% | 98.9% | 99.5% | 100.0% | 98.9% |
| Alligator crack | 176 | 11.9% | 11.9% | 11.9% | 9.7% | 13.6% | 11.9% |
| Pothole | 59 | 13.6% | 13.6% | 11.9% | 16.9% | 15.3% | 15.3% |

*Patch, weathering, ravelling and block crack are never named at any resolution, including full, so their 0% recall is not a resolution effect.*

**Table 6.6d — Control arms: same 1024² resolution, JPEG round-trip only**

| Arm (compared with plain 1024²) | Stage 1 agreement | Stage 2 exact match | Stage 2 Jaccard | Attain accuracy, no false positives |
|---|---:|---:|---:|---|
| JPEG quality 95 (visually identical) | 100.0% | 81.6% | 0.881 | 87.5% → 88.0% (p = 1.00) |
| JPEG quality 75 (uploads are stored at ≈80) | 96.5% | 67.3% | 0.793 | 87.5% → 88.0% (p = 1.00) |

**Table 6.6e — Whole-worker time and peak GPU memory, measured in isolation (full-size 1736² photographs, 4-bit, 24 GB GPU)**

| Cap | Visual tokens | Worker time, median | Peak reserved GPU memory | Fits in 24 GB | n |
|---|---:|---:|---:|---|---:|
| Full | 3,844 | 271.8 s | 47.8 GB | No | 2 |
| 1280² | 2,025 | 30.3 s | 26.2 GB | No | 6 |
| **1024²** | **1,296** | **11.1 s** | **19.8 GB** | **Yes** | 6 |
| 768² | 729 | 9.2 s | 16.1 GB | Yes | 6 |

*Pavement pre-filter + detection + classification. Source: `eval_results/resolution_timing.json`.*

**Against the pre-registered criteria, every cap failed.** 1280² failed on Stage 2 Jaccard (0.779); 1024² and 768² on Stage 1 agreement (91.2%) and Jaccard; 640² on Jaccard; 512² on both. No cap failed the safety criterion (at most 1.8% Distress → Normal) or the Attain accuracy criterion. §7.6 explains why we nevertheless recommend 1024².

### 6.7 Gating on the First Label Alone

Most production predictions name two distress types, and the second is often a hedge ("Potholes, Bleeding"). The whole-field confidence falls when the model hedges, so these images go to review even when the first label is a confident pothole. We tested the alternative: show the first label as the main one, gate on its confidence alone, and treat the others as side labels. That only works if (a) the first label is the dominant distress and (b) its confidence predicts whether it is right. To help (a), we added two lines to the Stage 2 prompt asking for the most prominent distress first. Both changes were evaluated on the same 407 Attain images as §6.5, re-run at the production 1024² cap and paired image by image against the stored run.

**Table 6.7a — The ordering rule and the multi-label answer (paired, n = 407)**

| Correctness rule | Without rule | With rule | Worse | Better | McNemar p |
|---|---:|---:|---:|---:|---:|
| No false positives | 85.3% | 83.3% | 9 | 1 | 0.021 |
| Any overlap | 92.6% | 93.1% | 1 | 3 | 0.63 |
| Jaccard ≥ 0.5 | 38.3% | 39.3% | 11 | 15 | 0.56 |
| Exact set | 3.7% | 4.7% | 5 | 9 | 0.42 |

*Exact two-sided McNemar on discordant pairs. The stored run was at full resolution; restricted to the 379 frames the cap leaves unchanged, the no-false-positives comparison is 8 worse and 0 better (p = 0.008), so the difference is the prompt's. Source: `eval_results/primary_confidence.md`, `scripts/primary_confidence_report.py`.*

The rule changed the label set on 12% of images, and mostly not in the intended way. In six of the nine images that got worse, a correct second label (Alligator or Transverse Cracking) was replaced by "Potholes", which the annotators had not marked.

**Table 6.7b — Is the first label the dominant distress?**

| | Without rule | With rule |
|---|---:|---:|
| First label = Longitudinal Cracking | 380 | 390 |
| First label = Alligator Cracking | 17 | 9 |
| First label = Potholes | 10 | 8 |
| First label is the annotated class with the largest box area | 150 (36.9%, CI 32–42) | 148 (36.4%, CI 32–41) |

*McNemar p = 0.69 (4 worse, 2 better). Box area is a proxy for prominence: it overstates thin diagonal cracks and double counts overlapping boxes.*

The model leads with Longitudinal Cracking on 96% of Attain images with or without the rule. That is the first step of the inspection protocol in the prompt, and the rule did not override it.

**Table 6.7c — Does the first label's confidence predict whether the first label is right? (n = 407: 371 right, 36 wrong)**

| Score | AUC ± SE | Mean, right | Mean, wrong |
|---|---:|---:|---:|
| First label, joint probability | 0.468 ± 0.051 | 0.636 | 0.649 |
| First label, geometric mean | 0.476 ± 0.051 | 0.909 | 0.907 |
| Whole field (current gate) | **0.635 ± 0.044** | 0.878 | 0.863 |

*Right = the dataset annotates the first label's class on the image. Hanley–McNeil SE.*

The first label's own confidence carries no information about whether it is right; both variants are within one standard error of chance. The whole-field score, designed for a different question, predicts first-label correctness better than the first label's own score does. On the same fresh run it also reproduces its §6.5 result under the no-false-positives rule (0.726 ± 0.030, against 0.744 in the stored run).

At the live 0.80 threshold, the joint score would send 91% of Attain images to review, and the 35 it would pass are wrong at 17% (95% CI 8–33%), against a base rate of 8.8% if nothing were reviewed. No threshold on it does better than the base rate by more than the width of its interval.

**Production photographs.** Every production row now stores both scores, so the two gates can be compared on the same Bengaluru uploads without re-inference (68 photographs that reached Stage 2; no expert labels, so this compares review load, not accuracy). Potholes lead the label list in 50 of 68, so ordering is less of a problem here than on Attain. The first-label gate still does not lower the review load: 69% against 68% for the whole-field gate, and 90% for both on multi-label photographs. The first label's own confidence is itself low on hedged photographs (mean 0.654 against 0.778 for the whole field): when the model adds a second label, it is also less sure of the first. The gates disagree on 12 photographs, 8 passed only by the whole-field gate and 4 only by the first-label gate. Source: `scripts/production_gate_comparison.py`, `eval_results/production_gate_comparison.json`.

### 6.8 Per-Type Probing Against the Free-Form List

**What the free-form list does.** On the 547 test frames the free-form answer took 8 distinct label sets. "Longitudinal Cracking, Transverse Cracking" alone accounted for 332 (61%), and "N/A" or "Normal" for 100 more. It named Alligator Cracking on 8% of the frames that have it, and Ravelling, Hungry Surface (weathering) or a patch on none. The probe produced 35 distinct label sets on the same frames.

**Table 6.8a — Stage 2 on the Attain test split (n = 547, 23 without distress; five headline classes)**

| Metric | Free-form list (production) | Per-type probe | Constant "Linear + Alligator" | Probe − production [95% CI] |
|---|---:|---:|---:|---:|
| **Macro MCC** | 0.111 | **0.223** | 0.000 | **+0.112 [+0.033, +0.183]**, p = 0.002 |
| Macro F1 | 0.264 | 0.545 | 0.346 | +0.281 [+0.219, +0.332] |
| Macro balanced accuracy | 0.552 | 0.590 | 0.500 | +0.037 [−0.000, +0.075] |
| Mean per-image Jaccard | 0.376 | 0.623 | 0.621 | +0.246 [+0.195, +0.297] |
| Hamming loss (lower is better) | 0.294 | 0.254 | 0.209 | −0.041 [−0.081, +0.003] |
| Exact label set | 9.7% | 25.2% | 27.2% | McNemar 118 vs 33, p < 0.001 |
| No false positive | 85.6% | 43.7% | 60.3% | — |

*Variant `min:name` (name-only questions), chosen on the 300-frame development split. Thresholds MCC-optimal per class on the development split. Cluster bootstrap over 20-frame blocks. Source: `eval_results/stage2_probe_report.json`, `scripts/stage2_probe_report.py`.*

The pre-registered criterion is met: macro MCC doubles, with an interval that excludes zero. Two rows temper this. On per-image Jaccard, exact-set agreement and Hamming loss, the probe is only level with, or behind, a constant predictor that answers "Linear crack and Alligator crack" for every frame. Those metrics are dominated by the two classes that are present most of the time, which is why MCC is the headline. And the probe says "yes" much more often, so its share of answers without a false positive falls from 85.6% to 43.7%. The earlier headline correctness rule (no false positives, §6.5) rewards the free-form list for naming almost nothing.

**Table 6.8b — Per class, test split**

| Class | Prevalence | Production P / R | Production MCC | Probe P / R | Probe MCC [95% CI] | Probe AUROC | ΔMCC [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|
| Linear crack | 83.9% | 0.89 / 0.82 | 0.257 | 0.87 / 0.97 | 0.358 [0.147, 0.529] | 0.672 | +0.101 [−0.044, +0.229] |
| Alligator crack | 69.1% | 0.89 / 0.08 | 0.114 | 0.74 / 0.97 | 0.336 [0.196, 0.456] | 0.703 | **+0.222 [+0.069, +0.371]** |
| Pothole | 18.5% | 0.43 / 0.25 | 0.219 | 0.37 / 0.40 | 0.234 [0.032, 0.412] | 0.650 | +0.016 [−0.067, +0.112] |
| Raveling | 12.8% | 0.00 / 0.00 | 0.000 | 0.09 / 0.14 | −0.063 [−0.164, 0.043] | 0.539 | −0.063 [−0.162, +0.053] |
| Weathering | 26.3% | 0.00 / 0.00 | −0.036 | 0.40 / 0.58 | 0.249 [−0.040, 0.472] | 0.659 | +0.285 [−0.021, +0.508] |
| Patch and utility cut | 19.4% | — (no such label) | 0.000 | 0.40 / 0.32 | 0.224 [0.035, 0.442] | 0.616 | not in headline |
| Block crack (diagnostic) | 3.3% | — | 0.000 | 0.07 / 0.72 | 0.138 [−0.052, 0.285] | 0.774 | not in headline |

In the continuity terms of §6.3, recall of the classes RDD trained on rises from 46.1% to 90.8% (938 annotated class-instances). Recall of the zero-shot classes both methods can name (ravelling, weathering) rises from 0 of 214 to 93 of 214. Recall figures of this kind must be read with the precision column.

**Table 6.8c — Question wording (test split, each variant with its own development-tuned thresholds)**

| Variant | Dev macro AUROC | Test macro AUROC | Test macro MCC | Median time |
|---|---:|---:|---:|---:|
| Name only (**chosen on dev**) | **0.716** | 0.644 | 0.223 | 1.08 s |
| Name + IRC definition and visual cues | 0.683 | 0.643 | 0.227 | 1.64 s |
| Definition + full taxonomy in the system prompt | 0.653 | 0.672 | 0.168 | 2.70 s |

Adding the IRC definition to each question did not help. Putting the whole taxonomy in the system prompt raised test AUROC but lowered test MCC. The development choice and the test ordering disagree, which is a first sign that 300 frames from 15 road blocks are too few to tune per-class choices on.

**Table 6.8d — Grouped 5-fold cross-validation over all 847 frames (43 blocks; secondary analysis)**

| Method | Macro MCC [95% CI] | Macro F1 | Mean Jaccard | Exact set | Δ macro MCC vs production [95% CI] |
|---|---:|---:|---:|---:|---:|
| Free-form list (production) | 0.125 [0.074, 0.170] | 0.249 | 0.410 | 14.1% | — |
| Probe, all five classes decided | 0.248 [0.148, 0.335] | 0.508 | 0.585 | 23.6% | +0.123 [+0.046, +0.194], p = 0.002 |
| Probe, eligible classes only | 0.241 [0.159, 0.302] | 0.443 | 0.594 | 24.8% | +0.116 [+0.064, +0.157], p = 0.001 |
| Constant "Linear + Alligator" | 0.000 | 0.339 | 0.606 | 29.4% | — |

*Thresholds and Platt calibration fitted on four folds, applied to the fifth. "Eligible" (rule fixed before running): the probe decides a class only if its pooled out-of-fold AUROC has a 95% lower bound above 0.5. That holds for Linear crack (0.766 [0.652, 0.854]), Alligator crack (0.758 [0.672, 0.837]) and Raveling (0.624 [0.506, 0.741]), and not for Pothole (0.620 [0.486, 0.739]), Weathering (0.655 [0.432, 0.802]) or Patch (0.579 [0.467, 0.737]), which become verify-only. Source: `eval_results/stage2_probe_cv.json`, `scripts/stage2_probe_cv.py`.*

Cross-validation over every frame confirms the single-split result: macro MCC roughly doubles, with the lower end of the interval well above zero. The per-class picture is less stable than the single split suggested (Raveling moves from no skill on the test split to modest skill pooled), which is why the production configuration uses the pooled eligibility rule rather than per-class choices from one split.

**The probe's own confidence.** Once its per-class Platt calibration was fitted correctly, the probe's image-level confidence ranks its own errors far better than the field confidence ranks the free-form list's. Out of fold, its AUROC for "the label set is exactly right" is 0.778 [0.686, 0.854] against 0.483 for the field score on the same outputs, and 0.874 [0.804, 0.926] against 0.473 for "no false positive". It is also roughly calibrated: frames it scores 0.4–0.6 are exactly right 54% of the time, frames it scores below 0.2 8% of the time (ECE 0.076). For comparison, frames the field score puts at 0.80–0.90 are exactly right 8.5% of the time. It does not beat the field score on the Jaccard ≥ 0.5 event (0.629 against 0.662). A calibrated confidence also cannot exceed the model's real accuracy, so at the live 0.80 threshold it would send almost every Attain frame to review. Under the rule fixed before the analysis, the production gate stays on the field score; the probe confidence is stored on every row.

*A calibration bug was found and fixed during this analysis. Plain Newton iterations for the Platt fit overshot and stuck at the parameter bound for four classes when P(yes) spanned 10⁻⁴ to 1. With a damped, line-searched step the fit matches scikit-learn to three decimals. Thresholds and MCC were unaffected; only the confidence results changed. The fix is recorded in the pre-registration addendum.*

**Table 6.8e — Stage 1 on Attain, and the safety net (test split: 524 frames with distress, 23 without)**

| Detector | Recall of distress | Specificity | MCC | AUROC |
|---|---:|---:|---:|---:|
| Production Stage 1 (Normal/Distress word) | 51.7% | 100.0% | 0.208 | 0.957 |
| Probe, largest headline P(yes) | 93.3% | 87.0% | 0.536 | 0.966 |

Production Stage 1 ranks frames well (AUROC 0.957) but decides conservatively: it calls half of the frames with annotated distress Normal, and Stage 2 never sees them. With Stage 1 in front, end-to-end macro MCC is 0.094 for the free-form list and 0.159 for the probe. Out of fold over all 847 frames, Stage 1 called 449 frames Normal, 371 of them with annotated distress. The safety net would send 313 of those 371 to review, at the cost of 22 of the 78 frames without distress. Only 78 frames have no distress, so the specificity figures are uncertain.

**Table 6.8f — The same configuration on 229 Bengaluru uploads since the production switch (no labels)**

| | |
|---|---|
| Uploads that reached Stage 2 | 204 |
| Label sets the probe rule would leave unchanged | **3 of 204** |
| Types the probe would add | Ravelling 197, Transverse 122, Alligator 110, Longitudinal 84 |
| Types the probe would remove | Bleeding 45 of 52, Potholes 37 of 167, Edge Breaking 25, Hungry Surface 13 |
| Review load at 0.80: field gate / probe gate | 59.8% / 99.0% |
| Stage-1-Normal uploads / already sent to review / newly flagged by the safety net | 25 / 19 / 1 of the 6 auto-accepted |
| Probe time, median | 1.43 s |

*Source: `eval_results/probe_production_compare.json`, `scripts/probe_production_compare.py` (read-only).*

Nothing measures accuracy here. What the table shows is that thresholds tuned on Attain road strips do not transfer to Bengaluru close-ups. The Ravelling threshold (P(yes) ≥ 0.089) lies below the 10th percentile of Bengaluru uploads, so a label that would be added to 98% of photographs carries no information. The probe's removals are harder to dismiss: a manual look at four random "Bleeding" uploads found no visible bitumen film in any of them (tree shadow, loose aggregate, repair patches, a wet night-time surface). That is an inspection, not ground truth. The configuration therefore went to production in **shadow mode** with the safety net on (§3.4.1).

---

## 7. Discussion

### 7.1 Baseline Performance Interpretation

The baseline results establish that a general-purpose VLM (Qwen2.5-VL-7B) has some capability for pavement assessment but falls far short of practical deployment:

- **Stage 1** demonstrates that the model understands the concept of pavement damage but has a conservative (Normal-biased) decision boundary. The 96.51% Distress precision indicates that the visual features of damage are distinguishable — the model simply has a high threshold for declaring damage.

- **Stage 2** reveals that fine-grained distress type classification is beyond the base model's capability. The model can detect "crack" as a general concept but cannot distinguish between crack types (longitudinal vs transverse vs alligator) or identify potholes. This is expected: pavement distress classification is a specialized domain that requires training data to establish the visual-linguistic mapping between specific damage patterns and their technical names.

### 7.2 Why Zero-Shot Stage 2 Partially Works (48.66%)

The 48.66% accuracy (above 20% random) is attributable to:
1. **Taxonomy injection:** The comprehensive distress type list in the system prompt gives the model a vocabulary to work with
2. **Visual-semantic pre-training:** The VLM's pre-trained alignment between visual features and text enables partial recognition of obvious patterns (cracks)
3. **D00 bias:** "Longitudinal crack" is the most generic and visually common crack type, serving as a default when the model recognizes crack-like patterns

### 7.3 Empirical Impact of Fine-Tuning

Three of the four predictions in the original Phase 1 plan held; one did not. Verified deltas:

1. **D10 (Transverse) recall** — predicted gain from 0% to functional. Actual: 0% → 1.72%. **Did not generalize.** Hypothesis: discriminating longitudinal from transverse needs orientation-aware visual grounding the model does not learn from these training pairs.
2. **D20 (Alligator) recall** — predicted gain from 2.57%. Actual: **2.57% → 35.22% (+32.65 pp)**. The interconnected mesh pattern is now recognizable.
3. **D40 (Pothole) recall** — predicted gain from 1.94%. Actual: **1.94% → 34.56% (+32.61 pp)**. Depth/shadow cues are now associated with the term "pothole".
4. **Unparseable rate** — predicted to drop from 34.24%. Actual: format compliance is now near-universal; the "Unparseable / Normal" class has 94.73% recall (vs 64.12% baseline), confirming the model now uses the exact `DISTRESS_TYPES: ... SEVERITY: ... DESCRIPTION: ...` schema almost always.

Stage 1 also dramatically improved (Distress recall +30.27 pp) — fine-tuning shifted the decision boundary closer to ground truth and away from the Normal-bias.

### 7.4 What Fine-Tuning Did NOT Solve

D10 transverse-crack recall remains at ~2%. This is the open problem after Phase 2. Two hypotheses:
- **Insufficient orientation discrimination signal in the loss** — the model can copy the structured response format but the visual feature distinguishing horizontal from vertical cracks isn't leveraged.
- **Class imbalance in training data** — D10 is the smallest of the four RDD classes. Resampling, focal loss, or pair-based contrastive examples could help.

Future work should target this specifically. The current adapter is otherwise production-ready.

### 7.5 The Engineering Cost — Not Trivial

Getting QLoRA on Qwen2.5-VL-7B to actually train end-to-end on a 24 GB consumer GPU required **6 commits of empirically-validated YAML hardening** — each fixing a specific failure mode reproduced in real OOM logs. Documented for reproducibility:

1. `enable_liger_kernel: false` — Liger is incompatible with QLoRA, silent slowdown
2. `image_max_pixels: 100352` — phone photos at native resolution OOM the activation memory
3. `cutoff_len: 1024` — 2048 was overkill given response lengths and pushed activations into thrashing
4. `label_smoothing_factor: 0.0` — the label smoother materializes a 5 GB log_softmax tensor that triggered the original OOM
5. `optim: paged_adamw_8bit` — pages optimizer state to CPU; lazy allocation on first step otherwise crashes at step 9
6. `lora_rank: 32` (not 64) and `per_device_train_batch_size: 1, gradient_accumulation_steps: 32` — final memory budget that fits comfortably
7. `torch_empty_cache_steps: 4` — without this, allocator fragmentation balloons step times from 22s to 400s by step 25

Most of these are not in the LLaMA-Factory documentation. We document them in `paper/PAPER_INGREDIENTS.md` §5 and the project's `CLAUDE.md` for future work.

### 7.6 Why 1024² Is Recommended Despite Failing the Pre-Registered Criteria

The Stage 2 agreement criterion assumed that the full-resolution answer is a stable reference. The control arms show it is not. A JPEG round-trip at quality 95, which removes no visible information, changes the distress-type set on 18.4% of images (Jaccard 0.881). Uploads are already stored as JPEG at a quality of about 80; one step further, at quality 75, the Jaccard falls to 0.793, below the 0.80 criterion. A criterion that is failed by compression close to what every upload already carries cannot distinguish resolution loss from the model's own instability; in hindsight it was mis-specified.

The evidence that bears on *loss* points the other way. On Attain, paired accuracy is unchanged at every cap: the largest drop is 1.5 percentage points at 512² (p = 0.45), and the changes that do occur are symmetric — at 768², nine predictions became worse and nine became better under the Jaccard rule. Recall of the finest-grained class, linear cracking, stays between 98.9% and 100% at every cap. The answers change, but not in a direction that costs accuracy; the pattern is a re-sampling of an unstable second label rather than a systematic loss of detail.

The Stage 1 changes are confined to uncertain images. Across all caps there were 19 Stage 1 flips on 7 images, and every one of those images had a full-resolution Stage 1 confidence below the 0.80 review threshold (the highest was 0.787). All three Distress → Normal flips were the same image, at full-resolution confidence 0.530. Every flipped image is routed to an expert at full resolution already, so capping changes no automated decision.

We therefore recommend a cap of 1024² — the largest that fits in GPU memory (Table 6.6e) — and state plainly that this departs from the rule we set in advance. The Attain evidence at 1024² itself rests on the 16 frames that cap changes; the stronger evidence is that accuracy holds at 768² and below, where most or all frames are downscaled by comparable or greater factors.

### 7.7 GPU Memory, Not Compute, Sets Production Throughput

At full resolution a single worker pass reserves a peak of 47.8 GB of GPU memory on a 24 GB card. The classification stage accounts for it: On Windows the display driver (WDDM) does not raise an out-of-memory error in this case: it pages the overflow to system memory over PCIe, and the pass continues an order of magnitude more slowly. on the same full-size image the pavement pre-filter and detection stages, which encode the same image but generate only a few tokens, take 3.9 s and 5.8 s, while classification takes about 260 s. At the 1024² cap the whole worker runs in 11.1 s against 271.8 s, a 24× reduction, with peak memory at 19.8 GB.

This also affects measurement. The PyTorch caching allocator keeps a pass's peak memory reserved after the pass ends, so any timing taken after a full-resolution image without clearing the cache inherits the overflowed state. In the first resolution run, the 1024² cap measured 109 s when timed immediately after a full-resolution pass and 10 s once the cache was cleared between passes. Throughput figures elsewhere in this project that were measured without this precaution should be re-measured.

Image *shape* has the same effect. Attain mixes three frame shapes. Without releasing the cache between images, Stage 2 took 8 s on 1479×508 frames, 24 s on 640×640 and 84 s on 1920×1080, with identical outputs. Releasing the cache after every image brought all three to about 8 s. The smaller square frames were slower than the larger wide ones, which rules out compute as the cause: each new shape grew the allocator's reserve until it spilled into system memory. Production uploads all arrive at 1200×1600 and have not been affected. The worker now releases the cache after every image as a guard.

### 7.8 Limitations of the Calibration and Resolution Results

- **No expert labels on production photographs.** On the Bengaluru uploads, the confidence and resolution results measure separation and change, not accuracy. Labelling a sample of those photographs is the most valuable next step.
- **Domain gap.** Attain is vehicle-mounted footage with wide frames (created in Tehran; §4.3); production is handheld close-ups from Bengaluru. Mean confidence differs by 0.13 between them, which is why thresholds were compared at matched review load.
- **Calibration population.** The 407 Attain images were selected by a Stage 1 run made before the IRC:82 prompt revision; 274 of the 407 stored predictions (67%) reproduced under the current prompts. Rankings are unaffected, but absolute rates carry this caveat.
- **Unstable second label.** An invisible perturbation changes 18% of Stage 2 answers. This caps how far any agreement metric can go. Asking for the most prominent type first did not stabilise it (§6.7); it moved the second label and left the first where it was.
- **Configuration.** All results are for the 7B model at 4-bit on a single 24 GB GPU under Windows.

### 7.9 Why the First Label's Confidence Does Not Work as a Gate

On Attain, the first label is almost always Longitudinal Cracking, because that is step 1 of the inspection protocol written into the prompt. The model's confidence in that token reflects how the prompt orders the inspection, not what the image shows. It comes out the same (0.64 on average) whether or not the frame contains a linear crack. The wrong first labels are frames where the annotators marked no linear crack at all, and the model is no less sure on those.

The whole-field score works better for a reason that is easy to miss: hedging is evidence. When the model is unsure what an image shows, it names more types and names them less confidently, and those are the images where its first label is also more often wrong. A score that ignores the second label ignores that evidence. For the same reason, shielding the first label from the hedge ("a confident pothole with an unsure extra label should pass") has no support on labelled data. The cases it would wave through are the cases the current gate catches.

Two limits apply. First, 96% of Attain first labels are one class, so the accuracy test is weak for Bengaluru, where potholes lead. The production comparison shows the first-label gate would not lower review load there either, but only labelled Bengaluru photographs can settle whether its score is informative. Every production row now records it, so that analysis will need no re-inference. Second, a prompt instruction is a weak lever on output order. Ordering by a separate step, such as asking which listed type dominates and reading the softmax over the listed types (as Stage 1 does for Normal/Distress), would make "main type" a decision the model is asked to make rather than a side effect of listing. We leave that to future work.

Production therefore keeps the whole-field gate at 0.80 and the unmodified prompt. The interfaces show the first label as the main distress and the others beside it. The rule-carrying prompt remains available (`PROMPTS_VERSION=v2_primary_first`) so these results can be reproduced.

### 7.10 Asking Instead of Listing, and Why It Is Not Yet Live

**Why the free-form list collapses.** A generated list is a sequence. The model writes the first label, then the second conditioned on the first, and it stops when stopping is likely. The v2 prompt walks through an inspection protocol whose first step is longitudinal and transverse cracking, and the list follows that protocol: 61% of test frames received exactly its first two items. Alligator cracking, present on 69% of frames, was named on 8% of them. The model does see it: asked directly, it rates alligator cracking higher on frames that have it (AUROC 0.70 test, 0.76 cross-validated). The information is in the model; the list is a poor way to get it out. Per-type probing reads the same model's belief about each type separately, and each belief can then be thresholded on its own.

**Why the evaluation had to change first.** Three earlier practices would have hidden or inflated this result. The headline correctness rule (no false positives) scores a model that names one label as right more often than one that names four, which rewards the collapsed list. Per-class F1 on classes present in 70–84% of frames is dominated by prevalence: production's 0.95 F1 for linear cracks (§6.5 data) is the score of a constant predictor. And a scoring bug silently removed 167 patch instances from the zero-shot tally. MCC against a constant baseline, block-level splits and bootstraps, and a recorded protocol are what make the +0.112 believable. The same changes also show its limits. The probe does not beat the constant predictor on per-image agreement, and on Attain it is still far from reliable (macro MCC 0.24).

**Why production runs it in shadow.** A threshold is a statement about a particular photographic domain. The probe's thresholds were learned on vehicle-mounted road strips, and on handheld close-ups they fire on almost everything. Deploying them would replace one uninformative default ("Longitudinal, Transverse") with another ("Ravelling"). What does transfer is the machinery: fast per-type probabilities, a calibrated way to set thresholds, a rule that separates types with labelled evidence from types without, and a confidence that measurably predicts error. Shadow mode puts that machinery on every production row now. The step that remains is local labels. Expert review in the existing UI is exactly the data the calibration script needs, which makes Phase 3 (expert-in-the-loop) concrete: each reviewed upload moves the probe towards being allowed to decide on Bengaluru roads.

**Where the probe helps now.** The Stage 1 safety net needs only that the probe's scores rank distress above clean pavement, which holds on Attain (AUROC 0.966) and plausibly on close-ups. It never changes a label; it only sends a confident "Normal" to a person when the probe disagrees. That matches the project's standing rule that the system must not silently classify a damaged road as normal.

**Limitations.** (1) All labelled evidence is from one dataset (Attain WS_V2.0), whose frames and annotation practice differ from production. (2) The 78 no-distress frames are the only all-negative examples, so specificity estimates are wide. (3) The development split (15 blocks) was too small to tune per-class choices; the cross-validated configuration is the one deployed. (4) Twelve of the eighteen IRC types have no labelled data at all, so the probe may only remove them; its removals (e.g. Bleeding on 45 of 52 uploads) are unvalidated. (5) Severity is unchanged and not evaluated here. Attain labels only High or Low and the image-level maximum is High on most frames, while the model mostly answers Medium, so exact-match severity accuracy on Attain (§6.3) measures the vocabulary mismatch more than the model. (6) The description sentence still comes from the free-form generation and can mention types the probe would remove.

---

# PART II: TECHNICAL NOTES

---

## T1. Data Pipeline Details

### T1.1 GAPs V2 Conversion (Script 01)

- Raw format: `.npy` chunks with shape `(N, 1, 160, 160)` float32 + int32 labels
- Conversion: Memory-mapped loading (`np.load(mmap_mode='r')`) to avoid RAM exhaustion
- Parallelism: Auto-scales to available CPU cores (uses `multiprocessing.Pool`)
- Chunk size: Calculated from 60% of available RAM / per-image byte size
- Output: RGB PNG files in `data/images/{split}/Normal/` and `data/images/{split}/Distress/`

### T1.2 RDD2022 Conversion (Script 02)

- YOLO `.txt` labels parsed per image: each line = `class_id x_center y_center width height`
- Duplicate class IDs collapsed to unique set (image-level multi-label)
- Training responses generated by `build_stage2_training_response()` with heuristic severity
- Combined training JSON: GAPs + RDD interleaved, registered as `combined_train` in LLaMA-Factory dataset_info

### T1.3 Dataset Sizes (Final)

| Dataset File | Entries | Purpose |
|---|---|---|
| gaps_train.json | 50,000 | Stage 1 training |
| gaps_valid.json | 10,000 | Stage 1 validation |
| gaps_test.json | 10,000 | Stage 1 evaluation |
| rdd_train.json | 26,869 | Stage 2 training |
| rdd_valid.json | 5,758 | Stage 2 validation |
| rdd_test.json | 5,758 | Stage 2 evaluation |
| **Total training** | **76,869** | Combined fine-tuning |

## T2. Inference Pipeline Technical Details

### T2.1 Model Loading Sequence

1. Check for GPU availability and compute capability
2. Configure BitsAndBytesConfig (4-bit NF4 for deployment, none for evaluation)
3. Set `PYTORCH_CUDA_ALLOC_CONF` for CUDA memory allocator tuning
4. Auto-detect attention implementation: Flash Attention 2 > SDPA > Eager
5. Load base model with `device_map="auto"` (automatic device placement)
6. If adapter exists: load as `PeftModel` (never merge)
7. Load processor with pixel limits: min=7,168, max=4,840,000
8. Cache "Normal" and "Distress" token IDs for Stage 1 confidence extraction
9. Set model to eval mode

### T2.2 Processor Pixel Limits

| Parameter | Value | Rationale |
|---|---|---|
| MIN_PIXELS | 7,168 (256 * 28) | Low enough for 160x160 GAPs images (25,600 pixels) |
| MAX_PIXELS | 4,840,000 (2200 * 2200) | Prevents memory explosion on very large images |

These bounds control how many visual tokens the Qwen2.5-VL processor generates. More pixels = more tokens = more VRAM and compute.

**Measured recommendation (§6.6, §7.6):** `MAX_PIXELS = 1,048,576` (1024²). At the current 4,840,000 cap a 1736² photograph is not downscaled and a worker pass reserves a peak of 47.8 GB of GPU memory; at 1024² it needs 19.8 GB and the worker runs 24× faster with no measured accuracy loss on labelled data. The production value remains 4,840,000 until this recommendation is applied.

### T2.3 Inference Performance

| Stage | Image Size | Avg Time (7B fp16) | Avg Time (3B 4-bit) |
|---|---|---|---|
| Stage 1 | 160x160 (GAPs) | 2.08 s | ~1.5 s |
| Stage 2 | 512x512 (RDD) | 5.32 s | ~4.7 s |
| Full Pipeline | Varies | ~7.4 s (if distressed) | ~6.2 s |

The figures above were measured on small dataset images. Production smartphone photographs behave very differently (§6.6, Table 6.6e; 7B, 4-bit, IRC:82 prompts, whole worker including the pavement pre-filter):

| Input | Visual tokens | Worker time (median) |
|---|---:|---:|
| 1736² photograph, full resolution | 3,844 | 271.8 s |
| Same photograph, 1024² cap | 1,296 | 11.1 s |
| Same photograph, 768² cap | 729 | 9.2 s |

### T2.4 Thread Safety

All model inference is wrapped in `threading.Lock()` to prevent concurrent access to CUDA tensors. The FastAPI server uses `asyncio.to_thread()` to run inference in a thread pool, keeping the event loop responsive for health checks and SSE streaming.

### T2.5 VRAM Usage

| Model | Config | VRAM Allocated | VRAM Reserved |
|---|---|---|---|
| Qwen2.5-VL-7B | fp16, no quantization | ~14 GB | ~16 GB |
| Qwen2.5-VL-7B | 4-bit NF4 | ~5.5 GB | ~7.3 GB |
| Qwen2.5-VL-3B | 4-bit NF4 | ~2 GB | ~3 GB |

The table above is weights at rest. Peak memory during Stage 2 inference on a 1736² photograph (7B, 4-bit) is dominated by activations and grows steeply with input resolution: 47.8 GB reserved at full resolution, 26.2 GB at a 1280² cap, 19.8 GB at 1024² and 16.1 GB at 768² (`scripts/resolution_timing.py`). Anything above physical memory is paged to system RAM by the Windows driver rather than failing, which is why the full-resolution pass is roughly 25× slower rather than crashing.

## T3. API and Deployment Architecture

### T3.1 Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/classify` | POST | Standard classification (multipart/form-data) |
| `/classify/stream` | POST | SSE streaming with per-stage progress events |
| `/classify/base64` | POST | Base64-encoded image input |
| `/health` | GET | Model status, device info, adapter status |
| `/retrain/start` | POST | Trigger incremental retraining |
| `/retrain/status` | GET | Training progress |
| `/dashboard/delete/{id}?confirm=true` | DELETE | Remove an upload everywhere: Cloudinary image first (signed destroy with CDN invalidation), then the app's `photos` row, which cascades to the assessment. Deletes nothing if the image cannot be confirmed gone. |

### T3.2 SSE Streaming Protocol

```
Client: POST /classify/stream (multipart/form-data with image)

Server events:
  event: progress  data: {"stage":"stage1","status":"running","message":"Detecting..."}
  event: progress  data: {"stage":"stage1","status":"complete","result":{...stage1 fields...}}
  event: progress  data: {"stage":"stage2","status":"running","message":"Classifying..."}
  event: progress  data: {"stage":"stage2","status":"complete","result":{...stage2 fields...}}
  event: result    data: {full ClassificationResponse JSON}
```

**Implementation note:** SSE uses `fetch()` + `ReadableStream` on the client (not `EventSource`) because `EventSource` only supports GET requests, and image upload requires POST with multipart/form-data.

### T3.3 Remote Access

Cloudflare Tunnel (`cloudflared tunnel --url http://localhost:8000`) provides zero-configuration HTTPS tunneling for remote testing. No DNS setup, no port forwarding, no firewall changes required. Each tunnel gets a random `*.trycloudflare.com` subdomain.

## T4. Evaluation Robustness

### T4.1 Checkpoint-Based Resumption (Stage 2)

Stage 2 evaluation runs for ~8.5 hours (5,758 images * 5.32 s/image). To survive system crashes/reboots:

- **Checkpoint interval:** Every 50 images (~4.4 minutes of work)
- **Checkpoint contents:** `y_true`, `y_pred`, counters, resume index, total sample count
- **Atomic writes:** Checkpoint written to `.tmp` file, then renamed (prevents corruption if crash occurs mid-write)
- **Resume logic:** On startup, checks for valid checkpoint with matching sample count. If found, resumes from saved index. If sample count changed (different `--max-samples`), starts fresh.
- **Cleanup:** Checkpoint file deleted only after successful full completion.

### T4.2 Graceful Interrupt Handling

- First `Ctrl+C`: Sets `_interrupted` flag, current image finishes, partial results saved
- Second `Ctrl+C`: Force quit (data from current session lost, but checkpoint preserves prior progress)
- System reboot: Process killed without signal handling, but checkpoint file preserves progress

### T4.3 Determinism

All evaluations use greedy decoding (`do_sample=False, temperature=None, top_p=None`). Given identical hardware, model weights, and input data, results are exactly reproducible. The only source of non-determinism is CUDA floating-point ordering on different GPU architectures, which produces negligible differences (<0.01% accuracy variation).

## T5. Known Bugs and Workarounds

| Bug | Impact | Workaround |
|---|---|---|
| `transformers >= 4.52` `Params4bit` bug | Model loading crashes | Pin `transformers==4.57.6` |
| `peft` `merge_and_unload()` on quantized models | Produces corrupted weights | Keep as PeftModel, never merge |
| 7B model OOM on 6GB VRAM | Cannot run on laptop | Use 3B for dev, 7B for eval on A5000 |
| SSE `\r\n` line endings on Windows | JS parser breaks | Normalize `\r\n` to `\n` in client |
| `capture="environment"` on mobile | Blocks gallery selection | Use `accept="image/*"` alone |
| JS `history` variable collision | Conflicts with `window.history` | Renamed to `sessionHistory` |
| CPU offload with bitsandbytes | Not supported | Use smaller model or larger GPU |

## T6. Dependency Pinning (Critical)

```
transformers==4.57.6
accelerate==0.34.2
bitsandbytes==0.44.1
peft==0.14.0
```

These four packages must be installed at these exact versions. Any upgrade to `transformers` past 4.52 will break `bitsandbytes` integration. This combination has been tested on:
- Windows 11 + RTX 4050 (6GB) + CUDA 12.1
- Windows 11 + RTX A5000 (24GB) + CUDA 12.1

---

*Last updated: 2026-05-07*
*Status: Phase 1 baseline complete. **Phase 2 fine-tuning + post-finetune eval + A/B promotion gate complete (PASSED).** Adapter promoted to `adapters/v2-rdd-2epochs-20260507/`. Cross-dataset eval on Attain pending. Phase 3 (expert-in-the-loop) not yet started.*
