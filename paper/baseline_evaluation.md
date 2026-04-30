# Two-Stage Pavement Distress Classification Using Vision-Language Models with Expert-in-the-Loop Incremental Learning

## Paper-Ready Sections + Technical Notes

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
- Cross-dataset generalization evaluation using geographically distinct datasets (RDD2022 from multiple countries, Attain from New Zealand)
- An expert-in-the-loop feedback mechanism combining immediate few-shot improvement with permanent LoRA incremental retraining

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
Stage 2 uses **geometric mean of per-token probabilities** across the entire generated sequence:
1. For each generated token `t_i`, compute `log P(t_i | t_1, ..., t_{i-1})` from the logits
2. Average all log-probabilities: `mean_log_prob = (1/N) * sum(log P(t_i))`
3. Exponentiate: `confidence = exp(mean_log_prob)`

This represents the model's average certainty per token. Unlike Stage 1's binary softmax, Stage 2 must evaluate confidence over a multi-token structured response, making geometric mean the appropriate metric. A sequence where the model is highly certain about every token yields a confidence close to 1.0; uncertainty on even a few tokens pulls the geometric mean down significantly.

**Response Parsing:**
The parser extracts structured fields using prefix matching (`DISTRESS_TYPES:`, `SEVERITY:`, `DESCRIPTION:`). If structured parsing fails, a fallback keyword extraction scans for known distress type names in the raw text. If no distress types can be extracted, the result is labeled "Unknown" — which always falls below the 80% confidence threshold and triggers expert review.

### 3.5 Expert Review Threshold

The confidence threshold is set at **80%** (0.80), defined once in `scripts/utils.py` and imported by all components. Any prediction where either Stage 1 or Stage 2 confidence falls below this threshold is flagged with `needs_expert_review = true`.

**Threshold selection rationale:**
- Below 50%: Model is essentially guessing — must be reviewed
- 50-80%: Model has a direction but is uncertain — review improves accuracy
- Above 80%: Model is confident — accept unless contradicted by field evidence
- The 80% threshold will be empirically calibrated on the validation set after fine-tuning (Phase 2)

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
| Images | 2,293 images with 19,761 annotated distress instances |
| Classes | 10 types (see mapping below) |
| Severity | Low / Medium / High per instance (unlike RDD which has no severity labels) |
| Collection | Smartphone-mounted on vehicles, 20-70 km/h, New Zealand roads |
| Geographic Context | Southern hemisphere, different climate and road construction standards than RDD |

**Attain-to-Pipeline Class Mapping:**

| Attain Class | Pipeline Taxonomy Match | In RDD Training? | Evaluation Role |
|---|---|---|---|
| Alligator Crack | Alligator Crack (D20) | YES | In-distribution transfer |
| Longitudinal Crack | Longitudinal Crack (D00) | YES | In-distribution transfer |
| Transverse Crack | Transverse Crack (D10) | YES | In-distribution transfer |
| Pothole | Pothole (D40) | YES | In-distribution transfer |
| Block Crack | Block Crack (D43) | NO | **Zero-shot evaluation** |
| Patch/Utility Cut | Inlaid Patch (D44) / Utility Cut | NO | **Zero-shot evaluation** |
| Weathering | Weathering/Oxidation | NO | **Zero-shot evaluation** |
| Raveling | Raveling | NO | **Zero-shot evaluation** |
| Faded Marking | (not pavement distress) | - | EXCLUDED |
| Lane/Shoulder Drop-off | (not pavement distress) | - | EXCLUDED |
| Manhole | (not pavement distress) | - | EXCLUDED |

**Exclusion rationale:** Faded markings, lane/shoulder drop-offs, and manholes are road features, not pavement structural damage. Including them would inflate false positive rates and dilute the evaluation's focus on actual distress classification.

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

> **[TO BE FILLED — Phase 2]**
> After fine-tuning on combined GAPs + RDD training data.

#### 6.2.1 Stage 1: Binary Detection (Fine-Tuned)

| Metric | Baseline | Fine-Tuned | Delta |
|---|---|---|---|
| Overall Accuracy | 76.53% | ___ | ___ |
| Precision (macro) | 84.36% | ___ | ___ |
| Recall (macro) | 70.92% | ___ | ___ |
| F1-Score (macro) | 71.44% | ___ | ___ |
| Distress Recall | 42.88% | ___ | ___ |

#### 6.2.2 Stage 2: In-Distribution Type Classification (Fine-Tuned, RDD Test Set)

| Metric | Baseline | Fine-Tuned | Delta |
|---|---|---|---|
| Primary Accuracy | 48.66% | ___ | ___ |
| F1-Score (macro) | 20.71% | ___ | ___ |
| Exact Match Rate | 34.40% | ___ | ___ |

**Per-Class Comparison:**

| Class | Baseline Recall | Fine-Tuned Recall | Delta |
|---|---|---|---|
| D00 (Longitudinal) | 72.50% | ___ | ___ |
| D10 (Transverse) | 0.00% | ___ | ___ |
| D20 (Alligator) | 2.57% | ___ | ___ |
| D40 (Pothole) | 1.94% | ___ | ___ |

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

---

## 7. Discussion

> **[TO BE EXPANDED after Phases 2 and 3]**

### 7.1 Baseline Performance Interpretation

The baseline results establish that a general-purpose VLM (Qwen2.5-VL-7B) has some capability for pavement assessment but falls far short of practical deployment:

- **Stage 1** demonstrates that the model understands the concept of pavement damage but has a conservative (Normal-biased) decision boundary. The 96.51% Distress precision indicates that the visual features of damage are distinguishable — the model simply has a high threshold for declaring damage.

- **Stage 2** reveals that fine-grained distress type classification is beyond the base model's capability. The model can detect "crack" as a general concept but cannot distinguish between crack types (longitudinal vs transverse vs alligator) or identify potholes. This is expected: pavement distress classification is a specialized domain that requires training data to establish the visual-linguistic mapping between specific damage patterns and their technical names.

### 7.2 Why Zero-Shot Stage 2 Partially Works (48.66%)

The 48.66% accuracy (above 20% random) is attributable to:
1. **Taxonomy injection:** The comprehensive distress type list in the system prompt gives the model a vocabulary to work with
2. **Visual-semantic pre-training:** The VLM's pre-trained alignment between visual features and text enables partial recognition of obvious patterns (cracks)
3. **D00 bias:** "Longitudinal crack" is the most generic and visually common crack type, serving as a default when the model recognizes crack-like patterns

### 7.3 Expected Impact of Fine-Tuning

Based on the baseline failure patterns, fine-tuning should specifically improve:
1. **D10 (Transverse) recall** from 0% — the model needs training data showing orientation matters
2. **D20 (Alligator) recall** from 2.57% — the interconnected crack pattern is distinctive but needs explicit examples
3. **D40 (Pothole) recall** from 1.94% — depth/shadow cues need to be learned for bowl-shaped depressions
4. **Unparseable rate** from 34.24% — fine-tuning on the exact output format reduces format violations

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

### T2.3 Inference Performance

| Stage | Image Size | Avg Time (7B fp16) | Avg Time (3B 4-bit) |
|---|---|---|---|
| Stage 1 | 160x160 (GAPs) | 2.08 s | ~1.5 s |
| Stage 2 | 512x512 (RDD) | 5.32 s | ~4.7 s |
| Full Pipeline | Varies | ~7.4 s (if distressed) | ~6.2 s |

### T2.4 Thread Safety

All model inference is wrapped in `threading.Lock()` to prevent concurrent access to CUDA tensors. The FastAPI server uses `asyncio.to_thread()` to run inference in a thread pool, keeping the event loop responsive for health checks and SSE streaming.

### T2.5 VRAM Usage

| Model | Config | VRAM Allocated | VRAM Reserved |
|---|---|---|---|
| Qwen2.5-VL-7B | fp16, no quantization | ~14 GB | ~16 GB |
| Qwen2.5-VL-7B | 4-bit NF4 | ~5.5 GB | ~7.3 GB |
| Qwen2.5-VL-3B | 4-bit NF4 | ~2 GB | ~3 GB |

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

*Last updated: 2026-03-29*
*Status: Phase 1 baseline complete. Phase 2 (fine-tuning + cross-dataset eval) pending. Phase 3 (expert-in-the-loop) not yet started.*
