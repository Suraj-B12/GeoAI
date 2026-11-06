# Weight-space programming - editor's work notes

Written by the editor (Claude) while working; one entry per step.

## 2026-11-06T06:01:33+00:00 - Step 0 - set-up and the target

Target: the Stage 1 decision (production prompt, first generated token, P(Distress) = softmax over the
'Normal' and 'Distress' first sub-tokens). Section 6.8 found that Stage 1 says Normal on about half of the
Attain frames that show distress (recall 51.8%) while its own ranking is good (test AUROC 0.957) and the
per-type probe sees distress on 93% of them. So the information is in the model and the answer does not use
it: the kind of "hidden ability behind an obstacle" the weight-space-programming post reports (their
obstacle was a repetition loop).

Harness: a forward-pass server (scripts/wsp_server.py) is my only way to run the model; every pass is in
eval_results/finetune/wsp_ledger.jsonl and counts against 50. Edits are rank-one output edits
W' = W - c u u^T W or column edits on decoder projections, applied as hooks that the server's self-test
showed equal to the dense edited (dequantized NF4) weight to within NF4 kernel rounding (relative error
0.0019 vs 0.0014 for the unedited module). The test split is refused by the server.

Sanity check (no pass spent): the server's teacher-forced read reproduces the stored production Stage 1
answers on the 256 dev frames of pass 1 - 256/256 identical labels, confidence within 0.011 (max), and the
first token is always 'Normal' or 'Distress'.

## 2026-11-06T10:23:12+00:00 - Step 1 - diagnosis (passes 1-3, base model, 726 practice+dev frames, captures at all 28 layers)

Base behaviour. Dev: recall 51.8% at specificity 100% (55/55), AUROC 0.913. The mean decision score
s = logit(Distress) - logit(Normal) is -0.01 on distress frames and -3.11 on clean frames: the distress frames sit
on the decision boundary. A constant shift would recover much of the recall; the best dev offset is +1.56
(recalibrated base: dev recall 76.7%, specificity 98.2%, MCC 0.600 vs 0.406). Any edit has to be compared with this.

Direct logit attribution (s decomposed exactly into every attention and MLP output at the decision position,
through the final RMSNorm; components sum to the captured score within 0.024). The decision is written by late
attention: layer 26 (mean -2.76) and 23 (-1.35) push toward Normal but less so on distress frames (they carry
evidence, dev AUROC 0.88/0.92); layers 25 (+2.19) and 27 (+1.61) push toward Distress almost constantly;
MLP 26 (+1.02) carries evidence (0.91). Three MLPs write a near-constant push toward Normal with no class
signal: layer 25 (-0.57, sd 0.10), 13 (-0.21, sd 0.02), 11 (-0.15, sd 0.01). Small anti-evidence writers exist
(MLP 27: contributes more Normal on distress frames, dev AUROC 0.10).

Offline screen (direct effect only, free): removing the anti-evidence writers or the bias writers changes the
dev AUROC by at most 0.002 - those edits would only move the threshold, i.e. they are recalibration.

## 2026-11-06T10:23:12+00:00 - Step 2 - where is the evidence? (CPU only)

A class-mean direction estimated on the EXTERNAL practice frames, applied to dev frames, separates distress from
clean at AUROC 0.54-0.63 in the residual stream up to layer 15, then 0.959 after layer 16, and 0.92-0.94 after
that - higher than the final decision's 0.913. So the decision loses part of what the model has computed.

What writes it: the projection of layer-16 ATTENTION's output on that direction separates dev frames at AUROC
0.971 (distress -0.99 vs clean -1.52); layer-16 MLP 0.872; the residual before layer 16 only 0.60. Layer-16
attention is where the decision position reads the image evidence.

Hypothesis e1: amplifying layer-16 attention's write along that direction (rank-one W' = W - c u u^T W on
layers.16.self_attn.o_proj with c < 0) lets more evidence reach the decision, which would raise AUROC rather than
just shift the threshold. Dose c = -0.5, -1, -2 on the fixed 256-frame dev subset, plus a norm-matched random
direction at c = -1. Hypothesis e2 (bias): remove the readout-direction write of MLPs 25, 13, 11 (u = the
effective readout g*(w_Distress - w_Normal)), c = 1, 1.5, 2; expected to act like a +0.9c logit shift.

## 2026-11-06T11:22:32+00:00 - Step 3 - e1 dose-response (passes 4-8, fixed 256-frame dev subset)

A base re-run through the dev path reproduces the diagnosis logits exactly (mean |ds| = 0.000), so every change
below is the edit's.

e1 (amplify layer-16 attention's write along the practice evidence direction):
  c = -0.5: recall 53.2%, specificity 100%, MCC 0.443, AUROC 0.912, mean |ds| 0.060
  c = -1  : recall 54.7%, specificity 100%, MCC 0.454, AUROC 0.912, mean |ds| 0.076
  c = -2  : recall 56.7%, specificity 100%, MCC 0.469, AUROC 0.913, mean |ds| 0.148
  random direction, c = -1: recall 52.7%, MCC 0.440, AUROC 0.911, mean |ds| 0.046
  (base on these frames: recall 52.2%, MCC 0.436, AUROC 0.912; recalibrated base: recall 77.6%, MCC 0.639)

Reading: the edit is direction-specific (distress frames move up by +0.16 at c = -2 while clean frames stay at
-3.12; the random direction moves nothing in particular), so the layer-16 evidence direction is causally
upstream of the decision - but tripling its write moves the decision very little. The later layers do not use
the size of that write; the ranking gap is not "too little evidence written at layer 16". e1 is not a useful edit.
Next: the bias edits (e2), and a unit-level look at the late layers (heads and MLP neurons of layers 20-27),
where the decision is actually written.

## 2026-11-06T11:46:41+00:00 - Step 4 - e2 dose-response (passes 9-12)

e2 (remove the readout-direction write of the near-constant Normal-pushing MLPs 25, 13, 11):
  c = 1  : recall 56.2%, MCC 0.465, AUROC 0.911, shift distress +0.17 / clean +0.20, mean |ds| 0.175
  c = 1.5: recall 59.7%, MCC 0.491, AUROC 0.914, mean |ds| 0.276
  c = 2  : recall 61.2%, MCC 0.503, AUROC 0.913, mean |ds| 0.354
  random directions, c = 1.5: recall 52.2%, MCC 0.436, mean |ds| 0.063
The direct-effect estimate for c = 1 was a +0.93 shift; the measured shift is +0.18. The later layers repair most of
what is removed in the middle of the network (a "self-repair" effect), so edits at layers 11-25 buy little. It is a
pure shift (AUROC unchanged), i.e. recalibration by other means, and far weaker than the recalibration control
(MCC 0.639 at a +1.56 offset). Conclusion so far: both the evidence edit and the bias edit are causal and
direction-specific, but the decision is robust to rank-one edits upstream of the last layers.
Next: unit-level capture (heads and neurons of layers 20-27), then edits AT the last layer, where nothing
downstream can repair them and the offline estimate is exact up to the final RMSNorm.

## 2026-11-06T12:10:51+00:00 - Step 5 - unit-level localisation (passes 13, 14, 16: o_proj and down_proj inputs, layers 20-27)

Per-head direct contributions to s (o_proj input of each 128-dim head times its o_proj columns, projected on the
effective readout g*(w_Distress - w_Normal), divided by the same frame's final rms; heads sum to the layer's
attention contribution within 0.04; the bf16 checkpoint weights were used for this attribution, the NF4 weights
that the server serves differ from them by their quantisation error).

The bias is a tug of war between a few heads that ignore the image:
  head 26.17: -6.80 on every frame (sd 0.14; class difference -0.007; AUROC 0.48 practice / 0.37 dev)
  head 27.1 : +4.62 (sd 0.14), head 26.5: +3.69 (sd 0.19), head 25.9: +2.17 (sd 0.15) - near-constant too
Head 26.17 is a pure "Normal prior": its output is the same whatever the photo shows. Evidence heads (class
difference > 0.15, dev AUROC 0.89-0.95): 23.20, 22.1, 25.9, 27.15, 23.19, 23.11, 26.16. Anti-evidence heads
(write less Distress on distress frames, dev AUROC 0.08-0.11): 27.16 (diff -0.23), 26.21, 26.4, 27.20.

Edit e3 (amplify the last attention layer's readout write) measured +0.31 at c = -0.5 against a +0.8 direct
estimate: even the last MLP and the final norm absorb part of a late edit.

Hypotheses: e5 - remove a fraction c of head 26.17's write along the readout direction (a column edit on that
head's 128 o_proj columns, D = c r r^T W[:,S]; the post's "edit one head's output columns", restricted to the
one direction the decision reads) - a bias fix located where the bias is made; e6 - remove the readout write of
the four anti-evidence heads (could raise AUROC, since they subtract evidence).

## 2026-11-06T12:46:09+00:00 - Step 6 - last-layer and head edits (passes 15, 17-23)

e3 amplify layer-27 attention readout write: c = -0.5 MCC 0.484 (shift +0.31); c = -1 MCC 0.523 (shift +0.62);
   random direction c = -1: MCC 0.440 (no shift).
e4 remove layer-27 MLP readout write: MCC 0.454, AUROC 0.915; distress +0.35, clean -0.36: the only edit so far
   that pulls the classes apart (mean gap 3.84 vs 3.13), and the review share drops 37.1% -> 27.7%.
e3 + e4: super-additive - recall 77.1%, specificity 94.5%, MCC 0.605; MLP 27 is what absorbed e3 on its own.
e5 remove head 26.17's readout write: c = 0.25 MCC 0.523 (shift +0.56); c = 0.5 MCC 0.561 (shift +1.08),
   specificity 100% at both.

Check against recalibration (same frames): for every edit, shifting the BASE logits by the edit's own mean shift
gives the same or a slightly better MCC (e5 c=0.5: 0.561 vs 0.575 for base + 1.09; e3+e4: 0.605 vs 0.639 for
base + 1.61). The edits are real and causal - random directions at the same places and doses do nothing - but on
this model what they buy is a threshold move, and the threshold can be moved without touching a weight. The
obstacle is a constant 'Normal prior', written mostly by one head (26.17), not missing or hidden evidence; the
ranking (AUROC 0.91 on dev) is unchanged by every edit tried.
Remaining: e5 c = 0.75 / 1.0, random control on head 26.17, e6 (anti-evidence heads), e5 + e4 combinations,
then pick the final edit on dev MCC and check Stage 2 / probe regressions before freezing.

## 2026-11-06T13:35:45+00:00 - Step 7 - combinations and the choice of the final edit (passes 24-29)

e5 c = 1.0 : recall 79.6%, specificity 92.7%, MCC 0.619 (shift +2.14, review share up to 42.6%)
e5 c = 0.75: recall 75.1%, specificity 100%,  MCC 0.627 (shift +1.62 - the recalibration offset is +1.56)
random direction on head 26.17, c = 0.5: nothing (mean |ds| 0.038)
e6 (anti-evidence heads' readout write removed): distress -0.27, clean -0.43, AUROC 0.912 - no gain
e5 0.5 + e4 : specificity 70.9% (overshoot - the two interact super-additively, MLP 27 was absorbing e5 too)
e5 0.75 + e4: specificity 27.3% (overshoot)
e5 0.25 + e4: recall 73.6%, specificity 100%, MCC 0.612, lowest review share (31.6%)
random direction on MLP 27, c = 1: nothing (mean |ds| 0.008)

Choice (pre-registered criterion: dev behaviour; simplest edit on ties): e5 at c = 0.75 - remove three quarters of
the Normal-prior head 26.17's write along the decision's readout direction. It is the best dev MCC of all edits
(0.627; recalibrated base 0.639 on the same frames), it touches 128 columns of one matrix
(layers.26.self_attn.o_proj, a rank-one change within that head's block), and its random control does nothing.
Expectation stated before the test: it will behave like the recalibrated base - same ranking (AUROC), a threshold
move of about the same size - so under the pre-registered rule it is unlikely to count as "better".
Next: Stage 2 / probe regression check (2 passes), freeze, bake, the one held-out run.

## 2026-11-06T15:22:24+00:00 - Step 8 - the held-out run (final_20261106T191810_bfc0, the only test evaluation)

Baked bf16 vs hooks on 20 practice frames: same decision 20/20, P(Distress) within 0.0075, logits within 0.125.
Test (547 frames): base recall 51.7%, MCC 0.208, AUROC 0.957; recalibrated base 77.1% / 0.352; edited weights
74.8% / 0.333 / 0.955; random control 52.1% / 0.209. Specificity 23/23 for all. Edited - recalibrated MCC
-0.019 [-0.036, -0.006]; AUROC edited - base -0.001 [-0.005, +0.001]. Bengaluru Distress share 19.7% -> 36.7%.
Verdict under the pre-registered rule: not better. The edit removes the obstacle it was aimed at, as precisely as
predicted on dev, and recalibration removes it as well, slightly better.

