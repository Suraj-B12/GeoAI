"""
Stage 2 confidence extraction — single source of truth.

Both the production classifier (`app/model.py`) and the offline evaluation
scripts compute Stage 2 confidence. Before this module they each owned a copy
of the maths, which is exactly the divergence that made
`scripts/model_loader.py` necessary for model loading: if the two drift, an
evaluation measures the evaluator rather than the model, and nothing announces
it. The functions here are the only implementation; everything else delegates.

The metric
----------
Confidence is the geometric mean of the per-token probabilities the model
assigned to the tokens it actually generated:

    confidence = exp( mean( log p_i ) )    over a chosen token span

Geometric rather than arithmetic because probabilities compound; the result is
the average per-token likelihood. Everything then turns on WHICH tokens are in
the span:

  sequence  every generated token, DESCRIPTION prose included. Measured
            anti-correlated with correctness (AUC 0.231, n=37): fluent wrong
            answers score high, hedged right answers score low.
  field     only the tokens of the DISTRESS_TYPES value — the classification
            itself. The production default since 2026-09-17.
  primary   the JOINT probability (product, not geometric mean) of the first
            label the model names; secondary labels do not lower it. Recorded
            on every row, not used as the gate: on 407 labelled Attain images
            it did not separate right first labels from wrong ones (AUC 0.468)
            where `field` did (0.635). eval_results/primary_confidence.md.

Locating the field span is the fiddly part. Byte-level BPE splits multi-byte
characters across token boundaries, so a token-by-token decode does not give
reliable character offsets. Instead the prefix is decoded cumulatively, which
yields an exact character offset at every token boundary, and the field's
character range is mapped onto a token range from that.

No string matching on the output, no heuristic scaling, no clamping to make
numbers look better: these are the model's own probabilities.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


def token_log_probs(scores: tuple, generated_ids) -> list:
    """Per-token log-probability of the token the model actually generated.

    Index i corresponds to generated token i, so the result can be sliced by a
    token span. Special/padding tokens and non-finite values become None so
    that slicing stays aligned with the token sequence rather than silently
    shifting by one.
    """
    out: list = []
    num_tokens = min(len(scores), generated_ids.shape[0])
    for i in range(num_tokens):
        token_id = generated_ids[i].item()
        if token_id <= 0:  # special / padding
            out.append(None)
            continue
        log_prob = torch.log_softmax(scores[i][0].float(), dim=0)[token_id].item()
        out.append(log_prob if math.isfinite(log_prob) else None)
    return out


def geomean(log_probs: list) -> float:
    """Geometric mean of probabilities = exp(mean(log p_i)). 0.0 if empty."""
    vals = [lp for lp in log_probs if lp is not None]
    if not vals:
        return 0.0
    return round(max(0.0, min(1.0, math.exp(sum(vals) / len(vals)))), 4)


def sequence_confidence(scores: tuple, generated_ids) -> float:
    """Geometric mean over EVERY generated token (the legacy metric)."""
    if not scores or len(scores) == 0:
        return 0.0
    return geomean(token_log_probs(scores, generated_ids))


def joint(log_probs: list) -> float:
    """Joint probability = exp(sum(log p_i)). 0.0 if empty.

    For a label that spans several tokens this is the probability the model
    gave to that exact label, not an average over its pieces.
    """
    vals = [lp for lp in log_probs if lp is not None]
    if not vals:
        return 0.0
    return round(max(0.0, min(1.0, math.exp(sum(vals)))), 4)


def _field_value_range(tokenizer, ids: list, field: str) -> Optional[tuple]:
    """(offsets, full_text, value_start, line_end) for `field`, or None.

    offsets[i] is the character length of the decoded prefix ids[:i+1].
    Decoding the prefix cumulatively gives an exact offset at every token
    boundary. Decoding token by token does not, because byte-level BPE splits
    multi-byte characters across tokens.
    """
    if not ids:
        return None
    offsets: list = []
    for i in range(len(ids)):
        offsets.append(len(tokenizer.decode(ids[: i + 1], skip_special_tokens=True)))
    full_text = tokenizer.decode(ids, skip_special_tokens=True)

    marker_pos = full_text.upper().find(f"{field}:")
    if marker_pos == -1:
        return None
    value_start = marker_pos + len(field) + 1
    line_end = full_text.find("\n", value_start)
    if line_end == -1:
        line_end = len(full_text)
    if line_end <= value_start:
        return None
    return offsets, full_text, value_start, line_end


def find_field_token_span(tokenizer, generated_ids,
                          field: str = "DISTRESS_TYPES") -> Optional[tuple]:
    """Token range [start, end) covering the value of `field` in the output.

    Returns None when the field is absent, empty, or shorter than two tokens —
    fewer than two tokens is too thin an estimate to be worth trusting, and the
    caller falls back to the whole sequence.
    """
    found = _field_value_range(tokenizer, generated_ids.tolist(), field)
    if found is None:
        return None
    offsets, _full_text, value_start, line_end = found

    # Map char range -> token range. A token belongs to the span if its end
    # offset lands inside (value_start, line_end].
    start_tok, end_tok = None, None
    for i, end_off in enumerate(offsets):
        if start_tok is None and end_off > value_start:
            start_tok = i
        if end_off <= line_end:
            end_tok = i + 1
    if start_tok is None or end_tok is None or end_tok <= start_tok:
        return None
    if end_tok - start_tok < 2:
        return None
    return start_tok, end_tok


def field_confidence(tokenizer, scores: tuple, generated_ids,
                     field: str = "DISTRESS_TYPES") -> tuple:
    """Confidence restricted to the tokens carrying the classification.

    Returns (confidence, used_field_span). When the span cannot be located the
    whole-sequence value is returned with used_field_span=False, so the caller
    always gets a usable number and can record which path was taken instead of
    the fallback being invisible.
    """
    if not scores or len(scores) == 0:
        return 0.0, False
    lp = token_log_probs(scores, generated_ids)
    span = find_field_token_span(tokenizer, generated_ids, field)
    if span is None:
        return geomean(lp), False
    start, end = span
    return geomean(lp[start:end]), True


def find_type_item_spans(tokenizer, generated_ids,
                         field: str = "DISTRESS_TYPES") -> list:
    """Token span of each comma-separated label in the field, in output order.

    Returns [(label_text, start_tok, end_tok), ...]. The split matches the
    parser in scripts/utils.py, which also splits the value on commas, so
    label_text canonicalises to the same name the parser produced.

    A token belongs to a label if its characters overlap the label's
    characters. The leading-space token (" Pot") therefore goes to its label.
    Separator tokens (",") belong to no label. Unlike the whole-field span,
    a single-token label is kept: one token is the whole label.
    """
    found = _field_value_range(tokenizer, generated_ids.tolist(), field)
    if found is None:
        return []
    offsets, full_text, value_start, line_end = found

    items = []
    pos = value_start
    for piece in full_text[value_start:line_end].split(","):
        lead = len(piece) - len(piece.lstrip())
        text = piece.strip()
        if text:
            a = pos + lead
            items.append((text, a, a + len(text)))
        pos += len(piece) + 1  # +1 for the comma

    spans = []
    for text, a, b in items:
        toks = [i for i, end_off in enumerate(offsets)
                if end_off > a and (offsets[i - 1] if i else 0) < b]
        if toks:
            spans.append((text, toks[0], toks[-1] + 1))
    return spans


def type_confidences(tokenizer, scores: tuple, generated_ids,
                     field: str = "DISTRESS_TYPES") -> list:
    """Confidence of each label the model named, in output order.

    `joint` is the probability of that exact label. The first label's joint
    probability is unconditional: it is where the model chose the primary
    type. Every later label's probability is CONDITIONAL on the labels
    already written, so later labels cannot be ranked against the first by
    this number. `geomean` is recorded for comparison only. It favours long
    names, because the tokens after the first are close to certain.
    """
    if not scores or len(scores) == 0:
        return []
    lp = token_log_probs(scores, generated_ids)
    return [{"label": text,
             "joint": joint(lp[s:e]),
             "geomean": geomean(lp[s:e]),
             "n_tokens": e - s}
            for text, s, e in find_type_item_spans(tokenizer, generated_ids, field)]


def primary_entry(type_confs: list, primary_label: Optional[str],
                  canonicalize) -> Optional[dict]:
    """The type_confidences entry that produced the parser's primary label.

    Usually the first label. It is not the first one when the parser dropped
    that label (for example "Other Distress", which canonicalises to None).
    Returns None when no label matches, for example a "Normal" answer or a
    keyword-fallback parse. The caller then falls back and records that.
    """
    if not primary_label:
        return None
    for tc in type_confs:
        if canonicalize(tc["label"]) == primary_label:
            return tc
    return None


def both_confidences(tokenizer, scores: tuple, generated_ids,
                     field: str = "DISTRESS_TYPES") -> dict:
    """Both metrics in one call — what every caller should record.

    Recording only the active one is how 1,769 images of Attain inference ended
    up with no confidence data at all, forcing a re-run to answer a question the
    original run could have answered for free.
    """
    seq = sequence_confidence(scores, generated_ids)
    fld, span_found = field_confidence(tokenizer, scores, generated_ids, field)
    return {
        "stage2_confidence_field": fld,
        "stage2_confidence_sequence": seq,
        "stage2_field_span_found": span_found,
    }
