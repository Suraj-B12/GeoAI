"""
Tests for primary-type Stage 2 confidence (GPU-free).

The primary-label confidence is the probability of the FIRST label the model names,
located by mapping characters back to tokens. Getting that mapping wrong does
not crash anything. It silently scores the wrong tokens, and every analysis
built on it measures something else. So this checks the mapping
against the real Qwen tokenizer, with synthetic logits whose probabilities
are known exactly.

What is NOT covered: whether the first label is the dominant distress, or
whether its confidence predicts correctness. Those are properties of the
model, measured on labelled data (scripts/primary_confidence_report.py).

Usage:
    venv/Scripts/python.exe scripts/tests/test_primary_confidence.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from scripts import confidence as conf  # noqa: E402
from scripts.irc82_taxonomy import canonicalize_to_irc  # noqa: E402
from scripts.utils import parse_stage2_response  # noqa: E402

FAILURES: list[str] = []
CHECKS = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if cond:
        print(f"  [OK]   {name}")
    else:
        print(f"  [FAIL] {name}" + (f" - {detail}" if detail else ""))
        FAILURES.append(name)


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")


def fake_generation(tok, text: str, prob_for):
    """Token ids for `text` plus per-step scores giving each token a chosen
    probability. prob_for(i, token_text) -> probability of token i."""
    ids = tok(text, add_special_tokens=False)["input_ids"]
    vocab = len(tok)
    scores = []
    for i, tid in enumerate(ids):
        p = prob_for(i, tok.decode([tid]))
        logits = torch.zeros(1, vocab)
        # softmax(target) = p when target logit = log(p * (V - 1) / (1 - p))
        logits[0, tid] = math.log(p * (vocab - 1) / (1 - p))
        scores.append(logits)
    return tuple(scores), torch.tensor(ids)


OUTPUT = ("DISTRESS_TYPES: Potholes, Bleeding\n"
          "SEVERITY: Medium\n"
          "DESCRIPTION: A medium pothole with a shiny patch (IRC:82 7.5.3).")


def main() -> int:
    tok = load_tokenizer()

    print("\n[1] label spans line up with the parser's labels")
    scores, ids = fake_generation(tok, OUTPUT, lambda i, t: 0.9)
    spans = conf.find_type_item_spans(tok, ids)
    labels = [s[0] for s in spans]
    check("two labels found, in output order", labels == ["Potholes", "Bleeding"], str(labels))
    parsed = parse_stage2_response(OUTPUT)["distress_types"]
    check("canonical names match the parser",
          [canonicalize_to_irc(x) for x in labels] == parsed, f"{labels} vs {parsed}")
    for text, s, e in spans:
        decoded = tok.decode(ids[s:e].tolist()).strip()
        check(f"tokens of '{text}' decode to that label", decoded == text, repr(decoded))

    print("\n[2] a hedged second label does not lower the primary confidence")
    # Primary tokens at 0.95, everything in the second label at 0.40.
    first = spans[0]
    second = spans[1]

    def probs(i, _t):
        if second[1] <= i < second[2]:
            return 0.40
        return 0.95

    scores, ids = fake_generation(tok, OUTPUT, probs)
    tcs = conf.type_confidences(tok, scores, ids)
    n1 = first[2] - first[1]
    expect_primary = round(0.95 ** n1, 4)
    check("primary joint = product of its token probabilities",
          abs(tcs[0]["joint"] - expect_primary) < 1e-3, f"{tcs[0]['joint']} vs {expect_primary}")
    check("primary is unaffected by the hedged label",
          tcs[0]["joint"] > 0.8, str(tcs[0]))
    fld, ok = conf.field_confidence(tok, scores, ids)
    check("whole-field confidence IS pulled down by it (the old behaviour)",
          ok and fld < tcs[0]["joint"], f"field={fld} primary={tcs[0]['joint']}")
    entry = conf.primary_entry(tcs, parsed[0], canonicalize_to_irc)
    check("primary_entry picks the first label", entry is tcs[0], str(entry))

    print("\n[3] single-label and single-token labels")
    single = "DISTRESS_TYPES: Bleeding\nSEVERITY: Low\nDESCRIPTION: Shiny patch."
    scores, ids = fake_generation(tok, single, lambda i, t: 0.7)
    tcs = conf.type_confidences(tok, scores, ids)
    check("one label found", [t["label"] for t in tcs] == ["Bleeding"], str(tcs))
    check("joint is never above geomean", tcs[0]["joint"] <= tcs[0]["geomean"] + 1e-9, str(tcs[0]))

    print("\n[4] labels the parser drops are skipped, not scored")
    dropped = ("DISTRESS_TYPES: Other Distress, Alligator Cracking, Potholes\n"
               "SEVERITY: High\nDESCRIPTION: x")
    scores, ids = fake_generation(tok, dropped, lambda i, t: 0.9)
    tcs = conf.type_confidences(tok, scores, ids)
    parsed = parse_stage2_response(dropped)["distress_types"]
    check("parser drops 'Other Distress'", parsed[0] == "Alligator Cracking", str(parsed))
    entry = conf.primary_entry(tcs, parsed[0], canonicalize_to_irc)
    check("primary entry is the parser's first label, not the dropped one",
          entry is not None and entry["label"] == "Alligator Cracking", str(entry))

    print("\n[5] answers with no distress label fall back cleanly")
    normal = "DISTRESS_TYPES: Normal - No distress detected\nSEVERITY: N/A\nDESCRIPTION: x"
    scores, ids = fake_generation(tok, normal, lambda i, t: 0.9)
    tcs = conf.type_confidences(tok, scores, ids)
    parsed = parse_stage2_response(normal)["distress_types"]
    check("'Normal' has no primary entry",
          conf.primary_entry(tcs, parsed[0], canonicalize_to_irc) is None, str(parsed))
    check("missing field -> no spans",
          conf.find_type_item_spans(tok, fake_generation(tok, "no field here", lambda i, t: 0.9)[1]) == [])
    check("no scores -> no confidences", conf.type_confidences(tok, (), torch.tensor([])) == [])

    print("\n[6] multi-byte characters before the field do not shift the spans")
    mb = "Note: §7.5.3 applies\nDISTRESS_TYPES: Potholes, Edge Breaking\nSEVERITY: Large"
    scores, ids = fake_generation(tok, mb, lambda i, t: 0.9)
    spans = conf.find_type_item_spans(tok, ids)
    check("labels still exact", [s[0] for s in spans] == ["Potholes", "Edge Breaking"], str(spans))
    check("span tokens decode to the label",
          all(tok.decode(ids[s:e].tolist()).strip() == t for t, s, e in spans))

    print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} checks passed")
    if FAILURES:
        print("FAILED:", ", ".join(FAILURES))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
