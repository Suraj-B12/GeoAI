"""
Stage 2 per-type probing: one Yes/No question per IRC:82 type, scored from the
model's own next-token probabilities.

Why this exists
---------------
The production Stage 2 asks the model to WRITE a comma-separated list of
distress types. Measured on 407 labelled Attain images (2026-09-18 run,
eval_results/calib_attain_407.json) that free-form list collapsed onto one
answer: "Longitudinal Cracking, Transverse Cracking" for 318 of 407 images
(78%) - the first two entries of the prompt's inspection protocol. Alligator
cracking was named on 11% of the images that have it, potholes on 18%, and
ravelling / weathering never. A list the model writes in protocol order is a
decoding artefact, not a per-type judgement.

Here the model is asked about every type separately and answers with one
token. P(Yes) = p(Yes)+p(yes) renormalised against p(No)+p(no) at that single
position, exactly how Stage 1 already reads Normal vs Distress. That gives:

  * a probability for EVERY type, not only the ones the model chose to write,
    so each type gets its own ROC curve and its own decision threshold;
  * no list-order effect: no type conditions on another type's answer;
  * a principled confidence: how close each decision is to its threshold.

Making it cheap
---------------
Every question shares the same system prompt and the same image, which is
more than 90% of the tokens. The shared prefix (system + image) is run ONCE
and its key/value cache is expanded to all questions, which then run as a
single batch of short suffixes. Qwen2.5-VL uses 3-D multimodal rotary
positions, so the suffix positions are taken from the model's own
get_rope_index() over the full sequence rather than assumed - and the code
asserts that they continue the prefix positions as plain text positions.

The fast path is verified against the slow one (one full forward pass per
question, no cache sharing) by `parity_check()`; production runs it at load
time and falls back to the slow path if the two disagree.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image
from transformers import DynamicCache

from scripts.irc82_taxonomy import (
    IRC82_CONDITION_INDICATORS,
    IRC82_DISTRESS_TAXONOMY,
    render_taxonomy_for_prompt,
)

# ============================================================
# Question sets
# ============================================================

YES_WORDS = ("Yes", "yes", " Yes", " yes", "YES")
NO_WORDS = ("No", "no", " No", " no", "NO")

# Things the photographs contain that are NOT distress. Shared by every
# question through the system prompt, so it costs nothing per question.
_EXCLUSIONS = (
    "Do NOT count as distress: shadows, painted lane markings or arrows, "
    "water on the road or wet patches, oil stains, tyre skid marks, manhole "
    "covers, drainage grates, leaves or debris lying on the surface, joints "
    "between lanes that are sealed and intact."
)

SYSTEM_PROMPTS = {
    # Minimal: role + answer protocol + exclusions.
    "min": (
        "You are a pavement inspection engineer who rates bituminous road "
        "surfaces under IRC:82-2015 (Indian Roads Congress Code of Practice "
        "for Maintenance of Bituminous Road Surfaces). You will be asked "
        "about ONE distress type at a time. Answer with a single word, Yes or "
        "No: Yes only if that distress is visible on the road surface in the "
        "photo, otherwise No. " + _EXCLUSIONS
    ),
}
# "tax": the same, plus the whole IRC:82 taxonomy, so that each question is
# answered knowing which neighbouring types exist (pothole vs shallow
# depression, alligator vs transverse block cracking). The block sits in the
# shared prefix, so it is paid for once per image, not once per question.
SYSTEM_PROMPTS["tax"] = (SYSTEM_PROMPTS["min"] + "\n\n"
                         + render_taxonomy_for_prompt())


@dataclass(frozen=True)
class ProbeType:
    """One Yes/No question and the label it produces when answered Yes."""
    key: str            # canonical name (IRC distress type or condition indicator)
    output_label: str   # label written to distress_types / indicators
    kind: str           # "distress" | "indicator"
    section: str        # IRC:82 section
    description: str
    visual_cues: str


# Sub-patterns: a visually distinct form that IRC:82 files under an existing
# type. A Yes emits the parent type's label. IRC:82-2015 §7.3.5.1 defines
# transverse cracks as cracks "in the transverse directions or as
# interconnected cracks forming series of large blocks perpendicular to the
# direction of the road" - i.e. block cracking is a form of Transverse
# Cracking, not of Alligator Cracking (which §7.3.3 defines by SMALL
# irregular blocks). Asking about the block form separately lets it be
# measured against Attain's "Block crack" class without inventing a 19th type.
SUB_PATTERNS = [
    ProbeType(
        key="Block Cracking",
        output_label="Transverse Cracking",
        kind="pattern",
        section="7.3.5",
        description=("Interconnected cracks dividing the surface into large, "
                     "roughly rectangular blocks - the block form of Transverse "
                     "Cracking in IRC:82 §7.3.5.1. The blocks are much larger "
                     "than the small irregular pieces of alligator cracking."),
        visual_cues=("A grid of long straight cracks running both along and "
                     "across the road, enclosing rectangular blocks typically "
                     "0.3 m to 3 m on a side."),
    ),
]


def all_probe_types(include_indicators: bool = True,
                    include_patterns: bool = True) -> list[ProbeType]:
    """Every IRC:82 distress type, plus condition indicators and sub-patterns.

    Generated from scripts/irc82_taxonomy.py so the probes can never drift from
    the taxonomy the rest of the pipeline uses.
    """
    out = [ProbeType(k, k, "distress", e["irc_section"], e["description"],
                     e["visual_cues"])
           for k, e in IRC82_DISTRESS_TAXONOMY.items()]
    if include_indicators:
        out += [ProbeType(k, k, "indicator", e["irc_section"], e["description"],
                          e["visual_cues"])
                for k, e in IRC82_CONDITION_INDICATORS.items()]
    if include_patterns:
        out += list(SUB_PATTERNS)
    return out


def question_for(p: ProbeType, style: str) -> str:
    """The user-turn text for one probe.

    style:
      name  the IRC name only
      def   name + IRC definition + visual cues (the taxonomy's own text)
    """
    tail = " Answer Yes or No."
    if style == "name":
        return f"Does this road photo show {p.key}?" + tail
    if style == "def":
        return (f"Does this road photo show {p.key} (IRC:82 §{p.section})? "
                f"Definition: {p.description} What it looks like: "
                f"{p.visual_cues}" + tail)
    raise ValueError(f"unknown question style {style!r}")


# ============================================================
# The prober
# ============================================================

class TypeProber:
    """Scores P(Yes) for a set of Yes/No questions about one image.

    Stateless between calls apart from cached token ids. Not thread-safe on
    its own: the caller (PavementClassifier) holds its inference lock.
    """

    def __init__(self, model, processor, probe_types: list[ProbeType],
                 system_style: str = "min", question_style: str = "def"):
        if system_style not in SYSTEM_PROMPTS:
            raise ValueError(f"unknown system style {system_style!r}")
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.probe_types = list(probe_types)
        self.system_style = system_style
        self.question_style = question_style
        self.system_prompt = SYSTEM_PROMPTS[system_style]
        self.questions = [question_for(p, question_style) for p in self.probe_types]

        self.yes_ids = self._single_token_ids(YES_WORDS)
        self.no_ids = self._single_token_ids(NO_WORDS)
        if not self.yes_ids or not self.no_ids:
            raise RuntimeError("Yes/No do not tokenize to single tokens for this "
                               "tokenizer; per-type probing cannot be used")
        self.vision_end_id = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        pad = self.tokenizer.pad_token_id
        self.pad_id = pad if pad is not None else self.tokenizer.eos_token_id
        self.rope_owner = self._find_rope_owner()
        self._head_w = self._fp32_head_rows()
        # GPU memory the suffix batch may use for its copies of the prefix
        # keys/values. 1.5 GB keeps a 1024x1024 photo's 19 questions in one or
        # two chunks while leaving room for the production server on the
        # same 24 GB card.
        self.batch_budget_bytes = int(float(
            os.environ.get("PROBE_BATCH_BUDGET_GB", "1.5")) * 1024 ** 3)
        self.last_chunk_size = None

    # ---------------- helpers ----------------

    def _single_token_ids(self, words) -> list[int]:
        ids = set()
        for w in words:
            enc = self.tokenizer.encode(w, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(enc[0])
        return sorted(ids)

    def _find_rope_owner(self):
        """The module that implements get_rope_index (differs under PEFT)."""
        m = self.model
        for _ in range(4):
            if hasattr(m, "get_rope_index"):
                return m
            inner = getattr(m, "model", None)
            if inner is None:
                inner = getattr(m, "base_model", None)
            if inner is None:
                break
            m = inner
        return None

    def _fp32_head_rows(self):
        """The lm_head rows for the Yes/No tokens, in float32, or None.

        bf16 logits are rounded to ~0.06-0.125 at the magnitudes a 7B model
        produces, so P(yes) comes out in visible steps (0.469, 0.500, 0.531)
        and many images tie - which throws away ranking information that an
        ROC curve needs. Recomputing only these ~10 logits in float32 from the
        final hidden state removes the rounding at no measurable cost. Returns
        None (plain bf16 logits are used) if the head is quantized or unusual.
        """
        try:
            head = self.model.get_output_embeddings()
            w = getattr(head, "weight", None)
            if w is None or w.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                return None
            if getattr(head, "bias", None) is not None:
                return None
            ids = torch.tensor(self.yes_ids + self.no_ids, device=w.device)
            return w.index_select(0, ids).float().clone()
        except Exception:
            return None

    def _chat_text(self, image: Image.Image, question: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ]},
        ]
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

    def _yes_prob(self, logits: torch.Tensor,
                  hidden: Optional[torch.Tensor] = None) -> tuple[float, float]:
        """(P(yes | yes-or-no), probability mass on yes+no tokens) at one
        position. The mass is a health check: near 1 means the model is
        answering the question as asked. With `hidden` (the final hidden state
        at that position) the Yes/No logits are recomputed in float32."""
        lp = torch.log_softmax(logits.float(), dim=-1)
        y = torch.logsumexp(lp[self.yes_ids], dim=0)
        n = torch.logsumexp(lp[self.no_ids], dim=0)
        mass = (y.exp() + n.exp()).item()
        if hidden is not None and self._head_w is not None:
            z = self._head_w @ hidden.float().to(self._head_w.device)
            k = len(self.yes_ids)
            p_yes = torch.sigmoid(torch.logsumexp(z[:k], 0)
                                  - torch.logsumexp(z[k:], 0)).item()
        else:
            p_yes = torch.sigmoid(y - n).item()
        return p_yes, mass

    class _HeadInput:
        """Forward pre-hook that keeps the input of the output head (the final
        hidden state for the positions logits_to_keep selected)."""

        def __init__(self, module):
            self.value = None
            self._h = module.register_forward_pre_hook(self._hook) if module is not None else None

        def _hook(self, _module, args):
            self.value = args[0].detach()

        def close(self):
            if self._h is not None:
                self._h.remove()
                self._h = None

    def _capture(self):
        head = self.model.get_output_embeddings() if self._head_w is not None else None
        return self._HeadInput(head)

    # ---------------- slow reference path ----------------

    @torch.inference_mode()
    def probe_reference(self, image: Image.Image) -> list[dict]:
        """One complete forward pass per question. Slow and obviously correct;
        used to verify the fast path, and as the fallback if it ever fails."""
        out = []
        for p, q in zip(self.probe_types, self.questions):
            text = self._chat_text(image, q)
            inputs = self.processor(text=[text], images=[image], padding=True,
                                    return_tensors="pt").to(self.model.device)
            cap = self._capture()
            try:
                res = self.model(**inputs, use_cache=False, logits_to_keep=1)
            finally:
                cap.close()
            hid = cap.value[0, -1] if cap.value is not None else None
            p_yes, mass = self._yes_prob(res.logits[0, -1], hid)
            out.append({"key": p.key, "p_yes": p_yes, "yes_no_mass": mass})
            del res, inputs, cap
        return out

    # ---------------- fast shared-prefix path ----------------

    @torch.inference_mode()
    def probe(self, image: Image.Image) -> list[dict]:
        """All questions against one shared (system + image) prefix."""
        if self.rope_owner is None:
            raise RuntimeError("model exposes no get_rope_index; cannot build "
                               "multimodal positions for the shared prefix")
        device = self.model.device
        texts = [self._chat_text(image, q) for q in self.questions]

        # Full encoding of question 0: gives the prefix tokens, the image
        # tensors, and the reference positions.
        full = self.processor(text=[texts[0]], images=[image], padding=True,
                              return_tensors="pt")
        ids0 = full["input_ids"][0]
        ve = (ids0 == self.vision_end_id).nonzero()
        if len(ve) != 1:
            raise RuntimeError(f"expected exactly one <|vision_end|>, found {len(ve)}")
        P = int(ve[0, 0]) + 1

        # Suffixes = everything after <|vision_end|>. Special tokens split the
        # BPE stream, so tokenizing a suffix on its own gives the same ids as
        # tokenizing the whole text; asserted below for question 0.
        marker = "<|vision_end|>"
        head0 = texts[0].split(marker, 1)[0]
        suffix_ids = []
        for t in texts:
            head, tail = t.split(marker, 1)
            if head != head0:
                raise RuntimeError("questions do not share the same prefix text")
            suffix_ids.append(self.tokenizer(tail, add_special_tokens=False)["input_ids"])
        if ids0[P:].tolist() != suffix_ids[0]:
            raise RuntimeError("suffix tokenization does not match the full "
                               "encoding at the image boundary")

        # 3-D rope positions from the model itself, over the full sequence.
        full_dev = full.to(device)
        pos_full, _ = self.rope_owner.get_rope_index(
            full_dev["input_ids"], full_dev.get("image_grid_thw"), None,
            attention_mask=full_dev["attention_mask"])
        pos_prefix = pos_full[:, :, :P]
        base = pos_full[:, 0, P]
        if not bool((base == base[0]).all()):
            raise RuntimeError("first text position after the image differs "
                               "across rope dimensions")
        base = int(base[0])
        exp_tail = torch.arange(base, base + len(suffix_ids[0]), device=device)
        if not torch.equal(pos_full[0, 0, P:], exp_tail):
            raise RuntimeError("text positions after the image are not "
                               "contiguous; shared-prefix probing is unsafe")

        # 1) prefix once
        pre = self.model(
            input_ids=full_dev["input_ids"][:, :P],
            attention_mask=full_dev["attention_mask"][:, :P],
            pixel_values=full_dev["pixel_values"],
            image_grid_thw=full_dev["image_grid_thw"],
            position_ids=pos_prefix,
            use_cache=True,
            logits_to_keep=1,
        )
        prefix_kv = [(layer.keys, layer.values) for layer in pre.past_key_values.layers]
        del pre, full, full_dev

        # 2) the suffixes, left-padded so each question's final token sits in
        # the last column and logits_to_keep=1 reads all of them. They run in
        # chunks sized to a memory budget: every row needs its own copy of the
        # prefix keys/values, and on a card shared with the production server
        # an unbounded batch pushes the process past physical VRAM, where
        # Windows pages to system RAM instead of failing (measured: 1.1 s ->
        # 23-111 s per image with a 3k-token prefix and 19 rows at once).
        N = len(suffix_ids)
        S = max(len(s) for s in suffix_ids)
        ids = torch.full((N, S), self.pad_id, dtype=torch.long)
        smask = torch.zeros((N, S), dtype=torch.long)
        pos = torch.full((N, S), base, dtype=torch.long)
        for i, s in enumerate(suffix_ids):
            L = len(s)
            ids[i, S - L:] = torch.tensor(s, dtype=torch.long)
            smask[i, S - L:] = 1
            pos[i, S - L:] = torch.arange(base, base + L)
        per_row = sum(k.numel() * k.element_size() + v.numel() * v.element_size()
                      for k, v in prefix_kv)
        chunk = max(1, min(N, int(self.batch_budget_bytes // max(2 * per_row, 1))))

        out = []
        for a in range(0, N, chunk):
            b = min(N, a + chunk)
            m = b - a
            cache = DynamicCache(ddp_cache_data=[
                (k.expand(m, -1, -1, -1), v.expand(m, -1, -1, -1)) for k, v in prefix_kv])
            attn = torch.cat([torch.ones((m, P), dtype=torch.long), smask[a:b]], dim=1)
            cap = self._capture()
            try:
                res = self.model(
                    input_ids=ids[a:b].to(device),
                    attention_mask=attn.to(device),
                    position_ids=pos[a:b].to(device).unsqueeze(0).expand(3, m, S),
                    past_key_values=cache,
                    cache_position=torch.arange(P, P + S, device=device),
                    use_cache=True,
                    logits_to_keep=1,
                )
            finally:
                cap.close()
            for j in range(m):
                hid = cap.value[j, -1] if cap.value is not None else None
                p_yes, mass = self._yes_prob(res.logits[j, -1], hid)
                out.append({"key": self.probe_types[a + j].key, "p_yes": p_yes,
                            "yes_no_mass": mass})
            del res, cache, cap
        del prefix_kv
        self.last_chunk_size = chunk
        return out

    # ---------------- verification ----------------

    def parity_check(self, image: Optional[Image.Image] = None,
                     tol: float = 0.05) -> dict:
        """Fast path vs slow path on one image. `tol` is the largest allowed
        absolute difference in P(yes) - bf16 batching changes the last digits
        of the logits, not the answer."""
        if image is None:
            # A plain synthetic "road": grey with a dark line. Content does not
            # matter for parity, only that both paths see the same pixels.
            image = Image.new("RGB", (448, 448), (110, 110, 110))
            for x in range(40, 400):
                for y in range(220, 226):
                    image.putpixel((x, y), (25, 25, 25))
        t0 = time.time()
        fast = self.probe(image)
        t1 = time.time()
        slow = self.probe_reference(image)
        t2 = time.time()
        diffs = [abs(a["p_yes"] - b["p_yes"]) for a, b in zip(fast, slow)]
        return {
            "ok": max(diffs) <= tol,
            "max_abs_diff": round(max(diffs), 5),
            "mean_abs_diff": round(sum(diffs) / len(diffs), 5),
            "fast_s": round(t1 - t0, 2),
            "slow_s": round(t2 - t1, 2),
            "n_questions": len(diffs),
        }
