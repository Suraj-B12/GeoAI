"""
Grad-CAM for the Stage 2 per-type probe: where in the photo the model looked
when it answered "Yes, this road shows <type>".

What is explained
-----------------
The probe (scripts/stage2_probe.py) asks one Yes/No question per IRC:82 type
and reads P(yes) from the next-token logits. Its log-odds,

    z = logsumexp(logits[yes tokens]) - logsumexp(logits[no tokens]),

is one number per (photo, type), so it can be traced back to the image. The
free-form Stage 2 list has no such single number, which is why the probe, not
the list, is explained.

Where the map comes from
------------------------
Qwen2.5-VL turns the photo into a grid of image tokens, one per 28 x 28 px
block of the resized photo (14 px patches, merged 2 x 2). At the 1024^2
production cap that is about 36 x 36 tokens. For a chosen set of layers we keep
the hidden states at the image-token positions (A, tokens x channels) and the
gradient of z with respect to them (G), then form two maps per layer:

    gradcam  ReLU( sum_c mean_t(G[:, c]) * A[t, c] )     (Selvaraju et al. 2017)
    gxa      ReLU( sum_c G[t, c] * A[t, c] )             (gradient x activation)

Layer 0 is the vision encoder's output as it enters the language model; layers
k >= 1 are decoder layers. Which (method, layer) localises best is chosen on
the Attain DEV split only (scripts/gradcam_localization_report.py).

Cost and safety
---------------
One forward pass with gradients through the language model and one backward
pass per (photo, type). The 4-bit weights stay frozen; only activations get
gradients. The vision encoder runs without a graph (its output is detached and
made a leaf), so no memory is spent on its activations. Nothing here changes a
label or a probability the pipeline reports.
"""

from __future__ import annotations

import functools
from typing import Optional

import numpy as np
import torch
import torch.utils.checkpoint
from PIL import Image

from scripts.stage2_probe import NO_WORDS, SYSTEM_PROMPTS, YES_WORDS, question_for

DEFAULT_LAYERS = (0, 4, 8, 14, 20, 27)
METHODS = ("gradcam", "gxa")


class ProbeGradCAM:
    """Per-type relevance maps over the image-token grid."""

    def __init__(self, model, processor, probe_types, system_style: str = "min",
                 question_style: str = "name", layers=DEFAULT_LAYERS,
                 checkpoint: bool = True):
        self.checkpoint = checkpoint
        self.model = model
        self.processor = processor
        self.tok = processor.tokenizer
        self.system_prompt = SYSTEM_PROMPTS[system_style]
        self.question_style = question_style
        self.types = {p.key: p for p in probe_types}
        self.yes_ids = self._ids(YES_WORDS)
        self.no_ids = self._ids(NO_WORDS)
        if not self.yes_ids or not self.no_ids:
            raise RuntimeError("Yes/No are not single tokens for this tokenizer")
        core = self._core()
        self.visual = core.visual
        self.decoder_layers = core.language_model.layers
        self.merge = int(getattr(self.visual, "spatial_merge_size", 2))
        n = len(self.decoder_layers)
        self.layers = tuple(sorted({int(k) for k in layers if 0 <= int(k) <= n}))
        self.image_token_id = int(model.config.image_token_id)

    # ---------------- helpers ----------------

    def _ids(self, words) -> list[int]:
        out = set()
        for w in words:
            enc = self.tok.encode(w, add_special_tokens=False)
            if len(enc) == 1:
                out.add(enc[0])
        return sorted(out)

    def _core(self):
        m = self.model
        for _ in range(4):
            if hasattr(m, "visual") and hasattr(m, "language_model"):
                return m
            m = getattr(m, "model", None) or getattr(m, "base_model", None)
            if m is None:
                break
        raise RuntimeError("could not find the visual / language_model modules")

    def _chat(self, image: Image.Image, question: str) -> str:
        msgs = [{"role": "system", "content": self.system_prompt},
                {"role": "user", "content": [{"type": "image", "image": image},
                                             {"type": "text", "text": question}]}]
        return self.processor.apply_chat_template(msgs, tokenize=False,
                                                  add_generation_prompt=True)

    # ---------------- the explanation ----------------

    def explain(self, image: Image.Image, key: str) -> dict:
        """Relevance maps for one probe type on one photo.

        Returns {p_yes, logodds, grid: (rows, cols), resized: (h, w) in px of
        the photo the model saw, maps: {"gradcam@L": float32 (rows, cols),
        "gxa@L": ...}}. Map cell (i, j) covers rows i*28..(i+1)*28 and columns
        j*28..(j+1)*28 of the resized photo, i.e. the same relative region of
        the original photo.
        """
        q = question_for(self.types[key], self.question_style)
        inputs = self.processor(text=[self._chat(image, q)], images=[image],
                                padding=True, return_tensors="pt").to(self.model.device)
        thw = inputs["image_grid_thw"][0].tolist()
        rows, cols = thw[1] // self.merge, thw[2] // self.merge
        img_pos = (inputs["input_ids"][0] == self.image_token_id).nonzero()[:, 0]
        if len(img_pos) != rows * cols:
            raise RuntimeError(f"{len(img_pos)} image tokens for a {rows}x{cols} grid")

        acts: dict = {}
        hooks = []

        def vis_hook(_m, _a, out):
            leaf = out.detach().requires_grad_(True)
            acts[0] = leaf
            return leaf

        # Always installed: the detached leaf is what lets gradients flow
        # into the decoder at all (every weight is frozen).
        hooks.append(self.visual.register_forward_hook(vis_hook))
        for k in self.layers:
            if k == 0:
                continue

            def dec_hook(_m, _a, out, k=k):
                # With checkpointing the layer runs again during backward; that
                # recomputed output is not part of the graph z was built from,
                # so only the first (real) forward output is kept.
                if k in acts:
                    return
                h = out[0] if isinstance(out, tuple) else out
                if h.requires_grad:
                    h.retain_grad()
                    acts[k] = h
            hooks.append(self.decoder_layers[k - 1].register_forward_hook(dec_hook))

        # Gradient checkpointing on the decoder layers only: activations are
        # recomputed in backward instead of stored, which keeps the peak near
        # the forward pass (measured without it: 16.4 GB for a 954-token photo,
        # more than the card has free next to the production server). Only
        # these layer objects are switched to training mode - the checkpoint
        # switch reads that flag - and every attention/MLP submodule stays in
        # eval mode (Qwen2.5-VL has no dropout either way). Restored after.
        saved = [(l, l.training, l.gradient_checkpointing,
                  getattr(l, "_gradient_checkpointing_func", None)) for l in self.decoder_layers]
        if self.checkpoint:
            for l in self.decoder_layers:
                l.gradient_checkpointing = True
                l._gradient_checkpointing_func = functools.partial(
                    torch.utils.checkpoint.checkpoint, use_reentrant=False)
                l.training = True
        try:
            with torch.enable_grad():
                res = self.model(**inputs, use_cache=False, logits_to_keep=1)
                lg = res.logits[0, -1].float()
                z = torch.logsumexp(lg[self.yes_ids], 0) - torch.logsumexp(lg[self.no_ids], 0)
                z.backward()
        finally:
            for h in hooks:
                h.remove()
            for l, tr, gc_on, fn in saved:
                l.training = tr
                l.gradient_checkpointing = gc_on
                if fn is None:
                    l.__dict__.pop("_gradient_checkpointing_func", None)
                else:
                    l._gradient_checkpointing_func = fn

        maps = {}
        for k, a in acts.items():
            if k not in self.layers:
                continue
            if k == 0:
                A, G = a.detach().float(), a.grad
            else:
                A, G = a.detach()[0, img_pos].float(), a.grad[0, img_pos] if a.grad is not None else None
            if G is None:
                continue
            G = G.float()
            gc = torch.relu((A * G.mean(0, keepdim=True)).sum(1))
            gx = torch.relu((A * G).sum(1))
            maps[f"gradcam@{k}"] = gc.reshape(rows, cols).cpu().numpy().astype(np.float32)
            maps[f"gxa@{k}"] = gx.reshape(rows, cols).cpu().numpy().astype(np.float32)
        z_val = float(z.detach())
        del res, acts, inputs
        self.model.zero_grad(set_to_none=True)
        return {"p_yes": float(torch.sigmoid(torch.tensor(z_val))), "logodds": z_val,
                "grid": (rows, cols), "resized": (thw[1] * 14, thw[2] * 14), "maps": maps}


def upsample(cam: np.ndarray, size_wh) -> np.ndarray:
    """Nearest-neighbour map at photo size (w, h), values 0..1 (for display)."""
    m = cam / cam.max() if cam.max() > 0 else cam
    im = Image.fromarray((m * 255).astype(np.uint8)).resize(tuple(size_wh), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def overlay(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Red heat overlay for viewing: strong cells red, weak cells transparent."""
    w, h = image.size
    m = upsample(cam, (w, h))
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    red = np.zeros_like(base)
    red[..., 0] = 255
    a = (alpha * m)[..., None]
    return Image.fromarray((base * (1 - a) + red * a).clip(0, 255).astype(np.uint8))


def cell_boxes(grid, size_wh) -> np.ndarray:
    """(rows*cols, 4) original-photo boxes [x1, y1, x2, y2] of each map cell."""
    rows, cols = grid
    w, h = size_wh
    xs = np.linspace(0, w, cols + 1)
    ys = np.linspace(0, h, rows + 1)
    out = np.zeros((rows, cols, 4), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            out[i, j] = (xs[j], ys[i], xs[j + 1], ys[i + 1])
    return out.reshape(-1, 4)


def hottest_box(cam: np.ndarray, size_wh, frac: float = 0.5) -> Optional[list]:
    """Bounding box of the connected region (4-neighbour) around the hottest
    cell, over cells >= frac * max. The 'suggested box' a reviewer would see."""
    if cam.max() <= 0:
        return None
    rows, cols = cam.shape
    on = cam >= frac * cam.max()
    i0, j0 = np.unravel_index(int(np.argmax(cam)), cam.shape)
    seen = np.zeros_like(on)
    stack = [(i0, j0)]
    seen[i0, j0] = True
    ii, jj = [], []
    while stack:
        i, j = stack.pop()
        ii.append(i)
        jj.append(j)
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < rows and 0 <= b < cols and on[a, b] and not seen[a, b]:
                seen[a, b] = True
                stack.append((a, b))
    w, h = size_wh
    return [min(jj) * w / cols, min(ii) * h / rows, (max(jj) + 1) * w / cols, (max(ii) + 1) * h / rows]
