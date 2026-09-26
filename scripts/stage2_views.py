"""
Where the per-type probe looks: the whole photo, tiles of it, or zoomed
crops around damage the model itself located.

Why this exists
---------------
The probe sees each photo resized to at most 1024x1024. A crack a few pixels
wide can vanish in that resize, and a small pothole is a small fraction of a
wide road frame. Asking the model to "focus" does not bring pixels back;
cropping does - it spends the same pixel budget on a smaller patch of road.
Two ways to choose the patches:

  tiles  - cut the photo into a grid of roughly square, slightly overlapping
           tiles (about four), optionally upscaled; probe every tile.
  zoom   - ask Qwen2.5-VL to box the damaged spots (it outputs JSON boxes in
           the coordinates of the resized image it saw), crop those boxes
           with a margin, upscale, probe every crop.

A type's score is then the maximum P(yes) over the views (the full photo is
always one of them): damage anywhere in the photo is damage in the photo.

Measured before building this (Attain dev frames): the model's boxes found
about half of the annotated damage when asked for tight boxes, and boxed the
whole frame when asked loosely - see eval_results/stage2_views_report.md for
the full comparison.
"""

from __future__ import annotations

import json
import math
import re
import time
from typing import Optional

import torch
from PIL import Image

GROUND_PROMPT = (
    "Detect each individual damaged spot on the road surface - every crack, "
    "crack network, pothole, ravelled patch or patch repair - and draw a "
    "separate tight bounding box around each one. Do not draw one box around "
    "the whole road. Output JSON: [{\"bbox_2d\": [x1, y1, x2, y2], \"label\": "
    "\"<damage type>\"}], at most 10 boxes. If there is no damage, output []."
)


# ============================================================
# Geometry
# ============================================================

def grid_boxes(w: int, h: int, target: int = 4, overlap: float = 0.1) -> list[list[float]]:
    """About `target` roughly square tiles covering the image, each extended
    by `overlap` of its size so damage on a tile border is seen whole."""
    side = math.sqrt(w * h / target)
    cols = max(1, round(w / side))
    rows = max(1, round(h / side))
    tw, th = w / cols, h / rows
    ox, oy = tw * overlap, th * overlap
    out = []
    for r in range(rows):
        for c in range(cols):
            out.append([max(0.0, c * tw - ox), max(0.0, r * th - oy),
                        min(float(w), (c + 1) * tw + ox), min(float(h), (r + 1) * th + oy)])
    return out


def crop_view(img: Image.Image, box, upscale_limit: float = 2.0,
              max_pixels: int = 1024 * 1024) -> Image.Image:
    """Crop `box` and upscale it by up to `upscale_limit` (linear) without
    exceeding `max_pixels` - more pixels per crack, same processor budget."""
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    crop = img.crop((x1, y1, max(x2, x1 + 1), max(y2, y1 + 1)))
    cw, ch = crop.size
    scale = min(upscale_limit, math.sqrt(max_pixels / max(cw * ch, 1)))
    if scale > 1.01:
        crop = crop.resize((max(1, int(cw * scale)), max(1, int(ch * scale))), Image.BICUBIC)
    return crop


def _expand(b, margin, w, h, min_side):
    x1, y1, x2, y2 = b
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    nw = max(bw * (1 + 2 * margin), min_side)
    nh = max(bh * (1 + 2 * margin), min_side)
    return [max(0.0, cx - nw / 2), max(0.0, cy - nh / 2), min(float(w), cx + nw / 2), min(float(h), cy + nh / 2)]


def _overlap(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def merge_boxes(boxes: list) -> list:
    """Union overlapping boxes until none overlap."""
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        out = []
        while boxes:
            b = boxes.pop()
            for o in out:
                if _overlap(b, o):
                    o[0], o[1] = min(o[0], b[0]), min(o[1], b[1])
                    o[2], o[3] = max(o[2], b[2]), max(o[3], b[3])
                    changed = True
                    break
            else:
                out.append(b)
        boxes = out
    return boxes


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def select_zoom_boxes(boxes: list, w: int, h: int, max_boxes: int = 4,
                      max_area_frac: float = 0.5, margin: float = 0.2,
                      min_side_frac: float = 0.25, nms_iou: float = 0.3) -> list:
    """Crops to zoom into, from damage boxes.

    Each box is expanded by `margin` and to at least `min_side_frac` of the
    image's short side (context: a pothole cropped to its own rim is hard to
    recognise). Boxes covering more than `max_area_frac` of the image are
    dropped - the full view already sees them at that scale. The rest are
    kept largest-first, skipping any that overlap a kept one by IoU >
    `nms_iou`, up to `max_boxes`. No transitive merging: on road-filling
    frames merging turned many small damage boxes into one near-full-frame
    blob, i.e. no zoom at all.
    """
    min_side = max(56.0, min_side_frac * min(w, h))
    ex = [_expand(b, margin, w, h, min_side) for b in boxes]
    ex = [b for b in ex if (b[2] - b[0]) * (b[3] - b[1]) <= max_area_frac * w * h]
    ex.sort(key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
    keep = []
    for b in ex:
        if all(_iou(b, k) <= nms_iou for k in keep):
            keep.append(b)
        if len(keep) >= max_boxes:
            break
    return keep


# ============================================================
# Model-side locator
# ============================================================

class DamageLocator:
    """Qwen2.5-VL grounding: box the damaged spots, in original-image pixels."""

    def __init__(self, model, processor, prompt: str = GROUND_PROMPT,
                 max_new_tokens: int = 400):
        self.model = model
        self.processor = processor
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens

    @torch.inference_mode()
    def locate(self, img: Image.Image) -> dict:
        msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text", "text": self.prompt}]}]
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = self.processor(text=[text], images=[img], return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inp, max_new_tokens=self.max_new_tokens, do_sample=False,
                                  temperature=None, top_p=None)
        raw = self.processor.batch_decode(out[:, inp["input_ids"].shape[1]:],
                                          skip_special_tokens=True)[0]
        # Qwen2.5-VL answers in the coordinates of the RESIZED image it was
        # shown (grid x 14 px); checked on 1920x1080 frames, where box recall
        # was 0.30 read this way and 0.06 read as original pixels.
        rh = int(inp["image_grid_thw"][0][1]) * 14
        rw = int(inp["image_grid_thw"][0][2]) * 14
        del out, inp
        return {"boxes": parse_boxes(raw, img.width / rw, img.height / rh, img.width, img.height),
                "raw": raw, "resized": [rw, rh]}


def parse_boxes(raw: str, sx: float, sy: float, w: int, h: int) -> list:
    """JSON boxes from the model's reply, scaled to the original image and
    clipped. Malformed output returns [] (the full view is always probed)."""
    m = re.search(r"\[.*\]", raw or "", re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except Exception:
        return []
    out = []
    for it in items if isinstance(items, list) else []:
        try:
            x1, y1, x2, y2 = [float(v) for v in it["bbox_2d"]]
        except Exception:
            continue
        b = [max(0.0, min(x1, x2) * sx), max(0.0, min(y1, y2) * sy),
             min(float(w), max(x1, x2) * sx), min(float(h), max(y1, y2) * sy)]
        if b[2] - b[0] > 4 and b[3] - b[1] > 4:
            out.append(b)
    return out


def max_over_views(view_ps: list) -> dict:
    """Per type, the largest P(yes) over the views."""
    keys = view_ps[0].keys()
    return {k: max(v[k] for v in view_ps) for k in keys}


AGGREGATIONS = ("+full", "only", "avgmax", "mean", "logitmean")


def aggregate(full: dict, views: list, mode: str = "+full") -> dict:
    """Combine the full-photo scores with the extra views' scores.

    +full      max over {full} U views
    only       max over the views (full if there are none)
    avgmax     (full + max(views)) / 2 - the full photo keeps a vote. On the
               Attain DEV split this beat plain max for zoom crops (macro
               AUROC 0.734 vs 0.717): max over several views gives every clean
               frame more chances for one spurious high "yes".
    mean       arithmetic mean over {full} U views
    logitmean  mean of logits over {full} U views
    """
    if not views:
        return dict(full)
    if mode == "+full":
        return max_over_views(views + [full])
    if mode == "only":
        return max_over_views(views)
    if mode == "avgmax":
        mx = max_over_views(views)
        return {k: (full[k] + mx[k]) / 2 for k in full}
    pool = views + [full]
    if mode == "mean":
        return {k: sum(v[k] for v in pool) / len(pool) for k in full}
    if mode == "logitmean":
        def lg(q):
            q = min(max(q, 1e-6), 1 - 1e-6)
            return math.log(q / (1 - q))
        return {k: 1 / (1 + math.exp(-sum(lg(v[k]) for v in pool) / len(pool))) for k in full}
    raise ValueError(f"unknown aggregation {mode!r}")


def _extra_views(prober, image, spec: dict, locator, release, max_pixels) -> tuple[list, list, dict]:
    """Probe the tiles or zoom crops a view spec asks for.
    Returns (per-view scores, boxes, extra info)."""
    w, h = image.size
    ul = float(spec.get("upscale_limit", 2.0))
    extra = {}
    if spec["mode"] == "tile":
        t = spec.get("tile", {})
        boxes = grid_boxes(w, h, target=t.get("target", 4), overlap=t.get("overlap", 0.1))
    elif spec["mode"] == "zoom":
        if locator is None:
            raise RuntimeError("zoom mode needs a DamageLocator")
        loc = locator.locate(image)
        release()
        boxes = select_zoom_boxes(loc["boxes"], w, h, **spec.get("zoom", {}))
        extra["located_boxes"] = len(loc["boxes"])
    else:
        return [], [], extra
    views = []
    for b in boxes:
        res = prober.probe(crop_view(image, b, ul, max_pixels))
        release()
        views.append({r["key"]: r["p_yes"] for r in res})
    return views, boxes, extra


def probe_scores(prober, image: Image.Image, views_cfg: Optional[dict],
                 locator: Optional["DamageLocator"] = None, release=None,
                 max_pixels: int = 1024 * 1024) -> tuple[dict, dict]:
    """Per-type P(yes) for an image under a views configuration.

    The full photo is always probed first and is the fallback: if tiling or
    zooming fails for any reason, the full-photo scores are returned and the
    failure is recorded, so extra views can never make the answer worse than
    the plain probe.

    views_cfg["record"] (optional) names a second view spec whose per-view
    scores are STORED but never used for the answer - how production collects
    tile scores on real uploads while still deciding on the full photo, so the
    two can later be compared on expert-labelled Bengaluru photos without
    re-running the model. A recording failure never affects the scores.

    Returns (scores, info).
    """
    release = release or (lambda: None)
    t0 = time.time()
    full_res = prober.probe(image)
    release()
    full = {r["key"]: r["p_yes"] for r in full_res}
    info = {"mode": "full", "n_views": 0,
            "min_yes_no_mass": min(r["yes_no_mass"] for r in full_res)}
    views_cfg = views_cfg or {}
    scores = full
    mode = views_cfg.get("mode", "full")
    if mode != "full":
        try:
            views, boxes, extra = _extra_views(prober, image, views_cfg, locator, release, max_pixels)
            scores = aggregate(full, views, views_cfg.get("aggregate", "+full"))
            info.update(mode=mode, n_views=len(views),
                        boxes=[[round(v, 1) for v in b] for b in boxes], **extra)
        except Exception as e:
            scores = full
            info.update(views_error=f"{type(e).__name__}: {e}", fell_back_to="full")
    rec = views_cfg.get("record")
    if rec:
        t1 = time.time()
        try:
            views, boxes, extra = _extra_views(prober, image, rec, locator, release, max_pixels)
            info["recorded"] = {
                "mode": rec["mode"], "aggregate": rec.get("aggregate", "+full"),
                "boxes": [[round(v, 1) for v in b] for b in boxes],
                "views": [{k: round(v, 5) for k, v in vw.items()} for vw in views],
                "time_ms": round((time.time() - t1) * 1000, 1), **extra}
        except Exception as e:
            info["record_error"] = f"{type(e).__name__}: {e}"
    info["time_ms"] = round((time.time() - t0) * 1000, 1)
    return scores, info
