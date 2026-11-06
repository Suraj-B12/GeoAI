"""
Weight-space programming (WSP) harness - shared core (paper §6.11).

Replicates the setting of Takehi, Teoh, Zhang & Khattab, "Sparks of Weight-Space
Programming: LLMs can natively read & write neural network weights" (MIT CSAIL,
blog post, 2 Nov 2026) on our production model: a frontier LLM (here: Claude, the
editor) may only (1) read weights and compute statistics on CPU, and (2) spend a
fixed budget of forward passes; it edits weights directly. No backward pass, no
autograd, no optimizer, no iterative fitting of weights to a loss.

This module holds what both the forward-pass server and the CPU analysis use:

  * Edit specifications and the RULES, enforced in code:
      - only decoder-layer projections may be edited (never the embedding, the
        unembedding / lm_head, or the final norm);
      - a whole-tensor scalar multiple is refused;
      - kinds: "rank1_out"  y' = y - c * u (u^T y)          (W' = W - c u u^T W)
               "columns"    y' = y - D x_S                   (W'[:, S] = W[:, S] - D)
  * Exact application as forward hooks on the untouched NF4 modules (the dense
    equivalent is checked by scripts/tests/test_wsp_core.py and by the server's
    start-up self-test on real dequantized layers).
  * The pass ledger and budget, and the data pools (practice / dev / test /
    bengaluru) with the test guard.

Nothing here runs the model.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from scripts.ft_common import (FT_DIR, atomic_write_json, atomic_write_text,  # noqa: E402
                               read_json, utc_now)

WSP_DIR = Path(os.environ.get("WSP_DIR", r"C:\GeoAI\wsp"))
EDIT_DIR = WSP_DIR / "edits"
CAPTURE_DIR = WSP_DIR / "captures"
LEDGER = FT_DIR / "wsp_ledger.jsonl"
NOTES = FT_DIR / "wsp_work_notes.md"
FINAL_DIR = FT_DIR / "wsp_final_edit"
BUDGET = 50                     # forward passes for diagnosis + editing (pre-registered)
MAX_IMAGES_PER_PASS = 256
MAX_GENERATIONS_PER_PASS = 64
PORT = 8011

# ============================================================
# Rules
# ============================================================

ALLOWED_MODULE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\."
    r"(mlp\.(down_proj|up_proj|gate_proj)|self_attn\.(o_proj|q_proj|k_proj|v_proj))$")
FORBIDDEN_WORDS = ("embed", "lm_head", "norm", "visual", "merger")


class RuleViolation(ValueError):
    pass


def check_module_name(name: str) -> int:
    """Return the layer index; raise RuleViolation for anything not editable."""
    low = name.lower()
    for w in FORBIDDEN_WORDS:
        if w in low:
            raise RuleViolation(f"edits to {name!r} are forbidden (embedding/unembedding/norm/vision)")
    m = ALLOWED_MODULE.match(name)
    if not m:
        raise RuleViolation(f"{name!r} is not an editable decoder projection")
    return int(m.group(1))


def validate_edit(edit: dict, weight: torch.Tensor | None = None) -> dict:
    """Check an edit against the rules. `weight` (out x in, any dtype) enables the
    whole-tensor-scalar-multiple check for column edits."""
    for k in ("name", "kind", "module", "explanation"):
        if not edit.get(k):
            raise RuleViolation(f"edit is missing {k!r}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", edit["name"]):
        raise RuleViolation(f"bad edit name {edit['name']!r}")
    check_module_name(edit["module"])
    if edit["kind"] == "rank1_out":
        u = edit["u"].float()
        if u.ndim != 1 or not torch.isfinite(u).all() or float(u.norm()) < 1e-6:
            raise RuleViolation("u must be a finite non-zero vector")
        c = float(edit["c"])
        if not (abs(c) <= 3.0 and c == c):
            raise RuleViolation("c must be finite with |c| <= 3")
        if u.numel() == 1:
            raise RuleViolation("a rank-one edit of a 1-d output is a whole-tensor scalar multiple")
    elif edit["kind"] == "columns":
        S = edit["cols"]
        D = edit["D"].float()
        if S.ndim != 1 or D.ndim != 2 or D.shape[1] != S.numel():
            raise RuleViolation("columns edit needs cols (k,) and D (out, k)")
        if not torch.isfinite(D).all():
            raise RuleViolation("D must be finite")
        if weight is not None:
            W = weight.float()
            if S.numel() == W.shape[1]:
                Ws = W[:, S]
                cos = float((Ws.flatten() @ D.flatten()) / (Ws.norm() * D.norm() + 1e-12))
                rel = float((D - (D.flatten() @ Ws.flatten() / (Ws.norm() ** 2)) * Ws).norm() / (D.norm() + 1e-12))
                if cos > 0.999 or rel < 1e-3:
                    raise RuleViolation("column edit over all columns proportional to W is a "
                                        "whole-tensor scalar multiple")
    else:
        raise RuleViolation(f"unknown edit kind {edit['kind']!r}")
    return edit


def normalized(edit: dict) -> dict:
    e = dict(edit)
    if e["kind"] == "rank1_out":
        u = e["u"].float()
        e["u"] = u / u.norm()
    return e


# ============================================================
# Hooks
# ============================================================

def make_hook(edit: dict):
    """Forward hook implementing the edit exactly on a module's output."""
    e = normalized(edit)
    if e["kind"] == "rank1_out":
        u = e["u"]
        c = float(e["c"])

        def hook(module, inputs, output):
            y = output.float()
            uu = u.to(y.device)
            y = y - c * (y @ uu).unsqueeze(-1) * uu
            return y.to(output.dtype)
        return hook
    S = e["cols"]
    D = e["D"].float()

    def hook(module, inputs, output):
        x = inputs[0]
        xs = x[..., S.to(x.device)].float()
        y = output.float() - xs @ D.to(x.device).T
        return y.to(output.dtype)
    return hook


def apply_edits(model, edits: list[dict]) -> list:
    """Register hooks; returns handles (call .remove() on each to undo)."""
    mods = dict(model.named_modules())
    handles = []
    for e in edits:
        name = e["module"]
        mod = mods.get(name)
        if mod is None:     # PEFT or wrapper prefixes
            cands = [k for k in mods if k.endswith(name)]
            if len(cands) != 1:
                for h in handles:
                    h.remove()
                raise RuleViolation(f"module {name!r} not found uniquely ({len(cands)} matches)")
            mod = mods[cands[0]]
        handles.append(mod.register_forward_hook(make_hook(e)))
    return handles


def dense_edited_weight(W: torch.Tensor, edit: dict) -> torch.Tensor:
    """W' such that x @ W'^T equals the hooked output (reference for tests)."""
    e = normalized(edit)
    W = W.float()
    if e["kind"] == "rank1_out":
        u = e["u"]
        return W - float(e["c"]) * torch.outer(u, u @ W)
    W2 = W.clone()
    W2[:, e["cols"]] -= e["D"].float()
    return W2


# ============================================================
# The Stage 1 read (shared by the server and the final evaluation)
# ============================================================

def stage1_read(clf, model, img, n_id: int, d_id: int, sys_prompt: str, user_prompt: str) -> dict:
    """Teacher-forced Stage 1 read: the production inputs (clf._build_inputs),
    one forward pass, the first generated token's logits. P(Distress) is the
    softmax over the Normal / Distress first sub-tokens, exactly as
    PavementClassifier._compute_stage1_confidence reads them."""
    with clf._lock, torch.inference_mode():
        inputs, _n = clf._build_inputs(img, sys_prompt, user_prompt)
        out = model(**inputs, use_cache=False, logits_to_keep=1)
        lg = out.logits[0, -1].float()
    ln, ld = float(lg[n_id]), float(lg[d_id])
    top = int(lg.argmax())
    return {"logit_normal": round(ln, 4), "logit_distress": round(ld, 4),
            "p_distress": float(torch.softmax(torch.tensor([ln, ld]), 0)[1]),
            "top_token": top, "top_is_label": top in (n_id, d_id),
            "n_tokens": int(inputs["input_ids"].shape[1])}


# ============================================================
# Edit storage
# ============================================================

def save_edit(edit: dict, weight: torch.Tensor | None = None) -> Path:
    """Validate and store an edit as C:\\GeoAI\\wsp\\edits\\<name>.pt + manifest entry."""
    validate_edit(edit, weight)
    EDIT_DIR.mkdir(parents=True, exist_ok=True)
    p = EDIT_DIR / f"{edit['name']}.pt"
    if p.exists():
        raise RuleViolation(f"edit {edit['name']!r} exists; edits are immutable - use a new name")
    blob = {k: v for k, v in edit.items()}
    tmp = p.with_suffix(".pt.tmp")
    torch.save(blob, tmp)
    os.replace(tmp, p)
    man = read_json(EDIT_DIR / "manifest.json", {})
    man[edit["name"]] = {k: (v if not isinstance(v, torch.Tensor) else list(v.shape))
                         for k, v in edit.items()}
    man[edit["name"]]["saved_at"] = utc_now()
    atomic_write_json(EDIT_DIR / "manifest.json", man)
    return p


def load_edit(name: str) -> dict:
    p = EDIT_DIR / f"{name}.pt"
    if not p.exists():
        raise RuleViolation(f"no edit named {name!r}")
    e = torch.load(p, weights_only=False)
    validate_edit(e)
    return e


def load_edit_group(names: list[str]) -> list[dict]:
    return [load_edit(n) for n in names]


# ============================================================
# Ledger / budget / notes
# ============================================================

def ledger_rows() -> list[dict]:
    if not LEDGER.exists():
        return []
    return [json.loads(l) for l in LEDGER.read_text(encoding="utf-8").splitlines() if l.strip()]


def passes_used() -> int:
    return sum(1 for r in ledger_rows() if r.get("counted"))


def append_ledger(rec: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def note(title: str, text: str) -> None:
    """Append one work-notes entry ("explain every step")."""
    NOTES.parent.mkdir(parents=True, exist_ok=True)
    head = "" if NOTES.exists() else ("# Weight-space programming - editor's work notes\n\n"
                                      "Written by the editor (Claude) while working; one entry per step.\n\n")
    with open(NOTES, "a", encoding="utf-8") as f:
        f.write(head + f"## {utc_now()} - {title}\n\n{text.strip()}\n\n")


# ============================================================
# Data pools
# ============================================================

def pools() -> dict:
    """{'practice': [...], 'dev': [...], 'test': [...], 'bengaluru': [...]} of
    {key, path, gt, pool, block?}. Practice = WS_V1 + OS_V1 after the
    contamination drop list; dev/test = the pre-registered WS_V2.0 split."""
    from scripts.ft_common import load_train_subset, load_v2_rows
    dec = read_json(FT_DIR / "contamination.json")
    if not dec:
        raise RuntimeError("eval_results/finetune/contamination.json missing (run ft_prepare.py)")
    drop = set(dec["dropped"])
    prac = []
    for s in ("WS_V1.0", "OS_V1.0"):
        for r in load_train_subset(s):
            if r["image"] not in drop:
                prac.append({"key": r["image"], "path": r["image_path"], "gt": r["gt"],
                             "annotated": r["annotated"], "pool": "practice"})
    v2 = load_v2_rows()
    dev = [{"key": r["image"], "path": r["image_path"], "gt": r["gt"], "block": r["block"],
            "pool": "dev"} for r in v2 if r["split"] == "dev"]
    test = [{"key": r["image"], "path": r["image_path"], "gt": r["gt"], "block": r["block"],
             "pool": "test"} for r in v2 if r["split"] == "test"]
    beng = read_json(FT_DIR / "wsp_bengaluru_items.json", {"items": []})["items"]
    return {"practice": prac, "dev": dev, "test": test,
            "bengaluru": [dict(b, pool="bengaluru") for b in beng]}
