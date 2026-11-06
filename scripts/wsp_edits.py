"""
Edit recipes of the weight-space programming study (paper §6.11).

Each recipe turns statistics of captured activations (closed form: class means,
differences, normalisation) into a named, immutable edit saved by
wsp_core.save_edit. Directions are estimated on the EXTERNAL practice frames
(WS_V1.0 + OS_V1.0 after the contamination drop) - never on dev (dev is for
checking) and never on test.

  evidence <layer> <site>  u = unit class-mean difference (distress - clean) of
                           the rms-normalised residual stream after <layer>, on
                           practice frames; edit on <layer>'s attention o_proj
                           (site attn) or MLP down_proj (site mlp)
  readout                  u = unit effective readout direction
                           g * (w_Distress - w_Normal) (final RMSNorm weight times
                           the lm_head row difference); removing it from a
                           module's output removes that module's DIRECT write to
                           the Normal/Distress decision
  random                   seeded random unit vector, the norm-matched control
                           (u is always unit; same module, same c)
  head                     column edit on ONE attention head's 128 o_proj input columns:
                           D = c * r (r^T W[:, S]) with r = the unit readout direction
                           (or, with --random, a seeded random unit vector): removes the
                           fraction c of that head's write along r and nothing else.
                           W is the served NF4 weight, dequantized by the server.

Usage:
  venv/Scripts/python.exe scripts/wsp_edits.py --name e1_attn16_c-1 --recipe evidence --layer 16 \
      --site attn --c -1 --why "..."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts import wsp_analysis as A  # noqa: E402
from scripts import wsp_core as W  # noqa: E402

SITE_MODULE = {"attn": "self_attn.o_proj", "mlp": "mlp.down_proj"}


def evidence_direction(layer: int) -> np.ndarray:
    P = W.pools()
    meta = {r["key"]: r for r in P["dev"] + P["practice"]}
    keys, C, _s1 = A.load_base_captures()
    y = np.array([A.label(meta[k]) for k in keys])
    prac = np.array([meta[k]["pool"] == "practice" for k in keys])
    r = C[f"resid_{layer}"]
    r = r / np.sqrt((r ** 2).mean(1, keepdims=True) + A.EPS)
    d = r[prac & (y == 1)].mean(0) - r[prac & (y == 0)].mean(0)
    return d / np.linalg.norm(d)


def readout_direction() -> np.ndarray:
    ro = A.readout_rows()
    v = ro["g"] * (ro["w_distress"] - ro["w_normal"])
    return v / np.linalg.norm(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--recipe", choices=["evidence", "readout", "random", "head"], required=True)
    ap.add_argument("--head", type=int, default=None)
    ap.add_argument("--random", action="store_true", help="head recipe: random direction instead of the readout")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--site", choices=["attn", "mlp"], required=True)
    ap.add_argument("--c", type=float, required=True)
    ap.add_argument("--dir-layer", type=int, default=None, help="evidence: residual layer of the direction")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--why", required=True)
    a = ap.parse_args()
    module = f"model.language_model.layers.{a.layer}.{SITE_MODULE[a.site]}"
    if a.recipe == "head":
        from scripts import wsp_client as C
        if a.site != "attn" or a.head is None:
            sys.exit("head recipe needs --site attn and --head")
        Wt = C.weight(module)                                       # NF4 dequantized, out x in
        S = torch.arange(a.head * 128, (a.head + 1) * 128)
        if a.random:
            g = np.random.default_rng(a.seed)
            r = g.standard_normal(Wt.shape[0])
            src = f"seeded random unit vector (seed {a.seed}), norm-matched control"
        else:
            r = readout_direction()
            src = "effective readout g*(w_Distress - w_Normal)"
        r = torch.tensor(r / np.linalg.norm(r), dtype=torch.float32)
        D = a.c * torch.outer(r, r @ Wt[:, S])
        e = {"name": a.name, "kind": "columns", "module": module, "cols": S, "D": D, "c": a.c,
             "explanation": f"{a.why} | head {a.head}: D = c r (r^T W[:,S]), r = {src}",
             "recipe": "head", "head": a.head, "direction": r, "direction_source": src}
        p = W.save_edit(e, Wt)
        print(f"[edit] {a.name}: {module} head {a.head} columns c={a.c} -> {p}")
        return 0
    if a.recipe == "evidence":
        u = evidence_direction(a.dir_layer if a.dir_layer is not None else a.layer)
        src = f"practice class-mean difference of the rms-normalised residual after layer {a.dir_layer or a.layer}"
    elif a.recipe == "readout":
        u = readout_direction()
        src = "effective readout g*(w_Distress - w_Normal)"
    else:
        g = np.random.default_rng(a.seed)
        u = g.standard_normal(3584)
        u /= np.linalg.norm(u)
        src = f"seeded random unit vector (seed {a.seed}), norm-matched control"
    e = {"name": a.name, "kind": "rank1_out", "module": module,
         "u": torch.tensor(u, dtype=torch.float32), "c": a.c,
         "explanation": f"{a.why} | u = {src}", "recipe": a.recipe, "direction_source": src}
    p = W.save_edit(e)
    print(f"[edit] {a.name}: {module} rank1_out c={a.c} -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
