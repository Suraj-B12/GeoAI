"""
Bake weight-space edits into real weights (paper §6.11 deliverable).

An edit found with the forward-pass server is a formula (wsp_core): rank-one
output edit W' = W - c u u^T W, or column edit W'[:,S] = W[:,S] - D. This script
turns the chosen edits into EDITED WEIGHT MATRICES:

  make   for every edited module: dequantize the ORIGINAL NF4 weight
         (bitsandbytes.functional.dequantize_4bit), apply the edit(s) in float32
         (several edits on one module are applied in the order given), and store
         the result as a bf16 tensor <out>/<module>.pt, with manifest.json
         (module, shape, sha256, formula, c, direction u / column set, source
         edit names). No re-quantization: NF4 would re-round the whole matrix.

  apply  (library) load_baked_model(): the production model through
         scripts/model_loader.py, then each edited Linear4bit module is replaced
         by a bf16 torch.nn.Linear holding the edited weights. Nothing else
         changes.

Usage:
    venv/Scripts/python.exe scripts/wsp_apply_edit.py make --edits e1,e2 --out C:/GeoAI/wsp/edits/final
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from scripts import wsp_core as W  # noqa: E402
from scripts.ft_common import atomic_write_json, read_json, sha256_file, utc_now  # noqa: E402

torch.set_grad_enabled(False)


def _parent_and_child(model, name: str):
    mods = dict(model.named_modules())
    if name not in mods:
        cands = [k for k in mods if k.endswith(name)]
        if len(cands) != 1:
            raise KeyError(f"module {name!r} not found uniquely")
        name = cands[0]
    parent_name, child = name.rsplit(".", 1)
    return mods[parent_name], child, mods[name]


def dequant_original(mod) -> torch.Tensor:
    import bitsandbytes.functional as bnbf
    w = mod.weight
    if getattr(w, "quant_state", None) is not None:
        return bnbf.dequantize_4bit(w.data, w.quant_state).float()
    return w.detach().float()


def make_baked(model, edit_names: list[str], out_dir: Path) -> dict:
    """Compute and store the edited bf16 matrices for the named edits."""
    edits = W.load_edit_group(edit_names)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_mod: dict = {}
    for e in edits:
        by_mod.setdefault(e["module"], []).append(e)
    manifest = {"made_at": utc_now(), "source_edits": edit_names,
                "formula": {"rank1_out": "W' = W - c * u (u^T W)  (u unit)",
                            "columns": "W'[:, S] = W[:, S] - D"},
                "storage": "bf16 replacement weight (out x in); bias edited the same way when present",
                "modules": {}}
    for name, es in by_mod.items():
        _p, _c, mod = _parent_and_child(model, name)
        Wd = dequant_original(mod).cpu()
        Wn = Wd.clone()
        b = mod.bias.detach().float().cpu() if getattr(mod, "bias", None) is not None else None
        for e in es:
            W.validate_edit(e, Wn)
            Wn = W.dense_edited_weight(Wn, e)
            if b is not None and e["kind"] == "rank1_out":
                u = (e["u"].float() / e["u"].float().norm())
                b = b - float(e["c"]) * u * (u @ b)
        t = Wn.to(torch.bfloat16).contiguous()
        f = out_dir / f"{name}.pt"
        tmp = f.with_suffix(".pt.tmp")
        torch.save({"weight": t, "bias": None if b is None else b.to(torch.bfloat16)}, tmp)
        os.replace(tmp, f)
        delta = (Wn - Wd)
        manifest["modules"][name] = {
            "shape": list(t.shape), "file": f.name, "sha256": sha256_file(f),
            "edits": [{"name": e["name"], "kind": e["kind"],
                       "c": float(e["c"]) if "c" in e else None,
                       "u_file": None, "cols": e["cols"].tolist() if "cols" in e else None}
                      for e in es],
            "changed_entries": int((delta.abs() > 0).sum()),
            "delta_fro_over_w_fro": float(delta.norm() / Wd.norm()),
        }
        for k, e in enumerate(es):
            if e["kind"] == "rank1_out":
                uf = out_dir / f"{name}.u{k}.pt"
                torch.save(e["u"].float(), uf)
                manifest["modules"][name]["edits"][k]["u_file"] = uf.name
    atomic_write_json(out_dir / "manifest.json", manifest)
    return manifest


def bake_into(model, baked_dir: Path) -> dict:
    """Replace each edited Linear4bit with a bf16 nn.Linear holding the edited weight.
    Returns {module: original module} so the caller can restore it."""
    man = read_json(baked_dir / "manifest.json")
    originals = {}
    for name, info in man["modules"].items():
        f = baked_dir / info["file"]
        if sha256_file(f) != info["sha256"]:
            raise RuntimeError(f"{f} does not match its manifest sha256")
        blob = torch.load(f, weights_only=True)
        parent, child, mod = _parent_and_child(model, name)
        dev = next(mod.parameters()).device
        out_f, in_f = blob["weight"].shape
        lin = torch.nn.Linear(in_f, out_f, bias=blob["bias"] is not None, dtype=torch.bfloat16, device=dev)
        lin.weight.data.copy_(blob["weight"].to(dev))
        if blob["bias"] is not None:
            lin.bias.data.copy_(blob["bias"].to(dev))
        lin.requires_grad_(False)
        originals[name] = mod
        setattr(parent, child, lin)
    return originals


def restore(model, originals: dict) -> None:
    for name, mod in originals.items():
        parent, child, _cur = _parent_and_child(model, name)
        setattr(parent, child, mod)


def load_baked_model(baked_dir: Path, max_pixels: int = 1048576):
    """Production model (scripts/model_loader.load_vlm, 4-bit) with the edited
    modules swapped for bf16 Linear layers."""
    from scripts.model_loader import load_vlm
    model, processor, info = load_vlm("Qwen/Qwen2.5-VL-7B-Instruct", quantization_bits=4,
                                      min_pixels=256 * 28, max_pixels=max_pixels)
    info["baked"] = list(bake_into(model, Path(baked_dir)).keys())
    return model, processor, info


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--edits", required=True)
    m.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "make":
        from scripts.ft_common import GpuLock
        from scripts.model_loader import load_vlm
        with GpuLock("wsp bake edits"):
            model, _proc, _info = load_vlm("Qwen/Qwen2.5-VL-7B-Instruct", quantization_bits=4,
                                           min_pixels=256 * 28, max_pixels=1048576, run_self_test=False)
            man = make_baked(model, [e for e in a.edits.split(",") if e], Path(a.out))
        print({k: (v["shape"], v["changed_entries"]) for k, v in man["modules"].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
