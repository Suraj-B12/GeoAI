"""
Editor-side helpers for weight-space programming (paper §6.11): the "kernel".

The editor works in a CPU Python process. It may read weights (dequantized by
the server, saved as fp16 files), read captured activations, compute
statistics, and save edits (wsp_core.save_edit). It runs the model ONLY by
asking the forward-pass server for a pass (wsp_server.py, port 8011), and every
pass is counted in the ledger.

    from scripts import wsp_client as C
    res = C.stage1(keys, purpose="...", edits=[], capture={"layers": [..], "sites": ["mlp"]})
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from scripts import wsp_core as W  # noqa: E402

URL = f"http://127.0.0.1:{W.PORT}"


def _post(path: str, body: dict, timeout: float = 3600) -> dict:
    import httpx
    r = httpx.post(URL + path, json=body, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"{path} -> {r.status_code}: {r.text[:500]}")
    return r.json()


def status() -> dict:
    import httpx
    return httpx.get(URL + "/status", timeout=10).json()


def run(kind: str, keys: list[str], purpose: str, edits: list[str] | None = None,
        capture: dict | None = None, final: bool = False, run_id: str | None = None) -> dict:
    """One pass (the caller keeps len(keys) within the per-pass cap)."""
    return _post("/pass", {"kind": kind, "items": keys, "purpose": purpose, "edits": edits or [],
                           "capture": capture, "final": final, "run_id": run_id})


def stage1(keys, purpose, edits=None, capture=None) -> dict:
    return run("stage1", keys, purpose, edits, capture)


def weight(module: str):
    """Dequantized weight (out x in, float32 on CPU) of one editable module."""
    import torch
    W.check_module_name(module)
    cached = W.WSP_DIR / "weights" / f"{module}.pt"      # written by the server's /weight
    p = cached if cached.exists() else _post("/weight", {"module": module})["path"]
    return torch.load(p, weights_only=True).float()


def load_capture(path: str) -> dict:
    z = np.load(path)
    return {k: z[k] for k in z.files}


def save_json(name: str, obj) -> Path:
    p = W.WSP_DIR / "analysis" / name
    from scripts.ft_common import atomic_write_json
    atomic_write_json(p, obj)
    return p
