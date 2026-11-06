"""
CPU tests of the weight-space programming harness (paper §6.11).

  * the forward hooks equal the dense edited weight (rank-one output edit and
    column edit), on a plain layer and on a layer whose weight went through an
    NF4-like rounding (what dequantize_4bit returns is just a dense tensor);
  * the rules refuse forbidden edits (embedding, lm_head, final norm, vision,
    whole-tensor scalar multiples, unknown kinds, bad names);
  * the ledger counts passes and only counted ones;
  * the WSP code imports no optimizer, no autograd training and no peft.

Run: CUDA_VISIBLE_DEVICES=-1 venv/Scripts/python.exe scripts/tests/test_wsp_core.py
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from scripts import wsp_core as W  # noqa: E402

MOD = "model.language_model.layers.3.mlp.down_proj"


class Tiny(torch.nn.Module):
    """Mimics the module path of the real model so apply_edits finds it."""

    def __init__(self, W_):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()
        self.model.language_model.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(4)])
        lay = self.model.language_model.layers[3]
        lay.mlp = torch.nn.Module()
        lay.mlp.down_proj = torch.nn.Linear(W_.shape[1], W_.shape[0], bias=False)
        with torch.no_grad():
            lay.mlp.down_proj.weight.copy_(W_)

    def forward(self, x):
        return self.model.language_model.layers[3].mlp.down_proj(x)


def nf4_like(Wt):
    """Round each 64-block to 16 levels scaled by its absmax (NF4's structure)."""
    levels = torch.tensor([-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
                           0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0])
    flat = Wt.flatten().view(-1, 64)
    amax = flat.abs().amax(1, keepdim=True)
    q = levels[(flat / amax).unsqueeze(-1).sub(levels).abs().argmin(-1)] * amax
    return q.view_as(Wt)


def check_equal(Wt):
    g = torch.Generator().manual_seed(1)
    m = Tiny(Wt)
    x = torch.randn(5, Wt.shape[1], generator=g)
    for e in (
        {"name": "r1", "kind": "rank1_out", "module": MOD, "explanation": "t",
         "u": torch.randn(Wt.shape[0], generator=g), "c": 1.5},
        {"name": "r1neg", "kind": "rank1_out", "module": MOD, "explanation": "t",
         "u": torch.randn(Wt.shape[0], generator=g), "c": -0.8},
        {"name": "cols", "kind": "columns", "module": MOD, "explanation": "t",
         "cols": torch.tensor([1, 5, 7, 30]), "D": torch.randn(Wt.shape[0], 4, generator=g)},
    ):
        W.validate_edit(e, Wt)
        h = W.apply_edits(m, [e])
        with torch.no_grad():
            hooked = m(x)
        for hh in h:
            hh.remove()
        dense = x @ W.dense_edited_weight(Wt, e).T
        err = (hooked - dense).abs().max().item()
        assert err < 1e-4, (e["kind"], err)
        with torch.no_grad():
            assert (m(x) - x @ Wt.T).abs().max().item() < 1e-4   # hooks removed cleanly
    # composition of two edits on the same module = sequential dense application
    e1 = {"name": "a", "kind": "rank1_out", "module": MOD, "explanation": "t",
          "u": torch.randn(Wt.shape[0], generator=g), "c": 1.0}
    e2 = {"name": "b", "kind": "rank1_out", "module": MOD, "explanation": "t",
          "u": torch.randn(Wt.shape[0], generator=g), "c": 0.5}
    h = W.apply_edits(m, [e1, e2])
    with torch.no_grad():
        hooked = m(x)
    for hh in h:
        hh.remove()
    dense = x @ W.dense_edited_weight(W.dense_edited_weight(Wt, e1), e2).T
    assert (hooked - dense).abs().max().item() < 1e-4


def test_hooks_equal_dense():
    torch.manual_seed(0)
    Wt = torch.randn(48, 64)
    check_equal(Wt)
    check_equal(nf4_like(Wt))


def test_rules_refuse():
    u = torch.randn(10)
    bad_modules = ["model.language_model.embed_tokens", "lm_head", "model.language_model.norm",
                   "model.visual.blocks.0.mlp.down_proj", "model.language_model.layers.3.input_layernorm",
                   "model.language_model.layers.3.mlp", "something.else"]
    for m in bad_modules:
        try:
            W.validate_edit({"name": "x", "kind": "rank1_out", "module": m, "explanation": "t", "u": u, "c": 1.0})
        except W.RuleViolation:
            continue
        raise AssertionError(f"{m} was not refused")
    Wt = torch.randn(10, 6)
    for e in (
        {"name": "x", "kind": "columns", "module": MOD, "explanation": "t",
         "cols": torch.arange(6), "D": 0.3 * Wt},                          # whole-tensor scalar multiple
        {"name": "x", "kind": "rank1_out", "module": MOD, "explanation": "t", "u": torch.zeros(10), "c": 1.0},
        {"name": "x", "kind": "rank1_out", "module": MOD, "explanation": "t", "u": u, "c": 9.0},
        {"name": "x", "kind": "scale", "module": MOD, "explanation": "t"},
        {"name": "bad name!", "kind": "rank1_out", "module": MOD, "explanation": "t", "u": u, "c": 1.0},
        {"name": "x", "kind": "rank1_out", "module": MOD, "explanation": "", "u": u, "c": 1.0},
    ):
        try:
            W.validate_edit(e, Wt)
        except W.RuleViolation:
            continue
        raise AssertionError(f"not refused: {e['kind']} {e.get('name')}")
    # a column edit over all columns that is NOT proportional to W is allowed
    W.validate_edit({"name": "ok", "kind": "columns", "module": MOD, "explanation": "t",
                     "cols": torch.arange(6), "D": torch.randn(10, 6)}, Wt)


def test_ledger_counts():
    with tempfile.TemporaryDirectory() as d:
        old = W.LEDGER
        W.LEDGER = Path(d) / "ledger.jsonl"
        try:
            assert W.passes_used() == 0
            W.append_ledger({"kind": "stage1", "counted": True})
            W.append_ledger({"kind": "stage1", "counted": True})
            W.append_ledger({"kind": "stage1", "counted": False, "final": True})
            assert W.passes_used() == 2
        finally:
            W.LEDGER = old


def test_gpu_lock_convention():
    """Both owner formats are read; a dead owner's lock is taken over; a live one is
    waited for; release removes the lock only when owner.txt still names us."""
    import subprocess
    from scripts import ft_common as F
    with tempfile.TemporaryDirectory() as d:
        old = F.LOCK_DIR
        F.LOCK_DIR = Path(d) / "gpu.lock"
        try:
            # a dead PID: start and finish a process, keep its PID
            dead = subprocess.Popen([sys.executable, "-c", "pass"]); dead.wait()
            for fmt in (f"pid: {dead.pid}\npurpose: x\n", f"owner=y; pid={dead.pid}; purpose=z\n"):
                F.LOCK_DIR.mkdir()
                (F.LOCK_DIR / "owner.txt").write_text(fmt, encoding="utf-8")
                assert F._owner_pid() == dead.pid
                with F.GpuLock("test", poll_s=0.01, max_wait_s=5, quiet=True):
                    assert F._owner_pid() == os.getpid()
                assert not F.LOCK_DIR.exists()
            # a live owner (this process's parent shell is alive; use our own child)
            live = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
            try:
                F.LOCK_DIR.mkdir()
                (F.LOCK_DIR / "owner.txt").write_text(f"pid: {live.pid}\n", encoding="utf-8")
                try:
                    with F.GpuLock("test", poll_s=0.05, max_wait_s=0.3, quiet=True):
                        raise AssertionError("took a live owner's lock")
                except TimeoutError:
                    pass
                assert F.LOCK_DIR.exists()
            finally:
                live.kill()
            import shutil
            shutil.rmtree(F.LOCK_DIR)
            # release leaves a lock that someone else took over
            with F.GpuLock("test", poll_s=0.01, quiet=True):
                (F.LOCK_DIR / "owner.txt").write_text("pid: 4\n", encoding="utf-8")
            assert F.LOCK_DIR.exists()
            shutil.rmtree(F.LOCK_DIR)
        finally:
            F.LOCK_DIR = old


def test_bake_equals_hook():
    """wsp_apply_edit: baked bf16 nn.Linear == hooked original (to bf16 rounding),
    the manifest records the edit, and restore() puts the original module back."""
    from scripts import wsp_apply_edit as BK
    torch.manual_seed(3)
    Wt = torch.randn(48, 64)
    m = Tiny(Wt)
    with tempfile.TemporaryDirectory() as d:
        old = W.EDIT_DIR
        W.EDIT_DIR = Path(d) / "edits"
        try:
            W.save_edit({"name": "bk1", "kind": "rank1_out", "module": MOD, "explanation": "t",
                         "u": torch.randn(48), "c": 1.2})
            W.save_edit({"name": "bk2", "kind": "columns", "module": MOD, "explanation": "t",
                         "cols": torch.tensor([0, 3, 9]), "D": torch.randn(48, 3)})
            man = BK.make_baked(m, ["bk1", "bk2"], Path(d) / "final")
            info = man["modules"][MOD]
            assert info["shape"] == [48, 64] and [e["name"] for e in info["edits"]] == ["bk1", "bk2"]
            x = torch.randn(6, 64)
            h = W.apply_edits(m, W.load_edit_group(["bk1", "bk2"]))
            with torch.no_grad():
                hooked = m(x)
            for hh in h:
                hh.remove()
            orig = BK.bake_into(m, Path(d) / "final")
            lin = m.model.language_model.layers[3].mlp.down_proj
            assert isinstance(lin, torch.nn.Linear) and lin.weight.dtype == torch.bfloat16
            with torch.no_grad():
                baked = m(x.to(torch.bfloat16)).float()
            rel = ((baked - hooked).abs().mean() / hooked.abs().mean()).item()
            assert rel < 1e-2, rel
            BK.restore(m, orig)
            assert m.model.language_model.layers[3].mlp.down_proj is orig[MOD]
            try:
                W.save_edit({"name": "bk1", "kind": "rank1_out", "module": MOD, "explanation": "t",
                             "u": torch.randn(48), "c": 1.0})
                raise AssertionError("edit names must be immutable")
            except W.RuleViolation:
                pass
        finally:
            W.EDIT_DIR = old


def test_no_training_machinery():
    for p in sorted((ROOT / "scripts").glob("wsp_*.py")):
        src = p.read_text(encoding="utf-8")
        code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
        for pat in (r"torch\.optim", r"\.backward\(", r"import peft", r"from peft",
                    r"requires_grad_\(True\)", r"autograd\.grad"):
            assert not re.search(pat, code), f"{p.name} contains {pat}"


if __name__ == "__main__":
    test_hooks_equal_dense()
    test_rules_refuse()
    test_ledger_counts()
    test_gpu_lock_convention()
    test_bake_equals_hook()
    test_no_training_machinery()
    print("test_wsp_core: 6 passed")
