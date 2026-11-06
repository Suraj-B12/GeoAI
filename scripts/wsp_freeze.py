"""
Freeze the final edit of the weight-space programming study (paper §6.11).

After this, editing is over: the forward-pass server and wsp_final_eval.py treat
eval_results/finetune/wsp_final_edit/manifest.json as the signal that the ONE
held-out evaluation may run. This script
  1. checks the named edits exist and pass the rules,
  2. bakes the final edit and the random-direction control into bf16 edited
     weight matrices (scripts/wsp_apply_edit.py make; needs the GPU lock briefly),
  3. copies the small edit definitions (u vectors, c, module, explanation) into
     eval_results/finetune/wsp_final_edit/ - the baked matrices themselves stay in
     C:\\GeoAI\\wsp\\edits\\{final,random_baked} (136 MB per down_proj, too large for git),
  4. writes the manifest (refuses to overwrite an existing one).

    venv/Scripts/python.exe scripts/wsp_freeze.py --final e_a,e_b --random r_a,r_b --why "..."
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import wsp_core as W  # noqa: E402
from scripts.ft_common import atomic_write_json, read_json, sha256_file, utc_now  # noqa: E402

PY = sys.executable


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", required=True)
    ap.add_argument("--random", required=True)
    ap.add_argument("--why", required=True)
    a = ap.parse_args()
    man_path = W.FINAL_DIR / "manifest.json"
    if man_path.exists():
        sys.exit(f"{man_path} exists: the final edit is already frozen")
    final = [x for x in a.final.split(",") if x]
    rand = [x for x in a.random.split(",") if x]
    fe, re_ = W.load_edit_group(final), W.load_edit_group(rand)
    if sorted(e["module"] for e in fe) != sorted(e["module"] for e in re_):
        sys.exit("the random control must edit the same modules as the final edit")
    if sorted(float(e["c"]) for e in fe) != sorted(float(e["c"]) for e in re_):
        sys.exit("the random control must use the same c values")
    out = {"final": W.EDIT_DIR / "final", "random": W.EDIT_DIR / "random_baked"}
    for k, names in (("final", final), ("random", rand)):
        if not (out[k] / "manifest.json").exists():
            r = subprocess.call([PY, str(PROJECT_ROOT / "scripts" / "wsp_apply_edit.py"), "make",
                                 "--edits", ",".join(names), "--out", str(out[k])], cwd=str(PROJECT_ROOT))
            if r != 0:
                sys.exit(f"baking {k} failed ({r})")
    W.FINAL_DIR.mkdir(parents=True, exist_ok=True)
    for n in final + rand:
        shutil.copy2(W.EDIT_DIR / f"{n}.pt", W.FINAL_DIR / f"{n}.pt")
    led = W.ledger_rows()
    man = {"frozen_at": utc_now(), "why": a.why, "final_edits": final, "random_edits": rand,
           "baked": {k: str(v) for k, v in out.items()},
           "baked_manifests": {k: read_json(v / "manifest.json") for k, v in out.items()},
           "edit_files_sha256": {n: sha256_file(W.FINAL_DIR / f"{n}.pt") for n in final + rand},
           "edits": {e["name"]: {"module": e["module"], "kind": e["kind"], "c": float(e.get("c", 0)),
                                 "explanation": e["explanation"]} for e in fe + re_},
           "passes_used_at_freeze": sum(1 for r in led if r.get("counted")),
           "test_evaluations_before_freeze": sum(1 for r in led if r.get("final"))}
    atomic_write_json(man_path, man)
    print(f"[frozen] {man_path}: final {final}, random {rand}, passes used {man['passes_used_at_freeze']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
