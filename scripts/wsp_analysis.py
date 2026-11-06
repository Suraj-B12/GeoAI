"""
The editor's analysis steps for weight-space programming (paper §6.11).

Every step is a sub-command, so the evidence trail can be re-run. Steps that
need the model ask the forward-pass server for passes (counted in the ledger);
everything else is CPU statistics on captured activations and weights.

  diagnose   base passes on practice + dev with captures (residual stream, MLP
             output, attention output at the decision position, all 28 layers)
  dla        direct logit attribution of the Stage 1 decision
             (logit(Distress) - logit(Normal)) to every layer's attention and MLP,
             through the final RMSNorm, split by class; plus per-layer
             separability of the residual stream

Outputs go to C:\\GeoAI\\wsp\\analysis\\ (JSON) - small summaries are copied into
eval_results/finetune/wsp_dev_runs.json by the report.
"""

from __future__ import annotations

import argparse
import math
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from scripts import wsp_core as W  # noqa: E402
from scripts.ft_common import FT_DIR, atomic_write_json, read_json  # noqa: E402

AN = W.WSP_DIR / "analysis"
HF_SNAP = Path(os.path.expanduser(
    r"~\.cache\huggingface\hub\models--Qwen--Qwen2.5-VL-7B-Instruct\snapshots"
    r"\cc594898137f460bfe9f0759e9844b3ce807cfb5"))
N_LAYERS = 28
EPS = 1e-6


def label(r) -> int:
    return 1 if r.get("gt") else 0


def auroc(s, y) -> float:
    s = np.asarray(s, float)
    y = np.asarray(y, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    # average ranks for ties
    i = 0
    r = np.arange(1, len(allv) + 1, dtype=float)
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        r[i:j + 1] = (i + j + 2) / 2
        i = j + 1
    ranks[order] = r
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def readout_rows():
    """Final RMSNorm weight and the lm_head rows of the Normal / Distress first
    tokens, read from the checkpoint (bf16; lm_head and norms are not quantized)."""
    from safetensors import safe_open
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(HF_SNAP))
    n_id = tok.encode("Normal", add_special_tokens=False)[0]
    d_id = tok.encode("Distress", add_special_tokens=False)[0]
    idx = json.loads((HF_SNAP / "model.safetensors.index.json").read_text())["weight_map"]
    out = {}
    for k in ("model.norm.weight", "lm_head.weight"):
        with safe_open(str(HF_SNAP / idx[k]), framework="pt") as f:
            t = f.get_tensor(k)
        out[k] = t
    g = out["model.norm.weight"].float().numpy()
    wN = out["lm_head.weight"][n_id].float().numpy()
    wD = out["lm_head.weight"][d_id].float().numpy()
    return {"g": g, "w_normal": wN, "w_distress": wD, "n_id": n_id, "d_id": d_id}


# ============================================================
# diagnose
# ============================================================

def cmd_diagnose(a) -> None:
    from scripts import wsp_client as C
    P = W.pools()
    items = P["dev"] + P["practice"]
    keys = [r["key"] for r in items]
    out = read_json(AN / "base_diagnose.json", {"passes": [], "s1": {}})
    done = set(out["s1"])
    todo = [k for k in keys if k not in done]
    cap = {"layers": list(range(N_LAYERS)), "sites": ["resid", "mlp", "attn"]}
    for i in range(0, len(todo), W.MAX_IMAGES_PER_PASS):
        chunk = todo[i:i + W.MAX_IMAGES_PER_PASS]
        r = C.stage1(chunk, purpose=("Diagnosis: base Stage 1 first-token logits on practice+dev, with "
                                     "residual/MLP/attention outputs at the decision position for all 28 "
                                     "layers, to locate what writes the Normal-vs-Distress decision"),
                     capture=cap)
        for x in r["results"]:
            out["s1"][x["key"]] = x
        out["passes"].append({"pass_id": r["ledger"]["pass_id"], "capture": r["capture_file"],
                              "keys": chunk})
        atomic_write_json(AN / "base_diagnose.json", out)
        print(f"[diagnose] pass {r['ledger']['pass_id']} n={len(chunk)} used {r['ledger']['budget_used']}")


# ============================================================
# dla
# ============================================================

def load_base_captures():
    d = read_json(AN / "base_diagnose.json")
    keys, arrs = [], {}
    for p in d["passes"]:
        z = np.load(p["capture"])
        keys += list(z["keys"])
        for k in z.files:
            if k != "keys":
                arrs.setdefault(k, []).append(z[k].astype(np.float32))
    return keys, {k: np.concatenate(v) for k, v in arrs.items()}, d["s1"]


def cmd_dla(a) -> None:
    P = W.pools()
    meta = {r["key"]: r for r in P["dev"] + P["practice"]}
    keys, A, s1 = load_base_captures()
    ro = readout_rows()
    v = ro["g"] * (ro["w_distress"] - ro["w_normal"])          # effective readout direction
    h = A[f"resid_{N_LAYERS - 1}"]                              # final residual, pre-norm
    rms = np.sqrt((h ** 2).mean(1) + EPS)
    delta_cap = (h @ v) / rms
    delta_srv = np.array([s1[k]["logit_distress"] - s1[k]["logit_normal"] for k in keys])
    y = np.array([label(meta[k]) for k in keys])
    pool = np.array([meta[k]["pool"] for k in keys])
    # residual before layer 0 = embeddings: resid_0 - attn_0 - mlp_0
    emb = A["resid_0"] - A["attn_0"] - A["mlp_0"]
    comps = {"embed": emb}
    for l in range(N_LAYERS):
        comps[f"attn_{l}"] = A[f"attn_{l}"]
        comps[f"mlp_{l}"] = A[f"mlp_{l}"]
    contrib = {k: (c @ v) / rms for k, c in comps.items()}
    recon = sum(contrib.values())
    res = {"n": len(keys), "n_distress": int(y.sum()), "n_clean": int((1 - y).sum()),
           "consistency": {"capture_vs_server_delta_max_abs": float(np.abs(delta_cap - delta_srv).max()),
                           "sum_of_components_vs_capture_max_abs": float(np.abs(recon - delta_cap).max())},
           "by_pool": {}, "components": {}, "resid_separability": {}}
    for pl in ("dev", "practice"):
        m = pool == pl
        dd = delta_srv[m]
        yy = y[m]
        res["by_pool"][pl] = {"n": int(m.sum()), "distress": int(yy.sum()),
                              "recall_argmax": float((dd[yy == 1] > 0).mean()),
                              "specificity_argmax": float((dd[yy == 0] <= 0).mean()) if (yy == 0).any() else None,
                              "auroc": auroc(dd, yy),
                              "delta_mean_distress": float(dd[yy == 1].mean()),
                              "delta_mean_clean": float(dd[yy == 0].mean()) if (yy == 0).any() else None}
    dev = pool == "dev"
    for k, c in contrib.items():
        res["components"][k] = {
            "mean_all": float(c.mean()), "mean_distress": float(c[y == 1].mean()),
            "mean_clean": float(c[y == 0].mean()), "sd": float(c.std()),
            "auroc_dev": auroc(c[dev], y[dev]), "auroc_all": auroc(c, y)}
    # separability of the residual stream per layer: class-mean difference
    # direction estimated on PRACTICE, scored on DEV (no dev statistic in the direction)
    prac = pool == "practice"
    for l in range(N_LAYERS):
        r = A[f"resid_{l}"]
        r = r / np.sqrt((r ** 2).mean(1, keepdims=True) + EPS)
        d = r[prac & (y == 1)].mean(0) - r[prac & (y == 0)].mean(0)
        res["resid_separability"][l] = {"auroc_dev_of_practice_direction": auroc(r[dev] @ d, y[dev]),
                                        "cos_with_readout": float(d @ v / (np.linalg.norm(d) * np.linalg.norm(v)))}
    atomic_write_json(AN / "dla.json", res)
    print(json.dumps({k: res[k] for k in ("n", "n_distress", "n_clean", "consistency", "by_pool")}, indent=1))
    top = sorted(res["components"].items(), key=lambda kv: kv[1]["mean_all"])
    print("most Normal-pushing (mean contribution):")
    for k, c in top[:10]:
        print(f"  {k:8s} mean {c['mean_all']:+.3f}  distress {c['mean_distress']:+.3f}  clean {c['mean_clean']:+.3f}  "
              f"sd {c['sd']:.3f}  auroc_dev {c['auroc_dev']:.3f}")
    print("most Distress-pushing:")
    for k, c in top[-8:]:
        print(f"  {k:8s} mean {c['mean_all']:+.3f}  distress {c['mean_distress']:+.3f}  clean {c['mean_clean']:+.3f}  "
              f"sd {c['sd']:.3f}  auroc_dev {c['auroc_dev']:.3f}")
    sep = sorted(res["components"].items(), key=lambda kv: -abs(kv[1]["mean_distress"] - kv[1]["mean_clean"]))
    print("largest class difference (distress - clean):")
    for k, c in sep[:10]:
        print(f"  {k:8s} diff {c['mean_distress'] - c['mean_clean']:+.3f}  auroc_dev {c['auroc_dev']:.3f}")
    print("residual separability (practice direction, dev AUROC):",
          {l: round(x["auroc_dev_of_practice_direction"], 3) for l, x in res["resid_separability"].items()})


# ============================================================
# devrun: one pass on the fixed 256-frame dev subset with given edits
# ============================================================

def dev256() -> list[str]:
    """All clean dev frames + a seeded sample of distress dev frames = 256 (one pass)."""
    p = AN / "dev256.json"
    if p.exists():
        return read_json(p)["keys"]
    import random
    dev = W.pools()["dev"]
    clean = sorted(r["key"] for r in dev if not r["gt"])
    dis = sorted(r["key"] for r in dev if r["gt"])
    rng = random.Random(20261106)
    keys = sorted(clean + rng.sample(dis, W.MAX_IMAGES_PER_PASS - len(clean)))
    atomic_write_json(p, {"keys": keys, "n_clean": len(clean), "rule": "all clean dev frames + seeded distress sample"})
    return keys


def s1_metrics(res: dict, keys: list[str], meta: dict, offset: float = 0.0) -> dict:
    s = np.array([res[k]["logit_distress"] - res[k]["logit_normal"] + offset for k in keys])
    y = np.array([label(meta[k]) for k in keys])
    pred = s > 0
    tp = int((pred & (y == 1)).sum()); fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum()); fp = int((pred & (y == 0)).sum())
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) or 1.0
    p = 1 / (1 + np.exp(-s))
    return {"n": len(keys), "recall": tp / max(1, tp + fn), "specificity": tn / max(1, tn + fp),
            "mcc": (tp * tn - fp * fn) / den, "auroc": auroc(s, y),
            "review": float((np.maximum(p, 1 - p) < 0.80).mean()),
            "mean_score_distress": float(s[y == 1].mean()), "mean_score_clean": float(s[y == 0].mean()),
            "top_is_label": float(np.mean([res[k]["top_is_label"] for k in keys]))}


def record_run(run: dict) -> None:
    p = FT_DIR / "wsp_dev_runs.json"
    d = read_json(p, {"runs": []})
    d["runs"].append(run)
    atomic_write_json(p, d)


def cmd_devrun(a) -> dict:
    from scripts import wsp_client as C
    keys = dev256()
    P = W.pools()
    meta = {r["key"]: r for r in P["dev"]}
    edits = [e for e in (a.edits or "").split(",") if e]
    r = C.stage1(keys, purpose=a.purpose, edits=edits)
    res = {x["key"]: x for x in r["results"]}
    diag = read_json(AN / "base_diagnose.json")["s1"]
    m = s1_metrics(res, keys, meta)
    base = s1_metrics(diag, keys, meta)
    run = {"name": a.name, "edits": edits, "pass_id": r["ledger"]["pass_id"], "dev": m,
           "dev_base_same_frames": base, "dose_group": a.dose_group, "c": a.c, "control": a.control,
           "explanation": a.purpose,
           "mean_abs_score_change": float(np.mean([abs((res[k]["logit_distress"] - res[k]["logit_normal"]) -
                                                       (diag[k]["logit_distress"] - diag[k]["logit_normal"]))
                                                   for k in keys]))}
    record_run(run)
    print(json.dumps({"name": a.name, "dev": {k: round(v, 4) for k, v in m.items()},
                      "budget_used": r["ledger"]["budget_used"]}), flush=True)
    return run


# ============================================================
# units: capture o_proj / down_proj inputs, then per-head and per-neuron attribution
# ============================================================

UNIT_LAYERS = list(range(20, 28))


def cmd_units_capture(a) -> None:
    """Passes: o_proj and down_proj INPUTS at the decision position, layers 20-27,
    on practice + dev, base model (what a head or neuron column edit acts on)."""
    from scripts import wsp_client as C
    P = W.pools()
    keys = [r["key"] for r in P["dev"] + P["practice"]]
    out = read_json(AN / "units_capture.json", {"passes": [], "keys_done": []})
    todo = [k for k in keys if k not in set(out["keys_done"])]
    cap = {"layers": UNIT_LAYERS, "sites": ["attn_in", "mlp_in", "resid"]}
    for i in range(0, len(todo), W.MAX_IMAGES_PER_PASS):
        chunk = todo[i:i + W.MAX_IMAGES_PER_PASS]
        r = C.stage1(chunk, purpose=("Localisation at unit level: inputs of o_proj (per-head outputs) and down_proj "
                                     "(MLP neurons) at the decision position, layers 20-27, base model, practice+dev"),
                     capture=cap)
        out["passes"].append({"pass_id": r["ledger"]["pass_id"], "capture": r["capture_file"], "keys": chunk})
        out["keys_done"] += chunk
        atomic_write_json(AN / "units_capture.json", out)
        print(f"[units] pass {r['ledger']['pass_id']} n={len(chunk)} used {r['ledger']['budget_used']}", flush=True)


def ckpt_weight(key: str) -> np.ndarray:
    """A weight from the bf16 checkpoint on CPU (for attribution only: the served
    model is NF4, which differs from these values by its quantisation error)."""
    from safetensors import safe_open
    idx = json.loads((HF_SNAP / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(str(HF_SNAP / idx[key]), framework="pt") as f:
        return f.get_tensor(key).float().numpy()


def cmd_units(a) -> None:
    """Per-head (o_proj column blocks of 128) and per-neuron (down_proj columns)
    direct contribution to s = logit(Distress) - logit(Normal), through the final
    RMSNorm of the SAME image (rms of the captured final residual)."""
    P = W.pools()
    meta = {r["key"]: r for r in P["dev"] + P["practice"]}
    uc = read_json(AN / "units_capture.json")
    keys, arr = [], {}
    for p in uc["passes"]:
        z = np.load(p["capture"])
        keys += list(z["keys"])
        for k in z.files:
            if k != "keys":
                arr.setdefault(k, []).append(z[k].astype(np.float32))
    arr = {k: np.concatenate(v) for k, v in arr.items()}
    bk, bA, _ = load_base_captures()
    idx = {k: i for i, k in enumerate(bk)}
    h = bA["resid_27"][[idx[k] for k in keys]]
    rms = np.sqrt((h ** 2).mean(1) + EPS)
    ro = readout_rows()
    v = ro["g"] * (ro["w_distress"] - ro["w_normal"])
    y = np.array([label(meta[k]) for k in keys])
    pool = np.array([meta[k]["pool"] for k in keys])
    dev, prac = pool == "dev", pool == "practice"
    res = {"heads": {}, "neurons": {}}
    for l in UNIT_LAYERS:
        Wo = ckpt_weight(f"model.layers.{l}.self_attn.o_proj.weight")   # 3584 x 3584 (bf16 checkpoint)
        Wd = ckpt_weight(f"model.layers.{l}.mlp.down_proj.weight")      # 3584 x 18944
        vo = v @ Wo          # per input dim: how much a unit of o_proj input moves s (before /rms)
        vd = v @ Wd
        xa = arr[f"attn_in_{l}"]
        xm = arr[f"mlp_in_{l}"]
        ca = (xa * vo) / rms[:, None]                 # per input dim contribution
        cm = (xm * vd) / rms[:, None]
        heads = ca.reshape(len(keys), 28, 128).sum(-1)
        for hh in range(28):
            c = heads[:, hh]
            res["heads"][f"{l}.{hh}"] = {"mean": float(c.mean()), "sd": float(c.std()),
                                         "diff": float(c[y == 1].mean() - c[y == 0].mean()),
                                         "auroc_prac": auroc(c[prac], y[prac]), "auroc_dev": auroc(c[dev], y[dev])}
        mean = cm.mean(0)
        diff = cm[y == 1].mean(0) - cm[y == 0].mean(0)
        sd = cm.std(0)
        order = np.argsort(mean)
        res["neurons"][str(l)] = {"most_normal": [[int(j), float(mean[j]), float(diff[j]), float(sd[j])] for j in order[:64]],
                                  "most_distress": [[int(j), float(mean[j]), float(diff[j]), float(sd[j])] for j in order[::-1][:16]],
                                  "sum_mean": float(mean.sum()), "check_vs_mlp_out": None}
        # consistency: neurons sum to the MLP's direct contribution
        res["neurons"][str(l)]["check_sum_minus_component"] = float(np.abs(cm.sum(1) - (bA[f"mlp_{l}"][[idx[k] for k in keys]] @ v) / rms).max())
        res["heads_check_" + str(l)] = float(np.abs(heads.sum(1) - (bA[f"attn_{l}"][[idx[k] for k in keys]] @ v) / rms).max())
    atomic_write_json(AN / "units.json", res)
    hs = sorted(res["heads"].items(), key=lambda kv: kv[1]["mean"])
    print("heads most toward Normal:")
    for k, x in hs[:12]:
        print(f"  {k:6s} mean {x['mean']:+.3f} sd {x['sd']:.3f} diff {x['diff']:+.3f} auroc prac {x['auroc_prac']:.3f} dev {x['auroc_dev']:.3f}")
    print("heads most toward Distress:")
    for k, x in hs[-8:]:
        print(f"  {k:6s} mean {x['mean']:+.3f} sd {x['sd']:.3f} diff {x['diff']:+.3f} auroc prac {x['auroc_prac']:.3f} dev {x['auroc_dev']:.3f}")
    print("checks:", {k: round(v_, 4) for k, v_ in res.items() if k.startswith("heads_check")},
          {l: round(res["neurons"][l]["check_sum_minus_component"], 4) for l in res["neurons"]})


def cmd_regress(a) -> None:
    """Pre-freeze regression check of a candidate (2 passes): Stage 2 free-form on
    the 30 seeded dev frames the final evaluation also uses, and the per-type
    probe on the 256-frame dev subset - compared with the stored BASE outputs of
    the same frames (eval_results/stage2_probe_raw_attain.json, production config)."""
    import random
    from scripts import wsp_client as C
    edits = [e for e in a.edits.split(",") if e]
    dev = sorted(W.pools()["dev"], key=lambda r: r["key"])
    dev30 = sorted(random.Random(20261106).sample(dev, 30), key=lambda r: r["key"])
    stored = {r["image"]: r for r in read_json(EVAL_RAW)["rows"]}
    out = read_json(AN / f"regress_{a.name}.json", {})
    if "stage2" not in out:
        r = C.run("stage2", [x["key"] for x in dev30], purpose=f"Regression check before freezing ({a.name}): "
                  "Stage 2 free-form output still parses on 30 dev frames", edits=edits)
        out["stage2"] = r["results"]
        atomic_write_json(AN / f"regress_{a.name}.json", out)
    if "probe" not in out:
        r = C.run("probe", dev256(), purpose=f"Regression check before freezing ({a.name}): per-type probe "
                  "P(yes) on the 256-frame dev subset barely moves", edits=edits)
        out["probe"] = r["results"]
        atomic_write_json(AN / f"regress_{a.name}.json", out)

    def ok(x):
        raw = (x.get("raw") or x.get("stage2_raw") or "").upper()
        t = x.get("distress_types") or []
        return all(f in raw for f in ("DISTRESS_TYPES", "SEVERITY", "DESCRIPTION")) and bool(t) and \
            not all(str(v).lower() in ("unknown", "other") for v in t)
    base_fail = sum(1 for x in dev30 if not ok(stored[x["key"]]["v0"]))
    ed_fail = sum(1 for x in out["stage2"] if not ok(x))
    same_types = sum(1 for x in out["stage2"] if (x.get("distress_types") or []) ==
                     (stored[x["key"]]["v0"].get("distress_types") or []))
    dp = {}
    for x in out["probe"]:
        bp = stored[x["key"]]["probe"]["min:name"]["p"]
        for t, p in x["p"].items():
            dp.setdefault(t, []).append(abs(p - bp[t]))
    summ = {"stage2_parse_failures": {"base_stored": base_fail, "edited": ed_fail, "n": len(dev30)},
            "stage2_same_type_list_as_base": same_types,
            "probe_mean_abs_dp": {t: float(np.mean(v)) for t, v in dp.items()},
            "probe_max_mean_abs_dp": float(max(np.mean(v) for v in dp.values()))}
    out["summary"] = summ
    atomic_write_json(AN / f"regress_{a.name}.json", out)
    print(json.dumps(summ, indent=1), flush=True)


EVAL_RAW = PROJECT_ROOT / "eval_results" / "stage2_probe_raw_attain.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    rg = sub.add_parser("regress")
    rg.add_argument("--name", required=True)
    rg.add_argument("--edits", required=True)
    sub.add_parser("diagnose")
    sub.add_parser("dla")
    d = sub.add_parser("devrun")
    d.add_argument("--name", required=True)
    d.add_argument("--edits", default="")
    d.add_argument("--purpose", required=True)
    d.add_argument("--dose-group", default=None)
    d.add_argument("--c", type=float, default=None)
    d.add_argument("--control", action="store_true")
    sub.add_parser("units-capture")
    sub.add_parser("units")
    a = ap.parse_args()
    AN.mkdir(parents=True, exist_ok=True)
    {"diagnose": cmd_diagnose, "dla": cmd_dla, "devrun": cmd_devrun,
     "units-capture": cmd_units_capture, "units": cmd_units, "regress": cmd_regress}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
