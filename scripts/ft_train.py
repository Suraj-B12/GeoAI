"""
QLoRA post-training of the per-type Yes/No probe (paper §6.11, PostTrainBench-style).

The model is trained on exactly the quantity production reads: for one photo and
one IRC:82 type, P(Yes) from the Yes/No logits after the probe's own system
prompt and question (scripts/stage2_probe.py, variant min:name, the production
variant). Each step runs the probe's shared-prefix forward WITH gradients:
system prompt + image once, then the image's question suffixes as one batch on
the expanded key/value cache, then binary cross-entropy on
z = logit(Yes) - logit(No) (Linear crack: max over the Longitudinal and
Transverse questions, as the evaluation scores it). At start-up the
differentiable forward is checked against the unmodified production
TypeProber.probe() on a real image, so training and evaluation measure the same
number.

Model: Qwen2.5-VL-7B-Instruct loaded ONLY through scripts/model_loader.load_vlm
(4-bit NF4, bf16 compute, production pixel cap). LoRA on the language model's
attention and MLP projections; vision tower frozen; never merged.

Crash-safe and detached-friendly: holds C:\\GeoAI\\gpu.lock for the whole run
(released in finally), checkpoints the adapter + optimizer + scheduler + data
position atomically every --ckpt-every optimizer steps, and resumes from the
last checkpoint when re-run with the same arguments. Every epoch's adapter is
kept (epoch_<n>/) so the epoch can be chosen on DEV afterwards.

Usage (one run directory per configuration):
    venv/Scripts/python.exe scripts/ft_train.py --run A_r16_lr1e-4_s1 \
        --manifest eval_results/finetune/manifest_armA.json --seed 1
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512,garbage_collection_threshold:0.9")

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import DynamicCache  # noqa: E402

from scripts.ft_common import (ART_DIR, GpuLock, atomic_write_json, read_json,  # noqa: E402
                               sha256_file, utc_now)
from scripts.stage2_probe import TypeProber, all_probe_types  # noqa: E402

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
VARIANT = ("min", "name")          # production probe variant (configs/stage2_probe.json)
PROD_MAX_PIXELS = 1048576
MIN_PIXELS = 256 * 28
LORA_TARGET = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


# ============================================================
# Differentiable shared-prefix probe forward
# ============================================================

class TrainableProbe:
    """TypeProber.probe() without inference_mode: returns z = logit(Yes) -
    logit(No) per requested question as a tensor that carries gradients.

    Mirrors TypeProber.probe step by step (same chat text, same prefix/suffix
    split at <|vision_end|>, same 3-D rope positions from get_rope_index, same
    float32 recomputation of the Yes/No head rows from the final hidden state).
    """

    def __init__(self, prober: TypeProber):
        self.p = prober
        self.key_index = {t.key: i for i, t in enumerate(prober.probe_types)}
        head = prober.model.get_output_embeddings()
        self._stash = {}
        self._hook = head.register_forward_pre_hook(self._keep)

    def _keep(self, _m, args):
        self._stash["h"] = args[0]

    def close(self):
        self._hook.remove()

    def z(self, image: Image.Image, keys: list[str]) -> torch.Tensor:
        """Packed form used for training: ONE sequence = prefix + every question
        suffix, with a block attention mask (each suffix sees the whole prefix
        and its own earlier tokens, never another suffix) and each suffix's 3-D
        rope positions restarting where the prefix ends. Mathematically the same
        as the shared-cache batch, but cache-free, so standard (non-reentrant)
        gradient checkpointing applies and activations stay small."""
        p = self.p
        model = p.model
        device = model.device
        qs = [p.questions[self.key_index[k]] for k in keys]
        texts = [p._chat_text(image, q) for q in qs]
        full = p.processor(text=[texts[0]], images=[image], padding=True, return_tensors="pt")
        ids0 = full["input_ids"][0]
        ve = (ids0 == p.vision_end_id).nonzero()
        if len(ve) != 1:
            raise RuntimeError(f"expected one <|vision_end|>, found {len(ve)}")
        P = int(ve[0, 0]) + 1
        marker = "<|vision_end|>"
        head0 = texts[0].split(marker, 1)[0]
        suffix_ids = []
        for t in texts:
            h, tail = t.split(marker, 1)
            if h != head0:
                raise RuntimeError("questions do not share the same prefix text")
            suffix_ids.append(p.tokenizer(tail, add_special_tokens=False)["input_ids"])
        if ids0[P:].tolist() != suffix_ids[0]:
            raise RuntimeError("suffix tokenization does not match the full encoding")
        full_dev = full.to(device)
        with torch.no_grad():
            pos_full, _ = p.rope_owner.get_rope_index(
                full_dev["input_ids"], full_dev.get("image_grid_thw"), None,
                attention_mask=full_dev["attention_mask"])
        base = int(pos_full[0, 0, P])
        L = P + sum(len(s) for s in suffix_ids)
        ids = torch.empty(L, dtype=torch.long)
        ids[:P] = ids0[:P]
        pos = torch.empty((3, 1, L), dtype=torch.long, device=device)
        pos[:, :, :P] = pos_full[:, :, :P]
        mask = torch.zeros((L, L), dtype=torch.bool)
        mask[:P, :P] = torch.tril(torch.ones((P, P), dtype=torch.bool))
        last = []
        o = P
        for s in suffix_ids:
            n = len(s)
            ids[o:o + n] = torch.tensor(s, dtype=torch.long)
            pos[:, :, o:o + n] = torch.arange(base, base + n, device=device)
            mask[o:o + n, :P] = True
            mask[o:o + n, o:o + n] = torch.tril(torch.ones((n, n), dtype=torch.bool))
            last.append(o + n - 1)
            o += n
        self._stash.clear()
        model(input_ids=ids.unsqueeze(0).to(device),
              attention_mask=mask.view(1, 1, L, L).to(device),
              pixel_values=full_dev["pixel_values"], image_grid_thw=full_dev["image_grid_thw"],
              position_ids=pos, use_cache=False,
              logits_to_keep=torch.tensor(last, device=device))
        h = self._stash.pop("h")[0].float()               # (N, hidden)
        zz = h @ p._head_w.to(h.device).T
        k = len(p.yes_ids)
        return torch.logsumexp(zz[:, :k], 1) - torch.logsumexp(zz[:, k:], 1)

    def z_cached(self, image: Image.Image, keys: list[str]) -> torch.Tensor:
        """The shared-cache form (identical to TypeProber.probe); kept as a
        second reference for the parity check."""
        p = self.p
        model = p.model
        device = model.device
        qs = [p.questions[self.key_index[k]] for k in keys]
        texts = [p._chat_text(image, q) for q in qs]
        full = p.processor(text=[texts[0]], images=[image], padding=True, return_tensors="pt")
        ids0 = full["input_ids"][0]
        ve = (ids0 == p.vision_end_id).nonzero()
        if len(ve) != 1:
            raise RuntimeError(f"expected one <|vision_end|>, found {len(ve)}")
        P = int(ve[0, 0]) + 1
        marker = "<|vision_end|>"
        head0 = texts[0].split(marker, 1)[0]
        suffix_ids = []
        for t in texts:
            h, tail = t.split(marker, 1)
            if h != head0:
                raise RuntimeError("questions do not share the same prefix text")
            suffix_ids.append(p.tokenizer(tail, add_special_tokens=False)["input_ids"])
        if ids0[P:].tolist() != suffix_ids[0]:
            raise RuntimeError("suffix tokenization does not match the full encoding")
        full_dev = full.to(device)
        with torch.no_grad():
            pos_full, _ = p.rope_owner.get_rope_index(
                full_dev["input_ids"], full_dev.get("image_grid_thw"), None,
                attention_mask=full_dev["attention_mask"])
        pos_prefix = pos_full[:, :, :P]
        base = int(pos_full[0, 0, P])
        pre = model(input_ids=full_dev["input_ids"][:, :P],
                    attention_mask=full_dev["attention_mask"][:, :P],
                    pixel_values=full_dev["pixel_values"],
                    image_grid_thw=full_dev["image_grid_thw"],
                    position_ids=pos_prefix, use_cache=True, logits_to_keep=1)
        prefix_kv = [(layer.keys, layer.values) for layer in pre.past_key_values.layers]
        del pre
        N = len(suffix_ids)
        S = max(len(s) for s in suffix_ids)
        ids = torch.full((N, S), p.pad_id, dtype=torch.long)
        smask = torch.zeros((N, S), dtype=torch.long)
        pos = torch.full((N, S), base, dtype=torch.long)
        for i, s in enumerate(suffix_ids):
            L = len(s)
            ids[i, S - L:] = torch.tensor(s, dtype=torch.long)
            smask[i, S - L:] = 1
            pos[i, S - L:] = torch.arange(base, base + L)
        cache = DynamicCache(ddp_cache_data=[
            (k.expand(N, -1, -1, -1), v.expand(N, -1, -1, -1)) for k, v in prefix_kv])
        attn = torch.cat([torch.ones((N, P), dtype=torch.long), smask], dim=1)
        self._stash.clear()
        model(input_ids=ids.to(device), attention_mask=attn.to(device),
              position_ids=pos.to(device).unsqueeze(0).expand(3, N, S),
              past_key_values=cache, cache_position=torch.arange(P, P + S, device=device),
              use_cache=True, logits_to_keep=1)
        h = self._stash.pop("h")[:, -1].float()          # (N, hidden)
        zz = h @ p._head_w.to(h.device).T                # (N, |yes|+|no|) float32
        k = len(p.yes_ids)
        return torch.logsumexp(zz[:, :k], 1) - torch.logsumexp(zz[:, k:], 1)


def resize_to_cap(img: Image.Image, max_pixels: int) -> Image.Image:
    w, h = img.size
    if w * h <= max_pixels:
        return img
    s = math.sqrt(max_pixels / (w * h))
    return img.resize((max(28, int(w * s)), max(28, int(h * s))), Image.BICUBIC)


# ============================================================
# Checkpoints (atomic directory swap)
# ============================================================

def save_ckpt(run_dir: Path, model, opt, sched, state: dict) -> None:
    tmp = run_dir / "ckpt.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    model.save_pretrained(tmp / "adapter")
    torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(),
                "rng_torch": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all()},
               tmp / "train_state.pt")
    atomic_write_json(tmp / "state.json", state)     # written last = checkpoint complete
    cur, old = run_dir / "ckpt", run_dir / "ckpt.old"
    shutil.rmtree(old, ignore_errors=True)
    if cur.exists():
        os.replace(cur, old)
    os.replace(tmp, cur)
    shutil.rmtree(old, ignore_errors=True)


def find_ckpt(run_dir: Path):
    for d in (run_dir / "ckpt", run_dir / "ckpt.old"):
        if (d / "state.json").exists() and (d / "adapter" / "adapter_config.json").exists():
            return d
    return None


# ============================================================
# Main
# ============================================================

def class_weights(manifest: dict, mode: str) -> dict:
    """Per-class (w_pos, w_neg). 'none' = 1/1; 'balanced' = n/(2*count), clipped."""
    out = {}
    for c, k in manifest["target_counts"].items():
        if mode == "none" or k["pos"] == 0 or k["neg"] == 0:
            out[c] = (1.0, 1.0)
        else:
            n = k["pos"] + k["neg"]
            out[c] = (min(5.0, max(0.2, n / (2 * k["pos"]))), min(5.0, max(0.2, n / (2 * k["neg"]))))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name (directory under C:/GeoAI/finetune/runs)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--accum", type=int, default=8, help="images per optimizer step")
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--class-weights", choices=["none", "balanced"], default="balanced")
    ap.add_argument("--train-max-pixels", type=int, default=409600,
                    help="images above this are downscaled for TRAINING only")
    ap.add_argument("--ckpt-every", type=int, default=25, help="optimizer steps")
    ap.add_argument("--max-steps", type=int, default=0, help="smoke test: stop after N images")
    ap.add_argument("--no-lock", action="store_true", help="debug only")
    a = ap.parse_args()

    run_dir = ART_DIR / "runs" / a.run
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(a.manifest)
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path
    manifest = read_json(manifest_path)
    cfg = {k: v for k, v in vars(a).items() if k not in ("no_lock", "max_steps")}
    cfg["manifest_sha256"] = sha256_file(manifest_path)
    cfg["code_sha256"] = {f: sha256_file(PROJECT_ROOT / f) for f in
                          ("scripts/ft_train.py", "scripts/ft_common.py", "scripts/stage2_probe.py")}
    prev = read_json(run_dir / "config.json")
    if prev and {k: prev[k] for k in prev if k != "code_sha256"} != \
            {k: cfg[k] for k in cfg if k != "code_sha256"}:
        sys.exit(f"ERROR: {run_dir} holds a run with a different configuration; use another --run")
    if prev and prev.get("code_sha256") != cfg["code_sha256"]:
        print("[warn] code changed since this run started (bug fix?) - recorded in config_history")
        hist = read_json(run_dir / "config_history.json", [])
        hist.append({"at": utc_now(), "code_sha256": cfg["code_sha256"]})
        atomic_write_json(run_dir / "config_history.json", hist)
    atomic_write_json(run_dir / "config.json", cfg)
    if (run_dir / "final" / "adapter_config.json").exists() and not a.max_steps:
        print(f"[done] {run_dir} already finished")
        return 0

    lock = GpuLock(f"train {a.run}") if not a.no_lock else None
    if lock:
        lock.__enter__()
    try:
        return _train(a, cfg, manifest, run_dir)
    finally:
        if lock:
            lock.__exit__(None, None, None)


def _train(a, cfg, manifest, run_dir: Path) -> int:
    from peft import LoraConfig, PeftModel, get_peft_model
    from scripts.model_loader import load_vlm

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    model, processor, info = load_vlm(MODEL, quantization_bits=4, min_pixels=MIN_PIXELS,
                                      max_pixels=PROD_MAX_PIXELS, run_self_test=True)
    for prm in model.parameters():
        prm.requires_grad_(False)

    ck = find_ckpt(run_dir)
    if ck is not None:
        model = PeftModel.from_pretrained(model, ck / "adapter", is_trainable=True)
        state = read_json(ck / "state.json")
        print(f"[resume] from {ck.name}: epoch {state['epoch']} image {state['pos']} "
              f"opt step {state['opt_step']}", flush=True)
    else:
        lcfg = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout, bias="none",
                          target_modules=LORA_TARGET, task_type="CAUSAL_LM")
        model = get_peft_model(model, lcfg)
        state = {"epoch": 0, "pos": 0, "opt_step": 0, "images_seen": 0}
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_vis = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "visual" in n)
    if n_vis:
        sys.exit(f"ERROR: {n_vis} trainable parameters in the vision tower")
    print(f"[lora] trainable parameters: {n_train:,}", flush=True)

    probe_types = all_probe_types(include_indicators=True)
    prober = TypeProber(model, processor, probe_types, *VARIANT)
    tp = TrainableProbe(prober)

    # --- parity: differentiable forward == production probe (no grad, eval mode)
    items = manifest["items"]
    model.eval()
    img0 = Image.open(items[0]["image_path"]).convert("RGB")
    keys_all = [t.key for t in probe_types]
    with torch.no_grad():
        z = tp.z(img0, keys_all)
    ref = prober.probe(img0)
    pz = torch.sigmoid(z).cpu().tolist()
    diffs = [abs(pz[i] - r["p_yes"]) for i, r in enumerate(ref)]
    parity = {"max_abs_diff": max(diffs), "mean_abs_diff": sum(diffs) / len(diffs)}
    print(f"[parity] trainable forward vs TypeProber.probe: {parity}", flush=True)
    if parity["max_abs_diff"] > 0.02:
        sys.exit("ERROR: differentiable forward disagrees with the production probe")
    img0.close()
    gc.collect()
    torch.cuda.empty_cache()

    weights = class_weights(manifest, a.class_weights)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay, betas=(0.9, 0.999))
    steps_per_epoch = math.ceil(len(items) / a.accum)
    total = steps_per_epoch * a.epochs
    warm = max(1, int(a.warmup * total))

    def lr_at(s):
        if s < warm:
            return (s + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    if ck is not None:
        ts = torch.load(ck / "train_state.pt", weights_only=False)
        opt.load_state_dict(ts["opt"])
        sched.load_state_dict(ts["sched"])
        torch.set_rng_state(ts["rng_torch"])
        torch.cuda.set_rng_state_all(ts["rng_cuda"])

    log_path = run_dir / "train_log.jsonl"
    model.train()
    t_start = time.time()
    acc_loss, acc_n, acc_imgs = 0.0, 0, 0
    opt.zero_grad(set_to_none=True)
    for epoch in range(state["epoch"], a.epochs):
        order = list(range(len(items)))
        random.Random(a.seed * 1000 + epoch).shuffle(order)
        start = state["pos"] if epoch == state["epoch"] else 0
        # resume lands on an optimizer-step boundary (checkpoints are taken there)
        for pos in range(start, len(order)):
            it = items[order[pos]]
            img = resize_to_cap(Image.open(it["image_path"]).convert("RGB"), a.train_max_pixels)
            keys = []
            for t in it["targets"]:
                for k in t["keys"]:
                    if k not in keys:
                        keys.append(k)
            t0 = time.time()
            z = tp.z(img, keys)
            terms = []
            for t in it["targets"]:
                zc = torch.stack([z[keys.index(k)] for k in t["keys"]]).max()
                wp, wn = weights.get(t["cls"], (1.0, 1.0))
                w = wp if t["y"] else wn
                terms.append(w * torch.nn.functional.binary_cross_entropy_with_logits(
                    zc, torch.tensor(float(t["y"]), device=zc.device)))
            loss = torch.stack(terms).mean()
            (loss / a.accum).backward()
            acc_loss += float(loss.detach())
            acc_n += 1
            acc_imgs += 1
            img.close()
            del z, loss, terms
            state["images_seen"] += 1
            last_of_epoch = pos == len(order) - 1
            if acc_imgs == a.accum or last_of_epoch:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                state["opt_step"] += 1
                acc_imgs = 0
                gc.collect()
                torch.cuda.empty_cache()
                rec = {"epoch": epoch, "pos": pos + 1, "opt_step": state["opt_step"],
                       "lr": sched.get_last_lr()[0], "loss": acc_loss / max(1, acc_n),
                       "s_per_img": (time.time() - t0),
                       "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
                       "elapsed_s": round(time.time() - t_start, 1), "at": utc_now()}
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                if state["opt_step"] % 5 == 0:
                    print(f"[train] ep {epoch} img {pos + 1}/{len(order)} step {state['opt_step']}/{total} "
                          f"loss {rec['loss']:.4f} lr {rec['lr']:.2e} peak {rec['peak_reserved_gb']} GB "
                          f"{rec['elapsed_s'] / max(1, state['images_seen'] - (0 if ck is None else 0)):.2f} s/img",
                          flush=True)
                acc_loss, acc_n = 0.0, 0
                state["epoch"], state["pos"] = epoch, pos + 1
                if last_of_epoch:
                    state["epoch"], state["pos"] = epoch + 1, 0
                if state["opt_step"] % a.ckpt_every == 0 or last_of_epoch:
                    save_ckpt(run_dir, model, opt, sched, dict(state, at=utc_now()))
                if last_of_epoch:
                    ed = run_dir / f"epoch_{epoch + 1}"
                    tmp = run_dir / f"epoch_{epoch + 1}.tmp"
                    shutil.rmtree(tmp, ignore_errors=True)
                    model.save_pretrained(tmp)
                    shutil.rmtree(ed, ignore_errors=True)
                    os.replace(tmp, ed)
                    print(f"[epoch] {epoch + 1} saved -> {ed}", flush=True)
            if a.max_steps and state["images_seen"] >= a.max_steps:
                print(f"[smoke] stopping after {a.max_steps} images; peak reserved "
                      f"{torch.cuda.max_memory_reserved() / 1e9:.2f} GB, "
                      f"{(time.time() - t_start) / a.max_steps:.2f} s/img", flush=True)
                tp.close()
                return 0
    tmp = run_dir / "final.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp)
    shutil.rmtree(run_dir / "final", ignore_errors=True)
    os.replace(tmp, run_dir / "final")
    atomic_write_json(run_dir / "train_summary.json", {
        "finished_at": utc_now(), "opt_steps": state["opt_step"], "images_seen": state["images_seen"],
        "seconds_this_session": round(time.time() - t_start, 1), "parity": parity,
        "trainable_params": n_train, "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
        "load_info": {k: info.get(k) for k in ("quantization_bits", "max_pixels", "attn_implementation")}})
    tp.close()
    print(f"[done] {run_dir / 'final'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
