"""Discriminator analysis 1: records + VLM answers (CPU, no torch).

Reads the send-all sidecars, gate-mode sidecars, run reports and batch_001/responses.jsonl (read-only),
writes C:/GeoAI/analysis/discriminator/out/tabular.json and joined.csv.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:/Users/PRO-LAB-3/Documents/GeoAI_Video/field")
OUT = Path(r"C:/GeoAI/analysis/discriminator/out")
OUT.mkdir(parents=True, exist_ok=True)
SESS = ["20261026T114332Z_SM-G991B-F0D1_f428", "20261026T114907Z_SM-G991B-F0D1_293d",
        "20261027T050257Z_SM-G991B-F0D1_d47c"]
rng = np.random.default_rng(0)

# ---- Qwen smart_resize (copied verbatim from geoai/qwen.py) ----
FACTOR, MIN_PIXELS, MAX_PIXELS = 28, 7168, 1_048_576


def smart_resize(height, width, factor=FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def load_recs(kind):
    rows = []
    for s in SESS:
        rep = json.loads((ROOT / kind / s / "report.json").read_text())
        for p in sorted((ROOT / kind / s / "sidecars").glob("*.json")):
            r = json.loads(p.read_text())
            rows.append(r)
    return rows


def flat(r):
    f = r["frame"] or {}
    q = r["quality"] or {}
    g = (r["gate"] or {})
    sc = g.get("scores") or {}
    c = r["capture"] or {}
    cam = r["camera"] or {}
    d = {"id": r["id"], "session": r["session_id"][-4:], "k": r["segment"]["k"], "status": r["status"],
         "decision_if_gated": g.get("decision_if_gated"), "reasons": ",".join(g.get("reasons") or []),
         "v_top": (f.get("rows_of_segment") or [None, None])[0], "v_bot": (f.get("rows_of_segment") or [None, None])[1],
         "near_edge": f.get("near_edge_ahead_m"), "far_edge": f.get("far_edge_ahead_m"),
         "speed": cam.get("speed_mps"), "alt_msl": cam.get("alt_msl_m"), "h_sigma": cam.get("h_sigma_m"),
         "exposure_ns": c.get("exposure_ns"), "iso": c.get("iso"), "focus_dpt": c.get("focus_diopters"),
         "focus_mode": c.get("focus_mode"), "low_light": c.get("low_light"),
         "chainage": r["segment"].get("chainage_start_km"), "survey_start_ch": (r.get("survey") or {}).get("start_chainage_km"),
         "flags": ",".join(r.get("flags") or [])}
    d.update({f"q_{k}": v for k, v in q.items()})
    d.update({f"s_{k}": v for k, v in sc.items()})
    return d


send = pd.DataFrame([flat(r) for r in load_recs("runs_segments_sendall")])
gate = pd.DataFrame([flat(r) for r in load_recs("runs_segments")])
near_m = {}
for s in SESS:
    rep = json.loads((ROOT / "runs_segments_sendall" / s / "report.json").read_text())
    near_m[s[-4:]] = rep["plan"]["near_m"]
send["near_m"] = send["session"].map(near_m)
send["win_pos"] = (send["near_edge"] - send["near_m"]) / 1.0  # 0 = nearest candidate position, 1 = farthest

# ---- VLM answers ----
resp = []
for line in (ROOT / "batches/batch_001/responses.jsonl").read_text(encoding="utf-8").splitlines():
    try:
        o = json.loads(line)
    except ValueError:
        continue
    if o.get("error") or not isinstance(o.get("response"), dict):
        continue
    x = o["response"]
    types = [t for t in (x.get("distress_types") or []) if t and t != "Normal"]
    sn = x.get("stage1_safety_net") or {}
    resp.append({"id": o["request_id"], "seconds": o["seconds"], "proc_ms": x.get("processing_time_ms"),
                 "s1_ms": x.get("stage1_time_ms"), "s2_ms": x.get("stage2_time_ms"), "sn_ms": sn.get("time_ms"),
                 "sn_flagged": sn.get("flagged"), "sn_score": sn.get("score"),
                 "s1_label": x.get("stage1_label"), "s1_conf": x.get("stage1_confidence"), "types": "|".join(types),
                 "vlm_distress": bool(types), "n_types": len(types),
                 "probe_time": ((x.get("stage2_probe") or {}).get("time_ms") if isinstance(x.get("stage2_probe"), dict) else None),
                 "needs_review": x.get("needs_expert_review")})
resp = pd.DataFrame(resp).drop_duplicates("id", keep="last")
J = send.merge(resp, on="id", how="inner")
J.to_csv(OUT / "joined.csv", index=False)
send.to_csv(OUT / "sendall_records.csv", index=False)
res = {"n_records_sendall": len(send), "n_answered": len(J), "answered_by_session": J["session"].value_counts().to_dict()}

# ---- AUROC with block bootstrap (blocks of 10 consecutive segments: neighbours are correlated) ----


def auroc(x, y):
    x = np.asarray(x, float); y = np.asarray(y, bool)
    ok = np.isfinite(x)
    x, y = x[ok], y[ok]
    P, N = y.sum(), (~y).sum()
    if P == 0 or N == 0:
        return float("nan")
    r = pd.Series(x).rank().to_numpy()
    return float((r[y].sum() - P * (P + 1) / 2) / (P * N))


def pass_at_recall(x, y, target=0.9):
    x = np.asarray(x, float); y = np.asarray(y, bool)
    xp = np.sort(x[y])[::-1]
    if xp.size == 0:
        return float("nan"), float("nan")
    k = math.ceil(target * xp.size)
    t = xp[k - 1]
    return float((x >= t).mean()), float(t)


def boot(fn, df, cols, B=2000, block=10):
    n = len(df)
    nb = math.ceil(n / block)
    starts = np.arange(0, n, block)
    out = []
    for _ in range(B):
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in rng.choice(starts, nb)])
        d = df.iloc[idx]
        out.append(fn(*[d[c].to_numpy() for c in cols]))
    out = np.asarray(out, float)
    return [float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))]


J = J.sort_values(["session", "k"]).reset_index(drop=True)
y = J["vlm_distress"].to_numpy()
res["vlm_positive"] = int(y.sum())
res["vlm_positive_rate"] = float(y.mean())
sig = ["s_probe_frac", "s_probe_max_logit", "s_evidence", "s_crack_mm", "s_z_max", "s_anomalous_patch_frac",
       "q_sharpness", "q_blur_px", "speed", "iso", "exposure_ns", "low_light", "win_pos"]
J["low_light"] = J["low_light"].astype(float)
au = {}
for s in sig:
    a = auroc(J[s], y)
    ci = boot(lambda x, yy: auroc(x, yy), J, [s, "vlm_distress"])
    pr, t = pass_at_recall(J[s].to_numpy(float), y)
    prci = boot(lambda x, yy: pass_at_recall(x.astype(float), yy.astype(bool))[0], J, [s, "vlm_distress"])
    au[s] = {"auroc": a, "auroc_ci95": ci, "pass_rate_at_recall90": pr, "pass_rate_ci95": prci, "threshold": t}
res["auroc"] = au
# Current default gate on the answered set
th = {"s_probe_frac": 0.05, "s_evidence": 30.0, "s_crack_mm": 300.0}
passes = np.zeros(len(J), bool)
for k, v in th.items():
    passes |= J[k].to_numpy(float) >= v
res["default_gate_on_answered"] = {"pass_rate": float(passes.mean()), "recall": float(passes[y].mean()) if y.any() else None}
# Stage-1 vs low light
res["distress_rate_by_low_light"] = J.groupby("low_light")["vlm_distress"].agg(["mean", "count"]).reset_index().to_dict("records")

# ---- timing ----
tm = {}
for lab, d in J.groupby("s1_label"):
    tm[lab] = {c: {"median": float(d[c].median()), "p10": float(d[c].quantile(.1)), "p90": float(d[c].quantile(.9)), "n": int(d[c].notna().sum())}
               for c in ["seconds", "proc_ms", "s1_ms", "s2_ms", "sn_ms"] if d[c].notna().any()}
tm["all_seconds"] = {"median": float(J["seconds"].median()), "mean": float(J["seconds"].mean()),
                     "p95": float(J["seconds"].quantile(.95))}
tm["share_stage1_normal"] = float((J["s1_label"] == "Normal").mean())
tm["sn_share_of_proc_normal"] = float((J.loc[J.s1_label == "Normal", "sn_ms"] / J.loc[J.s1_label == "Normal", "proc_ms"]).median())
tm["overhead_s_median"] = float((J["seconds"] - J["proc_ms"] / 1000).median())
tm["sn_flagged_share_of_normal"] = float(J.loc[J.s1_label == "Normal", "sn_flagged"].astype(float).mean())
res["timing"] = tm
res["label_counts"] = J["types"].value_counts().head(20).to_dict()
res["s1_label_counts"] = J["s1_label"].value_counts().to_dict()
res["s1_distress_but_no_types"] = int(((J.s1_label == "Distress") & (~J.vlm_distress)).sum())

# ---- best-frame choice ----
bf = {}
for s, d in send.groupby("session"):
    d = d[d.status != "not_covered"]
    bf[s] = {"n": len(d), "n_candidates_hist": d["q_n_candidates"].value_counts().sort_index().to_dict(),
             "rank_gt1": int((d["q_sharpness_rank"] > 1).sum()),
             "win_pos_median": float(d["win_pos"].median()), "win_pos_quartiles": d["win_pos"].quantile([.1, .25, .5, .75, .9]).round(3).tolist(),
             "share_win_pos_gt_0.5": float((d["win_pos"] > 0.5).mean()),
             "share_win_pos_gt_0.8": float((d["win_pos"] > 0.8).mean()),
             "share_win_pos_lt_0.2": float((d["win_pos"] < 0.2).mean()),
             "blur_px_p50_p90_max": d["q_blur_px"].quantile([.5, .9, 1]).round(2).tolist(),
             "over_blur_4px": int((d["q_blur_px"] > 4).sum()), "blur_gt_2px": int((d["q_blur_px"] > 2).sum()),
             "rot_share_gt_motion": float((d["q_blur_rotation_px"] > d["q_blur_motion_px"]).mean()),
             "low_light_chosen": float(d["low_light"].astype(float).mean()),
             "speed_p50": float(d["speed"].median()), "exposure_us_p50": float(d["exposure_ns"].median() / 1000),
             "exposure_us_p90": float(d["exposure_ns"].quantile(.9) / 1000),
             "iso_p50": float(d["iso"].median()), "iso_p90": float(d["iso"].quantile(.9)),
             "focus_dpt": d["focus_dpt"].round(3).value_counts().head(3).to_dict(), "focus_mode": d["focus_mode"].value_counts().to_dict(),
             "seg_rows_px_p50": float((d.v_bot.clip(upper=2159) - d.v_top.clip(lower=0)).median()),
             "chainage_null": int(d["chainage"].isna().sum()), "alt_msl_null": int(d["alt_msl"].isna().sum()),
             "h_sigma_p50": float(d["h_sigma"].median())}
res["best_frame"] = bf

# ---- Q1: VLM input geometry per frame ----
H, W = 2160, 3840
fh, fw = smart_resize(H, W)
res["full_frame_input"] = {"w": fw, "h": fh, "tokens": (fw // 28) * (fh // 28), "scale": fw / W}
g = send[send.status != "not_covered"].copy()
g["r0"] = g.v_top.clip(lower=0)
g["r1"] = g.v_bot.clip(upper=H)
g["seg_rows"] = g.r1 - g.r0
g["seg_share_rows"] = g.seg_rows / H
crops = []
for _, r in g.iterrows():
    for ctx_name, ctx_up, ctx_dn in (("rows_only", 0, 0), ("rows+ctx", 0.10, 0.0)):
        r0 = max(0, r.r0 - ctx_up * r.seg_rows)
        r1 = min(H, r.r1 + ctx_dn * r.seg_rows)
        hh = int(round(r1 - r0))
        oh, ow = smart_resize(hh, W)
        crops.append({"id": r.id, "variant": ctx_name, "crop_h": hh, "out_w": ow, "out_h": oh, "tokens": (ow // 28) * (oh // 28),
                      "scale": ow / W, "lin_gain": (ow / W) / (fw / W), "aspect": ow / oh})
    # same-scale crop (keep 0.35 scale): token saving at equal resolution
    oh = int(round((r.r1 - r.r0) * fw / W / 28)) * 28
    crops.append({"id": r.id, "variant": "same_scale_band", "crop_h": int(r.seg_rows), "out_w": fw, "out_h": oh,
                  "tokens": (fw // 28) * (oh // 28), "scale": fw / W, "lin_gain": 1.0, "aspect": fw / oh})
C = pd.DataFrame(crops)
res["vlm_view"] = {
    "segment_share_of_rows": g.seg_share_rows.quantile([.1, .5, .9]).round(3).tolist(),
    "segment_rows_px": g.seg_rows.quantile([.1, .5, .9]).round(0).tolist(),
    "crop_variants": {v: {"tokens_p50": float(d.tokens.median()), "lin_gain_p50": float(d.lin_gain.median()),
                          "lin_gain_p10_p90": d.lin_gain.quantile([.1, .9]).round(3).tolist(),
                          "aspect_p50": float(d.aspect.median()), "aspect_max": float(d.aspect.max()),
                          "out_p50": f"{int(d.out_w.median())}x{int(d.out_h.median())}"}
                      for v, d in C.groupby("variant")},
}
(OUT / "tabular.json").write_text(json.dumps(res, indent=1, default=float))
print(json.dumps(res, indent=1, default=float))
