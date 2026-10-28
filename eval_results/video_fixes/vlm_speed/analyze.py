"""Identity check + timing statistics + figure for the profile=video A/B and the runner end-to-end runs.

Inputs (this folder): ab_raw.jsonl (ab_measure.py), runner_before/ and runner_after/ (run_vlm_batch.py
outputs + runner_timing.json), hash_time.json. Output: results.json, fig_vlm_speed.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261028)
B = 10000
# Everything in the /classify reply that decides labels or review. Timings, and the record-only
# views inside stage1_safety_net, are excluded on purpose (they are what the profile changes).
DECISION_FIELDS = ["is_distressed", "stage1_label", "stage1_confidence", "stage1_raw", "distress_types",
                   "primary_distress_type", "secondary_distress_types", "severity", "description", "stage2_raw",
                   "stage2_confidence", "stage2_confidence_primary", "stage2_confidence_probe", "stage2_mode",
                   "condition_indicators", "needs_expert_review"]
SN_FIELDS = ["flagged", "score", "strongest_type", "threshold", "p", "error"]


def decision(resp: dict) -> dict:
    d = {k: resp.get(k) for k in DECISION_FIELDS}
    sn = resp.get("stage1_safety_net")
    d["stage1_safety_net"] = None if sn is None else {k: sn.get(k) for k in SN_FIELDS}
    return d


def boot_median_ci(x: np.ndarray) -> list[float]:
    idx = RNG.integers(0, len(x), size=(B, len(x)))
    m = np.median(x[idx], axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def stats(x) -> dict:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {"n": 0}
    return {"n": int(x.size), "median": float(np.median(x)), "median_ci95": boot_median_ci(x),
            "p90": float(np.percentile(x, 90)), "mean": float(x.mean()), "min": float(x.min()), "max": float(x.max())}


def paired(a, b) -> dict:
    """Median of per-frame differences a-b and ratio b/a, bootstrap over frames."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d, r = a - b, b / a
    return {"n": int(a.size), "median_saving_s": float(np.median(d)), "median_saving_ci95": boot_median_ci(d),
            "median_ratio_video_over_default": float(np.median(r)), "ratio_ci95": boot_median_ci(r)}


def ab() -> dict:
    lines = [json.loads(x) for x in (HERE / "ab_raw.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    by = {}
    for ln in lines:
        if ln["http_status"] == 200:
            by[(ln["request_id"], ln["profile"])] = ln   # last success wins (resumes)
    ids = sorted({rid for rid, _ in by if (rid, "default") in by and (rid, "video") in by})
    diffs, same_as_batch001 = [], 0
    hist = {}
    for x in (Path(r"C:\Users\PRO-LAB-3\Documents\GeoAI_Video\field\batches\batch_001") / "responses.jsonl").read_text(
            encoding="utf-8").splitlines():
        r = json.loads(x)
        hist[r["request_id"]] = r["response"]
    rows = []
    for rid in ids:
        dflt, vid = by[(rid, "default")], by[(rid, "video")]
        dd, dv = decision(dflt["response"]), decision(vid["response"])
        if dd != dv:
            diffs.append({"request_id": rid, "fields": [k for k in dd if dd[k] != dv[k]],
                          "default": {k: dd[k] for k in dd if dd[k] != dv[k]}, "video": {k: dv[k] for k in dd if dd[k] != dv[k]}})
        same_as_batch001 += decision(hist[rid]) == dd
        rv = vid["response"]
        sn_d = dflt["response"].get("stage1_safety_net") or {}
        rows.append({"request_id": rid, "session": dflt["session"], "answer": dflt["response"]["stage1_label"],
                     "default_s": dflt["seconds"], "video_s": vid["seconds"],
                     "default_server_ms": dflt["response"]["processing_time_ms"], "video_server_ms": rv["processing_time_ms"],
                     "stage1_ms": dflt["response"]["stage1_time_ms"],
                     "default_tiles_ms": ((sn_d.get("views") or {}).get("recorded") or {}).get("time_ms"),
                     "video_profile_echo": rv.get("profile"), "video_views_recorded": rv.get("views_recorded"),
                     "default_has_profile_key": "profile" in dflt["response"],
                     "default_first": dflt["position"] == 0})
    out = {"pairs": len(ids), "identical_pairs": len(ids) - len(diffs), "differences": diffs,
           "default_vs_batch001_identical": same_as_batch001,
           "video_echo_ok": all(r["video_profile_echo"] == "video" and r["video_views_recorded"] is False for r in rows),
           "default_reply_has_no_profile_key": not any(r["default_has_profile_key"] for r in rows),
           "by_answer": {}}
    for ans in ("Normal", "Distress", "all"):
        sel = [r for r in rows if ans == "all" or r["answer"] == ans]
        if not sel:
            continue
        out["by_answer"][ans] = {
            "default_wall_s": stats([r["default_s"] for r in sel]), "video_wall_s": stats([r["video_s"] for r in sel]),
            "default_server_s": stats([r["default_server_ms"] / 1000 for r in sel]),
            "video_server_s": stats([r["video_server_ms"] / 1000 for r in sel]),
            "paired_wall": paired([r["default_s"] for r in sel], [r["video_s"] for r in sel]),
            "stage1_s": stats([r["stage1_ms"] / 1000 for r in sel]),
        }
        tiles = [r["default_tiles_ms"] / 1000 for r in sel if r["default_tiles_ms"] is not None]
        if tiles:
            out["by_answer"][ans]["default_recorded_tiles_s_on_normal"] = stats(tiles)
    # order effect: does going first cost more (warm-up)?
    out["order_check"] = {
        "default_first_median_s": float(np.median([r["default_s"] for r in rows if r["default_first"]])) if rows else None,
        "default_second_median_s": float(np.median([r["default_s"] for r in rows if not r["default_first"]])) if rows else None}
    # Mix-weighted projection at batch_001's answer mix (649 Normal / 164 Distress of 813).
    bn, bd = out["by_answer"].get("Normal"), out["by_answer"].get("Distress")
    if bn and bd:
        wN, wD = 649 / 813, 164 / 813
        out["batch_mix_projection_s_per_image"] = {
            "default_mean": wN * bn["default_wall_s"]["mean"] + wD * bd["default_wall_s"]["mean"],
            "video_mean": wN * bn["video_wall_s"]["mean"] + wD * bd["video_wall_s"]["mean"],
            "note": "server wall time per request only; the runner adds its pacing (6.5 s floor before, 0 on loopback after)"}
    out["rows"] = rows
    return out


def runner() -> dict:
    res = {}
    for name in ("runner_before", "runner_after"):
        d = HERE / name
        t = d / "runner_timing.json"
        if not t.exists():
            continue
        tm = json.loads(t.read_text(encoding="utf-8"))
        lines = [json.loads(x) for x in (d / "responses.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        ok = [ln for ln in lines if not ln.get("error")]
        res[name] = {**tm, "answered": len(ok), "errors": len(lines) - len(ok),
                     "s_per_image_end_to_end": tm["wall_s"] / max(1, len(ok)),
                     "request_seconds": stats([ln["seconds"] for ln in ok]),
                     "run_log": json.loads((d / "vlm_run.json").read_text(encoding="utf-8"))["runs"][-1] | {"health": "omitted"}}
    if len(res) == 2:
        b = {json.loads(x)["request_id"]: decision(json.loads(x)["response"]) for x in
             (HERE / "runner_before" / "responses.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()}
        a = {json.loads(x)["request_id"]: decision(json.loads(x)["response"]) for x in
             (HERE / "runner_after" / "responses.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()}
        common = set(a) & set(b)
        res["identical_answers_before_vs_after"] = sum(a[k] == b[k] for k in common)
        res["compared"] = len(common)
        secs = [json.loads(x)["seconds"] for x in (HERE / "runner_after" / "responses.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        res["projection_video_with_6_5s_pacing_s_per_image"] = sum(max(s, 6.5) for s in secs) / len(secs)
        res["speedup_end_to_end"] = res["runner_before"]["s_per_image_end_to_end"] / res["runner_after"]["s_per_image_end_to_end"]
    return res


def figure(r: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C_BEFORE, C_AFTER = "#2a78d6", "#eb6834"   # categorical slots 1 and 2 (validated reference palette)
    INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": [2, 1.15]})
    ax = axes[0]
    rows = r["ab"]["rows"]
    groups = [("Normal", "Stage 1 Normal"), ("Distress", "Stage 1 Distress")]
    for gi, (ans, label) in enumerate(groups):
        sel = [x for x in rows if x["answer"] == ans]
        for off, key, col, nm in ((-0.17, "default_s", C_BEFORE, "Default request (records tiles)"),
                                  (0.17, "video_s", C_AFTER, "profile=video (no tile recording)")):
            v = np.array([x[key] for x in sel])
            jit = RNG.uniform(-0.06, 0.06, v.size)
            ax.scatter(gi + off + jit, v, s=9, color=col, alpha=0.35, linewidths=0, zorder=2)
            st = r["ab"]["by_answer"][ans]["default_wall_s" if key == "default_s" else "video_wall_s"]
            lo, hi = st["median_ci95"]
            ax.errorbar(gi + off, st["median"], yerr=[[st["median"] - lo], [hi - st["median"]]], fmt="o", ms=6,
                        color=col, ecolor=col, elinewidth=1.6, capsize=3, zorder=3, label=nm if gi == 0 else None)
            ax.annotate(f"{st['median']:.1f}", (gi + off, st["median"]), xytext=(9, -3), textcoords="offset points",
                        fontsize=8, color=INK)
        ax.text(gi, -0.16, f"n = {len(sel)}", transform=ax.get_xaxis_transform(), ha="center", fontsize=8, color=MUTED)
    ax.set_xticks([0, 1], [g[1] for g in groups])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, None)
    ax.set_ylabel("Wall time per /classify request (s)")
    ax.set_title("a  60 frames, each sent both ways (dot: median, bar: 95% CI)", loc="left", fontsize=9, color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax2 = axes[1]
    rr = r["runner"]
    if "runner_before" in rr and "runner_after" in rr:
        vals = [rr["runner_before"]["s_per_image_end_to_end"], rr["runner_after"]["s_per_image_end_to_end"]]
        ax2.bar([0, 1], vals, width=0.55, color=[C_BEFORE, C_AFTER], zorder=2)
        for i, v in enumerate(vals):
            ax2.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color=INK)
        ax2.set_xticks([0, 1], ["Before\n(default, 6.5 s pacing)", "After\n(video, no pacing)"], fontsize=7.5)
    ax2.set_ylabel("End-to-end runner time (s per image)")
    ax2.set_title("b  Runner, 30 frames (24 N + 6 D)", loc="left", fontsize=9, color=INK)
    ax2.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax2.set_ylim(0, None)
    for a in axes:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / "fig_vlm_speed.png", dpi=200)


def main() -> None:
    r = {"ab": ab(), "runner": runner()}
    ht = HERE / "hash_time.json"
    if ht.exists():
        r["batch_check"] = json.loads(ht.read_text(encoding="utf-8"))
    (HERE / "results.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
    figure(r)
    ab_ = r["ab"]
    print("pairs", ab_["pairs"], "identical", ab_["identical_pairs"], "vs batch_001", ab_["default_vs_batch001_identical"])
    for k, v in ab_["by_answer"].items():
        print(k, {kk: (round(vv["median"], 2), [round(c, 2) for c in vv["median_ci95"]], round(vv["p90"], 2), vv["n"])
                  for kk, vv in v.items() if isinstance(vv, dict) and "median" in vv}, v["paired_wall"])
    print(json.dumps({k: v for k, v in r["runner"].items() if not isinstance(v, dict)}, indent=1))


if __name__ == "__main__":
    main()
