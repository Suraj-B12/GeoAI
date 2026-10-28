"""Assemble results.json from out/loso.json, out/sharpness_identity.json, out/compare.json, out/vlm_time.json."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
O = HERE / "out"
loso = json.loads((O / "loso.json").read_text(encoding="utf-8"))
sharp = json.loads((O / "sharpness_identity.json").read_text(encoding="utf-8"))
cmp_ = json.loads((O / "compare.json").read_text(encoding="utf-8"))
vt = json.loads((O / "vlm_time.json").read_text(encoding="utf-8"))


def per_km(var):
    out = {}
    for s in ("f428", "293d", "d47c", "all"):
        segs = ["f428", "293d", "d47c"] if s == "all" else [s]
        km = sum(cmp_["sessions"][x]["planned_m"] for x in segs) / 1000
        reps = []
        for rep in (1, 2):
            tot = {}
            for x in segs:
                for k, v in cmp_["sessions"][x]["runs"][f"{var}_rep{rep}"]["timings_s"].items():
                    tot[k] = tot.get(k, 0.0) + v
            reps.append(tot)
        out[s] = {"km": km, "total_s": [r["total"] for r in reps], "total_s_per_km": [r["total"] / km for r in reps],
                  "stages_s_mean": {k: (reps[0][k] + reps[1][k]) / 2 for k in reps[0]}}
    return out


timing = {v: per_km(v) for v in ("before", "after")}
timing["speedup_all"] = [b / a for b, a in zip(timing["before"]["all"]["total_s"], timing["after"]["all"]["total_s"])]
res = {
    "schema": "gate_speed.results.v1", "date": "2026-10-28",
    "code": {"before": "GeoAI_Video git HEAD 3c72086 (C:/GeoAI/tmp/gate_speed_before)",
             "after": "HEAD + this study's changes only (C:/GeoAI/tmp/gate_speed_after); the same changes are in the repo working tree"},
    "preregistration_sha256": hashlib.sha256((HERE / "PREREGISTRATION.md").read_bytes()).hexdigest(),
    "gate_validation": {k: loso[k] for k in ("target_recall", "B", "block", "pooled", "sessions", "decision", "auroc",
                                               "full_batch_calibration")},
    "vlm_time": vt,
    "sharpness_luma": {"adopted": False, "rule": "adopt only if the chosen frame is identical in every segment of the 3 sessions",
                       "sessions": {k: {x: v[x] for x in ("segments", "n_differences", "chosen_frame_differences",
                                                          "old_vs_existing_sendall_run_differences", "spearman_old_new",
                                                          "sharpness_rel_diff", "ms_per_call")} for k, v in sharp.items()}},
    "extraction": {"per_session": {s: {k: v for k, v in d.items()} for s, d in cmp_["sessions"].items()},
                   "per_km": timing, "probe_frac_threshold": cmp_["probe_frac_threshold"]},
}
(HERE / "results.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
print(json.dumps(timing, indent=1)[:3000])
