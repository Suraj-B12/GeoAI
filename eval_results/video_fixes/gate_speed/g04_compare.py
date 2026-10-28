"""Compare the timed extraction runs (CPU only): outputs, decisions, identity, timings. Writes out/compare.json.

Runs (field/runs_eval/<variant>_rep<n>, made by g03_gpu_runs.sh under the GPU lock):
  before_rep1/2       git HEAD 3c72086, default config (bank + crack filter + uncalibrated thresholds)
  after_rep1/2        HEAD + this study's changes, --config calibration/segments_probe_gate.json (probe-only gate)
  afterdefault_rep1   HEAD + this study's changes, default config (must equal before byte for byte)
Usage: CUDA_VISIBLE_DEVICES=-1 python g04_compare.py
"""
import os
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
import json
import sys
from pathlib import Path

sys.path.insert(0, r"C:/GeoAI/tmp/gate_speed_after/pipeline")
import geoai  # noqa: E402
assert Path(geoai.__file__).as_posix().startswith("C:/GeoAI/tmp/gate_speed_after/")
from geoai.gate import audit_draw, read_calibration  # noqa: E402
from geoai.segment_records import check_segment_png, load_segment_records, validate_segment_record  # noqa: E402

EV = Path(r"C:/Users/PRO-LAB-3/Documents/GeoAI_Video/field/runs_eval")
CAL = Path(r"C:/GeoAI/tmp/gate_speed_after/pipeline/calibration/gate_20261028.json")
OUT = Path(__file__).resolve().parent / "out"
RUNS = ["before_rep1", "before_rep2", "after_rep1", "after_rep2", "afterdefault_rep1"]
SESS = sorted(p.name for p in (EV / "before_rep1").iterdir() if p.is_dir() and p.name != "logs")
T = read_calibration(CAL)["any_of"]["thresholds"]["probe_frac"]


def files(run: Path) -> dict:
    return {p.relative_to(run).as_posix(): p for p in run.rglob("*") if p.is_file()}


def same_run(a: Path, b: Path) -> dict:
    """Byte-identical run folders except report.json's created_utc and timings_s."""
    fa, fb = files(a), files(b)
    diff = sorted(set(fa) ^ set(fb))
    diff += [k for k in sorted(set(fa) & set(fb)) if k != "report.json" and fa[k].read_bytes() != fb[k].read_bytes()]
    ra, rb = (json.loads((x / "report.json").read_text(encoding="utf-8")) for x in (a, b))
    for r in (ra, rb):
        r.pop("created_utc"), r.pop("timings_s")
    rep_diff = sorted(k for k in set(ra) | set(rb) if ra.get(k) != rb.get(k))
    return {"files": len(fa), "differing_files": diff[:20], "n_differing_files": len(diff), "report_keys_differing": rep_diff}


res = {"probe_frac_threshold": T, "sessions": {}}
for s in SESS:
    short = s[-4:]
    R = {}
    for run in RUNS:
        d = EV / run / s
        rep = json.loads((d / "report.json").read_text(encoding="utf-8"))
        R[run] = {"dir": d, "rep": rep, "recs": {r["id"]: r for r in load_segment_records(d / "sidecars")}}
    out = {"planned_m": R["before_rep1"]["rep"]["coverage"]["planned_m"], "runs": {}}
    for run in RUNS:
        rep = R[run]["rep"]
        out["runs"][run] = {"records": rep["records"], "pass_rate": rep["gate"]["pass_rate"], "sent_rate": rep["gate"]["sent_rate"],
                            "thresholds": rep["gate"]["thresholds"], "bank": rep["bank"], "timings_s": rep["timings_s"],
                            "features_device": rep["features_device"], "dinov2_calls": rep["dinov2_calls"]}
    b, a = R["before_rep1"]["recs"], R["after_rep1"]["recs"]
    assert set(a) == set(b)

    def fkey(r):
        return None if r.get("frame") is None else (r["frame"]["segment"], r["frame"]["frame_in_segment"])
    frame_diff = {run: [i for i in sorted(b) if fkey(R[run]["recs"][i]) != fkey(b[i])] for run in RUNS[1:]}
    probe_diff = [i for i in sorted(b) if b[i]["gate"] and (a[i]["gate"]["scores"]["probe_frac"], a[i]["gate"]["scores"]["probe_max_logit"])
                  != (b[i]["gate"]["scores"]["probe_frac"], b[i]["gate"]["scores"]["probe_max_logit"])]
    # The probe-only decisions replayed from the BEFORE run's probe scores with the calibrated threshold.
    replay_diff = []
    for i, r in a.items():
        if r["status"] == "not_covered":
            continue
        pf = b[i]["gate"]["scores"]["probe_frac"]
        want = "send" if pf >= T else ("audit" if audit_draw(i) < 0.10 else "skip")
        if r["status"] != want:
            replay_diff.append(i)
    null_ok = all(r["gate"] is None or all(r["gate"]["scores"][k] is None for k in ("evidence", "crack_mm", "z_max", "anomalous_patch_frac"))
                  for r in a.values())
    invalid = [i for i, r in a.items() if validate_segment_record(r)]
    png_problems = {}
    for i, r in a.items():
        if r["status"] in ("send", "audit"):
            p = check_segment_png(R["after_rep1"]["dir"] / r["frame"]["file"], r, offset_minutes=330)
            if p:
                png_problems[i] = p
    out.update({"chosen_frame_differences_vs_before_rep1": frame_diff, "probe_score_differences_after_vs_before": probe_diff,
                "decision_replay_differences": replay_diff, "bank_scores_null_in_after": null_ok,
                "after_invalid_records": invalid, "after_png_problems": png_problems,
                "n_png_checked": sum(r["status"] in ("send", "audit") for r in a.values()),
                "identity": {"before_rep1_vs_rep2": same_run(R["before_rep1"]["dir"], R["before_rep2"]["dir"]),
                             "after_rep1_vs_rep2": same_run(R["after_rep1"]["dir"], R["after_rep2"]["dir"]),
                             "afterdefault_vs_before_rep1": same_run(R["afterdefault_rep1"]["dir"], R["before_rep1"]["dir"])}})
    res["sessions"][short] = out
    print(short, json.dumps({k: v for k, v in out.items() if k not in ("runs",)}, default=str)[:1500], flush=True)
(OUT / "compare.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
