"""Expected VLM time per km under each gate, replayed on batch_001 (CPU only). Writes out/vlm_time.json.

Every segment of the send-all batch has its measured VLM wall time (responses.jsonl `seconds`) and answer
class. A gate's VLM time on a session = the sum of the measured times of the segments it would send:
pass (decision_if_gated = send) plus the audit draw (SHA-256 of the id < audit_frac 0.10). Gates:
default thresholds; the shipped calibration (in-sample, all 3 sessions); the leave-one-session-out
thresholds (held-out session only, out/loso.json). Usage: CUDA_VISIBLE_DEVICES=-1 python g05_vlm_time.py
"""
import os
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r"C:/GeoAI/tmp/gate_speed_before/pipeline")
from geoai.analyze import load_batch  # noqa: E402
from geoai.config import Config  # noqa: E402
from geoai.gate import audit_draw  # noqa: E402
from geoai.vlm import answer_class  # noqa: E402

BATCH = Path(r"C:/Users/PRO-LAB-3/Documents/GeoAI_Video/field/batches/batch_001")
HERE = Path(__file__).resolve().parent
loso = json.loads((HERE / "out" / "loso.json").read_text(encoding="utf-8"))
cfg = Config()
data = load_batch(BATCH, cfg)
secs = {}
for line in (BATCH / "responses.jsonl").read_text(encoding="utf-8").splitlines():
    x = json.loads(line)
    if x.get("http_status") == 200:
        secs[x["request_id"]] = float(x["seconds"])
rows = []
for r in data.rows:
    rec = data.records[r["id"]]
    cls, _ = answer_class(data.answers.results[r["id"]], set(cfg.vlm.labels))
    rows.append({"id": r["id"], "s": rec["session_id"][-4:], "len": rec["segment"]["length_m"], "cls": cls,
                 "sec": secs[r["id"]], "sc": rec["gate"]["scores"], "draw": audit_draw(r["id"])})


def stats(v):
    v = np.array(v)
    return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)), "p90": float(np.percentile(v, 90))}


res = {"seconds_per_image": {c: stats([x["sec"] for x in rows if x["cls"] == c]) for c in ("normal", "distress")},
       "seconds_per_image_all": stats([x["sec"] for x in rows]), "audit_frac": 0.10}


def sent(x, th):
    p = any(th.get(k) is not None and x["sc"][k] is not None and x["sc"][k] >= th[k] for k in ("probe_frac", "evidence", "crack_mm"))
    return p, p or x["draw"] < 0.10


default = {"probe_frac": 0.05, "evidence": 30.0, "crack_mm": 300.0}
full = loso["full_batch_calibration"]["V1_probe_frac"]["thresholds"]
gates = {"send_all": None, "default": {s: default for s in ("f428", "293d", "d47c")},
         "probe_calibrated_in_sample": {s: full for s in ("f428", "293d", "d47c")},
         "probe_loso_held_out": {s: loso["sessions"]["V1_probe_frac"][s]["thresholds"] for s in ("f428", "293d", "d47c")}}
out = {}
for g, ths in gates.items():
    per = {}
    for s in ("f428", "293d", "d47c", "all"):
        xs = [x for x in rows if s == "all" or x["s"] == s]
        km = math.fsum(x["len"] for x in xs) / 1000
        if ths is None:
            flags = [(True, True)] * len(xs)
        else:
            flags = [sent(x, ths[x["s"]]) for x in xs]
        n_pass = sum(f[0] for f in flags)
        n_sent = sum(f[1] for f in flags)
        t = math.fsum(x["sec"] for x, f in zip(xs, flags) if f[1])
        per[s] = {"km": km, "segments": len(xs), "pass": n_pass, "sent": n_sent, "sent_per_km": n_sent / km,
                  "vlm_s": t, "vlm_min_per_km": t / 60 / km,
                  "vlm_positive_sent": sum(1 for x, f in zip(xs, flags) if f[1] and x["cls"] == "distress"),
                  "vlm_positive": sum(1 for x in xs if x["cls"] == "distress")}
    out[g] = per
res["gates"] = out
for g in ("probe_calibrated_in_sample", "probe_loso_held_out"):
    res[f"saved_min_per_km_{g}_vs_default"] = out["default"]["all"]["vlm_min_per_km"] - out[g]["all"]["vlm_min_per_km"]
    res[f"saved_min_per_km_{g}_vs_send_all"] = out["send_all"]["all"]["vlm_min_per_km"] - out[g]["all"]["vlm_min_per_km"]
(HERE / "out" / "vlm_time.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if k != "gates"}, indent=1))
for g, per in out.items():
    print(g, {s: (round(p["sent_per_km"], 1), round(p["vlm_min_per_km"], 1), f'{p["vlm_positive_sent"]}/{p["vlm_positive"]}') for s, p in per.items()})
