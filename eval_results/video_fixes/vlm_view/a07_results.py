"""VLM-view A/B, step 7: results.json from the step outputs and the visual judgements (CPU only, no model)."""
import csv
import json
import math
from pathlib import Path

from judgements import J

OUT = Path(r"C:/GeoAI/analysis/fixes/vlm_view")


def wilson(k, n, z=1.959964):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


flips = list(csv.DictReader(open(OUT / "flips.csv", encoding="utf-8")))
rows = []
for f in flips:
    key = f["id"].split("F0D1_")[1]
    verdict, note = J[key]
    rows.append({**{k: f[k] for k in ("id", "stratum", "frame_rerun", "frame_labels", "band", "band_labels")},
                 "verdict": verdict, "note": note})
with open(OUT / "judgements.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)


def tally(sub):
    t = {v: sum(1 for r in sub if r["verdict"] == v) for v in ("band", "frame", "unclear")}
    dec = t["band"] + t["frame"]
    t["band_share_of_decided"] = round(t["band"] / dec, 4) if dec else None
    t["band_share_ci95"] = wilson(t["band"], dec)
    return t


judged = {"all_flips": tally(rows),
          "frame_distress_to_band_normal": tally([r for r in rows if r["frame_rerun"] == "distress"]),
          "frame_normal_to_band_distress": tally([r for r in rows if r["frame_rerun"] == "normal"]),
          "out_of_segment_cases": [r["id"] for r in rows if "BELOW the near edge" in r["note"] or "beyond the far edge" in r["note"]],
          "footpath_false_positives_of_band": [r["id"] for r in rows if "footpath" in r["note"] and r["verdict"] == "frame"],
          "method": "one look per flip at the whole frame (1280 px) with the segment's edges marked; not ground truth"}
res = {"band_vs_frame": json.loads((OUT / "band_vs_frame.json").read_text()), "flip_judgements": judged}
for name in ("nearest_geometry.json", "nearest_vs_sharpest.json"):
    if (OUT / name).exists():
        res[name[:-5]] = json.loads((OUT / name).read_text())
json.dump(res, open(OUT / "results.json", "w"), indent=1)
print(json.dumps(judged, indent=1))
