"""VLM-view A/B, step 7: results.json from the step outputs and the visual judgements (CPU only, no model)."""
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, r"C:/GeoAI/analysis/fixes")
import atomic_io as aio  # noqa: E402  atomic result writes
from judgements import J  # noqa: E402

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
aio.write_csv(OUT / "judgements.csv", rows)


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

# nearest_sharp vs sharpest: visual judgement of the flips + the rule written before any answer
# (nearest_decision_rule.json): default only if the Wilson 95 % CI of the nearest share of decided flips > 0.5.
if (OUT / "judgements_nearest.py").exists() and (OUT / "nearest_vs_sharpest.json").exists():
    from judgements_nearest import JN  # noqa: E402
    nflips = json.loads((OUT / "nearest_vs_sharpest.json").read_text())["flips"]
    missing = [f["id"] for f in nflips if f["id"].split("F0D1_")[1] not in JN]
    assert not missing, f"flips without a judgement: {missing}"
    nrows = [{**f, "verdict": JN[f["id"].split("F0D1_")[1]][0], "note": JN[f["id"].split("F0D1_")[1]][1]} for f in nflips]
    aio.write_csv(OUT / "judgements_nearest.csv", nrows) if nrows else None

    def _margin(k, n):
        """How many 'nearest' verdicts could turn into 'unclear' before the rule fails (or, if it fails, how
        many more 'nearest' it would need): the robustness of the outcome in flips."""
        if (wilson(k, n) or [0])[0] > 0.5:
            j = 0
            while k - j - 1 >= 0 and (wilson(k - j - 1, n - j - 1) or [0])[0] > 0.5:
                j += 1
            return {"rule_met": True, "nearest_verdicts_that_can_be_lost": j}
        j = 0
        while (wilson(k + j, n + j) or [0])[0] <= 0.5 and j < 1000:
            j += 1
        return {"rule_met": False, "more_nearest_verdicts_needed": j}

    def ntally(sub):
        t = {v: sum(1 for r in sub if r["verdict"] == v) for v in ("nearest", "sharpest", "unclear")}
        dec = t["nearest"] + t["sharpest"]
        t["nearest_share_of_decided"] = round(t["nearest"] / dec, 4) if dec else None
        t["nearest_share_ci95"] = wilson(t["nearest"], dec)
        return t

    allt = ntally(nrows)
    ci = allt["nearest_share_ci95"]
    rule = json.loads((OUT / "nearest_decision_rule.json").read_text())
    res["nearest_flip_judgements"] = {
        "all_flips": allt,
        "sharpest_distress_to_nearest_normal": ntally([r for r in nrows if r["sharpest"] == "distress"]),
        "sharpest_normal_to_nearest_distress": ntally([r for r in nrows if r["sharpest"] == "normal"]),
        "method": ("one look per flip at both whole frames (900 px) with the segment's edges marked; not ground "
                   "truth; not blind (each arm and its answer named on the sheet)"),
        "rule": rule["rule"], "rule_written_utc": rule["written_utc"],
        # The rule is a necessary condition ("default only if ..."), not an automatic switch.
        "rule_met": bool(ci is not None and ci[0] > 0.5),
        "margin_flips": None if ci is None else _margin(allt["nearest"], allt["nearest"] + allt["sharpest"])}
aio.write_json(OUT / "results.json", res, indent=1)
print(json.dumps(judged, indent=1))
