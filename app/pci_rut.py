"""
Rut depth (IRC:82-2023 PCI weight 0.14) for a report section: a PRIOR interval, not a measurement.

Why only a prior: a forward-looking phone camera cannot measure a depth of a few mm across a wheel path,
and the IMU cannot either — the survey vehicle is a SCOOTER (owner, 2026-11-09): a single-track vehicle has
no left/right axle whose roll would encode a transverse depression, and its roll is dominated by steering /
lean. The per-type probe's 'Rutting' P(yes) (research.video_probe_scores) is stored for every video
segment but has never been checked against a measured rut (no Attain class, no expert label, no survey
yet), so it does not move the interval until eval_results/pci/rut/rut_calibration.json exists.

Sources and reasoning: eval_results/pci/rut/RUT_PRIOR.md.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CALIBRATION_PATH = PROJECT_ROOT / "eval_results" / "pci" / "rut" / "rut_calibration.json"
CALIBRATION_SCHEMA = "geoai.rut_calibration.v1"

# Literature range (RUT_PRIOR.md §2): every published Bengaluru value found lies in 0-23 mm (5 roads);
# IRC:37 designs bituminous pavements to a 20 mm rut at end of life. A judgement range, NOT a statistical
# 90 % interval (5 roads is far too few for that).
PRIOR_LO_MM = 0.0
PRIOR_HI_MM = 25.0
SCORE_FLAG = 0.5   # a section whose p90 Rutting P(yes) reaches this is flagged (not rated) for a look

PRIOR_BASIS = ("Prior range, not a measurement: published Bengaluru rut depths 0-23 mm on 5 urban roads "
               "(Anjana & Jamakhandi 2017, Silk Board junction, 2.5 m straight edge; Jagadeesh et al. 2013, "
               "Hosur Road 2.5-10.8 mm over 14 months) and the IRC:37 20 mm end-of-life design rut; "
               "widened to 25 mm to cover the largest single reading (23 mm). Not conditioned on the image: "
               "the 'Rutting' probe score is unvalidated for rut depth.")


def load_calibration(path: Optional[Path] = None) -> Optional[dict]:
    path = Path(path) if path else CALIBRATION_PATH
    if not path.exists():
        return None
    cal = json.loads(path.read_text(encoding="utf-8"))
    problems = validate_calibration(cal)
    if problems:
        raise ValueError(f"{path}: invalid rut calibration: {'; '.join(problems)}")
    return cal


def validate_calibration(cal: Mapping) -> list[str]:
    """Schema: {"schema", "stat": "p90"|"mean"|"max", "bins": [{"lo": score, "hi": score, "lo90": mm,
    "hi90": mm, "point": mm, "n": int}], "source": {...}, "probe_fp": str}. Bins must tile [0, 1]."""
    p = []
    if cal.get("schema") != CALIBRATION_SCHEMA:
        p.append(f"schema must be {CALIBRATION_SCHEMA}")
    if cal.get("stat") not in ("p90", "mean", "max"):
        p.append("stat must be p90, mean or max")
    bins = cal.get("bins")
    if not isinstance(bins, list) or not bins:
        p.append("bins missing")
    else:
        edge = 0.0
        for b in sorted(bins, key=lambda b: b.get("lo", -1)):
            if not all(isinstance(b.get(k), (int, float)) for k in ("lo", "hi", "lo90", "hi90", "point", "n")):
                p.append("each bin needs lo, hi, lo90, hi90, point, n")
                break
            if abs(b["lo"] - edge) > 1e-9 or not b["lo"] < b["hi"] or not 0 <= b["lo90"] <= b["point"] <= b["hi90"]:
                p.append("bins must tile [0, 1] with 0 <= lo90 <= point <= hi90")
                break
            edge = b["hi"]
        if not p and abs(edge - 1.0) > 1e-9:
            p.append("bins must end at 1")
    for k in ("source", "probe_fp"):
        if k not in cal:
            p.append(f"{k} missing")
    return p


def rut_prior(rutting_score_stats: Optional[Mapping] = None, calibration: Optional[Mapping] = None) -> dict:
    """Rut depth interval (mm) for one section.

    rutting_score_stats: summary of the section's per-segment 'Rutting' probe P(yes), e.g.
        {"n": 20, "mean": 0.07, "p90": 0.15, "max": 0.31} (None when no probe scores).
    calibration: a rut_calibration.json dict (fitted later on the survey's calibration half), or None.

    Returns {"point", "lo90", "hi90", "basis", "is_prior", "conditioned_on_score", "flag"}.
    Without a calibration the interval is the unconditional literature prior and point is None (the
    literature does not support a single typical value)."""
    stats = dict(rutting_score_stats or {})
    flag = None
    p90 = stats.get("p90")
    if isinstance(p90, (int, float)) and math.isfinite(p90) and p90 >= SCORE_FLAG:
        flag = (f"'Rutting' probe P(yes) p90 = {p90:.2f} >= {SCORE_FLAG}: check this section on site "
                "(probe unvalidated for rutting)")
    if calibration is not None:
        x = stats.get(calibration["stat"])
        if isinstance(x, (int, float)) and math.isfinite(x) and stats.get("n", 0) > 0:
            x = min(max(float(x), 0.0), 1.0)
            for b in sorted(calibration["bins"], key=lambda b: b["lo"]):
                if b["lo"] <= x < b["hi"] or (x == 1.0 and b["hi"] == 1.0):
                    return {"point": float(b["point"]), "lo90": float(b["lo90"]), "hi90": float(b["hi90"]),
                            "basis": f"rut_calibration.json bin {b['lo']}-{b['hi']} of {calibration['stat']} "
                                     f"(n = {b['n']}, source: {calibration['source']})",
                            "is_prior": False, "conditioned_on_score": True, "flag": flag}
    return {"point": None, "lo90": PRIOR_LO_MM, "hi90": PRIOR_HI_MM, "basis": PRIOR_BASIS, "is_prior": True,
            "conditioned_on_score": False, "flag": flag}


def score_stats(scores) -> dict:
    """{"n", "mean", "p90", "max"} of a section's per-segment Rutting P(yes) values (None/NaN dropped)."""
    xs = sorted(float(s) for s in scores if s is not None and math.isfinite(float(s)))
    if not xs:
        return {"n": 0, "mean": None, "p90": None, "max": None}
    k = 0.9 * (len(xs) - 1)
    lo, hi = math.floor(k), math.ceil(k)
    p90 = xs[lo] + (xs[hi] - xs[lo]) * (k - lo)
    return {"n": len(xs), "mean": sum(xs) / len(xs), "p90": p90, "max": xs[-1]}
