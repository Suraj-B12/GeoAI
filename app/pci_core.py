"""
IRC:82-2015 distress-based condition rating — the pure arithmetic of the A->B report (no I/O).

"IRC:82-2015 distress-based condition rating — screening estimate from automated image classification".
Never "PCI" alone, never ASTM. Spec: eval_results/pci/DESIGN.md (v2) §2-§5; tables and clause references:
eval_results/pci/RESEARCH.md §1.2-1.5, §6.

    TABLES / WEIGHTS / PARAMETERS     Tables 5.1 (highway), 5.2 (MDR/rural), 5.3 (urban), App. 4 weights
    band(table, param, x)             Poor / Fair / Good with tolerance 1e-9 at the limits
    rating_value(table, param, x)     App-4-inferred interpolation, truncated floor(10x+1e-9)/10, clamped to band
    weighted(w, r), mean1(values)     Decimal half-up to 1 decimal (App. 4 prints 0.75*1.5 = 1.125 -> 1.1)
    final_class(v)                    <= 1.0 Poor, 1.1-2.0 Fair, >= 2.1 Good
    unit_quantities(segments, ...)    lengths, coverage, E_lo / E_hi / uncorrected per parameter, runs
    rate_unit(q, table)               per-parameter bands + bracketed App-4 final
    treatments(...)                   candidate treatment text with clause references

A "segment" here is a dict with at least: length_m, status (send/audit/skip/not_covered), vlm_class
(distress/normal/unknown/pending/not_sent/not_covered), final_labels, final_severity, condition_indicators,
and for runs: run_key (pass id) and k. These are the GIS properties of app/gis_routes._segment_feature.
"""
from __future__ import annotations

import json
import math
import os
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Iterable, Mapping, Optional

from app import pci_irc2023 as P23

EPS = 1e-9
SKIP_DISTRESS_UPPER = 0.133    # audited skips distressed 8.1 % [4.8, 13.3] (paper §8.7): upper CI
SKIP_DISTRESS_POINT = 0.081
SMALL_POTHOLE_AREA_M2 = 0.0314  # pi * 0.1^2: IRC §7.5.3.4 "small" pothole, 200 mm wide (a severity class)
DEFAULT_LANE_WIDTH_M = 3.5
RATED_COVERAGE = 0.80          # our choice (RESEARCH §5.1.7), not IRC
LOW_COVERAGE = 0.50

NAME = ("IRC:82-2015 distress-based condition rating — screening estimate from automated image "
        "classification")

# ------------------------------------------------------------------ tables
# (good_below L, poor_above U): Good x < L, Fair L <= x <= U, Poor x > U. Units % except rut depth (mm).
TABLES: dict[str, dict] = {
    "urban": {
        "table": "5.3", "title": "Table 5.3 — Pavement Distress Based Rating for Urban Roads (p. 5)",
        "params": {"cracking": (5.0, 15.0), "ravelling": (5.0, 10.0), "potholes": (0.0, 0.5),
                   "settlement": (1.0, 5.0), "rut_depth": (5.0, 10.0)},
    },
    "highway": {
        "table": "5.1", "title": "Table 5.1 — Pavement Distress Based Rating for Highways (p. 4)",
        "params": {"cracking": (5.0, 10.0), "ravelling": (1.0, 10.0), "potholes": (0.1, 1.0),
                   "shoving": (0.1, 1.0), "patching": (1.0, 10.0), "settlement": (1.0, 5.0),
                   "rut_depth": (5.0, 10.0)},
    },
    "rural": {
        "table": "5.2", "title": "Table 5.2 — Pavement Distress Based Rating for MDR(s) and Rural Roads (p. 5)",
        "params": {"cracking": (10.0, 20.0), "ravelling": (10.0, 20.0), "potholes": (0.5, 1.0),
                   "patching": (5.0, 20.0), "settlement": (2.0, 5.0)},
    },
}
TABLES["mdr"] = dict(TABLES["rural"])
CATEGORIES = ("urban", "highway", "mdr", "rural")

# Appendix 4, p. 73 (fixed multiplier factors)
WEIGHTS = {"cracking": 1.00, "ravelling": 0.75, "potholes": 0.50, "shoving": 1.00, "patching": 0.75,
           "settlement": 0.75, "rut_depth": 1.00}

PARAM_NAMES = {"cracking": "Cracking", "ravelling": "Ravelling", "potholes": "Potholes", "shoving": "Shoving",
               "patching": "Patching", "settlement": "Settlement and Depression", "rut_depth": "Rut depth"}
PARAM_UNITS = {"rut_depth": "mm"}

# ------------------------------------------------------------------ label -> parameter (DESIGN §3)
CRACK_ALLIGATOR = ("Alligator Cracking",)
CRACK_LINEAR = ("Longitudinal Cracking", "Transverse Cracking", "Edge Cracking", "Hairline Cracks")
PARAM_LABELS: dict[str, tuple] = {
    "cracking": CRACK_ALLIGATOR + CRACK_LINEAR,
    "potholes": ("Potholes",),
    "ravelling": ("Ravelling",),
    "settlement": ("Settlement", "Shallow Depression"),   # Upheaval is not in the taxonomy
    "shoving": ("Shoving",),
    "patching": (),          # condition indicator "Patching", not a distress label
    "rut_depth": (),         # needs a 3 m straight edge
}
SUB_PARAMS = {"cracking_alligator": CRACK_ALLIGATOR, "cracking_linear": CRACK_LINEAR}
# Proforma-1 columns reported but never rated (labels unvalidated)
REPORTED_ONLY = ("Bleeding", "Edge Breaking", "Stripping", "Hungry Surface", "Streaking", "Slippage",
                 "Corrugation", "Rutting")

ASSESSMENT: dict[str, dict] = {
    "cracking": {"assessed": True, "strength": "weak",
                 "evidence": "linear crack MCC 0.257 (P 0.89 / R 0.82); alligator recall 0.08 (Attain test)"},
    "potholes": {"assessed": True, "strength": "weak", "evidence": "MCC 0.219, P 0.43, R 0.25 (Attain test)"},
    "ravelling": {"assessed": False, "evidence": "recall 0 on Attain; 0 labels on video"},
    "settlement": {"assessed": False, "evidence": "never evaluated; 0 labels on video"},
    "shoving": {"assessed": False, "evidence": "never evaluated; 0 labels on video"},
    "patching": {"assessed": False, "evidence": "0 Patching indicators stored on video"},
    "rut_depth": {"assessed": False, "evidence": "needs a 3 m straight edge (not measurable from images)"},
}


def is_assessed(param: str) -> bool:
    return bool(ASSESSMENT.get(param, {}).get("assessed"))


# ================================================================== IRC:82-2023 PCI (0-100)
# Pothole occurrences in gate-skipped (never classified) segments: 9 of 161 audited skipped segments held a
# pothole, 0.056 [0.030, 0.103] (Wilson 95 %), eval_results/pci/validation.md.
POTHOLE_SKIP_POINT = 9 / 161
POTHOLE_SKIP_UPPER = 0.1028
SEGMENT_M = 5.0

# Probe questions (scripts/stage2_probe.py) that stand for each 2023 parameter; a segment is positive when the
# largest P(yes) over its keys reaches the calibrated threshold.
PROBE_KEYS: dict[str, tuple] = {
    "cracking": ("Alligator Cracking", "Longitudinal Cracking", "Transverse Cracking", "Edge Cracking",
                 "Hairline Cracks", "Block Cracking"),
    "potholes": ("Potholes",), "ravelling": ("Ravelling",), "patching": ("Patching",), "rut_depth": ("Rutting",),
}
DEFAULT_CALIBRATION = Path(__file__).resolve().parent.parent / "eval_results" / "pci" / "param_calibration.json"
_DEFAULT = object()
_cal_cache: dict = {"sig": None, "cal": None}
SOURCE_NAMES = {"entered": "entered by requester (unverified)", "probe": "per-type probe, calibrated",
                "labels": "Stage 2 labels", "labels_calibrated": "Stage 2 labels, scale calibrated",
                "roughness": "phone accelerometer, calibrated", "not_assessed": "not assessed"}


def validate_calibration(cal: Mapping) -> dict:
    """Checked copy of the parameter calibration (eval_results/pci/param_calibration.json, written only by
    scripts/pci_survey_compare.py from the survey's CALIBRATION half). Raises ValueError when malformed.

    {"schema": 2, "edition": "IRC:82-2023", "probe_fp": "<16 hex>", "fitted_on": {...},
     "params": {param: {"source": "probe", "keys": [...], "threshold": t, "a": a, "b": b, "evidence": "..."}
                       | {"source": "labels", "b": scale, "evidence": "..."}}}
    probe: value = max(0, a + b * f), f = fraction of the probe-scored length whose largest P(yes) over keys is
    >= threshold (b = 100 x area factor for % parameters; per-section count factor for potholes; mm for rut
    depth). labels: the label quantity (extent %, pothole runs) times b. IRI is calibrated separately
    (eval_results/pci/iri/iri_calibration.json) and arrives on the segments as `iri` (m/km)."""
    if not isinstance(cal, Mapping) or cal.get("schema") != 2 or cal.get("edition") != P23.EDITION:
        raise ValueError("calibration: schema 2 / edition IRC:82-2023 required")
    fp = cal.get("probe_fp")
    if not (isinstance(fp, str) and len(fp) == 16 and all(c in "0123456789abcdef" for c in fp)):
        raise ValueError("calibration: probe_fp must be 16 hex characters")
    out = {"schema": 2, "edition": P23.EDITION, "probe_fp": fp, "fitted_on": dict(cal.get("fitted_on") or {}),
           "params": {}}
    for p, spec in (cal.get("params") or {}).items():
        if p not in PROBE_KEYS:
            raise ValueError(f"calibration: parameter {p!r} cannot be calibrated here")
        if not isinstance(spec, Mapping):
            raise ValueError(f"calibration: {p}: entry must be an object")
        src = spec.get("source")
        b = float(spec.get("b", 1.0))
        if not math.isfinite(b) or b < 0:
            raise ValueError(f"calibration: {p}: b must be finite and >= 0")
        ev = str(spec.get("evidence") or "calibrated on survey sections")
        if src == "labels":
            if p not in ("cracking", "potholes"):
                raise ValueError(f"calibration: {p} has no Stage 2 label quantity")
            out["params"][p] = {"source": "labels", "b": b, "evidence": ev}
            continue
        if src != "probe":
            raise ValueError(f"calibration: {p}: source must be 'probe' or 'labels'")
        keys = list(spec.get("keys") or PROBE_KEYS[p])
        if not set(keys) <= set(PROBE_KEYS[p]):
            raise ValueError(f"calibration: {p}: keys must be among {PROBE_KEYS[p]}")
        t, a = float(spec["threshold"]), float(spec.get("a", 0.0))
        if not (0.0 < t < 1.0) or not math.isfinite(a):
            raise ValueError(f"calibration: {p}: threshold in (0,1) and finite a required")
        out["params"][p] = {"source": "probe", "keys": keys, "threshold": t, "a": a, "b": b, "evidence": ev}
    return out


def current_calibration() -> Optional[dict]:
    """The calibration file named by PCI_PARAM_CALIBRATION (default eval_results/pci/param_calibration.json),
    re-read when it changes; None when absent. A malformed file raises: a report never silently changes method."""
    path = Path(os.environ.get("PCI_PARAM_CALIBRATION") or DEFAULT_CALIBRATION)
    try:
        st = path.stat()
    except OSError:
        _cal_cache.update(sig=None, cal=None)
        return None
    sig = (str(path), st.st_mtime_ns, st.st_size)
    if _cal_cache["sig"] != sig:
        _cal_cache.update(sig=sig, cal=validate_calibration(json.loads(path.read_text(encoding="utf-8"))))
    return _cal_cache["cal"]


def calibration_info() -> Optional[dict]:
    cal = current_calibration()
    if cal is None:
        return None
    return {"fitted_on": cal["fitted_on"], "params": {p: {k: v for k, v in s.items() if k != "keys"}
                                                     for p, s in cal["params"].items()}}


def probe_value(segments: Iterable[Mapping], spec: Mapping, imaged_m: float, per_km: Optional[float] = None) -> dict:
    """Probe-based value of one parameter over a unit's imaged segments (see validate_calibration). per_km: the
    unit length in km for a COUNT parameter (potholes: b is potholes per km at f = 1)."""
    scored = pos = 0.0
    keys, t = spec["keys"], spec["threshold"]
    for s in segments:
        sc = s.get("probe")
        if s.get("status") == "not_covered" or not isinstance(sc, Mapping):
            continue
        L = _len(s)
        scored += L
        if max(float(sc.get(k) or 0.0) for k in keys) >= t - EPS:
            pos += L
    frac = pos / scored if scored > 0 else None
    ok = frac is not None and imaged_m > 0 and scored / imaged_m >= LOW_COVERAGE - EPS
    scale = 1.0 if per_km is None else per_km
    return {"value": max(0.0, spec["a"] + spec["b"] * frac * scale) if ok else None, "fraction": frac,
            "scored_m": scored, "positive_m": pos}


_iri_cal_cache: dict = {"sig": None, "cal": None}


def iri_calibration() -> Optional[dict]:
    """eval_results/pci/iri/iri_calibration.json via app.pci_iri (re-read when it changes); None when absent."""
    from app import pci_iri
    path = Path(os.environ.get("PCI_IRI_CALIBRATION") or pci_iri.CALIBRATION_PATH)
    try:
        st = path.stat()
    except OSError:
        _iri_cal_cache.update(sig=None, cal=None)
        return None
    sig = (str(path), st.st_mtime_ns, st.st_size)
    if _iri_cal_cache["sig"] != sig:
        _iri_cal_cache.update(sig=sig, cal=pci_iri.load_calibration(path))
    return _iri_cal_cache["cal"]


def roughness_value(segments: Iterable[Mapping], imaged_m: float, cal=_DEFAULT) -> Optional[dict]:
    """Calibrated IRI (m/km) of a unit from the phone-IMU features on its segments (`roughness`, table
    research.segment_roughness): segments are grouped into 100 m bins by measure (the calibration's unit),
    (bins of the calibration's unit_m) each bin -> app.pci_iri.aggregate() with the IRI calibration, and the calibrated bins are averaged by
    length. None without a calibration, or when calibrated bins cover < 50 % of the imaged length."""
    cal = iri_calibration() if cal is _DEFAULT else cal
    if cal is None:
        return None
    from app import pci_iri
    bins: dict[int, list] = {}
    for s in segments:
        r = s.get("roughness")
        if s.get("status") == "not_covered" or not isinstance(r, Mapping):
            continue
        L = _len(s)
        row = dict(r, s0_m=0.0, s1_m=L)
        unit = float(cal.get("unit_m") or 100.0)
        bins.setdefault(int(math.floor(float(s.get("measure_m") or 0.0) / unit)), []).append(row)
    tot = acc = 0.0
    statuses: dict[str, int] = {}
    for rows in bins.values():
        agg = pci_iri.aggregate(rows, calibration=cal)
        statuses[agg["iri_status"]] = statuses.get(agg["iri_status"], 0) + 1
        if agg["iri_status"] == "calibrated":
            tot += agg["length_m"]
            acc += agg["length_m"] * agg["iri"]
    if tot <= 0 or imaged_m <= 0 or tot / imaged_m < LOW_COVERAGE - EPS:
        return None
    return {"value": acc / tot, "measured_m": tot, "bins": statuses}


def band23(road_class: str, param: str, x: float, surface: Optional[str] = None) -> Optional[str]:
    """Tables 5.1-5.3 range of one value: 'Excellent to Good' / 'Satisfactory to Fair' / 'Poor to Fail'."""
    if param == "iri":
        r = P23.IRI_RANGES.get(road_class, {}).get(surface or P23.DEFAULT_SURFACE[road_class])
        if r is None:
            return None
        a, b = r
        return "Excellent to Good" if x < a - EPS else ("Poor to Fail" if x > b + EPS else "Satisfactory to Fair")
    a, b = P23.RANGES[road_class][param]
    if param == "potholes":
        if x <= a + EPS:
            return "Excellent to Good"
        return "Poor to Fail" if x > b + EPS else "Satisfactory to Fair"
    return "Excellent to Good" if x < a - EPS else ("Poor to Fail" if x > b + EPS else "Satisfactory to Fair")


def pci_unit(q: Mapping, road_class: str, surface: Optional[str] = None, entered: Optional[Mapping] = None,
             calibration=_DEFAULT) -> dict:
    """IRC:82-2023 PCI of one unit (100 m section, km row or stretch).

    Per parameter the value comes from, in order: entered by the requester; calibrated probe scores; calibrated
    accelerometer roughness (IRI); Stage 2 labels (cracking extent %, pothole number = label runs in the unit);
    else not assessed. Label values carry the gate-skip range (lo = fewer distress = better, hi = skipped
    segments distressed at the audited upper bound, point = audited point rate). Unknown parameters are
    bracketed PCI_i = 0 / 100. pci.single = sum w_i PCI_i(point) when every parameter has a value, else None."""
    cal = current_calibration() if calibration is _DEFAULT else calibration
    cparams = (cal or {}).get("params") or {}
    out = {"edition": P23.EDITION, "road_class": road_class,
           "surface": surface or P23.DEFAULT_SURFACE[road_class],
           "rated": q["coverage_status"] != "insufficient", "coverage_status": q["coverage_status"], "params": []}
    if not out["rated"]:
        out["pci"] = None
        return out
    lo_hi, point, missing = {}, {}, []
    skip_segs = q["skip_m"] / SEGMENT_M
    for p in P23.PARAMS:
        row = {"param": p, "name": P23.PARAM_NAMES[p], "unit": P23.PARAM_UNITS[p], "weight": P23.WEIGHTS[p]}
        spec = cparams.get(p)
        v_lo = v_hi = v_pt = None
        if entered and entered.get(p) is not None:
            src, ev = "entered", "entered by requester for the whole stretch (unverified)"
            v_lo = v_hi = v_pt = float(entered[p])
            if p == "potholes":          # entered as potholes per km; a unit's number scales with its length
                v_lo = v_hi = v_pt = float(entered[p]) * q["length_m"] / 1000.0
                ev = "entered by requester as potholes per km (unverified), scaled to this unit's length"
        elif spec and spec["source"] == "probe" and (q.get("probe") or {}).get(p, {}).get("value") is not None:
            pv = q["probe"][p]
            src, ev = "probe", spec["evidence"]
            v_lo = v_hi = v_pt = pv["value"]
            row.update(probe_fraction=pv["fraction"], probe_threshold=spec["threshold"])
        elif p == "iri" and q.get("roughness"):
            src, ev = "roughness", "phone accelerometer, calibrated on survey sections (iri_calibration.json)"
            v_lo = v_hi = v_pt = q["roughness"]["value"]
        elif p in ("cracking", "potholes") and not (spec and spec["source"] == "probe"):
            b = spec["b"] if spec else 1.0
            src = "labels_calibrated" if spec else "labels"
            if p == "cracking":
                e = q["extent"]["cracking"]
                v_lo, v_hi, v_pt = b * e["lo"], b * e["hi"], b * e.get("point", e["lo"])
                ev = ASSESSMENT["cracking"]["evidence"] + "; extent = % of surveyed LENGTH (upper bound of % area)"
                if not spec:
                    # % of length only bounds the IRC % of AREA from above: the best case is no cracked area
                    v_lo = 0.0
                    ev += "; best case taken as 0 % area until calibrated on a survey"
            else:
                n = q["runs"]["potholes"]
                v_lo, v_hi = b * n, b * (n + POTHOLE_SKIP_UPPER * skip_segs)
                v_pt = b * (n + POTHOLE_SKIP_POINT * skip_segs)
                ev = (ASSESSMENT["potholes"]["evidence"] + "; number = runs of consecutive pothole-labelled 5 m "
                      "segments in this unit (one run counted as one pothole)")
            if spec:
                ev += f"; scale {b:g} from {spec['evidence']}"
        else:
            src = "not_assessed"
            if spec and spec["source"] == "probe":
                pv = (q.get("probe") or {}).get(p) or {}
                ev = ("probe calibrated, but probe scores cover less than 50 % of this unit's imaged length "
                      f"({pv.get('scored_m', 0.0):.0f} m scored)")
            else:
                ev = {"ravelling": "probe not calibrated on Bengaluru (Attain thresholds put it on 98 % of photos)",
                      "patching": "probe not calibrated on Bengaluru",
                      "rut_depth": "needs a straight edge or laser profiler; not measurable from one forward camera",
                      "iri": "accelerometer roughness not yet calibrated against a reference IRI"}.get(p, "not assessed")
        row.update(source=src, source_name=SOURCE_NAMES[src], evidence=ev)
        if src == "not_assessed":
            row.update(known=False)
            missing.append(p)
        else:
            try:
                pw, pb = P23.pci_param_bounds(road_class, p, v_lo, v_hi, surface)
                pp = P23.pci_param(road_class, p, v_pt, surface)
            except P23.Unavailable as e:
                row.update(known=False, value_lo=v_lo, value_hi=v_hi, value_point=v_pt, evidence=f"{ev}; {e}")
                missing.append(p)
                out["params"].append(row)
                continue
            row.update(known=True, value_lo=v_lo, value_hi=v_hi, value_point=v_pt, pci_best=max(pb, pw),
                       pci_worst=min(pb, pw), pci_point=pp, range=band23(road_class, p, v_pt, surface))
            lo_hi[p] = (v_lo, v_hi)
            point[p] = v_pt
        out["params"].append(row)
    rng = P23.pci_range(road_class, {p: lo_hi.get(p) for p in P23.PARAMS if p not in missing}, surface)
    lo, hi = rng["lo"], rng["hi"]
    single = None
    off_scale = [r["name"] for r in out["params"] if r.get("known") and r["source"] == "labels"]
    if not missing and not off_scale:
        single = sum(P23.WEIGHTS[p] * r["pci_point"] for p, r in zip(P23.PARAMS, out["params"]))
    single_note = None
    if missing:
        single_note = "no single PCI: " + ", ".join(P23.PARAM_NAMES[p] for p in missing) + " not assessed"
    elif off_scale:
        single_note = ("no single PCI: " + ", ".join(off_scale) + " from uncalibrated Stage 2 labels (% of length, "
                       "pothole runs), which are not on the IRC % area / pothole-number scale until calibrated")
    known_w = sum(P23.WEIGHTS[p] for p in P23.PARAMS if p not in missing)
    out["pci"] = {
        "lo": lo, "hi": hi, "class_lo": P23.pci_class(lo), "class_hi": P23.pci_class(hi),
        "class": (P23.pci_class(lo) if P23.pci_class(lo) == P23.pci_class(hi)
                  else f"{P23.pci_class(lo)}–{P23.pci_class(hi)}"),
        "single": single, "single_class": P23.pci_class(single) if single is not None else None,
        "single_note": single_note,
        "recommendation": P23.RECOMMENDATION[P23.pci_class(single)] if single is not None else None,
        "missing": [P23.PARAM_NAMES[p] for p in missing], "known_weight": known_w,
        "assessed_part": (sum(P23.WEIGHTS[p] * r["pci_point"] for p, r in zip(P23.PARAMS, out["params"])
                              if r.get("known")) / known_w) if known_w > 0 else None,
    }
    return out


# ------------------------------------------------------------------ arithmetic
def d1(x: float) -> float:
    """Round half-up to 1 decimal (Decimal of the float's shortest repr, so 1.25 -> 1.3, 2.35 -> 2.4)."""
    return float(Decimal(repr(float(x))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def trunc1(x: float) -> float:
    """Truncate to 1 decimal, float-safe: floor(10x + 1e-9)/10 (2 - 4/5 = 1.1999.. -> 1.2)."""
    return math.floor(10.0 * x + EPS) / 10.0


def band(category: str, param: str, x: float) -> str:
    """'Poor' / 'Fair' / 'Good' for value x (%, or mm for rut depth) on the category's table."""
    L, U = TABLES[category]["params"][param]
    if category == "urban" and param == "potholes":       # NIL (0) / >0 and <=0.5 / >0.5
        if x <= EPS:
            return "Good"
        return "Poor" if x > U + EPS else "Fair"
    if x > U + EPS:
        return "Poor"
    if x < L - EPS:
        return "Good"
    return "Fair"


def rating_value(category: str, param: str, x: float) -> float:
    """App-4-inferred rating (RESEARCH §1.5): Poor 1.0; Fair 2 - (x-L)/(U-L); Good 3 - x/L; urban
    pothole Good 3.0, Fair 2 - x/0.5. Truncated to 1 decimal and clamped to its band's range."""
    L, U = TABLES[category]["params"][param]
    b = band(category, param, x)
    if b == "Poor":
        return 1.0
    if category == "urban" and param == "potholes":
        if b == "Good":
            return 3.0
        return min(2.0, max(1.1, trunc1(2.0 - x / U)))
    if b == "Fair":
        return min(2.0, max(1.1, trunc1(2.0 - (x - L) / (U - L))))
    return min(3.0, max(2.1, trunc1(3.0 - x / L)))


def _dec(x: float) -> Decimal:
    return Decimal(repr(float(x)))


def weighted(w: float, r: float) -> float:
    """w*r in exact decimal arithmetic, rounded half-up to 1 decimal: App. 4 prints 0.75*1.5 = 1.125 as 1.1
    and 0.75*1.2 = 0.9 (pinned by the App-4 worked-example test)."""
    return float((_dec(w) * _dec(r)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def mean1(values: Iterable[float]) -> float:
    vals = [_dec(v) for v in values]
    return float((sum(vals) / Decimal(len(vals))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def final_class(v: float) -> str:
    """App. 4 / Tables 5.x bands for the final value (rounded to 1 decimal first)."""
    v = float(_dec(v).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
    if v <= 1.0 + EPS:
        return "Poor"
    if v <= 2.0 + EPS:
        return "Fair"
    return "Good"


CLASS_ORDER = ("Poor", "Fair", "Good")


def class_range(a: str, b: str) -> str:
    """'Fair' when equal, else 'Poor–Fair' (worse first)."""
    if a == b:
        return a
    lo, hi = sorted((a, b), key=CLASS_ORDER.index)
    return f"{lo}–{hi}"


def app4_final(category: str, values: Mapping[str, float]) -> dict:
    """The App-4 computation over EVERY parameter of the table, from input values (% or mm).
    Returns rows + final value + class. Used for the worked-example test and by rate_unit()."""
    rows = []
    for p in TABLES[category]["params"]:
        x = values[p]
        r = rating_value(category, p, x)
        rows.append({"param": p, "input": x, "band": band(category, p, x), "rating": r, "weight": WEIGHTS[p],
                     "weighted": weighted(WEIGHTS[p], r)})
    f = mean1(r["weighted"] for r in rows)
    return {"rows": rows, "final": f, "class": final_class(f)}


# ------------------------------------------------------------------ quantities
def _labels(s: Mapping) -> set:
    return {str(x) for x in (s.get("final_labels") or [])}


def _is_classified(s: Mapping) -> bool:
    return s.get("status") in ("send", "audit") and s.get("vlm_class") in ("distress", "normal")


def _has(s: Mapping, labels: Iterable[str]) -> bool:
    return s.get("vlm_class") == "distress" and bool(_labels(s) & set(labels))


def _len(s: Mapping) -> float:
    try:
        v = float(s.get("length_m") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return v if math.isfinite(v) and v > 0 else 0.0


def runs(segments: list[Mapping], labels: Iterable[str]) -> list[dict]:
    """Runs of consecutive positive segments of ONE pass (run_key) and consecutive k: one occurrence each.
    Any non-positive segment (negative, skipped, unanswered, not covered) or a k jump ends a run."""
    labels = tuple(labels)
    out, cur = [], None
    for s in sorted(segments, key=lambda s: (str(s.get("run_key")), int(s.get("k") or 0))):
        pos = _is_classified(s) and _has(s, labels)
        key, k = str(s.get("run_key")), int(s.get("k") or 0)
        if pos and cur is not None and cur["run_key"] == key and k == cur["k_last"] + 1:
            cur["k_last"] = k
            cur["n"] += 1
            cur["length_m"] += _len(s)
            cur["segment_ids"].append(s.get("segment_id"))
            m = s.get("measure_m")
            if m is not None and (cur["measure_m"] is None or m < cur["measure_m"]):
                cur["measure_m"] = m
            continue
        if cur is not None:
            out.append(cur)
            cur = None
        if pos:
            cur = {"run_key": key, "k_first": k, "k_last": k, "n": 1, "length_m": _len(s),
                   "segment_ids": [s.get("segment_id")], "measure_m": s.get("measure_m")}
    if cur is not None:
        out.append(cur)
    return out


def run_length_hist(rs: list[dict]) -> dict:
    """{segments_in_run: count} (str keys, JSON-ready)."""
    h: dict[str, int] = {}
    for r in rs:
        h[str(r["n"])] = h.get(str(r["n"]), 0) + 1
    return dict(sorted(h.items(), key=lambda kv: int(kv[0])))


def unit_quantities(segments: list[Mapping], length_m: float, *, gap_m: float = 0.0,
                    lane_width_m: float = DEFAULT_LANE_WIDTH_M, runs_in_unit: Optional[Mapping[str, list]] = None,
                    pothole_runs_touching: Optional[int] = None, calibration=_DEFAULT) -> dict:
    """Summed quantities of one unit (100 m section, km row or whole stretch) from its chosen segments.

    length_m: the unit's length along the path (measure). gap_m: unsurveyed path length (gap edges).
    runs_in_unit: {"potholes": [...], "cracking": [...]} runs already assigned to this unit (a run that crosses
    a unit boundary counts once, in the unit of its first metre); default: runs() over `segments`."""
    q = {"length_m": float(length_m), "imaged_m": 0.0, "classified_m": 0.0, "skip_m": 0.0, "not_covered_m": 0.0,
         "unanswered_m": 0.0, "gap_m": float(gap_m), "segments": len(segments), "distress_segments": 0,
         "expert_reviewed": 0, "needs_expert_review": 0}
    affected = {p: 0.0 for p in PARAM_LABELS}
    affected.update({p: 0.0 for p in SUB_PARAMS})
    reported = {lab: 0.0 for lab in REPORTED_ONLY}
    labelled_unassessed = {p: 0 for p in PARAM_LABELS}
    worst_sev: dict[str, str] = {}
    sev_rank = {"Low": 1, "Medium": 2, "High": 3}
    for s in segments:
        L = _len(s)
        st = s.get("status")
        if st == "not_covered":
            q["not_covered_m"] += L
            continue
        q["imaged_m"] += L
        if st == "skip":
            q["skip_m"] += L
            continue
        if not _is_classified(s):
            q["unanswered_m"] += L
            continue
        q["classified_m"] += L
        q["expert_reviewed"] += int(bool(s.get("expert_reviewed")))
        q["needs_expert_review"] += int(bool(s.get("needs_expert_review")))
        if s.get("vlm_class") != "distress":
            continue
        q["distress_segments"] += 1
        labs = _labels(s)
        for p, pl in PARAM_LABELS.items():
            if labs & set(pl):
                affected[p] += L
                if not is_assessed(p):
                    labelled_unassessed[p] += 1
        for p, pl in SUB_PARAMS.items():
            if labs & set(pl):
                affected[p] += L
        for lab in REPORTED_ONLY:
            if lab in labs:
                reported[lab] += L
        sev = s.get("final_severity")
        if sev in sev_rank:
            for lab in labs:
                if sev_rank[sev] > sev_rank.get(worst_sev.get(lab, ""), 0):
                    worst_sev[lab] = sev
    D = q["classified_m"] + q["skip_m"]
    q["denominator_m"] = D
    q["coverage"] = min(1.0, q["imaged_m"] / length_m) if length_m > 0 else 0.0
    if D <= 0 or q["classified_m"] <= 0 or q["coverage"] < LOW_COVERAGE - EPS:
        q["coverage_status"] = "insufficient"
    elif q["coverage"] < RATED_COVERAGE - EPS:
        q["coverage_status"] = "low_coverage"
    else:
        q["coverage_status"] = "rated"
    q["rated_80"] = q["coverage_status"] == "rated"
    q["rated_50"] = q["coverage_status"] in ("rated", "low_coverage")

    def est(a):
        if D <= 0:
            return None
        return {"lo": 100.0 * a / D, "hi": 100.0 * (a + SKIP_DISTRESS_UPPER * q["skip_m"]) / D,
                "point": 100.0 * (a + SKIP_DISTRESS_POINT * q["skip_m"]) / D,
                "uncorrected": (100.0 * a / q["classified_m"]) if q["classified_m"] > 0 else None,
                "affected_m": a}
    q["extent"] = {p: est(a) for p, a in affected.items()}
    q["reported"] = {lab: est(a) for lab, a in reported.items()}
    q["labelled_unassessed_segments"] = {p: n for p, n in labelled_unassessed.items() if n}
    q["worst_severity"] = dict(sorted(worst_sev.items()))
    q["labels_present"] = sorted({lab for s in segments if _is_classified(s) and s.get("vlm_class") == "distress"
                                  for lab in _labels(s)})
    if runs_in_unit is not None:
        pr, cr = list(runs_in_unit.get("potholes") or []), list(runs_in_unit.get("cracking") or [])
    else:
        pr, cr = runs(list(segments), PARAM_LABELS["potholes"]), runs(list(segments), PARAM_LABELS["cracking"])
    q["runs"] = {"potholes": len(pr), "cracking": len(cr)}
    q["runs_per_km"] = {k: (v / (D / 1000.0) if D > 0 else None) for k, v in q["runs"].items()}
    q["run_length_hist"] = {"potholes": run_length_hist(pr), "cracking": run_length_hist(cr)}
    q["lane_width_m"] = lane_width_m
    # nominal area counts every run with a flagged segment in this unit (a run crossing a boundary counts in both
    # units here, so the area band never contradicts a flagged segment); runs/km count each run once
    n_area = len(pr) if pothole_runs_touching is None else pothole_runs_touching
    q["pothole_runs_touching"] = n_area
    q["pothole_nominal_area_pct"] = (100.0 * n_area * SMALL_POTHOLE_AREA_M2 / (D * lane_width_m)) if D > 0 else None
    cal = current_calibration() if calibration is _DEFAULT else calibration
    q["probe"] = {p: probe_value(segments, spec, q["imaged_m"], (length_m / 1000.0) if p == "potholes" else None)
                  for p, spec in ((cal or {}).get("params") or {}).items() if spec["source"] == "probe"}
    q["roughness"] = roughness_value(segments, q["imaged_m"])
    return q


# ------------------------------------------------------------------ rating
def rate_unit(q: Mapping, category: str) -> dict:
    """Per-parameter bands and the bracketed App-4 final for one unit's quantities.

    Assessed parameters: band/rating for E_lo and E_hi (lo = fewer distress = better). Not-assessed
    parameters: bracketed r = 1.0 (final_lo) / 3.0 (final_hi). Insufficient coverage -> rated False."""
    tab = TABLES[category]
    out = {"category": category, "table": tab["table"], "rated": q["coverage_status"] != "insufficient",
           "coverage_status": q["coverage_status"], "params": [], "not_assessed": []}
    if not out["rated"]:
        out.update(final=None, headline=None)
        return out
    lo_w, hi_w, assessed_w_best, assessed_w_worst = [], [], [], []
    unc_lo, unc_hi, unc_ok = [], [], True
    for p in tab["params"]:
        w = WEIGHTS[p]
        row = {"param": p, "name": PARAM_NAMES[p], "unit": PARAM_UNITS.get(p, "%"), "weight": w,
               "assessed": is_assessed(p), "evidence": ASSESSMENT[p]["evidence"],
               "limits": list(tab["params"][p])}
        if is_assessed(p):
            e = q["extent"][p]
            x_best, x_worst = e["lo"], e["hi"]
            r_best, r_worst = rating_value(category, p, x_best), rating_value(category, p, x_worst)
            row.update(E_lo=x_best, E_hi=x_worst, E_uncorrected=e["uncorrected"],
                       band_lo=band(category, p, x_best), band_hi=band(category, p, x_worst),
                       band=class_range(band(category, p, x_worst), band(category, p, x_best)),
                       rating_best=r_best, rating_worst=r_worst,
                       weighted_best=weighted(w, r_best), weighted_worst=weighted(w, r_worst))
            if row["band_lo"] != row["band_hi"] and e["affected_m"] <= 0:
                # the worse end comes only from the gate-skipped length (REVIEW_backend #5)
                row["band_note"] = (f"{row['band_lo']} (no {PARAM_NAMES[p].lower()} label); {row['band_hi']} if "
                                    f"skipped segments hold {'a pothole' if p == 'potholes' else 'cracks'}")
            if e["uncorrected"] is not None:
                r_u = rating_value(category, p, e["uncorrected"])
                row.update(band_uncorrected=band(category, p, e["uncorrected"]), rating_uncorrected=r_u)
                unc_lo.append(weighted(w, r_u))
                unc_hi.append(weighted(w, r_u))
            else:
                unc_ok = False
            hi_w.append(row["weighted_best"])
            lo_w.append(row["weighted_worst"])
            assessed_w_best.append(row["weighted_best"])
            assessed_w_worst.append(row["weighted_worst"])
        else:
            row.update(rating_best=3.0, rating_worst=1.0, weighted_best=weighted(w, 3.0),
                       weighted_worst=weighted(w, 1.0), band="not assessed")
            hi_w.append(row["weighted_best"])
            lo_w.append(row["weighted_worst"])
            unc_lo.append(row["weighted_worst"])
            unc_hi.append(row["weighted_best"])
            out["not_assessed"].append(p)
        out["params"].append(row)
    f_lo, f_hi = mean1(lo_w), mean1(hi_w)
    c_lo, c_hi = final_class(f_lo), final_class(f_hi)
    na = ", ".join(PARAM_NAMES[p].lower() for p in out["not_assessed"])
    out["final"] = {"lo": f_lo, "hi": f_hi, "class_lo": c_lo, "class_hi": c_hi,
                    "class": class_range(c_lo, c_hi),
                    "label": class_range(c_lo, c_hi) + (f" ({na} not assessed)" if na else ""),
                    "n_params": len(tab["params"]), "n_assessed": len(tab["params"]) - len(out["not_assessed"])}
    # sensitivity lines (DESIGN §4)
    sens = {}
    sw = sum(WEIGHTS[p] for p in tab["params"])
    sens["sum_wr_over_sum_w"] = {"lo": d1(sum(lo_w) / sw), "hi": d1(sum(hi_w) / sw)}
    sens["sum_wr_over_sum_w"]["class"] = class_range(final_class(sens["sum_wr_over_sum_w"]["lo"]),
                                                     final_class(sens["sum_wr_over_sum_w"]["hi"]))
    if assessed_w_best:
        a_lo, a_hi = mean1(assessed_w_worst), mean1(assessed_w_best)
        sens["assessed_only"] = {"lo": a_lo, "hi": a_hi, "class": class_range(final_class(a_lo), final_class(a_hi))}
    if unc_ok:
        u_lo, u_hi = mean1(unc_lo), mean1(unc_hi)
        sens["uncorrected"] = {"lo": u_lo, "hi": u_hi, "class": class_range(final_class(u_lo), final_class(u_hi))}
    sens["coverage_rule_80"] = q["rated_80"]
    sens["coverage_rule_50"] = q["rated_50"]
    out["sensitivity"] = sens
    # F5: with 5 m segments one flagged segment may already exceed the Poor limit of the unit: then the band is
    # in effect "flagged / not flagged" (the IRC arithmetic is unchanged; this only says how to read it)
    D = q.get("denominator_m") or 0.0
    for row in out["params"]:
        if not row["assessed"] or D <= 0:
            continue
        one = 100.0 * 5.0 / D
        U = TABLES[category]["params"][row["param"]][1]
        if one > U + EPS:
            row["resolution_note"] = (
                f"One flagged 5 m segment is {one:.1f} % of this unit's surveyed length, above the Poor limit "
                f"({U:g} %), so this band reads Poor whenever any {PARAM_NAMES[row['param']].lower()} is flagged: "
                f"read it as 'flagged: yes/no'" + ("; use runs per km for the amount" if row["param"] == "potholes"
                                                   else ""))
    head = {}
    for row in out["params"]:
        if row["assessed"]:
            head[row["param"]] = {"band": row["band"], "E_lo": row["E_lo"], "E_hi": row["E_hi"]}
    head["pothole_runs_per_km"] = q["runs_per_km"]["potholes"]
    head["pothole_runs"] = q["runs"]["potholes"]
    if q.get("pothole_nominal_area_pct") is not None:
        head["pothole_nominal_area_pct"] = q["pothole_nominal_area_pct"]
        head["pothole_nominal_area_band"] = band(category, "potholes", q["pothole_nominal_area_pct"])
    out["headline"] = head
    return out


# ------------------------------------------------------------------ treatments (RESEARCH §6)
TREATMENT_HEADING = ("Candidate treatments per IRC:82-2015 — to be confirmed by field inspection. No label in this "
                     "report has been validated against Bengaluru expert labels.")

_RENEWAL = {
    "urban": "periodic renewal: 30-40 BC/SDBC/2 Coat Micro Surfacing (Table 9.1, urban; roughness trigger 2600 mm/km)",
    "highway": "periodic renewal: NH 30-40 BC, SH 30 BC/30 SDBC (Table 9.1)",
    "rural": "periodic renewal 'when condition rating reaches 2 to 1': SD/PC/MSS/SDBC (20-25) (Table 9.1, MDR/RR)",
    "mdr": "periodic renewal 'when condition rating reaches 2 to 1': SD/PC/MSS/SDBC (20-25) (Table 9.1, MDR/RR)",
}
CLASS_TEXT = {
    "Good": ("Routine maintenance (pothole filling, crack repair, patching, markings); consider preventive "
             "maintenance (crack sealing, fog seal, slurry seal, microsurfacing, surface dressing, thin surfacing) "
             "so that the rating does not drop below 2 (§4.1, §5.3 p. 7, §8.2.4, Tables 8.1-8.7)."),
    "Fair": ("At a rating of about 2: {renewal}. Preventive treatments are effective only before the rating drops "
             "below 2; slurry seal/microsurfacing is not preventive on an extensively cracked surface (§5.3 p. 7, "
             "§8.2.3, §8.2.4)."),
    "Poor": ("Detailed investigation to assess the strengthening requirement (Table 9.1 note *); if structurally "
             "adequate, periodic renewal suffices, otherwise rehabilitation per IRC:81 / IRC:115 / IRC:37, outside "
             "IRC:82 (§2.4, §9.3)."),
}
ANY_CLASS_TEXT = ("Distress that adversely influences riding quality (e.g. potholes) should be rectified straightway "
                  "(§5.3, p. 6). Recycling of the wearing course is preferred where feasible (Table 9.1, second "
                  "footnote).")

DISTRESS_TREATMENT: dict[str, dict] = {
    "Potholes": {"any": "Rectify straightway (§5.3). Cut the deteriorated edges, fill with open- or dense-graded "
                        "premix (immediate-use, storable cold mix per IRC:116/IRC:100, ready-made mix, or patching "
                        "machine); potholes deeper than 75 mm in layers of 25-75 mm", "clause": "7.5.3.5; 10.3.8; 11.4"},
    "Alligator Cracking": {"any": "Pit/core to find the cause (structural adequacy, subsurface moisture) "
                                  "(§7.3.3.5). Crack sealing with bitumen emulsion or rubberised/modified bitumen; "
                                  "milling and surface recycling; designed overlay if structurally inadequate",
                           "clause": "7.3.3.5"},
    "Longitudinal Cracking": {"Low": "Crack sealing, preferably with rubberised bitumen",
                              "Medium": "Crack sealing, preferably with rubberised bitumen",
                              "High": "Remove and replace the cracked layer with a fresh overlay",
                              "clause": "7.3.4.5"},
    "Transverse Cracking": {"any": "Slurry seal or rubberised bitumen", "clause": "7.3.5.5"},
    "Hairline Cracks": {"any": "Fog seal, liquid rejuvenating agents, slurry seal, microsurfacing", "clause": "7.3.2.5"},
    "Edge Cracking": {"any": "Improve edge drainage, replace shoulder material with permeable material; with heavy "
                             "truck traffic a 75-100 mm structural overlay", "clause": "7.3.6.5"},
    "Ravelling": {"Low": "Fog seal, sand seal, seal coat, slurry seal or microsurfacing",
                  "Medium": "Seal coat, slurry seal or microsurfacing in the affected area",
                  "High": "Single or multiple liquid seal coat, slurry seal, microsurfacing, or a thin overlay "
                          "(mix seal surfacing / open-graded premix carpet)", "clause": "7.5.2.5"},
    "Hungry Surface": {"any": "Fog seal / liquid rejuvenators; slurry seal or microsurfacing 2-4 mm", "clause": "7.2.4.5"},
    "Stripping": {"any": "Hot coarse sand (>= 150 °C) rolled in; liquid seal; else remove and relay; rejuvenating "
                         "sealants, slurry seal, microsurfacing", "clause": "7.5.1.5"},
    "Bleeding": {"Low": "Heated coarse sand (passing 1.18 mm), light rolling",
                 "Medium": "Pre-coated chips (4.75-1.18 mm, 1 % bitumen), light rolling",
                 "High": "Mill out and repair; RAP plus fresh aggregate, then slurry seal/microsurfacing; or "
                         "open-graded premix carpet", "clause": "7.2.1.5"},
    "Rutting": {"any": "Fill ruts with premix and compact (tack coat first); microsurfacing fills ruts up to 40 mm "
                       "(depth not measured)", "clause": "7.4.2.5; 10.3.4"},
    "Corrugation": {"any": "Scarify and recompact, then new surfacing; or cut high spots / sand-bitumen premix",
                    "clause": "7.4.3.5"},
    "Shoving": {"any": "Remove down to a firm base and lay a stable premix patch", "clause": "7.4.4.5"},
    "Slippage": {"any": "Remove the surface around the area to well-bonded material; tack coat; premix patch",
                 "clause": "7.4.1.5"},
    "Shallow Depression": {"any": "Fill with premix and compact to profile", "clause": "7.4.5.5"},
    "Settlement": {"any": "Excavate a defective fill and rebuild; under-drains where drainage failed; designed "
                          "pavement if thickness is inadequate", "clause": "7.4.6.5"},
    "Edge Breaking": {"any": "Remove to a regular section; rebuild pavement and shoulder together; shoulder slope "
                             "1 % steeper than camber; improve drainage", "clause": "7.5.4.5"},
    "Streaking": {"any": "Remove the streaked surface and apply a new surface", "clause": "7.2.3.5"},
}
VALIDATED_LABELS = set(PARAM_LABELS["cracking"]) | set(PARAM_LABELS["potholes"])   # "some validation" (Attain)


def class_treatment(cls_range: Optional[str], category: str) -> str:
    """Treatment text for a (possibly bracketed) final class."""
    if not cls_range:
        return "Insufficient data: no treatment suggested."
    parts = cls_range.split("–")
    if len(parts) == 1:
        return CLASS_TEXT[parts[0]].format(renewal=_RENEWAL[category])
    lo, hi = parts[0], parts[-1]
    txt = f"{cls_range}: "
    if lo == "Poor":
        txt += ("investigate strengthening need if Poor is confirmed (Table 9.1 note *); else "
                + (_lower1(CLASS_TEXT["Fair"].format(renewal=_RENEWAL[category])) if hi in ("Fair", "Good") else ""))
    else:
        txt += ("if Fair is confirmed: " + _lower1(CLASS_TEXT["Fair"].format(renewal=_RENEWAL[category])) +
                " If Good: " + CLASS_TEXT["Good"])
    return txt


def distress_treatments(q: Mapping) -> list[dict]:
    """One candidate line per distress label present in the unit, by its worst visually judged severity."""
    worst = q.get("worst_severity") or {}
    out = []
    for lab in sorted(set(q.get("labels_present") or []) | set(worst)):
        t = DISTRESS_TREATMENT.get(lab)
        if not t:
            continue
        sev = worst.get(lab)
        text = t[sev] if sev in t else (t.get("any") or t.get("Medium") or t.get("Low"))
        out.append({"label": lab, "severity": sev or "not judged", "treatment": text, "clause": t["clause"],
                    "validated": lab in VALIDATED_LABELS,
                    "note": "" if lab in VALIDATED_LABELS else "label unvalidated"})
    return out


def _lower1(s: str) -> str:
    return s[:1].lower() + s[1:] if s else s
