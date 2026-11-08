"""
IRC:82-2023 (Second Revision) Pavement Condition Index (PCI, 0-100) — the pure arithmetic (no I/O).

Source: IRC:82-2023 "Code of Practice for Maintenance of Bituminous Roads", §5.1-5.2 (Tables 5.1-5.5, Fig. 5.1)
and Appendix-2 (Tables A2.1-A2.3, equations transcribed from the printed images, worked examples A.1-C.3).
Transcription notes and the example check: eval_results/pci/IRC82_2023.md. Copy read: a third-party hosted PDF
whose title page reads "IRC:82-2023 ... (Second Revision), Indian Roads Congress" (C:\\GeoAI\\pci\\irc82_2023).

    PCI = sum_i w_i * PCI_i                                      (§5.2, Table 5.4)
    weights: IRI 0.40, pothole 0.16, rut depth 0.14, cracking 0.12, ravelling 0.10, patching 0.08  (sum 1.00)
    PCI_i    = the Appendix-2 equation of the road class (highway / MDR / rural / urban), clamped to [0, 100]
    extents  = % of the surface area of ALL lanes of one carriageway (eq. 1); potholes = NUMBER per section =
               pothole area (m2) / 0.1 m2 (eq. 2); rut depth and IRI = average of all lanes of one direction
    section  = 100 m to 1000 m (project / network level)
    classes  (Table 5.5): >90 Excellent, >80 Good, >60 Satisfactory, >40 Fair, >20 Poor, <=20 Fail

Missing parameters are not invented: pci_range() gives the lowest and highest PCI the known parameters allow
(unknown PCI_i = 0 and = 100).
"""
from __future__ import annotations

import math
from typing import Mapping, Optional

EDITION = "IRC:82-2023"
NAME = ("IRC:82-2023 Pavement Condition Index (0-100) — screening estimate from automated image classification "
        "and phone sensors")

PARAMS = ("cracking", "ravelling", "potholes", "patching", "rut_depth", "iri")
WEIGHTS = {"iri": 0.40, "potholes": 0.16, "rut_depth": 0.14, "cracking": 0.12, "ravelling": 0.10,
           "patching": 0.08}
PARAM_NAMES = {"cracking": "Cracking", "ravelling": "Ravelling", "potholes": "Potholes", "patching": "Patching",
               "rut_depth": "Rut depth", "iri": "Roughness (IRI)"}
PARAM_UNITS = {"cracking": "%", "ravelling": "%", "potholes": "number", "patching": "%", "rut_depth": "mm",
               "iri": "m/km"}
POTHOLE_UNIT_AREA_M2 = 0.1          # eq. (2)
CLASSES = ((90.0, "Excellent"), (80.0, "Good"), (60.0, "Satisfactory"), (40.0, "Fair"), (20.0, "Poor"))
RECOMMENDATION = {"Excellent": "Routine maintenance", "Good": "Preventive maintenance",
                  "Satisfactory": "Renewal (as per section 9.5)",
                  "Fair": "Minor rehabilitation (based on structural evaluation)",
                  "Poor": "Major rehabilitation / structural overlay", "Fail": "Reconstruction"}
ROAD_CLASSES = ("highway", "mdr", "rural", "urban")
SURFACES = ("SD", "OGPC", "MSS", "SDBC", "BC")
SURFACE_NAMES = {"SD": "Surface Dressing", "OGPC": "Open Graded Premix Carpet", "MSS": "Mix Seal Surfacing",
                 "SDBC": "Semi Dense Bituminous Concrete/Course", "BC": "Bituminous Concrete"}

# Table 5.1-5.3 distress ranges (Excellent-Good / Satisfactory-Fair / Poor-Fail): (upper of EG, upper of SF).
# Potholes are a number (per km or per section). Shown beside each parameter; the PCI itself uses Appendix 2.
RANGES = {
    "highway": {"cracking": (5, 10), "ravelling": (1, 10), "potholes": (0, 2), "patching": (1, 10),
                "rut_depth": (5, 10)},
    "mdr": {"cracking": (10, 20), "ravelling": (10, 20), "potholes": (1, 5), "patching": (5, 20),
            "rut_depth": (10, 20)},
    "urban": {"cracking": (5, 15), "ravelling": (5, 10), "potholes": (0, 3), "patching": (5, 10),
              "rut_depth": (5, 10)},
}
RANGES["rural"] = RANGES["mdr"]
# Maximum permissible roughness (IRI m/km): (upper of EG, upper of SF), Tables 5.1-5.3
IRI_RANGES = {
    "highway": {"BC": (2.55, 3.30), "SMA": (2.55, 3.30), "SDBC": (2.55, 3.30)},
    "urban": {"BC": (2.55, 3.30), "SMA": (2.55, 3.30), "SDBC": (2.55, 3.30)},
    "mdr": {"SD": (4.03, 4.98), "OGPC": (3.79, 4.62), "MSS": (3.55, 4.27), "SDBC": (3.05, 4.03), "BC": (2.81, 3.55)},
    "rural": {"SD": (4.27, 4.98), "OGPC": (4.03, 4.62), "MSS": (3.79, 4.27), "SDBC": (3.33, 4.03)},
}


class Unavailable(ValueError):
    """A printed equation that cannot be used as printed (see IRC82_2023.md)."""


def _rat(x: float, num: tuple, den: tuple) -> float:
    """Polynomial ratio; coefficients highest power first."""
    n = sum(c * x ** (len(num) - 1 - i) for i, c in enumerate(num))
    d = sum(c * x ** (len(den) - 1 - i) for i, c in enumerate(den))
    return n / d


def _f7231(x):          # 7231 / (x^2 - 0.737 x + 73.09)
    return 7231.0 / (x * x - 0.737 * x + 73.09)


def _fexp2(x):          # 52.92 e^-0.02525x + 44.1 e^-0.2899x
    return 52.92 * math.exp(-0.02525 * x) + 44.1 * math.exp(-0.2899 * x)


def _fcubic(x):         # (-0.5662 x^3 + 85.14 x^2 - 3377 x + 5.182e4) / (x^2 - 29.31 x + 523.4)
    return _rat(x, (-0.5662, 85.14, -3377.0, 5.182e4), (1.0, -29.31, 523.4))


def _iri_power(x):      # 100 / (IRI^1.91 - 3.542 IRI + 4.315)
    return 100.0 / (x ** 1.91 - 3.542 * x + 4.315)


def _pot_highway(n):
    return _rat(n, (-80.32, 1129.0, -3524.0, 3566.0), (1.0, 5.791, -28.67, 35.72))


def _pot_mdr(n):
    """Printed (Tables A2.2 A/B): 0.0113 PN^4 - 0.2825 PN^3 + 2.6246 PN^2 + 99.529. As printed it RISES (192 at
    PN = 10) while its printed curve falls from ~83 at 1 to 0 at 10: a term is missing (a linear term near
    -18.9 PN would fit), so it is not used; MDR/rural pothole PCI is reported as not computable."""
    raise Unavailable("the printed MDR/rural pothole equation (Table A2.2) is incomplete: it rises to 192 at 10 "
                      "potholes while its curve falls to 0; pothole PCI not computed")


def _patch_mdr(x):
    return _rat(x, (-1.723e-3, -8.273e-2, 21.39, 419.1), (1.0, 4.267))


def _pot_urban(n):
    return _rat(n, (-130.6, 2112.0, -9550.0, 14390.0), (1.0, 6.619, -66.73, 144.1))


def _crack_urban(x):
    return 100.0 * math.exp(-0.0534 * x) - 0.006641 * x


IRI_MDR = {
    "SD": lambda x: _rat(x, (-0.09535, 4.235, -74.43, 642.4, -2625.0, 4034.0), (1.0, -9.065, 20.71)),
    "OGPC": lambda x: _rat(x, (-4.842, 121.5, -875.9, 2029.0), (1.0, -8.389, 19.85)),
    "MSS": lambda x: _rat(x, (-2.686, 64.96, -437.2, 1046.0, -596.2), (1.0, -9.766, 31.09, -30.94)),
    "SDBC": lambda x: _rat(x, (-4.468, 106.2, -694.7, 1507.0), (1.0, -6.778, 14.51)),
    "BC": lambda x: _rat(x, (-2.52, 55.18, -292.4, 476.4, -5.565), (1.0, -7.458, 17.77, -12.09)),
}


def _unusable(msg):
    def f(_x):
        raise Unavailable(msg)
    f.unusable = msg
    return f


# Rural SD and MSS as printed are NOT used (REVIEW_2023.md R1/R2). Every other roughness equation gives PCI ~80 at
# the Table 5.1-5.3 "Excellent to Good" IRI limit and ~40 at the "Satisfactory to Fair" limit (75-82 / 36-45 for
# all 8 usable curves); these two do not, and the rural roughness curve in our copy (p. 85) does not render, so
# no reading can be checked against a curve. The candidate readings are kept for the record only.
IRI_RURAL_CANDIDATES = {
    # printed denominator "IRI^2 - 9.272 IRI^2 + 21.13 IRI + 3.037" (two IRI^2 terms). Read as IRI^3: PCI 60.8 at
    # the EG limit 4.27 and 14.7 at the SF limit 4.98, and it dips to 14.5 at IRI 4.9 then rises to 27.8 at 6.
    "SD": lambda x: _rat(x, (-25.68, 646.2, -4465.0, 9408.0), (1.0, -9.272, 21.13, 3.037)),
    # printed "IRI^3 + 7.173 IRI^2 - 9.017 IRI + 15.43": PCI 48 at IRI 2 and 0.6 at the EG limit 3.79 (contradicts
    # Table 5.2). With both middle signs flipped ("- 7.173 IRI^2 + 9.017 IRI", the alternating pattern of every
    # other denominator) it gives 79.0 / 38.4 at 3.79 / 4.27 like every other curve -- two sign errors, unconfirmed.
    "MSS_printed": lambda x: _rat(x, (-24.69, 593.2, -3663.0, 6786.0), (1.0, 7.173, -9.017, 15.43)),
    "MSS_signs_flipped": lambda x: _rat(x, (-24.69, 593.2, -3663.0, 6786.0), (1.0, -7.173, 9.017, 15.43)),
}
IRI_RURAL = {
    "SD": _unusable("the printed rural SD roughness equation (Table A2.2 B) has a garbled denominator "
                    "('IRI^2 - 9.272 IRI^2 ...'); no reading fits Table 5.2 and the curve is not legible; "
                    "roughness PCI not computed"),
    "OGPC": lambda x: _rat(x, (-4.02, 98.33, -670.6, 1444.0), (1.0, -8.215, 17.79)),
    "MSS": _unusable("the printed rural MSS roughness equation (Table A2.2 B) gives PCI 48 at IRI 2 and 0.6 at "
                     "the Table 5.2 Excellent-to-Good limit (3.79); likely sign errors; roughness PCI not computed"),
    "SDBC": lambda x: _rat(x, (25.16, 48.45), (1.0, -5.079, 7.595)),
}

EQUATIONS = {
    "highway": {"cracking": _f7231, "ravelling": _fexp2, "potholes": _pot_highway, "patching": _fexp2,
                "rut_depth": _f7231, "iri": _iri_power},
    "mdr": {"cracking": _fcubic, "ravelling": _fcubic, "potholes": _pot_mdr, "patching": _patch_mdr,
            "rut_depth": _fcubic, "iri": IRI_MDR},
    "rural": {"cracking": _fcubic, "ravelling": _fcubic, "potholes": _pot_mdr, "patching": _patch_mdr,
              "rut_depth": _fcubic, "iri": IRI_RURAL},
    "urban": {"cracking": _crack_urban, "ravelling": _f7231, "potholes": _pot_urban, "patching": _f7231,
              "rut_depth": _f7231, "iri": _iri_power},
}
DEFAULT_SURFACE = {"highway": "BC", "urban": "BC", "mdr": "BC", "rural": "SDBC"}

# Largest value any equation is evaluated at (beyond these every curve is 0 or ~0; avoids float overflow of
# IRI^5 etc. for absurd inputs). Extents are percentages and cannot exceed 100.
X_MAX = {"cracking": 100.0, "ravelling": 100.0, "patching": 100.0, "potholes": 1.0e4, "rut_depth": 1.0e4,
         "iri": 1.0e3}
IRI_LEFT = 2.0      # left end of every printed roughness curve


def _iri_peak(f) -> float:
    """IRI (>= 2) at which a roughness equation is largest on [2, 20]. A smoother road must not score lower than a
    rougher one, so IRI below the peak is scored as the peak (rural OGPC rises 86.6 -> 100.3 from IRI 2 to 3.4,
    rural SDBC 68.7 -> 98.0 from 2 to 2.8; every other curve falls from IRI 2, where the peak is 2)."""
    if getattr(f, "unusable", None):
        return IRI_LEFT
    best_x, best_v = IRI_LEFT, -math.inf
    for i in range(18001):
        x = IRI_LEFT + i * 0.001
        try:
            v = f(x)
        except ZeroDivisionError:
            continue
        if math.isfinite(v) and v > best_v + 1e-12:
            best_x, best_v = x, v
    return best_x


IRI_PEAK = {(rc, s): _iri_peak(f) for rc, d in (("mdr", IRI_MDR), ("rural", IRI_RURAL)) for s, f in d.items()}


def _equation(road_class: str, param: str, surface: Optional[str]):
    f = EQUATIONS[road_class][param]
    left = None
    if param == "iri":
        left = IRI_LEFT
        if isinstance(f, dict):
            s = surface or DEFAULT_SURFACE[road_class]
            if s not in f:
                if s in SURFACES:
                    raise Unavailable(f"IRC:82-2023 gives no roughness equation for {SURFACE_NAMES[s]} ({s}) on "
                                      f"{road_class} roads; roughness PCI not computed")
                raise ValueError(f"surface must be one of {', '.join(SURFACES)}")
            left = IRI_PEAK[(road_class, s)]
            f = f[s]
    return f, left


def pci_param(road_class: str, param: str, x: float, surface: Optional[str] = None) -> float:
    """PCI_i (0-100) of one parameter value. IRI below 2 m/km (the left end of every printed curve), or below the
    peak of a curve that first rises, is scored as the peak; extents above 100 % as 100 %; any equation result
    is clamped to [0, 100] (the curves are only drawn inside it). Raises Unavailable for an equation that
    cannot be used as printed (MDR/rural potholes, rural SD/MSS roughness, rural BC roughness)."""
    if road_class not in EQUATIONS:
        raise ValueError(f"road class must be one of {', '.join(ROAD_CLASSES)}")
    if param not in WEIGHTS:
        raise ValueError(f"unknown parameter {param!r}")
    x = float(x)
    if not math.isfinite(x) or x < 0:
        raise ValueError(f"{param}: value must be a finite number >= 0")
    f, left = _equation(road_class, param, surface)
    if left is not None:
        x = max(x, left)
    x = min(x, X_MAX[param])
    try:
        v = f(x)
    except ZeroDivisionError:           # no pole lies inside the evaluated domain (REVIEW_2023.md, grid scan)
        raise ValueError(f"{param}: equation undefined at {x}") from None
    if not math.isfinite(v):
        raise ValueError(f"{param}: equation not finite at {x}")
    return max(0.0, min(100.0, v))


def pci_param_bounds(road_class: str, param: str, a: float, b: float, surface: Optional[str] = None) -> tuple:
    """(worst, best) PCI_i over every value in [a, b] (not only the ends: several printed curves are not
    monotone, e.g. MDR SD roughness rises 36.3 -> 37.4 between IRI 4.86 and 5.31)."""
    a, b = (a, b) if a <= b else (b, a)
    vals = [pci_param(road_class, param, a, surface), pci_param(road_class, param, b, surface)]
    if b > a:
        n = 400
        vals += [pci_param(road_class, param, a + (b - a) * i / n, surface) for i in range(1, n)]
    return min(vals), max(vals)


def pci_class(v: float) -> str:
    for lo, name in CLASSES:
        if v > lo:
            return name
    return "Fail"


def pci(road_class: str, values: Mapping[str, float], surface: Optional[str] = None) -> dict:
    """Full PCI from all six values."""
    rows = []
    for p in PARAMS:
        v = pci_param(road_class, p, values[p], surface)
        rows.append({"param": p, "value": float(values[p]), "pci": v, "weight": WEIGHTS[p], "weighted": WEIGHTS[p] * v})
    total = sum(r["weighted"] for r in rows)
    return {"rows": rows, "pci": total, "class": pci_class(total), "recommendation": RECOMMENDATION[pci_class(total)]}


def pci_range(road_class: str, known: Mapping[str, Optional[float]], surface: Optional[str] = None) -> dict:
    """PCI bounds when some parameters are unknown (value None or absent): unknown PCI_i taken as 0 (lo) and 100
    (hi). known values may be (lo, hi) pairs for a parameter whose value is itself a range (lo = better)."""
    lo = hi = 0.0
    missing = []
    rows = []
    for p in PARAMS:
        w = WEIGHTS[p]
        v = known.get(p)
        if v is None:
            missing.append(p)
            hi += 100.0 * w
            rows.append({"param": p, "known": False, "weight": w})
            continue
        a, b = (v, v) if not isinstance(v, (tuple, list)) else (v[0], v[1])
        try:
            worst, best = pci_param_bounds(road_class, p, a, b, surface)
        except Unavailable as e:
            missing.append(p)
            hi += 100.0 * w
            rows.append({"param": p, "known": False, "weight": w, "reason": str(e)})
            continue
        lo += w * worst
        hi += w * best
        rows.append({"param": p, "known": True, "weight": w, "value_lo": a, "value_hi": b, "pci_best": best,
                     "pci_worst": worst})
    single = None if missing or any(r["known"] and r["pci_best"] != r["pci_worst"] for r in rows) else lo
    return {"lo": lo, "hi": hi, "class_lo": pci_class(lo), "class_hi": pci_class(hi), "single": single,
            "missing": missing, "rows": rows, "known_weight": sum(WEIGHTS[p] for p in PARAMS if p not in missing)}


def _split_normal_quantile(u: float, lo: float, mid: float, hi: float) -> float:
    """Value at probability u of a split-normal with 5 % quantile lo, median mid, 95 % quantile hi (each side its
    own spread); values below 0 are clipped to 0 (all PCI inputs are >= 0)."""
    from statistics import NormalDist
    z = NormalDist().inv_cdf(min(max(u, 1e-9), 1 - 1e-9))
    s = (mid - lo) / 1.6448536 if z < 0 else (hi - mid) / 1.6448536
    return max(0.0, mid + z * s)


def pci_estimate(road_class: str, inputs: Mapping[str, Mapping[str, float]], surface: Optional[str] = None,
                 n: int = 4000, seed: int = 20261109) -> dict:
    """PCI estimate with a 90 % interval from per-parameter estimates.

    inputs: {param: {"point": x, "lo90": a, "hi90": b[, "dist": "uniform"]}} for EVERY parameter ("uniform": a
    prior range drawn uniformly between lo90 and hi90, point = its middle) (a parameter without an estimate must be
    passed with its widest defensible bounds by the caller, or this returns None). Each parameter is drawn
    independently from a split-normal with those 5/50/95 % quantiles (independence is an assumption: in reality
    distresses co-occur, which would widen the interval somewhat); PCI = sum w_i PCI_i per draw; the result is the
    median and the 5-95 % range of the draws, plus the PCI of the point values. Deterministic (seeded)."""
    import random
    if any(p not in inputs for p in PARAMS):
        return None
    rnd = random.Random(seed)
    for p in PARAMS:                                  # fail early on an unusable equation
        pci_param(road_class, p, inputs[p]["point"], surface)
    draws = []
    for _ in range(n):
        tot = 0.0
        for p in PARAMS:
            q = inputs[p]
            if q.get("dist") == "uniform":            # a prior range without a typical value
                x = q["lo90"] + rnd.random() * (q["hi90"] - q["lo90"])
            else:
                x = _split_normal_quantile(rnd.random(), q["lo90"], q["point"], q["hi90"])
            tot += WEIGHTS[p] * pci_param(road_class, p, x, surface)
        draws.append(tot)
    draws.sort()
    point = sum(WEIGHTS[p] * pci_param(road_class, p, inputs[p]["point"], surface) for p in PARAMS)
    med = draws[n // 2]
    lo, hi = draws[int(0.05 * n)], draws[int(0.95 * n) - 1]
    contrib = {}
    for p in PARAMS:                                  # how much of the interval each parameter causes
        others = {k: inputs[k]["point"] for k in PARAMS}
        vals = []
        for x in (inputs[p]["lo90"], inputs[p]["hi90"]):
            others[p] = x
            vals.append(sum(WEIGHTS[k] * pci_param(road_class, k, others[k], surface) for k in PARAMS))
        contrib[p] = abs(vals[0] - vals[1])
    return {"point": point, "median": med, "lo90": lo, "hi90": hi, "class": pci_class(med),
            "class_lo": pci_class(lo), "class_hi": pci_class(hi), "recommendation": RECOMMENDATION[pci_class(med)],
            "width_by_param": contrib, "draws": n}


def potholes_number(area_m2: float) -> float:
    """eq. (2): equivalent pothole number = pothole area / 0.1 m2."""
    return float(area_m2) / POTHOLE_UNIT_AREA_M2


# Appendix-2 worked examples: (road class, values, printed PCI)
EXAMPLES = [
    ("highway", dict(cracking=0.5, ravelling=0.3, potholes=0, patching=0.05, rut_depth=0.10, iri=2.4), 93.58),
    ("highway", dict(cracking=5, ravelling=3, potholes=2, patching=2.6, rut_depth=10, iri=4.5), 41.37),
    ("highway", dict(cracking=14, ravelling=12, potholes=4, patching=15, rut_depth=14, iri=8), 20.54),
    ("mdr", dict(cracking=0.6, ravelling=0.1, potholes=1, patching=0.5, rut_depth=1, iri=2.8), 89.45),
    ("mdr", dict(cracking=6.5, ravelling=4.7, potholes=6, patching=6, rut_depth=8, iri=9.2), 47.17),
    ("mdr", dict(cracking=20, ravelling=16, potholes=10, patching=12, rut_depth=22, iri=12), 22.44),
    ("urban", dict(cracking=0.5, ravelling=0.2, potholes=0, patching=0.1, rut_depth=0, iri=2.4), 96.1),
    ("urban", dict(cracking=5, ravelling=2, potholes=3, patching=3.5, rut_depth=4, iri=6), 56.86),
    ("urban", dict(cracking=12, ravelling=8, potholes=4, patching=10, rut_depth=12, iri=7), 34.68),
]
