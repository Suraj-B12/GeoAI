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
IRI_RURAL = {
    # printed denominator "IRI^2 - 9.272 IRI^2 + 21.13 IRI + 3.037" read as IRI^3 - 9.272 IRI^2 ... (typo: two
    # IRI^2 terms; the cubic reading is the only one that follows the printed curve) — see IRC82_2023.md
    "SD": lambda x: _rat(x, (-25.68, 646.2, -4465.0, 9408.0), (1.0, -9.272, 21.13, 3.037)),
    "OGPC": lambda x: _rat(x, (-4.02, 98.33, -670.6, 1444.0), (1.0, -8.215, 17.79)),
    "MSS": lambda x: _rat(x, (-24.69, 593.2, -3663.0, 6786.0), (1.0, 7.173, -9.017, 15.43)),
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


def pci_param(road_class: str, param: str, x: float, surface: Optional[str] = None) -> float:
    """PCI_i (0-100) of one parameter value. IRI below 2 m/km (the left end of every printed curve) is scored as
    2 m/km would be; any equation result is clamped to [0, 100] (the curves are only drawn inside it)."""
    if road_class not in EQUATIONS:
        raise ValueError(f"road class must be one of {', '.join(ROAD_CLASSES)}")
    if param not in WEIGHTS:
        raise ValueError(f"unknown parameter {param!r}")
    x = float(x)
    if not math.isfinite(x) or x < 0:
        raise ValueError(f"{param}: value must be a finite number >= 0")
    f = EQUATIONS[road_class][param]
    if param == "iri":
        if isinstance(f, dict):
            s = surface or DEFAULT_SURFACE[road_class]
            if s not in f:
                raise ValueError(f"IRI equation for {s} not given for {road_class} roads")
            f = f[s]
        x = max(x, 2.0)
    try:
        v = f(x)
    except ZeroDivisionError:
        v = 0.0
    if not math.isfinite(v):
        v = 0.0
    return max(0.0, min(100.0, v))


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
            pa, pb = pci_param(road_class, p, a, surface), pci_param(road_class, p, b, surface)
        except Unavailable as e:
            missing.append(p)
            hi += 100.0 * w
            rows.append({"param": p, "known": False, "weight": w, "reason": str(e)})
            continue
        best, worst = max(pa, pb), min(pa, pb)
        lo += w * worst
        hi += w * best
        rows.append({"param": p, "known": True, "weight": w, "value_lo": a, "value_hi": b, "pci_best": best,
                     "pci_worst": worst})
    single = None if missing or any(r["known"] and r["pci_best"] != r["pci_worst"] for r in rows) else lo
    return {"lo": lo, "hi": hi, "class_lo": pci_class(lo), "class_hi": pci_class(hi), "single": single,
            "missing": missing, "rows": rows, "known_weight": sum(WEIGHTS[p] for p in PARAMS if p not in missing)}


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
