"""
How per-type probe probabilities become the Stage 2 answer - single source of
truth for production (app/model.py) and evaluation (stage2_probe_report.py).

If the two computed this separately, a test-set number could describe a rule
production does not run. Both call combine() below.

The rule ("probe-verified list", schema 1)
------------------------------------------
Types are split by whether labelled data exists to set their threshold.

  CALIBRATED groups (config["groups"]): types Attain annotates - linear
    (longitudinal / transverse) cracking, alligator cracking, potholes,
    ravelling, hungry surface (weathering) - plus the Patching indicator.
    The probe DECIDES these: a type is reported when its P(yes) reaches the
    threshold tuned on the DEV split, whatever the free-form list said. This
    is where the measured gain is: the list almost never names alligator
    cracking, potholes on wide frames, ravelling or weathering.

  UNCALIBRATED types (the other 12 IRC types: bleeding, rutting, edge
    breaking, ...): no labelled data, so no threshold can be justified. The
    probe may only REMOVE them: a type the free-form list named is kept if the
    probe also gives it P(yes) >= verify_threshold, and dropped otherwise. The
    probe never adds an uncalibrated type on its own.

  Condition indicators (Patching) never enter distress_types; they are
  reported separately.

Order: calibrated types by calibrated probability, then verified types by raw
P(yes). distress_types[0] is therefore "the type the model is most confident
is present", NOT "the most prominent distress" - nothing here measures extent.

Confidence: the probability that every calibrated decision is right, under
per-type Platt calibration fitted on DEV and treating the decisions as
independent: prod_g ( q_g if reported else 1 - q_g ).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

SCHEMA_VERSION = 1


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    e = math.exp(z)
    return e / (1 + e)


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def calibrate(p: float, platt) -> float:
    if not platt:
        return p
    a, b = platt
    return _sigmoid(a * _logit(p) + b)


def load_config(path) -> dict:
    """Load and validate a probe configuration. Raises ValueError with the
    reason; the caller decides whether that disables the probe (production:
    yes, it falls back to the free-form list)."""
    path = Path(path)
    if not path.exists():
        raise ValueError(f"probe config not found: {path}")
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"probe config unreadable ({type(e).__name__}: {e})")
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict, known_keys: Optional[set] = None) -> None:
    if cfg.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"probe config schema_version {cfg.get('schema_version')!r} "
                         f"!= {SCHEMA_VERSION}")
    v = cfg.get("variant") or {}
    if not isinstance(v.get("system_style"), str) or not isinstance(v.get("question_style"), str):
        raise ValueError("probe config variant needs system_style and question_style")
    groups = cfg.get("groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("probe config has no groups")
    for name, g in groups.items():
        if g.get("kind") not in ("distress", "indicator"):
            raise ValueError(f"group {name!r}: kind must be distress or indicator")
        keys = g.get("keys")
        if not isinstance(keys, list) or not keys:
            raise ValueError(f"group {name!r}: keys must be a non-empty list")
        if known_keys is not None:
            missing = [k for k in keys if k not in known_keys]
            if missing:
                raise ValueError(f"group {name!r}: unknown probe keys {missing}")
        t = g.get("threshold")
        if not isinstance(t, (int, float)) or not (0.0 < t < 1.0):
            raise ValueError(f"group {name!r}: threshold must be in (0, 1), got {t!r}")
        pl = g.get("platt")
        if pl is not None and (not isinstance(pl, list) or len(pl) != 2
                               or not all(isinstance(x, (int, float)) and math.isfinite(x)
                                          for x in pl)):
            raise ValueError(f"group {name!r}: platt must be [a, b] or null")
    vt = cfg.get("verify_threshold")
    if not isinstance(vt, (int, float)) or not (0.0 < vt < 1.0):
        raise ValueError("verify_threshold must be in (0, 1)")
    sn = cfg.get("stage1_safety_net")
    if sn is not None:
        if not isinstance(sn.get("keys"), list) or not sn["keys"]:
            raise ValueError("stage1_safety_net.keys must be a non-empty list")
        if known_keys is not None and any(k not in known_keys for k in sn["keys"]):
            raise ValueError("stage1_safety_net.keys contains unknown probe keys")
        t = sn.get("threshold")
        if not isinstance(t, (int, float)) or not (0.0 < t < 1.0):
            raise ValueError("stage1_safety_net.threshold must be in (0, 1)")


def safety_net(p: dict, cfg: dict) -> Optional[dict]:
    """Stage 1 said Normal: does the probe see distress anyway?

    Returns None when the config has no safety net. A flag never changes a
    label - it only routes the image to expert review instead of letting a
    possibly-missed distress be auto-classified as Normal.
    """
    sn = cfg.get("stage1_safety_net")
    if not sn:
        return None
    key = max(sn["keys"], key=lambda k: p[k])
    score = p[key]
    return {"flagged": score >= sn["threshold"], "score": round(score, 6),
            "strongest_type": key, "threshold": sn["threshold"]}


def combine(generated: list, p: dict, cfg: dict, output_label: Optional[dict] = None) -> dict:
    """Merge the free-form list with the per-type probabilities.

    generated     distress_types parsed from the free-form generation
    p             {probe key: P(yes)} for every probe question
    cfg           a validated probe configuration
    output_label  {probe key: label written when positive}; defaults to the
                  key itself (sub-patterns map to their parent type)
    """
    output_label = output_label or {}
    groups = cfg["groups"]
    vt = float(cfg["verify_threshold"])
    calibrated_keys = {k for g in groups.values() for k in g["keys"]}

    scored: dict = {}          # distress label -> ordering score
    indicators: list = []
    group_out: dict = {}
    confidence = 1.0
    for name, g in groups.items():
        s = max(p[k] for k in g["keys"])
        dec = s >= g["threshold"]
        q = calibrate(s, g.get("platt"))
        group_out[name] = {"score": round(s, 6), "threshold": g["threshold"],
                           "reported": dec, "calibrated_p": round(q, 6)}
        if g["kind"] == "indicator":
            if dec:
                indicators.append(name if not g.get("label") else g["label"])
            continue
        confidence *= q if dec else (1 - q)
        if dec:
            for k in g["keys"]:
                if p[k] >= g["threshold"]:
                    lab = output_label.get(k, k)
                    scored[lab] = max(scored.get(lab, 0.0), 1.0 + q)  # calibrated first

    verified, rejected, dropped = [], [], []
    for t in generated or []:
        if t in calibrated_keys:
            continue                       # decided by the probe above
        if t not in p:
            dropped.append(t)              # "Normal", "Unknown", non-IRC text
        elif p[t] >= vt:
            verified.append(t)
            scored.setdefault(t, p[t])     # raw P(yes) < 1, so after calibrated
        else:
            rejected.append(t)

    types = sorted(scored, key=lambda t: -scored[t])
    gen = list(generated or [])
    return {
        "types": types,
        "indicators": indicators,
        "confidence": round(confidence, 6),
        "groups": group_out,
        "added_by_probe": [t for t in types if t not in gen],
        "removed_by_probe": [t for t in gen if t in calibrated_keys and t not in types]
                            + rejected,
        "verified_uncalibrated": verified,
        "dropped_non_taxonomy": dropped,
        "rule": f"probe_verified_list/schema{SCHEMA_VERSION}",
    }
