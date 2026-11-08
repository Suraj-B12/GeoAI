"""
Compare our IRC:82-2023 PCI with an external road survey: import, pre-registered split, calibration on one half,
blind evaluation on the other.

Protocol: eval_results/pci/survey_split_rule.json (registered 2026-11-08, before any survey data).

Survey CSV (UTF-8, header row; template eval_results/pci/survey_template.csv). Required per section:
    section_id, start_lat, start_lon, end_lat, end_lon
Optional per section (any subset; blanks allowed):
    road_class (urban|highway|mdr|rural), surface (SD|OGPC|MSS|SDBC|BC),
    cracking_pct, ravelling_pct, potholes_number, patching_pct, rut_depth_mm, iri_m_km,
    pci            the survey's own IRC:82-2023 PCI (0-100)
    other_score    any other per-section score (compared by rank correlation only)
Start/end positions are the section's ends on the road (GPS). Sections of 100-1000 m.

    python scripts/pci_survey_compare.py check    SURVEY.csv   # parse + split + which sections we can build
    python scripts/pci_survey_compare.py fit      SURVEY.csv   # calibration half -> param_calibration_candidate.json
    python scripts/pci_survey_compare.py evaluate SURVEY.csv   # test half, WITH the candidate -> survey_comparison.json
    python scripts/pci_survey_compare.py promote  SURVEY.csv   # candidate -> param_calibration.json (production reads
                                                               # it live), only after an evaluate of that candidate
Options: --database geoai (default from DATABASE_URL), --road-class urban, --surface BC.

Calibration (fit, calibration half only), per parameter the survey reports:
  - candidates: Stage 2 labels times a scale b (cracking extent, pothole runs), or the per-type probe at threshold t
    with value = a + b * f (f = fraction of the section's probe-scored length at or above t; a = 0 except rut depth);
  - t on a grid 0.05-0.95, b on a log grid 0.01-1000; the candidate with the lowest mean absolute error IN PCI_i
    POINTS wins (in native units for a parameter whose equation is unavailable for some section, e.g. MDR potholes);
  - its calibration-half MAE is written beside the MAE of a constant (the calibration-half median). Both are
    IN-SAMPLE (thousands of candidates are tried on the same half); a parameter whose best candidate is not
    better than the constant is NOT written to the calibration (it stays "not assessed").
  - IRI is NOT fitted here (the roughness workstream, app/pci_iri.py, owns iri_calibration.json).
Evaluate (test half only): our single PCI vs the survey PCI (MAE, bias, RMSE, Spearman, Bland-Altman, Table 5.5
classes exact / within one / quadratic kappa) with block-bootstrap 95 % CIs, beside the constant baseline; per
parameter MAE; and an equation check (survey parameters through our equations vs the survey's own PCI).
Every evaluate run is appended to eval_results/pci/survey_ledger.jsonl; a later fit on the same file is post hoc.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import pci_irc2023 as P23  # noqa: E402

OUT = PROJECT_ROOT / "eval_results" / "pci"
RULE = OUT / "survey_split_rule.json"
LEDGER = OUT / "survey_ledger.jsonl"
IRI_CAL_PATH = OUT / "iri" / "iri_calibration_candidate.json"   # fit writes; production never reads it
IRI_LIVE_PATH = OUT / "iri" / "iri_calibration.json"
CAL_PATH = OUT / "param_calibration_candidate.json"      # written by fit; never read by production
LIVE_PATH = OUT / "param_calibration.json"               # read live by app/pci_core.current_calibration()
COLS = {"cracking": "cracking_pct", "ravelling": "ravelling_pct", "potholes": "potholes_number",
        "patching": "patching_pct", "rut_depth": "rut_depth_mm", "iri": "iri_m_km"}
BLOCK = 3
T_GRID = [round(0.05 + 0.025 * i, 3) for i in range(37)]                      # 0.05 .. 0.95
# b: % parameters need b up to 100 (b = 100 x area factor), probe potholes hundreds per km at f = 1
B_GRID = sorted({float(f"{10 ** (-2 + 5 * i / 240):.4g}") for i in range(241)}
                | {round(0.05 * i, 4) for i in range(1, 61)})
SEED = 20261108
N_BOOT = 2000


# ============================================================ survey file
def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _num(v) -> Optional[float]:
    if v is None:
        return None
    v = str(v).strip()
    if v == "" or v.lower() in ("na", "n/a", "nan", "-"):
        return None
    x = float(v)
    if not math.isfinite(x):
        raise ValueError(f"not finite: {v}")
    return x


def read_survey(path: Path) -> list[dict]:
    rows = []
    pct = {"cracking", "ravelling", "patching"}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rd = csv.DictReader(fh)
        hdr = {h.strip().lower() for h in (rd.fieldnames or [])}
        need = {"section_id", "start_lat", "start_lon", "end_lat", "end_lon"}
        if not need <= hdr:
            raise SystemExit(f"survey CSV needs columns {sorted(need)}; has {sorted(hdr)}")
        seen = set()
        for i, raw in enumerate(rd, start=2):
            r = {k.strip().lower(): (v.strip() if isinstance(v, str) else v) for k, v in raw.items() if k}
            sid = r.get("section_id") or ""
            if not sid or sid in seen:
                raise SystemExit(f"line {i}: section_id missing or repeated ({sid!r})")
            seen.add(sid)
            try:
                row = {"section_id": sid, "line": i,
                       "a": {"lat": _num(r["start_lat"]), "lon": _num(r["start_lon"])},
                       "b": {"lat": _num(r["end_lat"]), "lon": _num(r["end_lon"])},
                       "road_class": (r.get("road_class") or "").lower() or None,
                       "surface": (r.get("surface") or "").upper() or None,
                       "pci": _num(r.get("pci")), "other_score": _num(r.get("other_score")),
                       "survey": {p: _num(r.get(c)) for p, c in COLS.items()}}
            except (ValueError, TypeError) as e:
                raise SystemExit(f"line {i}: {e}") from None
            if None in (row["a"]["lat"], row["a"]["lon"], row["b"]["lat"], row["b"]["lon"]):
                raise SystemExit(f"line {i}: start/end position missing")
            if row["road_class"] and row["road_class"] not in P23.ROAD_CLASSES:
                raise SystemExit(f"line {i}: road_class must be one of {P23.ROAD_CLASSES}")
            if row["surface"] and row["surface"] not in P23.SURFACES:
                raise SystemExit(f"line {i}: surface must be one of {P23.SURFACES}")
            if row["pci"] is not None and not 0 <= row["pci"] <= 100:
                raise SystemExit(f"line {i}: pci must be within 0-100 (IRC:82-2023); put other scales in other_score")
            for end in ("a", "b"):
                if not (-90 <= row[end]["lat"] <= 90 and -180 <= row[end]["lon"] <= 180):
                    raise SystemExit(f"line {i}: start/end position is not a latitude/longitude")
            for p, v in row["survey"].items():
                if v is not None and (v < 0 or (p in pct and v > 100)):
                    raise SystemExit(f"line {i}: {COLS[p]} must be >= 0" + (" and <= 100" if p in pct else ""))
            rc, sf = row["road_class"], row["surface"]
            if rc in ("mdr", "rural") and sf and sf not in P23.EQUATIONS[rc]["iri"]:
                raise SystemExit(f"line {i}: IRC:82-2023 gives no roughness equation for surface {sf} on {rc} roads")
            rows.append(row)
    if not rows:
        raise SystemExit("survey CSV has no sections")
    return rows


def test_half_sha(rows: list[dict]) -> str:
    """Hash of the test half's section ids: survives edits of the CSV (an extra column, a corrected value), so a
    fit after an evaluate is still recognised as post hoc."""
    ids = sorted(r["section_id"] for r in rows if r.get("half") == "test")
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def is_post_hoc(ledger: list[dict], file_sha: str, test_sha: str) -> bool:
    return any(e.get("cmd") == "evaluate" and (e.get("survey_sha256") == file_sha or e.get("test_half_sha256") == test_sha)
               for e in ledger)


def split(rows: list[dict]) -> list[dict]:
    """The pre-registered split (survey_split_rule.json): CSV order, blocks of 3, alternate calib / test."""
    for i, r in enumerate(rows):
        r["block"] = i // BLOCK
        r["half"] = "calibration" if r["block"] % 2 == 0 else "test"
    return rows


# ============================================================ our measurements per section
def measure(rows: list[dict], road_class: str, surface: Optional[str], probe_fp: Optional[str]) -> None:
    """Adds r["ours"] (or r["excluded"]) to every row: the report of A=start -> B=end on our data, the chosen
    segments with their probe scores, and the label quantities."""
    from app import pci_chainage as PC
    g = PC.load_graph(force=True)
    by_id = {n.segment_id: n.props for n in g.nodes}
    probe = PC._probe_scores(probe_fp) if probe_fp else {}
    rough = PC._roughness_rows()
    for r in rows:
        req = {"a": r["a"], "b": r["b"], "category": r["road_class"] or road_class}
        s = r["surface"] or surface
        if s:
            req["surface"] = s
        try:
            rep = PC.build_report(g, req)
        except PC.PciError as e:
            r["excluded"] = f"no report: {e}"
            continue
        st = rep["stretch"]
        if st["quantities"]["coverage_status"] == "insufficient":
            r["excluded"] = "our coverage insufficient (< 50 % imaged)"
            continue
        segs = []
        for sid in rep["segment_ids"]:
            pr = by_id.get(sid) or {}
            segs.append({"segment_id": sid, "length_m": float(pr.get("length_m") or 0.0), "status": pr.get("status"),
                         "probe": probe.get(sid), "roughness": rough.get(sid)})
        q = st["quantities"]
        km = rep["km"]
        params = ({x["param"]: x.get("value_point") for x in km[0]["pci"]["params"] if x.get("known")}
                  if len(km) == 1 and km[0]["pci"].get("pci") else None)
        r["ours"] = {"params": params, "report_length_m": rep["summary"]["length_m"], "segments": segs,
                     "cracking_point": q["extent"]["cracking"]["point"] if q["extent"]["cracking"] else None,
                     "pothole_runs": q["runs"]["potholes"], "skip_m": q["skip_m"],
                     "pci_report": rep["summary"]["pci"], "road_class": req["category"], "surface": req.get("surface")}


def probe_fraction(segs: list[dict], keys, t: float) -> Optional[float]:
    tot = pos = 0.0
    for s in segs:
        if s["status"] == "not_covered" or not s.get("probe"):
            continue
        tot += s["length_m"]
        if max(float(s["probe"].get(k) or 0.0) for k in keys) >= t - 1e-9:
            pos += s["length_m"]
    return pos / tot if tot > 0 else None


# ============================================================ fit
def _pci_err(rc, surf, p, pred, truth) -> Optional[float]:
    """|error| in PCI_i points; None when the parameter's equation is unavailable for this section."""
    try:
        return abs(P23.pci_param(rc, p, max(0.0, pred), surf) - P23.pci_param(rc, p, truth, surf))
    except P23.Unavailable:
        return None


def _errs(rcs, surfs, p, preds, truths) -> tuple:
    """(mean |error|, unit): in PCI_i points when every section has a usable equation, else in the parameter's
    own unit for ALL sections (never a mix of the two)."""
    e = [_pci_err(rc, s, p, a, y) for a, y, rc, s in zip(preds, truths, rcs, surfs)]
    if all(x is not None for x in e):
        return sum(e) / len(e), "PCI points"
    return sum(abs(max(0.0, a) - y) for a, y in zip(preds, truths)) / len(preds), P23.PARAM_UNITS[p]


def _fit_scale(xs, ys, rcs, surfs, p, a_free=False):
    """(a, b, MAE, unit) minimising MAE over the b grid (a = 0, or a from the median residual)."""
    best = None
    for b in B_GRID:
        a = 0.0
        if a_free:
            res = sorted(y - b * x for x, y in zip(xs, ys))
            a = res[len(res) // 2]
        mae, unit = _errs(rcs, surfs, p, [a + b * x for x in xs], ys)
        if best is None or mae < best[2] - 1e-12:
            best = (a, b, mae, unit)
    return best


def fit(rows: list[dict], probe_fp: str) -> dict:
    from app import pci_core as C
    cal_rows = [r for r in rows if r["half"] == "calibration" and "ours" in r]
    params, report = {}, {}
    for p in ("cracking", "ravelling", "potholes", "patching", "rut_depth"):
        rs = [r for r in cal_rows if r["survey"][p] is not None]
        if len(rs) < 5:
            report[p] = {"n": len(rs), "fitted": False, "reason": "fewer than 5 calibration sections with a value"}
            continue
        ys = [r["survey"][p] for r in rs]
        rcs = [r["ours"]["road_class"] for r in rs]
        sfs = [r["ours"]["surface"] for r in rs]
        med = sorted(ys)[len(ys) // 2]
        const, unit = _errs(rcs, sfs, p, [med] * len(ys), ys)
        cands = []
        if p in ("cracking", "potholes"):
            xs = [r["ours"]["cracking_point"] if p == "cracking" else
                  r["ours"]["pothole_runs"] + C.POTHOLE_SKIP_POINT * r["ours"]["skip_m"] / C.SEGMENT_M for r in rs]
            if all(x is not None for x in xs):
                a, b, mae, _u = _fit_scale(xs, ys, rcs, sfs, p)
                cands.append({"source": "labels", "b": b, "mae": mae})
        keys = C.PROBE_KEYS[p]
        for t in T_GRID:
            fs = [probe_fraction(r["ours"]["segments"], keys, t) for r in rs]
            ok = [(f, y, rc, s) for f, y, rc, s in zip(fs, ys, rcs, sfs) if f is not None]
            if len(ok) < len(rs):
                continue
            f_, y_, rc_, s_ = zip(*ok)
            if p == "potholes":
                # f is a length fraction; the survey reports a NUMBER per section: scale by section length
                lens = [r["ours"]["report_length_m"] / 1000.0 for r in rs]
                f_ = tuple(fi * li for fi, li in zip(f_, lens))
            a, b, mae, _u = _fit_scale(list(f_), list(y_), rc_, s_, p, a_free=(p == "rut_depth"))
            cands.append({"source": "probe", "keys": list(keys), "threshold": t, "a": a, "b": b, "mae": mae})
        if not cands:
            report[p] = {"n": len(rs), "fitted": False, "reason": "no usable candidate (missing probe scores?)"}
            continue
        best = min(cands, key=lambda c: (c["mae"], c["source"] != "labels"))
        better = best["mae"] < const - 1e-9
        ev = (f"fitted on {len(rs)} calibration sections of the survey (in-sample, best of {len(cands)} candidates): "
              f"MAE {best['mae']:.1f} {unit} vs {const:.1f} for a constant")
        spec = {k: v for k, v in best.items() if k != "mae"}
        spec["evidence"] = ev
        if p == "potholes" and spec["source"] == "probe":
            spec["evidence"] += "; b per km of section"
        report[p] = {"n": len(rs), "fitted": better, "mae": round(best["mae"], 3), "constant_mae": round(const, 3),
                     "error_unit": unit, "better_than_constant": better, "chosen": spec if better else None,
                     "best_candidate": spec, "candidates_tried": len(cands)}
        if better:
            params[p] = spec
        else:
            report[p]["reason"] = "best candidate not better than a constant (in-sample): not written"
    return {"params": params, "report": report}


# ============================================================ evaluate
def _cls_idx(v: float) -> int:
    order = ["Fail", "Poor", "Fair", "Satisfactory", "Good", "Excellent"]
    return order.index(P23.pci_class(v))


def _spearman(x, y) -> Optional[float]:
    if len(x) < 3:
        return None
    from scipy.stats import spearmanr
    r = spearmanr(x, y).statistic
    return None if r is None or not math.isfinite(r) else float(r)


def _qkappa(a: list[int], b: list[int], k: int = 6) -> Optional[float]:
    n = len(a)
    if n == 0:
        return None
    O = [[0] * k for _ in range(k)]
    for i, j in zip(a, b):
        O[i][j] += 1
    ra = [sum(O[i]) for i in range(k)]
    cb = [sum(O[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = (i - j) ** 2 / (k - 1) ** 2
            num += w * O[i][j]
            den += w * ra[i] * cb[j] / n
    return None if den == 0 else 1.0 - num / den


def metrics(pairs: list[tuple], const: float) -> dict:
    """pairs: (ours, survey, block). Point metrics + block-bootstrap CIs."""
    def point(ps):
        o = [p[0] for p in ps]
        s = [p[1] for p in ps]
        d = [a - b for a, b in zip(o, s)]
        n = len(d)
        mae = sum(abs(x) for x in d) / n
        bias = sum(d) / n
        sd = math.sqrt(sum((x - bias) ** 2 for x in d) / (n - 1)) if n > 1 else 0.0
        ci, si = [_cls_idx(v) for v in o], [_cls_idx(v) for v in s]
        return {"n": n, "mae": mae, "bias": bias, "rmse": math.sqrt(sum(x * x for x in d) / n),
                "constant_mae": sum(abs(const - b) for b in s) / n, "spearman": _spearman(o, s),
                "class_exact": sum(a == b for a, b in zip(ci, si)) / n,
                "class_within_one": sum(abs(a - b) <= 1 for a, b in zip(ci, si)) / n,
                "qkappa": _qkappa(ci, si), "bland_altman": [bias - 1.96 * sd, bias + 1.96 * sd]}
    out = point(pairs)
    blocks = sorted({p[2] for p in pairs})
    rnd = random.Random(SEED)
    boots = {k: [] for k in ("mae", "constant_mae", "bias", "spearman", "class_exact")}
    diff = []
    for _ in range(N_BOOT):
        pick = [rnd.choice(blocks) for _ in blocks]
        ps = [p for b in pick for p in pairs if p[2] == b]
        if len(ps) < 2:
            continue
        m = point(ps)
        for k in boots:
            if m[k] is not None:
                boots[k].append(m[k])
        diff.append(m["mae"] - m["constant_mae"])

    def ci(v):
        if len(v) < 50:
            return None
        v = sorted(v)
        return [v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]]
    out["ci95"] = {k: ci(v) for k, v in boots.items()}
    out["mae_minus_constant_ci95"] = ci(diff)
    return out


def evaluate(rows: list[dict]) -> dict:
    test = [r for r in rows if r["half"] == "test" and "ours" in r]
    calib_pci = [r["pci"] for r in rows if r["half"] == "calibration" and r["pci"] is not None]
    # pre-registered (survey_split_rule.json): the calibration-half MEAN survey PCI
    const = (sum(calib_pci) / len(calib_pci)) if calib_pci else None
    res = {"n_test_sections": len(test), "constant_from_calibration_half": const}
    single = [(r["ours"]["pci_report"]["single"], r["pci"], r["block"]) for r in test
              if r["pci"] is not None and r["ours"]["pci_report"] and r["ours"]["pci_report"]["single"] is not None]
    mid = [((r["ours"]["pci_report"]["lo"] + r["ours"]["pci_report"]["hi"]) / 2, r["pci"], r["block"]) for r in test
           if r["pci"] is not None and r["ours"]["pci_report"]]
    inside = [r["ours"]["pci_report"]["lo"] - 1e-9 <= r["pci"] <= r["ours"]["pci_report"]["hi"] + 1e-9 for r in test
              if r["pci"] is not None and r["ours"]["pci_report"]]
    res["single_pci"] = metrics(single, const) if (single and const is not None) else None
    res["single_pci_available"] = f"{len(single)} of {sum(1 for r in test if r['pci'] is not None)}"
    res["range_midpoint"] = metrics(mid, const) if (mid and const is not None) else None
    res["survey_pci_inside_our_range"] = (sum(inside) / len(inside)) if inside else None
    res["insufficient"] = len(single) < 10
    # addendum 1 (survey_split_rule.json, registered before any survey data): partial PCI over the parameters
    # that have a value in BOTH our report and the survey, sum w_i PCI_i / sum w_i, same equations both sides
    part = []
    for r in test:
        ours = r["ours"].get("params") or {}
        common = [p for p in P23.PARAMS if ours.get(p) is not None and r["survey"][p] is not None]
        if not common:
            continue
        try:
            o = [P23.pci_param(r["ours"]["road_class"], p, ours[p], r["ours"]["surface"]) for p in common]
            y = [P23.pci_param(r["ours"]["road_class"], p, r["survey"][p], r["ours"]["surface"]) for p in common]
        except (P23.Unavailable, ValueError):
            continue
        w = [P23.WEIGHTS[p] for p in common]
        part.append((sum(a * b for a, b in zip(w, o)) / sum(w), sum(a * b for a, b in zip(w, y)) / sum(w),
                     r["block"], tuple(common)))
    if part:
        sets: dict = {}
        for *_, c in part:
            sets[",".join(c)] = sets.get(",".join(c), 0) + 1
        cpart = []
        for r in rows:
            if r["half"] != "calibration":
                continue
            vals = [(P23.WEIGHTS[p], r["survey"][p], p) for p in P23.PARAMS if r["survey"][p] is not None]
            try:
                cpart.append(sum(w * P23.pci_param(r["road_class"] or "urban", p, v, r["surface"]) for w, v, p in vals)
                             / sum(w for w, _, _ in vals))
            except (P23.Unavailable, ValueError, ZeroDivisionError):
                continue
        pconst = (sum(cpart) / len(cpart)) if cpart else None
        res["partial_pci"] = metrics([(a, b, c) for a, b, c, _ in part], pconst) if pconst is not None else None
        res["partial_pci_parameter_sets"] = sets
    res["per_parameter"] = {}
    for p in P23.PARAMS:
        prs = [(r["ours"]["params"][p], r["survey"][p], r) for r in test
               if r["survey"][p] is not None and r["ours"].get("params") and r["ours"]["params"].get(p) is not None]
        n_sv = sum(1 for r in test if r["survey"][p] is not None)
        if not prs:
            res["per_parameter"][p] = {"n_survey_values": n_sv, "n_compared": 0}
            continue
        e_nat = [abs(o - y) for o, y, _ in prs]
        e_pci = [_pci_err(r["ours"]["road_class"], r["ours"]["surface"], p, o, y) for o, y, r in prs]
        e_pci = [x for x in e_pci if x is not None]
        res["per_parameter"][p] = {"n_survey_values": n_sv, "n_compared": len(prs),
                                   "mae": sum(e_nat) / len(e_nat), "unit": P23.PARAM_UNITS[p],
                                   "mae_pci_points": (sum(e_pci) / len(e_pci)) if e_pci else None,
                                   "n_pci_points": len(e_pci),
                                   "spearman": _spearman([o for o, _, _ in prs], [y for _, y, _ in prs])}
    rows_eq = [r for r in rows if r["pci"] is not None and all(r["survey"][p] is not None for p in P23.PARAMS)]
    if rows_eq:
        d = []
        for r in rows_eq:
            rc = r["road_class"] or (r.get("ours") or {}).get("road_class") or "urban"
            try:
                d.append(P23.pci(rc, r["survey"], r["surface"])["pci"] - r["pci"])
            except P23.Unavailable:
                continue
        if d:
            res["equation_check"] = {"n": len(d), "mean_diff": sum(d) / len(d),
                                     "max_abs_diff": max(abs(x) for x in d)}
    return res


# ============================================================ main
def _probe_fp() -> Optional[str]:
    from app import gis_routes as GR
    r = GR._query(lambda cur: cur.execute(
        "SELECT probe_fp, count(*) AS n FROM research.video_probe_scores GROUP BY 1 ORDER BY 2 DESC LIMIT 1"
    ).fetchone())
    return r["probe_fp"] if r else None


def _load_env(database: Optional[str]):
    env = PROJECT_ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    if database:
        os.environ["DATABASE_URL"] = re.sub(r"/[^/?]+(\?|$)", f"/{database}\\1", os.environ["DATABASE_URL"], count=1)


def fit_iri(rows: list[dict], file_sha: str, rule_sha: str, post_hoc: bool) -> dict:
    """IRI calibration candidate from the calibration half: per section, app.pci_iri.aggregate() of its segments'
    IMU features vs the survey IRI; app.pci_iri.fit_calibration (log-linear). Written only when, in-sample, it
    beats a constant (calibration-half mean IRI) in PCI_roughness points."""
    from app import pci_iri
    rs = [r for r in rows if r["half"] == "calibration" and "ours" in r and r["survey"]["iri"] is not None]
    data = []
    for r in rs:
        segrows = [dict(s["roughness"], s0_m=0.0, s1_m=s["length_m"]) for s in r["ours"]["segments"]
                   if s.get("roughness") and s["status"] != "not_covered"]
        agg = pci_iri.aggregate(segrows)
        if agg.get("rms_av") and agg.get("speed_mps"):
            data.append(dict(agg, iri_ref=r["survey"]["iri"], section_length_m=r["ours"]["report_length_m"],
                             road_class=r["ours"]["road_class"], surface=r["ours"]["surface"]))
    if len(data) < 7:
        return {"n": len(data), "fitted": False, "reason": "fewer than 7 calibration sections with IRI and IMU data"}
    lens = sorted(d["section_length_m"] for d in data)
    unit = lens[len(lens) // 2]
    try:
        cal = pci_iri.fit_calibration(data, unit_m=unit, source={
            "survey_file_sha256": file_sha, "split_rule_sha256": rule_sha, "half": "calibration",
            "sections": len(data), "post_hoc": post_hoc})
    except ValueError as e:
        return {"n": len(data), "fitted": False, "reason": str(e)}
    mean_ref = sum(d["iri_ref"] for d in data) / len(data)

    def err(pred, d):
        try:
            return abs(P23.pci_param(d["road_class"], "iri", pred, d["surface"])
                       - P23.pci_param(d["road_class"], "iri", d["iri_ref"], d["surface"]))
        except (P23.Unavailable, ValueError):
            return None
    e_fit, e_const = [], []
    for d in data:
        pred, _ = pci_iri.calibrated_iri(d, cal)
        a, b = err(pred, d) if pred is not None else None, err(mean_ref, d)
        if a is not None and b is not None:
            e_fit.append(a)
            e_const.append(b)
    if not e_fit:
        return {"n": len(data), "fitted": False, "reason": "IRI PCI not computable for these sections"}
    mf, mc = sum(e_fit) / len(e_fit), sum(e_const) / len(e_const)
    rep = {"n": len(data), "unit_m": unit, "mae_pci_in_sample": mf, "constant_mae_pci": mc,
           "r2_log": cal.get("r2_log"), "residual_sd_log": cal.get("residual_sd_log")}
    if mf >= mc:
        rep.update(fitted=False, reason="not better than a constant (in-sample): not written")
        return rep
    tmp = IRI_CAL_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(cal, indent=2), encoding="utf-8")
    os.replace(tmp, IRI_CAL_PATH)
    rep.update(fitted=True, written=str(IRI_CAL_PATH))
    return rep


def promote(file_sha: str) -> int:
    """Copy the candidate calibration to the live path, only when the ledger holds an evaluate of exactly this
    candidate (same fitted_on) on this survey file."""
    if not CAL_PATH.exists():
        raise SystemExit(f"no candidate calibration at {CAL_PATH}: run fit first")
    cand = json.loads(CAL_PATH.read_text(encoding="utf-8"))
    from app import pci_core as C
    C.validate_calibration(cand)
    ledger = [json.loads(x) for x in LEDGER.read_text(encoding="utf-8").splitlines() if x.strip()] \
        if LEDGER.exists() else []
    if not any(e.get("cmd") == "evaluate" and e.get("survey_sha256") == file_sha
               and e.get("calibration") == cand.get("fitted_on") for e in ledger):
        raise SystemExit("refused: this candidate was never evaluated on the test half of this survey file")
    tmp = LIVE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(cand, indent=2), encoding="utf-8")
    os.replace(tmp, LIVE_PATH)
    if IRI_CAL_PATH.exists():
        ic = json.loads(IRI_CAL_PATH.read_text(encoding="utf-8"))
        if (ic.get("source") or {}).get("survey_file_sha256") == file_sha:
            IRI_LIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = IRI_LIVE_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(ic, indent=2), encoding="utf-8")
            os.replace(tmp, IRI_LIVE_PATH)
            print(f"[promote] {IRI_CAL_PATH.name} -> {IRI_LIVE_PATH}")
    print(f"[promote] {CAL_PATH.name} -> {LIVE_PATH} (production reads it on the next report)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "fit", "evaluate", "promote"])
    ap.add_argument("survey", type=Path)
    ap.add_argument("--database", default=None)
    ap.add_argument("--road-class", default="urban", choices=list(P23.ROAD_CLASSES))
    ap.add_argument("--surface", default=None, choices=list(P23.SURFACES))
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="write calibration / comparison / ledger here instead of eval_results/pci (tests)")
    args = ap.parse_args()
    global OUT, LEDGER, CAL_PATH, LIVE_PATH, IRI_CAL_PATH, IRI_LIVE_PATH
    if args.out_dir:
        OUT = args.out_dir
        OUT.mkdir(parents=True, exist_ok=True)
        LEDGER = OUT / "survey_ledger.jsonl"
        CAL_PATH, LIVE_PATH = OUT / "param_calibration_candidate.json", OUT / "param_calibration.json"
        IRI_CAL_PATH, IRI_LIVE_PATH = OUT / "iri_calibration_candidate.json", OUT / "iri_calibration.json"
    if args.cmd == "promote":
        return promote(sha256_file(args.survey))
    # evaluate measures our reports WITH the candidate calibration (never whatever production has live)
    os.environ["PCI_PARAM_CALIBRATION"] = str(CAL_PATH)
    os.environ["PCI_IRI_CALIBRATION"] = str(IRI_CAL_PATH)
    _load_env(args.database)
    rule_sha = sha256_file(RULE)
    file_sha = sha256_file(args.survey)
    rows = split(read_survey(args.survey))
    fp = _probe_fp()
    none = str(OUT / "__no_calibration__.json")
    cand = json.loads(CAL_PATH.read_text(encoding="utf-8")) if CAL_PATH.exists() else None
    if args.cmd == "fit" or cand is None or cand["fitted_on"].get("survey_file_sha256") != file_sha:
        os.environ["PCI_PARAM_CALIBRATION"] = none          # fit: uncalibrated; evaluate: never another file's fit
    icand = json.loads(IRI_CAL_PATH.read_text(encoding="utf-8")) if IRI_CAL_PATH.exists() else None
    if args.cmd == "fit" or icand is None or (icand.get("source") or {}).get("survey_file_sha256") != file_sha:
        os.environ["PCI_IRI_CALIBRATION"] = none
    measure(rows, args.road_class, args.surface, fp)
    excl = [{"section_id": r["section_id"], "half": r["half"], "reason": r["excluded"]} for r in rows if "excluded" in r]
    n_c = sum(1 for r in rows if r["half"] == "calibration" and "ours" in r)
    n_t = sum(1 for r in rows if r["half"] == "test" and "ours" in r)
    print(f"[survey] {len(rows)} sections, {n_c} calibration + {n_t} test usable, {len(excl)} excluded; "
          f"file {file_sha[:12]}, rule {rule_sha[:12]}, probe_fp {fp}")
    for e in excl:
        print(f"  excluded {e['section_id']} ({e['half']}): {e['reason']}")
    if args.cmd == "check":
        return 0
    ledger = [json.loads(x) for x in LEDGER.read_text(encoding="utf-8").splitlines() if x.strip()] \
        if LEDGER.exists() else []
    if args.cmd == "fit":
        post_hoc = is_post_hoc(ledger, file_sha, test_half_sha(rows))
        f = fit(rows, fp)
        cal = {"schema": 2, "edition": P23.EDITION, "probe_fp": fp,
               "fitted_on": {"survey_file_sha256": file_sha, "split_rule_sha256": rule_sha, "half": "calibration",
                             "sections": n_c, "fitted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "post_hoc": post_hoc},
               "params": f["params"]}
        from app import pci_core as C
        C.validate_calibration(cal)
        tmp = CAL_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(cal, indent=2), encoding="utf-8")
        os.replace(tmp, CAL_PATH)
        iri_rep = fit_iri(rows, file_sha, rule_sha, post_hoc)
        f["report"]["iri"] = iri_rep
        (OUT / "param_calibration_report.json").write_text(json.dumps(f["report"], indent=2), encoding="utf-8")
        print(json.dumps(f["report"], indent=1))
        print(f"[fit] wrote {CAL_PATH}" + (" (POST HOC: this survey was already evaluated)" if post_hoc else ""))
        return 0
    cal = cand if (cand is not None and cand["fitted_on"].get("survey_file_sha256") == file_sha) else None
    if cal is None:
        print("[evaluate] WARNING: no candidate calibration fitted on this survey file; evaluating uncalibrated")
    res = evaluate(rows)
    res.update(survey_file_sha256=file_sha, split_rule_sha256=rule_sha, excluded=excl,
               calibration=(cal or {}).get("fitted_on"), evaluated_utc=datetime.now(timezone.utc).isoformat(
                   timespec="seconds"))
    (OUT / "survey_comparison.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    with open(LEDGER, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"cmd": "evaluate", "survey_sha256": file_sha, "test_half_sha256": test_half_sha(rows),
                             "utc": res["evaluated_utc"], "calibration": res["calibration"]}) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "excluded"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
