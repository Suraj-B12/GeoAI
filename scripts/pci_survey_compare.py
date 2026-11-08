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
    python scripts/pci_survey_compare.py fit      SURVEY.csv   # calibration half -> eval_results/pci/param_calibration.json
    python scripts/pci_survey_compare.py evaluate SURVEY.csv   # test half -> eval_results/pci/survey_comparison.{md,json}
Options: --database geoai (default from DATABASE_URL), --road-class urban, --surface BC.

Calibration (fit, calibration half only), per parameter the survey reports:
  - candidates: Stage 2 labels times a scale b (cracking extent, pothole runs), or the per-type probe at threshold t
    with value = a + b * f (f = fraction of the section's probe-scored length at or above t; a = 0 except rut depth);
  - t on a grid 0.05-0.95, b on a grid; the candidate with the lowest mean absolute error IN PCI_i POINTS wins;
  - its calibration-half MAE is written beside the MAE of a constant (the calibration-half median), so a
    parameter the images cannot see is visible as "not better than a constant".
  - IRI: app.pci_iri.fit_calibration (when the roughness workstream provides it) on the same half.
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
CAL_PATH = OUT / "param_calibration.json"
COLS = {"cracking": "cracking_pct", "ravelling": "ravelling_pct", "potholes": "potholes_number",
        "patching": "patching_pct", "rut_depth": "rut_depth_mm", "iri": "iri_m_km"}
BLOCK = 3
T_GRID = [round(0.05 + 0.025 * i, 3) for i in range(37)]                      # 0.05 .. 0.95
B_GRID = sorted({round(x, 4) for x in [0.05 * i for i in range(1, 61)] + [0.25 * i for i in range(13, 81)]})
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
            rows.append(row)
    if not rows:
        raise SystemExit("survey CSV has no sections")
    return rows


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
                         "probe": probe.get(sid), "roughness": pr.get("roughness")})
        q = st["quantities"]
        km = rep["km"]
        params = ({x["param"]: x.get("value_point") for x in km[0]["pci"]["params"]}
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
def _pci_err(rc, surf, p, pred, truth) -> float:
    try:
        return abs(P23.pci_param(rc, p, max(0.0, pred), surf) - P23.pci_param(rc, p, truth, surf))
    except P23.Unavailable:
        return abs(pred - truth)


def _fit_scale(xs, ys, rcs, surfs, p, a_free=False):
    """(a, b, MAE in PCI points) minimising MAE over the b grid (a = 0, or a from the median residual)."""
    best = None
    for b in B_GRID:
        a = 0.0
        if a_free:
            res = sorted(y - b * x for x, y in zip(xs, ys))
            a = res[len(res) // 2]
        mae = sum(_pci_err(rc, s, p, a + b * x, y) for x, y, rc, s in zip(xs, ys, rcs, surfs)) / len(xs)
        if best is None or mae < best[2] - 1e-12:
            best = (a, b, mae)
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
        const = sum(_pci_err(rc, s, p, med, y) for y, rc, s in zip(ys, rcs, sfs)) / len(ys)
        cands = []
        if p in ("cracking", "potholes"):
            xs = [r["ours"]["cracking_point"] if p == "cracking" else
                  r["ours"]["pothole_runs"] + C.POTHOLE_SKIP_POINT * r["ours"]["skip_m"] / C.SEGMENT_M for r in rs]
            if all(x is not None for x in xs):
                a, b, mae = _fit_scale(xs, ys, rcs, sfs, p)
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
            a, b, mae = _fit_scale(list(f_), list(y_), rc_, s_, p, a_free=(p == "rut_depth"))
            cands.append({"source": "probe", "keys": list(keys), "threshold": t, "a": a, "b": b, "mae": mae})
        if not cands:
            report[p] = {"n": len(rs), "fitted": False, "reason": "no usable candidate (missing probe scores?)"}
            continue
        best = min(cands, key=lambda c: (c["mae"], c["source"] != "labels"))
        ev = (f"fitted on {len(rs)} calibration sections of the survey: MAE {best['mae']:.1f} PCI points vs "
              f"{const:.1f} for a constant" + ("" if best["mae"] < const else " (NOT better than a constant)"))
        spec = {k: v for k, v in best.items() if k != "mae"}
        spec["evidence"] = ev
        if p == "potholes" and spec["source"] == "probe":
            spec["evidence"] += "; b per km of section"
        params[p] = spec
        report[p] = {"n": len(rs), "fitted": True, "mae_pci": round(best["mae"], 3), "constant_mae_pci": round(const, 3),
                     "better_than_constant": best["mae"] < const, "chosen": spec,
                     "candidates_tried": len(cands)}
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
    const = (sorted(calib_pci)[len(calib_pci) // 2]) if calib_pci else None
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
        res["per_parameter"][p] = {"n_survey_values": n_sv, "n_compared": len(prs),
                                   "mae": sum(e_nat) / len(e_nat), "unit": P23.PARAM_UNITS[p],
                                   "mae_pci_points": sum(e_pci) / len(e_pci),
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "fit", "evaluate"])
    ap.add_argument("survey", type=Path)
    ap.add_argument("--database", default=None)
    ap.add_argument("--road-class", default="urban", choices=list(P23.ROAD_CLASSES))
    ap.add_argument("--surface", default=None, choices=list(P23.SURFACES))
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="write calibration / comparison / ledger here instead of eval_results/pci (tests)")
    args = ap.parse_args()
    global OUT, LEDGER, CAL_PATH
    if args.out_dir:
        OUT = args.out_dir
        OUT.mkdir(parents=True, exist_ok=True)
        LEDGER, CAL_PATH = OUT / "survey_ledger.jsonl", OUT / "param_calibration.json"
        os.environ["PCI_PARAM_CALIBRATION"] = str(CAL_PATH)
    _load_env(args.database)
    rule_sha = sha256_file(RULE)
    file_sha = sha256_file(args.survey)
    rows = split(read_survey(args.survey))
    fp = _probe_fp()
    if args.cmd == "fit":
        os.environ["PCI_PARAM_CALIBRATION"] = str(OUT / "__no_calibration_while_fitting__.json")
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
        post_hoc = any(e["survey_sha256"] == file_sha and e["cmd"] == "evaluate" for e in ledger)
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
        (OUT / "param_calibration_report.json").write_text(json.dumps(f["report"], indent=2), encoding="utf-8")
        print(json.dumps(f["report"], indent=1))
        print(f"[fit] wrote {CAL_PATH}" + (" (POST HOC: this survey was already evaluated)" if post_hoc else ""))
        return 0
    cal = json.loads(CAL_PATH.read_text(encoding="utf-8")) if CAL_PATH.exists() else None
    if cal is None or cal["fitted_on"].get("survey_file_sha256") != file_sha:
        print("[evaluate] WARNING: no calibration fitted on this survey file; evaluating uncalibrated")
    res = evaluate(rows)
    res.update(survey_file_sha256=file_sha, split_rule_sha256=rule_sha, excluded=excl,
               calibration=(cal or {}).get("fitted_on"), evaluated_utc=datetime.now(timezone.utc).isoformat(
                   timespec="seconds"))
    (OUT / "survey_comparison.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    with open(LEDGER, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"cmd": "evaluate", "survey_sha256": file_sha, "utc": res["evaluated_utc"],
                             "calibration": res["calibration"]}) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "excluded"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
