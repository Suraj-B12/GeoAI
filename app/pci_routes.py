"""
A->B chainage + IRC:82-2015 condition-rating report endpoints (`geoai.pci.v1`), in the /gis router family.

    GET  /gis/pci/meta                    categories, limits, data bbox, data_as_of (for the WebGIS form)
    POST /gis/pci/path                    path GeoJSON, snapped points, length, passes, gaps   (stateless, 30/min)
    POST /gis/pci/report                  full report JSON (stateless preview)                 (10/min)
    POST /gis/pci/report.pdf | .csv       the same, rendered (stateless download)              (10/min)
    POST /gis/pci/reports                 SAVE a snapshot (report JSON + PDF bytes): X-API-Key = an operator key
                                          (API_KEYS); hard cap 200 stored; dedupe by input hash  (10/min)
    GET  /gis/pci/reports                 list of stored reports (samples first)               (60/min)
    GET  /gis/pci/reports/{id}            stored JSON + stale flag + segments changed since data_as_of
    GET  /gis/pci/reports/{id}.pdf|.csv|.geojson   the stored PDF bytes (frozen) / CSV / GeoJSON

Request body (JSON; unknown fields are refused):
    a, b: {lat, lon} (required); via: [{lat, lon}] (<= 8); category: urban | highway | rural (default urban);
    lane_width_m 2-30 (default 3.5); chainage_at_a_km 0-3000 with chainage_increases_towards_b true/false
    (both or neither); road_name, road_number, surface_type, prepared_by, title: text <= 120 chars,
    printed as "entered by requester, unverified".

Rate limits are per client: the Cf-Connecting-Ip header when the TCP peer is loopback (the Cloudflare
tunnel connects from 127.0.0.1), else the peer address. Own small sliding-window limiter, so this module does
not import app.main (the model server).

Reports read the same rows the map shows (app.gis_routes._fetch_segment_rows / _segment_feature) through the
cached graph of app.pci_chainage. Storage: research.pci_reports (migration 009, applied on first use).
"""
from __future__ import annotations

import asyncio
import csv
import hmac
import io
import ipaddress
import json
import math
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from app import pci_chainage as PC
from app import pci_core as C

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MIGRATION = PROJECT_ROOT / "migrations" / "009_pci_reports.sql"

router = APIRouter(prefix="/gis/pci", tags=["gis-pci"])

MAX_STORED = 200
MAX_TEXT = 120
BBOX_MARGIN_KM = 1.0
_JSON = "application/json"
_NO_CACHE = {"Cache-Control": "no-cache"}
_FIELDS = {"a", "b", "via", "category", "lane_width_m", "chainage_at_a_km", "chainage_increases_towards_b",
           "road_name", "road_number", "surface_type", "prepared_by", "title", "surface", "entered"}
# values a requester may enter for the whole stretch (IRC:82-2023 parameters): (min, max)
ENTERED_LIMITS = {"cracking": (0.0, 100.0), "ravelling": (0.0, 100.0), "patching": (0.0, 100.0),
                  "potholes": (0.0, 1000.0), "rut_depth": (0.0, 200.0), "iri": (0.5, 30.0)}
LIMITS = {"path": (30, 60.0), "report": (10, 60.0), "save": (10, 60.0), "read": (60, 60.0)}
KEY_FAILURES = (5, 600.0)          # wrong operator keys per client per 10 min before 429


# ============================================================ rate limit
class RateLimiter:
    """Sliding window per (bucket, client)."""

    def __init__(self):
        self._hits: dict[tuple, deque] = {}
        self._lock = threading.Lock()

    def check(self, bucket: str, key: str, n: int, window: float) -> Optional[float]:
        """None when allowed (and counted), else seconds until the next slot."""
        now = time.monotonic()
        with self._lock:
            q = self._hits.setdefault((bucket, key), deque())
            while q and now - q[0] >= window:
                q.popleft()
            if len(q) >= n:
                return max(0.1, window - (now - q[0]))
            q.append(now)
            if len(self._hits) > 10000:                 # forget idle clients
                for k in [k for k, v in self._hits.items() if not v]:
                    self._hits.pop(k, None)
            return None

    def recent(self, bucket: str, key: str, window: float) -> tuple[int, float]:
        """(hits in the window, seconds until the oldest leaves it) without counting a new hit."""
        now = time.monotonic()
        with self._lock:
            q = self._hits.setdefault((bucket, key), deque())
            while q and now - q[0] >= window:
                q.popleft()
            return len(q), (window - (now - q[0])) if q else 0.0

    def add(self, bucket: str, key: str) -> None:
        with self._lock:
            self._hits.setdefault((bucket, key), deque()).append(time.monotonic())

    def reset(self):
        with self._lock:
            self._hits.clear()


limiter = RateLimiter()


def client_key(request: Request) -> str:
    host = request.client.host if request.client else ""
    try:
        ip = ipaddress.ip_address(host)
        if getattr(ip, "ipv4_mapped", None):
            ip = ip.ipv4_mapped
        loop = ip.is_loopback
    except ValueError:
        loop = False
    if loop:
        cf = (request.headers.get("cf-connecting-ip") or "").strip()
        return f"cf:{cf}" if cf else "local"
    return f"ip:{host}"


def _limit(request: Request, bucket: str) -> None:
    n, window = LIMITS[bucket]
    wait = limiter.check(bucket, client_key(request), n, window)
    if wait is not None:
        raise HTTPException(status_code=429, detail=f"rate limit: {n} per {int(window)} s for this client",
                            headers={"Retry-After": str(int(math.ceil(wait)))})


# ============================================================ validation
def _finite(v, name) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        raise HTTPException(status_code=422, detail=f"{name} must be a number")
    try:
        f = float(v)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"{name} must be a number") from None
    if not math.isfinite(f):
        raise HTTPException(status_code=422, detail=f"{name} must be finite")
    return f


def _point(v, name, bbox) -> dict:
    if not isinstance(v, dict) or set(v) - {"lat", "lon"} or "lat" not in v or "lon" not in v:
        raise HTTPException(status_code=422, detail=f"{name} must be {{lat, lon}}")
    lat, lon = _finite(v["lat"], f"{name}.lat"), _finite(v["lon"], f"{name}.lon")
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise HTTPException(status_code=422, detail=f"{name} is not a valid position")
    if bbox:
        w, s, e, n = bbox
        dlat = BBOX_MARGIN_KM / 111.32
        dlon = dlat / max(0.1, math.cos(math.radians((s + n) / 2)))
        if not (s - dlat <= lat <= n + dlat and w - dlon <= lon <= e + dlon):
            raise HTTPException(status_code=422, detail=f"{name} is outside the surveyed area (+{BBOX_MARGIN_KM:.0f} km)")
    return {"lat": lat, "lon": lon}


def validate_request(body: Any, bbox: Optional[list]) -> dict:
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON object expected")
    unknown = sorted(set(body) - _FIELDS)
    if unknown:
        raise HTTPException(status_code=400, detail=f"unknown field(s): {', '.join(unknown)}; "
                                                    f"allowed: {', '.join(sorted(_FIELDS))}")
    out: dict = {"a": _point(body.get("a"), "a", bbox), "b": _point(body.get("b"), "b", bbox)}
    via = body.get("via") or []
    if not isinstance(via, list) or len(via) > PC.MAX_VIA:
        raise HTTPException(status_code=422, detail=f"via must be a list of at most {PC.MAX_VIA} points")
    out["via"] = [_point(v, f"via[{i}]", bbox) for i, v in enumerate(via)]
    cat = body.get("category") or "urban"
    if cat not in C.CATEGORIES:
        raise HTTPException(status_code=422, detail=f"category must be one of {', '.join(C.CATEGORIES)}")
    out["category"] = cat
    if body.get("lane_width_m") not in (None, ""):
        w = _finite(body["lane_width_m"], "lane_width_m")
        if not 2.0 <= w <= 30.0:
            raise HTTPException(status_code=422, detail="lane_width_m must be within 2-30 m")
        out["lane_width_m"] = w
    at_a, inc = body.get("chainage_at_a_km"), body.get("chainage_increases_towards_b")
    if at_a not in (None, ""):
        km = _finite(at_a, "chainage_at_a_km")
        if not 0.0 <= km <= 3000.0:
            raise HTTPException(status_code=422, detail="chainage_at_a_km must be within 0-3000")
        if not isinstance(inc, bool):
            raise HTTPException(status_code=422, detail="with chainage_at_a_km, say whether road chainage increases "
                                                        "towards B (chainage_increases_towards_b: true/false)")
        out["chainage_at_a_km"], out["chainage_increases_towards_b"] = km, inc
    elif inc not in (None, ""):
        if not isinstance(inc, bool):
            raise HTTPException(status_code=422, detail="chainage_increases_towards_b must be true or false")
    surf = body.get("surface")
    if surf not in (None, ""):
        from app import pci_irc2023 as P23
        if surf not in P23.SURFACES:
            raise HTTPException(status_code=422, detail=f"surface must be one of {', '.join(P23.SURFACES)}")
        out["surface"] = surf
        if cat in ("mdr", "rural") and surf not in (P23.IRI_MDR if cat == "mdr" else P23.IRI_RURAL):
            raise HTTPException(status_code=422, detail=f"IRC:82-2023 gives no roughness equation for {surf} on "
                                                        f"{cat} roads; surface must be one of "
                                                        + ", ".join(P23.IRI_MDR if cat == "mdr" else P23.IRI_RURAL))
    ent = body.get("entered")
    if ent not in (None, "", {}):
        if not isinstance(ent, dict) or set(ent) - set(ENTERED_LIMITS):
            raise HTTPException(status_code=422, detail="entered must be {parameter: value} with parameters among "
                                                        + ", ".join(ENTERED_LIMITS))
        vals = {}
        for k, v in ent.items():
            if v in (None, ""):
                continue
            x = _finite(v, f"entered.{k}")
            lo, hi = ENTERED_LIMITS[k]
            if not lo <= x <= hi:
                raise HTTPException(status_code=422, detail=f"entered.{k} must be within {lo:g}-{hi:g}")
            vals[k] = x
        if vals:
            out["entered"] = vals
    for f in PC.TEXT_FIELDS:
        v = body.get(f)
        if v in (None, ""):
            continue
        if not isinstance(v, str):
            raise HTTPException(status_code=422, detail=f"{f} must be text")
        v = " ".join(v.split())
        if len(v) > MAX_TEXT:
            raise HTTPException(status_code=422, detail=f"{f} is longer than {MAX_TEXT} characters")
        import unicodedata
        if any(unicodedata.category(ch) in ("Cc", "Cf", "Co", "Cs", "Cn") for ch in v):
            raise HTTPException(status_code=422, detail=f"{f} contains control, format (e.g. right-to-left override, "
                                                        f"zero-width) or unassigned characters")
        from app.pci_report_pdf import unsupported_text
        bad = unsupported_text(v)
        if bad:
            raise HTTPException(status_code=422, detail=f"{f}: {bad}")
        if v:
            out[f] = v
    return out


async def _body(request: Request) -> Any:
    try:
        return await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        raise HTTPException(status_code=400, detail="request body must be JSON") from None


# ============================================================ data access
def _graph() -> PC.Graph:
    from app.gis_routes import _Unavailable
    try:
        return PC.load_graph()
    except _Unavailable as e:
        raise HTTPException(status_code=503, detail=f"survey data unavailable: {e}") from None
    except PC.PciError as e:
        raise HTTPException(status_code=503, detail=str(e)) from None


async def _graph_async() -> PC.Graph:
    return await asyncio.to_thread(_graph)


def _report_sync(req: dict, report_id=None) -> dict:
    g = _graph()
    try:
        return PC.build_report(g, req, report_id=report_id)
    except PC.PciError as e:
        raise HTTPException(status_code=422, detail=str(e)) from None


def _json(obj: Any, media: str = _JSON, headers: Optional[dict] = None) -> Response:
    return Response(content=json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
                    media_type=media, headers={**_NO_CACHE, **(headers or {})})


def image_fetcher(segment_id: str) -> Optional[bytes]:
    """A 640 px JPEG preview of a segment's stored frame (shared disk cache with /gis/image), or None."""
    from app import gis_routes as GR
    try:
        found = GR.resolve_segment_image(segment_id)
        if not found:
            return None
        path, _media, sha = found
        return GR.make_preview(path, 640, sha)
    except Exception as e:   # an unreadable image must not break the report
        print(f"[PCI] annex image {segment_id}: {type(e).__name__}")
        return None


def render_pdf(report: dict) -> bytes:
    from app.pci_report_pdf import render
    return render(report, image_fetcher=image_fetcher)


# ============================================================ CSV / GeoJSON
_CSV_DANGER = ("=", "+", "-", "@", "\t", "\r")


def csv_cell(v) -> Any:
    """Spreadsheet-safe cell: text starting with = + - @ (or tab/CR) gets a leading apostrophe."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.4f}".rstrip("0").rstrip(".") if math.isfinite(v) else ""
    if isinstance(v, (list, tuple)):
        v = "; ".join(str(x) for x in v)
    v = str(v)
    return "'" + v if v.startswith(_CSV_DANGER) else v


CSV_PARAMS = (("cracking", "cracking_pct"), ("ravelling", "ravelling_pct"), ("potholes", "potholes_no"),
              ("patching", "patching_pct"), ("rut_depth", "rut_depth_mm"), ("iri", "iri_m_per_km"))
CSV_COLUMNS = (("unit", "index", "chainage_from", "chainage_to", "from_m", "to_m", "road_chainage_from",
                "road_chainage_to", "start_lat", "start_lon", "end_lat", "end_lon", "length_m", "imaged_pct",
                "coverage_status")
               + tuple(c for p, n in CSV_PARAMS for c in (n, n + "_lo", n + "_hi", p + "_pci_i", p + "_source"))
               + ("pci_lo", "pci_hi", "pci_single", "pci_class", "recommendation_table_5_5", "missing_parameters")
               + ("pci_est", "pci_est_lo90", "pci_est_hi90", "class_est", "recommendation_est")
               + tuple(c for p, _n in CSV_PARAMS for c in (p + "_est", p + "_est_lo90", p + "_est_hi90",
                                                           p + "_est_basis"))
               + ("pothole_frames", "impact_felt", "jolt"))


def _csv_est_inputs(rep: dict, m0: float, m1: float) -> dict:
    """{param: (point, lo90, hi90)} over the 100 m sections inside [m0, m1] that have an estimate:
    length-weighted means, potholes summed (same as the PDF's km / stretch rows)."""
    items = []
    for sec in rep["sections"]:
        e = (sec.get("pci") or {}).get("estimate") or {}
        if e.get("estimate") and sec["from_m"] >= m0 - 1e-6 and sec["to_m"] <= m1 + 1e-6:
            items.append((e.get("inputs") or {}, sec["to_m"] - sec["from_m"]))
    out = {}
    for p, _n in CSV_PARAMS:
        xs = [(i[p], wt) for i, wt in items if p in i]
        if not xs:
            continue
        if p == "potholes":
            out[p] = tuple(sum(x[k] for x, _ in xs) for k in ("point", "lo90", "hi90"))
        else:
            W = sum(wt for _, wt in xs)
            out[p] = tuple(sum(x[k] * wt for x, wt in xs) / W for k in ("point", "lo90", "hi90"))
    return out


def _csv_lonlat(rep: dict, m: float) -> tuple:
    """(lat, lon) on the report path at measure m (metres from A)."""
    v = rep["path"]["vertices"]
    for (x1, y1, a), (x2, y2, b) in zip(v, v[1:]):
        if a <= m <= b:
            t = 0.0 if b == a else (m - a) / (b - a)
            return f"{y1 + t * (y2 - y1):.7f}", f"{x1 + t * (x2 - x1):.7f}"
    return f"{v[-1][1]:.7f}", f"{v[-1][0]:.7f}"


def report_csv(rep: dict) -> str:
    """IRC:82-2023 condition report as CSV: '#' comment lines (edition, road, surface, headline PCI), then one row
    per 100 m section, per km row and for the whole stretch. Parameter values: point estimate + lo/hi (gate-skip
    allowance), PCI_i (point) and source; a parameter that is not assessed has empty values and source
    'not_assessed'. Whole stretch: values over the full length; PCI and PCI_i = length-weighted mean of the rated
    km rows (the report's headline)."""
    from app import pci_irc2023 as P23
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\r\n")
    h, s = rep["header"], rep["summary"]
    hp = s.get("pci") or {}
    head_pci = ("insufficient data" if not hp else
                (f"{hp['single']:.1f} ({hp['single_class']})" if hp.get("single") is not None
                 else f"{hp['lo']:.1f} to {hp['hi']:.1f} ({hp['class']}); no single PCI: not assessed "
                      + ", ".join(hp.get("missing") or [])))
    for k, v in (("report", "Road condition report - IRC:82-2023 Pavement Condition Index (screening estimate "
                            "from phone video)"),
                 ("edition", h.get("edition") or P23.EDITION), ("report_id", rep.get("report_id")),
                 ("road_name (as entered)", h.get("road_name")), ("road_number (as entered)", h.get("road_number")),
                 ("road_class", h.get("category")),
                 ("surface", f"{h.get('surface')} {h.get('surface_name') or ''} ({h.get('surface_note') or ''})"),
                 ("A lat/lon", f"{h['a']['snapped_lat']:.7f} {h['a']['snapped_lon']:.7f}"),
                 ("B lat/lon", f"{h['b']['snapped_lat']:.7f} {h['b']['snapped_lon']:.7f}"),
                 ("length_m", s["length_m"]), ("survey_dates", ", ".join(h.get("survey_dates_ist") or [])),
                 ("data_as_of", rep.get("data_as_of")),
                 ("whole-stretch PCI estimate", "insufficient data" if not s.get("pci_estimate") else
                  f"{s['pci_estimate']['median']:.1f} (90 % interval {s['pci_estimate']['lo90']:.1f} to "
                  f"{s['pci_estimate']['hi90']:.1f}), {s['pci_estimate']['class']} (median); an estimate, not a "
                  "measurement (eval_results/pci/ESTIMATE.md)"),
                 ("whole-stretch PCI worst case", head_pci),
                 ("note", "one lane imaged; cracking % = share of surveyed LENGTH (upper bound of % area); potholes "
                          "= runs of consecutive pothole-labelled 5 m frames; chainage measured from A"),
                 ("columns", "<param> = best estimate (gate-skipped frames at the audited rate); _lo = classified "
                             "frames only (potholes_no_lo = runs counted); _hi = skipped frames at the upper 95 % "
                             "rate; <param>_pci_i at the best estimate; empty = not assessed (bracketed 0-100 in "
                             "pci_lo / pci_hi = the strict worst case). pci_est* / <param>_est* = the PCI estimate "
                             "and its inputs (median / point and 90 % interval; km and stretch rows: length-weighted "
                             "over the 100 m sections, potholes summed). unit=pothole rows: one run of pothole frames; "
                             "impact_felt = the scooter's accelerometer recorded a jolt there")):
        w.writerow(["# " + k, csv_cell(v)])
    w.writerow(CSV_COLUMNS)
    kms = [k for k in rep["km"] if ((k.get("pci") or {}).get("pci"))]
    kw = sum(k["to_m"] - k["from_m"] for k in kms) or 1.0

    def row(kind, idx, u, whole=False):
        q = u["quantities"]
        pb = u.get("pci") or {}
        pc = (hp or None) if whole else pb.get("pci")
        prm = {p["param"]: p for p in pb.get("params") or []}
        a_lat, a_lon = _csv_lonlat(rep, u["from_m"])
        b_lat, b_lon = _csv_lonlat(rep, u["to_m"])
        vals = [kind, idx, u["chainage_from"], u["chainage_to"], u["from_m"], u["to_m"],
                (u.get("road_chainage_from") or {}).get("text"), (u.get("road_chainage_to") or {}).get("text"),
                a_lat, a_lon, b_lat, b_lon, q["length_m"], round(100.0 * q["coverage"], 1), q["coverage_status"]]
        for p, _n in CSV_PARAMS:
            r = prm.get(p) or {}
            src = r.get("source") or ("not_assessed" if pc else None)
            v = r.get("value_point") if r.get("value_point") is not None else r.get("value_lo")
            if src == "not_assessed":
                v = None
            pci_i = r.get("pci_point") if r.get("known") else None
            if whole:
                ks = [((k.get("pci") or {}).get("params") or []) for k in kms]
                rows_p = [next((x for x in ps if x["param"] == p), {}) for ps in ks]
                pci_i = (sum(x["pci_point"] * (k["to_m"] - k["from_m"]) for x, k in zip(rows_p, kms)) / kw
                         if kms and all(x.get("known") for x in rows_p) else None)
            vals += [v, None if v is None else r.get("value_lo"), None if v is None else r.get("value_hi"), pci_i,
                     src]
        if pc:
            single = pc.get("single")
            cls = pc.get("single_class") if single is not None else pc.get("class")
            vals += [pc.get("lo"), pc.get("hi"), single, cls,
                     P23.RECOMMENDATION.get(pc.get("single_class")) if single is not None else None,
                     ", ".join(pc.get("missing") or [])]
        else:
            vals += [None, None, None, "insufficient data", None, None]
        # PCI estimate (median, 90 % interval) and its inputs
        eb = pb.get("estimate") or {}
        est = s.get("pci_estimate") if whole else (eb.get("estimate") if "inputs" in eb else (eb or None))
        if est:
            vals += [est.get("median"), est.get("lo90"), est.get("hi90"), est.get("class"),
                     est.get("recommendation")]
        else:
            vals += [None, None, None, None if not pc else "no estimate", None]
        ins = _csv_est_inputs(rep, u["from_m"], u["to_m"]) if est else {}
        basis = (eb.get("basis") or {}) if kind == "100m" else {}
        for p, _n in CSV_PARAMS:
            x = ins.get(p)
            vals += [x[0], x[1], x[2], basis.get(p)] if x else [None, None, None, None]
        vals += [None, None, None]
        w.writerow([csv_cell(x) for x in vals])
    for sec in rep["sections"]:
        row("100m", sec["index"], sec)
    for k in rep["km"]:
        row("km", k["km"], k)
    row("stretch", "", rep["stretch"], whole=True)
    blank = len(CSV_COLUMNS) - 15 - 3
    for j, r in enumerate(s.get("pothole_runs_list") or [], start=1):
        a_lat, a_lon = _csv_lonlat(rep, r["from_m"])
        b_lat, b_lon = _csv_lonlat(rep, r["to_m"])
        felt = r.get("impact_felt")
        vals = (["pothole", j, r.get("chainage"), None, r["from_m"], r["to_m"], None, None, a_lat, a_lon, b_lat,
                 b_lon, round(r["to_m"] - r["from_m"] + 5.0, 1), None, None] + [None] * blank
                + [r.get("segments"), None if felt is None else ("yes" if felt else "no"), r.get("jolt")])
        w.writerow([csv_cell(x) for x in vals])
    return "﻿" + buf.getvalue()


def report_geojson(rep: dict) -> dict:
    feats = []
    for s in rep["sections"]:
        r = s["rating"]
        f = r.get("final") or {}
        feats.append({"type": "Feature", "geometry": s["geometry"], "properties": {
            "kind": "section", "index": s["index"], "chainage_from": s["chainage_from"],
            "chainage_to": s["chainage_to"], "class": f.get("class") or "Insufficient data",
            "class_lo": f.get("class_lo"), "class_hi": f.get("class_hi"),
            "bands": {p["param"]: p["band"] for p in r.get("params", []) if p.get("assessed")},
            "coverage": s["quantities"]["coverage"]}})
    feats.append({"type": "Feature", "geometry": rep["path"]["geometry"]["gaps"], "properties": {"kind": "gap"}})
    for name, p in (("A", rep["header"]["a"]), ("B", rep["header"]["b"])):
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [p["snapped_lon"], p["snapped_lat"]]},
                      "properties": {"kind": "endpoint", "name": name}})
    for a in rep["annex"]:
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [a["lon"], a["lat"]]},
                      "properties": {"kind": "annex_frame", **{k: a[k] for k in ("segment_id", "chainage", "reason")}}})
    return {"type": "FeatureCollection", "schema": PC.SCHEMA, "report_id": rep.get("report_id"), "features": feats}


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ============================================================ storage
_schema_ready = False
_schema_lock = threading.Lock()


def ensure_schema(conn) -> None:
    global _schema_ready
    if _schema_ready:
        return
    with _schema_lock:
        if _schema_ready:
            return
        if conn.execute("SELECT to_regclass('research.pci_reports')").fetchone()[0] is None:
            from scripts.setup_local_db import statements_without_outer_transaction
            body = statements_without_outer_transaction(MIGRATION.read_text(encoding="utf-8-sig"))
            with conn.transaction():
                conn.execute(body)
        _schema_ready = True


def _connect():
    from app import video_store
    from app.gis_routes import CONNECT_TIMEOUT_S
    import psycopg
    try:
        return video_store.connect(connect_timeout=CONNECT_TIMEOUT_S)
    except (RuntimeError, psycopg.OperationalError) as e:
        raise HTTPException(status_code=503, detail=f"database unavailable: {type(e).__name__}") from None


def save_report(req: dict, *, sample: bool = False, sample_rule: Optional[str] = None,
                cap: Any = "default", graph: Optional[PC.Graph] = None) -> dict:
    """Build + render + store one report snapshot. Dedupe by input hash. Returns {id, created, ...}.
    `sample` is set only by scripts/pci_samples.py (samples do not count towards the cap).
    cap: "default" = MAX_STORED at call time; None = no cap."""
    if cap == "default":
        cap = MAX_STORED
    from psycopg.types.json import Jsonb
    g = graph or _graph()
    canon = PC.canonical_request(req)
    h = PC.input_hash(canon, g.data_as_of, sample)
    conn = _connect()
    try:
        ensure_schema(conn)
        row = conn.execute("SELECT id, created_at FROM research.pci_reports WHERE input_hash = %s", (h,)).fetchone()
        if row:
            return {"id": row[0], "created": False, "input_hash": h, "created_at": PC._iso(row[1])}
        if cap is not None and not sample:
            n = conn.execute("SELECT count(*) FROM research.pci_reports WHERE NOT sample").fetchone()[0]
            if n >= cap:
                raise HTTPException(status_code=409, detail=f"report store is full ({cap} saved reports); "
                                                            f"ask the operator to remove old ones")
        rid = conn.execute("SELECT nextval(pg_get_serial_sequence('research.pci_reports', 'id'))").fetchone()[0]
        try:
            rep = PC.build_report(g, req, report_id=rid)
        except PC.PciError as e:
            raise HTTPException(status_code=422, detail=str(e)) from None
        pdf = render_pdf(rep)
        title = canon.get("title") or canon.get("road_name") or \
            f"{rep['summary']['chainage_a']} to {rep['summary']['chainage_b']} ({rep['header']['category']})"
        with conn.transaction():
            ins = conn.execute(
                "INSERT INTO research.pci_reports (id, title, input, input_hash, report, pdf, sample, sample_rule, "
                "data_as_of) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (input_hash) DO NOTHING "
                "RETURNING id, created_at",
                (rid, title, Jsonb(canon), h, Jsonb(rep), pdf, bool(sample), sample_rule,
                 g.data_as_of or PC._iso(datetime.now(timezone.utc)))).fetchone()
        if ins is None:      # a concurrent save of the same input won
            row = conn.execute("SELECT id, created_at FROM research.pci_reports WHERE input_hash = %s", (h,)).fetchone()
            return {"id": row[0], "created": False, "input_hash": h, "created_at": PC._iso(row[1])}
        return {"id": ins[0], "created": True, "input_hash": h, "created_at": PC._iso(ins[1]), "title": title}
    finally:
        conn.close()


def _load(rid: int, cols: str) -> dict:
    from psycopg.rows import dict_row
    conn = _connect()
    try:
        ensure_schema(conn)
        with conn.cursor(row_factory=dict_row) as cur:
            row = cur.execute(f"SELECT {cols} FROM research.pci_reports WHERE id = %s", (rid,)).fetchone()
    finally:
        conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="no such report")
    return row


def _bands(summary) -> dict:
    hl = (summary or {}).get("headline") or {}
    return {p: (hl.get(p) or {}).get("band") for p in ("potholes", "cracking")}


def headline_text(summary) -> str:
    """Saved-report list line: IRC:82-2023 PCI (single or range) and class, then length."""
    s = summary or {}
    est = s.get("pci_estimate")
    if est:
        return (f"PCI ≈ {est['median']:.0f} (90 % {est['lo90']:.0f}–{est['hi90']:.0f}) · {est['class']} · "
                f"{'≥ ' if s.get('gaps') else ''}{s.get('length_m', 0):.0f} m")
    pci = s.get("pci")
    if pci:
        v = (f"PCI {pci['single']:.0f}" if pci.get("single") is not None
             else f"PCI {pci['lo']:.0f}–{pci['hi']:.0f}")
        cls = pci.get("single_class") or pci.get("class")
        return f"{v} · {cls} · {'≥ ' if s.get('gaps') else ''}{s.get('length_m', 0):.0f} m"
    hl = s.get("headline") or {}
    if not hl:
        return f"{s.get('length_m', 0):.0f} m · insufficient data"
    rpk = hl.get("pothole_runs_per_km")
    parts = [f"Potholes {(hl.get('potholes') or {}).get('band', '–')}"
             + (f" ({rpk:.1f} runs/km)" if isinstance(rpk, (int, float)) else ""),
             f"Cracking {(hl.get('cracking') or {}).get('band', '–')}",
             f"{'≥ ' if s.get('gaps') else ''}{s.get('length_m', 0):.0f} m"]
    f = s.get("final") or {}
    if f.get("class"):
        parts.append(f"overall, secondary: {f['class']}")
    return " · ".join(parts)


def _list() -> list[dict]:
    from psycopg.rows import dict_row
    conn = _connect()
    try:
        ensure_schema(conn)
        with conn.cursor(row_factory=dict_row) as cur:
            rows = cur.execute(
                "SELECT id, created_at, title, sample, sample_rule, data_as_of, input, "
                "report->'summary' AS summary, report->'header'->>'category' AS category "
                "FROM research.pci_reports ORDER BY sample DESC, id DESC LIMIT %s", (MAX_STORED + 50,)).fetchall()
    finally:
        conn.close()
    return [{"id": r["id"], "created_at": PC._iso(r["created_at"]), "title": r["title"], "sample": r["sample"],
             "sample_rule": r["sample_rule"], "data_as_of": PC._iso(r["data_as_of"]), "category": r["category"],
             "headline_text": headline_text(r["summary"]), "bands": _bands(r["summary"]),
             "overall_secondary": ((r["summary"] or {}).get("final") or {}).get("class"),
             "input": r["input"], "summary": r["summary"],
             "links": {"json": f"/gis/pci/reports/{r['id']}", "pdf": f"/gis/pci/reports/{r['id']}.pdf",
                       "csv": f"/gis/pci/reports/{r['id']}.csv", "geojson": f"/gis/pci/reports/{r['id']}.geojson"}}
            for r in rows]


def _check_key(request: Request) -> None:
    """Operator key check, with a failed-key bucket per client (REVIEW_backend #2): after KEY_FAILURES wrong
    keys in the window every further try is 429 until the window passes, so the key cannot be guessed."""
    from app import security
    keys = security.VALID_API_KEYS
    given = request.headers.get("x-api-key") or ""
    if not keys:
        raise HTTPException(status_code=403, detail="saving reports needs an operator API key, and none is "
                                                    "configured on this server (API_KEYS)")
    who = client_key(request)
    n, wait = limiter.recent("key_fail", who, KEY_FAILURES[1])
    if n >= KEY_FAILURES[0]:
        raise HTTPException(status_code=429, detail="too many wrong operator keys from this client; try later",
                            headers={"Retry-After": str(int(math.ceil(wait)) or 1)})
    if not given or not any(hmac.compare_digest(given.encode(), k.encode()) for k in keys):
        limiter.add("key_fail", who)
        print(f"[PCI] wrong operator key on save from {who} ({n + 1} in {int(KEY_FAILURES[1])} s)")
        raise HTTPException(status_code=401, detail="saving a report needs the operator API key (X-API-Key)")


# ============================================================ endpoints
@router.get("/meta")
async def pci_meta(request: Request):
    _limit(request, "read")
    g = await _graph_async()
    return _json({"schema": PC.SCHEMA, "name": C.NAME, "data_as_of": g.data_as_of, "bbox": g.bbox,
                  "bbox_margin_km": BBOX_MARGIN_KM,
                  "categories": [{"id": k, "name": PC.CATEGORY_NAMES[k], "table": v["table"], "title": v["title"],
                                  "params": list(v["params"])} for k, v in C.TABLES.items()],
                  "default_category": "urban", "default_lane_width_m": C.DEFAULT_LANE_WIDTH_M,
                  "limits": {"via_max": PC.MAX_VIA, "snap_max_m": PC.SNAP_MAX_M, "min_path_m": PC.MIN_PATH_M,
                             "max_path_m": PC.MAX_PATH_M, "explicit_gap_max_m": PC.EXPLICIT_GAP_MAX_M,
                             "lane_width_m": [2, 30], "chainage_at_a_km": [0, 3000], "text_max": MAX_TEXT,
                             "stored_max": MAX_STORED,
                             "rate_per_minute": {k: v[0] for k, v in LIMITS.items()}},
                  "assessment": C.ASSESSMENT, "fields": sorted(_FIELDS),
                  "irc2023": _meta_2023()})


def _meta_2023() -> dict:
    """IRC:82-2023 facts the WebGIS form needs (classes, parameters with units/limits, surfaces per road class).
    Surfaces per class = those with a roughness entry in Tables 5.1-5.3 / an Appendix-2 IRI equation (SMA has
    no equation and is not offered; rural roads have no BC equation)."""
    from app import pci_irc2023 as P23
    surf = {"highway": ["BC", "SDBC"], "urban": ["BC", "SDBC"],
            "mdr": [s for s in P23.SURFACES if s in P23.IRI_MDR], "rural": [s for s in P23.SURFACES if s in P23.IRI_RURAL]}
    return {"edition": P23.EDITION, "name": P23.NAME,
            "road_classes": [{"id": k, "name": PC.CATEGORY_NAMES[k], "surfaces": surf[k],
                              "default_surface": P23.DEFAULT_SURFACE[k]} for k in P23.ROAD_CLASSES],
            "surface_names": dict(P23.SURFACE_NAMES),
            "params": [{"id": p, "name": P23.PARAM_NAMES[p], "unit": P23.PARAM_UNITS[p], "weight": P23.WEIGHTS[p],
                        "entered_limits": list(ENTERED_LIMITS[p])} for p in P23.PARAMS],
            "classes": [{"name": n, "above": lo, "recommendation": P23.RECOMMENDATION[n]} for lo, n in P23.CLASSES]
                       + [{"name": "Fail", "above": None, "recommendation": P23.RECOMMENDATION["Fail"]}],
            "curves": _curves_2023(surf)}


_CURVES: dict = {}


def _curves_2023(surf: dict) -> dict:
    """PCI_i sampled from app.pci_irc2023.pci_param (the one implementation) so the WebGIS can show PCI_i at an
    estimate by linear interpolation. {"x": {grid: [...]}, "grid": {param: grid}, "y": {road class: {param: [...]}}}
    with roughness per surface ({"iri": {surface: [...]}}); y = null where the printed equation is not usable."""
    if _CURVES:
        return _CURVES
    from app import pci_irc2023 as P23

    def grid(a, b, step):
        n = int(round((b - a) / step))
        return [round(a + i * step, 4) for i in range(n + 1)]
    xs = {"pct": grid(0, 5, .1) + grid(5.5, 30, .5) + grid(32, 100, 2),
          "count": grid(0, 10, .1) + grid(10.5, 30, .5) + grid(32, 100, 2),
          "mm": grid(0, 50, .5), "iri": grid(.5, 10, .05) + grid(10.1, 30, .1)}
    gridof = {"cracking": "pct", "ravelling": "pct", "patching": "pct", "potholes": "count", "rut_depth": "mm",
              "iri": "iri"}

    def ys(cls, p, s):
        out = []
        for v in xs[gridof[p]]:
            try:
                out.append(round(P23.pci_param(cls, p, v, s), 2))
            except ValueError:            # Unavailable is a ValueError: printed equation not usable
                out.append(None)
        return out
    _CURVES.update(x=xs, grid=gridof, y={cls: {p: ({s: ys(cls, p, s) for s in surf[cls]} if p == "iri"
                                                    else ys(cls, p, None)) for p in P23.PARAMS}
                                         for cls in P23.ROAD_CLASSES})
    return _CURVES
    from app import pci_irc2023 as P23

    def grid(a, b, step):
        n = int(round((b - a) / step))
        return [round(a + i * step, 4) for i in range(n + 1)]
    xs = {"pct": grid(0, 5, .1) + grid(5.5, 30, .5)[0:] + grid(32, 100, 2),
          "potholes": grid(0, 10, .1) + grid(10.5, 30, .5) + grid(32, 100, 2),
          "rut_depth": grid(0, 50, .5),
          "iri": grid(.5, 10, .05) + grid(10.1, 30, .1)}

    def curve(cls, p, s, x):
        y = []
        for v in x:
            try:
                y.append(round(P23.pci_param(cls, p, v, s), 3))
            except ValueError:            # Unavailable is a ValueError: printed equation not usable
                y.append(None)
        return {"x": x, "y": y}
    for cls in P23.ROAD_CLASSES:
        d = {}
        for p in P23.PARAMS:
            if p == "iri":
                d["iri"] = {s: curve(cls, p, s, xs["iri"]) for s in surf[cls]}
            else:
                d[p] = curve(cls, p, None, xs["pct"] if p in ("cracking", "ravelling", "patching") else xs[p])
        _CURVES[cls] = d
    return _CURVES


@router.post("/path")
async def pci_path(request: Request):
    _limit(request, "path")
    body = await _body(request)
    g = await _graph_async()
    req = validate_request(body, g.bbox)
    try:
        doc = await asyncio.to_thread(PC.path_document, g, req)
    except PC.PciError as e:
        raise HTTPException(status_code=422, detail=str(e)) from None
    out = PC.public(doc)
    out.pop("vertices", None)
    out.pop("steps", None)
    return _json(PC._clean(out))


async def _stateless_report(request: Request) -> dict:
    _limit(request, "report")
    body = await _body(request)
    g = await _graph_async()
    req = validate_request(body, g.bbox)
    return await asyncio.to_thread(_report_sync, req)


@router.post("/report")
async def pci_report(request: Request):
    return _json(await _stateless_report(request))


@router.post("/report.pdf")
async def pci_report_pdf(request: Request):
    rep = await _stateless_report(request)
    pdf = await asyncio.to_thread(render_pdf, rep)
    return Response(content=pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="irc82_condition_{_stamp()}.pdf"',
                             **_NO_CACHE})


@router.post("/report.csv")
async def pci_report_csv(request: Request):
    rep = await _stateless_report(request)
    return Response(content=report_csv(rep).encode("utf-8"), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="irc82_condition_{_stamp()}.csv"',
                             **_NO_CACHE})


@router.post("/reports")
async def pci_save(request: Request):
    _limit(request, "save")          # before the key check: wrong keys count against the rate limit too
    _check_key(request)
    body = await _body(request)
    g = await _graph_async()
    req = validate_request(body, g.bbox)
    out = await asyncio.to_thread(save_report, req, graph=g)
    rid = out["id"]
    out["links"] = {"json": f"/gis/pci/reports/{rid}", "pdf": f"/gis/pci/reports/{rid}.pdf"}
    return _json(out, headers={})


@router.get("/reports")
async def pci_list(request: Request):
    _limit(request, "read")
    rows = await asyncio.to_thread(_list)
    return _json({"schema": PC.SCHEMA, "count": len(rows), "reports": rows})


def _parse_name(name: str) -> tuple[int, str]:
    base, _, ext = name.partition(".")
    if not base.isdigit() or len(base) > 12 or ext not in ("", "json", "pdf", "csv", "geojson"):
        raise HTTPException(status_code=404, detail="no such report")
    return int(base), ext or "json"


@router.get("/reports/{name}")
async def pci_stored(name: str, request: Request):
    _limit(request, "read")
    rid, ext = _parse_name(name)
    if ext == "pdf":
        row = await asyncio.to_thread(_load, rid, "pdf")
        return Response(content=bytes(row["pdf"]), media_type="application/pdf",
                        headers={"Content-Disposition": f'attachment; filename="irc82_condition_report_{rid}.pdf"',
                                 "Cache-Control": "public, max-age=86400"})
    row = await asyncio.to_thread(_load, rid, "id, created_at, title, sample, sample_rule, data_as_of, report")
    rep = row["report"]
    if ext == "csv":
        return Response(content=report_csv(rep).encode("utf-8"), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="irc82_condition_report_{rid}.csv"',
                                 **_NO_CACHE})
    if ext == "geojson":
        return _json(report_geojson(rep), "application/geo+json")
    as_of = row["data_as_of"].astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
    try:
        changed = await asyncio.to_thread(PC.changed_since, rep.get("segment_ids") or [], as_of)
    except Exception as e:      # the stored report is still served when the check fails
        print(f"[PCI] stale check failed: {type(e).__name__}")
        changed = None
    return _json({"schema": PC.SCHEMA, "id": row["id"], "created_at": PC._iso(row["created_at"]),
                  "title": row["title"], "sample": row["sample"], "sample_rule": row["sample_rule"],
                  "data_as_of": as_of, "stale": bool(changed), "changed_segments": changed,
                  "stale_note": ("labels or reviews of segments in this report changed after it was generated; "
                                 "the stored report is a snapshot and was not updated") if changed else None,
                  "report": rep})
