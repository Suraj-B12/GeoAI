"""
The stdlib-only core of the read-only GIS API (app/gis_routes.py): query filters, the
aggregates (summary, per-km, drive track), the FeatureCollection and CSV builders.

Standard library only, on purpose: scripts/export_gis_sample.py copies this file into the
portable sample bundle, where the bundle's mock_server.py imports it to answer the same query
strings on the teammate's machine (no database, no GPU). The API and the mock therefore filter
and aggregate with the SAME code, and a WebGIS built against the mock behaves identically
against the real server (scripts/tests/test_gis_routes.py compares the two).

Everything here works on a feature's flattened `properties` (the GeoJSON the API returns),
never on database rows:

    parse_filters(query, layer)  -> Filters      (FilterError = HTTP 400 on a bad value)
    apply(features, filters)     -> (page, numberMatched)
    feature_collection(...), segments_csv(...), aggregate(...), per_km(...), track_geometry(...)

Filter semantics (one rule each; the handoff document repeats them):
  session        any of (video only)                       properties.session_id
  date_from/to   ISO date or date-time, UTC when naive;    video: utc (frame time)
                 from inclusive, to exclusive; a plain     citizen: created_at (upload time)
                 date as `to` includes that whole day
  label          any of the 18 IRC:82-2015 names + Normal  properties.final_labels
  severity       any of None/Low/Medium/High/Unknown       properties.final_severity
  status         any of send/audit/skip/not_covered        properties.status (video only)
  vlm_class      any of VLM_CLASSES                        properties.vlm_class
  answered       true/false (video only)                   properties.answered
  reviewed       true/false                                properties.expert_reviewed
  needs_review   true/false                                properties.needs_expert_review
  min_confidence 0..1                                      properties.confidence (null never passes)
  bbox           minLon,minLat,maxLon,maxLat (RFC 7946)    video: centre_lon/centre_lat
                                                           citizen: longitude/latitude
  include_review true/false (citizen only, default false)  false: assessment_status classified/done only;
                                                           true: also expert_review
  limit, offset  page of the matching features, in the API's order

`final_*` = the expert's correction when one exists, else the VLM's answer.
Parameters starting with "_" are ignored (cache busters); any other unknown
parameter is an error, so a typo never silently returns everything.
"""

from __future__ import annotations

import csv
import io
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Optional

SCHEMA_VERSION = "geoai.gis.v1"

# The VLM's answer vocabulary: GeoAI_Video pipeline/geoai/config.py VlmConfig.labels (same order),
# i.e. the 18 vision-only IRC:82-2015 types of scripts/irc82_taxonomy.py plus "Normal".
LABELS: tuple[str, ...] = (
    "Bleeding", "Streaking", "Hungry Surface", "Hairline Cracks", "Alligator Cracking",
    "Longitudinal Cracking", "Transverse Cracking", "Edge Cracking", "Slippage", "Rutting",
    "Corrugation", "Shoving", "Shallow Depression", "Settlement", "Stripping", "Ravelling",
    "Potholes", "Edge Breaking", "Normal",
)
DISTRESS_LABELS: tuple[str, ...] = LABELS[:-1]

# Which label colours a marker when a segment / photo has several: the first one in this
# order. A display convention (structural failures before surface defects), not an IRC ranking.
MARKER_PRIORITY: tuple[str, ...] = (
    "Potholes", "Edge Breaking", "Ravelling", "Stripping", "Alligator Cracking", "Settlement",
    "Shallow Depression", "Rutting", "Shoving", "Corrugation", "Slippage", "Edge Cracking",
    "Transverse Cracking", "Longitudinal Cracking", "Hairline Cracks", "Bleeding", "Hungry Surface",
    "Streaking", "Normal",
)

SEVERITIES: tuple[str, ...] = ("None", "Low", "Medium", "High", "Unknown")
STATUSES: tuple[str, ...] = ("send", "audit", "skip", "not_covered")
# distress / normal / unknown: answered (unknown = answered without a valid label);
# pending: sent to the VLM, no answer stored yet; not_sent: gate said skip; not_covered: never imaged.
VLM_CLASSES: tuple[str, ...] = ("distress", "normal", "unknown", "pending", "not_sent", "not_covered")

DEFAULT_LIMIT = 5000
MAX_LIMIT = 20000

_LAYER_PARAMS = {
    "video": {"session", "date_from", "date_to", "label", "severity", "status", "vlm_class", "answered",
              "reviewed", "needs_review", "min_confidence", "bbox", "limit", "offset"},
    "citizen": {"date_from", "date_to", "label", "severity", "vlm_class", "reviewed", "needs_review",
                "min_confidence", "bbox", "limit", "offset", "include_review"},
}
_LAYER_KEYS = {"video": {"time": "utc", "lat": "centre_lat", "lon": "centre_lon"},
               "citizen": {"time": "created_at", "lat": "latitude", "lon": "longitude"}}


class FilterError(ValueError):
    """A query parameter that can't be used (the API answers 400 with this message)."""


@dataclass(frozen=True)
class Filters:
    layer: str
    sessions: Optional[frozenset] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    labels: Optional[frozenset] = None
    severities: Optional[frozenset] = None
    statuses: Optional[frozenset] = None
    vlm_classes: Optional[frozenset] = None
    answered: Optional[bool] = None
    reviewed: Optional[bool] = None
    needs_review: Optional[bool] = None
    min_confidence: Optional[float] = None
    bbox: Optional[tuple] = None
    include_review: bool = False
    limit: int = DEFAULT_LIMIT
    offset: int = 0
    keys: dict = field(default_factory=dict, compare=False)

    def echo(self) -> dict:
        """The filters in force, JSON-ready (returned with every response)."""
        out = {}
        for name in ("sessions", "labels", "severities", "statuses", "vlm_classes"):
            v = getattr(self, name)
            if v is not None:
                out[name] = sorted(v)
        for name in ("date_from", "date_to"):
            v = getattr(self, name)
            if v is not None:
                out[name] = iso_utc(v)
        for name in ("answered", "reviewed", "needs_review", "min_confidence", "bbox"):
            v = getattr(self, name)
            if v is not None:
                out[name] = list(v) if isinstance(v, tuple) else v
        if self.layer == "citizen":
            out["include_review"] = self.include_review
        out["limit"], out["offset"] = self.limit, self.offset
        return out


# ============================================================
# Small helpers (also used by gis_routes)
# ============================================================

def iso_utc(dt: Optional[datetime]) -> Optional[str]:
    """2026-10-26T11:44:08.930Z (UTC, milliseconds), or None."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_time(s: Any) -> Optional[datetime]:
    """An ISO date-time string from a feature (…Z or …+00:00) as an aware datetime; None if absent/bad."""
    if not isinstance(s, str) or not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def canonical_label(name: str) -> Optional[str]:
    """The LABELS spelling of a label (case-insensitive), or None."""
    return _LABEL_LOOKUP.get(str(name).strip().lower())


_LABEL_LOOKUP = {lab.lower(): lab for lab in LABELS}


def marker_label(labels: Iterable[str]) -> Optional[str]:
    """The label that colours the marker: first of MARKER_PRIORITY present; else the first label; else None."""
    present = {canonical_label(x) or x for x in labels or []}
    for lab in MARKER_PRIORITY:
        if lab in present:
            return lab
    labels = list(labels or [])
    return labels[0] if labels else None


# ============================================================
# Parsing
# ============================================================

def _split(v: str) -> list[str]:
    return [x.strip() for x in str(v).split(",") if x.strip()]


def _choices(name: str, raw: str, allowed: tuple[str, ...], canon=None) -> frozenset:
    out = set()
    lookup = {a.lower(): a for a in allowed}
    for x in _split(raw):
        c = canon(x) if canon else lookup.get(x.lower())
        if c is None or c not in allowed:
            raise FilterError(f"{name}={x!r} is not one of: {', '.join(allowed)}")
        out.add(c)
    if not out:
        raise FilterError(f"{name} is empty")
    return frozenset(out)


def _bool(name: str, raw: str) -> bool:
    s = str(raw).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    raise FilterError(f"{name}={raw!r} is not true/false")


def _int(name: str, raw: str, lo: int, hi: int) -> int:
    try:
        v = int(str(raw).strip())
    except ValueError:
        raise FilterError(f"{name}={raw!r} is not an integer") from None
    if not lo <= v <= hi:
        raise FilterError(f"{name}={v} is outside {lo}..{hi}")
    return v


def _date(name: str, raw: str, end: bool) -> datetime:
    """ISO date or date-time; naive = UTC. A plain date as the END of a range covers that whole day."""
    s = str(raw).strip()
    try:
        if len(s) == 10:
            d = date.fromisoformat(s)
            dt = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
            return dt + timedelta(days=1) if end else dt
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        raise FilterError(f"{name}={raw!r} is not an ISO date (YYYY-MM-DD) or date-time") from None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _bbox(raw: str) -> tuple:
    parts = _split(raw)
    try:
        v = tuple(float(x) for x in parts)
    except ValueError:
        raise FilterError(f"bbox={raw!r}: four numbers expected (minLon,minLat,maxLon,maxLat)") from None
    if len(v) != 4:
        raise FilterError(f"bbox={raw!r}: four numbers expected (minLon,minLat,maxLon,maxLat)")
    w, s, e, n = v
    if not (-180 <= w <= e <= 180 and -90 <= s <= n <= 90):
        raise FilterError(f"bbox={raw!r}: need -180<=minLon<=maxLon<=180 and -90<=minLat<=maxLat<=90 (lon first)")
    return v


def parse_filters(query: Mapping[str, Any], layer: str = "video", *, allow: Iterable[str] = ()) -> Filters:
    """Filters from a query mapping (FastAPI request.query_params, or a dict of str).

    `allow`: extra parameter names the endpoint handles itself (e.g. geom), accepted here unread.
    """
    if layer not in _LAYER_PARAMS:
        raise ValueError(f"layer {layer!r}")
    known = _LAYER_PARAMS[layer] | set(allow)
    unknown = sorted(k for k in query.keys() if k not in known and not str(k).startswith("_"))
    if unknown:
        raise FilterError(f"unknown parameter(s) for the {layer} layer: {', '.join(unknown)}; "
                          f"allowed: {', '.join(sorted(known))}")

    def get(name):
        v = query.get(name)
        return None if v is None or str(v).strip() == "" else str(v)

    kw: dict = {"layer": layer, "keys": _LAYER_KEYS[layer]}
    if get("session"):
        kw["sessions"] = frozenset(_split(get("session")))
    if get("date_from"):
        kw["date_from"] = _date("date_from", get("date_from"), end=False)
    if get("date_to"):
        kw["date_to"] = _date("date_to", get("date_to"), end=True)
    if kw.get("date_from") and kw.get("date_to") and kw["date_from"] >= kw["date_to"]:
        raise FilterError("date_from must be before date_to")
    if get("label"):
        kw["labels"] = _choices("label", get("label"), LABELS, canon=canonical_label)
    if get("severity"):
        kw["severities"] = _choices("severity", get("severity"), SEVERITIES)
    if get("status"):
        kw["statuses"] = _choices("status", get("status"), STATUSES)
    if get("vlm_class"):
        kw["vlm_classes"] = _choices("vlm_class", get("vlm_class"), VLM_CLASSES)
    for name in ("answered", "reviewed", "needs_review"):
        if get(name):
            kw[name] = _bool(name, get(name))
    if get("include_review"):
        kw["include_review"] = _bool("include_review", get("include_review"))
    if get("min_confidence"):
        try:
            c = float(get("min_confidence"))
        except ValueError:
            raise FilterError(f"min_confidence={get('min_confidence')!r} is not a number") from None
        if not 0.0 <= c <= 1.0:
            raise FilterError("min_confidence must be within 0..1")
        kw["min_confidence"] = c
    if get("bbox"):
        kw["bbox"] = _bbox(get("bbox"))
    kw["limit"] = _int("limit", get("limit"), 1, MAX_LIMIT) if get("limit") else DEFAULT_LIMIT
    kw["offset"] = _int("offset", get("offset"), 0, 10_000_000) if get("offset") else 0
    return Filters(**kw)


# ============================================================
# Matching
# ============================================================

def matches(p: Mapping[str, Any], f: Filters) -> bool:
    """Does one feature's properties pass every filter?"""
    keys = f.keys or _LAYER_KEYS[f.layer]
    if f.layer == "citizen" and not f.include_review and p.get("assessment_status") == "expert_review":
        return False
    if f.sessions is not None and p.get("session_id") not in f.sessions:
        return False
    if f.date_from is not None or f.date_to is not None:
        t = parse_time(p.get(keys["time"]))
        if t is None or (f.date_from is not None and t < f.date_from) or (f.date_to is not None and t >= f.date_to):
            return False
    if f.labels is not None and not (f.labels & {canonical_label(x) for x in (p.get("final_labels") or [])}):
        return False
    if f.severities is not None and p.get("final_severity") not in f.severities:
        return False
    if f.statuses is not None and p.get("status") not in f.statuses:
        return False
    if f.vlm_classes is not None and p.get("vlm_class") not in f.vlm_classes:
        return False
    if f.answered is not None and bool(p.get("answered")) != f.answered:
        return False
    if f.reviewed is not None and bool(p.get("expert_reviewed")) != f.reviewed:
        return False
    if f.needs_review is not None and bool(p.get("needs_expert_review")) != f.needs_review:
        return False
    if f.min_confidence is not None:
        c = p.get("confidence")
        if not isinstance(c, (int, float)) or c < f.min_confidence:
            return False
    if f.bbox is not None:
        lon, lat = p.get(keys["lon"]), p.get(keys["lat"])
        if not isinstance(lon, (int, float)) or not isinstance(lat, (int, float)):
            return False
        w, s, e, n = f.bbox
        if not (w <= lon <= e and s <= lat <= n):
            return False
    return True


def apply(features: Iterable[Mapping], f: Filters) -> tuple[list, int]:
    """(the requested page of matching features, number of matching features)."""
    hit = [ft for ft in features if matches(ft.get("properties") or {}, f)]
    return hit[f.offset:f.offset + f.limit], len(hit)


# ============================================================
# Values and geometry
# ============================================================

PREVIEW_WIDTHS: tuple[int, ...] = (160, 320, 480, 640, 960, 1280, 1600, 1920)   # ?w= snaps up to one of these


def snap_width(w: int) -> int:
    """The smallest allowed preview width >= w (the largest one for anything bigger)."""
    for allowed in PREVIEW_WIDTHS:
        if w <= allowed:
            return allowed
    return PREVIEW_WIDTHS[-1]


def num(x, nd: Optional[int] = None) -> Optional[float]:
    """A finite float (rounded to nd decimals), else None. Accepts numbers and numeric strings."""
    if isinstance(x, bool) or x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return round(v, nd) if nd is not None else v


def lonlat(lon, lat) -> Optional[list]:
    """[lon, lat] rounded to 7 decimals (~1 cm), or None when either is missing / out of range."""
    lon, lat = num(lon), num(lat)
    if lon is None or lat is None or not (-180 <= lon <= 180 and -90 <= lat <= 90):
        return None
    return [round(lon, 7), round(lat, 7)]


def _coords(geom: Optional[Mapping]) -> Iterable[list]:
    if not geom:
        return
    t, c = geom.get("type"), geom.get("coordinates")
    if t == "Point":
        yield c
    elif t in ("LineString", "MultiPoint"):
        yield from c
    elif t in ("Polygon", "MultiLineString"):
        for part in c:
            yield from part
    elif t == "MultiPolygon":
        for poly in c:
            for part in poly:
                yield from part


def bbox_of(features: Iterable[Mapping]) -> Optional[list]:
    """[minLon, minLat, maxLon, maxLat] over every coordinate of the features' geometries."""
    xs, ys = [], []
    for f in features:
        for x, y in _coords(f.get("geometry")):
            xs.append(x)
            ys.append(y)
    return [min(xs), min(ys), max(xs), max(ys)] if xs else None


def now_utc() -> str:
    return iso_utc(datetime.now(timezone.utc))


def feature_collection(features: list, total: int, f: Filters, layer: str, *, generated_at: Optional[str] = None,
                       **extra) -> dict:
    """RFC 7946 FeatureCollection with foreign members (schema, counts, the filters in force, bbox)."""
    return {"type": "FeatureCollection", "schema": SCHEMA_VERSION, "layer": layer,
            "generated_at": generated_at or now_utc(), "numberMatched": total, "numberReturned": len(features),
            "filters": f.echo(), **extra, "bbox": bbox_of(features), "features": features}


# ============================================================
# CSV
# ============================================================

CSV_COLUMNS: tuple[str, ...] = (
    "segment_id", "session_id", "k", "status", "vlm_class", "answered", "assessment_status",
    "final_labels", "final_primary_label", "marker_label", "final_severity", "labels", "primary_label", "severity",
    "confidence", "stage1_label", "stage1_confidence", "stage2_confidence", "needs_expert_review",
    "expert_reviewed", "expert_corrected_types", "expert_corrected_severity", "description",
    "condition_indicators", "utc", "local_time", "centre_lat", "centre_lon", "position_sigma_m",
    "s0_m", "s1_m", "length_m", "partial", "chainage_start_km", "chainage_end_km",
    "road_id", "road_name", "direction", "lane", "camera_lat", "camera_lon", "camera_heading_deg",
    "camera_speed_kmh", "camera_h_sigma_m", "gate_mode", "gate_decision", "gate_decision_if_gated",
    "gate_reasons", "gate_probe_frac", "gate_evidence", "gate_crack_mm", "quality_sharpness",
    "quality_blur_px", "quality_reason", "low_light", "image_url",
)


def segments_csv(features: Iterable[Mapping]) -> str:
    """CSV text with a UTF-8 BOM (Excel), CRLF lines; lists joined with '; ', booleans true/false,
    missing values empty."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\r\n")
    w.writerow(CSV_COLUMNS)
    for ft in features:
        p = ft["properties"]
        row = []
        for col in CSV_COLUMNS:
            v = p.get(col)
            if isinstance(v, list):
                v = "; ".join(str(x) for x in v)
            elif isinstance(v, bool):
                v = "true" if v else "false"
            elif v is None:
                v = ""
            row.append(v)
        w.writerow(row)
    return "﻿" + buf.getvalue()


# ============================================================
# Aggregates (GeoAI_Video pipeline/geoai/analyze.py definitions)
# ============================================================

def _length(p: Mapping) -> float:
    return num(p.get("length_m")) or 0.0


def pct(part: float, whole: float) -> Optional[float]:
    """Percent (0-100, 3 decimals), None when the whole is 0."""
    return round(100.0 * part / whole, 3) if whole > 0 else None


def valid_distress(p: Mapping) -> list[str]:
    """The segment's distinct IRC distress labels (final labels, canonical spelling, Normal excluded)."""
    seen = []
    for x in p.get("final_labels") or []:
        c = canonical_label(x)
        if c and c != "Normal" and c not in seen:
            seen.append(c)
    return seen


def aggregate(props: list[Mapping]) -> dict:
    """Lengths and counts of a set of video segments.

    surveyed = covered (send, audit, skip); not_covered is reported separately;
    classified = answered with class distress or normal. Lengths are real lengths (a partial last
    segment counts with its own). A segment with several labels counts once per label, so label
    rows don't add up to distress_m. Labels are the FINAL labels (expert correction when present).
    """
    m = defaultdict(float)
    by_status, by_class, by_sev, by_astatus = Counter(), Counter(), Counter(), Counter()
    lab_n, lab_m = Counter(), defaultdict(float)
    for p in props:
        length = _length(p)
        by_status[p["status"]] += 1
        by_class[p["vlm_class"]] += 1
        m["route_m"] += length
        m[f"{p['vlm_class']}_m"] += length
        if p["status"] == "not_covered":         # its length is in not_covered_m (via vlm_class)
            continue
        m["surveyed_m"] += length
        if p["status"] in ("send", "audit"):
            m["sent_m"] += length
        if p["vlm_class"] in ("distress", "normal"):
            m["classified_m"] += length
        if p["answered"]:
            by_sev[p["final_severity"]] += 1
            by_astatus[p["assessment_status"] or "no_assessment"] += 1
        for lab in valid_distress(p) if p["vlm_class"] == "distress" else []:
            lab_n[lab] += 1
            lab_m[lab] += length
        if p["vlm_class"] == "normal":
            lab_n["Normal"] += 1
            lab_m["Normal"] += length
    surveyed, classified = m["surveyed_m"], m["classified_m"]
    labels = [{"label": lab, "segments": lab_n[lab], "length_m": round(lab_m[lab], 2),
               "pct_of_surveyed": pct(lab_m[lab], surveyed), "pct_of_classified": pct(lab_m[lab], classified)}
              for lab in LABELS]
    return {
        "segments": len(props),
        "segments_by_status": {s: by_status[s] for s in STATUSES},
        "segments_by_vlm_class": {c: by_class[c] for c in VLM_CLASSES},
        "answered": sum(1 for p in props if p["answered"]),
        "answered_by_assessment_status": dict(sorted(by_astatus.items())),
        "answered_by_final_severity": {s: by_sev[s] for s in SEVERITIES},
        "needs_expert_review": sum(1 for p in props if p["needs_expert_review"]),
        "expert_reviewed": sum(1 for p in props if p["expert_reviewed"]),
        "with_image": sum(1 for p in props if p["has_image"]),
        "route_m": round(m["route_m"], 2), "surveyed_m": round(surveyed, 2),
        "not_covered_m": round(m["not_covered_m"], 2), "sent_m": round(m["sent_m"], 2),
        "classified_m": round(classified, 2), "distress_m": round(m["distress_m"], 2),
        "normal_m": round(m["normal_m"], 2), "unknown_m": round(m["unknown_m"], 2),
        "pending_m": round(m["pending_m"], 2), "not_sent_m": round(m["not_sent_m"], 2),
        "pct_distress_of_surveyed": pct(m["distress_m"], surveyed),
        "pct_distress_of_classified": pct(m["distress_m"], classified),
        "labels": labels,
    }


def summary_document(features: list[Mapping], f: Filters, *, generated_at: Optional[str] = None) -> dict:
    """/gis/video/summary: aggregate() over EVERY feature matching f (limit/offset ignored)."""
    hit = [ft for ft in features if matches(ft.get("properties") or {}, f)]
    props = [ft["properties"] for ft in hit]
    times = sorted(p["utc"] for p in props if p.get("utc"))
    return {"schema": SCHEMA_VERSION, "generated_at": generated_at or now_utc(), "filters": f.echo(),
            "sessions": len({p["session_id"] for p in props}), **aggregate(props),
            "time_range": {"from": times[0] if times else None, "to": times[-1] if times else None},
            "bbox": bbox_of(hit)}


def per_km(props: list[Mapping], bin_m: float = 1000.0) -> list[dict]:
    """1 km stretches of ONE session from its first segment (analyze.per_km): a segment straddling a
    boundary is split by length, so every column sums to the session total."""
    props = [p for p in props if num(p.get("s0_m")) is not None and num(p.get("s1_m")) is not None]
    if not props:
        return []
    base = min(p["s0_m"] for p in props)
    end = max(p["s1_m"] for p in props)
    n = max(1, int(math.ceil((end - base) / bin_m - 1e-9)))
    acc = [defaultdict(float) for _ in range(n)]
    for p in props:
        a, b = p["s0_m"] - base, p["s1_m"] - base
        keys = ["not_covered_m"] if p["status"] == "not_covered" else ["surveyed_m", f"{p['vlm_class']}_m"]
        if p["status"] != "not_covered" and p["vlm_class"] in ("distress", "normal"):
            keys.append("classified_m")
        if p["vlm_class"] == "distress":
            keys += [f"label:{lab}" for lab in valid_distress(p)]
        for i in range(int(a // bin_m), min(n, int(math.ceil(b / bin_m - 1e-12)))):
            ov = min(b, (i + 1) * bin_m) - max(a, i * bin_m)
            if ov > 0:
                for key in keys:
                    acc[i][key] += ov
    rows = []
    for i, d in enumerate(acc):
        row = {"km": i, "from_m": round(i * bin_m, 2), "to_m": round(min((i + 1) * bin_m, end - base), 2)}
        for key in ("surveyed_m", "not_covered_m", "classified_m", "distress_m", "normal_m", "unknown_m",
                    "pending_m", "not_sent_m"):
            row[key] = round(d[key], 2)
        row["labels_m"] = {lab: round(d[f"label:{lab}"], 2) for lab in DISTRESS_LABELS if d[f"label:{lab}"] > 0}
        row["pct_distress_of_surveyed"] = pct(d["distress_m"], d["surveyed_m"])
        row["pct_distress_of_classified"] = pct(d["distress_m"], d["classified_m"])
        rows.append(row)
    return rows


def track_geometry(props: list[Mapping]) -> Optional[dict]:
    """The drive as a line through segment centres in route order (k); a new part starts where k
    jumps or a centre is missing. LineString, MultiLineString when there are gaps, None if < 2 points."""
    parts, cur, last_k = [], [], None
    for p in sorted(props, key=lambda p: p["k"]):
        pt = lonlat(p.get("centre_lon"), p.get("centre_lat"))
        if pt is None or (last_k is not None and p["k"] != last_k + 1):
            if len(cur) >= 2:
                parts.append(cur)
            cur = []
        if pt is not None and (not cur or cur[-1] != pt):
            cur.append(pt)
        last_k = p["k"]
    if len(cur) >= 2:
        parts.append(cur)
    if not parts:
        return None
    if len(parts) == 1:
        return {"type": "LineString", "coordinates": parts[0]}
    return {"type": "MultiLineString", "coordinates": parts}
