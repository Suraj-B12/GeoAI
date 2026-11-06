"""
A->B chainage over the drive-video survey network and the per-unit attribution of the IRC:82-2015 report.

Spec: eval_results/pci/DESIGN.md (v2) §1 (chainage), §2 (units), §8 (report content). The arithmetic of
the rating is app/pci_core.py; this module decides WHICH segments feed each unit and at what chainage.

Graph (nodes = segments with a position; undirected; weight w, true length `length`):
  track   consecutive k within a drive                                   w = distance
  same    other drive, distance < 6 m, |dheading| < 30 deg or > 150 deg  w = distance + 50 (switch cost)
  cross   other drive, distance < 8 m, 30 <= |dheading| <= 150           w = distance + 50
          (no link when |d alt_ellipsoid_m| > 4 m: possible flyover / grade separation)
  gap     end of a drive -> start of the time-next drive of the same device, dt < 10 min
                                                                         w = 1.5 x straight line, flagged
  explicit gap (only on a leg that starts or ends at a via point, when no surveyed path exists):
          drive end -> drive start <= 300 m, straight line, flagged the same way.

Path = Dijkstra through A, via points (<= 8), B. Chainage = cumulative distance between the path's segment
centres (NO smoothing); a gap counts its straight line ("at least x m"). Passes = runs of one drive on the
path. Segments of other drives join a path pass only through same-road links, projected into a +-30 m measure
window around the linked node; same travel direction only (opposite-direction ones are listed, not rated).
Per 100 m section the pass with the most classified length is used (ties: latest date), never averaged.

build_report(graph, request) -> the report JSON (everything the PDF / CSV / GeoJSON render).
load_graph() -> the cached Graph of the server's database (rebuilt when segments or reviews change).
"""
from __future__ import annotations

import bisect
import hashlib
import json
import math
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Optional

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from app import pci_core as C

SCHEMA = "geoai.pci.v1"
R_EARTH = 6371008.8
IST = timezone(timedelta(hours=5, minutes=30))

SAME_ROAD_M = 6.0
CROSS_M = 8.0
SAME_DH = 30.0
OPPOSITE_DH = 150.0
MAX_DALT_M = 4.0
SWITCH_COST_M = 50.0
GAP_FACTOR = 1.5
GAP_MAX_DT_S = 600.0
EXPLICIT_GAP_MAX_M = 300.0
SNAP_MAX_M = 40.0
MIN_PATH_M = 50.0
MAX_PATH_M = 25000.0
MAX_VIA = 8
WINDOW_M = 30.0
SAME_SESSION_MIN_DK = 12          # a same-drive revisit (loop / U-turn) must be >= 60 m away in k
FILL_DK = 3                       # holes of <= 3 segments inside an attached run are filled
SECTION_M = 100.0
TAIL_MERGE_M = 50.0
ANNEX_MAX = 24

ACCURACY = ("Segment position ±10 m on surveyed runs (phone GNSS σ 4–6 m, receiver latency not corrected, 5 m "
            "segments); unknown extra error after each gap. A labelled distress may lie up to ~30 m ahead of the "
            "marked segment in the travel direction (forward-looking camera).")


class PciError(ValueError):
    """A request that cannot be answered (HTTP 422 with this message)."""


# ============================================================ geometry helpers
def angdiff(a: float, b: float) -> float:
    """|a - b| on the circle, 0..180 degrees."""
    d = abs((a - b) % 360.0)
    return 360.0 - d if d > 180.0 else d


def bearing_xy(dx: float, dy: float) -> float:
    """Compass bearing (0 = north, 90 = east) of a local (east, north) vector."""
    return math.degrees(math.atan2(dx, dy)) % 360.0


def fmt_chainage(m: float) -> str:
    """'Km 1+234' from metres (rounded to the metre)."""
    m = int(round(max(0.0, float(m))))
    return f"Km {m // 1000}+{m % 1000:03d}"


def road_chainage(m: float, at_a_km: Optional[float], increases: Optional[bool]) -> Optional[dict]:
    """Calibrated road chainage at measure m, or None when not calibrated."""
    if at_a_km is None or increases is None:
        return None
    km = float(at_a_km) + (m / 1000.0 if increases else -m / 1000.0)
    metres = int(round(km * 1000.0))
    if metres < 0:
        return {"km": round(km, 3), "text": f"{km:.3f} (before Km 0)"}
    return {"km": round(metres / 1000.0, 3), "text": f"Km {metres // 1000}+{metres % 1000:03d} ({metres / 1000.0:.3f})"}


def season_of(d: datetime) -> str:
    """IMD season of an IST date (IRC §5.3 asks for pre- and post-monsoon surveys)."""
    m = d.month
    if m in (1, 2):
        return "winter (Jan–Feb)"
    if m in (3, 4, 5):
        return "pre-monsoon (Mar–May)"
    if m in (6, 7, 8, 9):
        return "SW monsoon (Jun–Sep)"
    return "post-monsoon (Oct–Dec; NE monsoon in Karnataka)"


# ============================================================ graph
@dataclass
class Node:
    idx: int
    segment_id: str
    session_id: str
    k: int
    lat: float
    lon: float
    x: float
    y: float
    heading: Optional[float]
    alt: Optional[float]
    utc: Optional[datetime]
    device: str
    props: dict
    weather: Optional[str] = None
    image_sha256: Optional[str] = None


@dataclass
class Graph:
    nodes: list
    G: Any
    tree: Any
    lat0: float
    lon0: float
    bbox: list                      # [minLon, minLat, maxLon, maxLat] of the nodes
    key: Any = None
    data_as_of: Optional[str] = None
    gaps: list = field(default_factory=list)
    ends: set = field(default_factory=set)
    by_sk: dict = field(default_factory=dict)
    sessions: dict = field(default_factory=dict)     # session_id -> {pipeline_version, config_sha256}

    def xy(self, lat: float, lon: float) -> tuple[float, float]:
        return (R_EARTH * math.radians(lon - self.lon0) * math.cos(math.radians(self.lat0)),
                R_EARTH * math.radians(lat - self.lat0))

    def dist(self, i: int, j: int) -> float:
        a, b = self.nodes[i], self.nodes[j]
        return math.hypot(a.x - b.x, a.y - b.y)


def _num(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _time(v) -> Optional[datetime]:
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    if isinstance(v, str) and v:
        try:
            t = datetime.fromisoformat(v.replace("Z", "+00:00"))
            return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def build_graph(records: Iterable[Mapping], *, key=None, data_as_of: Optional[str] = None,
                sessions: Optional[dict] = None) -> Graph:
    """Graph from records: {"props": GIS properties, "alt": alt_ellipsoid_m, "device": id, "weather": str,
    "image_sha256": str}. Records without a centre position are not nodes."""
    recs = []
    for r in records:
        p = r["props"]
        lat, lon = _num(p.get("centre_lat")), _num(p.get("centre_lon"))
        if lat is None or lon is None:
            continue
        recs.append((r, p, lat, lon))
    if not recs:
        raise PciError("no surveyed segments with a position")
    lat0 = sum(x[2] for x in recs) / len(recs)
    lon0 = sum(x[3] for x in recs) / len(recs)
    g = Graph(nodes=[], G=nx.Graph(), tree=None, lat0=lat0, lon0=lon0, bbox=[], key=key, data_as_of=data_as_of,
              sessions=sessions or {})
    for r, p, lat, lon in recs:
        x, y = g.xy(lat, lon)
        n = Node(idx=len(g.nodes), segment_id=str(p["segment_id"]), session_id=str(p["session_id"]), k=int(p["k"]),
                 lat=lat, lon=lon, x=x, y=y, heading=_num(p.get("camera_heading_deg")), alt=_num(r.get("alt")),
                 utc=_time(p.get("utc")), device=str(r.get("device") or p.get("device_model") or "?"), props=dict(p),
                 weather=r.get("weather"), image_sha256=r.get("image_sha256"))
        g.nodes.append(n)
        g.by_sk[(n.session_id, n.k)] = n.idx
    g.bbox = [min(n.lon for n in g.nodes), min(n.lat for n in g.nodes),
              max(n.lon for n in g.nodes), max(n.lat for n in g.nodes)]
    xy = np.array([(n.x, n.y) for n in g.nodes])
    g.tree = cKDTree(xy)
    G = g.G
    G.add_nodes_from(range(len(g.nodes)))

    def add(i, j, w, kind, length):
        if G.has_edge(i, j) and G[i][j]["w"] <= w:
            return
        G.add_edge(i, j, w=w, kind=kind, length=length)

    # track edges
    by_session: dict[str, list[Node]] = {}
    for n in g.nodes:
        by_session.setdefault(n.session_id, []).append(n)
    for sid, ns in by_session.items():
        ns.sort(key=lambda n: n.k)
        for a, b in zip(ns, ns[1:]):
            if b.k == a.k + 1:
                add(a.idx, b.idx, g.dist(a.idx, b.idx), "track", g.dist(a.idx, b.idx))
        # drive ends (a k break also makes an end)
        for i, n in enumerate(ns):
            prev_ok = i > 0 and ns[i - 1].k == n.k - 1
            next_ok = i + 1 < len(ns) and ns[i + 1].k == n.k + 1
            if not (prev_ok and next_ok):
                g.ends.add(n.idx)
    # same-road / crossing links between drives
    for i, j in g.tree.query_pairs(CROSS_M):
        a, b = g.nodes[i], g.nodes[j]
        if a.session_id == b.session_id:
            continue
        if a.heading is None or b.heading is None:
            continue
        if a.alt is not None and b.alt is not None and abs(a.alt - b.alt) > MAX_DALT_M:
            continue
        d = g.dist(i, j)
        dh = angdiff(a.heading, b.heading)
        if d < SAME_ROAD_M and (dh < SAME_DH or dh > OPPOSITE_DH):
            add(i, j, d + SWITCH_COST_M, "same", d)
        elif SAME_DH <= dh <= OPPOSITE_DH:
            add(i, j, d + SWITCH_COST_M, "cross", d)
    # gap edges: end of a drive -> start of the time-next drive of the same device
    spans = []
    for sid, ns in by_session.items():
        timed = [n for n in ns if n.utc is not None]
        if timed:
            spans.append((ns[0].device, min(n.utc for n in timed), max(n.utc for n in timed), sid, ns))
    spans.sort(key=lambda s: (s[0], s[1]))
    for (dev, t0, t1, sid, ns), (dev2, u0, u1, sid2, ns2) in zip(spans, spans[1:]):
        if dev != dev2:
            continue
        dt = (u0 - t1).total_seconds()
        if not 0 <= dt < GAP_MAX_DT_S:
            continue
        a, b = ns[-1], ns2[0]
        d = g.dist(a.idx, b.idx)
        add(a.idx, b.idx, GAP_FACTOR * d, "gap", d)
        g.gaps.append({"from_session": sid, "to_session": sid2, "from_segment": a.segment_id,
                       "to_segment": b.segment_id, "straight_m": d, "dt_s": dt})
    return g


# ============================================================ path
def snap(g: Graph, lat: float, lon: float, what: str) -> dict:
    x, y = g.xy(lat, lon)
    d, i = g.tree.query([x, y], k=1)
    if not math.isfinite(d) or d > SNAP_MAX_M:
        raise PciError(f"{what} is {d:.0f} m from the nearest surveyed segment (limit {SNAP_MAX_M:.0f} m); "
                       f"pick a point on a surveyed road")
    n = g.nodes[int(i)]
    return {"node": int(i), "segment_id": n.segment_id, "distance_m": round(float(d), 1), "lat": lat, "lon": lon,
            "snapped_lat": n.lat, "snapped_lon": n.lon}


def _explicit_gap(g: Graph, u: int, v: int) -> Optional[tuple]:
    du = nx.single_source_dijkstra_path_length(g.G, u, weight="w")
    dv = nx.single_source_dijkstra_path_length(g.G, v, weight="w")
    a_c = [a for a in g.ends if a in du]
    b_c = [b for b in g.ends if b in dv]
    if not a_c or not b_c:
        return None
    tb = cKDTree(np.array([(g.nodes[b].x, g.nodes[b].y) for b in b_c]))
    best = None
    for a in a_c:
        for jj in tb.query_ball_point([g.nodes[a].x, g.nodes[a].y], EXPLICIT_GAP_MAX_M):
            b = b_c[jj]
            d = g.dist(a, b)
            cost = du[a] + GAP_FACTOR * d + dv[b]
            if best is None or cost < best[0]:
                best = (cost, a, b, d)
    return best


def find_path(g: Graph, a: Mapping, b: Mapping, via: Iterable[Mapping] = ()) -> dict:
    """Snap and route. Returns {nodes, steps, measures, snapped, gaps, length_m}."""
    via = list(via or [])
    if len(via) > MAX_VIA:
        raise PciError(f"at most {MAX_VIA} via points")
    pts = [("A", a)] + [(f"via {i + 1}", v) for i, v in enumerate(via)] + [("B", b)]
    snapped = [snap(g, float(p["lat"]), float(p["lon"]), name) for name, p in pts]
    for s, (name, _) in zip(snapped, pts):
        s["name"] = name
    ax, ay = g.xy(float(a["lat"]), float(a["lon"]))
    bx, by = g.xy(float(b["lat"]), float(b["lon"]))
    if not via and (snapped[0]["node"] == snapped[-1]["node"] or math.hypot(ax - bx, ay - by) < 5.0):
        raise PciError("A and B are the same point; pick two points at least 50 m apart along the road")
    nodes: list[int] = []
    explicit: set = set()
    for li in range(len(snapped) - 1):
        u, v = snapped[li]["node"], snapped[li + 1]["node"]
        if u == v:
            leg = [u]
        else:
            try:
                leg = nx.dijkstra_path(g.G, u, v, weight="w")
            except nx.NetworkXNoPath:
                leg = None
                if li > 0 or li + 1 < len(snapped) - 1:       # the leg touches a via point
                    best = _explicit_gap(g, u, v)
                    if best is not None:
                        _, ga, gb, _d = best
                        leg = nx.dijkstra_path(g.G, u, ga, weight="w") + nx.dijkstra_path(g.G, gb, v, weight="w")
                        explicit.add((ga, gb))
                if leg is None:
                    raise PciError(f"no surveyed path from {snapped[li]['name']} to {snapped[li + 1]['name']}"
                                   + ("" if (li > 0 or li + 1 < len(snapped) - 1) else
                                      "; add a via point where the vehicle drove to allow a flagged gap "
                                      f"(at most {EXPLICIT_GAP_MAX_M:.0f} m, straight line)"))
        nodes.extend(leg if not nodes else (leg[1:] if leg[0] == nodes[-1] else leg))
    steps, measures, gaps = [], [0.0], []
    for i in range(1, len(nodes)):
        u, v = nodes[i - 1], nodes[i]
        d = g.dist(u, v)
        if (u, v) in explicit or (v, u) in explicit:
            kind = "explicit_gap"
        elif g.G.has_edge(u, v):
            kind = g.G[u][v]["kind"]
        else:
            kind = "explicit_gap"
        steps.append(kind)
        if kind in ("gap", "explicit_gap"):
            gaps.append({"from_m": measures[-1], "to_m": measures[-1] + d, "straight_m": d, "kind": kind,
                         "from_segment": g.nodes[u].segment_id, "to_segment": g.nodes[v].segment_id})
        measures.append(measures[-1] + d)
    L = measures[-1]
    if L < MIN_PATH_M:
        raise PciError(f"the path from A to B is {L:.0f} m; at least {MIN_PATH_M:.0f} m is needed")
    if L > MAX_PATH_M:
        raise PciError(f"the path from A to B is {L / 1000:.1f} km; at most {MAX_PATH_M / 1000:.0f} km")
    return {"nodes": nodes, "steps": steps, "measures": measures, "snapped": snapped, "gaps": gaps, "length_m": L}


# ============================================================ passes and attribution
def _path_passes(g: Graph, path: dict) -> list[dict]:
    nodes, steps, ms = path["nodes"], path["steps"], path["measures"]
    passes, cur = [], [0]
    for i in range(1, len(nodes)):
        if steps[i - 1] == "track":
            cur.append(i)
        else:
            passes.append(cur)
            cur = [i]
    passes.append(cur)
    out = []
    for pi, pos in enumerate(passes):
        idxs = [nodes[i] for i in pos]
        cosines = []
        for j, i in enumerate(pos):
            lo, hi = pos[max(0, j - 1)], pos[min(len(pos) - 1, j + 1)]
            if lo == hi:
                continue
            a, b = g.nodes[nodes[lo]], g.nodes[nodes[hi]]
            h = g.nodes[nodes[i]].heading
            if h is None or (a.x == b.x and a.y == b.y):
                continue
            cosines.append(math.cos(math.radians(angdiff(h, bearing_xy(b.x - a.x, b.y - a.y)))))
        c = sum(cosines) / len(cosines) if cosines else None
        first = g.nodes[idxs[0]]
        ks = [g.nodes[i].k for i in idxs]
        times = [g.nodes[i].utc for i in idxs if g.nodes[i].utc]
        out.append({"id": f"P{pi}", "kind": "path", "session_id": first.session_id, "device": first.device,
                    "positions": pos, "nodes": idxs, "k_first": ks[0], "k_last": ks[-1],
                    "from_m": ms[pos[0]], "to_m": ms[pos[-1]],
                    "direction": ("towards B" if c is not None and c > 0 else
                                  "towards A" if c is not None else "unknown"),
                    "heading_agreement": None if c is None else round(c, 3),
                    "date_utc": max(times) if times else None})
    return out


def _attach_other(g: Graph, path: dict, passes: list[dict]) -> tuple[list[dict], list[dict]]:
    """Segments of other passes linked to the path by same-road links. Returns (same_dir, opposite) passes,
    each {id, session_id, segs: [(node, measure)], ...}."""
    nodes, ms, steps = path["nodes"], path["measures"], path["steps"]
    on_path = set(nodes)
    pos_of_pass = {}
    for p in passes:
        for pos in p["positions"]:
            pos_of_pass[pos] = p
    best: dict[int, tuple] = {}
    for pos, i in enumerate(nodes):
        a = g.nodes[i]
        if a.heading is None:
            continue
        for j in g.tree.query_ball_point([a.x, a.y], SAME_ROAD_M):
            if j in on_path:
                continue
            b = g.nodes[j]
            if b.heading is None:
                continue
            if b.session_id == a.session_id and abs(b.k - a.k) < SAME_SESSION_MIN_DK:
                continue
            if a.alt is not None and b.alt is not None and abs(a.alt - b.alt) > MAX_DALT_M:
                continue
            dh = angdiff(a.heading, b.heading)
            if dh < SAME_DH:
                cat = "same"
            elif dh > OPPOSITE_DH:
                cat = "opposite"
            else:
                continue
            d = math.hypot(a.x - b.x, a.y - b.y)
            if j not in best or d < best[j][0]:
                best[j] = (d, pos, cat)
    # project into a +-30 m measure window around the linked path node
    attached = {}
    for j, (d, pos, cat) in best.items():
        b = g.nodes[j]
        m0 = ms[pos]
        bm, bd = None, None
        lo = pos
        while lo > 0 and ms[lo - 1] >= m0 - WINDOW_M:
            lo -= 1
        hi = pos
        while hi + 1 < len(nodes) and ms[hi + 1] <= m0 + WINDOW_M:
            hi += 1
        for q in range(lo, hi):
            if steps[q] in ("gap", "explicit_gap"):
                continue
            p1, p2 = g.nodes[nodes[q]], g.nodes[nodes[q + 1]]
            vx, vy = p2.x - p1.x, p2.y - p1.y
            L2 = vx * vx + vy * vy
            t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((b.x - p1.x) * vx + (b.y - p1.y) * vy) / L2))
            px, py = p1.x + t * vx, p1.y + t * vy
            dd = math.hypot(b.x - px, b.y - py)
            if bd is None or dd < bd:
                bd, bm = dd, ms[q] + t * (ms[q + 1] - ms[q])
        if bm is None:
            bm = m0
        if bm <= 0.0 or bm >= ms[-1]:
            continue
        attached[j] = (bm, cat)
    # group per (session, category) into runs of k; fill holes of <= FILL_DK
    groups: dict[tuple, list] = {}
    for j, (m, cat) in attached.items():
        groups.setdefault((g.nodes[j].session_id, cat), []).append((g.nodes[j].k, j, m))
    same, opp = [], []
    for (sid, cat), items in sorted(groups.items()):
        items.sort()
        runs_, cur = [], [items[0]]
        for it in items[1:]:
            if it[0] - cur[-1][0] <= FILL_DK + 1:
                cur.append(it)
            else:
                runs_.append(cur)
                cur = [it]
        runs_.append(cur)
        for r in runs_:
            segs = []
            for (k1, j1, m1), (k2, j2, m2) in zip(r, r[1:]):
                segs.append((j1, m1))
                for kk in range(k1 + 1, k2):
                    jj = g.by_sk.get((sid, kk))
                    if jj is not None and jj not in on_path and jj not in attached:
                        segs.append((jj, m1 + (m2 - m1) * (kk - k1) / (k2 - k1)))
            segs.append((r[-1][1], r[-1][2]))
            times = [g.nodes[j].utc for j, _ in segs if g.nodes[j].utc]
            item = {"session_id": sid, "device": g.nodes[r[0][1]].device, "segs": segs,
                    "k_first": r[0][0], "k_last": r[-1][0], "from_m": min(m for _, m in segs),
                    "to_m": max(m for _, m in segs), "date_utc": max(times) if times else None,
                    "kind": "other", "direction": "same as linked path pass" if cat == "same" else "opposite"}
            (same if cat == "same" else opp).append(item)
    for i, p in enumerate(same):
        p["id"] = f"O{i}"
    for i, p in enumerate(opp):
        p["id"] = f"X{i}"
    return same, opp


def sections_of(L: float) -> list[tuple[float, float]]:
    """100 m sub-sections from A; a tail < 50 m merges into the previous one."""
    n = int(math.floor(L / SECTION_M + 1e-9))
    edges = [i * SECTION_M for i in range(n + 1)]
    if not edges or edges[-1] < L - 1e-9:
        if n >= 1 and L - edges[-1] < TAIL_MERGE_M:
            edges[-1] = L
        else:
            edges.append(L)
    if len(edges) == 1:
        edges.append(L)
    return list(zip(edges, edges[1:]))


def slice_geometry(vertices: list, steps: list, m0: float, m1: float) -> dict:
    """MultiLineString of the path between measures m0 and m1 (cut at gaps), endpoints interpolated."""
    parts, cur = [], []

    def interp(i, m):
        (x1, y1, a), (x2, y2, b) = vertices[i], vertices[i + 1]
        t = 0.0 if b == a else (m - a) / (b - a)
        return [round(x1 + t * (x2 - x1), 7), round(y1 + t * (y2 - y1), 7)]
    for i in range(len(vertices) - 1):
        a, b = vertices[i][2], vertices[i + 1][2]
        if b < m0 or a > m1:
            continue
        if steps[i] in ("gap", "explicit_gap"):
            if len(cur) >= 2:
                parts.append(cur)
            cur = []
            continue
        p0 = interp(i, max(a, m0)) if a < m0 else [vertices[i][0], vertices[i][1]]
        p1 = interp(i, min(b, m1)) if b > m1 else [vertices[i + 1][0], vertices[i + 1][1]]
        if not cur or cur[-1] != p0:
            if len(cur) >= 2:
                parts.append(cur)
            cur = [p0]
        if p1 != cur[-1]:
            cur.append(p1)
    if len(cur) >= 2:
        parts.append(cur)
    return {"type": "MultiLineString", "coordinates": parts}


def _overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _seg_dict(g: Graph, j: int, m: float, pid: str) -> dict:
    d = dict(g.nodes[j].props)
    d["run_key"] = pid
    d["measure_m"] = m
    return d


def _trim_q(q: dict) -> dict:
    """Quantities without per-run segment lists, JSON-ready (rounded)."""
    out = {}
    for k, v in q.items():
        if isinstance(v, float):
            out[k] = round(v, 4)
        elif k in ("extent", "reported"):
            out[k] = {p: (None if e is None else {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                                                  for kk, vv in e.items()}) for p, e in v.items()}
        elif k == "runs_per_km":
            out[k] = {p: (None if x is None else round(x, 3)) for p, x in v.items()}
        else:
            out[k] = v
    return out


def _round_rating(r: dict) -> dict:
    def rr(x):
        if isinstance(x, float):
            return round(x, 4)
        if isinstance(x, dict):
            return {k: rr(v) for k, v in x.items()}
        if isinstance(x, list):
            return [rr(v) for v in x]
        return x
    return rr(r)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _ist(dt: Optional[datetime]) -> Optional[str]:
    return dt.astimezone(IST).strftime("%Y-%m-%d %H:%M:%S IST") if dt else None


# ============================================================ request
CATEGORY_NAMES = {"urban": "Urban road", "highway": "Highway (NH/SH)", "rural": "MDR / rural road (ODR, VR)"}
TEXT_FIELDS = ("road_name", "road_number", "surface_type", "prepared_by", "title")


def canonical_request(req: Mapping) -> dict:
    """The fields that decide a report, normalised (for the input hash and the echo)."""
    def pt(p):
        return {"lat": round(float(p["lat"]), 7), "lon": round(float(p["lon"]), 7)}
    out = {"a": pt(req["a"]), "b": pt(req["b"]), "via": [pt(v) for v in (req.get("via") or [])],
           "category": req.get("category") or "urban",
           "lane_width_m": float(req.get("lane_width_m") or C.DEFAULT_LANE_WIDTH_M),
           "chainage_at_a_km": None if req.get("chainage_at_a_km") is None else float(req["chainage_at_a_km"]),
           "chainage_increases_towards_b": req.get("chainage_increases_towards_b")}
    for f in TEXT_FIELDS:
        v = req.get(f)
        out[f] = v.strip() if isinstance(v, str) and v.strip() else None
    return out


def input_hash(canon: Mapping, data_as_of: Optional[str], sample: bool = False) -> str:
    blob = json.dumps({"input": canon, "data_as_of": data_as_of, "sample": bool(sample)}, sort_keys=True,
                      separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ============================================================ path document (POST /gis/pci/path)
def path_geometry(g: Graph, path: dict) -> dict:
    """MultiLineString of the surveyed parts (split at gaps) + the gaps as their own lines."""
    parts, cur = [], [path["nodes"][0]]
    gap_lines = []
    for i, kind in enumerate(path["steps"]):
        v = path["nodes"][i + 1]
        if kind in ("gap", "explicit_gap"):
            if len(cur) >= 2:
                parts.append(cur)
            gap_lines.append([path["nodes"][i], v])
            cur = [v]
        else:
            cur.append(v)
    if len(cur) >= 2:
        parts.append(cur)

    def coords(ix):
        return [[round(g.nodes[i].lon, 7), round(g.nodes[i].lat, 7)] for i in ix]
    return {"surveyed": {"type": "MultiLineString", "coordinates": [coords(p) for p in parts]},
            "gaps": {"type": "MultiLineString", "coordinates": [coords(p) for p in gap_lines]}}


def path_document(g: Graph, req: Mapping) -> dict:
    canon = canonical_request(req)
    path = find_path(g, canon["a"], canon["b"], canon["via"])
    passes = _path_passes(g, path)
    same, opp = _attach_other(g, path, passes)
    geom = path_geometry(g, path)
    L = path["length_m"]
    return {
        "schema": SCHEMA, "length_m": round(L, 1),
        "chainage_a": fmt_chainage(0), "chainage_b": fmt_chainage(L),
        "snapped": path["snapped"],
        "gaps": [{**{k: (round(v, 1) if isinstance(v, float) else v) for k, v in gp.items()},
                  "label": f"≥ {gp['straight_m']:.0f} m (straight line)"} for gp in path["gaps"]],
        "gap_m": round(sum(gp["straight_m"] for gp in path["gaps"]), 1),
        "switches": sum(1 for s in path["steps"] if s in ("same", "cross")),
        "passes": [_pass_echo(g, p) for p in passes],
        "other_passes": [_pass_echo(g, p) for p in same], "opposite_passes": [_pass_echo(g, p) for p in opp],
        "geometry": geom,
        "vertices": [[round(g.nodes[i].lon, 7), round(g.nodes[i].lat, 7), round(m, 2)]
                     for i, m in zip(path["nodes"], path["measures"])],
        "steps": list(path["steps"]),
        "geojson": {"type": "FeatureCollection", "features": [
            {"type": "Feature", "geometry": geom["surveyed"], "properties": {"kind": "surveyed"}},
            {"type": "Feature", "geometry": geom["gaps"], "properties": {"kind": "gap"}}]},
        "_path": path, "_passes": passes, "_same": same, "_opp": opp, "_canon": canon,
    }


def _pass_echo(g: Graph, p: dict) -> dict:
    return {"id": p["id"], "kind": p["kind"], "session_id": p["session_id"], "device": p["device"],
            "k_first": p["k_first"], "k_last": p["k_last"], "from_m": round(p["from_m"], 1),
            "to_m": round(p["to_m"], 1), "direction": p["direction"],
            "date_ist": p["date_utc"].astimezone(IST).strftime("%Y-%m-%d") if p.get("date_utc") else None,
            "segments": len(p.get("nodes") or p.get("segs") or [])}


def public(doc: Mapping) -> dict:
    """The document without the private (underscore) working fields."""
    return {k: v for k, v in doc.items() if not k.startswith("_")}


# ============================================================ report
def build_report(g: Graph, req: Mapping, *, generated_at: Optional[datetime] = None,
                 report_id: Optional[Any] = None) -> dict:
    """The full report JSON (DESIGN §8). Deterministic for the same graph + request + generated_at."""
    doc = path_document(g, req)
    canon = doc["_canon"]
    path, passes, same, opp = doc["_path"], doc["_passes"], doc["_same"], doc["_opp"]
    cat = canon["category"]
    if cat not in C.TABLES:
        raise PciError(f"category must be one of {', '.join(C.TABLES)}")
    width = canon["lane_width_m"]
    L = path["length_m"]
    secs = sections_of(L)
    starts = [s for s, _ in secs]

    def sec_index(m):
        return max(0, min(len(secs) - 1, bisect.bisect_right(starts, m) - 1))

    # candidate segments per pass per section
    cand: list[dict] = [dict() for _ in secs]          # section -> pass id -> [seg dicts]
    pinfo = {}
    for p in passes:
        pinfo[p["id"]] = p
        for pos, j in zip(p["positions"], p["nodes"]):
            m = path["measures"][pos]
            cand[sec_index(m)].setdefault(p["id"], []).append(_seg_dict(g, j, m, p["id"]))
    for p in same:
        pinfo[p["id"]] = p
        for j, m in p["segs"]:
            cand[sec_index(m)].setdefault(p["id"], []).append(_seg_dict(g, j, m, p["id"]))
    opp_by_sec: list[list] = [[] for _ in secs]
    for p in opp:
        pinfo[p["id"]] = p
        for si in {sec_index(m) for _, m in p["segs"]}:
            opp_by_sec[si].append(p["id"])

    def classified(segs):
        return sum(float(s.get("length_m") or 0) for s in segs
                   if s.get("status") in ("send", "audit") and s.get("vlm_class") in ("distress", "normal"))

    chosen_segs: list[list] = []
    sec_choice = []
    for si, c in enumerate(cand):
        if not c:
            chosen_segs.append([])
            sec_choice.append({"chosen": None, "also": []})
            continue
        ranked = sorted(c.items(), key=lambda kv: (classified(kv[1]), pinfo[kv[0]]["date_utc"] or
                                                   datetime.min.replace(tzinfo=timezone.utc), kv[0]), reverse=True)
        chosen_segs.append(ranked[0][1])
        sec_choice.append({"chosen": ranked[0][0], "also": [k for k, _ in ranked[1:]]})
    # runs over the whole stretch's chosen segments, each assigned to the unit of its first metre
    all_chosen = [s for segs in chosen_segs for s in segs]
    run_lists = {"potholes": C.runs(all_chosen, C.PARAM_LABELS["potholes"]),
                 "cracking": C.runs(all_chosen, C.PARAM_LABELS["cracking"])}

    def runs_in(m0, m1, last):
        return {t: [r for r in rs if r["measure_m"] is not None and m0 <= r["measure_m"] and
                    (r["measure_m"] < m1 or last)] for t, rs in run_lists.items()}

    def gap_in(m0, m1):
        return sum(_overlap(m0, m1, gp["from_m"], gp["to_m"]) for gp in path["gaps"])

    def unit(m0, m1, segs, last, extra=None):
        q = C.unit_quantities(segs, m1 - m0, gap_m=gap_in(m0, m1), lane_width_m=width,
                              runs_in_unit=runs_in(m0, m1, last))
        r = C.rate_unit(q, cat)
        u = {"from_m": round(m0, 1), "to_m": round(m1, 1), "chainage_from": fmt_chainage(m0),
             "chainage_to": fmt_chainage(m1),
             "road_chainage_from": road_chainage(m0, canon["chainage_at_a_km"], canon["chainage_increases_towards_b"]),
             "road_chainage_to": road_chainage(m1, canon["chainage_at_a_km"], canon["chainage_increases_towards_b"]),
             "quantities": _trim_q(q), "rating": _round_rating(r), "treatment": {
                 "class": C.class_treatment(r["final"]["class"] if r.get("final") else None, cat),
                 "distress": C.distress_treatments(q) if r.get("rated") else []}}
        if extra:
            u.update(extra)
        u["geometry"] = slice_geometry(doc["vertices"], doc["steps"], m0, m1)
        return u

    def pass_label(pid):
        p = pinfo[pid]
        d = p["date_utc"].astimezone(IST).strftime("%Y-%m-%d") if p.get("date_utc") else "?"
        return {"pass": pid, "session_id": p["session_id"], "date_ist": d}

    sections = []
    for si, (m0, m1) in enumerate(secs):
        ch = sec_choice[si]
        sections.append(unit(m0, m1, chosen_segs[si], si == len(secs) - 1, {
            "index": si, "chosen_pass": pass_label(ch["chosen"]) if ch["chosen"] else None,
            "also_surveyed": [pass_label(x) for x in ch["also"]],
            "other_direction": [pass_label(x) for x in sorted(set(opp_by_sec[si]))]}))
    km_groups: dict[int, list[int]] = {}
    for si, (m0, _m1) in enumerate(secs):
        km_groups.setdefault(int(math.floor(m0 / 1000.0 + 1e-9)), []).append(si)
    kms = []
    for kmi, sis in sorted(km_groups.items()):
        m0, m1 = secs[sis[0]][0], secs[sis[-1]][1]
        kms.append(unit(m0, m1, [s for si in sis for s in chosen_segs[si]], sis[-1] == len(secs) - 1,
                        {"km": kmi, "part_km": (m1 - m0) < 1000.0 - 1e-6, "sections": sis}))
    stretch = unit(0.0, L, all_chosen, True, {"part_km": L < 1000.0 - 1e-6})
    stretch.pop("geometry", None)          # = the path geometry

    # survey facts of the used passes
    used_ids = sorted({s["segment_id"] for s in all_chosen})
    used_nodes = [g.nodes[g.by_sk[(s["session_id"], s["k"])]] for s in all_chosen]
    dates = sorted({n.utc.astimezone(IST).date().isoformat() for n in used_nodes if n.utc})
    seasons = sorted({season_of(n.utc.astimezone(IST)) for n in used_nodes if n.utc})
    weather = sorted({n.weather for n in used_nodes if n.weather})
    devices = sorted({str(n.props.get("device_model") or n.device) for n in used_nodes})
    sess_used = sorted({n.session_id for n in used_nodes})
    pipe = sorted({(g.sessions.get(s, {}).get("pipeline_version"), g.sessions.get(s, {}).get("config_sha256"))
                   for s in sess_used}, key=lambda t: (str(t[0]), str(t[1])))
    reviewed = sum(1 for s in all_chosen if s.get("expert_reviewed"))
    gen = generated_at or datetime.now(timezone.utc)

    sh = stretch["rating"]
    summary = {
        "length_m": round(L, 1), "chainage_a": fmt_chainage(0), "chainage_b": fmt_chainage(L),
        "coverage": stretch["quantities"]["coverage"], "coverage_status": stretch["quantities"]["coverage_status"],
        "gap_m": round(sum(gp["straight_m"] for gp in path["gaps"]), 1), "gaps": len(path["gaps"]),
        "headline": sh.get("headline"), "final": sh.get("final"),
        "sections_by_class": _count_classes(sections), "km_by_class": _count_classes(kms),
        "segments_used": len(all_chosen), "segments_expert_reviewed": reviewed,
    }
    report = {
        "schema": SCHEMA, "name": C.NAME, "report_id": report_id,
        "generated_at": _iso(gen), "generated_at_ist": _ist(gen), "data_as_of": g.data_as_of,
        "input": canon,
        "input_note": "road name/number, surface type and prepared by: entered by requester, unverified",
        "header": {
            "road_name": canon["road_name"], "road_number": canon["road_number"],
            "category": cat, "category_name": CATEGORY_NAMES[cat], "table": C.TABLES[cat]["table"],
            "table_title": C.TABLES[cat]["title"],
            "km_from": fmt_chainage(0), "km_to": fmt_chainage(L),
            "road_chainage_from": road_chainage(0.0, canon["chainage_at_a_km"], canon["chainage_increases_towards_b"]),
            "road_chainage_to": road_chainage(L, canon["chainage_at_a_km"], canon["chainage_increases_towards_b"]),
            "road_chainage_note": ("calibrated by requester at A (unverified)" if canon["chainage_at_a_km"] is not None
                                   and canon["chainage_increases_towards_b"] is not None
                                   else "Road chainage not calibrated (no km-stone reference)"),
            "a": path["snapped"][0], "b": path["snapped"][-1], "via": path["snapped"][1:-1],
            "directions": [{"pass": p["id"], "session_id": p["session_id"], "direction": p["direction"]}
                           for p in passes],
            "lane_imaged": "one lane (the lane the vehicle drove in); lane position not recorded",
            "lane_width_m": width, "lane_width_note": ("default 3.5 m (not measured)"
                                                      if req.get("lane_width_m") in (None, "") else "entered by requester"),
            "surface_type": canon["surface_type"], "survey_dates_ist": dates, "season": seasons,
            "weather_as_entered_on_survey_form": weather,
            "method": "Phone video (forward-looking, windshield mount), one frame per 5 m segment, frame gate, "
                      "VLM classification (pipeline P4)",
            "devices": devices,
            "model": {"pipeline": "P4", "model": "Qwen/Qwen2.5-VL-7B-Instruct (base, 4-bit NF4, 1024² cap)",
                      "prompts": "v2 (Stage 2 prompt id 9da1601cfec54f76)",
                      "gate": "field confidence 0.80 (Stage 1 and Stage 2); Stage 1 safety net on",
                      "video_pipeline": [{"pipeline_version": a, "config_sha256": b} for a, b in pipe]},
            "prepared_by": canon["prepared_by"],
            "prepared_by_note": "entered by requester, unverified",
        },
        "summary": summary,
        "path": public(doc),
        "sections": sections, "km": kms, "stretch": stretch,
        "treatment_heading": C.TREATMENT_HEADING, "treatment_any_class": C.ANY_CLASS_TEXT,
        "annex": _annex(g, sections, chosen_segs, run_lists, canon),
        "accuracy": ACCURACY,
        "method": _method_text(cat),
        "limitations": LIMITATIONS,
        "segment_ids": used_ids,
        "assessment": {p: dict(v) for p, v in C.ASSESSMENT.items()},
    }
    return _clean(report)


def _count_classes(units: list[dict]) -> dict:
    out: dict[str, int] = {}
    for u in units:
        f = u["rating"].get("final")
        key = f["class"] if f else "Insufficient data"
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def _annex(g: Graph, sections: list[dict], chosen_segs: list[list], run_lists: dict, canon: Mapping) -> list[dict]:
    """<= 24 frames: one per pothole run and one per section whose assessed band can be Poor (the most
    confident distress frame of the section), in chainage order; more than 24 are thinned evenly."""
    picks: dict[str, dict] = {}
    by_id = {n.segment_id: n for n in g.nodes}

    def add(sid, reason, m):
        n = by_id.get(sid)
        if n is None or not n.props.get("has_image") or sid in picks:
            return
        p = n.props
        picks[sid] = {"segment_id": sid, "reason": reason, "measure_m": round(m, 1), "chainage": fmt_chainage(m),
                      "road_chainage": road_chainage(m, canon["chainage_at_a_km"], canon["chainage_increases_towards_b"]),
                      "lat": n.lat, "lon": n.lon, "time_ist": _ist(n.utc), "labels": p.get("final_labels") or [],
                      "severity": p.get("final_severity"), "confidence": p.get("confidence"),
                      "review_state": ("expert reviewed" if p.get("expert_reviewed") else
                                       "needs expert review (not reviewed)" if p.get("needs_expert_review")
                                       else "model only (not reviewed)"),
                      "image_sha256": n.image_sha256, "session_id": n.session_id, "k": n.k,
                      "note": "frame shows road ahead of the marked chainage"}

    for r in run_lists["potholes"]:
        for sid in r["segment_ids"]:
            if by_id.get(sid) is not None and by_id[sid].props.get("has_image"):
                add(sid, "pothole run", r["measure_m"])
                break
    for sec, segs in zip(sections, chosen_segs):
        rt = sec["rating"]
        if not rt.get("rated") or not any(p.get("assessed") and p.get("band_hi") == "Poor" for p in rt["params"]):
            continue
        dis = [x for x in segs if x.get("vlm_class") == "distress" and x.get("has_image")]
        if dis:
            best = max(dis, key=lambda x: (x.get("confidence") or 0.0, -x["measure_m"]))
            add(best["segment_id"], "section rated Poor (" + sec["chainage_from"] + ")", best["measure_m"])
    out = sorted(picks.values(), key=lambda x: x["measure_m"])
    if len(out) > ANNEX_MAX:
        # keep an even spread along the stretch, deterministic
        step = len(out) / ANNEX_MAX
        out = [out[int(i * step)] for i in range(ANNEX_MAX)]
    return out


def _method_text(cat: str) -> list[str]:
    return [
        "Chainage: the path from A to B follows the surveyed drives (segment centres 5 m apart along the camera "
        "path); chainage is the cumulative distance between consecutive segment centres from A, without smoothing. "
        "Gaps between drives count their straight line and are printed as lower bounds.",
        "Units: 100 m sub-sections from A (a tail under 50 m merges into the previous one), km rows "
        "(IRC:82 'kilometre-wise'; a row under 1 km is a part-km), and the whole stretch. Quantities are summed, "
        "then rated; ratings are never averaged.",
        "Where several drives cover a section in the same direction, the pass with the most classified length is "
        "used (ties: latest date); the others are listed. Opposite-direction passes are listed, not rated.",
        "Extent E (% of surveyed length with the distress visible, est.) = affected classified length / "
        "(classified + gate-skipped length). E_lo assumes skipped segments are distress-free; E_hi assumes they "
        "are distressed at the audited rate's upper 95 % limit (13.3 %). Per-type skip rates are not estimable "
        "(13 distressed audits).",
        "Coverage = imaged length / section length. Rated at >= 80 % (our choice, not IRC); 50-80 % rated and "
        "flagged 'low coverage'; under 50 % or nothing classified: insufficient data.",
        f"Rating: IRC:82-2015 {C.TABLES[cat]['title']}; rating value interpolated inside each band as inferred "
        "from the Appendix-4 worked example (6 of 7 rows reproduced; the class never depends on it); weighted "
        "values rounded half-up to 0.1; final = mean over ALL parameters of the table (Appendix 4). Parameters "
        "the pipeline cannot assess are bracketed (rating 1.0 for the low end, 3.0 for the high end).",
        "Potholes: runs per km (consecutive positive segments of one pass = one occurrence) is the primary figure; "
        "a nominal small-pothole area (runs x 0.0314 m², IRC §7.5.3.4) over the surveyed lane is a sensitivity "
        "figure, not a bound.",
    ]


LIMITATIONS = [
    "Not an IRC:82 survey: IRC §5.3 asks for visual surveys by experienced engineers, at least twice a year on "
    "urban roads (pre- and post-monsoon). This report is a screening / prioritisation aid only.",
    "Area vs length: IRC % inputs are distress area / pavement area (implied by Proforma 2). This report measures "
    "the fraction of 5 m segments in which the model saw the distress: an UPPER BOUND on % area for localised "
    "distress, cracks included (one transverse crack flags a whole 5 m segment).",
    "Look-ahead: the camera sees about 1.4-33 m ahead; a distress may be labelled on several consecutive segments "
    "and up to ~30 m behind its true position. Runs (consecutive positives) are reported for this reason.",
    "Not assessed (never printed as 0 %): ravelling, settlement/depression, shoving, patching, rut depth (3 m "
    "straight edge), drainage and shoulder condition, roughness and skid resistance (Tables 5.4/5.5), crack width "
    "and pothole depth (severity is visually judged).",
    "Model validity (Attain dataset, Tehran; not Bengaluru): Stage 1 recall on distress frames 51.8 %; linear "
    "crack P 0.89 / R 0.82, MCC 0.257; alligator cracking recall 0.08; pothole P 0.43 / R 0.25, MCC 0.219; "
    "ravelling never named.",
    "0 expert labels on the video segments: no label in this report (including Bleeding and Edge Breaking) has "
    "been validated against Bengaluru expert labels.",
    "Frame gate: segments the gate skipped were not classified (audited skips distressed 8.1 % [4.8, 13.3] vs "
    "19.7 % of sent segments); E_lo / E_hi bracket this.",
    "GNSS: phone position, receiver latency not corrected; ±10 m on surveyed runs, more after gaps. One lane, "
    "one direction per pass; distress outside the camera's view or in other lanes is not seen.",
    "Survey-form fields (road name, direction, weather) on the drives are form defaults or entries by the "
    "operator and are printed only as 'as entered'.",
]


def _clean(x):
    """JSON-safe: datetimes to ISO strings, non-finite floats to None, tuples to lists."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, datetime):
        return _iso(x)
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


# ============================================================ loading from the server's database
_cache: dict = {"key": None, "graph": None}
_cache_lock = threading.Lock()

_KEY_SQL = """
SELECT count(*) AS n, max(g.updated_at) AS seg_at, max(a.reviewed_at) AS rev_at, max(a.processed_at) AS proc_at
FROM video_segments g LEFT JOIN assessments a ON a.id = g.assessment_id
"""
_EXTRA_SQL = """
SELECT segment_id, image_sha256, record->'camera'->>'alt_ellipsoid_m' AS alt,
       record->'device'->>'device_id' AS device_id, record->'survey'->>'weather' AS weather,
       record->>'pipeline_version' AS pipeline_version, record->>'config_sha256' AS config_sha256
FROM video_segments
"""


def _key_row(cur) -> dict:
    return cur.execute(_KEY_SQL).fetchone()


def data_key() -> tuple[tuple, Optional[str]]:
    """(cache key, data_as_of ISO) of the current database."""
    from app import gis_routes as GR
    r = GR._query(_key_row)
    times = [t for t in (r["seg_at"], r["rev_at"], r["proc_at"]) if t is not None]
    # full precision: a second-truncated stamp would mark every report stale at once
    as_of = (max(times).astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
             if times else None)
    return (r["n"], _iso(r["seg_at"]), _iso(r["rev_at"]), _iso(r["proc_at"])), as_of


def load_graph(force: bool = False) -> Graph:
    """The cached graph; rebuilt when the segment count / updated_at or any review changed."""
    from app import gis_routes as GR
    key, as_of = data_key()
    with _cache_lock:
        if not force and _cache["graph"] is not None and _cache["key"] == key:
            return _cache["graph"]
        rows = GR._fetch_segment_rows(None)
        extra = {r["segment_id"]: r for r in GR._query(lambda cur: cur.execute(_EXTRA_SQL).fetchall())}
        recs, sessions = [], {}
        for r in rows:
            props = GR._segment_feature(r, "point")["properties"]
            e = extra.get(props["segment_id"], {})
            recs.append({"props": props, "alt": e.get("alt"), "device": e.get("device_id"),
                         "weather": e.get("weather"), "image_sha256": e.get("image_sha256")})
            sessions.setdefault(props["session_id"], {"pipeline_version": e.get("pipeline_version"),
                                                      "config_sha256": e.get("config_sha256")})
        g = build_graph(recs, key=key, data_as_of=as_of, sessions=sessions)
        _cache.update(key=key, graph=g)
        return g


def changed_since(segment_ids: list[str], as_of: Optional[str]) -> int:
    """How many of these segments changed (row, answer or review) after as_of."""
    from app import gis_routes as GR
    if not as_of or not segment_ids:
        return 0
    return GR._query(lambda cur: cur.execute(
        "SELECT count(*) AS n FROM video_segments g LEFT JOIN assessments a ON a.id = g.assessment_id "
        "WHERE g.segment_id = ANY(%s) AND (g.updated_at > %s OR a.reviewed_at > %s OR a.processed_at > %s)",
        (segment_ids, as_of, as_of, as_of)).fetchone())["n"]
