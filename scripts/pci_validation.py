"""
Validation of the A->B chainage + IRC:82-2015 condition rating (DESIGN §6) -> eval_results/pci/validation.json
and validation.md. Every number in validation.md comes from this script's run; the command line is recorded.

Parts (each checkpointed atomically to eval_results/pci/validation_parts/<part>.json; finished parts are
skipped unless --redo <part>):
  arithmetic     IRC arithmetic: App-4 worked example, every band boundary of every table, truncation, bracket
  synthetic      chainage on synthetic drives: straight / arc / 90 / 180 degree turns; correlated 2-D AR(1) GNSS
                 noise (sigma 4 and 6 m; correlation length 50, 100, 200 m; 50 seeds each) -> length error;
                 A/B swap symmetry; via points; refusals
  retest         test-retest on real overlaps: segments of two different drives within 6 m, |dheading| < 30 deg,
                 |d alt| <= 4 m (the same-road link rule); label agreement (Cohen's kappa) for Distress,
                 Potholes, Cracking; E and band agreement on overlapping 100 m blocks; offsets of matched segments
  osm            independent length: OpenStreetMap (Overpass API; data (c) OpenStreetMap contributors, ODbL) way
                 network routed through our path's surveyed points every 100 m, for the sample stretches; OSM route
                 length vs straight line for every time-adjacent gap. If Overpass cannot be reached: "not done".
  runs           run-length distribution of consecutive positive segments per label (all drives)
  sensitivity    every 100 m block of every drive (k order, Table 5.3): bracket width, E_lo vs E_hi band changes,
                 coverage rule 50 vs 80 %, sum(wr)/sum(w) and assessed-only classes
  samples        headline numbers of the stored sample reports (research.pci_reports, sample = true)

Reads (SELECT only) the database named with --database (default: geoai, production, read-only use).

Usage:
    venv/Scripts/python.exe scripts/pci_validation.py --database geoai
    venv/Scripts/python.exe scripts/pci_validation.py --database geoai --redo osm
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "tests"))

from scripts import setup_local_db  # noqa: E402

OUT_DIR = PROJECT_ROOT / "eval_results" / "pci"
PARTS_DIR = OUT_DIR / "validation_parts"
OSM_CACHE = Path(os.environ.get("PCI_OSM_CACHE") or r"C:\GeoAI\pci\osm_cache")
PARTS = ("arithmetic", "synthetic", "retest", "osm", "runs", "sensitivity", "samples")
OVERPASS = ("https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
            "https://maps.mail.ru/osm/tools/overpass/api/interpreter")
DRIVABLE = ("motorway|trunk|primary|secondary|tertiary|unclassified|residential|service|living_street|road|"
            "motorway_link|trunk_link|primary_link|secondary_link|tertiary_link")


def atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


def kappa(pairs) -> dict:
    """Cohen's kappa for binary pairs [(a, b)]."""
    n = len(pairs)
    if n == 0:
        return {"n": 0, "kappa": None}
    a1 = sum(1 for a, _ in pairs if a)
    b1 = sum(1 for _, b in pairs if b)
    agree = sum(1 for a, b in pairs if a == b) / n
    pe = (a1 / n) * (b1 / n) + (1 - a1 / n) * (1 - b1 / n)
    k = None if pe >= 1 else (agree - pe) / (1 - pe)
    return {"n": n, "agreement": round(agree, 4), "kappa": None if k is None else round(k, 4),
            "positive_a": a1, "positive_b": b1, "both_positive": sum(1 for a, b in pairs if a and b)}


def q(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    i = (len(xs) - 1) * p
    lo, hi = int(math.floor(i)), int(math.ceil(i))
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def summ(xs, nd=3) -> dict:
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "mean": round(statistics.fmean(xs), nd), "sd": round(statistics.pstdev(xs), nd),
            "min": round(min(xs), nd), "p50": round(q(xs, 0.5), nd), "p90": round(q(xs, 0.9), nd),
            "max": round(max(xs), nd)}


# ============================================================ parts
def part_arithmetic(ctx) -> dict:
    from app import pci_core as C
    v = dict(cracking=9, ravelling=8, potholes=0, shoving=0.09, patching=5, settlement=4, rut_depth=9)
    ex = C.app4_final("highway", v)
    checks, fails = 0, []
    for cat, tab in C.TABLES.items():
        for p, (L, U) in tab["params"].items():
            pts = [(0.0, "Good")] if not (cat == "urban" and p == "potholes") else [(0.0, "Good"), (0.001, "Fair"),
                                                                                    (0.5, "Fair"), (0.5001, "Poor")]
            if not (cat == "urban" and p == "potholes"):
                pts += [(L, "Fair"), (L - 1e-12, "Fair"), (L - 1e-6, "Good"), (U, "Fair"), (U + 1e-12, "Fair"),
                        (U + 1e-6, "Poor")]
            for x, want in pts:
                checks += 1
                if C.band(cat, p, x) != want:
                    fails.append([cat, p, x, C.band(cat, p, x), want])
    trunc = {"2-(9-5)/5": C.trunc1(2 - (9 - 5) / 5), "1.5555": C.trunc1(1.5555)}
    rnd = {"0.75*1.5": C.weighted(0.75, 1.5), "0.75*1.2": C.weighted(0.75, 1.2), "0.5*2.5": C.weighted(0.5, 2.5)}
    # bracket extremes on Table 5.3 (all distress / none)
    seg = lambda k, lab: {"segment_id": str(k), "k": k, "status": "send", "length_m": 5.0, "run_key": "P",
                          "vlm_class": "distress" if lab else "normal", "final_labels": lab or ["Normal"],
                          "final_severity": "High" if lab else "None", "measure_m": 5.0 * k}
    worst = C.rate_unit(C.unit_quantities([seg(k, ["Potholes", "Alligator Cracking"]) for k in range(20)], 100.0),
                        "urban")["final"]
    best = C.rate_unit(C.unit_quantities([seg(k, None) for k in range(20)], 100.0), "urban")["final"]
    return {"app4_example": {"rows": [[r["param"], r["input"], r["rating"], r["weighted"]] for r in ex["rows"]],
                             "final": ex["final"], "class": ex["class"],
                             "note": "IRC prints 1.3 for the rut-depth row; the inferred rule gives 1.2 (RESEARCH §1.5);"
                                     " final 1.3 Fair either way"},
            "band_boundary_checks": checks, "band_boundary_failures": fails, "truncation": trunc, "rounding": rnd,
            "urban_bracket_all_distress": worst, "urban_bracket_no_distress": best}


def part_synthetic(ctx) -> dict:
    from app import pci_chainage as PC
    from test_pci_chainage import ar1_noise, arc, drive, ll, polyline
    out = {}

    def L(pts_xy):
        d = drive("S", polyline(pts_xy))
        g = PC.build_graph(d)
        return PC.find_path(g, ll(d[0]), ll(d[-1]))["length_m"]
    shapes = {"straight_1000": ([(0, 0), (1000, 0)], 1000.0),
              "arc_r100_90deg": (arc(0, 0, 100, 0, 90, step=1.0), math.pi * 100 / 2),
              "turn90_r15": ([(0, 0), (300, 0)] + arc(300, 15, 15, -90, 0, step=1.0)[1:] + [(315, 315)],
                             600 + math.pi * 15 / 2),
              "turn180_r8": ([(0, 0), (300, 0)] + arc(300, 8, 8, -90, 90, step=1.0)[1:] + [(0, 16)],
                             600 + math.pi * 8)}
    out["shapes"] = {k: {"true_m": round(t, 2), "measured_m": round(L(p), 2), "error_m": round(L(p) - t, 2)}
                     for k, (p, t) in shapes.items()}
    noise = {}
    for sigma in (4.0, 6.0):
        for corr in (50.0, 100.0, 200.0):
            errs = []
            for seed in range(50):
                pts = polyline([(0, 0), (1000, 0)])
                d = drive("N", pts, noise=ar1_noise(len(pts), sigma, corr, seed=seed))
                g = PC.build_graph(d)
                errs.append(PC.find_path(g, ll(d[0]), ll(d[-1]))["length_m"] - 1000.0)
            noise[f"sigma{sigma:g}_corr{corr:g}m"] = summ(errs, 2)
    out["ar1_noise_length_error_m_per_1000m"] = noise
    # A/B swap symmetry + via + refusals (counts of cases behaving as specified)
    d = drive("W", polyline([(0, 0), (1230, 0)]), labels={k: ["Potholes"] for k in range(10, 14)})
    g = PC.build_graph(d)
    r1 = PC.build_report(g, {"a": ll(d[0]), "b": ll(d[-1])})
    r2 = PC.build_report(g, {"a": ll(d[-1]), "b": ll(d[0])})
    out["swap"] = {"length_diff_m": abs(r1["summary"]["length_m"] - r2["summary"]["length_m"]),
                   "pothole_E_lo_diff": abs(r1["stretch"]["quantities"]["extent"]["potholes"]["lo"] -
                                            r2["stretch"]["quantities"]["extent"]["potholes"]["lo"])}
    import test_pci_chainage as T
    results = {}
    for name in ("test_refusals", "test_via_points_and_swap_symmetry", "test_gap_rules", "test_parallel_road_not_joined",
                 "test_flyover_refused_level_crossing_used", "test_u_turn_and_loop_attribution"):
        try:
            getattr(T, name)()
            results[name] = "pass"
        except Exception as e:
            results[name] = f"FAIL: {type(e).__name__}: {e}"
    out["rule_tests"] = results
    return out


def _classified(p):
    return p.get("status") in ("send", "audit") and p.get("vlm_class") in ("distress", "normal")


def _has(p, labels):
    return p.get("vlm_class") == "distress" and bool(set(p.get("final_labels") or []) & set(labels))


def part_retest(ctx) -> dict:
    from app import pci_chainage as PC
    from app import pci_core as C
    g = ctx["graph"]
    start = {}
    for n in g.nodes:
        if n.utc and (n.session_id not in start or n.utc < start[n.session_id]):
            start[n.session_id] = n.utc
    best = {}     # (i, other session) -> (d, j)
    for i, j in g.tree.query_pairs(PC.SAME_ROAD_M):
        a, b = g.nodes[i], g.nodes[j]
        if a.session_id == b.session_id or a.heading is None or b.heading is None:
            continue
        if a.alt is not None and b.alt is not None and abs(a.alt - b.alt) > PC.MAX_DALT_M:
            continue
        if PC.angdiff(a.heading, b.heading) >= PC.SAME_DH:
            continue
        if (start.get(a.session_id), a.session_id) > (start.get(b.session_id), b.session_id):
            a, b, i, j = b, a, j, i
        d = math.hypot(a.x - b.x, a.y - b.y)
        key = (i, b.session_id)
        if key not in best or d < best[key][0]:
            best[key] = (d, j)
    pairs = [(i, j, d) for (i, _), (d, j) in best.items()]
    drive_pairs = Counter((g.nodes[i].session_id, g.nodes[j].session_id) for i, j, _ in pairs)
    both = [(g.nodes[i].props, g.nodes[j].props) for i, j, _ in pairs
            if _classified(g.nodes[i].props) and _classified(g.nodes[j].props)]
    out = {"rule": "nearest segment of each later drive within 6 m, |dheading| < 30 deg, |d alt| <= 4 m",
           "matched_pairs": len(pairs), "drive_pairs": len(drive_pairs), "pairs_both_classified": len(both),
           "status_pairs": dict(Counter(f"{g.nodes[i].props['status']}/{g.nodes[j].props['status']}"
                                        for i, j, _ in pairs).most_common())}
    out["kappa"] = {
        "Distress": kappa([(a["vlm_class"] == "distress", b["vlm_class"] == "distress") for a, b in both]),
        "Potholes": kappa([(_has(a, ["Potholes"]), _has(b, ["Potholes"])) for a, b in both]),
        "Cracking": kappa([(_has(a, C.PARAM_LABELS["cracking"]), _has(b, C.PARAM_LABELS["cracking"])) for a, b in both]),
    }
    along, cross = [], []
    for i, j, d in pairs:
        a, b = g.nodes[i], g.nodes[j]
        h = math.radians(a.heading)
        ux, uy = math.sin(h), math.cos(h)
        dx, dy = b.x - a.x, b.y - a.y
        along.append(abs(dx * ux + dy * uy))
        cross.append(abs(-dx * uy + dy * ux))
    out["offset_m"] = {"distance": summ([d for _, _, d in pairs], 2), "along_track_abs": summ(along, 2),
                       "cross_track_abs": summ(cross, 2),
                       "note": "nearest-neighbour matches inside the 6 m link radius: offsets are bounded by the "
                               "matching rule, so they describe the link geometry, not an independent chainage error"}
    # 100 m blocks: 20 consecutive segments of the earlier drive, all matched in the later one
    by_pair = defaultdict(dict)
    for i, j, _ in pairs:
        by_pair[(g.nodes[i].session_id, g.nodes[j].session_id)][g.nodes[i].k] = (i, j)
    blocks = []
    for (sx, sy), m in by_pair.items():
        ks = sorted(m)
        run = [ks[0]] if ks else []
        runs_ = []
        for k in ks[1:]:
            if k == run[-1] + 1:
                run.append(k)
            else:
                runs_.append(run)
                run = [k]
        if run:
            runs_.append(run)
        for r in runs_:
            for b0 in range(0, len(r) - 19, 20):
                xs = [m[k][0] for k in r[b0:b0 + 20]]
                ys = sorted({m[k][1] for k in r[b0:b0 + 20]})
                segx = [dict(g.nodes[x].props, run_key="X", measure_m=0.0) for x in xs]
                segy = [dict(g.nodes[y].props, run_key="Y", measure_m=0.0) for y in ys]
                qx = C.unit_quantities(segx, 5.0 * len(segx))
                qy = C.unit_quantities(segy, 5.0 * len(segy))
                if qx["coverage_status"] == "insufficient" or qy["coverage_status"] == "insufficient":
                    continue
                row = {"x": sx, "y": sy}
                for p in ("cracking", "potholes"):
                    ex, ey = qx["extent"][p]["lo"], qy["extent"][p]["lo"]
                    row[p] = (ex, ey, C.band("urban", p, ex), C.band("urban", p, ey))
                rx, ry = C.rate_unit(qx, "urban")["final"], C.rate_unit(qy, "urban")["final"]
                row["final"] = (rx["class"], ry["class"])
                blocks.append(row)
    bl = {"n_blocks": len(blocks), "drive_pairs_with_blocks": len({(b["x"], b["y"]) for b in blocks})}
    for p in ("cracking", "potholes"):
        if blocks:
            bl[p] = {"band_agreement": round(sum(1 for b in blocks if b[p][2] == b[p][3]) / len(blocks), 4),
                     "mean_abs_diff_E_lo_pct": round(statistics.fmean(abs(b[p][0] - b[p][1]) for b in blocks), 2),
                     "E_lo_earlier": summ([b[p][0] for b in blocks], 2), "E_lo_later": summ([b[p][1] for b in blocks], 2)}
    if blocks:
        bl["final_class_agreement"] = round(sum(1 for b in blocks if b["final"][0] == b["final"][1]) / len(blocks), 4)
        bl["final_class_pairs"] = dict(Counter(f"{a} / {b}" for a, b in (x["final"] for x in blocks)).most_common())
    out["blocks_100m"] = bl
    # how smooth the real tracks are: centre-to-centre path / the pipeline's s-path (NOT independent: both from
    # the same GNSS fit; it shows whether jitter would inflate our chainage, as the AR(1) synthetic case does)
    by = defaultdict(list)
    for n in g.nodes:
        by[n.session_id].append(n)
    ratios = []
    for ns in by.values():
        ns.sort(key=lambda n: n.k)
        cp = sum(math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(ns, ns[1:]) if b.k == a.k + 1)
        sp = sum((b.props.get("s0_m") or 0) - (a.props.get("s0_m") or 0) for a, b in zip(ns, ns[1:])
                 if b.k == a.k + 1 and a.props.get("s0_m") is not None and b.props.get("s0_m") is not None)
        if sp > 50:
            ratios.append(cp / sp)
    out["centre_path_over_s_path"] = summ(ratios, 4)
    return out


_osm_fail = {"consecutive": 0}
OSM_GIVE_UP_AFTER = 3          # consecutive failed items: Overpass is down, stop asking (status "not done"/"partial")


def _overpass(bbox, tag):
    import httpx
    OSM_CACHE.mkdir(parents=True, exist_ok=True)
    f = OSM_CACHE / f"{tag.replace(':', '_')}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    if _osm_fail["consecutive"] >= OSM_GIVE_UP_AFTER:
        raise RuntimeError("skipped: Overpass failed on the previous items")
    try:
        data = _overpass_fetch(bbox, f)
    except RuntimeError:
        _osm_fail["consecutive"] += 1
        raise
    _osm_fail["consecutive"] = 0
    time.sleep(1.0)            # be polite to the public server
    return data


def _overpass_fetch(bbox, f):
    import httpx
    s, w, n, e = bbox
    ql = f'[out:json][timeout:90];way["highway"~"^({DRIVABLE})$"]({s:.6f},{w:.6f},{n:.6f},{e:.6f});out geom;'
    last = None
    wait = 5.0
    for attempt in range(8):
        url = OVERPASS[attempt % len(OVERPASS)]
        try:
            r = httpx.post(url, data={"data": ql}, timeout=120,
                           headers={"User-Agent": "GeoAI-capstone-pci-validation/1.0 (research; contact via repo)"})
            if r.status_code == 200 and r.text.lstrip().startswith("{"):
                data = r.json()
                atomic(f, data)
                return data
            last = f"HTTP {r.status_code} from {url}"
        except Exception as ex:
            last = f"{type(ex).__name__} from {url}"
        time.sleep(wait)
        wait = min(wait * 2, 120)
    raise RuntimeError(last or "Overpass unreachable")


def _osm_graph(data, lat0, lon0):
    import networkx as nx
    from app import pci_chainage as PC
    G = nx.Graph()
    xy = {}
    kx = PC.R_EARTH * math.cos(math.radians(lat0)) * math.pi / 180
    ky = PC.R_EARTH * math.pi / 180
    for el in data.get("elements", []):
        if el.get("type") != "way" or "geometry" not in el:
            continue
        pts = el["geometry"]
        ids = el.get("nodes") or list(range(len(pts)))
        for (pa, a), (pb, b) in zip(zip(ids, pts), list(zip(ids, pts))[1:]):
            ka, kb = ("n", pa) if el.get("nodes") else ("w", el["id"], pa), ("n", pb) if el.get("nodes") else ("w", el["id"], pb)
            xy[ka] = ((a["lon"] - lon0) * kx, (a["lat"] - lat0) * ky)
            xy[kb] = ((b["lon"] - lon0) * kx, (b["lat"] - lat0) * ky)
            d = math.hypot(xy[ka][0] - xy[kb][0], xy[ka][1] - xy[kb][1])
            G.add_edge(ka, kb, w=d)
    return G, xy


def part_osm(ctx) -> dict:
    import networkx as nx
    import numpy as np
    from scipy.spatial import cKDTree
    from app import pci_chainage as PC
    g = ctx["graph"]
    rule = json.loads((OUT_DIR / "samples_rule.json").read_text(encoding="utf-8"))
    out = {"attribution": "Map data (c) OpenStreetMap contributors, ODbL 1.0 (https://www.openstreetmap.org/copyright),"
                          " via the Overpass API", "samples": [], "gaps": []}
    errors = []

    def route_len(G, tree, keys, pts_xy):
        total, snaps = 0.0, []
        nodes = []
        for x, y in pts_xy:
            d, i = tree.query([x, y])
            snaps.append(d)
            nodes.append(keys[i])
        for u, v in zip(nodes, nodes[1:]):
            if u != v:
                total += nx.shortest_path_length(G, u, v, weight="w")
        return total, snaps

    def bbox_of(points, margin_m=300):
        lats = [p[0] for p in points]
        lons = [p[1] for p in points]
        dl = margin_m / 111320.0
        dn = dl / math.cos(math.radians(statistics.fmean(lats)))
        return (min(lats) - dl, min(lons) - dn, max(lats) + dl, max(lons) + dn)

    for c in rule["chosen"]:
        try:
            doc = PC.path_document(g, {"a": c["a"], "b": c["b"]})
        except PC.PciError as e:
            out["samples"].append({"rule": c["_rule"], "error": str(e)})
            continue
        verts, steps = doc["vertices"], doc["steps"]
        pts_ll = [(v[1], v[0]) for v in verts]
        try:
            data = _overpass(bbox_of(pts_ll), f"sample_{c['_rule']}")
        except Exception as e:
            errors.append(f"{c['_rule']}: {e}")
            out["samples"].append({"rule": c["_rule"], "error": f"Overpass: {e}"})
            continue
        lat0, lon0 = pts_ll[0]
        G, xy = _osm_graph(data, lat0, lon0)
        if G.number_of_nodes() == 0:
            out["samples"].append({"rule": c["_rule"], "error": "no OSM ways in the bbox"})
            continue
        keys = list(xy)
        tree = cKDTree(np.array([xy[k] for k in keys]))
        kx = PC.R_EARTH * math.cos(math.radians(lat0)) * math.pi / 180
        ky = PC.R_EARTH * math.pi / 180
        # surveyed runs only (gaps split them): waypoints every ~100 m of measure
        runs_, cur = [], [0]
        for i, s in enumerate(steps):
            if s in ("gap", "explicit_gap"):
                runs_.append(cur)
                cur = [i + 1]
            else:
                cur.append(i + 1)
        runs_.append(cur)
        ours, osm, snap_d = 0.0, 0.0, []
        for r in runs_:
            if len(r) < 2:
                continue
            ms = [verts[i][2] for i in r]
            ours += ms[-1] - ms[0]
            way, nextm = [], ms[0]
            for i in r:
                if verts[i][2] >= nextm - 1e-9:
                    way.append(i)
                    nextm = verts[i][2] + 100.0
            if way[-1] != r[-1]:
                way.append(r[-1])
            pts = [((verts[i][0] - lon0) * kx, (verts[i][1] - lat0) * ky) for i in way]
            try:
                L, sd = route_len(G, tree, keys, pts)
            except nx.NetworkXNoPath:
                L, sd = None, []
            if L is None:
                osm = None
                break
            osm += L
            snap_d += sd
        out["samples"].append({"rule": c["_rule"], "surveyed_length_m": round(ours, 1),
                               "osm_route_length_m": None if osm is None else round(osm, 1),
                               "ratio_ours_over_osm": None if not osm else round(ours / osm, 4),
                               "waypoint_snap_m": summ(snap_d, 1)})
    # gaps: straight line vs OSM shortest route
    by = {n.segment_id: n for n in g.nodes}
    for gp in g.gaps:
        a, b = by[gp["from_segment"]], by[gp["to_segment"]]
        try:
            data = _overpass(bbox_of([(a.lat, a.lon), (b.lat, b.lon)], 600), f"gap_{gp['from_segment']}")
        except Exception as e:
            errors.append(f"gap {gp['from_segment']}: {e}")
            out["gaps"].append({"from_segment": gp["from_segment"], "error": f"Overpass: {e}"})
            continue
        G, xy = _osm_graph(data, a.lat, a.lon)
        if G.number_of_nodes() == 0:
            out["gaps"].append({"from_segment": gp["from_segment"], "error": "no OSM ways"})
            continue
        keys = list(xy)
        tree = cKDTree(np.array([xy[k] for k in keys]))
        kx = PC.R_EARTH * math.cos(math.radians(a.lat)) * math.pi / 180
        ky = PC.R_EARTH * math.pi / 180
        pa = (0.0, 0.0)
        pb = ((b.lon - a.lon) * kx, (b.lat - a.lat) * ky)
        try:
            L, sd = route_len(G, tree, keys, [pa, pb])
        except nx.NetworkXNoPath:
            L, sd = None, []
        out["gaps"].append({"from_segment": gp["from_segment"], "straight_m": round(gp["straight_m"], 1),
                            "dt_s": round(gp["dt_s"], 1), "osm_route_m": None if L is None else round(L, 1),
                            "ratio_osm_over_straight": None if not L or gp["straight_m"] < 1 else
                            round(L / gp["straight_m"], 3), "snap_m": [round(x, 1) for x in sd]})
    ok_gaps = [x for x in out["gaps"] if x.get("ratio_osm_over_straight")]
    out["gap_ratio_summary"] = summ([x["ratio_osm_over_straight"] for x in ok_gaps], 3)
    ok_s = [x for x in out["samples"] if x.get("ratio_ours_over_osm")]
    out["sample_ratio_summary"] = summ([x["ratio_ours_over_osm"] for x in ok_s], 4)
    out["errors"] = errors
    out["status"] = ("done" if not errors and ok_s else ("partial" if ok_s or ok_gaps else "not done"))
    out["caveat"] = ("OSM routes are shortest paths on an undirected drivable-way graph through our own points, "
                     "snapped to the nearest OSM vertex; they approximate the driven route, they do not survey it")
    return out


def part_runs(ctx) -> dict:
    from app import pci_core as C
    g = ctx["graph"]
    by = defaultdict(list)
    for n in g.nodes:
        by[n.session_id].append(dict(n.props, run_key=n.session_id, measure_m=0.0))
    types = {"Potholes": ["Potholes"], "Cracking (any)": list(C.PARAM_LABELS["cracking"]),
             "Alligator Cracking": ["Alligator Cracking"], "Longitudinal Cracking": ["Longitudinal Cracking"],
             "Transverse Cracking": ["Transverse Cracking"], "Bleeding": ["Bleeding"], "Edge Breaking": ["Edge Breaking"]}
    out = {}
    for name, labs in types.items():
        rs = [r for segs in by.values() for r in C.runs(segs, labs)]
        lens = [r["n"] for r in rs]
        pos = sum(lens)
        out[name] = {"positive_segments": pos, "runs": len(rs), "segments_per_run": summ(lens, 2),
                     "hist": C.run_length_hist(rs),
                     "share_of_positives_in_runs_ge3": round(sum(x for x in lens if x >= 3) / pos, 4) if pos else None}
    return out


def part_sensitivity(ctx) -> dict:
    from app import pci_core as C
    g = ctx["graph"]
    by = defaultdict(list)
    for n in g.nodes:
        by[n.session_id].append(n)
    rows = []
    for sid, ns in by.items():
        ns.sort(key=lambda n: n.k)
        m, cur, cur_m0 = 0.0, [], 0.0
        prev = None
        blocks = []
        for n in ns:
            if prev is not None:
                m += math.hypot(n.x - prev.x, n.y - prev.y)
            prev = n
            if m - cur_m0 >= 100.0 and cur:
                blocks.append((cur_m0, m, cur))
                cur, cur_m0 = [], m
            cur.append(dict(n.props, run_key=sid, measure_m=m))
        if cur:
            if blocks and m - cur_m0 < 50.0:
                a0, _, segs = blocks.pop()
                blocks.append((a0, m, segs + cur))
            else:
                blocks.append((cur_m0, max(m, cur_m0 + 5.0), cur))
        for a0, a1, segs in blocks:
            qq = C.unit_quantities(segs, a1 - a0)
            r = C.rate_unit(qq, "urban")
            rows.append((qq, r))
    rated = [(qq, r) for qq, r in rows if r["rated"]]
    out = {"blocks": len(rows), "coverage_status": dict(Counter(qq["coverage_status"] for qq, _ in rows)),
           "rated_under_80_rule": sum(1 for qq, _ in rows if qq["rated_80"]),
           "rated_under_50_rule": sum(1 for qq, _ in rows if qq["rated_50"])}
    if rated:
        out["final_class_bracket"] = dict(Counter(r["final"]["class"] for _, r in rated).most_common())
        out["final_class_lo"] = dict(Counter(r["final"]["class_lo"] for _, r in rated).most_common())
        out["final_class_hi"] = dict(Counter(r["final"]["class_hi"] for _, r in rated).most_common())
        out["bracket_width"] = summ([r["final"]["hi"] - r["final"]["lo"] for _, r in rated], 2)
        out["single_class"] = sum(1 for _, r in rated if r["final"]["class_lo"] == r["final"]["class_hi"])
        for p in ("cracking", "potholes"):
            prm = [next(x for x in r["params"] if x["param"] == p) for _, r in rated]
            out[f"{p}_band"] = dict(Counter(x["band"] for x in prm).most_common())
            out[f"{p}_band_lo_differs_from_hi"] = sum(1 for x in prm if x["band_lo"] != x["band_hi"])
        out["sum_wr_over_sum_w_class"] = dict(Counter(r["sensitivity"]["sum_wr_over_sum_w"]["class"]
                                                      for _, r in rated).most_common())
        out["assessed_only_class"] = dict(Counter(r["sensitivity"]["assessed_only"]["class"]
                                                  for _, r in rated).most_common())
        out["uncorrected_class"] = dict(Counter(r["sensitivity"].get("uncorrected", {}).get("class", "n/a")
                                                for _, r in rated).most_common())
        out["poor_reachable_lo_end"] = out["final_class_lo"].get("Poor", 0)
    out["note"] = ("blocks of each drive in k order (no routing, one pass), 100 m by centre distance, tail < 50 m merged; "
                   "Table 5.3 (urban)")
    return out


def part_samples(ctx) -> dict:
    import psycopg
    with psycopg.connect(ctx["dsn"], autocommit=True) as c:
        c.execute("SET default_transaction_read_only = on")
        try:
            rows = c.execute("SELECT id, sample_rule, data_as_of, report->'summary', length(pdf) FROM research.pci_reports "
                             "WHERE sample ORDER BY id").fetchall()
        except psycopg.errors.UndefinedTable:
            return {"status": "not done: research.pci_reports does not exist in this database (run pci_samples.py)"}
    out = []
    for rid, rule, as_of, s, size in rows:
        hl = s.get("headline") or {}
        out.append({"id": rid, "rule": rule, "data_as_of": str(as_of), "length_m": s["length_m"],
                    "coverage": s["coverage"], "gaps": s["gaps"], "gap_m": s["gap_m"],
                    "cracking": hl.get("cracking"), "potholes": hl.get("potholes"),
                    "pothole_runs_per_km": hl.get("pothole_runs_per_km"),
                    "final": (s.get("final") or {}).get("label"), "pdf_bytes": size})
    return {"status": "done" if out else "no sample reports stored", "reports": out}


# ============================================================ markdown
def fmt(x, nd=3):
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def markdown(res: dict) -> str:
    L = [f"# Validation — A→B chainage and IRC:82-2015 condition rating", "",
         f"Generated {res['generated_at']} by `{res['command']}` on database `{res['database']}` "
         f"(data as of {res['data_as_of']}). Every number below is from that run "
         f"(`eval_results/pci/validation.json`).", ""]
    a = res["parts"].get("arithmetic")
    if a:
        ex = a["app4_example"]
        L += ["## 1. IRC arithmetic", "",
              f"- App-4 worked example (Table 5.1): ratings {', '.join(f'{r[0]} {r[2]}' for r in ex['rows'])}; "
              f"final **{ex['final']} {ex['class']}** (IRC prints 1.3 {'Fair'}). {ex['note']}.",
              f"- Band boundaries: {a['band_boundary_checks']} checks, {len(a['band_boundary_failures'])} failures.",
              f"- Truncation 2−(9−5)/5 → {a['truncation']['2-(9-5)/5']}; rounding 0.75·1.5 → {a['rounding']['0.75*1.5']}, "
              f"0.75·1.2 → {a['rounding']['0.75*1.2']}.",
              f"- Table 5.3 bracket, every segment pothole + alligator: {a['urban_bracket_all_distress']['lo']}–"
              f"{a['urban_bracket_all_distress']['hi']} ({a['urban_bracket_all_distress']['class']}); no distress: "
              f"{a['urban_bracket_no_distress']['lo']}–{a['urban_bracket_no_distress']['hi']} "
              f"({a['urban_bracket_no_distress']['class']}). Poor is reachable at the low end.", ""]
    s = res["parts"].get("synthetic")
    if s:
        L += ["## 2. Chainage on synthetic drives (no smoothing)", "", "| Shape | true m | measured m | error m |",
              "|---|---|---|---|"]
        for k, v in s["shapes"].items():
            L.append(f"| {k} | {v['true_m']} | {v['measured_m']} | {v['error_m']} |")
        L += ["", "Correlated 2-D AR(1) GNSS noise on a 1,000 m straight drive (50 seeds each): length error (m).", "",
              "| noise | mean | sd | p90 | max |", "|---|---|---|---|---|"]
        for k, v in s["ar1_noise_length_error_m_per_1000m"].items():
            L.append(f"| {k} | {v['mean']} | {v['sd']} | {v['p90']} | {v['max']} |")
        L += ["", f"- A/B swap: length difference {s['swap']['length_diff_m']} m, pothole E_lo difference "
                  f"{s['swap']['pothole_E_lo_diff']}.",
              "- Rule tests: " + "; ".join(f"{k} {v}" for k, v in s["rule_tests"].items()) + ".", ""]
    r = res["parts"].get("retest")
    if r:
        L += ["## 3. Test–retest on real overlaps", "", f"Rule: {r['rule']}. Matched pairs {r['matched_pairs']} over "
              f"{r['drive_pairs']} drive pairs; both classified {r['pairs_both_classified']}.", "",
              "| Label | n | agreement | κ | positive (earlier / later / both) |", "|---|---|---|---|---|"]
        for k, v in r["kappa"].items():
            L.append(f"| {k} | {v['n']} | {fmt(v.get('agreement'))} | {fmt(v.get('kappa'))} | "
                     f"{v.get('positive_a')} / {v.get('positive_b')} / {v.get('both_positive')} |")
        b = r["blocks_100m"]
        L += ["", f"100 m blocks fully matched in both drives and rated in both: **{b['n_blocks']}** "
                  f"({b['drive_pairs_with_blocks']} drive pairs)."]
        for p in ("cracking", "potholes"):
            if p in b:
                L.append(f"- {p}: band agreement {fmt(b[p]['band_agreement'])}, mean |ΔE_lo| "
                         f"{b[p]['mean_abs_diff_E_lo_pct']} percentage points.")
        if "final_class_agreement" in b:
            L.append(f"- bracketed final class agreement {fmt(b['final_class_agreement'])}.")
        o = r["offset_m"]
        L += [f"- Matched-segment offsets (bounded by the 6 m rule): along-track p50 {o['along_track_abs'].get('p50')} m, "
              f"p90 {o['along_track_abs'].get('p90')} m; cross-track p50 {o['cross_track_abs'].get('p50')} m.", ""]
    o = res["parts"].get("osm")
    L += ["## 4. Independent length (OpenStreetMap)", ""]
    if not o or o.get("status") == "not done":
        L += ["**Not done**: the Overpass API could not be reached" + (f" ({o['errors'][0]})" if o and o.get("errors")
                                                                      else "") + ". No substitute numbers.", ""]
    else:
        L += [f"Status: {o['status']}. {o['attribution']}. {o['caveat']}.", "",
              "| Sample | surveyed m (ours) | OSM route m | ours / OSM |", "|---|---|---|---|"]
        for x in o["samples"]:
            if "error" in x:
                L.append(f"| {x['rule']} | – | – | {x['error']} |")
            else:
                L.append(f"| {x['rule']} | {x['surveyed_length_m']} | {fmt(x['osm_route_length_m'], 1)} | "
                         f"{fmt(x['ratio_ours_over_osm'], 4)} |")
        gs = o["gap_ratio_summary"]
        L += ["", f"Time-adjacent gaps: OSM route / straight line over {gs.get('n', 0)} gaps: median {gs.get('p50')}, "
                  f"p90 {gs.get('p90')}, max {gs.get('max')} (straight lines are lower bounds)."]
        if o["errors"]:
            L.append(f"- {len(o['errors'])} Overpass requests failed (listed in validation.json).")
        L.append("")
    u = res["parts"].get("runs")
    if u:
        L += ["## 5. Run lengths (consecutive positive segments of one drive)", "",
              "| Label | positive segments | runs | mean segments/run | p90 | share of positives in runs ≥ 3 |",
              "|---|---|---|---|---|---|"]
        for k, v in u.items():
            sp = v["segments_per_run"]
            L.append(f"| {k} | {v['positive_segments']} | {v['runs']} | {sp.get('mean', '–')} | {sp.get('p90', '–')} | "
                     f"{fmt(v['share_of_positives_in_runs_ge3'], 3)} |")
        L.append("")
    se = res["parts"].get("sensitivity")
    if se:
        L += ["## 6. Class sensitivity (all 100 m blocks of all drives, Table 5.3)", "",
              f"- Blocks {se['blocks']}; coverage {se['coverage_status']}; rated under the 80 % rule "
              f"{se['rated_under_80_rule']}, under the 50 % rule {se['rated_under_50_rule']}.",
              f"- Bracketed final class: {se.get('final_class_bracket')}; single class (both ends agree) "
              f"{se.get('single_class')}.",
              f"- Low end: {se.get('final_class_lo')}; high end: {se.get('final_class_hi')}; bracket width "
              f"{se.get('bracket_width')}.",
              f"- Cracking band {se.get('cracking_band')} (E_lo vs E_hi band differs on "
              f"{se.get('cracking_band_lo_differs_from_hi')}); pothole band {se.get('potholes_band')} (differs on "
              f"{se.get('potholes_band_lo_differs_from_hi')}).",
              f"- Σwr/Σw class {se.get('sum_wr_over_sum_w_class')}; assessed-only {se.get('assessed_only_class')}; "
              f"no skip correction {se.get('uncorrected_class')}.", f"- {se['note']}.", ""]
    sm = res["parts"].get("samples")
    if sm:
        L += ["## Sample reports (pre-registered rule, `samples_rule.json`)", ""]
        if sm.get("reports"):
            L += ["| id | rule | length m | coverage | gaps | cracking band (E) | pothole band (E) | runs/km | final |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for x in sm["reports"]:
                c, p = x.get("cracking") or {}, x.get("potholes") or {}
                L.append(f"| {x['id']} | {x['rule']} | {x['length_m']} | {fmt(x['coverage'], 3)} | {x['gaps']} "
                         f"(≥ {x['gap_m']} m) | {c.get('band', '–')} ({fmt(c.get('E_lo'), 1)}–{fmt(c.get('E_hi'), 1)}) | "
                         f"{p.get('band', '–')} ({fmt(p.get('E_lo'), 1)}–{fmt(p.get('E_hi'), 1)}) | "
                         f"{fmt(x.get('pothole_runs_per_km'), 1)} | {x['final'] or 'Insufficient data'} |")
        else:
            L.append(sm["status"])
        L.append("")
    L += ["## 7. Limitation", "",
          "There are **0** expert labels on the video segments and no Bengaluru validation of any label. Everything "
          "above measures arithmetic, geometry and repeatability between drives — not accuracy against ground truth. "
          "Repeatability between drives of the same model on the same road is not correctness.", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--database", default="geoai", help="database to READ (SELECT only); default production geoai")
    ap.add_argument("--redo", nargs="*", default=[], choices=PARTS)
    ap.add_argument("--only", nargs="*", default=None, choices=PARTS)
    a = ap.parse_args(argv)
    dsn = setup_local_db.with_database(setup_local_db.database_url(), a.database)
    os.environ["DATABASE_URL"] = dsn
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    from app import pci_chainage as PC
    g = PC.load_graph(force=True)
    ctx = {"graph": g, "dsn": dsn}
    command = "venv/Scripts/python.exe scripts/pci_validation.py " + " ".join(sys.argv[1:] if argv is None else argv)
    res = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "command": command.strip(),
           "database": a.database, "data_as_of": g.data_as_of, "parts": {}}
    for part in PARTS:
        f = PARTS_DIR / f"{part}.json"
        if a.only is not None and part not in a.only and not f.exists():
            continue
        if f.exists() and part not in a.redo and (a.only is None or part not in a.only):
            res["parts"][part] = json.loads(f.read_text(encoding="utf-8"))["result"]
            print(f"{part}: reused checkpoint")
            continue
        t0 = time.time()
        print(f"{part}: running ...", flush=True)
        out = globals()[f"part_{part}"](ctx)
        atomic(f, {"part": part, "database": a.database, "data_as_of": g.data_as_of, "command": command.strip(),
                   "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "seconds": round(time.time() - t0, 1), "result": out})
        res["parts"][part] = out
        print(f"{part}: done in {time.time() - t0:.1f} s", flush=True)
    atomic(OUT_DIR / "validation.json", res)
    tmp = OUT_DIR / "validation.md.tmp"
    tmp.write_text(markdown(res), encoding="utf-8")
    os.replace(tmp, OUT_DIR / "validation.md")
    print("wrote eval_results/pci/validation.json and validation.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
