"""
Tests for app/pci_chainage.py on synthetic drives (pure: no database, no GPU).

  1. Chainage length: straight, arc, 90 and 180 degree turns are measured without smoothing bias; with
     CORRELATED GNSS noise (2-D AR(1), correlation length 50 / 200 m, sigma 4 and 6 m) the length error is
     bounded (and reported by scripts/pci_validation.py).
  2. Network rules: a parallel road 10 m away is never joined; a crossing with > 4 m altitude difference
     (flyover) is refused, a level crossing is used; gap edges only to the time-next drive of the same device
     within 10 min; an explicit gap (<= 300 m) only through a via point.
  3. Refusals: A/B > 40 m from the survey, A = B, path < 50 m, no path.
  4. Via points force the route; A/B swap gives the same length and the same stretch quantities.
  5. Attribution: a U-turn's return leg and a loop's revisit are separate passes (opposite listed, not rated;
     same direction attached in a +-30 m window); overlap choice = most classified length, ties latest date.
  6. Units: 100 m sections with the < 50 m tail merged; chainage text "Km x+yyy"; calibrated road chainage.

Usage:
    venv/Scripts/python.exe scripts/tests/test_pci_chainage.py
    venv/Scripts/python.exe -m pytest scripts/tests/test_pci_chainage.py -q
"""
from __future__ import annotations

import math
import random
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import pci_chainage as PC  # noqa: E402

LAT0, LON0 = 12.95, 77.60
T0 = datetime(2026, 10, 27, 10, 0, 0, tzinfo=timezone.utc)


def to_ll(x, y):
    lat = LAT0 + math.degrees(y / PC.R_EARTH)
    lon = LON0 + math.degrees(x / (PC.R_EARTH * math.cos(math.radians(LAT0))))
    return lat, lon


def polyline(points, step=5.0):
    """Points every `step` metres along a polyline of (x, y) vertices, with the tangent heading."""
    out = []
    carry = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        L = math.hypot(x2 - x1, y2 - y1)
        h = PC.bearing_xy(x2 - x1, y2 - y1)
        t = carry
        while t < L - 1e-9:
            out.append((x1 + (x2 - x1) * t / L, y1 + (y2 - y1) * t / L, h))
            t += step
        carry = t - L
    xe, ye = points[-1]
    out.append((xe, ye, out[-1][2]))
    return out


def arc(cx, cy, r, a0, a1, step=5.0):
    """Points every step metres on a circle arc from angle a0 to a1 (degrees, maths convention)."""
    n = max(2, int(abs(math.radians(a1 - a0)) * r / step) + 1)
    pts = []
    for i in range(n + 1):
        a = math.radians(a0 + (a1 - a0) * i / n)
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def drive(sid, pts, *, t0=T0, alt=820.0, device="DEV1", labels=None, status=None, noise=None):
    """Records of one drive: pts = [(x, y, heading)]. labels/status: {k: ...}."""
    recs = []
    for k, (x, y, h) in enumerate(pts):
        if noise is not None:
            x, y = x + noise[k][0], y + noise[k][1]
        lat, lon = to_ll(x, y)
        st = (status or {}).get(k, "send")
        lab = (labels or {}).get(k)
        if st == "skip":
            vc, fl = "not_sent", []
        elif st == "not_covered":
            vc, fl = "not_covered", []
        elif lab:
            vc, fl = "distress", list(lab)
        else:
            vc, fl = "normal", ["Normal"]
        props = {"segment_id": f"{sid}_seg{k:05d}", "session_id": sid, "k": k, "centre_lat": lat, "centre_lon": lon,
                 "camera_heading_deg": h, "utc": (t0 + timedelta(seconds=k)).isoformat().replace("+00:00", "Z"),
                 "status": st, "vlm_class": vc, "final_labels": fl, "final_severity": "Medium" if lab else "None",
                 "length_m": 5.0, "has_image": False, "confidence": 0.9, "device_model": "SM-TEST"}
        recs.append({"props": props, "alt": alt, "device": device})
    return recs


def ll(rec):
    return {"lat": rec["props"]["centre_lat"], "lon": rec["props"]["centre_lon"]}


def ar1_noise(n, sigma, corr_len, step=5.0, seed=1):
    rho = math.exp(-step / corr_len)
    rnd = random.Random(seed)
    e = math.sqrt(1 - rho * rho) * sigma
    nx_, ny_ = rnd.gauss(0, sigma), rnd.gauss(0, sigma)
    out = []
    for _ in range(n):
        out.append((nx_, ny_))
        nx_, ny_ = rho * nx_ + rnd.gauss(0, e), rho * ny_ + rnd.gauss(0, e)
    return out


def raises(fn, text=""):
    try:
        fn()
    except PC.PciError as e:
        assert text in str(e), (text, str(e))
        return str(e)
    raise AssertionError("PciError expected")


# ------------------------------------------------------------------ 1. length
def test_straight_and_curves_length():
    d = drive("S1", polyline([(0, 0), (1000, 0)]))
    g = PC.build_graph(d)
    p = PC.find_path(g, ll(d[0]), ll(d[-1]))
    assert abs(p["length_m"] - 1000.0) < 0.5, p["length_m"]
    # 90 degree turn (r = 15 m) and a 180 degree turn (r = 8 m): true length known
    pts = [(0, 0), (300, 0)] + arc(300, 15, 15, -90, 0)[1:] + [(315, 315)]
    true90 = 300 + math.pi * 15 / 2 + 300
    d = drive("S2", polyline(pts))
    g = PC.build_graph(d)
    L = PC.find_path(g, ll(d[0]), ll(d[-1]))["length_m"]
    assert abs(L - true90) / true90 < 0.01, (L, true90)
    pts = [(0, 0), (300, 0)] + arc(300, 8, 8, -90, 90)[1:] + [(0, 16)]
    true180 = 300 + math.pi * 8 + 300
    d = drive("S3", polyline(pts))
    g = PC.build_graph(d)
    L = PC.find_path(g, ll(d[0]), ll(d[-1]))["length_m"]
    assert abs(L - true180) / true180 < 0.01, (L, true180)


def length_error(sigma, corr_len, seed):
    pts = polyline([(0, 0), (1000, 0)])
    d = drive("N", pts, noise=ar1_noise(len(pts), sigma, corr_len, seed=seed))
    g = PC.build_graph(d)
    return PC.find_path(g, ll(d[0]), ll(d[-1]))["length_m"] - 1000.0


def test_correlated_noise_length_error():
    for sigma in (4.0, 6.0):
        e200 = [length_error(sigma, 200.0, s) for s in range(10)]
        e50 = [length_error(sigma, 50.0, s) for s in range(10)]
        # raw centres lengthen a noisy path; bounded for correlated noise (no smoothing applied)
        assert max(abs(x) for x in e200) < 0.03 * 1000 * (sigma / 4.0) ** 2, (sigma, e200)
        assert max(abs(x) for x in e50) < 0.12 * 1000 * (sigma / 4.0) ** 2, (sigma, e50)
        assert sum(e50) / len(e50) > sum(e200) / len(e200)


# ------------------------------------------------------------------ 2. network rules
def test_parallel_road_not_joined():
    r1 = drive("R1", polyline([(0, 0), (500, 0)]))
    r2 = drive("R2", polyline([(0, 10), (500, 10)]), t0=T0 + timedelta(hours=2))
    g = PC.build_graph(r1 + r2)
    kinds = {d["kind"] for _, _, d in g.G.edges(data=True)}
    assert kinds == {"track"}, kinds
    raises(lambda: PC.find_path(g, ll(r1[0]), ll(r2[-1])), "no surveyed path")
    p = PC.find_path(g, ll(r1[0]), ll(r1[-1]))
    passes = PC._path_passes(g, p)
    same, opp = PC._attach_other(g, p, passes)
    assert not same and not opp


def test_flyover_refused_level_crossing_used():
    a = drive("F1", polyline([(-300, 0), (300, 0)]))
    b = drive("F2", polyline([(0, -300), (0, 300)]), t0=T0 + timedelta(hours=2), alt=826.5)
    g = PC.build_graph(a + b)
    assert not any(d["kind"] == "cross" for _, _, d in g.G.edges(data=True))
    raises(lambda: PC.find_path(g, ll(a[0]), ll(b[-1])), "no surveyed path")
    b = drive("F2", polyline([(0, -300), (0, 300)]), t0=T0 + timedelta(hours=2), alt=821.0)
    g = PC.build_graph(a + b)
    p = PC.find_path(g, ll(a[0]), ll(b[-1]))
    assert any(s == "cross" for s in p["steps"])
    assert 590 < p["length_m"] < 615, p["length_m"]
    assert len(PC._path_passes(g, p)) == 2


def _two_drives(sep, dt_min, device2="DEV1"):
    d1 = drive("G1", polyline([(0, 0), (500, 0)]))
    t2 = T0 + timedelta(seconds=len(d1)) + timedelta(minutes=dt_min)
    d2 = drive("G2", polyline([(500 + sep, 0), (1000 + sep, 0)]), t0=t2, device=device2)
    return d1, d2


def test_gap_rules():
    d1, d2 = _two_drives(100, 2)
    g = PC.build_graph(d1 + d2)
    assert len(g.gaps) == 1 and abs(g.gaps[0]["straight_m"] - 100) < 1
    p = PC.find_path(g, ll(d1[0]), ll(d2[-1]))
    assert len(p["gaps"]) == 1 and p["gaps"][0]["kind"] == "gap"
    doc = PC.path_document(g, {"a": ll(d1[0]), "b": ll(d2[-1])})
    assert doc["gaps"][0]["label"] == "≥ 100 m (straight line)"
    # 20 minutes later: no gap edge; refusal without via; explicit gap through a via point
    d1, d2 = _two_drives(100, 20)
    g = PC.build_graph(d1 + d2)
    assert not g.gaps
    raises(lambda: PC.find_path(g, ll(d1[0]), ll(d2[-1])), "add a via point")
    p = PC.find_path(g, ll(d1[0]), ll(d2[-1]), [ll(d2[10])])
    assert [x["kind"] for x in p["gaps"]] == ["explicit_gap"]
    # another device: no gap edge
    d1, d2 = _two_drives(100, 2, device2="DEV2")
    assert not PC.build_graph(d1 + d2).gaps
    # explicit gap limited to 300 m
    d1, d2 = _two_drives(400, 20)
    g = PC.build_graph(d1 + d2)
    raises(lambda: PC.find_path(g, ll(d1[0]), ll(d2[-1]), [ll(d2[10])]), "no surveyed path")


# ------------------------------------------------------------------ 3. refusals
def test_refusals():
    d = drive("Q", polyline([(0, 0), (500, 0)]))
    g = PC.build_graph(d)
    lat, lon = to_ll(0, 45)
    raises(lambda: PC.find_path(g, {"lat": lat, "lon": lon}, ll(d[-1])), "limit 40 m")
    lat, lon = to_ll(0, 35)
    PC.find_path(g, {"lat": lat, "lon": lon}, ll(d[-1]))      # 35 m is accepted
    raises(lambda: PC.find_path(g, ll(d[3]), ll(d[3])), "same point")
    raises(lambda: PC.find_path(g, ll(d[3]), ll(d[10])), "at least 50 m")
    raises(lambda: PC.find_path(g, ll(d[0]), ll(d[-1]), [ll(d[1])] * 9), "at most 8 via")


# ------------------------------------------------------------------ 4. via and symmetry
def test_via_points_and_swap_symmetry():
    # a square block: A at the SW corner, B at the NE corner; two equal ways round
    south = drive("V1", polyline([(0, 0), (300, 0), (300, 300)]))
    west = drive("V2", polyline([(0, 0), (0, 300), (300, 300)]), t0=T0 + timedelta(hours=3))
    g = PC.build_graph(south + west)
    via = ll(west[60])                  # on the west side
    p = PC.find_path(g, ll(south[0]), ll(south[-1]), [via])
    sessions = {g.nodes[i].session_id for i in p["nodes"]}
    assert "V2" in sessions
    labels = {k: ["Potholes"] for k in range(10, 14)} | {k: ["Transverse Cracking"] for k in range(70, 80)}
    d = drive("W", polyline([(0, 0), (1230, 0)]), labels=labels, status={k: "skip" for k in range(100, 110)})
    g = PC.build_graph(d)
    r1 = PC.build_report(g, {"a": ll(d[0]), "b": ll(d[-1])})
    r2 = PC.build_report(g, {"a": ll(d[-1]), "b": ll(d[0])})
    assert abs(r1["summary"]["length_m"] - r2["summary"]["length_m"]) < 1e-6
    q1, q2 = r1["stretch"]["quantities"], r2["stretch"]["quantities"]
    for key in ("classified_m", "skip_m", "imaged_m"):
        assert abs(q1[key] - q2[key]) < 1e-9, key
    for p_ in ("cracking", "potholes"):
        assert abs(q1["extent"][p_]["lo"] - q2["extent"][p_]["lo"]) < 1e-9
    assert q1["runs"] == q2["runs"] == {"potholes": 1, "cracking": 1}
    assert len(r1["sections"]) == len(r2["sections"]) == 12        # 1230 m: tail 30 m merged
    assert r1["path"]["passes"][0]["direction"] == "towards B"
    assert r2["path"]["passes"][0]["direction"] == "towards A"


# ------------------------------------------------------------------ 5. attribution
def test_u_turn_and_loop_attribution():
    # out 400 m, U-turn (r 4 m), back 400 m on the same road 8 m to the side
    pts = polyline([(0, 0), (400, 0)] + arc(400, 4, 4, -90, 90)[1:] + [(0, 8)])
    d = drive("U", pts, labels={k: ["Potholes"] for k in range(len(pts) - 30, len(pts) - 25)})
    g = PC.build_graph(d)
    a, b = ll(d[5]), ll(d[70])
    p = PC.find_path(g, a, b)
    passes = PC._path_passes(g, p)
    assert len(passes) == 1
    same, opp = PC._attach_other(g, p, passes)
    assert not same, "the return leg is the opposite direction"
    rep = PC.build_report(g, {"a": a, "b": b})
    assert rep["stretch"]["quantities"]["extent"]["potholes"]["lo"] == 0.0     # return leg not rated
    # the return leg is 8 m away: beyond the 6 m same-road link, so not listed either; at 4 m it is listed
    pts = polyline([(0, 0), (400, 0)] + arc(400, 2, 2, -90, 90)[1:] + [(0, 4)])
    d = drive("U2", pts)
    g = PC.build_graph(d)
    p = PC.find_path(g, ll(d[5]), ll(d[70]))
    same, opp = PC._attach_other(g, p, PC._path_passes(g, p))
    assert opp and not same
    assert all(0 < m < p["length_m"] for o in opp for _, m in o["segs"])
    # a loop: the drive goes round a block and drives the first side again (same direction, 3 m to the side)
    loop = [(0, 0), (300, 0), (300, 200), (-50, 200), (-50, 3), (300, 3)]
    d = drive("L", polyline(loop), labels={k: ["Alligator Cracking"] for k in range(270, 300)})
    g = PC.build_graph(d)
    p = PC.find_path(g, ll(d[2]), ll(d[50]))
    same, opp = PC._attach_other(g, p, PC._path_passes(g, p))
    assert len(same) == 1 and not opp
    ms = [m for _, m in same[0]["segs"]]
    assert min(ms) >= 0 and max(ms) <= p["length_m"]
    rep = PC.build_report(g, {"a": ll(d[2]), "b": ll(d[50])})
    chosen = {s["chosen_pass"]["pass"] for s in rep["sections"] if s["chosen_pass"]}
    assert chosen <= {"P0", "O0"}
    assert all(s["also_surveyed"] for s in rep["sections"][:2])


def test_overlap_choice_most_classified_then_latest():
    road = polyline([(0, 0), (600, 0)])
    early = drive("E1", road, status={k: "skip" for k in range(0, 60)})                 # sections 0-2 skipped
    late = drive("E2", [(x, y + 2, h) for x, y, h in road], t0=T0 + timedelta(days=2),
                 labels={k: ["Potholes"] for k in range(80, 83)})
    g = PC.build_graph(early + late)
    rep = PC.build_report(g, {"a": ll(early[0]), "b": ll(early[-1])})
    secs = rep["sections"]
    assert secs[0]["chosen_pass"]["session_id"] == "E2"      # E1 skipped: less classified
    assert secs[4]["chosen_pass"]["session_id"] == "E2"      # equal classified: latest date
    assert secs[4]["also_surveyed"][0]["session_id"] == "E1"
    assert rep["stretch"]["quantities"]["runs"]["potholes"] == 1


# ------------------------------------------------------------------ 6. units and formatting
def test_sections_and_chainage_text():
    assert PC.sections_of(1230.0)[-1] == (1100.0, 1230.0)
    assert PC.sections_of(1260.0)[-1] == (1200.0, 1260.0)
    assert PC.sections_of(70.0) == [(0.0, 70.0)]
    assert PC.sections_of(1000.0)[-1] == (900.0, 1000.0) and len(PC.sections_of(1000.0)) == 10
    assert PC.fmt_chainage(12350.4) == "Km 12+350" and PC.fmt_chainage(0) == "Km 0+000"
    assert PC.fmt_chainage(999.6) == "Km 1+000"
    assert PC.road_chainage(350, 12.0, True)["text"] == "Km 12+350 (12.350)"
    assert PC.road_chainage(350, 12.0, False)["text"] == "Km 11+650 (11.650)"
    assert PC.road_chainage(350, None, True) is None
    d = drive("K", polyline([(0, 0), (2450, 0)]))
    g = PC.build_graph(d)
    rep = PC.build_report(g, {"a": ll(d[0]), "b": ll(d[-1]), "chainage_at_a_km": 12.0,
                              "chainage_increases_towards_b": True})
    assert [k["km"] for k in rep["km"]] == [0, 1, 2] and rep["km"][-1]["part_km"]
    assert rep["header"]["road_chainage_from"]["text"] == "Km 12+000 (12.000)"
    assert "calibrated by requester" in rep["header"]["road_chainage_note"]
    assert rep["summary"]["sections_by_class"]
    for u in rep["sections"] + rep["km"] + [rep["stretch"]]:
        rt = u["rating"]
        assert rt["rated"] and rt["final"]["class_lo"] in ("Poor", "Fair", "Good")


TESTS = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    failures = []
    for t in TESTS:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception:
            failures.append(t.__name__)
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - len(failures)}/{len(TESTS)} tests passed")
    sys.exit(1 if failures else 0)
