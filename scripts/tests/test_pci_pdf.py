"""
Tests for app/pci_report_pdf.py (IRC:82-2023 PCI report): every section is there, no 2015 content is left, the
headline PCI equals the sum of weight x PCI_i printed in the parameter table, no two text spans overlap on any
page and no text leaves the page — in both states of the report:

  * range state (today): only cracking and potholes have values, the PCI is a range ("No single PCI yet");
  * single state: a parameter calibration (fake param_calibration.json via PCI_PARAM_CALIBRATION) + probe
    scores and calibrated IRI on synthetic segments + an entered rut depth give every parameter a value, so the
    single PCI, its Table 5.5 class and recommendation are printed.

The PDF is parsed with PyMuPDF. It is not in the project venv: set PYMUPDF_PATH to a folder that holds it
(default: the session scratchpad copy named in eval_results/pci/STATUS_backend.md). Without it the layout
tests are SKIPPED with a message (the build tests still run).

Span boxes: x from the span bbox, y from the baseline origin (ascent 0.78 x size above, descent 0.22 x size
below), i.e. the inked area, so tight-but-legal line spacing is not reported as overlap.

Reports used: synthetic drives (pure, no database): a long one with every kind of section (distress, skips,
not covered, a flagged gap, two passes, calibrated road chainage, a 120-character name of long words), a 60 m
rural stretch (pothole PCI not computable for rural roads) and the calibrated one; no images (annex shows
"image not available").

Usage:
    venv/Scripts/python.exe scripts/tests/test_pci_pdf.py [out_dir]     # out_dir: also write the PDFs
    venv/Scripts/python.exe -m pytest scripts/tests/test_pci_pdf.py -q
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
from datetime import timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_PYMUPDF = (r"C:\Users\PRO-LA~1\AppData\Local\Temp\claude\69f5d1fa5ca9c74f"
                   r"\8a7b1584-25d6-4c77-a4ce-be4c2864d62e\scratchpad\pdfpy")
_p = os.environ.get("PYMUPDF_PATH") or DEFAULT_PYMUPDF
if os.path.isdir(_p) and _p not in sys.path:
    sys.path.append(_p)               # appended: the venv's own numpy etc. stay first
try:
    import pymupdf as fitz  # noqa: E402
except ImportError:
    try:
        import fitz  # noqa: E402
    except ImportError:
        fitz = None

try:
    import pytest
except ImportError:
    pytest = None

from app import pci_chainage as PC  # noqa: E402
from app import pci_core as C  # noqa: E402
from app import pci_irc2023 as P23  # noqa: E402
from app import pci_report_pdf as PDF  # noqa: E402


def overlaps(doc) -> list[str]:
    """Problems: pairs of overlapping text spans, and spans outside the page."""
    out = []
    for pno, page in enumerate(doc):
        W, H = page.rect.width, page.rect.height
        boxes = []
        for b in page.get_text("dict")["blocks"]:
            for line in b.get("lines", []):
                for sp in line["spans"]:
                    t = sp["text"]
                    if not t.strip():
                        continue
                    x0, _, x1, _ = sp["bbox"]
                    oy, size = sp["origin"][1], sp["size"]
                    box = (x0, oy - 0.78 * size, x1, oy + 0.22 * size)
                    if box[0] < -0.5 or box[2] > W + 0.5 or box[1] < -0.5 or box[3] > H + 0.5:
                        out.append(f"page {pno + 1}: outside the page: {t[:40]!r} {box}")
                    boxes.append((box, t))
        for i in range(len(boxes)):
            (a, ta) = boxes[i]
            for j in range(i + 1, len(boxes)):
                (b, tb) = boxes[j]
                ox = min(a[2], b[2]) - max(a[0], b[0])
                oy = min(a[3], b[3]) - max(a[1], b[1])
                if ox > 0.6 and oy > 0.6:
                    out.append(f"page {pno + 1}: {ta[:30]!r} overlaps {tb[:30]!r} ({ox:.1f} x {oy:.1f} pt)")
    return out


# ------------------------------------------------------------------ synthetic reports
PROBE_FP = "0123456789abcdef"
CALIBRATION = {
    "schema": 2, "edition": P23.EDITION, "probe_fp": PROBE_FP,
    "fitted_on": {"survey": "synthetic test survey", "sections": 10},
    "params": {
        "cracking": {"source": "labels", "b": 0.6, "evidence": "test calibration"},
        "potholes": {"source": "labels", "b": 1.5, "evidence": "test calibration"},
        "ravelling": {"source": "probe", "keys": ["Ravelling"], "threshold": 0.5, "a": 0.0, "b": 40.0,
                      "evidence": "test calibration"},
        "patching": {"source": "probe", "keys": ["Patching"], "threshold": 0.5, "a": 0.0, "b": 30.0,
                     "evidence": "test calibration"},
        "rut_depth": {"source": "probe", "keys": ["Rutting"], "threshold": 0.5, "a": 2.0, "b": 20.0,
                      "evidence": "test calibration"},
    },
}


def _report(long=True, **req):
    from test_pci_chainage import T0, drive, ll, polyline
    if long:
        labels = {k: ["Potholes"] for k in (20, 21, 22, 300, 301)}
        labels |= {k: ["Longitudinal Cracking", "Transverse Cracking"] for k in range(40, 90)}
        labels |= {k: ["Alligator Cracking", "Bleeding"] for k in range(400, 420)}
        labels |= {k: ["Edge Breaking"] for k in range(500, 505)}
        status = {k: "skip" for k in range(150, 230)} | {k: "not_covered" for k in range(600, 640)}
        d1 = drive("20261027T100000Z_DEV_aaaa", polyline([(0, 0), (2000, 0), (2000, 1500)]), labels=labels,
                   status=status)
        t2 = T0 + timedelta(seconds=len(d1) + 120)
        d2 = drive("20261027T101500Z_DEV_bbbb", polyline([(2000, 1620), (2000, 2600), (1200, 2600)]), t0=t2,
                   labels={k: ["Potholes"] for k in range(30, 34)})
        recs = d1 + d2
        a, b = ll(d1[0]), ll(d2[-1])
    else:
        d1 = drive("20261027T100000Z_DEV_cccc", polyline([(0, 0), (60, 0)]))
        recs, a, b = d1, ll(d1[0]), ll(d1[-1])
    g = PC.build_graph(recs)
    return PC.build_report(g, {"a": a, "b": b, **req})


def _calibrated_report():
    """1.25 km urban drive where every parameter has a value: cracking + potholes from labels with a calibrated
    scale (uncalibrated label values never give a single PCI: pci_core 'off scale'); ravelling,
    patching and rut depth from (fake) calibrated probe scores on the segments; IRI entered by the requester
    (the accelerometer IRI calibration is app/pci_iri's, tested in test_pci_iri.py)."""
    from test_pci_chainage import drive, ll, polyline
    labels = {k: ["Potholes"] for k in (30, 31, 160)}
    labels |= {k: ["Longitudinal Cracking"] for k in range(60, 75)}
    labels |= {k: ["Alligator Cracking"] for k in range(200, 206)}
    d = drive("20261027T120000Z_DEV_dddd", polyline([(0, 0), (900, 0), (900, 350)]), labels=labels)
    for rec in d:
        k = rec["props"]["k"]
        rec["props"]["probe"] = {"Ravelling": 0.8 if 100 <= k < 130 else 0.1,
                                 "Patching": 0.7 if 180 <= k < 190 else 0.05,
                                 "Rutting": 0.9 if 220 <= k < 235 else 0.2}
    tmp = Path(tempfile.mkdtemp(prefix="pci_cal_")) / "param_calibration.json"
    tmp.write_text(json.dumps(CALIBRATION), encoding="utf-8")
    old = os.environ.get("PCI_PARAM_CALIBRATION")
    os.environ["PCI_PARAM_CALIBRATION"] = str(tmp)
    try:
        g = PC.build_graph(d)
        rep = PC.build_report(g, {"a": ll(d[0]), "b": ll(d[-1]), "category": "urban", "surface": "BC",
                                  "entered": {"iri": 2.8}, "road_name": "Calibrated test road"})
    finally:
        if old is None:
            os.environ.pop("PCI_PARAM_CALIBRATION", None)
        else:
            os.environ["PCI_PARAM_CALIBRATION"] = old
    return rep


def _estimate_report():
    """1.25 km urban drive in the state the production reports are in since 2026-11-09: per-type probe scores on
    every segment (cracking / ravelling / patching extents from app.pci_extent), IRI entered (the phone-IMU IRI
    needs app.pci_iri features), rut depth from its prior -> every 100 m section has a PCI estimate with a 90 %
    interval, summary.pci_estimate is set. Two pothole runs; the first is marked impact_felt (no IMU here)."""
    from test_pci_chainage import drive, ll, polyline
    labels = {k: ["Potholes"] for k in (30, 31, 160)}
    labels |= {k: ["Longitudinal Cracking", "Bleeding"] for k in range(60, 75)}
    labels |= {k: ["Alligator Cracking", "Edge Breaking"] for k in range(200, 206)}
    d = drive("20261027T130000Z_DEV_eeee", polyline([(0, 0), (900, 0), (900, 350)]), labels=labels)
    keys = sorted({k for ks in C.PROBE_KEYS.values() for k in ks})
    for rec in d:
        k = rec["props"]["k"]
        rec["props"]["probe"] = {key: (0.75 if (60 <= k < 75 and "Crack" in key) or (100 <= k < 120 and
                                                                                     key == "Ravelling")
                                       else 0.12) for key in keys}
    g = PC.build_graph(d)
    rep = PC.build_report(g, {"a": ll(d[0]), "b": ll(d[-1]), "category": "urban", "entered": {"iri": 3.2},
                              "road_name": "Estimate test road"})
    runs = rep["summary"].get("pothole_runs_list") or []
    if runs:
        runs[-1]["impact_felt"], runs[-1]["jolt"] = True, 4.2
        if len(runs) > 1:
            runs[0]["impact_felt"], runs[0]["jolt"] = False, 1.1
    return rep


LONG_NAME = ("Outer Ring Road service lane between Silk Board junction and the Agara flyover approach "
             "(eastbound) KarnatakaRoadDevelopmentCorporation")[:120]

_cache: dict = {}


def reports():
    if not _cache:
        _cache["long"] = _report(True, road_name="Hosur Road (Bengaluru)", road_number="SH-17",
                                 surface_type="BC", prepared_by="Test reviewer", chainage_at_a_km=12.0,
                                 chainage_increases_towards_b=True)
        _cache["long_name"] = _report(True, road_name=LONG_NAME, category="highway", lane_width_m=7.0)
        _cache["short"] = _report(False, category="rural")
        _cache["calibrated"] = _calibrated_report()
        _cache["estimate"] = _estimate_report()
    return _cache


_pdfs: dict = {}


def pdfs():
    if not _pdfs:
        _pdfs.update({k: PDF.render(r) for k, r in reports().items()})
    return _pdfs


def _need_fitz():
    if fitz is None:
        msg = "PyMuPDF not found (set PYMUPDF_PATH): PDF layout tests skipped"
        if pytest is not None and __name__ != "__main__":
            pytest.skip(msg)
        print("SKIP:", msg)
        return False
    return True


def _text(name):
    """All text of a PDF, whitespace (incl. line wraps) collapsed to single spaces."""
    return " ".join(" ".join(p.get_text() for p in fitz.open(stream=pdfs()[name], filetype="pdf")).split())


# ------------------------------------------------------------------ tests
def test_builds_and_is_pdf():
    for name, data in pdfs().items():
        assert data[:5] == b"%PDF-" and len(data) > 5000, name


def test_report_states():
    """The synthetic inputs reach the two states the PDF must handle."""
    r = reports()
    assert r["long"]["summary"]["pci"]["single"] is None
    cal = r["calibrated"]
    s = cal["summary"]["pci"]
    assert s["single"] is not None and s["single_class"] in PDF.CLASS_COLORS, s
    srcs = {p["param"]: p["source"] for p in cal["stretch"]["pci"]["params"]}
    assert srcs == {"cracking": "labels_calibrated", "ravelling": "probe", "potholes": "labels_calibrated",
                    "patching": "probe",
                    "rut_depth": "probe", "iri": "entered"}, srcs


def test_headline_equals_weighted_sum():
    """sum(weight x PCI_i) over the printed parameter rows reproduces the headline (lo, hi and single)."""
    for name, rep in reports().items():
        pc = rep["summary"]["pci"]
        if not pc:
            continue
        rows = PDF.stretch_param_rows(rep)
        lo = sum(r["weight"] * (r["mean_worst"] if r["mean_known"] else 0.0) for r in rows)
        hi = sum(r["weight"] * (r["mean_best"] if r["mean_known"] else 100.0) for r in rows)
        assert abs(lo - pc["lo"]) < 0.02 and abs(hi - pc["hi"]) < 0.02, (name, lo, hi, pc)
        if pc["single"] is not None:
            pt = sum(r["weight"] * r["mean_point"] for r in rows)
            assert abs(pt - pc["single"]) < 0.02, (name, pt, pc["single"])


def test_sections_present_range_state():
    if not _need_fitz():
        return
    text = _text("long")
    for s in ("Road condition report", "1. Summary", "PCI by parameter", "Key findings", "Sign-off",
              "Field-checked by", "Approved by", "2. Route", "2a. Route detail per km",
              "3. Condition by km and by 100 m section", "Kilometre-wise", "100 m sections",
              "4. Pothole locations", "5. Maintenance (IRC:82-2023)", "Recommendation (Table 5.5)",
              "Repairs for the distress found", "7.5.3.5", "6. Method and limitations", "One lane",
              "Annex — Photographs", "IRC:82-2023", "No single PCI yet", "Not assessed:",
              "% of the PCI weight", "Hosur Road (Bengaluru)", "SH-17", "as entered, unverified",
              "Km 12+000 (12.000)", "Road chainage", "Distress range (Table 5.3)", "PCI range",
              "Poor to Fail", "gap, not surveyed", "Whole stretch", "Table 10.1"):
        assert s in text, s
    for gone in ("IRC:82-2015", "Appendix 4", "Proforma", "E_lo", "E_hi", "Strip chart", "secondary"):
        assert gone not in text, gone
    assert all(p.rect.width < p.rect.height for p in fitz.open(stream=pdfs()["long"], filetype="pdf"))
    fonts = {f[3] for p in fitz.open(stream=pdfs()["long"], filetype="pdf") for f in p.get_fonts()}
    assert any("Segoe" in f for f in fonts) or not PDF.FONT_DIR.exists(), fonts      # embedded TTF


def test_single_state():
    if not _need_fitz():
        return
    rep = reports()["calibrated"]
    text = _text("calibrated")
    pc = rep["summary"]["pci"]
    assert f"PCI {pc['single']:.1f}" in text, pc
    assert "Recommended (Table 5.5)" in text and PDF.T.class_recommendation(pc["single_class"]) in text
    for s in ("Ravelling (%)", "Patching (%)", "Rut depth (mm)", "image probe, calibrated on survey",
              "entered by requester (unverified)",
              "Entered for the whole stretch (unverified): roughness (IRI) 2.80 m/km", "Calibration.",
              "synthetic test survey", "Excellent (>90 to 100)", "Calibrated test road"):
        assert s in text, s
    for gone in ("No single PCI yet", "Not assessed:", "Distress range (Table"):
        assert gone not in text, gone


def test_estimate_state():
    """Headline 'PCI ≈ m (90 % interval lo–hi)', parameter table with share of interval and short basis, km and
    100 m estimate columns, map legend 'colour = median estimate', impact-felt pothole list (confirmed first),
    other distresses with 2023 clauses, scooter wording."""
    rep = reports()["estimate"]
    e = rep["summary"].get("pci_estimate")
    assert e and e["lo90"] <= e["median"] <= e["hi90"], e
    rows = PDF.stretch_estimate_rows(rep)
    assert abs(sum(r["share"] for r in rows) - 100.0) < 0.5
    assert {r["param"] for r in rows if r.get("point") is not None} == set(P23.PARAMS)
    for r in rows:
        assert len(r["basis"].split()) <= 8, r["basis"]
    if not _need_fitz():
        return
    text = _text("estimate")
    for s in (f"PCI ≈ {e['median']:.0f}", f"90 % interval {e['lo90']:.0f}–{e['hi90']:.0f}",
              "Recommended for the median class (Table 5.5)", "The interval is mostly", "Worst case",
              "Share of interval", "literature prior 0–25 mm", "entered by requester (unverified)",
              "PCI ≈ (90 % interval)", "Class (median)", "colour = median estimate", "Impact felt",
              "potholes the rider steered around show no jolt", "Other distresses seen (not part of the IRC PCI)",
              "Bleeding", "7.2.1.5", "Edge Breaking", "7.5.5.6", "scooter's handlebar", "An estimate, not a measurement"):
        assert s in text, s
    for gone in ("windshield", "No single PCI yet", "Distress range (Table"):
        assert gone not in text, gone
    i_yes, i_no = text.index(" yes "), text.index(" no ", text.index("Impact felt"))
    assert i_yes < i_no, "confirmed potholes are listed first"


def test_rural_pothole_not_computable():
    if not _need_fitz():
        return
    text = _text("short")
    assert "not computable" in text and "MDR / rural pothole equation" in text


def test_no_overlap_and_inside_page():
    if not _need_fitz():
        return
    for name, data in pdfs().items():
        doc = fitz.open(stream=data, filetype="pdf")
        probs = overlaps(doc)
        assert not probs, f"{name}: {len(probs)} problems, e.g. " + "; ".join(probs[:5])


def test_treatments_2023():
    """Treatment rows carry IRC:82-2023 clauses; Table 5.5 has the six classes in order."""
    T = PDF.T
    assert [c for c, _, _ in T.TABLE_5_5] == ["Excellent", "Good", "Satisfactory", "Fair", "Poor", "Fail"]
    rows = T.distress_rows(["Transverse Cracking", "Potholes", "Longitudinal Cracking"],
                           {"Longitudinal Cracking": "High"})
    assert [r["label"] for r in rows] == ["Potholes", "Longitudinal Cracking", "Transverse Cracking"]
    assert rows[0]["clause"].startswith("7.5.3.5") and rows[1]["clause"] == "7.3.4.5"
    assert "overlay" in rows[1]["treatment"]
    for c, _r, rec in T.TABLE_5_5:      # same classes as the PCI code
        assert P23.pci_class({"Excellent": 95, "Good": 85, "Satisfactory": 70, "Fair": 50, "Poor": 30,
                              "Fail": 10}[c]) == c


def test_colour_key():
    rep = reports()["long"]
    keys = {PDF.colour_key(rep, s)[1] for s in rep["sections"]}
    assert keys <= set(PDF.RANGE_COLORS) | {"Insufficient data"}, keys
    cal = reports()["calibrated"]
    keys = {PDF.colour_key(cal, s)[1] for s in cal["sections"]}
    assert keys <= set(PDF.CLASS_COLORS) | {"Insufficient data", "PCI not available here (a parameter missing)"}


def test_unsupported_text():
    assert PDF.unsupported_text("Hosur Road") is None
    # no complex-script shaping here: Kannada would print wrong letters, so it is refused with a message
    assert "English letters" in PDF.unsupported_text("ಹೊಸೂರು ರಸ್ತೆ")
    assert PDF.unsupported_text("सड़क") is not None
    assert PDF.unsupported_text("道路") is not None


def test_basemap_projection():
    from app import pci_basemap as B
    x, y = B.world_px(0.0, 0.0, 0)
    assert abs(x - 128) < 1e-9 and abs(y - 128) < 1e-9          # centre of the single z0 tile
    import math
    lon, lat, z = 77.5946, 12.9716, 15                           # Bengaluru; OSM wiki slippy-map formula:
    tx = int((lon + 180) / 360 * 2 ** z)
    ty = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * 2 ** z)
    x, y = B.world_px(lon, lat, z)
    assert (int(x // 256), int(y // 256)) == (tx, ty) == (23446, 15193), (x // 256, y // 256, tx, ty)


def test_offline_route_figure_still_renders():
    """Without the street map (offline / PCI_BASEMAP=0) the PDF still builds and says the map is missing."""
    old = os.environ.get("PCI_BASEMAP")
    os.environ["PCI_BASEMAP"] = "0"
    try:
        data = PDF.render(_report(False, category="rural"))
    finally:
        if old is None:
            os.environ.pop("PCI_BASEMAP", None)
        else:
            os.environ["PCI_BASEMAP"] = old
    assert data[:5] == b"%PDF-"
    if fitz is not None:
        text = "".join(p.get_text() for p in fitz.open(stream=data, filetype="pdf"))
        assert "street map unavailable" in text


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
    if len(sys.argv) > 1:        # write the PDFs for a visual check
        for name, data in pdfs().items():
            Path(sys.argv[1], f"test_{name}.pdf").write_bytes(data)
    print(f"\n{len(TESTS) - len(failures)}/{len(TESTS)} tests passed")
    sys.exit(1 if failures else 0)
