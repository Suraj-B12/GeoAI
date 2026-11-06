"""
Tests for app/pci_report_pdf.py: the PDF has every section of DESIGN §8, no two text spans overlap on any
page, and no text leaves the page.

The PDF is parsed with PyMuPDF. It is not in the project venv: set PYMUPDF_PATH to a folder that holds it
(default: the session scratchpad copy named in eval_results/pci/STATUS_backend.md). Without it the layout
tests are SKIPPED with a message (the build test still runs).

Span boxes: x from the span bbox, y from the baseline origin (ascent 0.78 x size above, descent 0.22 x size
below), i.e. the inked area, so tight-but-legal line spacing is not reported as overlap.

Reports used: synthetic drives (pure, no database): a long one with every kind of section (distress,
skips, not covered, a flagged gap, two passes, calibrated chainage, a 120-character name
of long words) and a 60 m stretch; no images (annex shows "image not available").

Usage:
    venv/Scripts/python.exe scripts/tests/test_pci_pdf.py
    venv/Scripts/python.exe -m pytest scripts/tests/test_pci_pdf.py -q
"""
from __future__ import annotations

import io
import os
import sys
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
                    # trim leading/trailing blanks from the x extent
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


LONG_NAME = ("Outer Ring Road service lane between Silk Board junction and the Agara flyover approach "
             "(eastbound) KarnatakaRoadDevelopmentCorporation")[:120]


def _pdfs():
    return {
        "long": PDF.render(_report(True, road_name="Hosur Road (Bengaluru)", road_number="SH-17",
                                   surface_type="BC", prepared_by="Test reviewer", chainage_at_a_km=12.0,
                                   chainage_increases_towards_b=True)),
        "long_name": PDF.render(_report(True, road_name=LONG_NAME, category="highway", lane_width_m=7.0)),
        "short": PDF.render(_report(False, category="rural")),
    }


_cache: dict = {}


def pdfs():
    if not _cache:
        _cache.update(_pdfs())
    return _cache


def _need_fitz():
    if fitz is None:
        msg = "PyMuPDF not found (set PYMUPDF_PATH): PDF layout tests skipped"
        if pytest is not None and __name__ != "__main__":
            pytest.skip(msg)
        print("SKIP:", msg)
        return False
    return True


def test_builds_and_is_pdf():
    for name, data in pdfs().items():
        assert data[:5] == b"%PDF-" and len(data) > 5000, name


def test_sections_present():
    if not _need_fitz():
        return
    doc = fitz.open(stream=pdfs()["long"], filetype="pdf")
    text = "\n".join(p.get_text() for p in doc)
    for s in ("1. Header (Proforma-1 fields)", "2. Summary", "3. Route", "4. Strip chart", "5. Kilometre-wise",
              "6. Rating per Appendix 4", "7. 100 m sub-sections", "8. Candidate treatments", "9. Photo annex",
              "10. Method", "11. Limitations", "12. Sign-off", "Recorded by", "Checked by", "not assessed",
              "entered by requester, unverified", "Km 12+000 (12.000)", "Drainage", "Total",
              "to be confirmed by field inspection", "label unvalidated", "straight line"):
        assert s in text, s
    assert any(p.rect.width > p.rect.height for p in doc), "no landscape page"
    fonts = {f[3] for p in doc for f in p.get_fonts()}
    assert any("Segoe" in f for f in fonts) or not PDF.FONT_DIR.exists(), fonts      # embedded TTF


def test_no_overlap_and_inside_page():
    if not _need_fitz():
        return
    for name, data in pdfs().items():
        doc = fitz.open(stream=data, filetype="pdf")
        probs = overlaps(doc)
        assert not probs, f"{name}: {len(probs)} problems, e.g. " + "; ".join(probs[:5])


def test_unsupported_text():
    assert PDF.unsupported_text("Hosur Road") is None
    # no complex-script shaping here: Kannada would print wrong letters, so it is refused with a message
    assert "English letters" in PDF.unsupported_text("ಹೊಸೂರು ರಸ್ತೆ")
    assert PDF.unsupported_text("सड़क") is not None
    assert PDF.unsupported_text("道路") is not None


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
