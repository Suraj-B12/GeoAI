"""
A4 PDF of an A->B road condition report under IRC:82-2023 (Pavement Condition Index, PCI 0-100), from the report
JSON of app.pci_chainage.build_report. Written for a civil / road engineer: what the road is, its PCI (one value,
or the range the unassessed parameters allow), where the potholes are, what IRC:82-2023 recommends, and what
limits the numbers. ReportLab only; text in Paragraphs so it wraps; fonts embedded (Segoe UI; Nirmala UI kept
for Indic text). scripts/tests/test_pci_pdf.py checks the sections, that no two text spans overlap and that no
text leaves the page.

    render(report, image_fetcher=None) -> bytes      image_fetcher(segment_id) -> JPEG bytes | None
    unsupported_text(s) -> str | None                why the PDF cannot print s (None = fine)

Contents: 1 Summary (header, PCI, parameter table, key findings, sign-off) · 2 Route map · 3 Km-wise and 100 m
tables · 4 Pothole locations · 5 Maintenance (Table 5.5, Table 10.1, Section 7 treatments) · 6 Method and
limitations · Annex: photographs.
"""
from __future__ import annotations

import io
import math
import unicodedata
from pathlib import Path
from typing import Callable, Optional
from xml.sax.saxutils import escape

from reportlab.graphics.shapes import Circle, Drawing, Line, PolyLine, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Frame, Image, KeepTogether, PageBreak, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle)

from app import pci_core as C
from app import pci_irc2023 as P23
from app import pci_treatments_2023 as T

FONT_DIR = Path(r"C:\Windows\Fonts")
MARGIN = 14 * mm

# Table 5.5 classes (PCI) and Tables 5.1-5.3 distress ranges
CLASS_COLORS = {"Excellent": "#166534", "Good": "#4d7c0f", "Satisfactory": "#a16207", "Fair": "#c2410c",
                "Poor": "#b91c1c", "Fail": "#7f1d1d"}
RANGE_COLORS = {"Excellent to Good": "#15803d", "Satisfactory to Fair": "#d97706", "Poor to Fail": "#b91c1c"}
NO_DATA = "#9ca3af"
PART_DATA = "#a8a29e"
INK = colors.HexColor("#1c1917")
MUTED = colors.HexColor("#57534e")
RULE = colors.HexColor("#d6d3d1")
HEAD_BG = colors.HexColor("#f5f5f4")
TOTAL_BG = colors.HexColor("#ecebe9")
ROAD_CLASS_NAMES = {"urban": "Urban road", "highway": "Highway (NH / SH)", "mdr": "Major District Road",
                    "rural": "Rural road"}
APPENDIX_TABLE = {"highway": "A2.1", "mdr": "A2.2", "rural": "A2.2", "urban": "A2.3"}
LOWER_NAMES = {"cracking": "cracking", "ravelling": "ravelling", "potholes": "potholes", "patching": "patching",
               "rut_depth": "rut depth", "iri": "roughness (IRI)"}
DISTRESS_TABLE = {"highway": "5.1", "mdr": "5.2", "rural": "5.2", "urban": "5.3"}

# ============================================================ fonts
_fonts = {"regular": "Helvetica", "bold": "Helvetica-Bold", "indic": None, "indic_bold": None}
_font_cmaps: dict[str, set] = {}
_fonts_done = False


def _register_fonts() -> None:
    global _fonts_done
    if _fonts_done:
        return
    _fonts_done = True
    try:
        r = TTFont("SegoeUI", str(FONT_DIR / "segoeui.ttf"))
        b = TTFont("SegoeUI-Bold", str(FONT_DIR / "segoeuib.ttf"))
        pdfmetrics.registerFont(r)
        pdfmetrics.registerFont(b)
        pdfmetrics.registerFontFamily("SegoeUI", normal="SegoeUI", bold="SegoeUI-Bold", italic="SegoeUI",
                                      boldItalic="SegoeUI-Bold")
        _fonts.update(regular="SegoeUI", bold="SegoeUI-Bold")
        _font_cmaps["regular"] = set(r.face.charToGlyph)
    except Exception as e:   # not on Windows: Helvetica (Latin-1 only)
        print(f"[PCI-PDF] Segoe UI unavailable ({type(e).__name__}); using Helvetica")
        _font_cmaps["regular"] = set(range(32, 256))
    try:
        n = TTFont("NirmalaUI", str(FONT_DIR / "Nirmala.ttc"), subfontIndex=0)
        nb = TTFont("NirmalaUI-Bold", str(FONT_DIR / "Nirmala.ttc"), subfontIndex=1)
        pdfmetrics.registerFont(n)
        pdfmetrics.registerFont(nb)
        _fonts.update(indic="NirmalaUI", indic_bold="NirmalaUI-Bold")
        _font_cmaps["indic"] = set(n.face.charToGlyph)
    except Exception as e:
        print(f"[PCI-PDF] Nirmala UI unavailable ({type(e).__name__}); Indic text will be refused")
        _font_cmaps["indic"] = set()


def unsupported_text(s: str) -> Optional[str]:
    """None when every character can be printed with the embedded fonts, else a message."""
    _register_fonts()
    reg, ind = _font_cmaps["regular"], _font_cmaps["indic"]
    bad = sorted({ch for ch in s if ord(ch) >= 32 and ord(ch) not in reg and ord(ch) not in ind
                  and unicodedata.category(ch)[0] != "Z"})
    if bad:
        names = ", ".join(f"U+{ord(c):04X}" for c in bad[:5])
        return (f"characters the PDF cannot print ({names}); please write the text in English letters "
                f"(Kannada and other Indic scripts cannot be shaped by this server), or leave the field empty")
    # ReportLab here has no complex-script shaping (uharfbuzz is not installed): Kannada vowel signs and
    # conjuncts print as detached marks or wrong letters. Refuse rather than print a wrong road name.
    if any(0x0900 <= ord(ch) <= 0x0DFF for ch in s):
        return ("Kannada and other Indic scripts cannot be printed correctly by this server's PDF renderer "
                "(no complex-script shaping); please write the name in English letters")
    return None


def _needs_indic(s: str) -> bool:
    reg = _font_cmaps.get("regular") or set()
    return any(ord(ch) not in reg and ord(ch) >= 128 for ch in s)


def user_text(s: Optional[str], empty: str = "not entered") -> str:
    """Requester-entered text as Paragraph markup (escaped; Indic runs in Nirmala UI)."""
    if not s:
        return f"<font color='#57534e'>{empty}</font>" if empty else ""
    if _needs_indic(s) and _fonts["indic"]:
        return f'<font name="{_fonts["indic"]}">{escape(s)}</font>'
    return escape(s)


# ============================================================ styles / small helpers
def _styles():
    _register_fonts()
    R, B = _fonts["regular"], _fonts["bold"]
    base = dict(fontName=R, textColor=INK, leading=11.2, fontSize=8.6)
    st = {
        "body": ParagraphStyle("body", **base, spaceAfter=3),
        "bullet": ParagraphStyle("bullet", **base, leftIndent=9, bulletIndent=0, spaceAfter=2.2),
        "cell": ParagraphStyle("cell", **{**base, "fontSize": 7.6, "leading": 9.4}),
        "cellb": ParagraphStyle("cellb", **{**base, "fontName": B, "fontSize": 7.6, "leading": 9.4}),
        "cellc": ParagraphStyle("cellc", **{**base, "fontSize": 7.6, "leading": 9.4, "alignment": TA_CENTER}),
        "cellr": ParagraphStyle("cellr", **{**base, "fontSize": 7.6, "leading": 9.4, "alignment": TA_RIGHT}),
        "head": ParagraphStyle("head", **{**base, "fontName": B, "fontSize": 7.4, "leading": 9.0}),
        "headc": ParagraphStyle("headc", **{**base, "fontName": B, "fontSize": 7.4, "leading": 9.0,
                                            "alignment": TA_CENTER}),
        "title": ParagraphStyle("title", **{**base, "fontName": B, "fontSize": 16, "leading": 20}, spaceAfter=1),
        "subtitle": ParagraphStyle("subtitle", **{**base, "fontSize": 10, "leading": 13, "textColor": MUTED},
                                   spaceAfter=5),
        "h2": ParagraphStyle("h2", **{**base, "fontName": B, "fontSize": 11.5, "leading": 14.5}, spaceBefore=9,
                             spaceAfter=4),
        "h3": ParagraphStyle("h3", **{**base, "fontName": B, "fontSize": 9.2, "leading": 12}, spaceBefore=6,
                             spaceAfter=2),
        "big": ParagraphStyle("big", **{**base, "fontName": B, "fontSize": 21, "leading": 25}),
        "bigsub": ParagraphStyle("bigsub", **{**base, "fontName": B, "fontSize": 10.5, "leading": 13.5}),
        "note": ParagraphStyle("note", **{**base, "fontSize": 7.4, "leading": 9.4, "textColor": MUTED},
                               spaceAfter=2),
        "caption": ParagraphStyle("caption", **{**base, "fontSize": 7.2, "leading": 9.0}, alignment=TA_LEFT),
    }
    return st


def P(text, style, **kw) -> Paragraph:
    return Paragraph(text, style, **kw)


def fnum(x, nd=1) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    return f"{x:.{nd}f}"


def frange(lo, hi, nd=1) -> str:
    if lo is None:
        return "–"
    a, b = fnum(lo, nd), fnum(hi, nd)
    return a if a == b else f"{a}–{b}"


def _plural(n, word) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _swatch(color: str) -> str:
    return f'<font color="{color}">■</font>'


def _table(data, widths, *, header_rows=1, zebra=False, extra=None) -> Table:
    t = Table(data, colWidths=widths, repeatRows=header_rows, hAlign="LEFT")
    style = [("GRID", (0, 0), (-1, -1), 0.4, RULE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
             ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    if header_rows:
        style.append(("BACKGROUND", (0, 0), (-1, header_rows - 1), HEAD_BG))
        style.append(("VALIGN", (0, 0), (-1, header_rows - 1), "BOTTOM"))
    if zebra:
        for r in range(header_rows, len(data)):
            if (r - header_rows) % 2 == 1:
                style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#fafaf9")))
    t.setStyle(TableStyle(style + (extra or [])))
    return t


def _ch(m: float) -> str:
    """'0+154' from metres."""
    m = int(round(max(0.0, float(m))))
    return f"{m // 1000}+{m % 1000:03d}"


# ============================================================ report data views
def _road_class(rep) -> str:
    return rep["header"]["category"]


def _upci(u: dict) -> Optional[dict]:
    """The unit's PCI block (lo/hi/single/class...) or None when not rated."""
    return ((u or {}).get("pci") or {}).get("pci")


def _params(u: dict) -> dict:
    return {p["param"]: p for p in ((u or {}).get("pci") or {}).get("params") or []}


def single_mode(rep) -> bool:
    """True when the headline PCI is one value (every parameter has a value on every rated km row)."""
    return bool((rep["summary"].get("pci") or {}).get("single") is not None)


def pci_text(pc: Optional[dict], nd_single=1) -> str:
    if not pc:
        return "–"
    if pc.get("single") is not None:
        return fnum(pc["single"], nd_single)
    return f"{fnum(pc['lo'], 0)}–{fnum(pc['hi'], 0)}"


def class_text(pc: Optional[dict]) -> str:
    if not pc:
        return "insufficient data"
    if pc.get("single") is not None:
        return pc["single_class"]
    return pc["class"]


def _value_point(prm: dict) -> Optional[float]:
    v = prm.get("value_point")
    return v if v is not None else prm.get("value_lo")


def worst_range(u: dict, road_class: str, surface: Optional[str]) -> Optional[str]:
    """Worst Tables 5.1-5.3 range ('Excellent to Good' / 'Satisfactory to Fair' / 'Poor to Fail') among the
    parameters that have a value in this unit (entered stretch-wide values excluded); None if not rated/none."""
    if not _upci(u):
        return None
    order = ("Excellent to Good", "Satisfactory to Fair", "Poor to Fail")
    worst = None
    for p, prm in _params(u).items():
        if prm.get("source") in ("not_assessed", "entered", None):
            continue
        v = _value_point(prm)
        if v is None:
            continue
        b = prm.get("range") or C.band23(road_class, p, v, surface)
        if b in order and (worst is None or order.index(b) > order.index(worst)):
            worst = b
    return worst


def colour_key(rep, u: dict) -> tuple[str, str]:
    """(colour, label) of one unit for the map and the tables."""
    pc = _upci(u)
    if not pc:
        return NO_DATA, "Insufficient data"
    if single_mode(rep):
        if pc.get("single") is None:
            return PART_DATA, "PCI not available here (a parameter missing)"
        return CLASS_COLORS[pc["single_class"]], pc["single_class"]
    w = worst_range(u, _road_class(rep), rep["header"].get("surface"))
    if w is None:
        return NO_DATA, "Insufficient data"
    return RANGE_COLORS[w], w


def stretch_param_rows(rep) -> list[dict]:
    """Headline parameter table: whole-stretch values; PCI_i = length-weighted mean over the rated km rows (the
    same rows and weights as summary.pci, so sum(weight x PCI_i) reproduces the headline)."""
    kms = [k for k in rep["km"] if _upci(k)]
    sp = _params(rep["stretch"])
    W = sum(k["to_m"] - k["from_m"] for k in kms) or 1.0
    rows = []
    for p in P23.PARAMS:
        s = dict(sp.get(p) or {"param": p, "name": P23.PARAM_NAMES[p], "unit": P23.PARAM_UNITS[p],
                               "weight": P23.WEIGHTS[p], "source": "not_assessed", "known": False})
        kp = [(_params(k).get(p) or {}, k["to_m"] - k["from_m"]) for k in kms]
        known = bool(kms) and all(x.get("known") for x, _ in kp)
        if known:
            s["mean_best"] = sum(x["pci_best"] * w for x, w in kp) / W
            s["mean_worst"] = sum(x["pci_worst"] * w for x, w in kp) / W
            s["mean_point"] = sum(x["pci_point"] * w for x, w in kp) / W
        s["mean_known"] = known
        rows.append(s)
    return rows


# ============================================================ page furniture
class _Doc(BaseDocTemplate):
    def __init__(self, buf, report):
        name = report["header"].get("road_name")
        super().__init__(buf, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN + 6 * mm,
                         bottomMargin=MARGIN + 4 * mm,
                         title="Road condition report (IRC:82-2023 PCI)" + (f" — {name}" if name else ""),
                         author="GeoAI pavement survey (automated)", subject="A to B road condition report")
        self.report = report
        pw, ph = A4
        fp = Frame(MARGIN, MARGIN + 4 * mm, pw - 2 * MARGIN, ph - 2 * MARGIN - 10 * mm, id="p", leftPadding=0,
                   rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates([PageTemplate("portrait", [fp], pagesize=A4, onPage=self._furniture)])

    def _furniture(self, canv, doc):
        canv.saveState()
        w, h = canv._pagesize
        canv.setFont(_fonts["regular"], 6.8)
        canv.setFillColor(MUTED)
        rid = self.report.get("report_id")
        canv.drawString(MARGIN, h - MARGIN, "Road condition report — IRC:82-2023 Pavement Condition Index (screening "
                                            "estimate from phone video)")
        canv.drawRightString(w - MARGIN, h - MARGIN, f"Report {rid}" if rid is not None else "Preview (not stored)")
        canv.setStrokeColor(RULE)
        canv.setLineWidth(0.4)
        canv.line(MARGIN, h - MARGIN - 2.2 * mm, w - MARGIN, h - MARGIN - 2.2 * mm)
        canv.line(MARGIN, MARGIN + 2.2 * mm, w - MARGIN, MARGIN + 2.2 * mm)
        canv.drawString(MARGIN, MARGIN - 1 * mm, f"Data as of {self.report.get('data_as_of') or '?'} · generated "
                                                 f"{self.report.get('generated_at_ist')}")
        canv.drawRightString(w - MARGIN, MARGIN - 1 * mm, f"Page {canv.getPageNumber()}")
        canv.restoreState()


# ============================================================ figures
def _fit(vertices, w, h, pad):
    """Local metres -> drawing coordinates, same scale on both axes (no street map)."""
    lat0 = sum(v[1] for v in vertices) / len(vertices)
    kx = 111320.0 * math.cos(math.radians(lat0))
    ky = 110540.0
    xs = [v[0] * kx for v in vertices]
    ys = [v[1] * ky for v in vertices]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    span = max(x1 - x0, y1 - y0, 50.0)
    s = min((w - 2 * pad) / max(x1 - x0, span * 0.02), (h - 2 * pad) / max(y1 - y0, span * 0.02))
    ox = pad + ((w - 2 * pad) - (x1 - x0) * s) / 2
    oy = pad + ((h - 2 * pad) - (y1 - y0) * s) / 2

    def f(lon, lat):
        return ox + (lon * kx - x0) * s, oy + (lat * ky - y0) * s
    return f, s


class _Labels:
    """Places text labels in a drawing without overlaps (and inside the drawing)."""

    def __init__(self, w, h, font, halo=False):
        self.boxes, self.w, self.h, self.font, self.halo = [], w, h, font, halo

    def obstacle_line(self, pts, r=1.3):
        for i in range(0, len(pts) - 2, 2):
            x1, y1, x2, y2 = pts[i], pts[i + 1], pts[i + 2], pts[i + 3]
            n = max(1, int(math.hypot(x2 - x1, y2 - y1) / 3.0))
            for k in range(n + 1):
                x, y = x1 + (x2 - x1) * k / n, y1 + (y2 - y1) * k / n
                self.boxes.append((x - r, y - r, x + r, y + r))

    def obstacle_box(self, x0, y0, x1, y1):
        self.boxes.append((x0, y0, x1, y1))

    def place(self, d: Drawing, x, y, text, size=6.5, color=INK, anchor="start", force=False, bg=False) -> bool:
        bg = bg or self.halo
        tw = pdfmetrics.stringWidth(text, self.font, size)
        x0 = x if anchor == "start" else (x - tw if anchor == "end" else x - tw / 2)
        box = (x0 - 1, y - size * 0.3 - 1, x0 + tw + 1, y + size * 0.95 + 1)
        if box[0] < 0 or box[2] > self.w or box[1] < 0 or box[3] > self.h:
            if not force:
                return False
            dx = -box[0] if box[0] < 0 else (self.w - box[2] if box[2] > self.w else 0)
            dy = -box[1] if box[1] < 0 else (self.h - box[3] if box[3] > self.h else 0)
            x0, y = x0 + dx, y + dy
            box = (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)
        for b in self.boxes:
            if not (box[2] <= b[0] or box[0] >= b[2] or box[3] <= b[1] or box[1] >= b[3]):
                return False
        self.boxes.append(box)
        if bg:   # white label box so the text reads over the street map
            d.add(Rect(box[0], box[1], box[2] - box[0], box[3] - box[1], fillColor=colors.white, fillOpacity=0.88,
                       strokeColor=None))
        d.add(String(x0, y, text, fontName=self.font, fontSize=size, fillColor=color))
        return True


def _point_at(vertices, m):
    for (x1, y1, a), (x2, y2, b) in zip(vertices, vertices[1:]):
        if a <= m <= b:
            t = 0.0 if b == a else (m - a) / (b - a)
            return x1 + t * (x2 - x1), y1 + t * (y2 - y1), (x1, y1, x2, y2)
    x, y, _ = vertices[-1]
    return x, y, (vertices[-2][0], vertices[-2][1], x, y) if len(vertices) > 1 else (x, y, x, y)


def lonlat_at(rep, m: float) -> tuple[float, float]:
    lon, lat, _ = _point_at([[v[0], v[1], v[2]] for v in rep["path"]["vertices"]], m)
    return lon, lat


def _north_and_scale(d, labels, w, h, pt_per_m, choices):
    nx0, ny0 = w - 9 * mm, h - 15 * mm
    d.add(Rect(nx0 - 6, ny0 - 3, 12, 25, fillColor=colors.white, fillOpacity=0.88, strokeColor=None))
    d.add(Polygon([nx0, ny0 + 9, nx0 - 3, ny0, nx0 + 3, ny0], fillColor=INK, strokeColor=INK))
    labels.place(d, nx0, ny0 + 10.5, "N", size=7, anchor="middle", force=True)
    labels.obstacle_box(nx0 - 6, ny0 - 3, nx0 + 6, ny0 + 9)
    target = (w * 0.22) / pt_per_m
    nice = choices[0]
    for c in choices:
        if c <= target:
            nice = c
    bx0, by0 = 6 * mm, 6 * mm
    lab = f"{int(nice)} m" if nice < 1000 else f"{nice / 1000:g} km"
    d.add(Rect(bx0 - 3, by0 - 3, nice * pt_per_m + 6, 15, fillColor=colors.white, fillOpacity=0.88, strokeColor=None))
    d.add(Rect(bx0, by0, nice * pt_per_m, 2.2, fillColor=INK, strokeColor=INK))
    labels.place(d, bx0, by0 + 4, lab, size=6.5, force=True)
    labels.obstacle_box(bx0 - 3, by0 - 3, bx0 + nice * pt_per_m + 3, by0 + 3)


def route_figure(rep: dict, w: float, h: float, m0: Optional[float] = None, m1: Optional[float] = None,
                 title: Optional[str] = None) -> Drawing:
    """The track over a street map (app.pci_basemap), each 100 m section coloured by colour_key(); pothole runs
    as black rings. m0/m1: a detail panel of that measure window (ticks and labels every 100 m)."""
    from app import pci_basemap
    from reportlab.graphics.shapes import Image as GImage

    d = Drawing(w, h)
    L = rep["summary"]["length_m"]
    lo, hi = (0.0, L) if m0 is None else (m0, m1)
    allv = rep["path"]["vertices"]
    verts = [v for v in allv if lo - 1e-6 <= v[2] <= hi + 1e-6] or allv
    try:
        base = pci_basemap.view([(v[0], v[1]) for v in verts], w, h, 14 * mm)
    except Exception:
        base = None
    if base is not None:
        f, s_ = base.xy, base.pt_per_m
        d.add(GImage(0, 0, w, h, base.image_path))
    else:
        f, s_ = _fit(verts, w, h, 14 * mm)
    d.add(Rect(0, 0, w, h, strokeColor=RULE, fillColor=None, strokeWidth=0.6))
    labels = _Labels(w, h, _fonts["regular"], halo=base is not None)
    if title:
        labels.place(d, 4, h - 10, title, size=7, force=True, bg=True)
    parts = []
    for sec in rep["sections"]:
        if sec["to_m"] < lo - 1e-6 or sec["from_m"] > hi + 1e-6:
            continue
        col, _ = colour_key(rep, sec)
        for part in sec["geometry"]["coordinates"]:
            pts = []
            for lon, lat in part:
                pts.extend(f(lon, lat))
            if len(pts) >= 4:
                parts.append((pts, col))
    for pts, _ in parts:      # white casing first, so the colours read on the map
        d.add(PolyLine(pts, strokeColor=colors.white, strokeWidth=5.6, strokeLineCap=1, strokeLineJoin=1))
    for pts, col in parts:
        d.add(PolyLine(pts, strokeColor=colors.HexColor(col), strokeWidth=3.4, strokeLineCap=1, strokeLineJoin=1))
        labels.obstacle_line(pts, r=2.4)
    for gp, part in zip(rep["path"]["gaps"], rep["path"]["geometry"]["gaps"]["coordinates"]):
        if gp["to_m"] < lo or gp["from_m"] > hi:
            continue
        pts = []
        for lon, lat in part:
            pts.extend(f(lon, lat))
        d.add(PolyLine(pts, strokeColor=colors.white, strokeWidth=3.0))
        d.add(PolyLine(pts, strokeColor=INK, strokeWidth=1.2, strokeDashArray=[3, 2]))
        labels.obstacle_line(pts)
    for r in rep["summary"].get("pothole_runs_list") or []:        # pothole runs
        mid = (r["from_m"] + r["to_m"]) / 2
        if not lo - 1e-6 <= mid <= hi + 1e-6:
            continue
        x, y = f(*lonlat_at(rep, mid))
        d.add(Circle(x, y, 2.6, fillColor=colors.white, strokeColor=INK, strokeWidth=1.1))
        labels.obstacle_box(x - 3, y - 3, x + 3, y + 3)
    detail = m0 is not None
    if detail:
        tick, step_lab = 100.0, 100.0
    else:
        tick = 100.0 if L <= 3000 else 500.0
        step_lab = 100.0 if L <= 1200 else 200.0 if L <= 2500 else 500.0 if L <= 6000 else 1000.0
        while L / step_lab > 25:
            step_lab *= 2
    m = math.ceil(lo / tick - 1e-9) * tick
    while m <= hi + 1e-6:
        lon, lat, seg = _point_at([[v[0], v[1], v[2]] for v in allv], m)
        x, y = f(lon, lat)
        ax, ay = f(seg[0], seg[1])
        bx, by = f(seg[2], seg[3])
        dx, dy = bx - ax, by - ay
        n = math.hypot(dx, dy) or 1.0
        nx_, ny_ = -dy / n, dx / n
        major = abs(m / 1000.0 - round(m / 1000.0)) < 1e-6
        ln = 5.0 if major else 3.4
        d.add(Line(x - nx_ * ln, y - ny_ * ln, x + nx_ * ln, y + ny_ * ln, strokeColor=INK,
                   strokeWidth=1.0 if major else 0.6))
        if m > 0 and abs(m / step_lab - round(m / step_lab)) < 1e-6:
            anchor = "start" if nx_ > 0.35 else ("end" if nx_ < -0.35 else "middle")
            ly = y + ny_ * 8 + (0 if ny_ > 0.35 else (-6 if ny_ < -0.35 else -2))
            txt = _ch(m)
            if not labels.place(d, x + nx_ * 8, ly, txt, size=6.4, anchor=anchor):
                anchor = "end" if anchor == "start" else ("start" if anchor == "end" else "middle")
                labels.place(d, x - nx_ * 8, y - ny_ * 8 + (0 if ny_ < -0.35 else (-6 if ny_ > 0.35 else -2)),
                             txt, size=6.4, anchor=anchor)
        m += tick
    for name, p, mp in (("A", rep["header"]["a"], 0.0), ("B", rep["header"]["b"], L)):
        if not lo - 1e-6 <= mp <= hi + 1e-6:
            continue
        x, y = f(p["snapped_lon"], p["snapped_lat"])
        d.add(Rect(x - 4, y - 4, 8, 8, fillColor=INK, strokeColor=colors.white, strokeWidth=1.0))
        for dx, dy in ((6, 4), (-13, 4), (6, -11), (-13, -11)):
            if labels.place(d, x + dx, y + dy, name, size=10, bg=True):
                break
    _north_and_scale(d, labels, w, h, s_, (10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000))
    if base is not None:
        labels.place(d, w - 3, 3.5, base.attribution, size=5.6, color=MUTED, anchor="end", force=True, bg=True)
    else:
        labels.place(d, w - 3, 3.5, "street map unavailable (offline): route drawn without a background",
                     size=5.6, color=MUTED, anchor="end", force=True)
    return d


def locator_figure(rep: dict, w: float, h: float, half_km: float = 2.0) -> Optional[Drawing]:
    """Where the stretch is: a street map about 2 x half_km wide around it; the route in blue with A and B.
    None when the street map is unavailable (the route figure already says so)."""
    from app import pci_basemap
    from reportlab.graphics.shapes import Image as GImage

    allv = rep["path"]["vertices"]
    lons = [v[0] for v in allv]
    lats = [v[1] for v in allv]
    cy = (min(lats) + max(lats)) / 2
    dlat = half_km / 110.54
    dlon = half_km / (111.32 * math.cos(math.radians(cy)))
    box = [(min(lons) - dlon, min(lats) - dlat), (max(lons) + dlon, max(lats) + dlat)]
    try:
        base = pci_basemap.view(box, w, h, 0.0)
    except Exception:
        base = None
    if base is None:
        return None
    d = Drawing(w, h)
    d.add(GImage(0, 0, w, h, base.image_path))
    d.add(Rect(0, 0, w, h, strokeColor=RULE, fillColor=None, strokeWidth=0.6))
    labels = _Labels(w, h, _fonts["regular"], halo=True)
    pts = []
    for lon, lat, _m in allv:
        pts.extend(base.xy(lon, lat))
    if len(pts) >= 4:
        d.add(PolyLine(pts, strokeColor=colors.white, strokeWidth=6.0, strokeLineCap=1, strokeLineJoin=1))
        d.add(PolyLine(pts, strokeColor=colors.HexColor("#1d4ed8"), strokeWidth=3.4, strokeLineCap=1,
                       strokeLineJoin=1))
        labels.obstacle_line(pts, r=3.0)
    L = rep["summary"]["length_m"]
    for name, p, _mp in (("A", rep["header"]["a"], 0.0), ("B", rep["header"]["b"], L)):
        x, y = base.xy(p["snapped_lon"], p["snapped_lat"])
        d.add(Rect(x - 4, y - 4, 8, 8, fillColor=INK, strokeColor=colors.white, strokeWidth=1.0))
        for dx, dy in ((6, 4), (-13, 4), (6, -11), (-13, -11)):
            if labels.place(d, x + dx, y + dy, name, size=10, bg=True):
                break
    labels.place(d, 4, h - 10, "Location of the stretch (blue)", size=7, force=True, bg=True)
    _north_and_scale(d, labels, w, h, base.pt_per_m, (100, 200, 500, 1000, 2000, 5000))
    labels.place(d, w - 3, 3.5, base.attribution, size=5.6, color=MUTED, anchor="end", force=True, bg=True)
    return d


def legend_table(rep, st, width) -> Table:
    if single_mode(rep):
        items = [(CLASS_COLORS[c], f"{c} ({r})") for c, r, _ in T.TABLE_5_5]
        items.append((PART_DATA, "PCI not available here"))
    else:
        items = [(RANGE_COLORS[k], k) for k in RANGE_COLORS]
    items.append((NO_DATA, "Insufficient data (< 50 % imaged)"))
    cells = [P(f"{_swatch(c)} {escape(t)}", st["cell"]) for c, t in items]
    cells.append(P("<font color='#1c1917'>○</font> pothole (one run of flagged frames)", st["cell"]))
    cells.append(P("<font color='#1c1917'>- - -</font> gap, not surveyed (straight line)", st["cell"]))
    ncol = 4
    rows = [cells[i:i + ncol] for i in range(0, len(cells), ncol)]
    rows[-1] += [""] * (ncol - len(rows[-1]))
    t = Table(rows, colWidths=[width / ncol] * ncol, hAlign="LEFT")
    t.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 0),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
    return t


# ============================================================ 1. summary
def _entered(v) -> str:
    return (user_text(v) + " <font color='#57534e'>(as entered, unverified)</font>") if v else user_text(None)


def header_block(rep, st, width):
    h, s = rep["header"], rep["summary"]
    a, b = h["a"], h["b"]
    rc_a, rc_b = h.get("road_chainage_from"), h.get("road_chainage_to")
    road_ch = (f"{escape(rc_a['text'])} at A to {escape(rc_b['text'])} at B (as entered, unverified)" if rc_a
               else "not given (no km-stone reference); chainage below is measured from A")
    length = (f"{'≥ ' if s['gaps'] else ''}{s['length_m']:.0f} m"
              + (f" (incl. {s['gaps']} unsurveyed gap{'s' if s['gaps'] != 1 else ''}, ≥ {s['gap_m']:.0f} m)"
                 if s["gaps"] else ""))
    dates = ", ".join(h["survey_dates_ist"]) or "–"
    season = "; ".join(x.split(" (")[0] for x in h["season"])
    surf = f"{escape(h.get('surface_name') or h.get('surface') or '')} ({escape(h.get('surface') or '')})"
    surf += (" — default for the road class" if h.get("surface_note", "").startswith("default")
             else " — as entered, unverified")
    rows = [
        ("Road", _entered(h.get("road_name")), "Road number", _entered(h.get("road_number"))),
        ("Road class", escape(ROAD_CLASS_NAMES.get(h["category"], h["category"])), "Surface", surf),
        ("Start A", f"{a['snapped_lat']:.6f}, {a['snapped_lon']:.6f}<br/>chainage {_ch(0)}",
         "End B", f"{b['snapped_lat']:.6f}, {b['snapped_lon']:.6f}<br/>chainage {_ch(s['length_m'])}"),
        ("Length", length, "Road chainage", road_ch),
        ("Survey date", escape(dates + (f" ({season})" if season else "")), "Imaged",
         f"{100 * s['coverage']:.0f} % of the length, one lane"),
        ("Method", "Phone video, forward-looking camera; one frame per 5 m; automatic image classification",
         "Device", escape(", ".join(h["devices"]) or "–")),
        ("Code", "IRC:82-2023, PCI 0–100 (§5.2, Tables 5.4–5.5, Appendix 2 Table "
                 f"{APPENDIX_TABLE.get(h['category'], '')})", "Prepared by", _entered(h.get("prepared_by"))),
    ]
    data = [[P(k1, st["cellb"]), P(v1, st["cell"]), P(k2, st["cellb"]), P(v2, st["cell"])] for k1, v1, k2, v2 in rows]
    kw = 21 * mm
    vw = (width - 2 * kw) / 2
    return _table(data, [kw, vw, kw, vw], header_rows=0,
                  extra=[("BACKGROUND", (0, 0), (0, -1), HEAD_BG), ("BACKGROUND", (2, 0), (2, -1), HEAD_BG)])


def _missing_line(rep) -> str:
    """One line: which parameters are missing from the headline PCI and why."""
    rows = stretch_param_rows(rep)
    miss = [r for r in rows if not r["mean_known"]]
    if not miss:
        lab = [LOWER_NAMES[r["param"]] for r in rows if r.get("source") == "labels"]
        if lab:
            return (f"Every parameter has a value, but {' and '.join(lab)} come from uncalibrated image labels "
                    "(share of length, runs of frames), not yet scaled to IRC % area / pothole units on a "
                    "reference survey; the range shows what they allow.")
        return ""
    w = sum(r["weight"] for r in miss)
    names = ", ".join(f"{LOWER_NAMES[r['param']]} ({r['weight']:.2f})" for r in miss)
    why = []
    srcs = {r["param"] for r in miss}
    if srcs & {"iri", "ravelling", "patching"}:
        why.append("not yet calibrated against a reference survey")
    if "rut_depth" in srcs:
        why.append("rut depth needs a straight edge or profiler")
    if "potholes" in srcs and _road_class(rep) in ("mdr", "rural"):
        why.append("the printed MDR / rural pothole equation is incomplete")
    return (f"Not assessed: {names} = {100 * w:.0f} % of the PCI weight ({'; '.join(why)}). Each is counted "
            f"from 0 to 100, which gives the range.")


def headline_block(rep, st, width):
    s = rep["summary"]
    pc = s.get("pci")
    out = []
    if not pc:
        box = [[P("PCI: insufficient data", st["bigsub"])],
               [P(f"Only {100 * s['coverage']:.0f} % of the stretch was imaged; a unit needs at least 50 % imaged "
                  "and some classified frames.", st["body"])]]
        t = Table(box, colWidths=[width], hAlign="LEFT")
        t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 1.0, INK), ("LEFTPADDING", (0, 0), (-1, -1), 6)]))
        return [t]
    if pc.get("single") is not None:
        cls = pc["single_class"]
        col = CLASS_COLORS[cls]
        left = [P(f'PCI <font color="{col}">{fnum(pc["single"], 1)}</font>', st["big"]),
                P(f'<font color="{col}">{escape(cls)}</font> (Table 5.5)', st["bigsub"])]
        right = [P(f"<b>Recommended (Table 5.5):</b> {escape(T.class_recommendation(cls) or '')}.", st["body"])]
        if pc["single"] < 60:
            right.append(P("PCI below 60: carry out a detailed investigation of the strengthening requirement "
                           "first (Table 10.1 note).", st["body"]))
        right.append(P("Every parameter has a value; check the source of each in the table below.", st["note"]))
    else:
        left = [P(f'PCI {fnum(pc["lo"], 0)}–{fnum(pc["hi"], 0)}', st["big"]),
                P(f'{escape(pc["class_lo"])} to {escape(pc["class_hi"])} (Table 5.5)', st["bigsub"])]
        right = [P("<b>No single PCI yet.</b> " + escape(_missing_line(rep)), st["body"]),
                 P("A maintenance class (Table 5.5) cannot be chosen from this range. What is measured — "
                   "potholes and cracking — is below and is enough to plan pothole repair and crack sealing.",
                   st["body"])]
    t = Table([[left, right]], colWidths=[64 * mm, width - 64 * mm], hAlign="LEFT")
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 1.0, INK), ("LINEAFTER", (0, 0), (0, 0), 0.4, RULE),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 7),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 7), ("TOPPADDING", (0, 0), (-1, -1), 6),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    out.append(t)
    if pc.get("units_rated", 0) < pc.get("units", 0):
        out.append(P(f"{pc['units'] - pc['units_rated']} of {pc['units']} km rows had too little imaged length "
                     "and are left out of the PCI.", st["note"]))
    return out


def _source_text(r: dict, rep) -> str:
    src = r.get("source")
    if src == "not_assessed":
        why = {"ravelling": "not assessed — needs calibration", "patching": "not assessed — needs calibration",
               "rut_depth": "not assessed — needs straight edge / profiler",
               "iri": "not assessed — accelerometer not calibrated"}
        return why.get(r["param"], "not assessed")
    if src == "labels":
        return "image labels (frames classified)"
    if src == "labels_calibrated":
        return "image labels, scale calibrated on survey"
    if src == "probe":
        return "image probe, calibrated on survey"
    if src == "roughness":
        return "phone accelerometer, calibrated"
    if src == "entered":
        return "entered by requester (unverified)"
    return escape(r.get("source_name") or str(src))


def _value_text(r: dict, rep, stretch: bool) -> str:
    p = r["param"]
    if r.get("value_point") is None and r.get("value_lo") is None:
        return "–"
    v, lo, hi = _value_point(r), r.get("value_lo"), r.get("value_hi")
    if p == "potholes" and r.get("source") in ("labels", "labels_calibrated"):
        n = int(round(lo)) if r.get("source") == "labels" else lo
        txt = f"{fnum(n, 0) if r.get('source') == 'labels' else fnum(n, 1)}"
        if stretch:
            rpk = rep["stretch"]["quantities"]["runs_per_km"].get("potholes")
            txt += f" ({fnum(rpk, 1)} per km)"
        return txt
    unit = {"%": " %", "mm": " mm", "m/km": " m/km", "number": ""}[r["unit"]]
    nd = 2 if p == "iri" else 1
    if p == "cracking" and r.get("source") == "labels":
        # uncalibrated: the share of LENGTH bounds the cracked AREA from above (best case 0 % area)
        return (f"{fnum(v, 1)} % of length <font color='#57534e'>(area ≤ {fnum(hi, 1)} %)</font>" if hi
                else f"{fnum(v, 1)} % of length")
    txt = fnum(v, nd) + unit
    if lo is not None and hi is not None and fnum(lo, nd) != fnum(hi, nd):
        txt += f" <font color='#57534e'>({fnum(lo, nd)}–{fnum(hi, nd)})</font>"
    return txt


def param_table(rep, st, width):
    rc = _road_class(rep)
    surface = rep["header"].get("surface")
    rows = stretch_param_rows(rep)
    head = [P(h, st["head"]) for h in ("Parameter", "Weight", "Value (whole stretch)", "PCIᵢ",
                                       f"Table {DISTRESS_TABLE[rc]} range", "Source")]
    data = [head]
    tot_lo = tot_hi = 0.0
    for r in rows:
        name = escape(r["name"]) + (f" ({escape(r['unit'])})" if r["unit"] != "number" else " (number)")
        if r["mean_known"]:
            pci_i = fnum(r["mean_point"], 0)
            if fnum(r["mean_worst"], 0) != fnum(r["mean_best"], 0):
                pci_i += f" <font color='#57534e'>({fnum(r['mean_worst'], 0)}–{fnum(r['mean_best'], 0)})</font>"
            tot_lo += r["weight"] * r["mean_worst"]
            tot_hi += r["weight"] * r["mean_best"]
        elif r.get("value_lo") is not None and r.get("source") != "not_assessed":
            pci_i = "not computable"
            tot_hi += 100 * r["weight"]
        else:
            pci_i = "<font color='#57534e'>0–100</font>"
            tot_hi += 100 * r["weight"]
        rng = "–"
        v = _value_point(r)
        if v is not None and r.get("source") != "not_assessed":
            vv = v
            if r["param"] == "potholes":
                rpk = rep["stretch"]["quantities"]["runs_per_km"].get("potholes")
                vv = rpk if (rpk is not None and r.get("source") in ("labels", "labels_calibrated")) else v
            b = C.band23(rc, r["param"], vv, surface)
            if b:
                rng = f'{_swatch(RANGE_COLORS[b])} {escape(b)}'
        known = r["mean_known"]
        data.append([P(name, st["cellb"] if known else st["cell"]), P(f"{r['weight']:.2f}", st["cellc"]),
                     P(_value_text(r, rep, True) if r.get("source") != "not_assessed" else "–", st["cell"]),
                     P(pci_i, st["cellc"]), P(rng, st["cell"]), P(_source_text(r, rep), st["cell"])])
    pc = rep["summary"].get("pci") or {}
    tot = pci_text(pc) if pc else "–"
    data.append([P("<b>PCI = Σ weight × PCIᵢ</b>", st["cell"]), P("1.00", st["cellc"]), "",
                 P(f"<b>{tot}</b>", st["cellc"]), P(f"<b>{escape(class_text(pc) if pc else '')}</b>", st["cell"]), ""])
    cw = [31, 13, 36, 22, 37]
    cw.append(width / mm - sum(cw))
    t = _table(data, [c * mm for c in cw],
               extra=[("BACKGROUND", (0, len(data) - 1), (-1, len(data) - 1), TOTAL_BG),
                      ("SPAN", (4, len(data) - 1), (5, len(data) - 1))])
    return t, (tot_lo, tot_hi)


def key_findings(rep, st) -> list:
    s = rep["summary"]
    q = rep["stretch"]["quantities"]
    runs = s.get("pothole_runs_list") or []
    out = []
    if runs:
        worst = max(rep["km"], key=lambda k: k["quantities"]["runs"]["potholes"])
        out.append(f"<b>Potholes:</b> {_plural(len(runs), 'pothole location')} flagged "
                   f"({fnum(q['runs_per_km'].get('potholes'), 1)} per km); most in "
                   f"{_ch(worst['from_m'])}–{_ch(worst['to_m'])} ({worst['quantities']['runs']['potholes']}). "
                   "Locations with coordinates in §4; check the photos in the annex before sending a crew.")
    else:
        out.append("<b>Potholes:</b> none flagged on the imaged frames.")
    cr = (_params(rep["stretch"]).get("cracking") or {})
    if cr.get("value_point") is not None:
        types = [t for t in ("Alligator Cracking", "Longitudinal Cracking", "Transverse Cracking", "Edge Cracking",
                             "Hairline Cracks") if t in (q.get("labels_present") or [])]
        out.append(f"<b>Cracking:</b> seen on {fnum(cr['value_point'], 1)} % of the surveyed length"
                   + (f" ({escape(', '.join(t.split(' ')[0].lower() for t in types))})" if types else "")
                   + ". This is a length share; the cracked AREA is smaller.")
    low = [x for x in rep["sections"] if not _upci(x)]
    cov = f"<b>Coverage:</b> {100 * s['coverage']:.0f} % of the length imaged"
    if s["gaps"]:
        cov += f"; {_plural(s['gaps'], 'gap')} not surveyed (≥ {s['gap_m']:.0f} m in all)"
    if low:
        cov += f"; {_plural(len(low), '100 m section')} with too little imagery"
    out.append(cov + ".")
    return [P(x, st["bullet"], bulletText="•") for x in out]


# ============================================================ 3. tables
def _shown_params(rep) -> list[str]:
    """Parameters that get a column: a value from the survey (not entered) in at least one unit."""
    out = []
    for p in P23.PARAMS:
        for u in rep["sections"] + rep["km"]:
            r = _params(u).get(p) or {}
            if r.get("source") not in (None, "not_assessed", "entered") and _value_point(r) is not None:
                out.append(p)
                break
    return out


COL_HEAD = {"cracking": "Cracking (%)", "ravelling": "Ravelling (%)", "potholes": "Potholes (no.)",
            "patching": "Patching (%)", "rut_depth": "Rut depth (mm)", "iri": "IRI (m/km)"}


def _cell_value(u, p) -> str:
    r = _params(u).get(p) or {}
    if r.get("source") in (None, "not_assessed") or _value_point(r) is None:
        return "–"
    if p == "potholes" and r.get("source") == "labels":
        return str(int(round(r["value_lo"])))
    return fnum(_value_point(r), 2 if p == "iri" else 1)


def _not_shown_note(rep, shown) -> str:
    ent = rep["header"].get("entered") or {}
    miss = [LOWER_NAMES[p] for p in P23.PARAMS if p not in shown and p not in ent]
    parts = []
    if ent:
        parts.append("Entered for the whole stretch (unverified): " + ", ".join(
            f"{LOWER_NAMES[p]} {fnum(v, 2 if p == 'iri' else 1)} {P23.PARAM_UNITS[p]}".replace(" number", "")
            for p, v in sorted(ent.items())) + ".")
    if miss:
        parts.append("Not assessed: " + ", ".join(miss) + " (counted 0–100 in the PCI).")
    return " ".join(parts)


def km_table(rep, st, width):
    shown = _shown_params(rep)
    single = single_mode(rep)
    head = ["Km (from–to)", "Imaged (%)"] + [COL_HEAD[p] for p in shown] + ["PCI", "Class"]
    if single:
        head.append("Recommendation (Table 5.5)")
    data = [[P(h, st["headc"] if i else st["head"]) for i, h in enumerate(head)]]
    one = len(rep["km"]) == 1
    units = [(k, False) for k in rep["km"]] + ([] if one else [(rep["stretch"], True)])
    for u, whole in units:
        q = u["quantities"]
        pc = rep["summary"].get("pci") if whole else _upci(u)
        lab = f"{_ch(u['from_m'])}–{_ch(u['to_m'])}"
        if whole:
            lab = f"<b>Whole stretch</b><br/>{lab}"
        elif u.get("part_km"):
            lab += " <font color='#57534e'>(part)</font>"
        row = [P(lab, st["cell"]), P(fnum(100 * q["coverage"], 0), st["cellc"])]
        if _upci(u) or (whole and pc):
            row += [P(_cell_value(u, p), st["cellc"]) for p in shown]
            col = CLASS_COLORS.get(pc.get("single_class")) if pc and pc.get("single") is not None else None
            row += [P(("<b>" if whole else "") + pci_text(pc) + ("</b>" if whole else ""), st["cellc"]),
                    P((f'<font color="{col}">' if col else "") + escape(class_text(pc)) + ("</font>" if col else ""),
                      st["cellc"])]
            if single:
                row.append(P(escape(T.class_recommendation(pc.get("single_class")) or "–"), st["cell"]))
        else:
            row += [P("<font color='#57534e'>insufficient data</font>", st["cellc"])] + [""] * (len(shown) + 1 +
                                                                                              (1 if single else 0))
        data.append(row)
    fixed = [29, 14]
    if single:              # recommendation column gets what the parameter columns leave (>= 34 mm)
        tail = [14, 21]
        pw = min(19.0, (width / mm - sum(fixed) - sum(tail) - 34) / max(1, len(shown)))
        cw = fixed + [pw] * len(shown) + tail
        cw.append(width / mm - sum(cw))
    else:
        tail = [18, 26]
        pw = 19.0
        extra = (width / mm - sum(fixed) - sum(tail) - pw * len(shown)) / (len(shown) + 2)
        cw = fixed + [pw + extra] * len(shown) + [tail[0] + extra, tail[1] + extra]
    ext = []
    for i, (u, _w) in enumerate(units, start=1):
        if not (_upci(u) or _w):
            ext.append(("SPAN", (2, i), (-1, i)))
    if not one:
        ext.append(("BACKGROUND", (0, len(data) - 1), (-1, len(data) - 1), TOTAL_BG))
    return _table(data, [c * mm for c in cw], extra=ext)


def section_table(rep, st, width):
    shown = _shown_params(rep)
    single = single_mode(rep)
    rc = _road_class(rep)
    head = ["", "Chainage (from–to)", "Imaged (%)"] + [COL_HEAD[p] for p in shown]
    head += (["PCI", "Class"] if single else [f"Distress range (Table {DISTRESS_TABLE[rc]})", "PCI range"])
    data = [[P(h, st["headc"] if i > 1 else st["head"]) for i, h in enumerate(head)]]
    ext = []
    for i, s in enumerate(rep["sections"], start=1):
        q = s["quantities"]
        col, lab = colour_key(rep, s)
        low = " <font color='#57534e'>low</font>" if q["coverage_status"] == "low_coverage" else ""
        row = [P(_swatch(col), st["cellc"]), P(f"{_ch(s['from_m'])}–{_ch(s['to_m'])}", st["cell"]),
               P(fnum(100 * q["coverage"], 0) + low, st["cellc"])]
        pc = _upci(s)
        if pc:
            row += [P(_cell_value(s, p), st["cellc"]) for p in shown]
            if single:
                row += [P(pci_text(pc), st["cellc"]), P(escape(class_text(pc)), st["cellc"])]
            else:
                row += [P(escape(lab), st["cellc"]), P(pci_text(pc), st["cellc"])]
        else:
            row += [P("<font color='#57534e'>insufficient data</font>", st["cellc"])] + [""] * (len(shown) + 1)
            ext.append(("SPAN", (3, i), (-1, i)))
        data.append(row)
    fixed = [6, 27, 15]
    tail = [24, 24] if single else [34, 20]
    rest = width / mm - sum(fixed) - sum(tail)
    cw = fixed + [rest / max(1, len(shown))] * len(shown) + tail
    if not shown:
        cw[-1] += rest
    return _table(data, [c * mm for c in cw], zebra=True, extra=ext)


# ============================================================ 4. potholes
def pothole_table(rep, st, width):
    runs = rep["summary"].get("pothole_runs_list") or []
    annex = {}
    for i, a in enumerate(rep.get("annex") or [], start=1):
        annex.setdefault(a["segment_id"], i)
    inp = rep.get("input") or {}
    at_a, inc = inp.get("chainage_at_a_km"), inp.get("chainage_increases_towards_b")
    calibrated = at_a is not None and inc is not None
    head = ["#", "Chainage from A", "Flagged length"] + (["Road chainage"] if calibrated else []) + \
           ["Latitude, longitude", "Photo"]
    data = [[P(h, st["head"]) for h in head]]
    for i, r in enumerate(runs, start=1):
        lon, lat = lonlat_at(rep, (r["from_m"] + r["to_m"]) / 2)
        ch = _ch(r["from_m"]) + ("" if r["segments"] == 1 else f" to {_ch(r['to_m'])}")
        photo = next((annex[s] for s in r["segment_ids"] if s in annex), None)
        row = [P(str(i), st["cellc"]), P(ch, st["cell"]),
               P(f"{r['segments'] * 5} m ({_plural(r['segments'], 'frame')})", st["cell"])]
        if calibrated:
            km = float(at_a) + ((r["from_m"] if inc else -r["from_m"]) / 1000.0)
            row.append(P(f"Km {int(km)}+{int(round((km - int(km)) * 1000)) % 1000:03d}" if km >= 0 else "–",
                         st["cell"]))
        row += [P(f"{lat:.6f}, {lon:.6f}", st["cell"]), P(f"Photo {photo}" if photo else "–", st["cellc"])]
        data.append(row)
    cw = [9, 40, 34] + ([30] if calibrated else []) + [46, 20]
    return _table(data, [c * mm for c in cw], zebra=True)


# ============================================================ 5. maintenance
def maintenance_block(rep, st, width):
    out = []
    pc = rep["summary"].get("pci") or {}
    rc = _road_class(rep)
    if pc.get("single") is not None:
        cls = pc["single_class"]
        out.append(P(f"<b>Whole stretch: PCI {fnum(pc['single'], 1)}, {escape(cls)} — "
                     f"{escape(T.class_recommendation(cls))}.</b> Km rows and 100 m sections with a different class "
                     "are in §3.", st["body"]))
        hi_rows = {cls}
    elif pc:
        a, b = T.CLASS_ORDER.index(pc["class_hi"]), T.CLASS_ORDER.index(pc["class_lo"])
        hi_rows = set(T.CLASS_ORDER[a:b + 1])
        out.append(P(f"The PCI range {fnum(pc['lo'], 0)}–{fnum(pc['hi'], 0)} spans {escape(pc['class_hi'])} to "
                     f"{escape(pc['class_lo'])} (shaded), so the overall treatment cannot be chosen yet. "
                     "The distress-specific repairs below apply now.", st["body"]))
    else:
        hi_rows = set()
    data = [[P(h, st["head"]) for h in ("Class", "PCI", "Recommendation (Table 5.5)")]]
    ext = []
    for i, (c, r, rec) in enumerate(T.TABLE_5_5, start=1):
        data.append([P(f"{_swatch(CLASS_COLORS[c])} {escape(c)}", st["cell"]), P(escape(r), st["cell"]),
                     P(escape(rec), st["cell"])])
        if c in hi_rows:
            ext.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#fef3c7")))
    out.append(_table(data, [32 * mm, 26 * mm, 84 * mm], extra=ext))
    out.append(Spacer(1, 2))
    out.append(P(escape(T.renewal_note(rc)), st["note"]))
    # distress-specific
    q = rep["stretch"]["quantities"]
    sp = _params(rep["stretch"])
    labels = set(q.get("labels_present") or [])
    have_param = {p for p, r in sp.items() if r.get("source") not in (None, "not_assessed")}
    labs = [lab for lab in labels if T.PARAM_OF.get(lab) in have_param]
    for p, lab in (("ravelling", "Ravelling"), ("patching", "Patching"), ("rut_depth", "Rutting")):
        r = sp.get(p) or {}
        if r.get("source") not in (None, "not_assessed", "entered") and (_value_point(r) or 0) > 0:
            labs.append(lab)
    rows = T.distress_rows(labs, q.get("worst_severity"))
    out.append(P("Repairs for the distress found (IRC:82-2023 Section 7)", st["h3"]))
    if rows:
        data = [[P(h, st["head"]) for h in ("Distress", "Worst severity (visual)", "Treatment", "Clause")]]
        for r in rows:
            data.append([P(escape(r["label"]), st["cellb"]), P(escape(r["severity"]), st["cellc"]),
                         P(escape(r["treatment"]), st["cell"]), P(escape(r["clause"]), st["cellc"])])
        cw = [32, 22, 0, 22]
        cw[2] = width / mm - sum(cw)
        out.append(_table(data, [c * mm for c in cw]))
        out.append(P("Severity is judged from the image (crack width and pothole depth are not measured). "
                     "§5.3: distress that seriously affects riding quality, such as potholes, is to be rectified "
                     "by timely maintenance.", st["note"]))
    else:
        out.append(P("No pothole or cracking found on the imaged frames.", st["body"]))
    other = sorted(lab for lab in labels if lab not in T.PARAM_OF and lab != "Normal")
    if other:
        out.append(P("Also labelled on some frames, not part of the PCI and not validated: "
                     + escape(", ".join(other)) + ". Confirm on site before acting.", st["note"]))
    return out


# ============================================================ 6. method and limitations
def method_lines(rep) -> list[str]:
    rc = _road_class(rep)
    h = rep["header"]
    lines = [
        "<b>Survey.</b> A phone on the windshield records forward-looking video; one frame per 5 m of travel is "
        "classified by a vision-language model (Qwen2.5-VL-7B, pipeline P4). A frame gate skips frames it judges "
        "uninformative; the PCI range allows for distress in skipped frames at the audited rate (upper 95 % limit).",
        "<b>Chainage</b> is the distance along the surveyed track from A (frame centres 5 m apart). Gaps between "
        "drives are counted as straight lines, so lengths after a gap are lower bounds.",
        f"<b>PCI</b> = Σ weight × PCIᵢ (§5.2, Table 5.4 weights); PCIᵢ from the Appendix-2 "
        f"equations for {escape(ROAD_CLASS_NAMES.get(rc, rc).lower())}s (Table {APPENDIX_TABLE.get(rc, '')}), "
        "clamped to 0–100. Computed per km row (and per 100 m section); the whole-stretch PCI is the "
        "length-weighted mean of the km rows (§5.1 note: analyse 100–1000 m). A parameter without a value is "
        "counted as 0 and as 100, giving a range — never a guessed value.",
        "<b>Cracking (%)</b> = share of the surveyed length whose frame shows cracking. IRC uses % of area, so this "
        "is an upper bound (one transverse crack marks a whole 5 m frame).",
        "<b>Potholes (number)</b> = runs of consecutive frames labelled pothole; one run counts as one pothole. "
        "IRC counts 0.1 m² pothole units (eq. 2); pothole area is not measured, so a large pothole counts once.",
    ]
    srcs = {p["param"]: p.get("source") for p in ((rep["stretch"].get("pci") or {}).get("params") or [])}
    if "labels_calibrated" in (srcs.get("cracking"), srcs.get("potholes")):
        lines.append("<b>Label scale.</b> Where marked 'scale calibrated', the cracking share of length and the "
                     "pothole run count are multiplied by a factor fitted on a reference survey, to put them on "
                     "the IRC % area / pothole-unit scale.")
    cal = h.get("calibration")
    if cal:
        fo = cal.get("fitted_on") or {}
        lines.append("<b>Calibration.</b> Parameters marked 'calibrated' use thresholds fitted on the calibration "
                     "half of a reference survey" + (f" ({escape(str(fo.get('survey') or fo.get('name') or ''))})"
                                                     if (fo.get("survey") or fo.get("name")) else "") + ".")
    if rc == "urban" and any((_params(k).get("iri") or {}).get("known") for k in rep["km"]):
        lines.append("The printed urban worked examples (Appendix 2, C.2 and C.3) do not reproduce from the printed "
                     "equations (their roughness term differs); this report uses the printed equations.")
    if rc in ("mdr", "rural"):
        lines.append("The printed MDR / rural pothole equation (Table A2.2) rises with the pothole number while its "
                     "curve falls; a term is missing, so pothole PCIᵢ is not computed (counted 0–100).")
    return lines


def limitation_lines(rep) -> list[str]:
    ev_c = C.ASSESSMENT["cracking"]["evidence"]
    ev_p = C.ASSESSMENT["potholes"]["evidence"]
    return [
        "<b>One lane.</b> Only the lane driven in is imaged. IRC:82-2023 measures extents over all lanes of the "
        "carriageway and IRI / rut depth as the mean of all lanes in one direction.",
        f"<b>Label accuracy</b> (tested on another dataset, not yet on Bengaluru roads): potholes {escape(ev_p)}; "
        f"cracking {escape(ev_c)}. Tree shade and vehicle shadows are sometimes labelled as potholes — check the "
        "photos.",
        "<b>Position.</b> Phone GPS: about ±10 m along the road, more after a gap. The camera looks ahead, so a "
        "distress may lie up to ~30 m beyond the marked chainage in the direction of travel.",
        "<b>Not an IRC condition survey.</b> §5.3 asks for instrumented surveys (e.g. a Network Survey Vehicle) on "
        "two-lane and wider roads, or visual surveys by experienced engineers, twice a year on urban roads. Use "
        "this report to screen and prioritise; confirm on site before design or payment.",
    ]


def signoff_block(st) -> list:
    sign = [[P("Field-checked by", st["cellb"]), P("", st["cell"]), P("Date", st["cellb"]), P("", st["cell"])],
            [P("Approved by", st["cellb"]), P("", st["cell"]), P("Date", st["cellb"]), P("", st["cell"])]]
    t = Table(sign, colWidths=[30 * mm, 86 * mm, 16 * mm, 50 * mm], rowHeights=[11 * mm, 11 * mm], hAlign="LEFT")
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, RULE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("BACKGROUND", (0, 0), (0, -1), HEAD_BG), ("BACKGROUND", (2, 0), (2, -1), HEAD_BG)]))
    return [P("Sign-off", st["h3"]), t,
            P("Generated automatically from phone video; to be signed after a field check.", st["note"])]


# ============================================================ annex
def annex_block(rep, st, image_fetcher):
    items = rep.get("annex") or []
    if not items:
        return [P("No photographs selected (no pothole and no section in the worst distress range).", st["body"])]
    cells = []
    col_w = 88 * mm
    for i, a in enumerate(items, start=1):
        img = None
        data = image_fetcher(a["segment_id"]) if image_fetcher else None
        if data:
            try:
                from PIL import Image as PILImage
                with PILImage.open(io.BytesIO(data)) as im:
                    iw, ih = im.size
                hgt = min(col_w * ih / iw, 52 * mm)
                img = Image(io.BytesIO(data), width=hgt * iw / ih, height=hgt)
            except Exception:
                img = None
        reason = (a["reason"].replace("segments)", "frames)").replace("segment)", "frame)")
                  if a["reason"].startswith("pothole run") else "most distressed frame of its 100 m section")
        note = a.get("note") or ""
        look = ("camera looking towards A" if "towards A" in note else
                "camera looking towards B" if "towards B" in note else "")
        cap = (f"<b>Photo {i} — {escape(_ch(a['measure_m']))}</b> · {escape(reason)}<br/>"
               f"{a['lat']:.6f}, {a['lon']:.6f} · {escape(a.get('time_ist') or '?')}"
               + (f" · {look}" if look else "") + "<br/>"
               f"Labels: {escape(', '.join(a['labels']) or '–')} · {escape(a['review_state'])}<br/>"
               f"<font color='#57534e'>frame {escape(a['segment_id'])}</font>")
        cells.append([img if img is not None else P("<i>image not available</i>", st["caption"]), Spacer(1, 2),
                      P(cap, st["caption"])])
    rows = [cells[i:i + 2] for i in range(0, len(cells), 2)]
    if len(rows[-1]) == 1:
        rows[-1].append("")
    out = []
    for r in rows:
        t = Table([r], colWidths=[col_w + 3 * mm] * 2, hAlign="LEFT")
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                               ("RIGHTPADDING", (0, 0), (-1, -1), 3 * mm), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
        out.append(t)
    return out


# ============================================================ render
def render(rep: dict, image_fetcher: Optional[Callable[[str], Optional[bytes]]] = None) -> bytes:
    st = _styles()
    buf = io.BytesIO()
    doc = _Doc(buf, rep)
    pw = A4[0] - 2 * MARGIN
    s = rep["summary"]
    L = s["length_m"]
    name = rep["header"].get("road_name")
    story = [P("Road condition report", st["title"]),
             P(("" if not name else user_text(name, "") + " · ") + f"A to B, {L / 1000:.2f} km · IRC:82-2023 "
               "Pavement Condition Index", st["subtitle"])]

    # 1. summary
    story.append(P("1. Summary", st["h2"]))
    story.append(header_block(rep, st, pw))
    story.append(Spacer(1, 6))
    story.extend(headline_block(rep, st, pw))
    story.append(Spacer(1, 6))
    ptab, _tot = param_table(rep, st, pw)
    story.append(KeepTogether([P("PCI by parameter", st["h3"]), ptab,
                               P("PCIᵢ = the parameter's index (0–100) at the best estimate, length-weighted over "
                                 "the km rows. Grey brackets = what the PCI range allows: frames the gate skipped "
                                 "and, for cracking, a cracked area anywhere from 0 up to the cracked length. "
                                 f"Range column: the value against Table {DISTRESS_TABLE[_road_class(rep)]} "
                                 "(potholes per km).",
                                 st["note"])]))
    story.append(P("Key findings", st["h3"]))
    story.extend(key_findings(rep, st))
    story.append(KeepTogether(signoff_block(st)))

    # 2. route
    story.append(PageBreak())
    story.append(P("2. Route", st["h2"]))
    story.append(route_figure(rep, pw, 122 * mm))
    story.append(Spacer(1, 3))
    story.append(legend_table(rep, st, pw))
    colour_note = ("Each 100 m section is coloured by its PCI class (Table 5.5)." if single_mode(rep) else
                   f"Each 100 m section is coloured by the worst Table {DISTRESS_TABLE[_road_class(rep)]} range of "
                   "its measured parameters (cracking %, potholes) — not by PCI, which is still a range.")
    story.append(P(colour_note + " Ticks every " + ("100" if L <= 3000 else "500") + " m, labelled with chainage "
                   "from A. Street map © OpenStreetMap contributors, for orientation only.", st["note"]))
    if L <= 3000:
        locd = locator_figure(rep, pw, 70 * mm)
        if locd is not None:
            story.append(Spacer(1, 4))
            story.append(locd)
    if L > 2000:
        story.append(PageBreak())
        story.append(P("2a. Route detail per km", st["h2"]))
        for k in rep["km"]:
            story.append(KeepTogether([route_figure(rep, pw, 112 * mm, k["from_m"], k["to_m"],
                                                    title=f"{_ch(k['from_m'])} to {_ch(k['to_m'])}"),
                                       Spacer(1, 4)]))
        story.append(legend_table(rep, st, pw))

    # 3. tables
    story.append(PageBreak())
    story.append(P("3. Condition by km and by 100 m section", st["h2"]))
    note = _not_shown_note(rep, _shown_params(rep))
    story.append(P("Kilometre-wise", st["h3"]))
    story.append(km_table(rep, st, pw))
    story.append(P(("Whole stretch: values over the full length; PCI = length-weighted mean of the km rows. "
                    if len(rep["km"]) > 1 else "") + "Potholes per row; cracking as % of surveyed length. "
                   + escape(note), st["note"]))
    story.append(Spacer(1, 4))
    story.append(P("100 m sections", st["h3"]))
    story.append(section_table(rep, st, pw))
    story.append(P(("Colour square = map colour. " if not single_mode(rep) else "Colour square = map colour (PCI "
                    "class). ") + "'low' = 50–80 % imaged; under 50 % the section is not rated.", st["note"]))

    # 4. potholes
    story.append(CondPageBreak(60 * mm))
    story.append(P("4. Pothole locations", st["h2"]))
    runs = s.get("pothole_runs_list") or []
    if runs:
        story.append(P("Each row is one run of consecutive frames labelled pothole. Coordinates are the middle of "
                       "the run; the pothole may be up to ~30 m further on in the direction of travel. "
                       "Check the photo before sending a crew.", st["note"]))
        story.append(pothole_table(rep, st, pw))
    else:
        story.append(P("No pothole flagged on the imaged frames.", st["body"]))

    # 5. maintenance
    story.append(CondPageBreak(90 * mm))
    story.append(P("5. Maintenance (IRC:82-2023)", st["h2"]))
    story.extend(maintenance_block(rep, st, pw))

    # 6. method and limitations
    story.append(CondPageBreak(70 * mm))
    story.append(P("6. Method and limitations", st["h2"]))
    for line in method_lines(rep):
        story.append(P(line, st["bullet"], bulletText="•"))
    story.append(Spacer(1, 3))
    for line in limitation_lines(rep):
        story.append(P(line, st["bullet"], bulletText="•"))
    story.append(P(f"Report {rep.get('report_id') if rep.get('report_id') is not None else 'preview (not stored)'}"
                   f" · data as of {escape(str(rep.get('data_as_of') or '?'))} · {s['segments_used']} frames used, "
                   f"{s['segments_expert_reviewed']} expert-reviewed · model {escape(rep['header']['model']['model'])}",
                   st["note"]))

    # annex
    story.append(PageBreak())
    story.append(P("Annex — Photographs", st["h2"]))
    story.append(P(f"{_plural(len(rep.get('annex') or []), 'frame')}: one per pothole run and per section with "
                   "the worst distress (up to 24). Each shows the road AHEAD of the marked chainage.", st["note"]))
    story.extend(annex_block(rep, st, image_fetcher))
    doc.build(story)
    return buf.getvalue()
