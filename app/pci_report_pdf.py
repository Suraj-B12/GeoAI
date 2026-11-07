"""
A4 PDF of an A->B chainage + IRC:82-2015 condition-rating report (DESIGN §8), from the report JSON of
app.pci_chainage.build_report. ReportLab only; no network; text in Paragraphs so it wraps; wide tables on
landscape pages; fonts embedded (Segoe UI for the text; Nirmala UI for Indic scripts such as Kannada in
requester-entered text). scripts/tests/test_pci_pdf.py checks that no two text spans overlap on any page and
that none leaves the page.

    render(report, image_fetcher=None) -> bytes      image_fetcher(segment_id) -> JPEG bytes | None
    unsupported_text(s) -> str | None                why the PDF cannot print s (None = fine)
"""
from __future__ import annotations

import io
import math
import unicodedata
from pathlib import Path
from typing import Callable, Optional
from xml.sax.saxutils import escape

from reportlab.graphics.shapes import Drawing, Line, PolyLine, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Frame, Image, KeepTogether, NextPageTemplate,
                                PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle)

FONT_DIR = Path(r"C:\Windows\Fonts")
MARGIN = 14 * mm

CLASS_COLORS = {"Good": "#15803d", "Fair–Good": "#84cc16", "Fair": "#eab308", "Poor–Fair": "#f97316",
                "Poor–Good": "#7c3aed", "Poor": "#b91c1c", "Insufficient data": "#9ca3af"}
BAND_COLORS = CLASS_COLORS
OVERALL_GREYS = {"Good": "#d6d3d1", "Fair–Good": "#bfbab6", "Fair": "#a8a29e", "Poor–Fair": "#78716c",
                 "Poor–Good": "#8f8984", "Poor": "#44403c", "Insufficient data": "#f5f5f4"}
BAND_LEGEND = {"Good": "Good", "Fair–Good": "Fair–Good (E_lo / E_hi)", "Fair": "Fair",
               "Poor–Fair": "Poor–Fair (E_lo / E_hi)", "Poor–Good": "Good on labels, Poor via skips",
               "Poor": "Poor", "Insufficient data": "Insufficient data"}
INK = colors.HexColor("#1c1917")
MUTED = colors.HexColor("#57534e")
RULE = colors.HexColor("#d6d3d1")
HEAD_BG = colors.HexColor("#f5f5f4")

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
    # conjuncts print as detached marks or wrong letters (checked by eye: "ಹೊ" -> dotted circle, "ಸ್ತೆ" -> "ಸತೆ").
    # Refuse rather than print a wrong road name. Nirmala UI stays registered for when shaping is added.
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
        return f"<i>{empty}</i>" if empty else ""
    if _needs_indic(s) and _fonts["indic"]:
        return f'<font name="{_fonts["indic"]}">{escape(s)}</font>'
    return escape(s)


# ============================================================ styles
def _styles():
    _register_fonts()
    R, B = _fonts["regular"], _fonts["bold"]
    base = dict(fontName=R, textColor=INK, leading=11.0, fontSize=8.6)
    st = {
        "body": ParagraphStyle("body", **base, spaceAfter=3),
        "small": ParagraphStyle("small", **{**base, "fontSize": 7.2, "leading": 9.0}),
        "cell": ParagraphStyle("cell", **{**base, "fontSize": 7.4, "leading": 9.0}),
        "cellb": ParagraphStyle("cellb", **{**base, "fontName": B, "fontSize": 7.4, "leading": 9.0}),
        "cellc": ParagraphStyle("cellc", **{**base, "fontSize": 7.4, "leading": 9.0, "alignment": TA_CENTER}),
        "head": ParagraphStyle("head", **{**base, "fontName": B, "fontSize": 7.4, "leading": 9.0}),
        "h1": ParagraphStyle("h1", **{**base, "fontName": B, "fontSize": 15, "leading": 19}, spaceAfter=4),
        "h2": ParagraphStyle("h2", **{**base, "fontName": B, "fontSize": 11.2, "leading": 14}, spaceBefore=8,
                             spaceAfter=4),
        "h3": ParagraphStyle("h3", **{**base, "fontName": B, "fontSize": 9.2, "leading": 12}, spaceBefore=5,
                             spaceAfter=2),
        "note": ParagraphStyle("note", **{**base, "fontSize": 7.4, "leading": 9.4, "textColor": MUTED}),
        "caption": ParagraphStyle("caption", **{**base, "fontSize": 7.0, "leading": 8.6}, alignment=TA_LEFT),
    }
    st["small"].textColor = MUTED
    return st


def P(text, style) -> Paragraph:
    return Paragraph(text, style)


def fnum(x, nd=1) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    return f"{x:.{nd}f}"


def frange(lo, hi, nd=1) -> str:
    if lo is None:
        return "–"
    a, b = fnum(lo, nd), fnum(hi, nd)
    return a if a == b else f"{a}–{b}"


def _cls(u: dict) -> str:
    f = (u.get("rating") or {}).get("final")
    return f["class"] if f else "Insufficient data"


def _band(u: dict, param: str) -> str:
    r = u.get("rating") or {}
    if not r.get("rated"):
        return "Insufficient data"
    p = next((x for x in r.get("params", []) if x["param"] == param), None)
    return p["band"] if p else "Insufficient data"


def _band_text(p: dict, short: bool = False) -> str:
    """A parameter's band for print; a band whose worse end comes only from skipped segments is spelled out."""
    if p.get("band_note"):
        return (f"{escape(p['band_lo'])} / {escape(p['band_hi'])} if skips" if short else escape(p["band_note"]))
    return escape(p["band"])


def _area(u: dict) -> str:
    """' · area: Fair' — the nominal small-pothole area band beside the pothole % band (F5)."""
    h = (u.get("rating") or {}).get("headline") or {}
    b = h.get("pothole_nominal_area_band")
    return f"<br/><font size='6.2' color='#57534e'>area: {escape(b)}</font>" if b else ""


def _plural(n, word) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _badge(text: str, color: str) -> str:
    return f'<font color="{color}"><b>{escape(text)}</b></font>'


def _table(data, widths, *, header_rows=1, zebra=False, extra=None) -> Table:
    t = Table(data, colWidths=widths, repeatRows=header_rows, hAlign="LEFT")
    style = [("GRID", (0, 0), (-1, -1), 0.4, RULE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
             ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header_rows:
        style.append(("BACKGROUND", (0, 0), (-1, header_rows - 1), HEAD_BG))
    if zebra:
        for r in range(header_rows, len(data)):
            if (r - header_rows) % 2 == 1:
                style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#fafaf9")))
    t.setStyle(TableStyle(style + (extra or [])))
    return t


# ============================================================ page furniture
class _Doc(BaseDocTemplate):
    def __init__(self, buf, report):
        super().__init__(buf, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN + 6 * mm,
                         bottomMargin=MARGIN + 4 * mm, title="IRC:82-2015 condition rating (screening estimate)",
                         author="GeoAI pavement survey (automated)", subject="A to B chainage report")
        self.report = report
        pw, ph = A4
        lw, lh = landscape(A4)
        fp = Frame(MARGIN, MARGIN + 4 * mm, pw - 2 * MARGIN, ph - 2 * MARGIN - 10 * mm, id="p", leftPadding=0,
                   rightPadding=0, topPadding=0, bottomPadding=0)
        fl = Frame(MARGIN, MARGIN + 4 * mm, lw - 2 * MARGIN, lh - 2 * MARGIN - 10 * mm, id="l", leftPadding=0,
                   rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates([PageTemplate("portrait", [fp], pagesize=A4, onPage=self._furniture),
                               PageTemplate("landscape", [fl], pagesize=landscape(A4), onPage=self._furniture)])

    def _furniture(self, canv, doc):
        canv.saveState()
        w, h = canv._pagesize
        canv.setFont(_fonts["regular"], 6.8)
        canv.setFillColor(MUTED)
        rid = self.report.get("report_id")
        left = "IRC:82-2015 distress-based condition rating — screening estimate from automated image classification"
        canv.drawString(MARGIN, h - MARGIN, left)
        right = f"Report {rid}" if rid is not None else "Preview (not stored)"
        canv.drawRightString(w - MARGIN, h - MARGIN, right)
        canv.setStrokeColor(RULE)
        canv.setLineWidth(0.4)
        canv.line(MARGIN, h - MARGIN - 2.2 * mm, w - MARGIN, h - MARGIN - 2.2 * mm)
        canv.line(MARGIN, MARGIN + 2.2 * mm, w - MARGIN, MARGIN + 2.2 * mm)
        foot = f"Data as of {self.report.get('data_as_of') or '?'} · generated {self.report.get('generated_at_ist')}"
        canv.drawString(MARGIN, MARGIN - 1 * mm, foot)
        canv.drawRightString(w - MARGIN, MARGIN - 1 * mm, f"Page {canv.getPageNumber()}")
        canv.restoreState()


# ============================================================ figures
def _fit(vertices, w, h, pad):
    """Local metres -> drawing coordinates, same scale on both axes."""
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

    def __init__(self, w, h, font):
        self.boxes, self.w, self.h, self.font = [], w, h, font

    def obstacle_line(self, pts, r=1.3):
        """Mark a drawn line (flat [x0, y0, x1, y1, ...]) as an obstacle, sampled every ~3 pt."""
        for i in range(0, len(pts) - 2, 2):
            x1, y1, x2, y2 = pts[i], pts[i + 1], pts[i + 2], pts[i + 3]
            n = max(1, int(math.hypot(x2 - x1, y2 - y1) / 3.0))
            for k in range(n + 1):
                x, y = x1 + (x2 - x1) * k / n, y1 + (y2 - y1) * k / n
                self.boxes.append((x - r, y - r, x + r, y + r))

    def place(self, d: Drawing, x, y, text, size=6.5, color=INK, anchor="start", force=False) -> bool:
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
        d.add(String(x0, y, text, fontName=self.font, fontSize=size, fillColor=color))
        return True


def _point_at(vertices, m):
    for (x1, y1, a), (x2, y2, b) in zip(vertices, vertices[1:]):
        if a <= m <= b:
            t = 0.0 if b == a else (m - a) / (b - a)
            return x1 + t * (x2 - x1), y1 + t * (y2 - y1), (x1, y1, x2, y2)
    x, y, _ = vertices[-1]
    return x, y, (vertices[-2][0], vertices[-2][1], x, y) if len(vertices) > 1 else (x, y, x, y)


def route_figure(rep: dict, w: float, h: float, m0: Optional[float] = None, m1: Optional[float] = None,
                 title: Optional[str] = None) -> Drawing:
    """The track coloured by the pothole band. m0/m1: a detail panel of that measure window (ticks every 100 m,
    all labelled); without them the whole stretch (labels thinned by length)."""
    d = Drawing(w, h)
    d.add(Rect(0, 0, w, h, strokeColor=RULE, fillColor=None, strokeWidth=0.5))
    L = rep["summary"]["length_m"]
    lo, hi = (0.0, L) if m0 is None else (m0, m1)
    allv = rep["path"]["vertices"]
    verts = [v for v in allv if lo - 1e-6 <= v[2] <= hi + 1e-6] or allv
    f, s_ = _fit(verts, w, h, 14 * mm)
    labels = _Labels(w, h, _fonts["regular"])
    if title:
        labels.place(d, 4, h - 10, title, size=7, force=True)
    for sec in rep["sections"]:
        if sec["to_m"] < lo - 1e-6 or sec["from_m"] > hi + 1e-6:
            continue
        col = colors.HexColor(BAND_COLORS.get(_band(sec, "potholes"), "#9ca3af"))
        for part in sec["geometry"]["coordinates"]:
            pts = []
            for lon, lat in part:
                pts.extend(f(lon, lat))
            if len(pts) >= 4:
                d.add(PolyLine(pts, strokeColor=col, strokeWidth=2.6, strokeLineCap=1, strokeLineJoin=1))
                labels.obstacle_line(pts)
    for gp, part in zip(rep["path"]["gaps"], rep["path"]["geometry"]["gaps"]["coordinates"]):
        if gp["to_m"] < lo or gp["from_m"] > hi:
            continue
        pts = []
        for lon, lat in part:
            pts.extend(f(lon, lat))
        d.add(PolyLine(pts, strokeColor=MUTED, strokeWidth=1.0, strokeDashArray=[3, 2]))
        labels.obstacle_line(pts)
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
        ln = 4.5 if major else 2.6
        d.add(Line(x - nx_ * ln, y - ny_ * ln, x + nx_ * ln, y + ny_ * ln, strokeColor=INK,
                   strokeWidth=0.8 if major else 0.45))
        if m > 0 and abs(m / step_lab - round(m / step_lab)) < 1e-6:
            anchor = "start" if nx_ > 0.35 else ("end" if nx_ < -0.35 else "middle")
            ly = y + ny_ * 7 + (0 if ny_ > 0.35 else (-6 if ny_ < -0.35 else -2))
            mm_ = int(round(m))
            txt = f"Km {mm_ // 1000}" if mm_ % 1000 == 0 else f"{mm_ // 1000}+{mm_ % 1000:03d}"
            if not labels.place(d, x + nx_ * 7, ly, txt, size=6.2, anchor=anchor):
                anchor = "end" if anchor == "start" else ("start" if anchor == "end" else "middle")
                labels.place(d, x - nx_ * 7, y - ny_ * 7 + (0 if ny_ < -0.35 else (-6 if ny_ > 0.35 else -2)),
                             txt, size=6.2, anchor=anchor)
        m += tick
    for name, p, mp in (("A", rep["header"]["a"], 0.0), ("B", rep["header"]["b"], L)):
        if not lo - 1e-6 <= mp <= hi + 1e-6:
            continue
        x, y = f(p["snapped_lon"], p["snapped_lat"])
        d.add(Rect(x - 3, y - 3, 6, 6, fillColor=INK, strokeColor=colors.white, strokeWidth=0.6))
        for dx, dy in ((5, 3), (-11, 3), (5, -9), (-11, -9)):
            if labels.place(d, x + dx, y + dy, name, size=9):
                break
    nx0, ny0 = w - 9 * mm, h - 15 * mm
    d.add(Polygon([nx0, ny0 + 9, nx0 - 3, ny0, nx0 + 3, ny0], fillColor=INK, strokeColor=INK))
    labels.place(d, nx0, ny0 + 10.5, "N", size=7, anchor="middle", force=True)
    target = (w * 0.22) / s_
    nice = 10.0
    for c in (10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000):
        if c <= target:
            nice = c
    bx0, by0 = 6 * mm, 6 * mm
    d.add(Rect(bx0, by0, nice * s_, 2.2, fillColor=INK, strokeColor=INK))
    labels.place(d, bx0, by0 + 4, f"{int(nice)} m" if nice < 1000 else f"{nice / 1000:g} km", size=6.5, force=True)
    return d


def legend_table(st, title: str = "Pothole band") -> Table:
    cells = []
    for name, col in BAND_COLORS.items():
        cells.append(P(f'<font color="{col}">■</font> {escape(BAND_LEGEND[name])}', st["cell"]))
    cells.append(P('<font color="#57534e">- - -</font> gap (straight line)', st["cell"]))
    rows = [cells[i:i + 4] for i in range(0, len(cells), 4)]
    rows[-1] += [""] * (4 - len(rows[-1]))
    t = Table(rows, colWidths=[44 * mm] * 4, hAlign="LEFT")
    t.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 0),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
    return t


def strip_chart(rep: dict, w: float) -> Drawing:
    """Rows along chainage: final class, cracking band, pothole band, coverage, pothole runs."""
    rows = [("Pothole band", "potholes"), ("Cracking band", "cracking"), ("Coverage", "coverage"),
            ("Pothole runs", "runs"), ("Overall, App. 4 (secondary)", "final")]
    lab_w = 36 * mm
    row_h = 10.0
    top_pad, axis_h = 4.0, 16.0
    h = top_pad + row_h * len(rows) + axis_h
    d = Drawing(w, h)
    L = rep["summary"]["length_m"]
    x0, x1 = lab_w, w - 3
    sx = (x1 - x0) / L
    labels = _Labels(w, h, _fonts["regular"])
    for ri, (name, key) in enumerate(rows):
        y = h - top_pad - (ri + 1) * row_h
        if key == "final":
            d.add(String(0, y + 3.0, name, fontName=_fonts["regular"], fontSize=6.0, fillColor=MUTED))
        else:
            d.add(String(0, y + 3.0, name, fontName=_fonts["regular"], fontSize=6.4, fillColor=INK))
        d.add(Rect(x0, y + 0.8, x1 - x0, row_h - 1.6, fillColor=colors.HexColor("#f5f5f4"), strokeColor=None))
        for sec in rep["sections"]:
            a, b = x0 + sec["from_m"] * sx, x0 + sec["to_m"] * sx
            r = sec["rating"]
            if key == "final":
                col = OVERALL_GREYS.get(_cls(sec), "#e7e5e4")
            elif key in ("cracking", "potholes"):
                col = BAND_COLORS.get(_band(sec, key), "#e7e5e4")
            elif key == "coverage":
                cov = sec["quantities"]["coverage"]
                g = int(235 - 160 * max(0.0, min(1.0, cov)))
                col = f"#{g:02x}{g:02x}{g:02x}"
            else:
                col = None
                n = sec["quantities"]["runs"]["potholes"]
                if n:
                    d.add(Rect(a, y + 0.8, max(0.6, b - a), row_h - 1.6, fillColor=colors.HexColor("#7f1d1d"),
                               strokeColor=None, fillOpacity=min(1.0, 0.35 + 0.2 * n)))
                continue
            d.add(Rect(a, y + 0.8, max(0.4, b - a), row_h - 1.6, fillColor=colors.HexColor(col), strokeColor=None))
    # axis
    ya = axis_h - 5
    d.add(Line(x0, ya + 3, x1, ya + 3, strokeColor=INK, strokeWidth=0.5))
    for gp in rep["path"]["gaps"]:
        a, b = x0 + gp["from_m"] * sx, x0 + gp["to_m"] * sx
        d.add(Line(a, ya + 3, max(a + 0.8, b), ya + 3, strokeColor=colors.white, strokeWidth=1.6))
        d.add(Line(a, ya + 3, max(a + 0.8, b), ya + 3, strokeColor=MUTED, strokeWidth=0.8, strokeDashArray=[2, 1.5]))
    step = 100.0
    for cand in (100, 200, 500, 1000, 2000, 5000):
        step = cand
        if pdfmetrics.stringWidth("Km 00+000", _fonts["regular"], 6) * 1.6 < cand * sx:
            break
    m = 0.0
    while m <= L + 1e-6:
        x = x0 + m * sx
        d.add(Line(x, ya + 1, x, ya + 5, strokeColor=INK, strokeWidth=0.4))
        km, rest = int(round(m)) // 1000, int(round(m)) % 1000
        labels.place(d, x, ya - 6, f"{km}+{rest:03d}", size=6.0, anchor="middle")
        m += step
    labels.place(d, x1, ya - 6, f"{int(round(L)) // 1000}+{int(round(L)) % 1000:03d}", size=6.0, anchor="end")
    return d


# ============================================================ content blocks
def _entered(v) -> str:
    return (user_text(v) + "  <font color='#57534e'>(entered by requester, unverified)</font>") if v else "not entered"


def _header_block(rep, st):
    h = rep["header"]
    rc_from, rc_to = h.get("road_chainage_from"), h.get("road_chainage_to")
    road_ch = (f"{rc_from['text']} to {rc_to['text']} — {h['road_chainage_note']}" if rc_from
               else h["road_chainage_note"])
    a, b = h["a"], h["b"]
    dirs = "; ".join(f"{x['pass']}: " + ("direction unknown" if "unknown" in x["direction"]
                                         else f"vehicle travelled {x['direction']}") for x in h["directions"][:6])
    if len(h["directions"]) > 6:
        dirs += f"; … ({len(h['directions'])} passes)"
    pipe = "; ".join(f"video pipeline {x['pipeline_version']} (config {str(x['config_sha256'])[:12]}…)"
                     for x in h["model"]["video_pipeline"][:3])
    rows = [
        ("Name of the road", _entered(h.get("road_name"))),
        ("Road number", _entered(h.get("road_number"))),
        ("Road category / IRC table", f"{escape(h['category_name'])} — IRC:82-2015 {escape(h['table_title'])}"),
        ("Km from / to (relative)", f"{h['km_from']} at A to {h['km_to']} at B"
         + (" (≥: gaps counted as straight lines)" if rep["summary"]["gaps"] else "")),
        ("Road chainage", escape(road_ch)),
        ("Point A", f"{a['snapped_lat']:.6f}, {a['snapped_lon']:.6f} (entered point snapped {a['distance_m']:.0f} m to the nearest segment)"),
        ("Point B", f"{b['snapped_lat']:.6f}, {b['snapped_lon']:.6f} (entered point snapped {b['distance_m']:.0f} m to the nearest segment)"),
        ("Via points", ", ".join(f"{v['snapped_lat']:.5f}, {v['snapped_lon']:.5f}" for v in h["via"]) or "none"),
        ("Direction (derived from camera heading)", escape(dirs)),
        ("Lane imaged", escape(h["lane_imaged"])),
        ("Surveyed lane width", f"{h['lane_width_m']:.1f} m — {escape(h['lane_width_note'])}"),
        ("Type of surface", _entered(h.get("surface_type"))),
        ("Date(s) of observation", escape(", ".join(h["survey_dates_ist"]) or "–")),
        ("Season", escape("; ".join(h["season"]) or "–") + " — IRC §5.3 asks for pre- and post-monsoon surveys"),
        ("Weather (as entered on the survey form)", escape(", ".join(h["weather_as_entered_on_survey_form"]) or "not entered")),
        ("Survey method", escape(h["method"])),
        ("Device", escape(", ".join(h["devices"]))),
        ("Model / pipeline", escape(f"{h['model']['pipeline']}: {h['model']['model']}; prompts {h['model']['prompts']}; "
                                    f"gate {h['model']['gate']}; {pipe}")),
        ("Report id", escape(str(rep.get("report_id")) if rep.get("report_id") is not None
                             else "preview (not stored; numbers may change as reviews arrive)")),
        ("Data as of / generated", escape(f"{rep.get('data_as_of')} / {rep.get('generated_at_ist')}")),
        ("Prepared by", _entered(h.get("prepared_by"))),
    ]
    data = [[P(k, st["cellb"]), P(v, st["cell"])] for k, v in rows]
    return _table(data, [52 * mm, 130 * mm], header_rows=0)


def _summary_block(rep, st):
    s = rep["summary"]
    hl = s.get("headline") or {}
    f = s.get("final")
    out = []
    if not f:
        out.append(P(_badge("Insufficient data for the stretch", CLASS_COLORS["Insufficient data"]) +
                     f" — coverage {100 * s['coverage']:.0f} % (at least 50 % imaged and some classified length "
                     "needed).", st["body"]))
    rows = [[P("Headline (assessed parameters)", st["head"]), P("Estimate (% of surveyed length with the distress "
                                                                   "visible; E_lo–E_hi)", st["head"]),
             P("IRC band", st["head"]), P("Note", st["head"])]]
    sprm = {p["param"]: p for p in (rep["stretch"]["rating"].get("params") or [])}
    for p in ("potholes", "cracking"):
        e = hl.get(p)
        if not e:
            continue
        band = e["band"]
        col = BAND_COLORS.get(band, "#1c1917")
        note = ("label unvalidated on Bengaluru video (0 expert labels); Attain linear crack P 0.89 / R 0.82, "
                "alligator recall 0.08; upper bound on % area for localised cracks" if p == "cracking"
                else "label unvalidated on Bengaluru video (0 expert labels); Attain precision 0.43, recall 0.25; "
                     "one pothole flags a whole 5 m segment, see runs per km")
        rows.append([P(p.capitalize(), st["cellb"]), P(frange(e["E_lo"], e["E_hi"]) + " % (est.)", st["cell"]),
                     P(f'<font color="{col}"><b>{_band_text(sprm.get(p, e))}</b></font>'
                       + (_area(rep["stretch"]) if p == "potholes" else ""), st["cell"]),
                     P(note, st["cell"])])
    if hl.get("pothole_runs") is not None:
        rows.append([P("Pothole runs", st["cellb"]),
                     P(f"{_plural(hl['pothole_runs'], 'run')} = {fnum(hl.get('pothole_runs_per_km'))} per surveyed km "
                       f"(est.)",
                       st["cell"]), P("primary pothole figure", st["cell"]),
                     P("consecutive positive segments of one pass = one occurrence", st["cell"])])
    if hl.get("pothole_nominal_area_pct") is not None:
        rows.append([P("Nominal pothole area", st["cellb"]),
                     P(f"{fnum(hl['pothole_nominal_area_pct'], 3)} % of the surveyed lane", st["cell"]),
                     P(escape(hl.get("pothole_nominal_area_band") or ""), st["cell"]),
                     P("runs × 0.0314 m² (IRC §7.5.3.4 small pothole); sensitivity, not a bound", st["cell"])])
    if len(rows) > 1:
        out.append(_table(rows, [30 * mm, 38 * mm, 38 * mm, 76 * mm]))
        out.append(Spacer(1, 4))
    if s.get("resolution_note"):
        out.append(P("Resolution: " + escape(s["resolution_note"]), st["note"]))
        out.append(Spacer(1, 3))
    for note in s.get("priority_notes") or []:
        box = Table([[P("<b>" + escape(note.split(". ", 1)[0]) + ".</b> " + escape(note.split(". ", 1)[1])
                        if ". " in note else escape(note), st["body"])]], colWidths=[182 * mm], hAlign="LEFT")
        box.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 1.0, INK), ("LEFTPADDING", (0, 0), (-1, -1), 5),
                                 ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("TOPPADDING", (0, 0), (-1, -1), 4),
                                 ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
        out.append(box)
        out.append(Spacer(1, 4))
    if f:
        out.append(P(f"Overall (Appendix 4, secondary): {fnum(f['lo'])}–{fnum(f['hi'])}, {escape(f['class'])} — "
                     + escape((f.get("note") or "").replace("Overall (Appendix 4, secondary): ", ""))
                     + " Never use it on its own.", st["note"]))
        out.append(Spacer(1, 3))
    gaps = (f"{s['gaps']} gap(s) totalling ≥ {s['gap_m']:.0f} m (straight line, unsurveyed)" if s["gaps"]
            else "no gaps")
    facts = (f"Length {'≥ ' if s['gaps'] else ''}{s['length_m']:.0f} m ({s['chainage_a']} to {s['chainage_b']}); coverage "
             f"{100 * s['coverage']:.1f} % imaged ({s['coverage_status'].replace('_', ' ')}); {gaps}; "
             f"{s['segments_used']} segments used, {s['segments_expert_reviewed']} expert-reviewed.")
    out.append(P(escape(facts), st["body"]))
    secs = ", ".join(f"{k}: {v}" for k, v in s["sections_by_class"].items())
    kms = ", ".join(f"{k}: {v}" for k, v in s["km_by_class"].items())
    pot = {}
    for sec in rep["sections"]:
        b = _band(sec, "potholes")
        pot[b] = pot.get(b, 0) + 1
    pots = ", ".join(f"{k}: {v}" for k, v in sorted(pot.items()))
    out.append(P(escape(f"100 m sections by pothole band — {pots}."), st["body"]))
    out.append(P(escape(f"By bracketed overall class (secondary) — {secs}. Km rows — {kms}."), st["note"]))
    return out


def _e_cell(q, key, reported=False):
    e = (q["reported"] if reported else q["extent"]).get(key)
    if e is None:
        return "–"
    return frange(e["lo"], e["hi"])


def _km_rows(rep) -> list:
    """Km rows to print; a stretch with one km row prints the whole stretch only (they are identical)."""
    return rep["km"] if len(rep["km"]) > 1 else []


def _proforma_table(rep, st, width):
    head = ["Chainage from", "Chainage to", "Cracks (%, est.)", "Patch work (%)", "Potholes (%, est.)",
            "Bleeding (%, est.)", "Depression (%)", "Edge breaking (%, est.)", "Ravelling (%)", "Shoving (%)",
            "Rut depth (mm)", "Drainage and shoulder", "Total", "Coverage (%)"]
    rows = [[P(h, st["head"]) for h in head]]
    na = P("<font color='#57534e'>not assessed</font>", st["cellc"])
    for u in _km_rows(rep) + [rep["stretch"]]:
        q = u["quantities"]
        rated = u["rating"].get("rated")
        lab = "Stretch" if u is rep["stretch"] else ""
        frm = u["chainage_from"] + (f"<br/>{escape(u['road_chainage_from']['text'])}" if u.get("road_chainage_from") else "")
        to = u["chainage_to"] + (f"<br/>{escape(u['road_chainage_to']['text'])}" if u.get("road_chainage_to") else "")
        if lab:
            frm = f"<b>Whole stretch</b><br/>{frm}"
        if u.get("part_km"):
            to += "<br/><font color='#57534e'>(part-km)</font>"
        if not rated:
            vals = [P(frm, st["cell"]), P(to, st["cell"])] + \
                   [P("<font color='#57534e'>insufficient data</font>", st["cellc"])] * 11 + \
                   [P(fnum(100 * q["coverage"], 0), st["cellc"])]
        else:
            vals = [P(frm, st["cell"]), P(to, st["cell"]),
                    P(_e_cell(q, "cracking") + f"<br/><font size='6.2' color='#57534e'>allig. "
                      f"{_e_cell(q, 'cracking_alligator')} · linear {_e_cell(q, 'cracking_linear')}</font>", st["cellc"]),
                    na, P(_e_cell(q, "potholes") + f"<br/><font size='6.2' color='#57534e'>{_plural(q['runs']['potholes'], 'run')}</font>",
                          st["cellc"]),
                    P(_e_cell(q, "Bleeding", True) + "<br/><font size='6.2' color='#57534e'>unvalidated</font>", st["cellc"]),
                    na, P(_e_cell(q, "Edge Breaking", True) + "<br/><font size='6.2' color='#57534e'>unvalidated</font>",
                          st["cellc"]), na, na, na, na, na, P(fnum(100 * q["coverage"], 0), st["cellc"])]
        rows.append(vals)
    cw = [22, 22, 25, 17, 21, 21, 18, 21, 17, 16, 17, 20, 15, 16]
    scale = width / (sum(cw) * mm)
    t = _table(rows, [c * mm * scale for c in cw], zebra=True,
               extra=[("BACKGROUND", (0, len(rows) - 1), (-1, len(rows) - 1), colors.HexColor("#eef2ff"))])
    return t


def _rating_table(u, st, title, width):
    r = u["rating"]
    out = [P(title, st["h3"])]
    if not r.get("rated"):
        out.append(P(f"Insufficient data (coverage {100 * u['quantities']['coverage']:.0f} %).", st["body"]))
        return out
    rows = [[P(h, st["head"]) for h in ("Parameter", "Input E (lo–hi)", "Band", "Rating value (worst–best)",
                                        "Weight", "Weighted rating (worst–best)", "Basis")]]
    for p in r["params"]:
        unit = p["unit"]
        if p["assessed"]:
            rows.append([P(escape(p["name"]), st["cellb"]),
                         P(f"{frange(p['E_lo'], p['E_hi'])} {unit}", st["cellc"]),
                         P(f'<font color="{BAND_COLORS.get(p["band"], "#1c1917")}"><b>{_band_text(p)}</b></font>'
                           + (_area(u) if p["param"] == "potholes" else ""), st["cellc"]),
                         P(frange(p["rating_worst"], p["rating_best"]), st["cellc"]),
                         P(f"{p['weight']:.2f}", st["cellc"]),
                         P(frange(p["weighted_worst"], p["weighted_best"]), st["cellc"]),
                         P("assessed (weak): " + escape(p["evidence"]), st["cell"])])
        else:
            rows.append([P(escape(p["name"]), st["cellb"]), P("not assessed", st["cellc"]),
                         P("not assessed", st["cellc"]), P("1.0–3.0 (bracket)", st["cellc"]),
                         P(f"{p['weight']:.2f}", st["cellc"]),
                         P(frange(p["weighted_worst"], p["weighted_best"]), st["cellc"]),
                         P(escape(p["evidence"]), st["cell"])])
    f = r["final"]
    rows.append([P("<b>Final rating value</b>", st["cell"]), "", "", "", "",
                 P(f"<b>{fnum(f['lo'])}–{fnum(f['hi'])}</b>", st["cellc"]),
                 P("<font color='#57534e'>" + escape(f["label"]) + " — secondary; use the bands</font>",
                   st["cell"])])
    sens = r.get("sensitivity", {})
    lines = []
    if "sum_wr_over_sum_w" in sens:
        x = sens["sum_wr_over_sum_w"]
        lines.append(f"Σwr/Σw {fnum(x['lo'])}–{fnum(x['hi'])} ({x['class']})")
    if "assessed_only" in sens:
        x = sens["assessed_only"]
        lines.append(f"averaged over assessed only {fnum(x['lo'])}–{fnum(x['hi'])} ({x['class']})")
    if "uncorrected" in sens:
        x = sens["uncorrected"]
        lines.append(f"no skip correction {fnum(x['lo'])}–{fnum(x['hi'])} ({x['class']})")
    lines.append("coverage rule 80 %: " + ("rated" if sens.get("coverage_rule_80") else "not rated") +
                 "; 50 %: " + ("rated" if sens.get("coverage_rule_50") else "not rated"))
    cw = [30, 24, 20, 26, 13, 27, 0]
    cw[-1] = width / mm - sum(cw)
    t = _table(rows, [c * mm for c in cw],
               extra=[("SPAN", (0, len(rows) - 1), (4, len(rows) - 1)),
                      ("BACKGROUND", (0, len(rows) - 1), (-1, len(rows) - 1), HEAD_BG)])
    out.append(t)
    out.append(P("Sensitivity: " + escape("; ".join(lines)) + ".", st["note"]))
    return out


def _sections_table(rep, st, width):
    head = ["#", "From", "To", "Coverage (%)", "Cracking E (%, est.)", "Cracking band", "Potholes E (%, est.)",
            "Pothole band", "Runs", "Overall (lo–hi)", "Overall class (secondary)", "Pass(es) used (date)",
            "Also surveyed / other direction"]
    rows = [[P(h, st["head"]) for h in head]]
    for s in rep["sections"]:
        q, r = s["quantities"], s["rating"]
        prm = {p["param"]: p for p in r.get("params", [])}
        f = r.get("final")
        cls = _cls(s)
        ch = s.get("chosen_pass")
        also = [f"{x['date_ist']} ({x['session_id'][-4:]})" for x in s.get("also_surveyed", [])]
        other = [f"{x['date_ist']} ({x['session_id'][-4:]})" for x in s.get("other_direction", [])]
        extra = ", ".join(also) + (("; other direction: " + ", ".join(other)) if other else "")
        if r.get("rated"):
            cells = [_e_cell(q, "cracking"),
                     f'<font color="{BAND_COLORS.get(prm["cracking"]["band"], "#000")}"><b>{_band_text(prm["cracking"], True)}</b></font>',
                     _e_cell(q, "potholes"),
                     f'<font color="{BAND_COLORS.get(prm["potholes"]["band"], "#000")}"><b>{_band_text(prm["potholes"], True)}</b></font>'
                     + _area(s),
                     str(q["runs"]["potholes"]), f"{fnum(f['lo'])}–{fnum(f['hi'])}"]
        else:
            cells = ["–", "–", "–", "–", "–", "–"]
        low = " <font color='#57534e'>(low)</font>" if q["coverage_status"] == "low_coverage" else ""
        rows.append([P(str(s["index"]), st["cellc"]), P(s["chainage_from"], st["cell"]), P(s["chainage_to"], st["cell"]),
                     P(fnum(100 * q["coverage"], 0) + low, st["cellc"])] +
                    [P(c, st["cellc"]) for c in cells] +
                    [P(f"<font color='#57534e'>{escape(cls)}</font>", st["cellc"]),
                     P(escape(", ".join(f"{x['date_ist']} ({x['session_id'][-4:]})" for x in s.get("passes_used") or [])
                              or (f"{ch['date_ist']} ({ch['session_id'][-4:]})" if ch else "–")), st["cell"]),
                     P(escape(extra) or "–", st["cell"])])
    cw = [8, 18, 18, 15, 18, 17, 18, 17, 10, 17, 22, 24, 0]
    cw[-1] = width / mm - sum(cw)
    return _table(rows, [c * mm for c in cw], zebra=True)


def _treatments_block(rep, st):
    out = [P("<b>" + escape(rep["treatment_any_class"]) + "</b>", st["body"]),
           P(escape(rep["treatment_heading"]), st["note"]), Spacer(1, 3)]
    units = [("Whole stretch", rep["stretch"])] + [(f"Km row {u['chainage_from']} to {u['chainage_to']}", u)
                                                   for u in _km_rows(rep)]
    for title, u in units:
        t = u["treatment"]
        pb = _band(u, "potholes")
        block = [P(escape(title) + " — pothole band " + _badge(pb, BAND_COLORS.get(pb, "#000")) +
                   f" <font color='#57534e'>(overall {escape(_cls(u))}, secondary)</font>", st["h3"])]
        if (u["quantities"].get("runs") or {}).get("potholes"):
            block.append(P("<b>Potholes present: rectify straightway (IRC:82 §5.3), whatever the overall class.</b>",
                           st["body"]))
        block.append(P(escape(t["class"]), st["body"]))
        if t["distress"]:
            rows = [[P(h, st["head"]) for h in ("Distress (label)", "Worst severity (visually judged)",
                                                "Candidate treatment", "IRC:82 clause")]]
            for d in t["distress"]:
                lab = escape(d["label"]) + (" <font color='#b91c1c'>(label unvalidated)</font>" if d["note"] else "")
                rows.append([P(lab, st["cell"]), P(escape(d["severity"]), st["cellc"]), P(escape(d["treatment"]), st["cell"]),
                             P(escape(d["clause"]), st["cellc"])])
            block.append(_table(rows, [38 * mm, 26 * mm, 96 * mm, 22 * mm]))
        out.append(KeepTogether(block) if len(t["distress"]) < 12 else block[0])
        if len(t["distress"]) >= 12:
            out.extend(block[1:])
    return out


def _annex_block(rep, st, image_fetcher):
    items = rep["annex"]
    if not items:
        return [P("No frames selected (no pothole runs and no section whose band can be Poor, or no stored images).",
                  st["body"])]
    cells = []
    col_w = 88 * mm
    for a in items:
        img = None
        data = image_fetcher(a["segment_id"]) if image_fetcher else None
        if data:
            try:
                from PIL import Image as PILImage
                with PILImage.open(io.BytesIO(data)) as im:
                    iw, ih = im.size
                hgt = col_w * ih / iw
                if hgt > 52 * mm:
                    hgt = 52 * mm
                img = Image(io.BytesIO(data), width=hgt * iw / ih, height=hgt)
            except Exception:
                img = None
        rc = a.get("road_chainage")
        cap = (f"<b>{escape(a['chainage'])}</b>" + (f" · road {escape(rc['text'].split(' (')[0])}" if rc else "") +
               f" — {escape(a['reason'])}<br/>{a['lat']:.6f}, {a['lon']:.6f} · {escape(a.get('time_ist') or '?')}<br/>"
               f"Labels: {escape(', '.join(a['labels']) or '–')} · severity {escape(str(a.get('severity')))} · "
               f"confidence {fnum(a.get('confidence'), 2)}<br/>{escape(a['review_state'])} · "
               f"<i>{escape(a['note'])}</i><br/><font color='#57534e'>{escape(a['segment_id'])} · "
               f"sha256 {escape(str(a.get('image_sha256') or '')[:16])}…</font>")
        inner = [img if img is not None else P("<i>image not available</i>", st["caption"]), Spacer(1, 2),
                 P(cap, st["caption"])]
        cells.append(inner)
    rows = [cells[i:i + 2] for i in range(0, len(cells), 2)]
    if len(rows[-1]) == 1:
        rows[-1].append("")
    out = []
    for r in rows:
        t = Table([r], colWidths=[col_w + 3 * mm] * 2, hAlign="LEFT")
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                               ("RIGHTPADDING", (0, 0), (-1, -1), 3 * mm), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
        out.append(t)
    return out


# ============================================================ render
def render(rep: dict, image_fetcher: Optional[Callable[[str], Optional[bytes]]] = None) -> bytes:
    st = _styles()
    buf = io.BytesIO()
    doc = _Doc(buf, rep)
    pw = A4[0] - 2 * MARGIN
    lw = landscape(A4)[0] - 2 * MARGIN
    story = []
    title = rep["header"].get("road_name")
    story.append(P("IRC:82-2015 distress-based condition rating", st["h1"]))
    story.append(P("Screening estimate from automated image classification — A to B stretch"
                   + (f": {user_text(title, '')}" if title else ""), st["h3"]))
    story.append(P("Not an IRC:82 Appendix-1/2 survey. Every number is an estimate (est.) from phone video and a "
                   "vision-language model; no label has been validated against Bengaluru expert labels.", st["note"]))
    story.append(Spacer(1, 4))
    story.append(P("1. Header (Proforma-1 fields)", st["h2"]))
    story.append(_header_block(rep, st))
    story.append(P("2. Summary", st["h2"]))
    story.extend(_summary_block(rep, st))
    story.append(CondPageBreak(150 * mm))
    story.append(P("3. Route", st["h2"]))
    story.append(route_figure(rep, pw, 118 * mm))
    story.append(Spacer(1, 2))
    story.append(legend_table(st))
    L = rep["summary"]["length_m"]
    story.append(P("Track = the surveyed segment centres, coloured by each 100 m section's POTHOLE band (the headline; "
                   "the overall class is never used for colour); ticks every " + ("100" if L <= 3000 else "500") +
                   " m, longer ticks every km, chainage from A; no basemap."
                   + (" Per-km detail panels follow the strip chart." if L > 2000 else "") + " "
                   + escape(rep["accuracy"]), st["note"]))
    story.append(P("4. Strip chart along chainage", st["h2"]))
    story.append(strip_chart(rep, pw))
    story.append(P("Band colours as in the route legend (two names = E_lo and E_hi fall in different bands; violet = "
                   "Good on the labels, Poor only if gate-skipped segments hold the distress). Coverage: darker = more "
                   "imaged. The last row is the bracketed Appendix-4 overall class in greys (darker = worse): secondary, never a "
                   "verdict.",
                   st["note"]))
    if L > 2000:
        story.append(PageBreak())
        story.append(P("4a. Route detail per km (ticks and labels every 100 m)", st["h2"]))
        kms = [(k["from_m"], k["to_m"], k["chainage_from"], k["chainage_to"]) for k in rep["km"]]
        for i, (a, b, ca, cb) in enumerate(kms):
            story.append(KeepTogether([route_figure(rep, pw, 112 * mm, a, b, title=f"{ca} to {cb}"),
                                       Spacer(1, 4)]))
        story.append(legend_table(st))
    story.append(NextPageTemplate("landscape"))
    story.append(PageBreak())
    story.append(P("5. Kilometre-wise distress record (IRC:82 Proforma 1 columns)", st["h2"]))
    story.append(P("Values are E_lo–E_hi: % of surveyed length with the distress visible (est.); an upper bound on % "
                   "area for localised distress. 'not assessed' is never 0 %.", st["note"]))
    story.append(_proforma_table(rep, st, lw))
    story.append(P("6. Rating per Appendix 4 (bracketed)", st["h2"]))
    story.append(KeepTogether(_rating_table(rep["stretch"], st, "Whole stretch", lw)))
    if len(rep["km"]) == 1:
        story.append(P("The stretch is a single (part-)km: its km row equals the whole stretch.", st["note"]))
    for u in _km_rows(rep):
        story.append(KeepTogether(_rating_table(u, st, f"Km row {u['chainage_from']} to {u['chainage_to']}"
                                                + (" (part-km)" if u.get("part_km") else ""), lw)))
    story.append(PageBreak())
    story.append(P("7. 100 m sub-sections", st["h2"]))
    story.append(_sections_table(rep, st, lw))
    story.append(NextPageTemplate("portrait"))
    story.append(PageBreak())
    story.append(P("8. Candidate treatments", st["h2"]))
    story.extend(_treatments_block(rep, st))
    story.append(PageBreak())
    story.append(P("9. Photo annex", st["h2"]))
    story.append(P(f"Up to 24 frames: pothole runs and sections whose band can be Poor ({len(rep['annex'])} shown). Each "
                   "frame shows the road AHEAD of the marked chainage (forward-looking camera).", st["note"]))
    story.extend(_annex_block(rep, st, image_fetcher))
    story.append(CondPageBreak(80 * mm))
    story.append(P("10. Method", st["h2"]))
    for line in rep["method"]:
        story.append(P("• " + escape(line), st["body"]))
    story.append(P("11. Limitations", st["h2"]))
    for line in rep["limitations"]:
        story.append(P("• " + escape(line), st["body"]))
    story.append(P("Accuracy: " + escape(rep["accuracy"]), st["body"]))
    story.append(CondPageBreak(40 * mm))
    story.append(P("12. Sign-off", st["h2"]))
    sign = [[P("Recorded by", st["cellb"]), P("", st["cell"]), P("Date", st["cellb"]), P("", st["cell"])],
            [P("Checked by", st["cellb"]), P("", st["cell"]), P("Date", st["cellb"]), P("", st["cell"])]]
    t = Table(sign, colWidths=[28 * mm, 80 * mm, 16 * mm, 50 * mm], rowHeights=[12 * mm, 12 * mm], hAlign="LEFT")
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, RULE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story.append(t)
    story.append(P("Left blank on purpose: the report is generated automatically; an engineer signs after a field check.",
                   st["note"]))
    doc.build(story)
    return buf.getvalue()
