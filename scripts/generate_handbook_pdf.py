"""
Generate the GeoAI Technical Handbook PDF.

Produces:
  - paper/diagrams/*.png   (all generated diagrams)
  - paper/GeoAI_Technical_Handbook.pdf

Run:
    venv/Scripts/python.exe scripts/generate_handbook_pdf.py

Dependencies: reportlab, matplotlib, Pillow, numpy.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer,
    PageBreak, Image, Table, TableStyle, KeepTogether, HRFlowable,
)
from reportlab.pdfgen import canvas
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY


# ---------- Paths ----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = PROJECT_ROOT / "paper"
DIAGRAMS_DIR = PAPER_DIR / "diagrams"
EVAL_DIR = PROJECT_ROOT / "eval_results"
CM_DIR = EVAL_DIR / "confusion_matrices"

DIAGRAMS_DIR.mkdir(parents=True, exist_ok=True)
PDF_PATH = PAPER_DIR / "GeoAI_Technical_Handbook.pdf"


# ---------- Load baseline numbers ------------------------------------------
def load_baseline():
    with open(EVAL_DIR / "baseline_results.json", "r") as f:
        return json.load(f)


BASELINE = load_baseline()


# ===========================================================================
# DIAGRAM GENERATION (matplotlib)
# ===========================================================================

def _save_fig(fig, name: str):
    path = DIAGRAMS_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _box(ax, x, y, w, h, text, *, fill="#ffffff", edge="#1c1917",
         fontsize=9, weight="normal", lw=1.2):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=lw, edgecolor=edge, facecolor=fill,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize, weight=weight,
            color="#1c1917", wrap=True)


def _arrow(ax, x1, y1, x2, y2, label=None, color="#1c1917", style="->",
           lw=1.4, label_offset=(0, 0.04)):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))
    if label:
        ax.text((x1 + x2) / 2 + label_offset[0],
                (y1 + y2) / 2 + label_offset[1],
                label, ha="center", va="center", fontsize=8,
                style="italic", color="#525252")


def diagram_system_architecture():
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Row 1: phone -> Cloudinary + Supabase.photos
    _box(ax, 0.2, 5.0, 1.7, 1.0, "Citizen Phone\n(RoadSide App)\nimage + GPS",
         fill="#f5f5f4", weight="bold")
    _box(ax, 2.6, 5.6, 1.7, 0.9, "Cloudinary\n(image CDN)", fill="#fef3c7")
    _box(ax, 2.6, 4.4, 1.7, 0.9, "Supabase\n  photos table", fill="#dbeafe")

    _arrow(ax, 1.9, 5.7, 2.6, 5.95, "JPEG upload")
    _arrow(ax, 1.9, 5.4, 2.6, 4.85, "metadata + URL")

    # Trigger
    _box(ax, 5.0, 4.4, 1.6, 0.9,
         "Postgres trigger\nphotos -> assessments\n(status='pending')",
         fill="#fef9c3", fontsize=8)
    _arrow(ax, 4.3, 4.85, 5.0, 4.85, "AFTER INSERT")

    # assessments
    _box(ax, 7.2, 4.4, 1.7, 0.9, "Supabase\n  assessments\n  (job queue)",
         fill="#dbeafe", weight="bold")
    _arrow(ax, 6.6, 4.85, 7.2, 4.85)

    # Row 2: worker
    _box(ax, 7.2, 2.7, 1.7, 0.9,
         "Operator Worker\n(asyncio loop)\nA5000 GPU",
         fill="#f5f5f4", weight="bold")
    _arrow(ax, 8.0, 4.4, 8.0, 3.6, "claim batch\n(SKIP LOCKED)",
           label_offset=(-0.55, 0))
    _arrow(ax, 8.05, 3.6, 8.05, 4.4, style="->", color="#525252",
           lw=1.0)
    ax.text(8.45, 4.0, "write back", fontsize=8, style="italic",
            color="#525252", rotation=90, va="center")

    # Two-stage model
    _box(ax, 4.6, 2.7, 1.9, 0.9,
         "Qwen2.5-VL-7B\nStage 1 -> Stage 2",
         fill="#e0e7ff", weight="bold")
    _arrow(ax, 7.2, 3.15, 6.5, 3.15, "image bytes")
    _arrow(ax, 6.5, 3.0, 7.2, 3.0, "label + conf",
           label_offset=(0, -0.15))

    # Cloudinary download arrow
    _arrow(ax, 3.45, 5.6, 5.55, 3.6, "image download",
           color="#525252", lw=1.0, style="->",
           label_offset=(-0.3, 0.1))

    # Operator dashboard
    _box(ax, 1.0, 2.7, 2.4, 0.9,
         "Operator Dashboard\nlive metrics + SSE",
         fill="#f5f5f4")
    _arrow(ax, 4.6, 3.15, 3.4, 3.15, "metrics", style="->")

    # Row 3: expert review and webgis
    _box(ax, 1.0, 1.0, 2.4, 0.9,
         "Expert UI\nreview low-confidence\nrows",
         fill="#fef3c7")
    _arrow(ax, 2.2, 2.7, 2.2, 1.9, "expert_review",
           label_offset=(0.7, 0))

    _box(ax, 4.6, 1.0, 1.9, 0.9,
         "Few-shot prompt\n+ LoRA retrain",
         fill="#fef3c7", fontsize=8)
    _arrow(ax, 3.4, 1.45, 4.6, 1.45, "corrections")
    _arrow(ax, 5.55, 1.9, 5.55, 2.7, "improved model",
           label_offset=(0.7, 0))

    _box(ax, 7.7, 1.0, 2.4, 0.9,
         "WebGIS Map\n(civic authorities)\ncolor-coded hotspots",
         fill="#dcfce7", weight="bold")
    _arrow(ax, 8.05, 2.7, 8.9, 1.9, "status='done'\nrows + lat/long",
           label_offset=(0.6, 0))

    ax.set_title("GeoAI System Architecture",
                 fontsize=13, weight="bold", pad=14)
    return _save_fig(fig, "system_architecture")


def diagram_two_stage_pipeline():
    fig, ax = plt.subplots(figsize=(10, 5.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    _box(ax, 0.3, 2.0, 1.5, 1.0, "Input image\n(phone photo)",
         fill="#f5f5f4", weight="bold")

    _box(ax, 2.5, 2.0, 1.8, 1.0,
         "Stage 1\nBinary detection\n(Normal / Distress)",
         fill="#e0e7ff", weight="bold")
    _arrow(ax, 1.8, 2.5, 2.5, 2.5)

    # Decision diamond (drawn as polygon)
    diamond = mpatches.Polygon(
        [(5.0, 2.5), (5.6, 3.0), (6.2, 2.5), (5.6, 2.0)],
        closed=True, facecolor="#fef9c3", edgecolor="#1c1917", lw=1.2)
    ax.add_patch(diamond)
    ax.text(5.6, 2.5, "Distressed?", ha="center", va="center", fontsize=8)
    _arrow(ax, 4.3, 2.5, 5.0, 2.5)

    _box(ax, 6.7, 2.0, 1.8, 1.0,
         "Stage 2\nType + severity\n(D00/D10/D20/D40 + ...)",
         fill="#e0e7ff", weight="bold")
    _arrow(ax, 6.2, 2.5, 6.7, 2.5, label="yes")

    _box(ax, 6.7, 0.5, 1.8, 0.8, "Return: Normal\n(skip Stage 2)",
         fill="#dcfce7")
    _arrow(ax, 5.6, 2.0, 7.2, 1.3, label="no",
           label_offset=(-0.3, 0.1))

    # Confidence threshold check
    _box(ax, 3.0, 0.3, 4.2, 1.0,
         "Confidence < 80%?\n  yes -> needs_expert_review = true\n"
         "  no  -> classified",
         fill="#fef9c3", fontsize=8)
    _arrow(ax, 7.6, 2.0, 5.6, 1.3, color="#525252")

    # Confidence extraction labels
    ax.text(3.4, 3.4, "first-token softmax\nover {Normal, Distress}",
            ha="center", fontsize=7, style="italic", color="#525252")
    ax.text(7.6, 3.4, "geometric mean of\nper-token probs",
            ha="center", fontsize=7, style="italic", color="#525252")

    ax.set_title("Two-Stage Inference Pipeline",
                 fontsize=13, weight="bold", pad=14)
    return _save_fig(fig, "two_stage_pipeline")


def diagram_status_state_machine():
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.5)
    ax.axis("off")

    states = [
        ("pending", 0.3, 2.0, "#dbeafe"),
        ("processing", 2.4, 2.0, "#fef9c3"),
        ("classified", 5.0, 3.0, "#dcfce7"),
        ("expert_review", 5.0, 1.0, "#fef3c7"),
        ("done", 8.0, 2.0, "#dcfce7"),
        ("failed", 2.4, 0.3, "#fee2e2"),
    ]
    for label, x, y, fill in states:
        _box(ax, x, y, 1.7, 0.7, label, fill=fill, weight="bold")

    # Transitions
    _arrow(ax, 2.0, 2.35, 2.4, 2.35, label="claim")
    _arrow(ax, 4.1, 2.5, 5.0, 3.2, label="conf>=80%",
           label_offset=(0, 0.05))
    _arrow(ax, 4.1, 2.2, 5.0, 1.4, label="conf<80%",
           label_offset=(0, -0.18))
    _arrow(ax, 6.7, 3.2, 8.0, 2.5, label="auto-finalize")
    _arrow(ax, 6.7, 1.4, 8.0, 2.2, label="expert reviewed")

    # Failure path
    _arrow(ax, 3.25, 2.0, 3.25, 1.0, label="3 retries",
           label_offset=(0.6, 0))
    # Recovery path
    _arrow(ax, 2.4, 1.85, 1.0, 2.0,
           label="stale > 5min\n(retry++)",
           label_offset=(0.4, -0.25), style="->")

    ax.set_title("assessments.status State Machine",
                 fontsize=13, weight="bold", pad=12)
    return _save_fig(fig, "status_state_machine")


def diagram_er_diagram():
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    def table(x, y, w, h, title, columns, fill="#ffffff"):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            linewidth=1.2, edgecolor="#1c1917", facecolor=fill)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h - 0.25, title, ha="center", va="top",
                fontsize=10, weight="bold", color="#1c1917")
        ax.plot([x + 0.05, x + w - 0.05],
                [y + h - 0.45, y + h - 0.45], color="#1c1917", lw=0.8)
        for i, col in enumerate(columns):
            ax.text(x + 0.1, y + h - 0.7 - i * 0.22, col,
                    fontsize=8, color="#1c1917", va="top")

    # photos (top-left)
    table(0.3, 4.3, 2.7, 2.0, "photos (RoadSide app)",
          ["id BIGINT PK",
           "image_url TEXT",
           "address TEXT",
           "latitude FLOAT8",
           "longitude FLOAT8",
           "created_at TIMESTAMPTZ"],
          fill="#dbeafe")

    # assessments (center)
    table(4.0, 2.5, 3.6, 3.8, "assessments (pipeline core)",
          ["id UUID PK",
           "photo_id BIGINT FK -> photos(id)",
           "image_url, lat, long, address",
           "status TEXT (state machine)",
           "claimed_at, claimed_by, retry_count",
           "stage1_label, stage1_confidence",
           "is_distressed BOOLEAN",
           "distress_types JSONB, severity, desc",
           "stage2_confidence REAL",
           "needs_expert_review BOOLEAN",
           "expert_corrected_types JSONB",
           "processed_at TIMESTAMPTZ"],
          fill="#fef9c3")

    # worker_state (top right)
    table(8.2, 4.3, 2.6, 2.0, "worker_state",
          ["worker_id TEXT PK",
           "is_running BOOLEAN",
           "current_image_id UUID FK",
           "images_processed INT",
           "avg_stage1_time_ms",
           "last_heartbeat TIMESTAMPTZ"],
          fill="#dcfce7")

    # retrain_jobs (bottom right)
    table(8.2, 2.0, 2.6, 1.8, "retrain_jobs",
          ["id UUID PK",
           "status TEXT",
           "corrections_count INT",
           "progress_pct, current_step",
           "adapter_path TEXT"],
          fill="#fef3c7")

    # custom_distress_types (bottom-left)
    table(0.3, 2.0, 2.7, 1.8, "custom_distress_types",
          ["id UUID PK",
           "name TEXT UNIQUE",
           "description TEXT",
           "added_by UUID FK auth.users"],
          fill="#fef3c7")

    # FK arrows
    _arrow(ax, 3.0, 5.3, 4.0, 4.5, label="trigger\nphoto_id FK",
           label_offset=(0.0, 0.1), color="#dc2626")
    _arrow(ax, 7.6, 4.3, 8.2, 4.7, label="current_image_id",
           label_offset=(0.0, 0.1))

    ax.set_title("Database ER Diagram (key columns + foreign keys)",
                 fontsize=13, weight="bold", pad=12)
    return _save_fig(fig, "er_diagram")


def diagram_async_worker_flow():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    # Main loop boxes (left to right)
    _box(ax, 0.3, 2.4, 1.5, 0.9,
         "Poll loop\nevery 10s", fill="#f5f5f4", weight="bold")
    _box(ax, 2.2, 2.4, 1.7, 0.9,
         "Claim batch (RPC)\nFOR UPDATE\nSKIP LOCKED",
         fill="#dbeafe", fontsize=8)
    _box(ax, 4.3, 2.4, 1.7, 0.9, "Download image\n(httpx, max 30s)",
         fill="#fef9c3", fontsize=8)
    _box(ax, 6.4, 2.4, 1.7, 0.9, "Stage 1 inference\n(asyncio.to_thread)",
         fill="#e0e7ff", fontsize=8)
    _box(ax, 8.5, 2.4, 1.7, 0.9, "Stage 2 inference\nif distressed",
         fill="#e0e7ff", fontsize=8)

    _arrow(ax, 1.8, 2.85, 2.2, 2.85)
    _arrow(ax, 3.9, 2.85, 4.3, 2.85)
    _arrow(ax, 6.0, 2.85, 6.4, 2.85)
    _arrow(ax, 8.1, 2.85, 8.5, 2.85)

    # Loop back
    _arrow(ax, 9.35, 2.4, 9.35, 1.5, color="#525252")
    _arrow(ax, 9.35, 1.5, 1.05, 1.5, color="#525252")
    _arrow(ax, 1.05, 1.5, 1.05, 2.4, color="#525252",
           label="next poll", label_offset=(0.5, 0))

    # Write-back box
    _box(ax, 4.3, 0.4, 5.7, 0.7,
         "Write back: status=classified|expert_review|failed,  "
         "stage1/stage2 fields, processed_at",
         fill="#dcfce7", fontsize=8)
    _arrow(ax, 9.35, 2.4, 9.0, 1.1, color="#525252")

    # Top side flows
    _box(ax, 0.3, 4.4, 2.0, 0.7, "On startup:\nreset_stale_processing",
         fill="#fee2e2", fontsize=8)
    _arrow(ax, 1.05, 4.4, 1.05, 3.3, color="#525252")

    _box(ax, 6.0, 4.4, 2.5, 0.7,
         "Heartbeat task (every 30s)\nupserts worker_state",
         fill="#fef3c7", fontsize=8)

    _box(ax, 8.7, 4.4, 1.8, 0.7,
         "Stop signal -> drain\nin-flight image",
         fill="#fee2e2", fontsize=8)

    # Annotation labels
    ax.text(5.5, 3.55, "Per-row try/except: TRANSIENT errors -> "
                       "exponential backoff retry up to 3x",
            ha="center", fontsize=8, style="italic", color="#525252")

    ax.set_title("Operator Worker — Async Loop",
                 fontsize=13, weight="bold", pad=12)
    return _save_fig(fig, "async_worker_flow")


def diagram_confusion_matrix_stage1():
    cm = np.array(BASELINE["stage1"]["confusion_matrix"])
    labels = ["Normal", "Distress"]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "#1c1917",
                    fontsize=12, weight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Stage 1 Baseline Confusion Matrix\n(GAPs V2 test, 10,000 images)",
                 fontsize=11)
    return _save_fig(fig, "stage1_cm")


def diagram_confusion_matrix_stage2():
    cm = np.array(BASELINE["stage2"]["confusion_matrix"])
    labels = ["Unparse", "D00", "D10", "D20", "D40", "Other"]
    # Drop the empty last row (Other has 0 support)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cm[i, j]
            ax.text(j, i, str(v) if v > 0 else "",
                    ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "#1c1917",
                    fontsize=10, weight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Stage 2 Baseline Confusion Matrix\n(RDD2022 test, 5,758 images)",
                 fontsize=11)
    return _save_fig(fig, "stage2_cm")


# ===========================================================================
# PDF DOCUMENT
# ===========================================================================

# Color palette (matches expert_ui style)
TEXT_COLOR = colors.HexColor("#1c1917")
ACCENT = colors.HexColor("#1c1917")
MUTED = colors.HexColor("#525252")
BORDER = colors.HexColor("#e7e5e4")
SURFACE = colors.HexColor("#f5f5f4")
RED = colors.HexColor("#dc2626")
AMBER = colors.HexColor("#d97706")
GREEN = colors.HexColor("#16a34a")
BLUE = colors.HexColor("#1d4ed8")


# ---------- Header / footer canvas -----------------------------------------
# We use the standard "two-pass" Platypus pattern: first build collects states,
# then save() replays each state as a real page (single canvas, single output).
class HandbookCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_states = []

    def showPage(self):
        # Snapshot the per-page state and start a fresh dict (do NOT call
        # the parent showPage here — that would render the page twice).
        self._saved_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        page_count = len(self._saved_states)
        for i, state in enumerate(self._saved_states):
            self.__dict__.update(state)
            self.draw_header_footer(i + 1, page_count)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_header_footer(self, page_num, total_pages):
        # Skip header on cover page (page 1)
        if page_num > 1:
            self.setFont("Helvetica", 8)
            self.setFillColor(MUTED)
            self.drawString(2 * cm, A4[1] - 1.2 * cm,
                            "GeoAI Technical Handbook")
            chapter = getattr(self, "_chapter", "")
            if chapter:
                self.drawRightString(A4[0] - 2 * cm, A4[1] - 1.2 * cm, chapter)
            self.setStrokeColor(BORDER)
            self.setLineWidth(0.4)
            self.line(2 * cm, A4[1] - 1.4 * cm,
                      A4[0] - 2 * cm, A4[1] - 1.4 * cm)
        # Footer (skip on cover)
        if page_num > 1:
            self.setFont("Helvetica", 8)
            self.setFillColor(MUTED)
            self.drawCentredString(A4[0] / 2, 1.2 * cm,
                                   f"Page {page_num} of {total_pages}")


# Track current chapter via a flowable side-effect
_CHAPTER_STATE = {"name": ""}


class _ChapterMarker(Spacer):
    def __init__(self, chapter_name):
        Spacer.__init__(self, 1, 0.001)
        self.chapter_name = chapter_name

    def draw(self):
        # When this flowable draws, it is on a particular page.
        # Stash the chapter name onto the canvas via a hook.
        canv = self.canv
        canv._chapter = self.chapter_name


# ---------- Styles ---------------------------------------------------------
def make_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="CoverTitle", parent=styles["Title"],
        fontName="Helvetica-Bold", fontSize=32, textColor=TEXT_COLOR,
        alignment=TA_CENTER, spaceAfter=14, leading=36))
    styles.add(ParagraphStyle(
        name="CoverSubtitle", parent=styles["Title"],
        fontName="Helvetica", fontSize=18, textColor=MUTED,
        alignment=TA_CENTER, spaceAfter=20))
    styles.add(ParagraphStyle(
        name="CoverMeta", parent=styles["Normal"],
        fontName="Helvetica", fontSize=12, textColor=TEXT_COLOR,
        alignment=TA_CENTER, leading=18))

    styles.add(ParagraphStyle(
        name="ChapterTitle", parent=styles["Heading1"],
        fontName="Helvetica-Bold", fontSize=20, textColor=TEXT_COLOR,
        spaceBefore=0, spaceAfter=10, leading=24))
    styles.add(ParagraphStyle(
        name="SectionTitle", parent=styles["Heading2"],
        fontName="Helvetica-Bold", fontSize=13, textColor=TEXT_COLOR,
        spaceBefore=12, spaceAfter=5, leading=16))
    styles.add(ParagraphStyle(
        name="SubsectionTitle", parent=styles["Heading3"],
        fontName="Helvetica-Bold", fontSize=11, textColor=TEXT_COLOR,
        spaceBefore=8, spaceAfter=3, leading=14))

    styles.add(ParagraphStyle(
        name="Body", parent=styles["BodyText"],
        fontName="Times-Roman", fontSize=10.5, textColor=TEXT_COLOR,
        leading=14, alignment=TA_JUSTIFY, spaceAfter=6))
    styles.add(ParagraphStyle(
        name="BodyTight", parent=styles["BodyText"],
        fontName="Times-Roman", fontSize=10, textColor=TEXT_COLOR,
        leading=13, alignment=TA_JUSTIFY, spaceAfter=4))
    styles.add(ParagraphStyle(
        name="HBullet", parent=styles["BodyText"],
        fontName="Times-Roman", fontSize=10.5, textColor=TEXT_COLOR,
        leading=14, leftIndent=16, bulletIndent=4, spaceAfter=3,
        alignment=TA_LEFT))
    styles.add(ParagraphStyle(
        name="Caption", parent=styles["BodyText"],
        fontName="Helvetica-Oblique", fontSize=9, textColor=MUTED,
        alignment=TA_CENTER, spaceBefore=4, spaceAfter=10))
    styles.add(ParagraphStyle(
        name="HCode", parent=styles["BodyText"],
        fontName="Courier", fontSize=8.5, textColor=TEXT_COLOR,
        leading=11, leftIndent=12, rightIndent=12, spaceBefore=4,
        spaceAfter=8, backColor=SURFACE, borderColor=BORDER,
        borderWidth=0.5, borderPadding=6))
    styles.add(ParagraphStyle(
        name="Callout", parent=styles["BodyText"],
        fontName="Times-Roman", fontSize=10, textColor=TEXT_COLOR,
        leading=13, leftIndent=10, rightIndent=10, spaceBefore=6,
        spaceAfter=8, backColor=colors.HexColor("#fef9c3"),
        borderColor=colors.HexColor("#facc15"),
        borderWidth=0.6, borderPadding=8, alignment=TA_JUSTIFY))

    return styles


STYLES = make_styles()


# ---------- Helper builders ------------------------------------------------
def P(text, style="Body"):
    return Paragraph(text, STYLES[style])


def chapter(name, *flowables):
    out = [PageBreak(), _ChapterMarker(name),
           Paragraph(name, STYLES["ChapterTitle"]),
           HRFlowable(width="100%", thickness=0.6, color=BORDER,
                      spaceAfter=12)]
    out.extend(flowables)
    return out


def section(text):
    return Paragraph(text, STYLES["SectionTitle"])


def subsection(text):
    return Paragraph(text, STYLES["SubsectionTitle"])


def bullet(text):
    return Paragraph(text, STYLES["HBullet"], bulletText="•")


def code(text):
    text = text.replace("\n", "<br/>").replace(" ", "&nbsp;")
    return Paragraph(text, STYLES["HCode"])


def callout(text):
    return Paragraph(text, STYLES["Callout"])


def caption(text):
    return Paragraph(text, STYLES["Caption"])


def fig(path, width_cm=15.5, cap=None):
    """Embed an image scaled proportionally based on its real pixel dims."""
    from PIL import Image as PILImage
    with PILImage.open(str(path)) as pim:
        iw, ih = pim.size
    target_w = width_cm * cm
    target_h = target_w * (ih / iw)
    img = Image(str(path), width=target_w, height=target_h)
    items = [img]
    if cap:
        items.append(caption(cap))
    return KeepTogether(items)


def table(data, *, col_widths=None, header=True, fontSize=9, padding=4):
    style_cmds = [
        ("FONT", (0, 0), (-1, -1), "Helvetica", fontSize),
        ("TEXTCOLOR", (0, 0), (-1, -1), TEXT_COLOR),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), padding),
        ("RIGHTPADDING", (0, 0), (-1, -1), padding),
        ("TOPPADDING", (0, 0), (-1, -1), padding),
        ("BOTTOMPADDING", (0, 0), (-1, -1), padding),
        ("LINEBELOW", (0, 0), (-1, -1), 0.3, BORDER),
        ("LINEABOVE", (0, 0), (-1, 0), 0.6, ACCENT),
    ]
    if header:
        style_cmds.extend([
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", fontSize),
            ("BACKGROUND", (0, 0), (-1, 0), SURFACE),
            ("LINEBELOW", (0, 0), (-1, 0), 0.6, ACCENT),
        ])
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle(style_cmds))
    return t


# ===========================================================================
# CONTENT — Chapters
# ===========================================================================

def cover_page():
    spacer = Spacer(1, 4 * cm)
    title = Paragraph("GeoAI", STYLES["CoverTitle"])
    subtitle = Paragraph(
        "Pavement Distress Detection System<br/>"
        "<font size='14'>Technical Handbook</font>",
        STYLES["CoverSubtitle"])

    rule = HRFlowable(width="50%", thickness=0.8, color=BORDER,
                      spaceBefore=8, spaceAfter=14, hAlign="CENTER")
    meta = Paragraph(
        "<b>Authors</b><br/>"
        "Suraj &middot; Richik Chaudhuri &middot; Sushant Deo<br/><br/>"
        "<b>Date</b><br/>"
        "April 2026<br/><br/>"
        "<b>Institution</b><br/>"
        "Capstone Project, Bengaluru, India<br/><br/>"
        "<font size='10' color='#525252'>Document version 1.0 &middot; "
        "Auto-generated by scripts/generate_handbook_pdf.py</font>",
        STYLES["CoverMeta"])

    return [spacer, title, subtitle, Spacer(1, 0.6 * cm), rule, meta]


def executive_summary():
    return chapter(
        "1. Executive Summary",
        P(
            "GeoAI is a system that detects damage on roads from ordinary smartphone "
            "photos. A citizen opens an app, takes a picture of a road, and within a "
            "few seconds the system says whether the road is damaged, what kind of "
            "damage it is (crack, pothole, etc.), how severe it looks, and how confident "
            "it is in that answer. Photos with low confidence are routed to a human "
            "expert for review. Over time, the model learns from those corrections."
        ),
        P(
            "The pipeline is built around a Vision-Language Model (Qwen2.5-VL-7B) "
            "fine-tuned on two public road-damage datasets: GAPs V2 (binary distress "
            "detection) and RDD2022 (distress type classification). It runs on a single "
            "NVIDIA RTX A5000 GPU at the college lab, exposes a FastAPI server, stores "
            "every classification in a Supabase Postgres database, and surfaces the "
            "results to civic authorities through a WebGIS map."
        ),
        section("What is built today"),
        bullet("A two-stage AI pipeline (Stage 1 binary, Stage 2 multi-class) with confidence scores extracted from raw model logits, not from string matching."),
        bullet("A FastAPI server with three classification endpoints (standard, base64, and live SSE streaming) plus full operator endpoints."),
        bullet("A Supabase schema with a job-queue pattern: the RoadSide app inserts a photo, a Postgres trigger creates a pending row, the worker atomically claims a batch and writes back results."),
        bullet("An autonomous asyncio worker that polls the queue, downloads images from Cloudinary, runs the model, and writes back per-stage timings and confidences. Heartbeats every 30 seconds. Recovers from worker crashes via stale-claim reset."),
        bullet("An operator dashboard with live SSE metrics, a per-stage progress visualisation, hover tooltips, and a help panel."),
        bullet("A 5-test robustness suite covering 404 image, corrupt JPEG, network timeout, mid-process stop, and concurrent claim. All five pass."),
        bullet("A Phase 1 baseline evaluation with real numbers (full 10,000 GAPs and 5,758 RDD images) saved to eval_results/baseline_results.json with confusion matrices."),
        section("What is the headline result so far"),
        P(
            "On the raw (un-fine-tuned) Qwen2.5-VL-7B model, Stage 1 binary detection scored "
            "<b>76.5% accuracy</b> with a strong Normal-bias (Distress recall is only 42.9%). "
            "Stage 2 type classification scored <b>48.7% primary accuracy</b>, but only because "
            "the model defaults to D00 longitudinal cracks when in doubt. Recall on the other "
            "three classes (D10 transverse, D20 alligator, D40 pothole) is between 0% and 2.6%. "
            "These numbers establish the baseline floor; Phase 2 fine-tuning is expected to "
            "lift them substantially, and the gap is the headline story of the paper."
        ),
        section("What is left"),
        bullet("Phase 2: QLoRA fine-tuning on RDD + GAPs combined training set; cross-dataset zero-shot evaluation on the Attain dataset (10 classes, 7 are real distress)."),
        bullet("Phase 3: expert-in-the-loop few-shot prompt injection plus LoRA incremental retraining, with versioned adapters for rollback."),
        bullet("Phase 4: WebGIS map for civic authorities with color-coded distress hotspots."),
    )


def system_architecture_chapter():
    return chapter(
        "2. System Architecture",
        P(
            "GeoAI is composed of seven independent components that communicate "
            "through a Postgres database. Each component does one thing and only one thing. "
            "This makes it possible to develop, test, and replace each piece "
            "without breaking the rest of the system."
        ),
        fig(DIAGRAMS_DIR / "system_architecture.png", width_cm=16.5,
            cap="Figure 2.1 — End-to-end data flow from citizen phone to civic authority dashboard."),
        section("Component-by-component walkthrough"),
        subsection("1. RoadSide mobile app (citizen-facing)"),
        P(
            "An Android/iOS app that captures a photo with GPS coordinates and uploads it. "
            "The image goes to <b>Cloudinary</b> (a content delivery network for images). "
            "The metadata (image URL, latitude, longitude, address, timestamp) goes to the "
            "<b>photos</b> table in Supabase. The app does not know that the AI pipeline "
            "exists; it only writes to <code>photos</code>."
        ),
        subsection("2. Cloudinary"),
        P(
            "Stores the actual JPEG bytes. The free tier is enough for the project. "
            "Cloudinary returns a public URL that the worker can later download from. "
            "Project ID: <code>dnxpt5gea</code>."
        ),
        subsection("3. Supabase Postgres database"),
        P(
            "The single source of truth. Three roles: queue (assessments table), worker "
            "state (worker_state table), and reference data (custom_distress_types, "
            "retrain_jobs). Row-Level Security (RLS) is enabled — only the service role "
            "key (held by the worker and dashboard) can write to <code>assessments</code>."
        ),
        subsection("4. The photos&rarr;assessments trigger"),
        P(
            "An <code>AFTER INSERT</code> trigger on the photos table. Whenever the "
            "RoadSide app inserts a new photo, the trigger automatically inserts a "
            "matching <code>assessments</code> row with <code>status = 'pending'</code>, "
            "copying the image URL, GPS coordinates, and timestamp. The mobile app team "
            "never has to know about the pipeline. The trigger is idempotent: if a row "
            "for that <code>photo_id</code> already exists, nothing happens."
        ),
        subsection("5. The operator worker"),
        P(
            "An asyncio loop running inside the FastAPI process on the A5000 machine. "
            "Polls the assessments table every 10 seconds, atomically claims up to 5 "
            "pending rows (using <code>FOR UPDATE SKIP LOCKED</code>), downloads each "
            "image from Cloudinary, runs Stage 1 then Stage 2, and writes back the "
            "results. See Chapter 9 for full details."
        ),
        subsection("6. The two-stage AI model"),
        P(
            "Qwen2.5-VL-7B-Instruct loaded once at startup. Stage 1 (binary) decides "
            "Normal vs Distress. If Distress, Stage 2 (multi-class) names the type, "
            "estimates severity, and writes a description. Both stages produce a real "
            "confidence value extracted from raw logits. See Chapter 4."
        ),
        subsection("7. Operator dashboard + Expert UI + WebGIS"),
        P(
            "Three browser-based clients that read from Supabase. The operator dashboard "
            "watches the worker. The expert UI reviews low-confidence rows and submits "
            "corrections. The WebGIS map (planned) shows hotspots on a Leaflet map for "
            "civic authorities."
        ),
        section("Why this shape"),
        P(
            "Database-as-message-queue is a well-known pattern (Postgres has had "
            "<code>FOR UPDATE SKIP LOCKED</code> since 9.5). Using it here means we "
            "do not need a dedicated message broker like RabbitMQ or Redis. It also "
            "means that every state transition is durably persisted: a worker crash "
            "loses no work, only the time spent on the crashed image."
        ),
    )


def model_chapter():
    return chapter(
        "3. The AI Model",
        section("Why Qwen2.5-VL-7B"),
        P(
            "We needed a model that (a) can take an image and produce structured text, "
            "(b) fits on a single 24GB GPU, (c) supports fine-tuning with low-rank "
            "adapters, and (d) handles a wide range of input resolutions. Qwen2.5-VL-7B "
            "from Alibaba ticks all four boxes. It has a Vision Transformer encoder, "
            "a 7-billion-parameter language model decoder, dynamic resolution support "
            "(images from 7,168 to 4.84 million pixels), and an instruction-tuned chat "
            "format."
        ),
        P(
            "Other models we considered: YOLOv8 (only does object detection, no description "
            "and no zero-shot for unseen classes); CLIP (only does similarity, no generation); "
            "Qwen-VL-13B (does not fit on the A5000 budget at fp16); Qwen-VL-3B (fits the laptop "
            "but is much weaker)."
        ),
        section("Two-stage design"),
        fig(DIAGRAMS_DIR / "two_stage_pipeline.png", width_cm=15.5,
            cap="Figure 3.1 — Stage 1 makes a binary decision, Stage 2 only runs if Stage 1 says Distress."),
        P(
            "Calling a 7B VLM is expensive (around 5 seconds per image for Stage 2 on an "
            "A5000). Most photos of public roads show normal pavement. By making the first "
            "stage a cheap binary check, we skip Stage 2 entirely on roughly 60% of images "
            "in the wild, which roughly halves the average inference cost."
        ),
        P(
            "Each stage produces its own confidence score. This matters: a high-confidence "
            "Distress detection followed by a low-confidence type classification correctly "
            "triggers expert review for the type only — we don't need a human to confirm "
            "that the road is damaged, only what kind of damage it is."
        ),
        section("Confidence extraction (the cookie-point)"),
        callout(
            "Most VLM systems fake confidence by doing string matching on the response "
            "(e.g. 'if the model says \"definitely\", call it 90%'). We do the real thing: "
            "we read the model's raw logits at generation time and compute calibrated "
            "probabilities from them. This is the difference between a research-grade "
            "system and a demo."
        ),
        subsection("Stage 1: binary softmax over class tokens"),
        P(
            "At model load time we cache the token IDs for the words \"Normal\" and \"Distress\". "
            "When Stage 1 generates its first token, we read the full logit vector "
            "(<i>vocab_size</i> long), pick out the two logits corresponding to those "
            "two words, apply <code>softmax</code> over only those two, and report "
            "<code>P(Normal)</code> or <code>P(Distress)</code> depending on which one "
            "the model produced. The result is a probability in [0, 1] that "
            "directly represents the model's certainty between the two classes."
        ),
        subsection("Stage 2: geometric mean of per-token probabilities"),
        P(
            "Stage 2's output is a multi-line structured response (DISTRESS_TYPES, "
            "SEVERITY, DESCRIPTION). For each generated token <i>t<sub>i</sub></i> we "
            "compute its conditional log-probability "
            "<code>log P(t<sub>i</sub> | t<sub>1</sub>...t<sub>i-1</sub>)</code>. We "
            "average all log-probabilities and exponentiate: confidence = "
            "exp(mean(log p<sub>i</sub>)). This is the geometric mean of per-token "
            "probabilities. A response where the model is highly certain about every "
            "token yields a confidence near 1.0; uncertainty on even a few tokens pulls "
            "the geometric mean down sharply."
        ),
        section("Failure-safe parsing"),
        P(
            "If Stage 1 outputs a word that is neither Normal nor Distress (e.g. the "
            "model rambles or switches language), we default to <b>Distress with 0% "
            "confidence</b>. The 0% guarantees the row is flagged for expert review. "
            "We err on the side of caution: a damaged road silently classified as "
            "Normal is a safety hazard, while a normal road sent for review only wastes "
            "human time."
        ),
        P(
            "If Stage 2's structured output is unparseable, a fallback keyword "
            "extractor scans the raw text for any known distress-type name. If even "
            "that fails, the result is labeled \"Unknown\" with confidence below the "
            "threshold, which automatically triggers expert review."
        ),
        section("Why the model can recognize distress types it was never trained on"),
        P(
            "The Stage 2 system prompt includes a <b>taxonomy</b>: a list of 20+ "
            "distress types with short descriptions (longitudinal crack, transverse "
            "crack, alligator crack, pothole, block crack, raveling, weathering, etc.). "
            "Even though the model is fine-tuned only on RDD's 4 classes, the "
            "taxonomy text gives it a vocabulary, and Qwen2.5-VL's pre-trained "
            "visual-semantic alignment lets it match image features against the "
            "textual descriptions of unseen classes. This is what enables zero-shot "
            "generalization to the Attain dataset."
        ),
    )


def datasets_chapter():
    s1_table = [
        ["Property", "Value"],
        ["Source", "German Asphalt Pavement (GAPs) v2"],
        ["Format", ".npy chunks, (N,1,160,160) float32"],
        ["Image size", "160 x 160 grayscale"],
        ["Train / Valid / Test", "50,000 / 10,000 / 10,000"],
        ["Class split", "~60% Normal, ~40% Distress"],
        ["Use", "Stage 1 binary training + evaluation"],
    ]
    s2_table = [
        ["Property", "Value"],
        ["Source", "Road Damage Dataset 2022 (IEEE Big Data)"],
        ["Format", "JPEG + YOLO .txt labels"],
        ["Image size", "~512 x 512 RGB"],
        ["Train / Valid / Test", "26,869 / 5,758 / 5,758"],
        ["Classes", "D00, D10, D20, D40 (4 types)"],
        ["Geographies", "Japan, India, Czech, Norway, US, China"],
        ["Use", "Stage 2 type training + in-distribution eval"],
    ]
    attain_table = [
        ["Property", "Value"],
        ["Source", "Mendeley Data, doi:10.17632/nykrzdm74f/1"],
        ["License", "CC BY 4.0"],
        ["Format", "JPEG + label files (10 classes)"],
        ["Images", "2,293 (with 19,761 instances)"],
        ["Severity labels", "Yes (Low / Medium / High)"],
        ["Geography", "New Zealand roads"],
        ["Use", "Cross-dataset zero-shot evaluation only"],
    ]
    taxonomy = [
        ["Attain class", "Pipeline taxonomy", "In RDD train?", "Eval role"],
        ["Alligator Crack", "Alligator Crack (D20)", "Yes", "in-distribution"],
        ["Longitudinal Crack", "Longitudinal Crack (D00)", "Yes", "in-distribution"],
        ["Transverse Crack", "Transverse Crack (D10)", "Yes", "in-distribution"],
        ["Pothole", "Pothole (D40)", "Yes", "in-distribution"],
        ["Block Crack", "Block Crack (D43)", "No", "zero-shot"],
        ["Patch / Utility cut", "Inlaid Patch / Utility Cut", "No", "zero-shot"],
        ["Weathering", "Weathering / Oxidation", "No", "zero-shot"],
        ["Raveling", "Raveling", "No", "zero-shot"],
        ["Faded marking", "(road feature)", "-", "EXCLUDED"],
        ["Lane drop-off", "(road feature)", "-", "EXCLUDED"],
        ["Manhole", "(road feature)", "-", "EXCLUDED"],
    ]
    return chapter(
        "4. Datasets",
        section("4.1 GAPs V2 — Stage 1 (binary)"),
        table(s1_table, col_widths=[5 * cm, 11 * cm]),
        Spacer(1, 4),
        P(
            "GAPs is a German asphalt-pavement dataset captured by an automated imaging "
            "vehicle. Each image is a 160 x 160 grayscale tile with a binary label. We "
            "convert the .npy chunks to RGB PNGs in parallel (using all CPU cores) so "
            "that LLaMA-Factory and the model processor can read them directly."
        ),
        section("4.2 RDD2022 — Stage 2 (multi-class)"),
        table(s2_table, col_widths=[5 * cm, 11 * cm]),
        Spacer(1, 4),
        P(
            "RDD2022 has bounding-box labels in YOLO format, but for VLM training we "
            "need image-level labels. We collapse all unique class IDs in each image's "
            "label file into a multi-label set (e.g. \"D00 + D40\") and synthesise a "
            "severity heuristic: pothole present or 5+ annotations -> High; 2+ types or "
            "3+ annotations -> Medium; otherwise Low. The training data is rendered into "
            "ShareGPT JSON for LLaMA-Factory."
        ),
        section("4.3 Attain — cross-dataset evaluation only"),
        table(attain_table, col_widths=[5 * cm, 11 * cm]),
        Spacer(1, 4),
        P(
            "Attain is used <b>only</b> for evaluation, never for training. It comes from "
            "New Zealand (Southern Hemisphere, different climate and construction) and "
            "has 10 classes with per-instance severity labels. Of those 10, 4 overlap "
            "with our RDD training set (the model has seen them), 3 are new types we want "
            "to test zero-shot generalization on, and 3 are road features (manhole, drop-off, "
            "faded marking) that we exclude because they are not pavement structural damage."
        ),
        subsection("Class taxonomy mapping"),
        table(taxonomy, col_widths=[4 * cm, 5 * cm, 2.5 * cm, 4 * cm], fontSize=8.5),
        Spacer(1, 4),
        P(
            "This three-way split (in-distribution / zero-shot / excluded) is the structural "
            "basis of three comparison tables in the paper. The accuracy gap between the "
            "in-distribution classes and the zero-shot classes quantifies the value of "
            "fine-tuning over taxonomy injection alone."
        ),
    )


def finetuning_chapter():
    hp_table = [
        ["Hyperparameter", "Value", "Rationale"],
        ["LoRA rank", "64", "captures domain-specific patterns"],
        ["LoRA alpha", "128", "alpha = 2 x rank, standard scaling"],
        ["LoRA dropout", "0.1", "sparsity regularizer (anti-overfit)"],
        ["LoRA target", "all linear layers", "broad adapter coverage"],
        ["Quantization", "4-bit NF4 + double quant", "minimum VRAM footprint"],
        ["Compute dtype", "bfloat16", "avoids fp16 overflow in attention"],
        ["Learning rate", "1e-4", "standard QLoRA LR for 7B"],
        ["LR schedule", "cosine + 10% warmup", "smooth decay, no forgetting"],
        ["Epochs (max)", "2", "50k+ datasets overfit at 3+ epochs"],
        ["Batch size", "4 x 8 grad accum = 32 effective", "stable gradient estimates"],
        ["Weight decay", "0.05", "L2 regularization"],
        ["Label smoothing", "0.1", "no overconfident logits"],
        ["NEFTune noise", "alpha = 5.0", "noisy embedding fine-tuning"],
        ["Optimizer", "AdamW (8-bit)", "memory-efficient"],
        ["Gradient checkpointing", "On", "trades compute for VRAM"],
        ["Vision tower", "Frozen", "fine-tune LM only"],
        ["Template", "qwen2_5_vl", "MUST match (NOT qwen2_vl)"],
        ["Max seq length", "2048 tokens", "enough for image + structured text"],
    ]
    return chapter(
        "5. Fine-tuning Strategy (QLoRA)",
        section("What QLoRA actually does"),
        P(
            "Fine-tuning a 7-billion-parameter model the normal way needs ~28GB just to "
            "store the weights at fp16, plus optimizer state, plus gradients. That is far "
            "more than the A5000's 24GB. <b>QLoRA</b> (Quantized Low-Rank Adaptation) solves "
            "this with two tricks:"
        ),
        bullet("<b>Freeze and quantize the base model to 4-bit NF4.</b> The 7B model now occupies about 4-5 GB of VRAM. Its weights are read-only — we never update them."),
        bullet("<b>Add tiny low-rank adapter matrices (LoRA) on top of every linear layer.</b> Each adapter is a pair of small matrices (rank 64). We only train the adapters. Total trainable parameters: a few hundred million instead of 7 billion."),
        P(
            "The result: training fits in ~16-20GB of VRAM and runs in 12-24 hours. The "
            "trained \"adapter\" is a single ~600MB file we can save, version, and ship "
            "separately from the base model."
        ),
        section("Hyperparameters"),
        table(hp_table, col_widths=[4 * cm, 5 * cm, 7 * cm], fontSize=8.5),
        Spacer(1, 4),
        section("The five-layer anti-overfit stack"),
        P("Five independent regularisation mechanisms work together so the adapter does not just memorise the training set:"),
        bullet("<b>Weight decay 0.05</b> — L2 penalty on the adapter parameter magnitudes."),
        bullet("<b>Label smoothing 0.1</b> — softens the target distribution, prevents the model from being 100% certain on training data."),
        bullet("<b>LoRA dropout 0.1</b> — randomly zeroes 10% of adapter activations during training."),
        bullet("<b>NEFTune noise alpha=5.0</b> — adds calibrated noise to embedding vectors during forward pass; published to improve generalization."),
        bullet("<b>Early stopping on eval_loss</b> — monitors validation loss every 500 steps and rolls back to the best checkpoint."),
        section("The trap we avoided: peft issue #2586"),
        callout(
            "After QLoRA training, the natural next step is to call <code>merge_and_unload()</code> "
            "and ship a single merged checkpoint. <b>This is broken on quantized models</b> "
            "(see github.com/huggingface/peft/issues/2586): merging 4-bit weights with LoRA "
            "produces corrupted output. We keep the model as a <code>PeftModel</code> for "
            "inference, with the adapter loaded on top of the frozen quantized base. Slower "
            "to load, but mathematically correct."
        ),
        section("Acceleration stack"),
        bullet("<b>Flash Attention 2</b> — fused attention kernel, O(N) memory instead of O(N^2). Auto-detected at startup based on GPU compute capability."),
        bullet("<b>SDPA</b> — PyTorch's native scaled-dot-product attention as fallback if Flash Attention 2 is not available."),
        bullet("<b>Liger Kernel</b> — fused MLP and RMS-norm kernels (Linux-only because of Triton)."),
        bullet("<b>Gradient checkpointing</b> — activations are recomputed during the backward pass instead of stored, trading compute for VRAM."),
    )


def baseline_chapter():
    s1 = BASELINE["stage1"]["classification_report"]
    cm1 = BASELINE["stage1"]["confusion_matrix"]
    s1_summary = [
        ["Metric", "Value"],
        ["Overall accuracy", f"{BASELINE['stage1']['accuracy']*100:.2f}%"],
        ["Precision (macro)", f"{BASELINE['stage1']['precision_macro']*100:.2f}%"],
        ["Recall (macro)", f"{BASELINE['stage1']['recall_macro']*100:.2f}%"],
        ["F1 (macro)", f"{BASELINE['stage1']['f1_macro']*100:.2f}%"],
        ["Unparseable responses", f"{BASELINE['stage1']['unparseable_responses']} / {BASELINE['stage1']['total_samples']:,}"],
        ["Avg inference time", f"{BASELINE['stage1']['avg_inference_time_sec']:.2f} s/image"],
    ]
    s1_perclass = [
        ["Class", "Precision", "Recall", "F1", "Support"],
        ["Normal",
         f"{s1['Normal']['precision']*100:.2f}%",
         f"{s1['Normal']['recall']*100:.2f}%",
         f"{s1['Normal']['f1-score']*100:.2f}%",
         f"{int(s1['Normal']['support']):,}"],
        ["Distress",
         f"{s1['Distress']['precision']*100:.2f}%",
         f"{s1['Distress']['recall']*100:.2f}%",
         f"{s1['Distress']['f1-score']*100:.2f}%",
         f"{int(s1['Distress']['support']):,}"],
        ["Macro avg",
         f"{s1['macro avg']['precision']*100:.2f}%",
         f"{s1['macro avg']['recall']*100:.2f}%",
         f"{s1['macro avg']['f1-score']*100:.2f}%",
         f"{int(s1['macro avg']['support']):,}"],
    ]
    s1_cm = [
        ["", "Pred Normal", "Pred Distress"],
        ["GT Normal", f"{cm1[0][0]:,} (TN)", f"{cm1[0][1]:,} (FP)"],
        ["GT Distress", f"{cm1[1][0]:,} (FN)", f"{cm1[1][1]:,} (TP)"],
    ]

    s2 = BASELINE["stage2"]["classification_report"]
    s2_summary = [
        ["Metric", "Value"],
        ["Primary accuracy", f"{BASELINE['stage2']['primary_accuracy']*100:.2f}%"],
        ["F1 (macro)", f"{BASELINE['stage2']['f1_macro']*100:.2f}%"],
        ["Exact match rate", f"{BASELINE['stage2']['exact_match_rate']*100:.2f}%"],
        ["Avg inference time", f"{BASELINE['stage2']['avg_inference_time_sec']:.2f} s/image"],
        ["Total samples", f"{BASELINE['stage2']['total_samples']:,}"],
    ]
    s2_perclass = [
        ["Class", "Precision", "Recall", "F1", "Support"],
        ["Unparseable / Normal",
         f"{s2['Unparseable/Normal']['precision']*100:.2f}%",
         f"{s2['Unparseable/Normal']['recall']*100:.2f}%",
         f"{s2['Unparseable/Normal']['f1-score']*100:.2f}%",
         f"{int(s2['Unparseable/Normal']['support']):,}"],
        ["D00 Longitudinal",
         f"{s2['Longitudinal Crack (D00)']['precision']*100:.2f}%",
         f"{s2['Longitudinal Crack (D00)']['recall']*100:.2f}%",
         f"{s2['Longitudinal Crack (D00)']['f1-score']*100:.2f}%",
         f"{int(s2['Longitudinal Crack (D00)']['support']):,}"],
        ["D10 Transverse",
         f"{s2['Transverse Crack (D10)']['precision']*100:.2f}%",
         f"{s2['Transverse Crack (D10)']['recall']*100:.2f}%",
         f"{s2['Transverse Crack (D10)']['f1-score']*100:.2f}%",
         f"{int(s2['Transverse Crack (D10)']['support']):,}"],
        ["D20 Alligator",
         f"{s2['Alligator Crack (D20)']['precision']*100:.2f}%",
         f"{s2['Alligator Crack (D20)']['recall']*100:.2f}%",
         f"{s2['Alligator Crack (D20)']['f1-score']*100:.2f}%",
         f"{int(s2['Alligator Crack (D20)']['support']):,}"],
        ["D40 Pothole",
         f"{s2['Pothole (D40)']['precision']*100:.2f}%",
         f"{s2['Pothole (D40)']['recall']*100:.2f}%",
         f"{s2['Pothole (D40)']['f1-score']*100:.2f}%",
         f"{int(s2['Pothole (D40)']['support']):,}"],
        ["Macro avg",
         f"{s2['macro avg']['precision']*100:.2f}%",
         f"{s2['macro avg']['recall']*100:.2f}%",
         f"{s2['macro avg']['f1-score']*100:.2f}%",
         f"{int(s2['macro avg']['support']):,}"],
    ]

    return chapter(
        "6. Phase 1 Baseline Results",
        P(
            "All numbers below come from running the <b>raw, un-fine-tuned</b> "
            "Qwen2.5-VL-7B-Instruct model on the full test splits. fp16, no quantization, "
            "RTX A5000. Greedy decoding (deterministic). Source: <code>eval_results/baseline_results.json</code>."
        ),
        section("6.1 Stage 1 — Binary detection (10,000 GAPs images)"),
        table(s1_summary, col_widths=[5 * cm, 5 * cm]),
        Spacer(1, 6),
        subsection("Per-class breakdown"),
        table(s1_perclass, col_widths=[3.5 * cm, 3 * cm, 3 * cm, 3 * cm, 3 * cm]),
        Spacer(1, 6),
        subsection("Confusion matrix"),
        table(s1_cm, col_widths=[3.5 * cm, 5 * cm, 5 * cm]),
        Spacer(1, 8),
        fig(DIAGRAMS_DIR / "stage1_cm.png", width_cm=11,
            cap="Figure 6.1 — Stage 1 confusion matrix on GAPs V2 test (10,000 images)."),
        callout(
            "<b>The headline failure mode: Normal-bias.</b> The model correctly identifies "
            "98.97% of normal pavement (Normal recall) but misses 57.12% of distressed "
            "pavement (Distress recall is only 42.88%). When it does say Distress it is "
            "almost always right (96.51% precision), but it is reluctant to say it. "
            "Fine-tuning needs to shift this decision boundary toward more aggressive "
            "Distress detection, even if Normal precision drops a little."
        ),
        section("6.2 Stage 2 — Type classification (5,758 RDD images)"),
        table(s2_summary, col_widths=[5 * cm, 5 * cm]),
        Spacer(1, 6),
        subsection("Per-class breakdown"),
        table(s2_perclass, col_widths=[3.5 * cm, 3 * cm, 3 * cm, 3 * cm, 3 * cm]),
        Spacer(1, 8),
        fig(DIAGRAMS_DIR / "stage2_cm.png", width_cm=14,
            cap="Figure 6.2 — Stage 2 confusion matrix on RDD2022 test. Note how almost everything not classified as Unparseable falls into the D00 column."),
        section("6.3 What the Stage 2 numbers actually mean"),
        bullet("<b>D00 Longitudinal recall 72.50%</b> — the only class that 'works'. The model recognises crack patterns and defaults to longitudinal."),
        bullet("<b>D10 Transverse recall 0.00%</b> — every single transverse crack is mis-labeled as either longitudinal or unparseable. Cracks are visually identical except for orientation, and the base model has no orientation prior."),
        bullet("<b>D20 Alligator recall 2.57%</b> — of 778 alligator-crack images, 593 are classified as longitudinal (D00). The interconnected pattern is distinctive but the base model lacks the domain vocabulary."),
        bullet("<b>D40 Pothole recall 1.94%</b> — even potholes, the most visually obvious distress, are barely recognised. The model needs to learn the 'bowl-shaped depression' visual cue."),
        bullet("<b>Unparseable rate 34.24%</b> — one in three responses cannot be cleanly parsed into an RDD class. Fine-tuning on the exact output format will fix this."),
        bullet("<b>48.66% overall accuracy is misleading</b>. Random-guess baseline with 5 effective classes is ~20%. The 48.66% is inflated by D00 and the unparseable bucket; the macro F1 of 20.71% reveals that most classes are not learnt."),
    )


def database_chapter():
    return chapter(
        "7. Database Schema",
        P(
            "All persistent state lives in Supabase Postgres. The schema is small (5 tables, "
            "2 views, 2 RPCs) but every column is load-bearing. Below is a tour of each table "
            "and how they connect."
        ),
        fig(DIAGRAMS_DIR / "er_diagram.png", width_cm=16.5,
            cap="Figure 7.1 — Entity-relationship diagram. The trigger from photos to assessments is the only place where the AI pipeline meets the mobile app."),
        section("7.1 photos (RoadSide app's table)"),
        P(
            "Owned by the mobile app team. The pipeline never writes to it. Mandatory "
            "columns: <code>id</code> (BIGINT PK), <code>image_url</code>, "
            "<code>address</code>, <code>latitude</code>, <code>longitude</code>, "
            "<code>created_at</code>."
        ),
        section("7.2 assessments (the heart of the pipeline)"),
        P("Every photo gets exactly one row here. It evolves through the status state machine as the worker processes it. Key columns:"),
        bullet("<code>id UUID PK</code>, <code>photo_id BIGINT FK photos(id)</code> with a unique index — guarantees one assessment per photo."),
        bullet("<code>status TEXT CHECK IN ('pending', 'processing', 'classified', 'expert_review', 'done', 'failed')</code> — the state machine."),
        bullet("<code>claimed_at, claimed_by, retry_count</code> — used by the worker to detect stale claims and stop runaway retries."),
        bullet("<code>stage1_label, stage1_confidence, is_distressed</code> — Stage 1 outputs."),
        bullet("<code>distress_types JSONB, severity, description, stage2_confidence</code> — Stage 2 outputs."),
        bullet("<code>needs_expert_review, expert_reviewed, expert_corrected_types, expert_corrected_severity, expert_notes, reviewed_by, reviewed_at</code> — the human-in-the-loop columns."),
        bullet("<code>processing_time_ms, model_version, raw_response JSONB, processed_at</code> — observability."),
        section("7.3 worker_state"),
        P(
            "Heartbeat table. The worker upserts its row every 30 seconds with the current "
            "in-flight image ID, cumulative counters, and rolling per-stage timings. The "
            "operator dashboard polls this row to know whether the worker is alive."
        ),
        section("7.4 retrain_jobs"),
        P(
            "One row per LoRA incremental retrain. Tracks <code>status</code> "
            "(pending/running/completed/failed), <code>corrections_count</code>, progress "
            "percentage, and the <code>adapter_path</code> the new adapter was saved to."
        ),
        section("7.5 custom_distress_types"),
        P(
            "Reference data — the dropdown options for the expert UI. Pre-seeded with 12 "
            "standard types, but experts can add custom ones."
        ),
        section("7.6 The photos -> assessments trigger"),
        P("This is the bridge between the mobile app's world and the pipeline's world. SQL excerpt:"),
        code(
            "CREATE TRIGGER trg_sync_photo_to_assessment\n"
            "  AFTER INSERT ON photos\n"
            "  FOR EACH ROW EXECUTE FUNCTION sync_photo_to_assessment();\n\n"
            "-- inside the function:\n"
            "INSERT INTO assessments (photo_id, image_url, latitude,\n"
            "  longitude, address, status, model_version, created_at)\n"
            "VALUES (NEW.id, NEW.image_url, NEW.latitude, NEW.longitude,\n"
            "  NEW.address, 'pending', 'pending', NEW.created_at);"
        ),
        P(
            "The trigger uses <code>SECURITY DEFINER</code> so it runs with the table owner's "
            "privileges (otherwise the mobile app's authenticated role could not bypass RLS "
            "on the assessments table). It is idempotent: an explicit "
            "<code>EXISTS</code> guard plus a unique index on <code>photo_id</code> ensures "
            "no duplicates if the trigger somehow fires twice."
        ),
        section("7.7 Two SQL views for the dashboard"),
        bullet("<code>pipeline_metrics</code> — single-row view aggregating queue depth per status, throughput in last hour and last minute, average confidences, and Distress-vs-Normal split."),
        bullet("<code>pipeline_class_distribution</code> — per-distress-type counts for the last 24 hours, used by the dashboard's pie chart."),
    )


def operator_pipeline_chapter():
    return chapter(
        "8. Operator Pipeline (Cookie-Point)",
        P(
            "This is the most technically interesting piece of the whole system. It turns "
            "a Postgres table into a reliable distributed job queue that survives crashes, "
            "ships incremental progress to a live dashboard, and never accidentally "
            "double-processes a row."
        ),
        fig(DIAGRAMS_DIR / "async_worker_flow.png", width_cm=16.5,
            cap="Figure 8.1 — The asyncio worker loop. Three concurrent tasks: poll loop, heartbeat, and stale-claim recovery."),
        section("8.1 Why an asyncio worker"),
        P(
            "The model itself is synchronous (CUDA forward passes block the calling thread), "
            "but everything around it — HTTP downloads, Supabase REST calls, dashboard SSE "
            "streaming — is I/O. We use Python's <code>asyncio</code> for the orchestration "
            "and wrap every model call in <code>asyncio.to_thread()</code>. This way the event "
            "loop stays responsive: while the GPU is busy on Stage 2, the heartbeat task can "
            "still write to <code>worker_state</code>, and the dashboard's SSE stream still "
            "delivers live metrics."
        ),
        section("8.2 Atomic batch claim — FOR UPDATE SKIP LOCKED"),
        callout(
            "PostgREST (Supabase's REST layer) cannot do <code>UPDATE ... LIMIT</code>. "
            "Solution: a database RPC that does the right thing. This is the <b>canonical "
            "PostgREST job-queue pattern</b>, well-documented in Supabase community resources."
        ),
        code(
            "CREATE FUNCTION claim_pending_assessments(\n"
            "  batch_size INT, worker_id TEXT) RETURNS SETOF assessments AS $$\n"
            "  UPDATE assessments\n"
            "  SET status='processing', claimed_at=NOW(), claimed_by=worker_id\n"
            "  WHERE id IN (\n"
            "    SELECT id FROM assessments\n"
            "    WHERE status='pending'\n"
            "    ORDER BY created_at ASC\n"
            "    FOR UPDATE SKIP LOCKED\n"
            "    LIMIT batch_size\n"
            "  )\n"
            "  RETURNING *;\n"
            "$$ LANGUAGE plpgsql SECURITY DEFINER;"
        ),
        P(
            "<code>FOR UPDATE SKIP LOCKED</code> is the magic. If two workers call this "
            "function at exactly the same time, Postgres gives each one a different "
            "set of rows. There is no overlap, no race condition, no deadlock. Our "
            "robustness test #5 verifies this experimentally with two concurrent calls."
        ),
        section("8.3 Heartbeat + stale-claim recovery"),
        P(
            "Every 30 seconds the worker upserts a row in <code>worker_state</code> with the "
            "current image ID, cumulative counters, and rolling timings. The dashboard polls "
            "this row to render the worker's live state."
        ),
        P(
            "Every 120 seconds the worker also calls <code>reset_stale_processing_assessments"
            "(stale_threshold_seconds=300)</code>. This finds any rows where "
            "<code>status='processing'</code> and <code>claimed_at &lt; NOW() - 5 minutes</code>, "
            "resets them to <code>pending</code>, and increments <code>retry_count</code>. "
            "If the worker process crashes, the rows it had claimed are eventually picked up "
            "by the next worker (or by the same worker on restart) without manual intervention."
        ),
        section("8.4 Retry strategy with exponential backoff"),
        P(
            "Each row gets up to 3 attempts. Errors are classified as <b>transient</b> "
            "(network/timeout/HTTP 5xx) or <b>terminal</b> (corrupt JPEG, unparseable image). "
            "Transient errors get retried with delays of 1s, 2s, 4s. Terminal errors fail "
            "fast on the first attempt — there is no point retrying a corrupt JPEG."
        ),
        section("8.5 Live per-stage progress"),
        P(
            "The worker exposes <code>current_stage</code> "
            "(<code>'downloading'</code> | <code>'stage1'</code> | <code>'stage2'</code> | "
            "<code>None</code>) and <code>current_stage_started_at</code>. "
            "The dashboard renders three stepped dots (Download &rarr; Stage 1 &rarr; Stage 2) "
            "with five visual states: queued, active+pulsing, done, failed, and skipped. "
            "When Stage 1 says Normal, the Stage 2 dot shows 'skipped (Normal)' — visually "
            "communicating that the bypass was intentional."
        ),
        P(
            "Implementation note: <code>predict()</code> was refactored into "
            "<code>predict_stage1()</code> and <code>predict_stage2()</code> so the worker "
            "can update <code>current_stage</code> between them. <code>_process_one</code> "
            "wraps everything in <code>try/finally</code> so <code>current_stage</code> is "
            "always cleared on exceptions, never leaving the dashboard with a stuck spinner."
        ),
        section("8.6 The status state machine"),
        fig(DIAGRAMS_DIR / "status_state_machine.png", width_cm=16.5,
            cap="Figure 8.2 — assessments.status transitions. Every transition is enforced by either an RPC or the worker."),
        section("8.7 Critical OOM fix"),
        callout(
            "Our first run on real Cloudinary photos crashed 1 of 15 images with CUDA OOM "
            "on a 3456 x 3456 photo (~12M pixels &rarr; ~15k visual tokens, attention "
            "O(n^2) blew past 24GB). Fix: pass <code>max_pixels=2200*2200</code> and "
            "<code>min_pixels=256*28</code> to <code>AutoProcessor.from_pretrained</code>. "
            "Don't pre-downscale on the worker side with PIL — let the processor handle "
            "it; it knows how to reduce token count without losing fidelity."
        ),
    )


def dashboard_chapter():
    return chapter(
        "9. Operator Dashboard UI",
        P(
            "A single-page HTML/JS dashboard at <code>GET /operator</code>. It uses the same "
            "design language as the expert UI (SK Modernist font, neutral palette, no "
            "frameworks, no build tools). The whole thing is one self-contained file."
        ),
        section("What it shows"),
        bullet("<b>Header status pill</b> — green dot if worker is running, red if stopped, with last-heartbeat seconds elapsed."),
        bullet("<b>Start / Stop buttons</b> — POST <code>/operator/start</code> or <code>/operator/stop</code>. Stop is graceful: in-flight image finishes."),
        bullet("<b>Queue depth metrics</b> — Pending, Processing, Classified, Expert review, Done, Failed counts (live)."),
        bullet("<b>Throughput metrics</b> — images/min and images/hr based on <code>processed_at</code> in the last window."),
        bullet("<b>Confidence averages</b> — rolling average Stage 1 and Stage 2 confidence over the last hour."),
        bullet("<b>Per-class distribution</b> — bar chart of distress-type counts in the last 24h."),
        bullet("<b>Currently-processing card</b> — live thumbnail of the in-flight image plus its three-dot per-stage progress (Download &rarr; Stage 1 &rarr; Stage 2)."),
        bullet("<b>Recent activity feed</b> — last 20 processed rows with confidence, class, and a small thumbnail."),
        section("How it stays live"),
        P(
            "The dashboard opens an SSE connection to <code>GET /operator/metrics/stream</code>. "
            "The server emits a JSON snapshot every 2 seconds. Because EventSource cannot send "
            "custom auth headers, the client uses <code>fetch()</code> + <code>ReadableStream</code>. "
            "When the worker is stopped, the stream still emits — only the values are different."
        ),
        section("Hover tooltips and help panel"),
        P(
            "Every metric has a hover tooltip explaining what it counts and where it comes from "
            "(e.g. \"throughput_last_hour: count of rows where processed_at >= now() - 1 hour AND "
            "status IN ('classified','expert_review','done')\"). A toggleable help panel "
            "documents the status state machine and the SSE event shapes. This means a new "
            "team member can sit down at the dashboard and figure out what is happening "
            "without reading the source code."
        ),
    )


def robustness_chapter():
    rt = [
        ["#", "Test", "What it does", "Pass criterion", "Result"],
        ["1", "404 image",
         "Inserts a row pointing at httpbin.org/status/404",
         "Row reaches status=failed after 3 retries with error message",
         "PASS"],
        ["2", "Corrupt JPEG",
         "Inserts a row pointing at junk-bytes file served as image/jpeg",
         "Row marked failed (non-transient — fast fail, retry_count=1)",
         "PASS"],
        ["3", "Network timeout",
         "Inserts a row pointing at httpbin.org/delay/120 (server hangs)",
         "Row marked failed with timeout error after 3 retries (~100-150s)",
         "PASS"],
        ["4", "Mid-process stop",
         "Hits /operator/stop while a row is in flight",
         "Row drains gracefully to classified/expert_review OR is released back to pending",
         "PASS"],
        ["5", "Concurrent claim",
         "Calls claim_pending_assessments RPC twice in parallel against 5 pending rows",
         "Zero overlap between the two returned sets (FOR UPDATE SKIP LOCKED works)",
         "PASS"],
    ]
    return chapter(
        "10. Robustness Verification",
        P(
            "Reliability cannot be argued, only tested. The file "
            "<code>scripts/tests/test_pipeline_robustness.py</code> runs five independent "
            "scenarios that each exercise a different failure mode of the pipeline. All five "
            "pass against the live system. The suite is re-runnable after any change to "
            "<code>app/worker.py</code> or <code>app/supabase_client.py</code>."
        ),
        table(rt, col_widths=[0.7 * cm, 2.6 * cm, 5.5 * cm, 5.5 * cm, 1.6 * cm], fontSize=8.5),
        Spacer(1, 6),
        section("Why each test matters"),
        bullet("<b>Test 1 (404)</b> — proves the worker doesn't crash on a missing image, and that the retry counter actually counts."),
        bullet("<b>Test 2 (corrupt JPEG)</b> — proves we distinguish non-transient errors so we don't waste retries on something that will never work."),
        bullet("<b>Test 3 (timeout)</b> — proves the httpx read timeout is enforced and surfaces correctly through the retry loop."),
        bullet("<b>Test 4 (mid-stop)</b> — proves the graceful drain logic works. Without this, hitting Stop could leave a row stuck in 'processing' forever."),
        bullet("<b>Test 5 (concurrent claim)</b> — proves the atomic claim RPC is actually atomic. This is the test that lets us claim the FOR UPDATE SKIP LOCKED design is correct."),
        section("End-to-end smoke test"),
        P(
            "Beyond the failure scenarios, a smoke test was run with 3 mock pending rows "
            "pointing at <code>test_fixtures/longitudinal_D00.jpg</code>, "
            "<code>alligator_D20.jpg</code>, and <code>pothole_D40.jpg</code>. Result: all 3 "
            "processed successfully. Confidence threshold logic verified at the boundary: "
            "0.971 -&gt; classified, 0.881 -&gt; classified, 0.798 -&gt; expert_review (just below "
            "the 0.80 cutoff). Stage 1 timing averaged 821 ms on A5000 with 4-bit quantization."
        ),
    )


def expert_loop_chapter():
    return chapter(
        "11. Expert-in-the-Loop (Phase 3 Plan)",
        P(
            "Confidence thresholds let the model say \"I'm not sure.\" The next question is what "
            "to do with that uncertainty. Phase 3 closes the loop with two complementary "
            "mechanisms — one immediate, one permanent."
        ),
        section("11.1 Immediate: few-shot prompt injection"),
        P(
            "When an expert corrects a prediction in the expert UI, the correction is "
            "(a) saved to Supabase as the authoritative record AND (b) sent to the API via "
            "<code>POST /corrections</code>. The API maintains an in-memory list of the 5 "
            "most recent corrections. Before the next inference, the Stage 2 system prompt "
            "is dynamically rebuilt by appending those 5 corrections as few-shot examples."
        ),
        P(
            "Effect: the model improves <b>instantly</b>, with no retraining. This is "
            "in-context learning. It costs only the prompt tokens. The corrections persist "
            "to <code>data/expert_corrections.json</code> on disk so they survive API restarts."
        ),
        section("11.2 Permanent: LoRA incremental retrain"),
        P(
            "Few-shot is fast but ephemeral — the prompt window only fits ~5 examples. Once "
            "we have accumulated enough corrections (configurable threshold), the operator "
            "clicks \"Retrain\" in the expert UI. This triggers "
            "<code>scripts/06_incremental_retrain.py</code>, which:"
        ),
        bullet("Builds a new ShareGPT JSON from the accumulated corrections."),
        bullet("Loads the previous adapter as the starting point."),
        bullet("Runs LLaMA-Factory training for 1-2 epochs on just the corrections (small, fast)."),
        bullet("Saves the result as <code>adapters/vN/</code> with a <code>metadata.json</code> noting the version, timestamp, corrections count, base adapter, and post-training metrics."),
        bullet("Updates <code>adapters/active_version.txt</code> (Windows-friendly — symlinks need admin)."),
        bullet("Calls <code>reload_adapter()</code> on the running classifier."),
        section("11.3 Versioned adapters with rollback"),
        code(
            "adapters/\n"
            "  v1/\n"
            "    adapter_config.json\n"
            "    adapter_model.safetensors\n"
            "    metadata.json\n"
            "  v2/\n"
            "    ...\n"
            "  active_version.txt   -> 'v2'"
        ),
        P(
            "The expert UI will have an adapter dropdown. Selecting an old version "
            "(via <code>POST /adapters/switch</code>) reloads the model with that adapter "
            "and updates <code>active_version.txt</code>. If a retrain produces a worse "
            "model, rollback is one click."
        ),
        section("11.4 Why both — the cookie-point"),
        callout(
            "The paper compares three states: (a) baseline (no corrections), (b) baseline + "
            "few-shot only, (c) baseline + few-shot + LoRA retrain. This gives us TWO "
            "methodology sections in the paper and a clear comparison table showing immediate "
            "vs permanent improvement. Published precedent: DamageQwen 2025 (same model "
            "family) showed few-shot alone gives ~18% improvement over zero-shot; we expect "
            "LoRA retrain to add another step on top."
        ),
    )


def webgis_chapter():
    return chapter(
        "12. WebGIS Layer (Phase 4 Plan)",
        P(
            "The end consumer of the system is not a citizen and not an expert — it is a "
            "civic authority deciding which roads to repair next. The WebGIS layer turns the "
            "individual assessments into a map a road department can act on."
        ),
        section("12.1 What it is"),
        P(
            "A web page, served from the same FastAPI app, that loads "
            "<code>status='done'</code> rows from <code>assessments</code> joined with "
            "<code>photos</code>, plots each one as a marker on a Leaflet.js map (OpenStreetMap "
            "tile layer), and color-codes the markers by distress type and severity."
        ),
        section("12.2 Data model"),
        bullet("<b>Marker color</b> — by primary distress type (D00 grey, D10 yellow, D20 orange, D40 red)."),
        bullet("<b>Marker size</b> — by severity (Low: 6px, Medium: 10px, High: 14px)."),
        bullet("<b>Hover tooltip</b> — distress type, severity, confidence, timestamp."),
        bullet("<b>Click popup</b> — full description, raw photo thumbnail, link to full-size image."),
        bullet("<b>Legend</b> — color/size key + counts per category for the visible map area."),
        section("12.3 Heatmap mode"),
        P(
            "When zoomed out, individual markers cluster into a heatmap (using "
            "<code>Leaflet.heat</code>). Heat intensity = count of distress assessments per "
            "tile, weighted by severity. This makes hotspots visually obvious at neighbourhood "
            "scale."
        ),
        section("12.4 Roles and access"),
        P(
            "Civic authorities log in via Supabase Auth with a custom 'authority' role. "
            "Anonymous users see the map but not the underlying photos. Authority users can "
            "click through to the original images and download a CSV of assessments in "
            "the visible map area."
        ),
        section("12.5 Future: temporal tracking"),
        P(
            "Because every assessment has a timestamp and (after Phase 4) a "
            "<code>road_segment_id</code>, we can plot deterioration of a single road over "
            "time. This is the data that justifies investment in proactive maintenance — "
            "showing that a road went from Low to High severity in 6 months is more "
            "actionable than a static snapshot."
        ),
    )


def critical_decisions_chapter():
    rows = [
        ["#", "Decision", "Why it matters"],
        ["1", "Template = qwen2_5_vl, not qwen2_vl",
         "Wrong template silently produces broken training. The architectures look similar but the chat-format tokens differ."],
        ["2", "Never call merge_and_unload() on a quantized model",
         "PEFT issue #2586: merging 4-bit weights with LoRA adapters produces corrupted output. Keep as PeftModel forever."],
        ["3", "CONFIDENCE_THRESHOLD = 0.80 in scripts/utils.py only",
         "Single source of truth. Any drift between API, worker, and dashboard would silently change thresholds."],
        ["4", "Max 2 training epochs",
         "50k+ datasets overfit at 3+ epochs (empirically observed in QLoRA literature). Epochs are a hyperparameter for SMALL datasets."],
        ["5", "lora_dropout = 0.1",
         "Acts as sparsity regularizer. Without it, adapters are too 'eager' to overfit specific patterns from the training set."],
        ["6", "Flash Attention 2 auto-detected",
         "Code checks GPU compute capability >= 8.0 AND flash-attn package install. Falls back to SDPA gracefully on hardware that doesn't support it."],
        ["7", "asyncio.to_thread() wraps model inference",
         "Without it, the synchronous CUDA call blocks the FastAPI event loop, freezing health checks and SSE streams during inference."],
        ["8", "Image format validation checks both .format AND .mode",
         "PIL leaves .format unset on some inputs; checking only one bypasses the validator. Both must agree."],
        ["9", "predict() split into predict_stage1() + predict_stage2()",
         "Lets the worker update current_stage between stages so the dashboard's per-stage progress dots are accurate."],
        ["10", "SSE uses fetch() + ReadableStream, not EventSource",
         "EventSource only supports GET. Image upload requires POST + multipart/form-data."],
        ["11", "Windows uses active_version.txt, not symlinks",
         "Symlinks need admin privileges on Windows. A plain text file works on every OS."],
        ["12", "Unparseable Stage 1 -> Distress with 0% confidence",
         "Errs on the side of caution. A damaged road silently classified as Normal is a safety hazard; a Normal road sent for review only wastes human time."],
        ["13", "image.load() after Image.open() in SSE endpoint",
         "Forces full pixel decode so the BytesIO buffer can be garbage collected. Without it, async streaming holds stale references."],
        ["14", "Client disconnect check before Stage 2 in SSE",
         "Avoids wasting GPU time if the client closed the connection during Stage 1 (~5 second window)."],
        ["15", "max_pixels = 2200*2200 in AutoProcessor",
         "Without it, a 3456x3456 phone photo blows up attention O(n^2) and crashes the worker with CUDA OOM."],
        ["16", "Pinned transformers/bitsandbytes/accelerate/peft versions",
         "transformers>=4.52 broke Params4bit compatibility. The exact pinned combination (4.57.6 / 0.44.1 / 0.34.2 / 0.14.0) is empirically verified."],
        ["17", "FOR UPDATE SKIP LOCKED in claim RPC",
         "Atomic batch claim. Multiple workers (or a misbehaving caller) cannot double-claim the same row. Verified by robustness test #5."],
        ["18", "Trigger uses SECURITY DEFINER",
         "Mobile app's authenticated role can INSERT into photos but not assessments. The trigger function bypasses RLS to create the matching assessments row."],
    ]
    return chapter(
        "13. Critical Technical Decisions",
        P(
            "These are the gotchas that took us hours to find and that will silently break "
            "the system if anyone changes them without understanding why. Each row is a "
            "constraint, not a preference."
        ),
        table(rows, col_widths=[0.8 * cm, 5.5 * cm, 9.7 * cm], fontSize=8.5),
    )


def status_chapter():
    rows = [
        ["Phase", "Component", "Status"],
        ["1", "Data prep (GAPs PNGs + RDD ShareGPT JSONs)", "DONE (80k + 32k generated)"],
        ["1", "Two-stage model with predict_stage1/predict_stage2", "DONE"],
        ["1", "Logit-based confidence extraction", "DONE"],
        ["1", "FastAPI server (/classify, /classify/stream, /health)", "DONE"],
        ["1", "SSE streaming with fetch + ReadableStream", "DONE"],
        ["1", "test_website/index.html for remote demo", "DONE"],
        ["1", "Supabase schema (assessments, custom_distress_types, retrain_jobs)", "DONE"],
        ["1", "Migration 001 — operator pipeline columns + RPC + views", "DONE (applied to live DB)"],
        ["1", "Migration 002 — photos -> assessments trigger", "DONE (applied to live DB)"],
        ["1", "Async worker (claim, retry, heartbeat, stale recovery)", "DONE"],
        ["1", "Operator dashboard (live SSE, per-stage progress)", "DONE"],
        ["1", "Robustness test suite (5 tests)", "DONE (all passed)"],
        ["1", "Baseline evaluation on 7B model (Stage 1 + Stage 2)", "DONE (76.5% / 48.7%)"],
        ["1", "Cloudflare Tunnel for remote access", "DONE"],
        ["", "", ""],
        ["2", "Attain dataset download + inspect format", "PENDING"],
        ["2", "scripts/07_cross_dataset_eval.py", "PENDING (depends on Attain format)"],
        ["2", "Run 07 on baseline (zero-shot Attain)", "PENDING"],
        ["2", "QLoRA fine-tune on RDD+GAPs (12-24 hours on A5000)", "PENDING"],
        ["2", "scripts/04_post_finetune_eval.py on RDD", "PENDING"],
        ["2", "Run 07 on fine-tuned model (in-distribution Attain transfer)", "PENDING"],
        ["2", "Generate three comparison tables for the paper", "PENDING"],
        ["", "", ""],
        ["3", "build_dynamic_stage2_prompt() in utils.py", "PENDING"],
        ["3", "POST /corrections endpoint + few-shot list in model.py", "PENDING"],
        ["3", "06_incremental_retrain.py with versioned adapters/vN/", "PENDING"],
        ["3", "GET /adapters, POST /adapters/switch, GET /adapters/active", "PENDING"],
        ["3", "reload_adapter() method in app/model.py", "PENDING"],
        ["3", "Expert UI: adapter dropdown + version display", "PENDING"],
        ["3", "End-to-end correction -> retrain -> rollback test", "PENDING"],
        ["", "", ""],
        ["4", "WebGIS Leaflet map with color-coded markers", "PENDING"],
        ["4", "Heatmap mode for zoom-out view", "PENDING"],
        ["4", "Authority role + private photo download", "PENDING"],
        ["4", "Field validation on real Bengaluru roads (50-100 photos)", "PENDING"],
    ]
    return chapter(
        "14. What Is Done vs What Is Pending",
        P(
            "Honest accounting of the project. Phase 1 is fully complete — code, tests, "
            "and the 76.5% / 48.7% baseline numbers in this document are real. Phases 2-4 "
            "are designed and partially scaffolded but not yet executed."
        ),
        table(rows, col_widths=[1.2 * cm, 9 * cm, 5.8 * cm], fontSize=8.5),
    )


def references_chapter():
    return chapter(
        "15. References and How to Reproduce",
        section("Where to start"),
        P(
            "<code>SETUP.md</code> in the project root has the full reproduction guide. "
            "Every environment variable, every CLI argument, every database column, and "
            "every file in the project is documented there. This handbook is the high-level "
            "tour; SETUP.md is the operator manual."
        ),
        section("Key commands (quick reference)"),
        code(
            "# One-time data prep\n"
            "python scripts/01_convert_gaps_to_images.py\n"
            "python scripts/02_build_training_data.py\n"
            "python scripts/05_setup_training_config.py\n\n"
            "# Phase 1: baseline evaluation (already done)\n"
            "QUANTIZATION_BITS=0 python scripts/03_baseline_eval.py --stage 1\n"
            "QUANTIZATION_BITS=0 python scripts/03_baseline_eval.py --stage 2\n\n"
            "# Phase 2: fine-tuning (12-24h on A5000)\n"
            "llamafactory-cli train configs/qwen25vl_qlora_sft.yaml\n"
            "python scripts/04_post_finetune_eval.py --adapter-path \\\n"
            "  outputs/qwen25vl-qlora-gaps-rdd/\n\n"
            "# Run the API + worker\n"
            "QUANTIZATION_BITS=0 uvicorn app.main:app \\\n"
            "  --host 0.0.0.0 --port 8000 --workers 1\n"
            "./cloudflared.exe tunnel --url http://localhost:8000\n\n"
            "# Robustness tests\n"
            "python scripts/tests/test_pipeline_robustness.py\n\n"
            "# Re-generate this PDF\n"
            "python scripts/generate_handbook_pdf.py"
        ),
        section("File map"),
        bullet("<code>app/main.py</code> — FastAPI server, all endpoints"),
        bullet("<code>app/model.py</code> — PavementClassifier, predict_stage1, predict_stage2, confidence extraction"),
        bullet("<code>app/worker.py</code> — operator pipeline asyncio worker"),
        bullet("<code>app/supabase_client.py</code> — Supabase REST + RPC + Cloudinary download"),
        bullet("<code>scripts/utils.py</code> — paths, prompts, parsing, CONFIDENCE_THRESHOLD"),
        bullet("<code>scripts/03_baseline_eval.py</code> — baseline evaluation harness"),
        bullet("<code>scripts/06_incremental_retrain.py</code> — Phase 3 LoRA incremental retrain"),
        bullet("<code>scripts/tests/test_pipeline_robustness.py</code> — 5-test robustness suite"),
        bullet("<code>db_schema.sql</code> — base schema (assessments, custom_distress_types, retrain_jobs, RLS)"),
        bullet("<code>migrations/001_operator_pipeline.sql</code> — status enum, claim RPC, worker_state, views"),
        bullet("<code>migrations/002_photos_integration.sql</code> — photos -> assessments trigger"),
        bullet("<code>operator_ui/index.html</code> — operator dashboard"),
        bullet("<code>expert_ui/index.html</code> — expert review dashboard"),
        bullet("<code>test_website/index.html</code> — public test website"),
        bullet("<code>configs/qwen25vl_qlora_sft.yaml</code> — LLaMA-Factory training config"),
        bullet("<code>paper/baseline_evaluation.md</code> — academic-style draft paper"),
        bullet("<code>paper/GeoAI_Technical_Handbook.pdf</code> — this document"),
        section("Citation seeds (for the paper)"),
        bullet("Qwen2.5-VL-7B: Bai et al., \"Qwen2.5-VL Technical Report\" (Alibaba, 2024)"),
        bullet("QLoRA: Dettmers et al., \"QLoRA: Efficient Finetuning of Quantized LLMs\" (NeurIPS 2023)"),
        bullet("LLaMA-Factory: Zheng et al., \"LLaMA-Factory: Unified Efficient Fine-Tuning of 100+ Language Models\" (ACL 2024 demo)"),
        bullet("RDD2022: Arya et al., \"RDD2022: A Multi-National Image Dataset for Automatic Road Damage Detection\" (IEEE Big Data 2022)"),
        bullet("GAPs V2: Stricker et al., \"German Asphalt Pavement Distress (GAPs) database v2\""),
        bullet("Attain dataset: Mendeley Data, doi:10.17632/nykrzdm74f/1 (CC BY 4.0)"),
        bullet("DamageQwen (2025): same Qwen2.5-VL family, +18% from few-shot over zero-shot"),
        bullet("Xu et al. (2025): zero-shot LLM matches expert PSCI"),
    )


# ===========================================================================
# Build the PDF
# ===========================================================================

def build_pdf():
    print("[1/3] Generating diagrams...")
    diagram_system_architecture()
    diagram_two_stage_pipeline()
    diagram_status_state_machine()
    diagram_er_diagram()
    diagram_async_worker_flow()

    # Confusion matrices: prefer existing PNGs if present, else generate
    s1_cm_existing = CM_DIR / "baseline_stage1_cm.png"
    s2_cm_existing = CM_DIR / "baseline_stage2_cm.png"
    s1_cm_target = DIAGRAMS_DIR / "stage1_cm.png"
    s2_cm_target = DIAGRAMS_DIR / "stage2_cm.png"

    if s1_cm_existing.exists() and s1_cm_existing.stat().st_size > 1000:
        # Use the existing one (copy / write through)
        from shutil import copyfile
        copyfile(s1_cm_existing, s1_cm_target)
        print(f"   Reused existing {s1_cm_existing.name}")
    else:
        diagram_confusion_matrix_stage1()
        print("   Generated stage1_cm.png from JSON")

    if s2_cm_existing.exists() and s2_cm_existing.stat().st_size > 1000:
        from shutil import copyfile
        copyfile(s2_cm_existing, s2_cm_target)
        print(f"   Reused existing {s2_cm_existing.name}")
    else:
        diagram_confusion_matrix_stage2()
        print("   Generated stage2_cm.png from JSON")

    print("[2/3] Building PDF flowables...")

    flowables = []
    # Cover page (no chapter marker, no page break)
    flowables.extend(cover_page())
    # All chapters (each starts with PageBreak via chapter())
    flowables.extend(executive_summary())
    flowables.extend(system_architecture_chapter())
    flowables.extend(model_chapter())
    flowables.extend(datasets_chapter())
    flowables.extend(finetuning_chapter())
    flowables.extend(baseline_chapter())
    flowables.extend(database_chapter())
    flowables.extend(operator_pipeline_chapter())
    flowables.extend(dashboard_chapter())
    flowables.extend(robustness_chapter())
    flowables.extend(expert_loop_chapter())
    flowables.extend(webgis_chapter())
    flowables.extend(critical_decisions_chapter())
    flowables.extend(status_chapter())
    flowables.extend(references_chapter())

    print("[3/3] Writing PDF...")

    # Document setup
    doc = BaseDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=1.8 * cm,
        title="GeoAI Technical Handbook",
        author="Suraj, Richik Chaudhuri, Sushant Deo",
        subject="Pavement Distress Detection — Capstone Technical Handbook",
    )
    frame = Frame(
        doc.leftMargin, doc.bottomMargin,
        doc.width, doc.height,
        id="normal", showBoundary=0,
    )
    doc.addPageTemplates([PageTemplate(id="all", frames=frame)])

    doc.build(flowables, canvasmaker=HandbookCanvas)

    size = PDF_PATH.stat().st_size
    print(f"\nPDF written: {PDF_PATH}")
    print(f"Size: {size:,} bytes ({size/1024:.1f} KB)")
    return PDF_PATH


if __name__ == "__main__":
    build_pdf()
