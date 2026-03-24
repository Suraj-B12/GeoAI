"""
Interactive Gradio UI for testing the Pavement Distress Classification pipeline.

Uses the shared PavementClassifier from app/model.py (single source of truth).

Usage:
    python gradio_ui/demo.py
    python gradio_ui/demo.py --adapter-path outputs/qwen25vl-qlora-gaps-rdd/
    python gradio_ui/demo.py --model Qwen/Qwen2.5-VL-3B-Instruct
"""

import argparse
import os
import sys
from pathlib import Path

import gradio as gr
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.model import PavementClassifier, CONFIDENCE_THRESHOLD

# Global classifier
CLASSIFIER = None


def classify_pavement(image):
    """Run the two-stage pipeline and format results."""
    if image is None:
        return "Upload an image", "", "", ""

    if CLASSIFIER is None:
        return "Model not loaded", "", "", ""

    pil_image = Image.fromarray(image).convert("RGB")
    result = CLASSIFIER.predict(pil_image)

    # Status line
    if not result["is_distressed"]:
        status = f"NORMAL ({result['stage1_confidence']*100:.1f}% confident)"
    elif result["needs_expert_review"]:
        status = f"DISTRESS DETECTED — LOW CONFIDENCE (needs expert review)"
    else:
        status = f"DISTRESS DETECTED ({result['stage1_confidence']*100:.1f}% confident)"

    # Summary
    lines = [
        f"Stage 1: {result['stage1_label']}  (confidence: {result['stage1_confidence']*100:.1f}%)",
        f"Stage 1 time: {result['stage1_time_ms']:.0f}ms",
    ]

    if result["is_distressed"]:
        types_str = ", ".join(result["distress_types"])
        lines += [
            "",
            f"Stage 2: {types_str}",
            f"Severity: {result['severity']}",
            f"Description: {result['description']}",
            f"Stage 2 confidence: {result['stage2_confidence']*100:.1f}%",
            f"Stage 2 time: {result['stage2_time_ms']:.0f}ms",
        ]

    if result["needs_expert_review"]:
        lines += [
            "",
            "--- FLAGGED FOR EXPERT REVIEW ---",
            f"Confidence threshold: {CONFIDENCE_THRESHOLD*100:.0f}%",
        ]

    lines.append(f"\nTotal time: {result['processing_time_ms']:.0f}ms")
    summary = "\n".join(lines)

    return status, result.get("stage1_raw", ""), result.get("stage2_raw", ""), summary


def build_ui():
    with gr.Blocks(title="Pavement Distress Classifier", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # Pavement Distress Classification
            ### Two-Stage AI Pipeline — Qwen2.5-VL

            **Stage 1:** Detects distress (Normal / Distress)
            **Stage 2:** Classifies type, severity, confidence

            Images with confidence below 80% are flagged for expert review.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(label="Upload Pavement Image", type="numpy", height=400)
                classify_btn = gr.Button("Analyze", variant="primary", size="lg")

            with gr.Column(scale=1):
                status_output = gr.Textbox(label="Status", lines=1, interactive=False)
                summary_output = gr.Textbox(label="Analysis", lines=14, interactive=False)

        with gr.Accordion("Raw Model Output", open=False):
            with gr.Row():
                s1_raw = gr.Textbox(label="Stage 1 Raw", lines=3, interactive=False)
                s2_raw = gr.Textbox(label="Stage 2 Raw", lines=6, interactive=False)

        classify_btn.click(
            fn=classify_pavement,
            inputs=[image_input],
            outputs=[status_output, s1_raw, s2_raw, summary_output],
        )

    return demo


def main():
    global CLASSIFIER

    parser = argparse.ArgumentParser(description="Gradio UI")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    CLASSIFIER = PavementClassifier(args.model, args.adapter_path)

    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
