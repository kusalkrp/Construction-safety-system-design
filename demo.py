"""
Gradio demo — Construction Safety Monitor.

Usage:
    python demo.py

Upload a construction site image to assess PPE compliance.
Outputs: annotated image, compliance score, full violation report.
"""

import logging
import os

import gradio as gr
import numpy as np
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

WEIGHTS_PATH = os.getenv("MODEL_WEIGHTS_PATH", "runs/train/weights/best.pt")
_pipeline = None


def load_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    from inference.pipeline import ConstructionSafetyPipeline
    _pipeline = ConstructionSafetyPipeline(weights_path=WEIGHTS_PATH)
    logger.info("Pipeline loaded.")
    return _pipeline


def analyse_image(image: np.ndarray | None):
    if image is None:
        return None, "No image provided.", "Upload an image to begin."

    try:
        pipeline = load_pipeline()
    except Exception as exc:
        logger.error("Failed to load pipeline: %s", exc)
        return None, "Model not loaded.", f"Error: {exc}\n\nEnsure MODEL_WEIGHTS_PATH is set in .env"

    try:
        result = pipeline.analyse(image)
        score_text = result.score_result.display
        temporal = pipeline.get_temporal_status()
        if temporal not in ("INSUFFICIENT_DATA", "CLEAR"):
            score_text += f"\nTemporal: {temporal}"
        return result.annotated_frame, score_text, result.formatted_report
    except Exception as exc:
        logger.error("Analysis failed: %s", exc, exc_info=True)
        return None, "Analysis failed.", f"Error during inference: {exc}"


SCORE_BAND_CSS = """
#score-box textarea {
    font-size: 1.8em !important;
    font-weight: bold;
    text-align: center;
}
"""

demo = gr.Interface(
    fn=analyse_image,
    inputs=gr.Image(
        type="numpy",
        label="Construction Site Image",
        image_mode="RGB",
    ),
    outputs=[
        gr.Image(label="Annotated Scene", type="numpy"),
        gr.Textbox(
            label="Compliance Score",
            lines=2,
            elem_id="score-box",
        ),
        gr.Textbox(
            label="Violation Report",
            lines=18,
            max_lines=40,
        ),
    ],
    title="Construction Safety Monitor",
    description=(
        "Upload a construction site image to assess PPE compliance.\n\n"
        "**Detects:** helmets, hi-vis vests, and violations thereof.\n"
        "**Applies:** 6 formal safety rules with context-aware severity.\n"
        "**Outputs:** Compliance score (0–100), annotated image, structured violation report."
    ),
    examples=[],   # add example image paths here after collecting test images
    css=SCORE_BAND_CSS,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
