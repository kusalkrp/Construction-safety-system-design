"""
End-to-end inference pipeline for Construction Safety Monitor.

Orchestrates:
  1. YOLOv8 person + PPE detection
  2. SafetyChecker — applies Rules 1–6, produces ViolationReports
  3. SiteScorer — computes 0–100 compliance score
  4. Annotator — draws annotated frame

For video input, also runs the 30-frame temporal sliding window analysis.

Usage:
    python inference/pipeline.py --source path/to/image.jpg --weights runs/train/weights/best.pt
    python inference/pipeline.py --source path/to/video.mp4 --weights runs/train/weights/best.pt
"""

import argparse
import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

from inference.annotator import draw_annotations
from inference.constants import (
    INTERMITTENT_RATIO_LOW,
    SUSTAINED_RATIO,
    TEMPORAL_WINDOW_FRAMES,
)
from inference.safety_checker import Detection, SafetyChecker, SiteReport
from inference.site_scorer import ScoreResult, SiteScorer

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Output of one pipeline.analyse() call."""
    site_report: SiteReport
    score_result: ScoreResult
    annotated_frame: np.ndarray          # RGB uint8
    formatted_report: str
    timestamp: float


class ConstructionSafetyPipeline:
    """
    Full inference pipeline: YOLO detection → SafetyChecker → SiteScorer → Annotator.

    Args:
        weights_path: Path to trained YOLOv8 .pt weights file.
        conf_threshold: Minimum YOLO detection confidence (default 0.30).
    """

    def __init__(self, weights_path: str | Path, conf_threshold: float = 0.30) -> None:
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")

        self.model = YOLO(str(weights_path))
        self.conf_threshold = conf_threshold
        self.scorer = SiteScorer()
        self._temporal_window: deque[bool] = deque(maxlen=TEMPORAL_WINDOW_FRAMES)

        logger.info("Pipeline loaded. Weights: %s", weights_path)

    def analyse(self, frame_rgb: np.ndarray) -> AnalysisResult:
        """
        Run the full pipeline on a single frame.

        Args:
            frame_rgb: Input frame as (H,W,3) uint8 RGB.

        Returns:
            AnalysisResult with report, score, and annotated frame.
        """
        h, w = frame_rgb.shape[:2]
        checker = SafetyChecker(frame_width=w, frame_height=h)

        detections = self._run_yolo(frame_rgb)
        site_report = checker.analyse(detections, frame_rgb)

        has_violation = any(
            r.violations for r in site_report.worker_reports if not r.is_visitor
        )
        self._temporal_window.append(has_violation)

        site_alert_triggered = site_report.site_alert is not None
        score_result = self.scorer.compute(site_report.worker_reports, site_alert_triggered)

        annotated = draw_annotations(
            frame_rgb,
            site_report.worker_reports,
            score_result=score_result,
            site_alert=site_report.site_alert,
        )

        report_text = self._format_report(site_report, score_result)

        return AnalysisResult(
            site_report=site_report,
            score_result=score_result,
            annotated_frame=annotated,
            formatted_report=report_text,
            timestamp=time.time(),
        )

    def get_temporal_status(self) -> str:
        """Classify current window as SUSTAINED / INTERMITTENT / CLEAR."""
        if len(self._temporal_window) < TEMPORAL_WINDOW_FRAMES:
            return "INSUFFICIENT_DATA"
        ratio = sum(self._temporal_window) / len(self._temporal_window)
        if ratio >= SUSTAINED_RATIO:
            return "SUSTAINED_VIOLATION"
        if ratio >= INTERMITTENT_RATIO_LOW:
            return "INTERMITTENT"
        return "CLEAR"

    def _run_yolo(self, frame_rgb: np.ndarray) -> list[Detection]:
        """Run YOLO inference. Returns list of Detection objects."""
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        results = self.model(frame_bgr, conf=self.conf_threshold, verbose=False)

        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            names = result.names
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(Detection(
                    class_name=names[cls_id],
                    confidence=conf,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                ))
        return detections

    def _format_report(self, site_report: SiteReport, score: ScoreResult) -> str:
        lines = [
            f"Construction Safety Analysis",
            f"{'─' * 44}",
            f"Compliance Score: {score.display}",
            f"Scene Context:    {site_report.scene_context}",
            f"Workers detected: {len(site_report.worker_reports)}",
            f"Violations: CRITICAL={score.n_critical}  HIGH={score.n_high}  WARNING={score.n_warning}",
            "",
        ]
        if site_report.site_alert:
            lines += [site_report.site_alert.human_readable, ""]
        for r in site_report.worker_reports:
            lines.append(r.human_readable)
            if r.recommended_action and r.recommended_action != "No action required.":
                lines.append(f"  → {r.recommended_action}")
            lines.append("")
        if score_trend := self.scorer.get_trend_summary():
            lines += [f"Trend: {score_trend}", ""]
        return "\n".join(lines)


def run_image(pipeline: ConstructionSafetyPipeline, source: Path, output_dir: Path) -> None:
    frame_bgr = cv2.imread(str(source))
    if frame_bgr is None:
        raise FileNotFoundError(f"Could not read image: {source}")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    result = pipeline.analyse(frame_rgb)
    print(result.formatted_report)

    out_path = output_dir / f"{source.stem}_annotated.jpg"
    out_bgr = cv2.cvtColor(result.annotated_frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), out_bgr)
    logger.info("Annotated image saved: %s", out_path)


def run_video(pipeline: ConstructionSafetyPipeline, source: Path, output_dir: Path) -> None:
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = output_dir / f"{source.stem}_annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (fw, fh))

    frame_idx = 0
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = pipeline.analyse(frame_rgb)
            out_bgr = cv2.cvtColor(result.annotated_frame, cv2.COLOR_RGB2BGR)
            writer.write(out_bgr)

            if frame_idx % 30 == 0:
                temporal = pipeline.get_temporal_status()
                logger.info("Frame %d | Score: %s | Temporal: %s",
                            frame_idx, result.score_result.display, temporal)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    logger.info("Annotated video saved: %s", out_path)
    logger.info("Session trend: %s", pipeline.scorer.get_trend_summary())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run construction safety inference pipeline")
    parser.add_argument("--source", required=True, help="Path to image or video file")
    parser.add_argument(
        "--weights",
        default=os.getenv("MODEL_WEIGHTS_PATH", "runs/train/weights/best.pt"),
        help="Path to YOLOv8 .pt weights file",
    )
    parser.add_argument("--conf", type=float, default=0.30, help="Detection confidence threshold")
    parser.add_argument("--output", default="output", help="Output directory")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = ConstructionSafetyPipeline(weights_path=args.weights, conf_threshold=args.conf)

    suffix = source.suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        run_image(pipeline, source, output_dir)
    elif suffix in {".mp4", ".avi", ".mov", ".mkv"}:
        run_video(pipeline, source, output_dir)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


if __name__ == "__main__":
    main()
