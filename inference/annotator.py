"""
Annotator — draws bounding boxes and compliance overlays on frames.

Takes an image and a list of ViolationReports and draws annotations.
Has NO inference logic, NO rule application, NO score computation.

Colour convention:
  Green  (#00AA44) — compliant workers
  Red    (#FF2200) — violations (labelled with severity + violation name)
  Yellow (#FFCC00) — unverifiable / review queue
  Orange (#FF6600) — AT RISK (HIGH severity)
  Blue   (#4488FF) — visitors
"""

import logging

import cv2
import numpy as np

from inference.safety_checker import ViolationReport
from inference.site_scorer import ScoreResult

logger = logging.getLogger(__name__)

# Box colours (BGR for OpenCV)
COLOUR_COMPLIANT   = (68, 170, 0)      # green
COLOUR_CRITICAL    = (0, 34, 255)      # red
COLOUR_HIGH        = (0, 102, 255)     # orange
COLOUR_WARNING     = (0, 204, 255)     # yellow
COLOUR_UNVERIFIABLE = (255, 204, 0)   # yellow
COLOUR_VISITOR     = (255, 136, 68)   # blue
COLOUR_SITE_ALERT  = (0, 0, 200)      # dark red

FONT = cv2.FONT_HERSHEY_SIMPLEX
BOX_THICKNESS = 2
LABEL_FONT_SCALE = 0.55
LABEL_THICKNESS = 1


def _severity_colour(severity: str) -> tuple[int, int, int]:
    mapping = {
        "COMPLIANT":        COLOUR_COMPLIANT,
        "CRITICAL":         COLOUR_CRITICAL,
        "CRITICAL-ELEVATED": COLOUR_CRITICAL,
        "HIGH":             COLOUR_HIGH,
        "WARNING":          COLOUR_WARNING,
        "UNVERIFIABLE":     COLOUR_UNVERIFIABLE,
        "INFO":             COLOUR_VISITOR,
        "SITE ALERT":       COLOUR_SITE_ALERT,
    }
    return mapping.get(severity, COLOUR_UNVERIFIABLE)


def _draw_label(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    colour: tuple[int, int, int],
) -> None:
    """Draw a filled label background with text."""
    (text_w, text_h), baseline = cv2.getTextSize(text, FONT, LABEL_FONT_SCALE, LABEL_THICKNESS)
    pad = 3
    cv2.rectangle(img, (x, y - text_h - pad * 2), (x + text_w + pad * 2, y + baseline), colour, -1)
    cv2.putText(img, text, (x + pad, y - pad), FONT, LABEL_FONT_SCALE, (255, 255, 255), LABEL_THICKNESS, cv2.LINE_AA)


def _draw_dashed_rect(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    colour: tuple[int, int, int],
    dash_len: int = 8,
) -> None:
    """Draw a dashed rectangle — used for PPE violation boxes."""
    pts = [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ]
    for (ax, ay), (bx, by) in pts:
        length = max(abs(bx - ax), abs(by - ay))
        steps = max(1, length // (dash_len * 2))
        for i in range(steps):
            t0 = i * 2 * dash_len / length
            t1 = min(1.0, (i * 2 + 1) * dash_len / length)
            sx, sy = int(ax + t0 * (bx - ax)), int(ay + t0 * (by - ay))
            ex, ey = int(ax + t1 * (bx - ax)), int(ay + t1 * (by - ay))
            cv2.line(img, (sx, sy), (ex, ey), colour, 1, cv2.LINE_AA)


def draw_annotations(
    frame_rgb: np.ndarray,
    worker_reports: list[ViolationReport],
    score_result: ScoreResult | None = None,
    site_alert: ViolationReport | None = None,
) -> np.ndarray:
    """
    Draw all annotations on the frame.

    Args:
        frame_rgb: Input frame as (H,W,3) uint8 RGB.
        worker_reports: Per-worker ViolationReports from SafetyChecker.
        score_result: ScoreResult from SiteScorer for overlay display.
        site_alert: Site-level alert report if Rule 6 triggered.

    Returns:
        Annotated frame as (H,W,3) uint8 RGB.
    """
    # Work in BGR for OpenCV, convert back at the end
    out = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    h, w = out.shape[:2]

    for report in worker_reports:
        if report.is_site_alert:
            continue

        x1, y1, x2, y2 = report.bbox
        colour = _severity_colour(report.severity)

        cv2.rectangle(out, (x1, y1), (x2, y2), colour, BOX_THICKNESS)

        # Draw the PPE violation box (where no_helmet/no_vest was detected) as a
        # thinner dashed rectangle so the viewer can see the actual detection location
        # rather than just the person anchor box.
        if report.violation_bbox is not None:
            vx1, vy1, vx2, vy2 = report.violation_bbox
            _draw_dashed_rect(out, vx1, vy1, vx2, vy2, colour)

        # Build label text
        if report.violations:
            v_short = "/".join(
                v.replace("no_helmet", "NO HELMET")
                 .replace("no_vest", "NO VEST")
                 .replace("partial_compliance", "PARTIAL")
                 .replace("ppe_gap", "PPE GAP")
                for v in report.violations[:2]  # max 2 on label
            )
            label = f"#{report.worker_id} {report.severity}: {v_short} [{report.confidence_tier}]"
        elif report.severity == "UNVERIFIABLE":
            label = f"#{report.worker_id} TOO FAR"
        elif report.is_visitor:
            label = f"#{report.worker_id} VISITOR"
        else:
            label = f"#{report.worker_id} OK"

        label_y = max(y1 - 4, 18)
        _draw_label(out, label, x1, label_y, colour)

    # ── Score overlay ─────────────────────────────────────────────────────────
    if score_result is not None:
        _draw_score_overlay(out, score_result, w, h)

    # ── Site alert banner ─────────────────────────────────────────────────────
    if site_alert is not None:
        _draw_site_alert_banner(out, site_alert, w)

    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def _draw_score_overlay(
    img: np.ndarray,
    result: ScoreResult,
    frame_w: int,
    frame_h: int,
) -> None:
    """Draw compliance score in top-right corner."""
    band_bgr = _hex_to_bgr(result.band_colour)
    score_text = f"{result.score}/100"
    band_text = result.band

    # Score number (large)
    scale_big = 1.2
    (sw, sh), _ = cv2.getTextSize(score_text, FONT, scale_big, 2)
    margin = 10
    x = frame_w - sw - margin * 2 - 6
    y = sh + margin + 6

    cv2.rectangle(img, (x - 6, margin), (frame_w - margin, y + 6), band_bgr, -1)
    cv2.putText(img, score_text, (x, y), FONT, scale_big, (255, 255, 255), 2, cv2.LINE_AA)

    # Band label below
    (bw, bh), _ = cv2.getTextSize(band_text, FONT, 0.5, 1)
    bx = frame_w - bw - margin * 2 - 4
    by = y + bh + 6
    cv2.putText(img, band_text, (bx, by), FONT, 0.5, band_bgr, 1, cv2.LINE_AA)


def _draw_site_alert_banner(img: np.ndarray, alert: ViolationReport, frame_w: int) -> None:
    """Draw a red site-alert banner at the bottom of the frame."""
    h = img.shape[0]
    banner_h = 32
    cv2.rectangle(img, (0, h - banner_h), (frame_w, h), COLOUR_SITE_ALERT, -1)
    cv2.putText(
        img,
        "⚠ SITE ALERT — SYSTEMIC NON-COMPLIANCE — SUPERVISOR REQUIRED",
        (8, h - 9),
        FONT,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _hex_to_bgr(hex_colour: str) -> tuple[int, int, int]:
    """Convert #RRGGBB to (B, G, R)."""
    h = hex_colour.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)
