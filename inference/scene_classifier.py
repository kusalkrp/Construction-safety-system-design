"""
Heuristic scene classifier — determines 'outdoor' or 'indoor' for Rule 2.

Approach: examine the top 20% of the frame in HSV colour space.
If ≥15% of sampled pixels fall in the sky-blue hue range with sufficient
saturation, classify as outdoor. Otherwise indoor.

This is a deliberate heuristic choice (not a ML classifier) because:
- Construction site sky is reliably blue/grey in the top portion of frame
- Zero additional model weight or inference cost
- Sufficiently accurate for the risk-modulation purpose of Rule 2
- A MobileNet classifier would add complexity without meaningful gain here

Known limitations:
- Overcast / cloudy skies may reduce sky-blue pixel count → false indoor
- Indoor scenes with skylights may trigger false outdoor
- Night scenes will not have sky-blue pixels → always indoor (conservative)
"""

import logging

import cv2
import numpy as np

from inference.constants import (
    SKY_BLUE_HUE_HIGH,
    SKY_BLUE_HUE_LOW,
    SKY_PIXEL_RATIO_OUTDOOR,
    SKY_SAMPLE_TOP_RATIO,
    SKY_SATURATION_MIN,
)

logger = logging.getLogger(__name__)


def classify_scene(frame_rgb: np.ndarray) -> str:
    """
    Classify a frame as 'outdoor' or 'indoor' using sky detection heuristic.

    Args:
        frame_rgb: Frame as (H, W, 3) uint8 RGB array.

    Returns:
        'outdoor' or 'indoor'
    """
    if frame_rgb is None or frame_rgb.size == 0:
        return "outdoor"  # conservative default

    h, w = frame_rgb.shape[:2]
    sample_h = max(1, int(h * SKY_SAMPLE_TOP_RATIO))
    top_region = frame_rgb[:sample_h, :, :]

    # Convert RGB → BGR → HSV (OpenCV expects BGR)
    bgr = cv2.cvtColor(top_region, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Sky-blue hue mask with minimum saturation
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]

    sky_mask = (
        (hue >= SKY_BLUE_HUE_LOW)
        & (hue <= SKY_BLUE_HUE_HIGH)
        & (sat >= SKY_SATURATION_MIN)
    )

    sky_ratio = sky_mask.sum() / sky_mask.size

    scene = "outdoor" if sky_ratio >= SKY_PIXEL_RATIO_OUTDOOR else "indoor"
    logger.debug("Scene classifier: sky_ratio=%.3f → %s", sky_ratio, scene)
    return scene
