"""
Named constants loaded from rules.yaml.
Import from here — never hardcode thresholds inline in inference code.
"""

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_RULES_PATH = Path(__file__).parent.parent / "rules.yaml"


def _load(rules_path: Path = _DEFAULT_RULES_PATH) -> dict:
    if not rules_path.exists():
        raise FileNotFoundError(f"rules.yaml not found: {rules_path}")
    with open(rules_path) as f:
        return yaml.safe_load(f)


_cfg = _load()

# ── Detection thresholds ──────────────────────────────────────────────────────
HIGH_CONFIDENCE_THRESHOLD: float      = _cfg["detection"]["high_confidence_threshold"]
MEDIUM_CONFIDENCE_THRESHOLD: float    = _cfg["detection"]["medium_confidence_threshold"]
PARTIAL_COMPLIANCE_CONF_LOW: float    = _cfg["detection"]["partial_compliance_conf_low"]
PARTIAL_COMPLIANCE_CONF_HIGH: float   = _cfg["detection"]["partial_compliance_conf_high"]
VIOLATION_CONF_MIN: float             = _cfg["detection"]["violation_conf_min"]

# ── Bounding box thresholds ───────────────────────────────────────────────────
FAR_FIELD_BBOX_THRESHOLD_PX: int      = _cfg["bbox"]["far_field_height_px"]
RULE_CONFIDENCE_BBOX_NORMALISE: float = _cfg["bbox"]["rule_confidence_bbox_normalise"]
CROWD_IOU_THRESHOLD: float            = _cfg["bbox"]["crowd_iou_threshold"]
PPE_PERSON_OVERLAP_IOU: float         = _cfg["bbox"]["ppe_person_overlap_iou"]
PERSON_CROP_HEAD_EXPAND: float        = _cfg["bbox"]["person_crop_head_expand"]
PPE_ABOVE_PERSON_X_OVERLAP: float     = _cfg["bbox"]["ppe_above_person_x_overlap"]
NO_HELMET_MAX_BODY_FRACTION: float    = _cfg["bbox"]["no_helmet_max_body_fraction"]

# ── Frame geometry ────────────────────────────────────────────────────────────
ELEVATION_ZONE_RATIO: float           = _cfg["frame"]["elevation_zone_ratio"]
VISITOR_EDGE_PX: int                  = _cfg["frame"]["visitor_edge_px"]

# ── rule_confidence weights ───────────────────────────────────────────────────
WEIGHT_DETECTION_CONF: float          = _cfg["confidence_weights"]["detection_conf"]
WEIGHT_BBOX_SIZE: float               = _cfg["confidence_weights"]["bbox_size_factor"]
WEIGHT_EDGE: float                    = _cfg["confidence_weights"]["edge_factor"]

# ── Compliance score ──────────────────────────────────────────────────────────
SCORE_CRITICAL_DEDUCTION: int         = _cfg["score"]["critical_deduction"]
SCORE_HIGH_DEDUCTION: int             = _cfg["score"]["high_deduction"]
SCORE_WARNING_DEDUCTION: int          = _cfg["score"]["warning_deduction"]
CROWD_MULTIPLIER: float               = _cfg["score"]["crowd_multiplier"]
TEMPORAL_DECAY_INTERVAL_S: float      = _cfg["score"]["temporal_decay_interval_s"]
TEMPORAL_DECAY_RATE: float            = _cfg["score"]["temporal_decay_rate"]

# ── Score bands ───────────────────────────────────────────────────────────────
SCORE_COMPLIANT_MIN: int              = _cfg["score_bands"]["compliant_min"]
SCORE_CAUTION_MIN: int                = _cfg["score_bands"]["caution_min"]
SCORE_AT_RISK_MIN: int                = _cfg["score_bands"]["at_risk_min"]

# ── Crowd rule ────────────────────────────────────────────────────────────────
CROWD_MIN_WORKERS: int                = _cfg["crowd"]["min_workers"]
CROWD_VIOLATION_RATIO: float          = _cfg["crowd"]["violation_ratio_threshold"]
CROWD_VIOLATION_CONF_MIN: float       = _cfg["crowd"]["crowd_violation_conf_min"]

# ── Temporal analysis ─────────────────────────────────────────────────────────
TEMPORAL_WINDOW_FRAMES: int           = _cfg["temporal"]["window_frames"]
SUSTAINED_RATIO: float                = _cfg["temporal"]["sustained_ratio"]
INTERMITTENT_RATIO_LOW: float         = _cfg["temporal"]["intermittent_ratio_low"]
SCORE_HISTORY_INTERVAL_S: float       = _cfg["temporal"]["score_history_interval_s"]

# ── Scene heuristic ───────────────────────────────────────────────────────────
SKY_SAMPLE_TOP_RATIO: float           = _cfg["scene"]["sky_sample_top_ratio"]
SKY_BLUE_HUE_LOW: int                 = _cfg["scene"]["sky_blue_hue_low"]
SKY_BLUE_HUE_HIGH: int                = _cfg["scene"]["sky_blue_hue_high"]
SKY_SATURATION_MIN: int               = _cfg["scene"]["sky_saturation_min"]
SKY_PIXEL_RATIO_OUTDOOR: float        = _cfg["scene"]["sky_pixel_ratio_outdoor"]

# ── Classes ───────────────────────────────────────────────────────────────────
CLASS_NAMES: list[str] = list(_cfg["classes"].keys())
CLASS_IDS: dict[str, int] = _cfg["classes"]
VIOLATION_CLASSES: list[str] = _cfg["violation_classes"]
SAFE_CLASSES: list[str] = _cfg["safe_classes"]
