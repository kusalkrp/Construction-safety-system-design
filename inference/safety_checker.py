"""
SafetyChecker — applies the 6 formal safety rules to YOLO detection outputs
and produces a ViolationReport per detected worker.

This class does NOT call YOLO. It receives detection outputs (boxes, classes,
confidences) and applies compliance logic only. This makes it independently
testable and decoupled from the model implementation.

Rules:
  1 — No helmet in active zone (CRITICAL / CRITICAL-ELEVATED)
  2 — No vest, context-aware indoor/outdoor (HIGH / WARNING)
  3 — Partial compliance, conf 0.35–0.65 (WARNING → review queue)
  4 — Far-field worker, bbox < 40px (UNVERIFIABLE)
  5 — Occlusion / PPE detection gap (WARNING → review queue)
  6 — Site-level crowd non-compliance, ≥4 workers ≥50% violating (SITE ALERT)
"""

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from inference.constants import (
    CROWD_MIN_WORKERS,
    CROWD_VIOLATION_CONF_MIN,
    CROWD_VIOLATION_RATIO,
    ELEVATION_ZONE_RATIO,
    FAR_FIELD_BBOX_THRESHOLD_PX,
    HIGH_CONFIDENCE_THRESHOLD,
    MEDIUM_CONFIDENCE_THRESHOLD,
    NO_HELMET_MAX_BODY_FRACTION,
    PARTIAL_COMPLIANCE_CONF_HIGH,
    PARTIAL_COMPLIANCE_CONF_LOW,
    PERSON_CROP_HEAD_EXPAND,
    PPE_ABOVE_PERSON_X_OVERLAP,
    PPE_PERSON_OVERLAP_IOU,
    RULE_CONFIDENCE_BBOX_NORMALISE,
    VIOLATION_CONF_MIN,
    VISITOR_EDGE_PX,
    WEIGHT_BBOX_SIZE,
    WEIGHT_DETECTION_CONF,
    WEIGHT_EDGE,
)
from inference.scene_classifier import classify_scene

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single YOLO detection box."""
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class ViolationReport:
    """
    Structured compliance report for a single detected worker.

    Produced by SafetyChecker.check_worker(). Contains all information
    needed by SiteScorer (severities + timestamp) and Annotator (bbox + tier).
    """
    worker_id: int
    bbox: tuple[int, int, int, int]          # x1, y1, x2, y2
    violations: list[str]
    detection_confidence: float
    rule_confidence: float
    confidence_tier: str                      # "HIGH" | "MEDIUM" | "LOW"
    severity: str                             # "CRITICAL" | "CRITICAL-ELEVATED" | "HIGH" | "WARNING" | "UNVERIFIABLE"
    scene_context: str                        # "outdoor" | "indoor"
    human_readable: str
    recommended_action: str
    timestamp: float = field(default_factory=time.time)
    is_visitor: bool = False
    is_site_alert: bool = False
    violation_bbox: tuple[int, int, int, int] | None = None  # PPE detection box (for annotator)


@dataclass
class SiteReport:
    """Aggregated result for a full frame."""
    worker_reports: list[ViolationReport]
    site_alert: ViolationReport | None
    frame_width: int
    frame_height: int
    scene_context: str
    timestamp: float = field(default_factory=time.time)


def compute_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    """Compute intersection-over-union for two (x1,y1,x2,y2) boxes."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, (xa2 - xa1) * (ya2 - ya1))
    area_b = max(0, (xb2 - xb1) * (yb2 - yb1))
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def expand_crop_for_ppe(
    bbox: tuple[int, int, int, int],
    frame_height: int,
    frame_width: int,
) -> tuple[int, int, int, int]:
    """
    Expand a person bounding box upward by PERSON_CROP_HEAD_EXPAND × box height.

    When the person detector anchors on the torso, the head region sits above y1.
    Expanding upward before PPE IoU matching ensures a helmet bbox above the torso
    is still associated with the correct worker.

    The original bbox is kept for ViolationReport annotation — this expanded box
    is only used for PPE overlap search.
    """
    x1, y1, x2, y2 = bbox
    box_height = y2 - y1
    y1_expanded = max(0, y1 - int(box_height * PERSON_CROP_HEAD_EXPAND))
    return x1, y1_expanded, x2, y2


def is_above_person(
    ppe_bbox: tuple[int, int, int, int],
    person_bbox: tuple[int, int, int, int],
) -> bool:
    """
    Return True when a PPE box sits directly above a person box with sufficient
    horizontal overlap — catches helmets that the IoU check misses entirely
    because the person detector anchored too low (chin/torso level).

    Conditions:
      - PPE box bottom (y2) is at or above the person box top (y1)
      - Horizontal overlap between the two boxes >= PPE_ABOVE_PERSON_X_OVERLAP
        of the PPE box width (30% default)
    """
    px1, py1, px2, _ = person_bbox
    hx1, _, hx2, hy2 = ppe_bbox

    if hy2 > py1:           # PPE bottom below person top — not above
        return False

    overlap_x = max(0, min(hx2, px2) - max(hx1, px1))
    ppe_width  = max(1, hx2 - hx1)
    return (overlap_x / ppe_width) >= PPE_ABOVE_PERSON_X_OVERLAP


def compute_rule_confidence(
    detection_conf: float,
    bbox_height: int,
    bbox_x_center: float,
    frame_width: int,
) -> float:
    """
    Compute rule_confidence from three signals.

    detection_conf: raw YOLO confidence
    bbox_height: height of the person bounding box in pixels
    bbox_x_center: x center of bbox (0–frame_width)
    frame_width: full frame width in pixels
    """
    bbox_size_factor = min(1.0, bbox_height / RULE_CONFIDENCE_BBOX_NORMALISE)
    edge_proximity_ratio = abs(bbox_x_center / frame_width - 0.5) * 2.0  # 0=centre, 1=edge
    edge_factor = 1.0 - edge_proximity_ratio

    return (
        WEIGHT_DETECTION_CONF * detection_conf
        + WEIGHT_BBOX_SIZE * bbox_size_factor
        + WEIGHT_EDGE * edge_factor
    )


def get_confidence_tier(rule_confidence: float) -> str:
    if rule_confidence >= HIGH_CONFIDENCE_THRESHOLD:
        return "HIGH"
    if rule_confidence >= MEDIUM_CONFIDENCE_THRESHOLD:
        return "MEDIUM"
    return "LOW"


class SafetyChecker:
    """
    Applies safety rules to YOLO detection outputs and produces
    ViolationReports per detected worker.

    Rules are loaded from rules.yaml (via constants.py) at import time.
    The checker does not perform model inference — it receives detection
    outputs and applies compliance logic only.

    Args:
        frame_width: Width of the input frame in pixels.
        frame_height: Height of the input frame in pixels.
    """

    def __init__(self, frame_width: int = 640, frame_height: int = 640) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height

    def analyse(
        self,
        detections: list[Detection],
        frame_rgb: "np.ndarray | None" = None,
    ) -> SiteReport:
        """
        Analyse all detections for a single frame.

        Args:
            detections: All YOLO detections for the frame.
            frame_rgb: Original frame (H,W,3) RGB — used for scene classification heuristic.

        Returns:
            SiteReport with per-worker ViolationReports and optional site-level alert.
        """
        scene_context = classify_scene(frame_rgb) if frame_rgb is not None else "outdoor"

        person_detections = [d for d in detections if d.class_name == "person"]
        ppe_detections = [d for d in detections if d.class_name != "person"]

        worker_reports: list[ViolationReport] = []
        for worker_id, person in enumerate(person_detections):
            report = self._check_worker(worker_id, person, ppe_detections, scene_context)
            worker_reports.append(report)

        # Fallback: if no person boxes detected, create violation reports from
        # standalone no_helmet / no_vest detections above the violation confidence threshold.
        # This handles the common case where the person class is not detected but PPE
        # violation classes are clearly visible.
        if not person_detections:
            helmet_on_dets = [d for d in ppe_detections if d.class_name == "helmet_on"]
            vest_on_dets   = [d for d in ppe_detections if d.class_name == "vest_on"]

            def _suppression_reason(det: Detection, safe_dets: list[Detection]) -> str | None:
                """
                Returns:
                  None          — not suppressed; generate violation report
                  "anatomical"  — box in physically impossible location; silently discard
                  "conflict"    — safe class provides contradicting evidence; generate
                                  partial_compliance WARNING instead of violation
                """
                det_bbox = (det.x1, det.y1, det.x2, det.y2)
                det_center_y = (det.y1 + det.y2) / 2

                # Anatomical impossibility — no safe_det needed
                if det.class_name == "no_vest" and det_center_y < self.frame_height * 0.25:
                    logger.debug(
                        "Fallback suppressed no_vest — center y=%.0f is in head zone "
                        "(top 25%% of %dpx frame)", det_center_y, self.frame_height,
                    )
                    return "anatomical"
                if det.class_name == "no_helmet" and det_center_y > self.frame_height * 0.60:
                    logger.debug(
                        "Fallback suppressed no_helmet — center y=%.0f is in lower body zone "
                        "(below 60%% of %dpx frame)", det_center_y, self.frame_height,
                    )
                    return "anatomical"

                for safe in safe_dets:
                    safe_bbox = (safe.x1, safe.y1, safe.x2, safe.y2)
                    iou_match = compute_iou(det_bbox, safe_bbox) > PPE_PERSON_OVERLAP_IOU

                    if iou_match and safe.confidence > det.confidence:
                        logger.debug(
                            "Fallback conflict: %s (conf %.2f) vs %s (conf %.2f) — "
                            "safe class wins on IoU; generating partial_compliance",
                            det.class_name, det.confidence, safe.class_name, safe.confidence,
                        )
                        return "conflict"

                    if is_above_person(safe_bbox, det_bbox):
                        logger.debug(
                            "Fallback conflict: %s directly above %s — "
                            "spatial evidence overrides confidence; generating partial_compliance",
                            safe.class_name, det.class_name,
                        )
                        return "conflict"

                return None

            worker_id = 0
            for det in [
                d for d in ppe_detections
                if d.class_name in ("no_helmet", "no_vest")
                and d.confidence >= VIOLATION_CONF_MIN
            ]:
                safe_dets = helmet_on_dets if det.class_name == "no_helmet" else vest_on_dets
                reason = _suppression_reason(det, safe_dets)
                if reason is None:
                    worker_reports.append(self._report_from_violation_det(worker_id, det, scene_context))
                    worker_id += 1
                elif reason == "conflict":
                    worker_reports.append(self._report_conflict_fallback(worker_id, det, scene_context))
                    worker_id += 1
                # "anatomical": silently discard

        site_alert = self._check_site_level(worker_reports)

        return SiteReport(
            worker_reports=worker_reports,
            site_alert=site_alert,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            scene_context=scene_context,
        )

    def _is_visitor(self, person: Detection) -> bool:
        """Heuristic: person near frame edge is likely a visitor, not an active worker."""
        return (
            person.x1 < VISITOR_EDGE_PX
            or person.x2 > self.frame_width - VISITOR_EDGE_PX
        )

    def _check_worker(
        self,
        worker_id: int,
        person: Detection,
        ppe_detections: list[Detection],
        scene_context: str,
    ) -> ViolationReport:
        """Apply Rules 1–5 to a single worker."""
        person_bbox = (person.x1, person.y1, person.x2, person.y2)
        bbox_x_center = (person.x1 + person.x2) / 2.0

        # ── Rule 4 — Far-field worker ─────────────────────────────────────────
        if person.height < FAR_FIELD_BBOX_THRESHOLD_PX:
            return ViolationReport(
                worker_id=worker_id,
                bbox=person_bbox,
                violations=[],
                detection_confidence=person.confidence,
                rule_confidence=0.0,
                confidence_tier="LOW",
                severity="UNVERIFIABLE",
                scene_context=scene_context,
                human_readable=(
                    f"[UNVERIFIABLE] Worker #{worker_id} — too distant to assess PPE state. "
                    f"Bounding box height {person.height}px < {FAR_FIELD_BBOX_THRESHOLD_PX}px threshold."
                ),
                recommended_action="PPE_STATE_UNVERIFIABLE — worker at distance. No alert generated. "
                                   "Production fix: PTZ camera with auto-zoom.",
            )

        # ── Visitor heuristic ─────────────────────────────────────────────────
        if self._is_visitor(person):
            return ViolationReport(
                worker_id=worker_id,
                bbox=person_bbox,
                violations=[],
                detection_confidence=person.confidence,
                rule_confidence=0.0,
                confidence_tier="LOW",
                severity="INFO",
                scene_context=scene_context,
                human_readable=f"[INFO] Person #{worker_id} near site perimeter — classified as visitor.",
                recommended_action="VISITOR_DETECTED — PPE rules not applied. Monitor if entering active zone.",
                is_visitor=True,
            )

        # Find PPE detections overlapping this worker.
        # Two association paths:
        #   1. IoU with expanded bbox — person detector often anchors on the torso;
        #      expanding upward 60% of box height captures most head regions.
        #   2. is_above_person — catches helmets that sit entirely above the expanded
        #      boundary (close-up images where the box starts at chin level).
        ppe_search_bbox = expand_crop_for_ppe(person_bbox, self.frame_height, self.frame_width)
        ppe_det_bbox = lambda d: (d.x1, d.y1, d.x2, d.y2)
        overlapping_ppe = [
            d for d in ppe_detections
            if compute_iou(ppe_search_bbox, ppe_det_bbox(d)) > PPE_PERSON_OVERLAP_IOU
            or is_above_person(ppe_det_bbox(d), person_bbox)
        ]

        ppe_class_names = {d.class_name for d in overlapping_ppe}
        # Fix: keep highest-confidence detection per class (last-wins dict would silently
        # replace a strong no_helmet with a weak one if iteration order puts the weak one last)
        ppe_by_class: dict[str, Detection] = {}
        for _d in overlapping_ppe:
            if _d.class_name not in ppe_by_class or _d.confidence > ppe_by_class[_d.class_name].confidence:
                ppe_by_class[_d.class_name] = _d

        violations: list[str] = []
        severity = "COMPLIANT"
        human_parts: list[str] = []
        action_parts: list[str] = []

        rule_conf = compute_rule_confidence(
            detection_conf=person.confidence,
            bbox_height=person.height,
            bbox_x_center=bbox_x_center,
            frame_width=self.frame_width,
        )

        # ── Rule 3 — Partial compliance ───────────────────────────────────────
        # Only fires for safety-critical required PPE (not bonus classes like mask_on).
        # Skips no_helmet/no_vest at conf >= VIOLATION_CONF_MIN — those are handled by
        # Rules 1/2 which produce the correct severity. Deduplicated: one partial_compliance
        # entry regardless of how many borderline detections are present.
        _REQUIRED_PPE = {"helmet_on", "vest_on", "no_helmet", "no_vest"}
        for ppe_det in overlapping_ppe:
            if ppe_det.class_name not in _REQUIRED_PPE:
                continue
            if ppe_det.class_name in ("no_helmet", "no_vest") and ppe_det.confidence >= VIOLATION_CONF_MIN:
                continue  # will be handled by Rule 1/2 at correct severity
            if PARTIAL_COMPLIANCE_CONF_LOW <= ppe_det.confidence <= PARTIAL_COMPLIANCE_CONF_HIGH:
                if "partial_compliance" not in violations:
                    violations.append("partial_compliance")
                    severity = "WARNING"
                    action_parts.append("PARTIAL_COMPLIANCE_SUSPECTED — add to human review queue.")
                human_parts.append(
                    f"{ppe_det.class_name} detected with borderline confidence "
                    f"({ppe_det.confidence:.2f}) — possible partial wear or occlusion."
                )

        # ── Rule 1 — No helmet ────────────────────────────────────────────────
        if "no_helmet" in ppe_class_names:
            no_helmet_det = ppe_by_class["no_helmet"]
            helmet_on_det = ppe_by_class.get("helmet_on")

            # Anatomical position check: no_helmet box center must be in the upper
            # NO_HELMET_MAX_BODY_FRACTION of the person bbox. A no_helmet detection
            # whose center sits in the torso/vest region is a mislocalized YOLO box
            # (possibly the model confusing vest texture with a head region).
            # For above-person detections (is_above_person path), person.y1 is the
            # reference — anything above that is always valid.
            no_helmet_center_y = (no_helmet_det.y1 + no_helmet_det.y2) / 2
            person_head_boundary = person.y1 + person.height * NO_HELMET_MAX_BODY_FRACTION
            no_helmet_in_head_region = (
                no_helmet_center_y <= person_head_boundary   # within person box, upper half
                or no_helmet_det.y2 <= person.y1             # fully above person box
            )

            if not no_helmet_in_head_region:
                logger.debug(
                    "Worker #%d: no_helmet box center y=%.0f is below head boundary y=%.0f — "
                    "mislocalized detection rejected",
                    worker_id, no_helmet_center_y, person_head_boundary,
                )
            # Conflict suppression: if helmet_on is detected with higher confidence
            # than no_helmet on the same worker, the model is contradicting itself.
            # Downgrade to partial_compliance (WARNING) rather than firing CRITICAL.
            elif (
                no_helmet_det.confidence >= VIOLATION_CONF_MIN
                and not (helmet_on_det and helmet_on_det.confidence > no_helmet_det.confidence)
            ):
                violations.append("no_helmet")
                is_elevated = person.y1 < self.frame_height * (1 - ELEVATION_ZONE_RATIO)
                current_severity = "CRITICAL-ELEVATED" if is_elevated else "CRITICAL"
                if severity not in ("CRITICAL-ELEVATED", "CRITICAL"):
                    severity = current_severity
                elif current_severity == "CRITICAL-ELEVATED":
                    severity = current_severity

                location_note = "elevated position (upper frame region)" if is_elevated else "site floor"
                human_parts.append(
                    f"No helmet detected. Location: {location_note}. "
                    f"Detection confidence: {no_helmet_det.confidence:.2f}."
                )
                action_parts.append(
                    "Immediate supervisor intervention required. "
                    "Worker must not continue work at height without head protection."
                    if is_elevated else
                    "Supervisor notification required. Worker must wear helmet in active zone."
                )
            elif no_helmet_det.confidence >= VIOLATION_CONF_MIN and helmet_on_det:
                # Conflicting signals — helmet_on outscores no_helmet — flag for review
                violations.append("partial_compliance")
                if severity == "COMPLIANT":
                    severity = "WARNING"
                human_parts.append(
                    f"Conflicting helmet signals: helmet_on ({helmet_on_det.confidence:.2f}) "
                    f"vs no_helmet ({no_helmet_det.confidence:.2f}) — helmet status uncertain."
                )
                action_parts.append("CONFLICTING_PPE_SIGNAL — add to human review queue.")

        # ── Rule 2 — No vest (context-aware) ─────────────────────────────────
        if "no_vest" in ppe_class_names:
            no_vest_det = ppe_by_class["no_vest"]
            vest_on_det = ppe_by_class.get("vest_on")
            # Conflict suppression: vest_on higher confidence than no_vest → review queue
            if no_vest_det.confidence >= VIOLATION_CONF_MIN and (
                vest_on_det and vest_on_det.confidence > no_vest_det.confidence
            ):
                violations.append("partial_compliance")
                if severity == "COMPLIANT":
                    severity = "WARNING"
                human_parts.append(
                    f"Conflicting vest signals: vest_on ({vest_on_det.confidence:.2f}) "
                    f"vs no_vest ({no_vest_det.confidence:.2f}) — vest status uncertain."
                )
                action_parts.append("CONFLICTING_PPE_SIGNAL — add to human review queue.")
            elif no_vest_det.confidence >= VIOLATION_CONF_MIN:
                violations.append("no_vest")
                vest_severity = "HIGH" if scene_context == "outdoor" else "WARNING"
                if severity not in ("CRITICAL-ELEVATED", "CRITICAL"):
                    severity = vest_severity

                human_parts.append(
                    f"No hi-vis vest detected ({scene_context} scene). "
                    f"Detection confidence: {no_vest_det.confidence:.2f}. "
                    f"Risk: {'vehicle / machinery strike exposure' if scene_context == 'outdoor' else 'lower but non-zero hazard'}."
                )
                action_parts.append(
                    "Worker must wear high-visibility vest. Vehicle strike risk in outdoor zone."
                    if scene_context == "outdoor" else
                    "Worker should wear hi-vis vest. Logged for supervisor review."
                )

        # ── Rule 5 — Occlusion / PPE detection gap ────────────────────────────
        # Fires when: (a) no PPE at all, (b) helmet detected but no vest-class,
        # or (c) vest detected but no helmet-class. Each PPE item is assessed
        # independently — presence of one does not confirm the other.
        has_helmet_class = any(
            d.class_name in ("helmet_on", "no_helmet") for d in overlapping_ppe
        )
        has_vest_class = any(
            d.class_name in ("vest_on", "no_vest") for d in overlapping_ppe
        )
        if not violations and (
            not overlapping_ppe
            or not has_helmet_class
            or not has_vest_class
        ):
            violations.append("ppe_gap")
            if severity == "COMPLIANT":
                severity = "WARNING"
            if not overlapping_ppe:
                human_parts.append(
                    "No PPE boxes overlap this worker — possible occlusion or detection gap."
                )
            elif not has_helmet_class:
                human_parts.append(
                    "Vest detected but no helmet-class associated — helmet state unverifiable."
                )
            else:
                human_parts.append(
                    "Helmet detected but no vest-class associated — vest state unverifiable."
                )
            action_parts.append(
                "OCCLUDED_WORKER — PPE state unknown. Human review recommended before clearing."
            )

        if not violations:
            severity = "COMPLIANT"

        confidence_tier = get_confidence_tier(rule_conf)
        worker_label = f"Worker #{worker_id}"

        if violations:
            severity_tag = f"[{severity}]"
            violation_str = ", ".join(v for v in violations if v != "partial_compliance")
            hr = (
                f"{severity_tag} {worker_label} — {violation_str or 'PPE state uncertain'}\n"
                + "\n".join(f"  {p}" for p in human_parts)
                + f"\n  Rule confidence: {confidence_tier} ({rule_conf:.2f})"
                + f"\n  Scene context: {scene_context} construction site"
            )
            action = " ".join(action_parts)
        else:
            hr = f"[COMPLIANT] {worker_label} — all PPE requirements met."
            action = "No action required."

        # Capture the PPE violation box for the annotator to draw
        primary_violation_det = (
            ppe_by_class.get("no_helmet") if "no_helmet" in violations
            else ppe_by_class.get("no_vest") if "no_vest" in violations
            else None
        )
        v_bbox = (
            (primary_violation_det.x1, primary_violation_det.y1,
             primary_violation_det.x2, primary_violation_det.y2)
            if primary_violation_det else None
        )

        return ViolationReport(
            worker_id=worker_id,
            bbox=person_bbox,
            violations=violations,
            detection_confidence=person.confidence,
            rule_confidence=rule_conf,
            confidence_tier=confidence_tier,
            severity=severity,
            scene_context=scene_context,
            human_readable=hr,
            recommended_action=action,
            violation_bbox=v_bbox,
        )

    def _report_from_violation_det(
        self,
        worker_id: int,
        det: Detection,
        scene_context: str,
    ) -> ViolationReport:
        """
        Create a ViolationReport directly from a no_helmet / no_vest detection
        when no person box was detected as an anchor.
        """
        bbox = (det.x1, det.y1, det.x2, det.y2)
        bbox_x_center = (det.x1 + det.x2) / 2.0
        rule_conf = compute_rule_confidence(
            detection_conf=det.confidence,
            bbox_height=det.height,
            bbox_x_center=bbox_x_center,
            frame_width=self.frame_width,
        )
        confidence_tier = get_confidence_tier(rule_conf)

        if det.class_name == "no_helmet":
            is_elevated = det.y1 < self.frame_height * (1 - ELEVATION_ZONE_RATIO)
            severity = "CRITICAL-ELEVATED" if is_elevated else "CRITICAL"
            location_note = "elevated position" if is_elevated else "site floor"
            hr = (
                f"[{severity}] Worker #{worker_id} — no_helmet\n"
                f"  No helmet detected. Location: {location_note}. "
                f"Detection confidence: {det.confidence:.2f}.\n"
                f"  Rule confidence: {confidence_tier} ({rule_conf:.2f})\n"
                f"  Scene context: {scene_context} construction site\n"
                f"  Note: person class not detected — report derived from PPE box directly."
            )
            action = (
                "Immediate supervisor intervention required."
                if is_elevated else
                "Supervisor notification required. Worker must wear helmet in active zone."
            )
        else:  # no_vest
            vest_severity = "HIGH" if scene_context == "outdoor" else "WARNING"
            severity = vest_severity
            hr = (
                f"[{severity}] Worker #{worker_id} — no_vest\n"
                f"  No hi-vis vest detected ({scene_context} scene). "
                f"Detection confidence: {det.confidence:.2f}.\n"
                f"  Rule confidence: {confidence_tier} ({rule_conf:.2f})\n"
                f"  Scene context: {scene_context} construction site\n"
                f"  Note: person class not detected — report derived from PPE box directly."
            )
            action = (
                "Worker must wear high-visibility vest. Vehicle strike risk in outdoor zone."
                if scene_context == "outdoor" else
                "Worker should wear hi-vis vest. Logged for supervisor review."
            )

        return ViolationReport(
            worker_id=worker_id,
            bbox=bbox,
            violations=[det.class_name],
            detection_confidence=det.confidence,
            rule_confidence=rule_conf,
            confidence_tier=confidence_tier,
            severity=severity,
            scene_context=scene_context,
            human_readable=hr,
            recommended_action=action,
        )

    def _report_conflict_fallback(
        self,
        worker_id: int,
        det: Detection,
        scene_context: str,
    ) -> ViolationReport:
        """
        Partial compliance report for the fallback path when a safe class (helmet_on/vest_on)
        contradicts a violation detection via IoU or spatial evidence.
        Previously this case returned nothing (score=100 COMPLIANT) — now surfaces as WARNING.
        """
        bbox = (det.x1, det.y1, det.x2, det.y2)
        bbox_x_center = (det.x1 + det.x2) / 2.0
        rule_conf = compute_rule_confidence(
            detection_conf=det.confidence,
            bbox_height=det.height,
            bbox_x_center=bbox_x_center,
            frame_width=self.frame_width,
        )
        confidence_tier = get_confidence_tier(rule_conf)
        safe_class = "helmet_on" if det.class_name == "no_helmet" else "vest_on"
        hr = (
            f"[WARNING] Worker #{worker_id} — partial_compliance\n"
            f"  Conflicting signals: {det.class_name} (conf {det.confidence:.2f}) detected "
            f"but {safe_class} present in same region with higher confidence — PPE state uncertain.\n"
            f"  Rule confidence: {confidence_tier} ({rule_conf:.2f})\n"
            f"  Scene context: {scene_context} construction site\n"
            f"  Note: person class not detected — report derived from PPE box directly."
        )
        return ViolationReport(
            worker_id=worker_id,
            bbox=bbox,
            violations=["partial_compliance"],
            detection_confidence=det.confidence,
            rule_confidence=rule_conf,
            confidence_tier=confidence_tier,
            severity="WARNING",
            scene_context=scene_context,
            human_readable=hr,
            recommended_action="CONFLICTING_PPE_SIGNAL — add to human review queue.",
        )

    def _check_site_level(
        self, worker_reports: list[ViolationReport]
    ) -> ViolationReport | None:
        """
        Rule 6 — Site-level crowd non-compliance.
        Triggers when >= CROWD_MIN_WORKERS detected and >= 50% are violating.
        Returns a site-level ViolationReport or None.
        """
        active_workers = [r for r in worker_reports if not r.is_visitor and r.severity != "UNVERIFIABLE"]
        n_workers = len(active_workers)

        if n_workers < CROWD_MIN_WORKERS:
            return None

        violating = [
            r for r in active_workers
            if r.violations
            and r.rule_confidence >= CROWD_VIOLATION_CONF_MIN
            and r.severity not in ("COMPLIANT", "UNVERIFIABLE", "INFO")
        ]
        n_violating = len(violating)
        ratio = n_violating / n_workers

        if ratio < CROWD_VIOLATION_RATIO:
            return None

        # Tally violation types
        violation_counts: dict[str, int] = {}
        for r in violating:
            for v in r.violations:
                violation_counts[v] = violation_counts.get(v, 0) + 1
        violation_summary = ", ".join(
            f"{v} ×{c}" for v, c in sorted(violation_counts.items(), key=lambda x: -x[1])
        )

        return ViolationReport(
            worker_id=-1,
            bbox=(0, 0, 0, 0),
            violations=["site_level_crowd"],
            detection_confidence=1.0,
            rule_confidence=1.0,
            confidence_tier="HIGH",
            severity="SITE ALERT",
            scene_context=active_workers[0].scene_context if active_workers else "outdoor",
            human_readable=(
                f"[SITE ALERT] Systemic PPE non-compliance detected\n"
                f"  Workers detected: {n_workers}\n"
                f"  Workers non-compliant: {n_violating} ({ratio:.0%})\n"
                f"  Active violations: {violation_summary or 'unknown'}"
            ),
            recommended_action=(
                "Site-wide work stoppage for PPE briefing. "
                "Supervisor: review site PPE supply immediately."
            ),
            is_site_alert=True,
        )
