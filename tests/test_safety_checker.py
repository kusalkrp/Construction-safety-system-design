"""
Unit tests for SafetyChecker — all 6 rules, edge cases, and ViolationReport structure.
"""

import pytest

from inference.safety_checker import Detection, SafetyChecker, compute_iou, compute_rule_confidence


def make_person(x1=100, y1=100, x2=200, y2=300, conf=0.90) -> Detection:
    return Detection(class_name="person", confidence=conf, x1=x1, y1=y1, x2=x2, y2=y2)


def make_ppe(class_name: str, x1=110, y1=110, x2=190, y2=200, conf=0.80) -> Detection:
    return Detection(class_name=class_name, confidence=conf, x1=x1, y1=y1, x2=x2, y2=y2)


# ── compute_iou ───────────────────────────────────────────────────────────────

def test_iou_identical_boxes():
    assert compute_iou((0, 0, 100, 100), (0, 0, 100, 100)) == pytest.approx(1.0)


def test_iou_no_overlap():
    assert compute_iou((0, 0, 50, 50), (60, 60, 100, 100)) == pytest.approx(0.0)


def test_iou_partial_overlap():
    iou = compute_iou((0, 0, 100, 100), (50, 50, 150, 150))
    assert 0.0 < iou < 1.0


# ── compute_rule_confidence ───────────────────────────────────────────────────

def test_rule_confidence_centred_large_box():
    conf = compute_rule_confidence(
        detection_conf=0.90,
        bbox_height=120,
        bbox_x_center=320,
        frame_width=640,
    )
    assert conf > 0.80


def test_rule_confidence_small_edge_box():
    conf = compute_rule_confidence(
        detection_conf=0.90,
        bbox_height=20,    # small
        bbox_x_center=620,  # near edge
        frame_width=640,
    )
    assert conf < 0.70


# ── Rule 4 — Far-field worker ─────────────────────────────────────────────────

def test_rule4_far_field_unverifiable():
    checker = SafetyChecker(frame_width=640, frame_height=640)
    # person bbox height = 30px < 40px threshold
    person = Detection(class_name="person", confidence=0.85, x1=100, y1=200, x2=160, y2=230)
    report = checker.analyse([person]).worker_reports[0]
    assert report.severity == "UNVERIFIABLE"
    assert report.rule_confidence == 0.0
    assert report.violations == []


# ── Rule 1 — No helmet ────────────────────────────────────────────────────────

def test_rule1_no_helmet_critical():
    checker = SafetyChecker(frame_width=640, frame_height=640)
    person = make_person(x1=100, y1=300, x2=200, y2=500)  # height=200, lower half
    no_helmet = make_ppe("no_helmet", x1=110, y1=310, x2=190, y2=380, conf=0.80)
    report = checker.analyse([person, no_helmet]).worker_reports[0]
    assert "no_helmet" in report.violations
    assert report.severity == "CRITICAL"


def test_rule1_no_helmet_elevated():
    checker = SafetyChecker(frame_width=640, frame_height=640)
    # Person in upper 40% of frame (y1=50 < 640*0.40=256)
    person = make_person(x1=100, y1=50, x2=200, y2=250)  # height=200
    no_helmet = make_ppe("no_helmet", x1=110, y1=60, x2=190, y2=130, conf=0.80)
    report = checker.analyse([person, no_helmet]).worker_reports[0]
    assert report.severity == "CRITICAL-ELEVATED"


def test_rule1_low_confidence_no_alert():
    """no_helmet below VIOLATION_CONF_MIN should NOT trigger Rule 1."""
    checker = SafetyChecker(frame_width=640, frame_height=640)
    person = make_person()
    no_helmet = make_ppe("no_helmet", conf=0.30)  # below 0.50 threshold
    report = checker.analyse([person, no_helmet]).worker_reports[0]
    assert "no_helmet" not in report.violations


# ── Rule 2 — No vest (context-aware) ─────────────────────────────────────────

def test_rule2_no_vest_outdoor_high():
    checker = SafetyChecker(frame_width=640, frame_height=640)
    person = make_person()
    no_vest = make_ppe("no_vest", conf=0.75)
    site_report = checker.analyse([person, no_vest], frame_rgb=None)
    # Without frame, defaults to outdoor
    report = site_report.worker_reports[0]
    assert "no_vest" in report.violations
    assert report.severity == "HIGH"


# ── Rule 3 — Partial compliance ───────────────────────────────────────────────

def test_rule3_partial_compliance_warning():
    checker = SafetyChecker(frame_width=640, frame_height=640)
    person = make_person()
    # Confidence in the partial-compliance range 0.35–0.65
    partial = make_ppe("helmet_on", conf=0.50)
    report = checker.analyse([person, partial]).worker_reports[0]
    assert "partial_compliance" in report.violations
    assert report.severity == "WARNING"


# ── Rule 5 — Occlusion / PPE detection gap ────────────────────────────────────

def test_rule5_no_ppe_detected_warning():
    checker = SafetyChecker(frame_width=640, frame_height=640)
    person = make_person()
    # No PPE detections at all
    report = checker.analyse([person]).worker_reports[0]
    assert "ppe_gap" in report.violations
    assert report.severity == "WARNING"


# ── Rule 6 — Site-level crowd ─────────────────────────────────────────────────

def test_rule6_triggers_with_majority_violations():
    checker = SafetyChecker(frame_width=640, frame_height=640)
    # 5 persons, all in lower half (not far-field, not elevated)
    persons = [
        Detection(class_name="person", confidence=0.90, x1=i*100, y1=300, x2=i*100+80, y2=500)
        for i in range(5)
    ]
    # All have no_helmet with high confidence
    ppe = [
        Detection(class_name="no_helmet", confidence=0.85, x1=i*100+5, y1=310, x2=i*100+75, y2=380)
        for i in range(5)
    ]
    site_report = checker.analyse(persons + ppe)
    assert site_report.site_alert is not None
    assert site_report.site_alert.severity == "SITE ALERT"


def test_rule6_does_not_trigger_below_threshold():
    checker = SafetyChecker(frame_width=640, frame_height=640)
    # Only 3 workers — below CROWD_MIN_WORKERS=4
    persons = [
        Detection(class_name="person", confidence=0.90, x1=i*100, y1=300, x2=i*100+80, y2=500)
        for i in range(3)
    ]
    site_report = checker.analyse(persons)
    assert site_report.site_alert is None


# ── Visitor heuristic ─────────────────────────────────────────────────────────

def test_visitor_near_frame_edge():
    checker = SafetyChecker(frame_width=640, frame_height=640)
    # Person within VISITOR_EDGE_PX=30 of left edge
    person = Detection(class_name="person", confidence=0.85, x1=5, y1=200, x2=80, y2=400)
    report = checker.analyse([person]).worker_reports[0]
    assert report.is_visitor is True
    assert report.severity == "INFO"


# ── Compliant worker ──────────────────────────────────────────────────────────

def test_compliant_worker_no_violations():
    checker = SafetyChecker(frame_width=640, frame_height=640)
    person = make_person()
    helmet = make_ppe("helmet_on", conf=0.90)
    vest = make_ppe("vest_on", conf=0.90)
    report = checker.analyse([person, helmet, vest]).worker_reports[0]
    assert report.violations == []
    assert report.severity == "COMPLIANT"
