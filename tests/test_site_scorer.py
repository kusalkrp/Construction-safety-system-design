"""Unit tests for SiteScorer — score formula, bands, temporal decay, trend."""

import time

import pytest

from inference.safety_checker import ViolationReport
from inference.site_scorer import SiteScorer


def make_report(severity: str, violations: list[str] | None = None, ts: float | None = None) -> ViolationReport:
    return ViolationReport(
        worker_id=0,
        bbox=(0, 0, 100, 200),
        violations=violations or (["no_helmet"] if "CRITICAL" in severity else ["no_vest"]),
        detection_confidence=0.85,
        rule_confidence=0.80,
        confidence_tier="HIGH",
        severity=severity,
        scene_context="outdoor",
        human_readable="test",
        recommended_action="test",
        timestamp=ts or time.time(),
    )


def test_perfect_score_no_violations():
    scorer = SiteScorer()
    result = scorer.compute([])
    assert result.score == 100
    assert result.band == "COMPLIANT"


def test_critical_violation_deducts_25():
    scorer = SiteScorer()
    reports = [make_report("CRITICAL")]
    result = scorer.compute(reports)
    assert result.score == 75
    assert result.n_critical == 1


def test_high_violation_deducts_15():
    scorer = SiteScorer()
    result = scorer.compute([make_report("HIGH")])
    assert result.score == 85


def test_warning_violation_deducts_5():
    scorer = SiteScorer()
    result = scorer.compute([make_report("WARNING")])
    assert result.score == 95


def test_score_clamped_to_zero():
    scorer = SiteScorer()
    reports = [make_report("CRITICAL")] * 10  # 10 × 25 = 250 deduction
    result = scorer.compute(reports)
    assert result.score == 0


def test_crowd_multiplier_applied():
    scorer = SiteScorer()
    reports = [make_report("CRITICAL")]       # -25 deduction
    result_no_crowd = scorer.compute(reports, site_alert_triggered=False)
    scorer2 = SiteScorer()
    result_crowd = scorer2.compute(reports, site_alert_triggered=True)
    assert result_crowd.score < result_no_crowd.score
    assert result_crowd.crowd_triggered is True


def test_score_band_compliant():
    scorer = SiteScorer()
    assert scorer._get_band(100)[0] == "COMPLIANT"
    assert scorer._get_band(80)[0] == "COMPLIANT"


def test_score_band_caution():
    scorer = SiteScorer()
    assert scorer._get_band(79)[0] == "CAUTION"
    assert scorer._get_band(60)[0] == "CAUTION"


def test_score_band_at_risk():
    scorer = SiteScorer()
    assert scorer._get_band(59)[0] == "AT RISK"
    assert scorer._get_band(40)[0] == "AT RISK"


def test_score_band_critical():
    scorer = SiteScorer()
    assert scorer._get_band(39)[0] == "CRITICAL"
    assert scorer._get_band(0)[0] == "CRITICAL"


def test_visitors_and_unverifiable_excluded():
    """Visitors and UNVERIFIABLE workers should not affect the score."""
    scorer = SiteScorer()
    visitor = make_report("INFO")
    visitor.is_visitor = True
    unverifiable = make_report("UNVERIFIABLE", violations=[])
    result = scorer.compute([visitor, unverifiable])
    assert result.score == 100


def test_temporal_decay_reduces_old_violations():
    scorer = SiteScorer()
    old_ts = time.time() - 120  # 2 minutes ago — past 1-minute decay interval
    old_report = make_report("CRITICAL", ts=old_ts)
    result = scorer.compute([old_report])
    # Score should be better than if the violation were fresh (-25)
    assert result.score > 75


def test_score_display_format():
    scorer = SiteScorer()
    result = scorer.compute([])
    assert "/" in result.display
    assert "—" in result.display
