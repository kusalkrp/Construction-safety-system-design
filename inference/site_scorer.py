"""
SiteScorer — computes a 0–100 compliance score from a list of ViolationReports.

This class does NOT apply safety rules. It receives a list of ViolationReports
(produced by SafetyChecker) and aggregates them into a site-level score.
It does not know what the violations were — only their severities and timestamps.

Score formula:
    score = 100
          - (critical_violations × 25)
          - (high_violations × 15)
          - (warning_violations × 5)
          × crowd_multiplier (1.3 if Rule 6 triggered)
          × temporal_decay (violations > 60s decay 50% per minute)

Score bands:
    80–100  COMPLIANT  (green)
    60–79   CAUTION    (amber)
    40–59   AT RISK    (orange)
    0–39    CRITICAL   (red)
"""

import logging
import time
from dataclasses import dataclass, field

from inference.constants import (
    CROWD_MULTIPLIER,
    SCORE_AT_RISK_MIN,
    SCORE_CAUTION_MIN,
    SCORE_COMPLIANT_MIN,
    SCORE_CRITICAL_DEDUCTION,
    SCORE_HIGH_DEDUCTION,
    SCORE_WARNING_DEDUCTION,
    TEMPORAL_DECAY_INTERVAL_S,
    TEMPORAL_DECAY_RATE,
)
from inference.safety_checker import ViolationReport

logger = logging.getLogger(__name__)


@dataclass
class ScoreResult:
    """Output of SiteScorer.compute()."""
    score: int
    band: str
    band_colour: str
    n_critical: int
    n_high: int
    n_warning: int
    crowd_triggered: bool
    display: str          # e.g. "67/100 — CAUTION"


@dataclass
class SiteScorer:
    """
    Aggregates ViolationReports into a 0–100 compliance score with trend tracking.

    score_history stores (timestamp, score) tuples every SCORE_HISTORY_INTERVAL_S
    seconds for trend analysis over a video session.
    """

    score_history: list[tuple[float, int]] = field(default_factory=list)
    _last_history_ts: float = field(default_factory=time.time)

    def compute(
        self,
        worker_reports: list[ViolationReport],
        site_alert_triggered: bool = False,
    ) -> ScoreResult:
        """
        Compute compliance score for the current frame's reports.

        Args:
            worker_reports: Per-worker ViolationReports from SafetyChecker.
            site_alert_triggered: True if Rule 6 site-level crowd alert fired.

        Returns:
            ScoreResult with score, band, and display string.
        """
        now = time.time()

        n_critical = 0
        n_high = 0
        n_warning = 0

        for report in worker_reports:
            if report.is_visitor or report.is_site_alert:
                continue
            if report.severity in ("UNVERIFIABLE", "INFO", "COMPLIANT"):
                continue

            decay = self._compute_decay(report.timestamp, now)

            if report.severity in ("CRITICAL", "CRITICAL-ELEVATED"):
                n_critical += decay
            elif report.severity == "HIGH":
                n_high += decay
            elif report.severity == "WARNING":
                n_warning += decay

        deduction = (
            n_critical * SCORE_CRITICAL_DEDUCTION
            + n_high * SCORE_HIGH_DEDUCTION
            + n_warning * SCORE_WARNING_DEDUCTION
        )

        if site_alert_triggered:
            deduction *= CROWD_MULTIPLIER

        raw_score = 100 - deduction
        score = max(0, min(100, int(raw_score)))
        band, colour = self._get_band(score)

        result = ScoreResult(
            score=score,
            band=band,
            band_colour=colour,
            n_critical=int(n_critical),
            n_high=int(n_high),
            n_warning=int(n_warning),
            crowd_triggered=site_alert_triggered,
            display=f"{score}/100 — {band}",
        )

        self._record_history(score, now)
        return result

    def _compute_decay(self, violation_ts: float, now: float) -> float:
        """Apply temporal decay — violations older than interval decay 50% per minute."""
        age_s = now - violation_ts
        if age_s <= TEMPORAL_DECAY_INTERVAL_S:
            return 1.0
        intervals_past = (age_s - TEMPORAL_DECAY_INTERVAL_S) / TEMPORAL_DECAY_INTERVAL_S
        return (1.0 - TEMPORAL_DECAY_RATE) ** intervals_past

    def _get_band(self, score: int) -> tuple[str, str]:
        if score >= SCORE_COMPLIANT_MIN:
            return "COMPLIANT", "#00AA44"
        if score >= SCORE_CAUTION_MIN:
            return "CAUTION", "#FFAA00"
        if score >= SCORE_AT_RISK_MIN:
            return "AT RISK", "#FF6600"
        return "CRITICAL", "#FF2200"

    def _record_history(self, score: int, now: float) -> None:
        from inference.constants import SCORE_HISTORY_INTERVAL_S
        if now - self._last_history_ts >= SCORE_HISTORY_INTERVAL_S:
            self.score_history.append((now, score))
            self._last_history_ts = now

    def get_trend_summary(self) -> str:
        """Return a human-readable summary of score trend over the session."""
        if len(self.score_history) < 2:
            return "Insufficient history for trend analysis."
        scores = [s for _, s in self.score_history]
        avg = sum(scores) / len(scores)
        trend = scores[-1] - scores[0]
        direction = "improving" if trend > 5 else "worsening" if trend < -5 else "stable"
        return (
            f"Session trend: {direction} | "
            f"Average score: {avg:.0f}/100 | "
            f"Start: {scores[0]} → End: {scores[-1]} | "
            f"Samples: {len(scores)}"
        )
