"""Pydantic request/response schemas for the FastAPI serving endpoint."""

from pydantic import BaseModel, Field


class ViolationReportSchema(BaseModel):
    worker_id: int
    bbox: tuple[int, int, int, int]
    violations: list[str]
    detection_confidence: float
    rule_confidence: float
    confidence_tier: str
    severity: str
    scene_context: str
    human_readable: str
    recommended_action: str
    is_visitor: bool
    is_site_alert: bool


class AnalyseResponse(BaseModel):
    compliance_score: int = Field(..., ge=0, le=100)
    score_band: str
    score_display: str
    scene_context: str
    n_workers: int
    n_critical: int
    n_high: int
    n_warning: int
    crowd_alert: bool
    worker_reports: list[ViolationReportSchema]
    site_alert: ViolationReportSchema | None
    formatted_report: str
    temporal_status: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
