"""
FastAPI serving endpoint for Construction Safety Monitor.

Endpoints:
  GET  /health       — health check + model status
  POST /analyse      — analyse a construction site image

Security:
  - Rate limiting: 20 req/min on /analyse (slowapi)
  - Body size limit: 10 MB
  - Generic 500 handler — no internal paths or stack traces exposed
  - Security headers middleware
"""

import logging
import os
from contextlib import asynccontextmanager
from io import BytesIO

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

from serving.schemas import AnalyseResponse, HealthResponse, ViolationReportSchema

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MAX_BODY_BYTES = 10 * 1024 * 1024  # 10 MB
WEIGHTS_PATH = os.getenv("MODEL_WEIGHTS_PATH", "runs/train/weights/best.pt")

limiter = Limiter(key_func=get_remote_address)
_pipeline = None


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    from inference.pipeline import ConstructionSafetyPipeline
    try:
        _pipeline = ConstructionSafetyPipeline(weights_path=WEIGHTS_PATH)
        logger.info("Model loaded: %s", WEIGHTS_PATH)
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        _pipeline = None
    yield
    _pipeline = None


app = FastAPI(
    title="Construction Safety Monitor API",
    description="PPE compliance detection for construction sites.",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error: %s", exc, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", model_loaded=_pipeline is not None)


@app.post("/analyse", response_model=AnalyseResponse)
@limiter.limit("20/minute")
async def analyse(request: Request, file: UploadFile = File(...)):
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    content = await file.read()
    if len(content) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 10 MB limit.")

    try:
        pil_img = Image.open(BytesIO(content)).convert("RGB")
        frame_rgb = np.array(pil_img)
    except Exception:
        raise HTTPException(status_code=422, detail="Could not decode image.")

    result = _pipeline.analyse(frame_rgb)
    temporal_status = _pipeline.get_temporal_status()

    worker_schemas = [
        ViolationReportSchema(
            worker_id=r.worker_id,
            bbox=r.bbox,
            violations=r.violations,
            detection_confidence=r.detection_confidence,
            rule_confidence=r.rule_confidence,
            confidence_tier=r.confidence_tier,
            severity=r.severity,
            scene_context=r.scene_context,
            human_readable=r.human_readable,
            recommended_action=r.recommended_action,
            is_visitor=r.is_visitor,
            is_site_alert=r.is_site_alert,
        )
        for r in result.site_report.worker_reports
    ]

    site_alert_schema = None
    if result.site_report.site_alert:
        sa = result.site_report.site_alert
        site_alert_schema = ViolationReportSchema(
            worker_id=sa.worker_id,
            bbox=sa.bbox,
            violations=sa.violations,
            detection_confidence=sa.detection_confidence,
            rule_confidence=sa.rule_confidence,
            confidence_tier=sa.confidence_tier,
            severity=sa.severity,
            scene_context=sa.scene_context,
            human_readable=sa.human_readable,
            recommended_action=sa.recommended_action,
            is_visitor=sa.is_visitor,
            is_site_alert=sa.is_site_alert,
        )

    return AnalyseResponse(
        compliance_score=result.score_result.score,
        score_band=result.score_result.band,
        score_display=result.score_result.display,
        scene_context=result.site_report.scene_context,
        n_workers=len(result.site_report.worker_reports),
        n_critical=result.score_result.n_critical,
        n_high=result.score_result.n_high,
        n_warning=result.score_result.n_warning,
        crowd_alert=result.score_result.crowd_triggered,
        worker_reports=worker_schemas,
        site_alert=site_alert_schema,
        formatted_report=result.formatted_report,
        temporal_status=temporal_status,
    )
