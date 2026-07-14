"""Drift monitoring and feature-importance endpoints."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.dependencies import get_monitoring_service, get_prediction_service
from app.core.config import get_settings
from app.core.security import (
    require_internal_or_permissions,
    require_permissions,
    require_token,
)
from app.schemas.monitoring import (
    ActualDemandRequest,
    ActualDemandResponse,
    DriftReportResponse,
)
from app.services.monitoring_service import MonitoringService
from app.services.prediction_service import PredictionService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["monitoring"])
settings = get_settings()

require_permissions_models = (
    require_permissions(settings.permissions_models)
    if settings.permissions_models
    else require_token
)
require_permissions_retrain = (
    require_internal_or_permissions(settings.permissions_retrain)
    if settings.permissions_retrain
    else require_token
)


@router.post(
    "/actuals",
    response_model=ActualDemandResponse,
    dependencies=[Depends(require_permissions_retrain)],
    summary="Records actual demand for drift monitoring",
)
async def record_actual(
    request: ActualDemandRequest,
    service: Annotated[MonitoringService | None, Depends(get_monitoring_service)],
) -> ActualDemandResponse:
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Monitoring service unavailable: database not configured.",
        )
    segment = {
        "vehicle_type": request.vehicle_type,
        "brand": request.brand,
        "model": request.model,
        "line": request.line,
    }
    await service.record_actual(
        model_version=request.model_version,
        segment=segment,
        month=request.month,
        actual_demand=request.actual_demand,
        predicted_demand=request.predicted_demand,
    )
    return ActualDemandResponse(
        model_version=request.model_version,
        month=request.month,
    )


@router.get(
    "/drift-report",
    response_model=DriftReportResponse,
    dependencies=[Depends(require_permissions_models)],
    summary="Generates a model drift report",
)
async def drift_report(
    service: Annotated[MonitoringService | None, Depends(get_monitoring_service)],
    model_version: str = Query(..., description="Model version"),
) -> DriftReportResponse:
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Monitoring service unavailable: database not configured.",
        )
    report = await service.get_drift_report(model_version)
    return DriftReportResponse(**report)


@router.get(
    "/models/latest/feature-importance",
    dependencies=[Depends(require_permissions_models)],
    summary="Gets feature importances of the current model",
)
async def feature_importance(
    service: Annotated[PredictionService, Depends(get_prediction_service)],
) -> dict:
    metadata = await service.get_latest_model()
    if not metadata:
        return {"detail": "No models available"}
    importances = (metadata.metrics or {}).get("feature_importances", {})
    return {
        "model_version": metadata.version,
        "feature_importances": importances,
    }
