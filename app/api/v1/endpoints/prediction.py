"""Demand prediction endpoints.

Endpoints delegate entirely to `PredictionService` and only handle
schema conversion and permission checks. Domain exceptions are
translated to HTTP in `app.api.exception_handlers`.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import get_prediction_service
from app.core.config import get_settings
from app.core.security import (
    require_internal_or_permissions,
    require_permissions,
    require_token,
)
from app.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
    PredictionWithHistoryResponse,
    RetrainRequest,
    RetrainResponse,
)
from app.services.prediction_service import PredictionService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["prediction"])
settings = get_settings()

# Permission dependencies resolved once at import time.
require_permissions_predict = (
    require_internal_or_permissions(settings.permissions_predict)
    if settings.permissions_predict
    else require_token
)
require_permissions_retrain = (
    require_internal_or_permissions(settings.permissions_retrain)
    if settings.permissions_retrain
    else require_token
)
require_permissions_models = (
    require_permissions(settings.permissions_models)
    if settings.permissions_models
    else require_token
)


@router.post(
    "/predict",
    response_model=PredictionResponse,
    dependencies=[Depends(require_permissions_predict)],
    summary="Predicts monthly demand by brand/model/line",
)
async def predict(
    request: PredictionRequest,
    service: Annotated[PredictionService, Depends(get_prediction_service)],
) -> PredictionResponse:
    result = await service.predict(
        filters=request.model_dump(exclude={"horizon_months", "confidence"}),
        horizon=request.horizon_months,
        confidence=request.confidence,
    )
    return PredictionResponse(**result.model_dump())


@router.post(
    "/predict-with-history",
    response_model=PredictionWithHistoryResponse,
    dependencies=[Depends(require_permissions_predict)],
    summary="Predicts and returns monthly history for charting",
)
async def predict_with_history(
    request: PredictionRequest,
    service: Annotated[PredictionService, Depends(get_prediction_service)],
) -> PredictionWithHistoryResponse:
    result = await service.predict_with_history(
        filters=request.model_dump(exclude={"horizon_months", "confidence"}),
        horizon=request.horizon_months,
        confidence=request.confidence,
    )
    dump = result.model_dump()
    dump["segment"] = result.segment.model_dump()
    return PredictionWithHistoryResponse(**dump)


@router.post(
    "/retrain",
    response_model=RetrainResponse,
    dependencies=[Depends(require_permissions_retrain)],
    summary="Launches a retrain with fresh data",
)
async def retrain(
    body: RetrainRequest,
    service: Annotated[PredictionService, Depends(get_prediction_service)],
) -> RetrainResponse:
    metadata = await service.retrain(start_date=body.start_date, end_date=body.end_date)
    return RetrainResponse(
        version=metadata.version,
        metrics=metadata.metrics or {},
        trained_at=metadata.trained_at or "",
        samples={
            "train": metadata.train_samples or 0,
            "test": metadata.test_samples or 0,
            "total": metadata.total_samples or 0,
        },
    )


@router.get(
    "/models/latest",
    dependencies=[Depends(require_permissions_models)],
    summary="Gets the latest model metadata",
)
async def latest_model(
    service: Annotated[PredictionService, Depends(get_prediction_service)],
) -> dict:
    metadata = await service.get_latest_model()
    if not metadata:
        return {"detail": "No models available"}
    return metadata.model_dump()
