"""Maps domain exceptions to HTTP responses."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.schemas.domain import (
    DataLoadError,
    DataQualityError,
    DomainError,
    InsufficientHistoryError,
    MissingVehicleLineError,
    ModelNotTrainedError,
    SegmentNotFoundError,
    TrainingError,
)

_STATUS_MAP: dict[type[DomainError], int] = {
    ModelNotTrainedError: 404,
    SegmentNotFoundError: 404,
    InsufficientHistoryError: 400,
    MissingVehicleLineError: 400,
    TrainingError: 400,
    DataQualityError: 400,
    DataLoadError: 500,
}


def register_exception_handlers(app: FastAPI) -> None:
    """Registers the global handler for domain exceptions."""

    @app.exception_handler(DomainError)
    async def domain_error_handler(request: Request, exc: DomainError) -> JSONResponse:
        status_code = _STATUS_MAP.get(type(exc), 500)
        return JSONResponse(
            status_code=status_code,
            content={"detail": exc.message},
        )
