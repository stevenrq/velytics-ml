"""Domain value objects and exceptions for demand forecasting."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class VehicleSegment(BaseModel):
    """Vehicle segment identified by type, brand, model and line."""

    vehicle_type: str
    brand: str
    model: str
    line: str


class ForecastPoint(BaseModel):
    """Forecast point with confidence interval."""

    month: str
    demand: float
    lower_ci: float
    upper_ci: float


class HistoricalPoint(BaseModel):
    """Historical sales point."""

    month: str
    sales_count: float


class ModelMetadata(BaseModel):
    """Full metadata of a trained model."""

    model_config = ConfigDict(extra="allow")

    version: str
    trained_at: str | None = None
    target: str | None = None
    features: list[str] | None = None
    metrics: dict[str, Any] | None = None
    candidates: list[dict[str, Any]] | None = None
    train_samples: int | None = None
    test_samples: int | None = None
    total_samples: int | None = None


class PredictionResult(BaseModel):
    """Result of a demand prediction."""

    predictions: list[ForecastPoint]
    model_version: str
    metrics: dict[str, Any] | None = None


class PredictionWithHistoryResult(BaseModel):
    """Result of a prediction with charting history."""

    predictions: list[ForecastPoint]
    history: list[HistoricalPoint]
    segment: VehicleSegment
    model_version: str
    trained_at: str | None = None
    metrics: dict[str, Any] | None = None


class DomainError(Exception):
    """Base domain exception."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class ModelNotTrainedError(DomainError):
    """No trained model is available."""


class InsufficientHistoryError(DomainError):
    """Data history is insufficient for the requested operation."""


class SegmentNotFoundError(DomainError):
    """The requested vehicle segment was not found in history."""


class DataLoadError(DomainError):
    """Error loading data from an external source, or no loader configured."""


class MissingVehicleLineError(DomainError):
    """The vehicle line is missing from the request."""


class TrainingError(DomainError):
    """Error during model training."""


class DataQualityError(DomainError):
    """Input data does not meet minimum quality requirements."""
