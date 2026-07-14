"""API request/response schemas for demand prediction endpoints."""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    """Monthly demand prediction request."""

    vehicle_type: str = Field(..., description="Vehicle type (CAR/MOTORCYCLE)")
    brand: str
    model: str
    line: str = Field(
        ...,
        description="Line/version (required)",
        min_length=1,
    )

    @field_validator("line", mode="before")
    @classmethod
    def _strip_line(cls, v: str) -> str:
        if isinstance(v, str):
            return v.strip()
        return v

    horizon_months: int = Field(6, gt=0, le=24)
    confidence: float = Field(0.95, ge=0.5, le=0.99)


class MonthlyPrediction(BaseModel):
    """Demand prediction for a single month."""

    month: str
    demand: float
    lower_ci: float
    upper_ci: float


class PredictionResponse(BaseModel):
    """Forecast series and model metadata."""

    predictions: List[MonthlyPrediction]
    model_version: str
    metrics: Optional[Dict[str, Any]] = None


class HistoricalPoint(BaseModel):
    """Historical sales point."""

    month: str
    sales_count: float


class PredictionWithHistoryResponse(BaseModel):
    """Prediction response including historical data."""

    predictions: List[MonthlyPrediction]
    history: List[HistoricalPoint]
    segment: Dict[str, str]
    model_version: str
    trained_at: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


class RetrainRequest(BaseModel):
    """Retrain request."""

    start_date: Optional[date] = Field(None, description="Start date (optional)")
    end_date: Optional[date] = Field(None, description="End date (optional)")


class RetrainResponse(BaseModel):
    """Retrain result."""

    version: str
    metrics: Dict[str, Any]
    trained_at: str
    samples: Dict[str, int]
