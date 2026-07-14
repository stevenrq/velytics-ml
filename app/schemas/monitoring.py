"""API request/response schemas for drift monitoring endpoints."""

from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field


class ActualDemandRequest(BaseModel):
    """Request to record actual observed demand."""

    model_version: str = Field(..., description="Reference model version")
    vehicle_type: str = Field(..., description="Vehicle type (CAR/MOTORCYCLE)")
    brand: str
    model: str
    line: str = Field(..., min_length=1)
    month: str = Field(
        ...,
        description="Month in ISO format (YYYY-MM-DD)",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    )
    actual_demand: float = Field(..., ge=0, description="Actual demand for the month")
    predicted_demand: float | None = Field(
        None, description="Original prediction (optional)"
    )


class ActualDemandResponse(BaseModel):
    """Response after recording actual demand."""

    status: str = "recorded"
    model_version: str
    month: str


class DriftReportResponse(BaseModel):
    """Model drift report."""

    model_version: str
    total_records: int
    records_with_prediction: int | None = None
    metrics: Dict[str, float]
    records: list[Dict[str, Any]]
