"""Drift monitoring service.

Records actual demand observations and generates drift reports against
stored predictions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

from app.repositories.monitoring_repository import MonitoringRepository
from app.repositories.prediction_repository import PredictionRepository

logger = logging.getLogger(__name__)


class MonitoringService:
    """Use cases: record actuals and generate drift reports."""

    def __init__(
        self,
        monitoring_repository: MonitoringRepository,
        prediction_repository: PredictionRepository | None = None,
    ) -> None:
        self._monitoring_repo = monitoring_repository
        self._prediction_repo = prediction_repository

    async def record_actual(
        self,
        model_version: str,
        segment: Dict[str, Any],
        month: str,
        actual_demand: float,
        predicted_demand: float | None = None,
    ) -> None:
        """Records actual demand for a given month and segment."""
        await self._monitoring_repo.save_actual(
            model_version=model_version,
            segment=segment,
            month=month,
            actual_demand=actual_demand,
            predicted_demand=predicted_demand,
        )
        logger.info(
            "Actual recorded: version=%s month=%s actual=%.2f predicted=%s",
            model_version,
            month,
            actual_demand,
            predicted_demand,
        )

    async def get_drift_report(
        self,
        model_version: str,
        segment: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Generates a drift report for the given model version."""
        records = await self._monitoring_repo.load_drift_records(model_version, segment)

        if not records:
            return {
                "model_version": model_version,
                "total_records": 0,
                "metrics": {},
                "records": [],
            }

        errors = [
            r["absolute_error"] for r in records if r["absolute_error"] is not None
        ]

        actuals = [
            r["actual_demand"] for r in records if r["predicted_demand"] is not None
        ]
        predicted = [
            r["predicted_demand"] for r in records if r["predicted_demand"] is not None
        ]

        metrics: Dict[str, float] = {}
        if errors:
            metrics["mae"] = float(np.mean(errors))
            metrics["max_error"] = float(np.max(errors))

        if actuals and predicted:
            safe_actuals = [max(a, 1e-3) for a in actuals]
            mape_values = [
                abs(a - p) / sa for a, p, sa in zip(actuals, predicted, safe_actuals)
            ]
            metrics["mape"] = float(np.mean(mape_values))

        return {
            "model_version": model_version,
            "total_records": len(records),
            "records_with_prediction": len(errors),
            "metrics": metrics,
            "records": records,
        }
