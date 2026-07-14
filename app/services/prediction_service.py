"""Demand prediction service.

Orchestrates predict, predict-with-history, retrain and model-metadata
use cases.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, List, Optional

import pandas as pd
from scipy.stats import norm

from app.clients.velytics_client import VelyticsClient
from app.core.config import Settings, get_settings
from app.ml.feature_engineering import FeatureEngineering
from app.ml.normalization import canonicalize_brand_model, canonicalize_label
from app.repositories.feature_repository import FeatureRepository
from app.repositories.model_registry import DatabaseModelRegistry, FileSystemModelRegistry
from app.repositories.prediction_repository import PredictionRepository
from app.schemas.domain import (
    DataLoadError,
    ForecastPoint,
    HistoricalPoint,
    InsufficientHistoryError,
    MissingVehicleLineError,
    ModelMetadata,
    ModelNotTrainedError,
    PredictionResult,
    PredictionWithHistoryResult,
    SegmentNotFoundError,
    VehicleSegment,
)
from app.services.training_service import TrainingService

logger = logging.getLogger(__name__)


class PredictionService:
    """Use cases: predict demand, retrain and query model metadata."""

    def __init__(
        self,
        registry: DatabaseModelRegistry | FileSystemModelRegistry,
        feature_engineering: FeatureEngineering,
        training_service: TrainingService,
        velytics_client: VelyticsClient | None = None,
        feature_repository: FeatureRepository | None = None,
        prediction_repository: PredictionRepository | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._registry = registry
        self._feature_engineering = feature_engineering
        self._training_service = training_service
        self._velytics_client = velytics_client
        self._feature_repository = feature_repository
        self._prediction_repository = prediction_repository
        self._settings = settings or get_settings()

    # ------------------------------------------------------------------
    # Public use cases
    # ------------------------------------------------------------------

    async def predict(
        self,
        filters: Dict[str, Any],
        horizon: int,
        confidence: float = 0.95,
    ) -> PredictionResult:
        """Predicts monthly demand for a vehicle segment.

        Raises
        ------
        DataLoadError
            If no data source is configured for prediction.
        ModelNotTrainedError
            If no trained model is available.
        SegmentNotFoundError
            If no history is found for the requested segment.
        """
        self._require_data_source()
        model, metadata = await self._load_model()

        segment = self._normalize_filters(filters)
        history = await self._load_history(segment, metadata.version)
        self._require_history(history)

        history = history.sort_values("event_month")
        forecast = self._forecast(model, metadata, history, horizon, confidence)

        result = PredictionResult(
            predictions=[ForecastPoint(**f) for f in forecast],
            model_version=metadata.version,
            metrics=metadata.metrics,
        )

        await self._store_prediction(
            metadata.version,
            {**filters, "horizon_months": horizon, "confidence": confidence},
            result.model_dump(),
            segment,
            horizon,
            confidence,
            with_history=False,
        )
        return result

    async def predict_with_history(
        self,
        filters: Dict[str, Any],
        horizon: int,
        confidence: float = 0.95,
    ) -> PredictionWithHistoryResult:
        """Predicts demand and also returns the history used for charting.

        Raises
        ------
        DataLoadError, ModelNotTrainedError, SegmentNotFoundError
            Same behavior as `predict`.
        """
        self._require_data_source()
        model, metadata = await self._load_model()

        segment = self._normalize_filters(filters)
        history = await self._load_history(segment, metadata.version)
        self._require_history(history)

        history = history.sort_values("event_month")
        forecast = self._forecast(model, metadata, history, horizon, confidence)
        history_points = [
            HistoricalPoint(month=ts.date().isoformat(), sales_count=float(sc))
            for ts, sc in zip(history["event_month"], history["sales_count"])
        ]

        result = PredictionWithHistoryResult(
            predictions=[ForecastPoint(**f) for f in forecast],
            history=history_points,
            segment=segment,
            model_version=metadata.version,
            trained_at=metadata.trained_at,
            metrics=metadata.metrics,
        )

        await self._store_prediction(
            metadata.version,
            {**filters, "horizon_months": horizon, "confidence": confidence},
            result.model_dump(),
            segment,
            horizon,
            confidence,
            with_history=True,
        )
        return result

    async def retrain(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> ModelMetadata:
        """Launches a retrain with fresh data.

        Raises
        ------
        DataLoadError
            If no client is configured to load data for retraining.
        InsufficientHistoryError
            If there isn't enough data to train in the requested range.
        TrainingError
            If the training process fails internally.
        """
        if not self._velytics_client:
            raise DataLoadError("No data client configured for retraining.")
        raw_df = await self._velytics_client.load_transactions(
            start_date=start_date, end_date=end_date
        )
        if raw_df.empty:
            raise InsufficientHistoryError(
                "No data to train on in the requested range."
            )
        return await self._training_service.train(raw_df)

    async def get_latest_model(self) -> ModelMetadata | None:
        """Fetches the latest trained model metadata."""
        return await self._registry.latest_metadata()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _require_data_source(self) -> None:
        if not self._velytics_client and not self._feature_repository:
            raise DataLoadError("No data client configured for prediction.")

    async def _load_model(self) -> tuple[Any, ModelMetadata]:
        try:
            return await self._registry.load_latest()
        except FileNotFoundError as exc:
            raise ModelNotTrainedError("No trained model exists yet.") from exc

    def _normalize_filters(self, filters: Dict[str, Any]) -> VehicleSegment:
        vehicle_type = canonicalize_label(filters.get("vehicle_type"))
        brand, model = canonicalize_brand_model(
            filters.get("brand"), filters.get("model")
        )
        line = canonicalize_label(filters.get("line"))
        if not line:
            raise MissingVehicleLineError(
                "Missing vehicle line. "
                "Include full brand/model/line to predict."
            )
        return VehicleSegment(
            vehicle_type=vehicle_type, brand=brand, model=model, line=line
        )

    @staticmethod
    def _require_history(history: pd.DataFrame) -> None:
        if history.empty:
            raise SegmentNotFoundError(
                "No history found for the requested combination. "
                "Check vehicle_type/brand/model/line or retrain with data "
                "that includes that segment."
            )

    async def _load_history(
        self, segment: VehicleSegment, model_version: str
    ) -> pd.DataFrame:
        """Tries to load history from the feature repository, else the client."""
        if self._feature_repository:
            history = await self._feature_repository.load_segment_history(
                model_version, segment.model_dump()
            )
            if not history.empty:
                return history

        if not self._velytics_client:
            return pd.DataFrame()

        raw_df = await self._velytics_client.load_transactions()
        feature_df = self._feature_engineering.build_feature_table(raw_df)
        if feature_df.empty:
            return feature_df
        if self._feature_repository:
            await self._feature_repository.save_snapshot(model_version, feature_df)
        return self._filter_history(feature_df, segment)

    @staticmethod
    def _filter_history(
        feature_df: pd.DataFrame, segment: VehicleSegment
    ) -> pd.DataFrame:
        mask = (
            (feature_df["vehicle_type"] == segment.vehicle_type)
            & (feature_df["brand"] == segment.brand)
            & (feature_df["model"] == segment.model)
            & (feature_df["line"] == segment.line)
        )
        return feature_df[mask].copy()

    async def _store_prediction(
        self,
        model_version: str,
        request_payload: Dict[str, Any],
        response_payload: Dict[str, Any],
        segment: VehicleSegment,
        horizon: int,
        confidence: float,
        with_history: bool,
    ) -> None:
        if not self._prediction_repository:
            return
        try:
            await self._prediction_repository.save_prediction(
                model_version=model_version,
                request_payload=request_payload,
                response_payload=response_payload,
                segment=segment.model_dump(),
                horizon=horizon,
                confidence=confidence,
                with_history=with_history,
            )
        except Exception as exc:
            logger.warning("Failed to save prediction: %s", exc)

    def _forecast(
        self,
        model: Any,
        metadata: ModelMetadata,
        history: pd.DataFrame,
        horizon: int,
        confidence: float,
    ) -> List[Dict[str, Any]]:
        """Generates a direct multi-step forecast using the trained model.

        Lags are read from the real history frozen at the prediction
        origin: previous predictions are never used as features, so
        there is no error accumulation between steps.
        """
        metrics = metadata.metrics or {}
        z_value = self._z_value(confidence)
        baseline = self._compute_segment_baseline(history)

        rate = self._settings.forecast_dampening_rate
        floor = self._settings.forecast_dampening_floor

        fe = self._feature_engineering
        frozen_history = history.copy()
        origin_month = frozen_history["event_month"].max()
        results: List[Dict[str, Any]] = []

        for step in range(1, horizon + 1):
            target_month = origin_month + pd.offsets.MonthBegin(step)
            future_row = fe.build_future_row_at_origin(
                frozen_history, target_month, step
            )

            model_features = metadata.features or (
                fe.category_cols + fe.direct_numeric_cols
            )
            optional_cols = [
                col for col in fe.optional_category_cols if col in future_row.columns
            ]
            all_cols = fe.category_cols + optional_cols + fe.direct_numeric_cols
            compatible_cols = [
                c for c in all_cols if c in model_features and c in future_row.columns
            ]
            features = future_row[compatible_cols]
            raw_prediction = float(model.predict(features)[0])

            alpha = max(floor, 1.0 - rate * step)
            prediction = alpha * raw_prediction + (1 - alpha) * baseline

            step_std = self._resolve_horizon_residual(metrics, step)
            margin = z_value * step_std
            lower = max(0.0, prediction - margin)
            upper = max(lower, prediction + margin)

            results.append(
                {
                    "month": target_month.date().isoformat(),
                    "demand": prediction,
                    "lower_ci": lower,
                    "upper_ci": upper,
                }
            )

        return results

    def _compute_segment_baseline(self, history: pd.DataFrame) -> float:
        """Mean sales over the last lag_window_long months (dampening baseline)."""
        if history.empty:
            return 0.0
        recent = history.tail(self._settings.lag_window_long)["sales_count"]
        return float(recent.mean())

    @staticmethod
    def _resolve_horizon_residual(metrics: Dict[str, Any], step: int) -> float:
        """Fetches the residual_std calibrated for horizon `step`.

        Prefers `horizon_residuals[step]` (computed per-horizon during
        direct multi-step training). Falls back to the global
        `residual_std` if unavailable.
        """
        horizon_residuals = metrics.get("horizon_residuals", {})
        if horizon_residuals:
            h_std = horizon_residuals.get(step) or horizon_residuals.get(str(step))
            if h_std is not None:
                return float(h_std)
        return float(metrics.get("residual_std", 1.0))

    @staticmethod
    def _z_value(confidence: float) -> float:
        """Z-value for a given confidence level using scipy."""
        conf = min(max(confidence, 0.5), 0.99)
        return float(norm.ppf((1 + conf) / 2))
