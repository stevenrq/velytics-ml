"""Model training service.

Orchestrates: feature engineering -> ML training -> model persistence.
Contains no ML logic directly (delegated to `FeatureEngineering` and
`ModelTrainer`) nor direct database access (delegated to the registry
and feature repository).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from app.core.config import Settings, get_settings
from app.ml.feature_engineering import FeatureEngineering
from app.ml.model_training import ModelTrainer
from app.repositories.feature_repository import FeatureRepository
from app.repositories.model_registry import DatabaseModelRegistry, FileSystemModelRegistry
from app.schemas.domain import ModelMetadata, TrainingError

logger = logging.getLogger(__name__)


class TrainingService:
    """Use case: train and version a demand model."""

    def __init__(
        self,
        registry: DatabaseModelRegistry | FileSystemModelRegistry,
        feature_engineering: FeatureEngineering,
        model_trainer: ModelTrainer,
        feature_repository: FeatureRepository | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._registry = registry
        self._feature_engineering = feature_engineering
        self._model_trainer = model_trainer
        self._feature_repository = feature_repository
        self._settings = settings or get_settings()

    async def train(self, raw_df: pd.DataFrame) -> ModelMetadata:
        """Trains on raw data and persists the best model.

        Raises
        ------
        TrainingError
            If the data is insufficient or training/evaluation fails.
        """
        try:
            dataset = self._feature_engineering.build_feature_table(raw_df)
        except ValueError as exc:
            raise TrainingError(str(exc)) from exc

        if dataset.empty:
            raise TrainingError("No historical data to train on.")

        direct_dataset = self._feature_engineering.build_direct_feature_table(
            dataset, self._settings.max_training_horizon
        )
        if direct_dataset.empty:
            raise TrainingError(
                "Not enough data to build the direct multi-step dataset."
            )

        try:
            evaluation = await asyncio.to_thread(
                self._model_trainer.train_and_evaluate,
                direct_dataset,
                self._feature_engineering.category_cols,
                self._feature_engineering.optional_category_cols,
                self._feature_engineering.direct_numeric_cols,
            )
        except ValueError as exc:
            raise TrainingError(str(exc)) from exc

        metadata_dict: dict[str, Any] = {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "target": self._settings.target_column,
            "features": (
                self._feature_engineering.category_cols
                + self._feature_engineering.direct_numeric_cols
            ),
            "metrics": {
                **evaluation.metrics,
                "residual_std": evaluation.residual_std,
                "segment_residuals": evaluation.segment_residuals,
                "horizon_residuals": evaluation.horizon_residuals,
                "feature_importances": evaluation.feature_importances,
            },
            "candidates": evaluation.candidates,
            "best_params": evaluation.best_params,
            "feature_importances": evaluation.feature_importances,
            "train_samples": evaluation.train_samples,
            "test_samples": evaluation.test_samples,
            "total_samples": len(direct_dataset),
        }

        saved = await self._registry.save(evaluation.pipeline, metadata_dict)

        if self._feature_repository:
            await self._feature_repository.save_snapshot(saved.version, dataset)

        logger.info("Model trained and versioned: %s", saved.version)
        return saved
