"""Shared fixtures for velytics-ml tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from app.core.config import Settings
from app.ml.model_training import TrainingEvaluation
from app.schemas.domain import ModelMetadata

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------


@pytest.fixture
def mock_settings() -> Settings:
    """Settings with minimal, controlled values for tests."""
    return Settings(
        model_dir="/tmp/velytics_test_models",
        min_history_months=1,
        target_column="sales_count",
        velytics_api_url="http://test-velytics-api",
        service_api_key="test-service-key",
        permissions_predict=["ml:predict"],
        permissions_retrain=["ml:retrain"],
        permissions_models=["ml:models"],
        cold_start_min_months=3,
        residual_shrinkage_prior=5,
        outlier_iqr_multiplier=3.0,
        enable_hyperparameter_tuning=False,
        cv_folds=3,
        cv_n_iter=5,
        model_selection_metric="weighted",
        model_selection_mape_weight=0.4,
        forecast_dampening_rate=0.05,
        forecast_dampening_floor=0.5,
        max_training_horizon=12,
        lag_window_short=3,
        lag_window_long=6,
        enable_feature_importance=True,
    )


# ------------------------------------------------------------------
# Domain entities
# ------------------------------------------------------------------


@pytest.fixture
def sample_metadata() -> ModelMetadata:
    """Representative metadata of a trained model."""
    return ModelMetadata(
        version="20250101120000",
        trained_at="2025-01-01T12:00:00+00:00",
        target="sales_count",
        features=[
            "vehicle_type",
            "brand",
            "model",
            "line",
            "purchases_count",
            "avg_margin",
            "avg_sale_price",
            "avg_purchase_price",
            "avg_days_inventory",
            "inventory_rotation",
            "lag_1",
            "lag_3",
            "lag_6",
            "rolling_mean_3",
            "rolling_mean_6",
            "month",
            "year",
            "month_sin",
            "month_cos",
            "quarter",
            "quarter_sin",
            "quarter_cos",
            "horizon_step",
        ],
        metrics={
            "rmse": 1.5,
            "mae": 1.0,
            "mape": 0.15,
            "r2": 0.85,
            "residual_std": 1.2,
            "horizon_residuals": {1: 1.1, 2: 1.3, 3: 1.5, 6: 1.9},
        },
        candidates=[{"model": "linear_regression", "rmse": 2.0}],
        train_samples=80,
        test_samples=20,
        total_samples=100,
    )


# ------------------------------------------------------------------
# Test DataFrames
# ------------------------------------------------------------------


@pytest.fixture
def sample_history_df() -> pd.DataFrame:
    """History DataFrame with 3 months for a single segment."""
    months = pd.date_range("2024-10-01", periods=3, freq="MS")
    return pd.DataFrame(
        {
            "event_month": months,
            "vehicle_type": ["CAR"] * 3,
            "brand": ["TOYOTA"] * 3,
            "model": ["COROLLA"] * 3,
            "line": ["XLE"] * 3,
            "sales_count": [10.0, 12.0, 11.0],
            "purchases_count": [8.0, 10.0, 9.0],
            "avg_margin": [5000.0, 5500.0, 5200.0],
            "avg_sale_price": [25000.0, 26000.0, 25500.0],
            "avg_purchase_price": [20000.0, 20500.0, 20300.0],
            "avg_days_inventory": [30.0, 28.0, 29.0],
            "inventory_rotation": [1.25, 1.2, 1.22],
            "lag_1": [0.0, 10.0, 12.0],
            "lag_3": [0.0, 0.0, 0.0],
            "lag_6": [0.0, 0.0, 0.0],
            "rolling_mean_3": [0.0, 10.0, 11.0],
            "rolling_mean_6": [0.0, 10.0, 11.0],
            "month": [10, 11, 12],
            "year": [2024, 2024, 2024],
            "month_sin": [np.sin(2 * np.pi * m / 12) for m in [10, 11, 12]],
            "month_cos": [np.cos(2 * np.pi * m / 12) for m in [10, 11, 12]],
            "quarter": [4, 4, 4],
            "quarter_sin": [np.sin(2 * np.pi * 4 / 4)] * 3,
            "quarter_cos": [np.cos(2 * np.pi * 4 / 4)] * 3,
        }
    )


@pytest.fixture
def sample_future_row() -> pd.DataFrame:
    """Future-month feature row (result of build_future_row)."""
    return pd.DataFrame(
        [
            {
                "event_month": pd.Timestamp("2025-01-01"),
                "vehicle_type": "CAR",
                "brand": "TOYOTA",
                "model": "COROLLA",
                "line": "XLE",
                "purchases_count": 9.0,
                "avg_margin": 5200.0,
                "avg_sale_price": 25500.0,
                "avg_purchase_price": 20300.0,
                "avg_days_inventory": 29.0,
                "inventory_rotation": 1.22,
                "lag_1": 11.0,
                "lag_3": 10.0,
                "lag_6": 0.0,
                "rolling_mean_3": 11.0,
                "rolling_mean_6": 11.0,
                "month": 1,
                "year": 2025,
                "month_sin": np.sin(2 * np.pi / 12),
                "month_cos": np.cos(2 * np.pi / 12),
                "quarter": 1,
                "quarter_sin": np.sin(2 * np.pi * 1 / 4),
                "quarter_cos": np.cos(2 * np.pi * 1 / 4),
                "horizon_step": 1.0,
            }
        ]
    )


# ------------------------------------------------------------------
# sklearn model mocks
# ------------------------------------------------------------------


@pytest.fixture
def mock_model() -> MagicMock:
    """Mock of a sklearn pipeline with predict()."""
    model = MagicMock()
    model.predict.return_value = np.array([13.0])
    return model


# ------------------------------------------------------------------
# Repository / client mocks
# ------------------------------------------------------------------


@pytest.fixture
def mock_model_registry(
    sample_metadata: ModelMetadata, mock_model: MagicMock
) -> AsyncMock:
    """Mock of the model registry (DB or filesystem)."""
    registry = AsyncMock()
    registry.load_latest.return_value = (mock_model, sample_metadata)
    registry.save.return_value = sample_metadata
    registry.latest_metadata.return_value = sample_metadata
    return registry


@pytest.fixture
def mock_feature_repository(sample_history_df: pd.DataFrame) -> AsyncMock:
    """Mock of FeatureRepository."""
    repo = AsyncMock()
    repo.load_segment_history.return_value = sample_history_df
    repo.save_snapshot.return_value = None
    repo.load_snapshot.return_value = sample_history_df
    return repo


@pytest.fixture
def mock_prediction_repository() -> AsyncMock:
    """Mock of PredictionRepository."""
    repo = AsyncMock()
    repo.save_prediction.return_value = None
    return repo


@pytest.fixture
def mock_velytics_client(sample_history_df: pd.DataFrame) -> AsyncMock:
    """Mock of VelyticsClient."""
    client = AsyncMock()
    client.load_transactions.return_value = sample_history_df
    return client


# ------------------------------------------------------------------
# ML infrastructure mocks
# ------------------------------------------------------------------


@pytest.fixture
def mock_feature_engineering(
    sample_history_df: pd.DataFrame,
    sample_future_row: pd.DataFrame,
) -> MagicMock:
    """Mock of FeatureEngineering with column attributes and stubbed methods."""
    fe = MagicMock()
    fe.category_cols = ["vehicle_type", "brand", "model", "line"]
    fe.optional_category_cols = []
    fe.numeric_cols = [
        "purchases_count",
        "avg_margin",
        "avg_sale_price",
        "avg_purchase_price",
        "avg_days_inventory",
        "inventory_rotation",
        "lag_1",
        "lag_3",
        "lag_6",
        "rolling_mean_3",
        "rolling_mean_6",
        "month",
        "year",
        "month_sin",
        "month_cos",
        "quarter",
        "quarter_sin",
        "quarter_cos",
    ]
    fe.direct_numeric_cols = fe.numeric_cols + ["horizon_step"]
    fe.build_future_row.return_value = sample_future_row
    fe.build_future_row_at_origin.return_value = sample_future_row
    fe.build_feature_table.return_value = sample_history_df
    direct_df = sample_history_df.copy()
    direct_df["horizon_step"] = 1.0
    fe.build_direct_feature_table.return_value = direct_df
    return fe


@pytest.fixture
def mock_model_trainer(mock_model: MagicMock) -> MagicMock:
    """Mock of ModelTrainer with a predefined evaluation."""
    trainer = MagicMock()
    trainer.train_and_evaluate.return_value = TrainingEvaluation(
        pipeline=mock_model,
        metrics={"rmse": 1.5, "mae": 1.0, "mape": 0.15, "r2": 0.85},
        residual_std=1.2,
        candidates=[{"model": "random_forest", "rmse": 1.5}],
        train_samples=80,
        test_samples=20,
        segment_residuals={"CAR|TOYOTA|COROLLA|XLE": 1.1},
        best_params={"model__n_estimators": 300},
        feature_importances={"lag_1": 0.3, "sales_count": 0.2},
        horizon_residuals={1: 1.1, 2: 1.3, 3: 1.5},
    )
    return trainer


# ------------------------------------------------------------------
# Application service mocks
# ------------------------------------------------------------------


@pytest.fixture
def mock_training_service(sample_metadata: ModelMetadata) -> AsyncMock:
    """Mock of TrainingService."""
    svc = AsyncMock()
    svc.train.return_value = sample_metadata
    return svc
