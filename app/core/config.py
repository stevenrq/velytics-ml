"""Application settings, centralized via Pydantic Settings."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from typing import Annotated

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
from pydantic_settings.sources import EnvSettingsSource


class LenientEnvSettingsSource(EnvSettingsSource):
    """Env settings source that tolerates malformed JSON for complex fields."""

    def decode_complex_value(self, field_name: str, field: Any, value: Any) -> Any:
        if value is None:
            return value
        if isinstance(value, (bytes, bytearray)):
            value = value.decode()
        if isinstance(value, str) and value.strip() == "":
            return value
        try:
            return super().decode_complex_value(field_name, field, value)
        except json.JSONDecodeError:
            return value


class Settings(BaseSettings):
    """Application settings loaded from environment / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "velytics-ml"
    app_version: str = "0.1.0"
    environment: str = "dev"

    # velytics-backend integration
    velytics_api_url: str = "http://localhost:3000/api/v1"
    service_api_key: str | None = None
    jwks_url: str = "http://localhost:3000/.well-known/jwks.json"
    jwt_issuer: str = "http://localhost:3000"

    # CORS
    cors_allowed_origins: Annotated[list[str], NoDecode] = ["http://localhost:4200"]

    # Database
    database_url: str | None = None
    database_run_migrations: bool = True
    database_echo: bool = False

    # Model registry
    model_dir: str = "models"
    model_name: str = "demand_forecaster"
    default_horizon_months: int = 6
    request_timeout_seconds: float = 15.0
    # Unique months required in the direct dataset. Requires N+1 months of
    # real transactions (e.g. 6 here → 7 months of data). Lower in dev.
    min_history_months: int = 6
    target_column: str = "sales_count"

    # -- Residuals and cold-start -----------------------------------------

    # Segments with fewer months than this receive lags filled with the
    # global mean (cold-start fallback).
    cold_start_min_months: int = 3
    # Bayesian shrinkage of per-segment residuals:
    #   blended = (n·seg_std + prior·global_std) / (n + prior)
    # High prior → trusts the global std more; low → trusts the segment more.
    residual_shrinkage_prior: int = 5

    # -- Data quality --------------------------------------------------------

    # IQR multiplier for outlier clipping on sale/purchase prices.
    # 1.5 = aggressive, 3.0 = conservative (only clips extreme values).
    outlier_iqr_multiplier: float = 3.0

    # -- Hyperparameter tuning ------------------------------------------------

    # Enables RandomizedSearchCV + TimeSeriesSplit. Disable in dev or with
    # small datasets to reduce training time.
    enable_hyperparameter_tuning: bool = True
    # TimeSeriesSplit folds. Use 2 with little data to avoid empty folds.
    cv_folds: int = 3
    # Parallel jobs in RandomizedSearchCV. -1 = all cores.
    cv_n_jobs: int = -1
    # Random combinations evaluated per candidate. More = better search,
    # slower. 5-10 is enough for small datasets.
    cv_n_iter: int = 10
    # Model selection metric: "rmse", "mape" or "weighted".
    model_selection_metric: str = "weighted"
    # MAPE weight in the "weighted" metric; the rest goes to RMSE.
    # 0.0 = only RMSE · 1.0 = only MAPE.
    model_selection_mape_weight: float = 0.4

    # -- Direct multi-step forecast -------------------------------------------

    # Model weight decay per step:
    #   alpha = max(floor, 1 − rate·h) → prediction = alpha·model + (1−alpha)·baseline
    # Use 0.0 to disable dampening.
    forecast_dampening_rate: float = 0.05
    # Minimum model weight in dampening. 0.5 = model always contributes ≥50%.
    forecast_dampening_floor: float = 0.5
    # Horizon steps for which training pairs are generated. Increasing
    # captures longer horizons at the cost of more training time.
    max_training_horizon: int = 12
    # Short window for lag and rolling mean (lag_3, rolling_mean_3).
    lag_window_short: int = 3
    # Long window for lag and rolling mean (lag_6, rolling_mean_6).
    # Use 12 to capture annual seasonality (requires ≥13 months of history).
    lag_window_long: int = 6

    # -- Feature importance ----------------------------------------------------

    # Extracts and persists feature importance after each training run.
    enable_feature_importance: bool = True

    permissions_predict: Annotated[list[str], NoDecode] = ["ml:predict"]
    permissions_retrain: Annotated[list[str], NoDecode] = ["ml:retrain"]
    permissions_models: Annotated[list[str], NoDecode] = ["ml:models"]

    @field_validator(
        "permissions_predict",
        "permissions_retrain",
        "permissions_models",
        "cors_allowed_origins",
        mode="before",
    )
    @classmethod
    def _split_comma_list(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [
                item.strip()
                for item in value.replace(" ", "").split(",")
                if item.strip()
            ]
        return value

    def model_path(self) -> Path:
        """Model artifact directory, created if missing."""
        path = Path(self.model_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _build_dsn(self, driver: str) -> str | None:
        """Rewrites the configured DSN with the given SQLAlchemy driver."""
        if not self.database_url:
            return None
        dsn = self.database_url
        for old_prefix in (
            "postgresql+psycopg2://",
            "postgresql+asyncpg://",
            "postgresql+psycopg://",
            "postgresql://",
        ):
            if dsn.startswith(old_prefix):
                return dsn.replace(old_prefix, f"postgresql+{driver}://", 1)
        return dsn

    def database_dsn(self) -> str | None:
        """Async DSN (asyncpg) for the application runtime."""
        return self._build_dsn("asyncpg")

    def database_sync_dsn(self) -> str | None:
        """Sync DSN (psycopg) for Alembic."""
        return self._build_dsn("psycopg")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type,
        init_settings: Any,
        env_settings: Any,
        dotenv_settings: Any,
        file_secret_settings: Any,
    ) -> tuple:
        return (
            init_settings,
            LenientEnvSettingsSource(settings_cls),
            dotenv_settings,
            file_secret_settings,
        )


@lru_cache
def get_settings() -> Settings:
    """Cached Settings instance for the process lifetime."""
    return Settings()
