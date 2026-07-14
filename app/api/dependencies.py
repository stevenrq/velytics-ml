"""FastAPI dependency factories.

Builds services with their concrete collaborators directly (no port
indirection): the model registry choice (DB vs filesystem) is the only
runtime branch, based on whether the database is configured.
"""

from __future__ import annotations

from fastapi import Depends

from app.clients.velytics_client import VelyticsClient
from app.core.config import Settings, get_settings
from app.core.database import database_enabled, get_session_factory
from app.ml.feature_engineering import FeatureEngineering
from app.ml.model_training import ModelTrainer
from app.repositories.feature_repository import FeatureRepository
from app.repositories.model_registry import DatabaseModelRegistry, FileSystemModelRegistry
from app.repositories.monitoring_repository import MonitoringRepository
from app.repositories.prediction_repository import PredictionRepository
from app.services.monitoring_service import MonitoringService
from app.services.prediction_service import PredictionService
from app.services.training_service import TrainingService

# ------------------------------------------------------------------
# Infrastructure
# ------------------------------------------------------------------


def get_model_registry(
    settings: Settings = Depends(get_settings),
) -> DatabaseModelRegistry | FileSystemModelRegistry:
    """Picks the DB-backed registry if the database is configured, else FS."""
    factory = get_session_factory()
    if factory is not None:
        return DatabaseModelRegistry(factory, settings.model_name)
    return FileSystemModelRegistry(settings)


def get_feature_repository(
    settings: Settings = Depends(get_settings),
) -> FeatureRepository | None:
    if not database_enabled(settings):
        return None
    factory = get_session_factory()
    if factory is None:
        return None
    return FeatureRepository(factory)


def get_prediction_repository(
    settings: Settings = Depends(get_settings),
) -> PredictionRepository | None:
    if not database_enabled(settings):
        return None
    factory = get_session_factory()
    if factory is None:
        return None
    return PredictionRepository(factory)


def get_velytics_client(
    settings: Settings = Depends(get_settings),
) -> VelyticsClient:
    return VelyticsClient(settings)


def get_feature_engineering(
    settings: Settings = Depends(get_settings),
) -> FeatureEngineering:
    return FeatureEngineering(settings)


def get_model_trainer(
    settings: Settings = Depends(get_settings),
) -> ModelTrainer:
    return ModelTrainer(settings)


# ------------------------------------------------------------------
# Application services
# ------------------------------------------------------------------


def get_training_service(
    registry: DatabaseModelRegistry | FileSystemModelRegistry = Depends(
        get_model_registry
    ),
    feature_engineering: FeatureEngineering = Depends(get_feature_engineering),
    model_trainer: ModelTrainer = Depends(get_model_trainer),
    feature_repository: FeatureRepository | None = Depends(get_feature_repository),
    settings: Settings = Depends(get_settings),
) -> TrainingService:
    return TrainingService(
        registry=registry,
        feature_engineering=feature_engineering,
        model_trainer=model_trainer,
        feature_repository=feature_repository,
        settings=settings,
    )


def get_monitoring_repository(
    settings: Settings = Depends(get_settings),
) -> MonitoringRepository | None:
    if not database_enabled(settings):
        return None
    factory = get_session_factory()
    if factory is None:
        return None
    return MonitoringRepository(factory)


def get_monitoring_service(
    monitoring_repository: MonitoringRepository | None = Depends(
        get_monitoring_repository
    ),
    prediction_repository: PredictionRepository | None = Depends(
        get_prediction_repository
    ),
) -> MonitoringService | None:
    """Builds the monitoring service if the repository is available."""
    if monitoring_repository is None:
        return None
    return MonitoringService(
        monitoring_repository=monitoring_repository,
        prediction_repository=prediction_repository,
    )


def get_prediction_service(
    registry: DatabaseModelRegistry | FileSystemModelRegistry = Depends(
        get_model_registry
    ),
    feature_engineering: FeatureEngineering = Depends(get_feature_engineering),
    training_service: TrainingService = Depends(get_training_service),
    velytics_client: VelyticsClient = Depends(get_velytics_client),
    feature_repository: FeatureRepository | None = Depends(get_feature_repository),
    prediction_repository: PredictionRepository | None = Depends(
        get_prediction_repository
    ),
    settings: Settings = Depends(get_settings),
) -> PredictionService:
    return PredictionService(
        registry=registry,
        feature_engineering=feature_engineering,
        training_service=training_service,
        velytics_client=velytics_client,
        feature_repository=feature_repository,
        prediction_repository=prediction_repository,
        settings=settings,
    )
