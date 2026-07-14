"""Model artifact registries: database-backed and filesystem-backed.

`DatabaseModelRegistry` is used when the database is configured;
`FileSystemModelRegistry` is the fallback otherwise (selected in
`app.api.dependencies.get_model_registry`).
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.models.ml_records import ModelArtifact
from app.schemas.domain import ModelMetadata

logger = logging.getLogger(__name__)


class DatabaseModelRegistry:
    """Model registry backed by a SQL database (async)."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        model_name: str,
    ) -> None:
        self._session_factory = session_factory
        self._model_name = model_name

    async def save(self, model: Any, metadata: dict[str, Any]) -> ModelMetadata:
        """Serializes the model with joblib and stores it as BYTEA."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        merged = {**metadata, "version": timestamp}

        buffer = io.BytesIO()
        joblib.dump({"model": model, "metadata": merged}, buffer)

        record = ModelArtifact(
            model_name=self._model_name,
            version=timestamp,
            model_metadata=merged,
            artifact=buffer.getvalue(),
        )

        async with self._session_factory() as session:
            async with session.begin():
                session.add(record)

        logger.info("Model saved in DB with version %s", timestamp)
        return ModelMetadata(**merged)

    async def load_latest(self) -> tuple[Any, ModelMetadata]:
        """Loads the most recently trained model from the database."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(ModelArtifact)
                .where(ModelArtifact.model_name == self._model_name)
                .order_by(ModelArtifact.created_at.desc())
                .limit(1)
            )
            record = result.scalars().first()

        if not record:
            raise FileNotFoundError("No trained model found.")

        artifact = joblib.load(io.BytesIO(record.artifact))
        model = artifact.get("model", artifact)
        raw_metadata = artifact.get("metadata", record.model_metadata)
        return model, ModelMetadata(**raw_metadata)

    async def latest_metadata(self) -> ModelMetadata | None:
        """Fetches the latest model metadata without loading the artifact."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(ModelArtifact)
                .where(ModelArtifact.model_name == self._model_name)
                .order_by(ModelArtifact.created_at.desc())
                .limit(1)
            )
            record = result.scalars().first()

        if not record:
            return None
        return ModelMetadata(**record.model_metadata)


class FileSystemModelRegistry:
    """Model registry backed by the local filesystem."""

    def __init__(self, settings: Settings) -> None:
        self._model_dir: Path = settings.model_path()
        self._model_name: str = settings.model_name
        self._latest_metadata_path = self._model_dir / "latest.json"

    def _artifact_path(self, version: str) -> Path:
        return self._model_dir / f"{self._model_name}_{version}.joblib"

    async def save(self, model: Any, metadata: dict[str, Any]) -> ModelMetadata:
        """Serializes and persists the model on the filesystem."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        merged = {**metadata, "version": timestamp}
        artifact_path = self._artifact_path(timestamp)

        def _write() -> None:
            joblib.dump({"model": model, "metadata": merged}, artifact_path)
            self._latest_metadata_path.write_text(json.dumps(merged, indent=2))

        await asyncio.to_thread(_write)
        logger.info("Model saved to filesystem with version %s", timestamp)
        return ModelMetadata(**merged)

    async def load_latest(self) -> tuple[Any, ModelMetadata]:
        """Loads the most recently trained model from the filesystem."""

        def _read() -> tuple[Any, dict[str, Any]]:
            if not self._latest_metadata_path.exists():
                raise FileNotFoundError("No trained model found.")
            raw = json.loads(self._latest_metadata_path.read_text())
            artifact_path = self._artifact_path(raw["version"])
            artifact = joblib.load(artifact_path)
            model = artifact.get("model", artifact)
            model_metadata = artifact.get("metadata", raw)
            return model, model_metadata

        model, raw_metadata = await asyncio.to_thread(_read)
        return model, ModelMetadata(**raw_metadata)

    async def latest_metadata(self) -> ModelMetadata | None:
        """Fetches the latest model metadata, if any."""

        def _read() -> dict[str, Any] | None:
            if not self._latest_metadata_path.exists():
                return None
            return json.loads(self._latest_metadata_path.read_text())

        raw = await asyncio.to_thread(_read)
        if raw is None:
            return None
        return ModelMetadata(**raw)
