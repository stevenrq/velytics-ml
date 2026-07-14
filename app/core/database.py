"""Async database engine lifecycle and Alembic migration runner."""

from __future__ import annotations

import logging
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import Settings

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Base(DeclarativeBase):
    """Declarative base for ORM models."""


_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def database_enabled(settings: Settings) -> bool:
    return bool(settings.database_dsn())


async def init_engine(settings: Settings) -> AsyncEngine | None:
    """Creates the async engine and session factory."""
    dsn = settings.database_dsn()
    if not dsn:
        return None

    global _engine, _session_factory
    _engine = create_async_engine(
        dsn,
        echo=settings.database_echo,
        pool_pre_ping=True,
    )
    _session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        autoflush=False,
        expire_on_commit=False,
    )
    return _engine


async def close_engine() -> None:
    """Disposes the engine and releases pool connections."""
    global _engine, _session_factory
    if _engine:
        await _engine.dispose()
    _engine = None
    _session_factory = None


def get_session_factory() -> async_sessionmaker[AsyncSession] | None:
    return _session_factory


def run_migrations(settings: Settings) -> None:
    """Applies all pending Alembic migrations (upgrade head)."""
    dsn = settings.database_sync_dsn()
    if not dsn:
        logger.warning("Database DSN not configured — skipping migrations")
        return

    alembic_cfg = Config(str(_PROJECT_ROOT / "alembic.ini"))
    alembic_cfg.set_main_option("script_location", str(_PROJECT_ROOT / "alembic"))
    alembic_cfg.set_main_option("sqlalchemy.url", dsn)

    logger.info("Running Alembic migrations (upgrade head)...")
    command.upgrade(alembic_cfg, "head")
    logger.info("Alembic migrations completed")
