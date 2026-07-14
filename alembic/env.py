"""Alembic migration environment.

Connects using the same settings as the FastAPI application
(`app.core.config.Settings`). Supports online and offline execution.
"""

from logging.config import fileConfig

from sqlalchemy import create_engine, pool

from alembic import context
from app.core.config import get_settings
from app.core.database import Base
from app.models import ml_records  # noqa: F401

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _get_sync_url() -> str:
    """Fetches the sync (psycopg) DSN from the application settings."""
    settings = get_settings()
    dsn = settings.database_sync_dsn()
    if not dsn:
        raise RuntimeError(
            "Database DSN is not configured. Set DATABASE_URL."
        )
    return dsn


def run_migrations_offline() -> None:
    """Runs migrations in 'offline' mode (generates SQL without connecting)."""
    context.configure(
        url=_get_sync_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Runs migrations in 'online' mode (connects to the database)."""
    connectable = create_engine(
        _get_sync_url(),
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
