"""velytics-ml FastAPI application entrypoint.

Configures the async lifespan, exception handlers, CORS and mounts the
API routers.
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.exception_handlers import register_exception_handlers
from app.api.v1.router import api_router, health_router
from app.core.config import get_settings
from app.core.database import close_engine, database_enabled, init_engine, run_migrations


@asynccontextmanager
async def lifespan(app_: FastAPI) -> AsyncIterator[None]:
    """Manages startup/shutdown of the async database engine."""
    settings = get_settings()
    if database_enabled(settings):
        if settings.database_run_migrations:
            run_migrations(settings)
        await init_engine(settings)
    yield
    await close_engine()


settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["Authorization", "Content-Type"],
)

register_exception_handlers(app)
app.include_router(health_router)
app.include_router(api_router)
