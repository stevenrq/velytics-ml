"""Aggregates v1 endpoint routers under the /api/v1/ml prefix."""

from fastapi import APIRouter

from app.api.v1.endpoints import health, monitoring, prediction

api_router = APIRouter(prefix="/api/v1/ml")
api_router.include_router(prediction.router)
api_router.include_router(monitoring.router)

health_router = health.router
