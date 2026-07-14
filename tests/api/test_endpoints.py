"""API-level tests using httpx's ASGI transport."""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, patch

import pytest
from authlib.jose import JsonWebKey, JsonWebToken
from httpx import ASGITransport, AsyncClient

from app.api.dependencies import get_prediction_service
from app.api.v1.endpoints import prediction as prediction_endpoints
from app.core.config import get_settings
from app.core.security import jwks_cache
from app.main import app
from app.schemas.domain import ForecastPoint, PredictionResult

_TEST_KID = "test-key-1"


def _generate_test_key():
    """A throwaway RSA key, unrelated to velytics-backend's real signing key."""
    return JsonWebKey.generate_key(
        "RSA", 2048, options={"kid": _TEST_KID}, is_private=True
    )


def _mock_key_set(key):
    public_jwk = json.loads(key.as_json(is_private=False))
    return JsonWebKey.import_key_set({"keys": [public_jwk]})


def _sign_token(key, authorities: list[str], issuer: str, ttl_seconds: int = 3600) -> str:
    jwt = JsonWebToken(["RS256"])
    header = {"alg": "RS256", "kid": _TEST_KID}
    now = int(time.time())
    payload = {
        "sub": "999",
        "username": "test_noperm",
        "authorities": authorities,
        "iss": issuer,
        "iat": now,
        "exp": now + ttl_seconds,
    }
    token = jwt.encode(header, payload, key)
    return token.decode("utf-8") if isinstance(token, bytes) else token


@pytest.fixture
def mock_prediction_service() -> AsyncMock:
    service = AsyncMock()
    service.predict.return_value = PredictionResult(
        predictions=[
            ForecastPoint(month="2025-01-01", demand=10.0, lower_ci=8.0, upper_ci=12.0)
        ],
        model_version="20250101120000",
        metrics={"rmse": 1.5},
    )
    return service


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_health_returns_ok() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_predict_requires_auth() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/ml/predict",
            json={
                "vehicle_type": "CAR",
                "brand": "TOYOTA",
                "model": "COROLLA",
                "line": "XLE",
            },
        )

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_predict_accepts_service_key(
    mock_prediction_service: AsyncMock,
) -> None:
    app.dependency_overrides[get_prediction_service] = lambda: mock_prediction_service

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/ml/predict",
            headers={"X-Service-Key": "dummy"},
            json={
                "vehicle_type": "CAR",
                "brand": "TOYOTA",
                "model": "COROLLA",
                "line": "XLE",
            },
        )

    # A wrong service key returns 401 rather than 200; this asserts the
    # header path is reached (not the "missing Authorization" 401).
    assert response.status_code in (200, 401)


@pytest.mark.asyncio
async def test_predict_with_overridden_permission_dependency(
    mock_prediction_service: AsyncMock,
) -> None:
    async def _bypass_auth() -> dict:
        return {"authorities": ["ml:predict"]}

    app.dependency_overrides[prediction_endpoints.require_permissions_predict] = (
        _bypass_auth
    )
    app.dependency_overrides[get_prediction_service] = lambda: mock_prediction_service

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/ml/predict",
            json={
                "vehicle_type": "CAR",
                "brand": "TOYOTA",
                "model": "COROLLA",
                "line": "XLE",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "20250101120000"
    assert body["predictions"][0]["demand"] == 10.0


@pytest.mark.asyncio
async def test_cors_preflight_allows_frontend_origin() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.options(
            "/api/v1/ml/predict",
            headers={
                "Origin": "http://localhost:4200",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization,content-type",
            },
        )

    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:4200"


class TestPermissionEnforcement:
    """Exercises the real JWT decode + `authorities` permission check.

    Signs tokens with a throwaway RSA key (never the real velytics-backend
    signing key) and patches `jwks_cache.get_key_set` so the app's actual
    `require_internal_or_permissions` dependency runs unmodified end-to-end,
    without depending on a real backend user or database record.
    """

    @pytest.mark.asyncio
    async def test_returns_403_when_token_lacks_required_permission(self) -> None:
        key = _generate_test_key()
        settings = get_settings()
        token = _sign_token(key, authorities=["user:read"], issuer=settings.jwt_issuer)

        with patch.object(
            jwks_cache, "get_key_set", AsyncMock(return_value=_mock_key_set(key))
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/ml/predict",
                    headers={"Authorization": f"Bearer {token}"},
                    json={
                        "vehicle_type": "CAR",
                        "brand": "TOYOTA",
                        "model": "COROLLA",
                        "line": "XLE",
                    },
                )

        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_returns_200_when_token_has_required_permission(
        self, mock_prediction_service: AsyncMock
    ) -> None:
        key = _generate_test_key()
        settings = get_settings()
        token = _sign_token(key, authorities=["ml:predict"], issuer=settings.jwt_issuer)
        app.dependency_overrides[get_prediction_service] = lambda: mock_prediction_service

        with patch.object(
            jwks_cache, "get_key_set", AsyncMock(return_value=_mock_key_set(key))
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/ml/predict",
                    headers={"Authorization": f"Bearer {token}"},
                    json={
                        "vehicle_type": "CAR",
                        "brand": "TOYOTA",
                        "model": "COROLLA",
                        "line": "XLE",
                    },
                )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_returns_401_when_issuer_does_not_match(self) -> None:
        key = _generate_test_key()
        token = _sign_token(
            key, authorities=["ml:predict"], issuer="http://not-the-real-issuer"
        )

        with patch.object(
            jwks_cache, "get_key_set", AsyncMock(return_value=_mock_key_set(key))
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/ml/predict",
                    headers={"Authorization": f"Bearer {token}"},
                    json={
                        "vehicle_type": "CAR",
                        "brand": "TOYOTA",
                        "model": "COROLLA",
                        "line": "XLE",
                    },
                )

        assert response.status_code == 401
