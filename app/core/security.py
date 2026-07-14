"""Auth dependencies: JWT (RS256, JWKS) and service-to-service key.

velytics-backend issues RS256 JWTs and exposes a plain JWKS endpoint
(no OIDC discovery document). Permissions travel in the `authorities`
claim. Service-to-service calls authenticate with the `X-Service-Key`
header instead of a user token.
"""

from __future__ import annotations

import hmac
import time
from typing import Any, Dict, Iterable, Set

import httpx
from authlib.jose import JoseError, JsonWebKey, JsonWebToken
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import get_settings

bearer_scheme = HTTPBearer(auto_error=False)
jwt_backend = JsonWebToken(["RS256"])


class JWKSCache:
    """Simple TTL cache for the JWKS public key set."""

    def __init__(self, ttl_seconds: float = 3600.0) -> None:
        self._cached: Dict[str, Any] = {}
        self._expires_at: float = 0.0
        self._ttl = ttl_seconds

    async def get_key_set(self, jwks_url: str, timeout: float) -> Any:
        if not jwks_url:
            return None

        now = time.time()
        if now >= self._expires_at:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(jwks_url)
                response.raise_for_status()
                payload = response.json()
                self._cached = payload
                self._expires_at = now + self._ttl

        if not self._cached:
            return None
        return JsonWebKey.import_key_set(self._cached)


jwks_cache = JWKSCache()


async def decode_token(raw_token: str) -> dict:
    """Decodes and validates a JWT against the backend's JWKS."""
    settings = get_settings()

    key_set = await jwks_cache.get_key_set(
        settings.jwks_url, settings.request_timeout_seconds
    )
    if not key_set:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No public key configured to validate the token.",
        )

    claims_options: Dict[str, Dict[str, Any]] = {
        "exp": {"essential": True},
        "iss": {"essential": True, "values": [settings.jwt_issuer]},
    }

    try:
        claims = jwt_backend.decode(raw_token, key_set, claims_options=claims_options)
        claims.validate()
        return dict(claims)
    except JoseError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
        ) from exc


def _extract_permissions(claims: dict) -> Set[str]:
    """Extracts permissions from the `authorities` claim."""
    raw = claims.get("authorities")
    if raw is None:
        return set()
    if isinstance(raw, str):
        parts = [
            token.strip() for token in raw.replace(",", " ").split() if token.strip()
        ]
        return set(parts)
    if isinstance(raw, (list, tuple, set)):
        return {str(item) for item in raw if item}
    return set()


async def require_token(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> dict:
    """Requires a valid Bearer token."""
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header.",
        )
    return await decode_token(credentials.credentials)


def require_permissions(required: Iterable[str]):
    """Dependency factory requiring at least one of the given permissions."""
    required_set = {permission for permission in required if permission}

    def dependency(claims: dict = Depends(require_token)) -> dict:
        permissions = _extract_permissions(claims)
        if required_set and not permissions.intersection(required_set):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Missing required permission to access this resource.",
            )
        return claims

    return dependency


def require_internal_or_permissions(required: Iterable[str]):
    """Dependency factory accepting either X-Service-Key or a scoped JWT."""
    required_set = {permission for permission in required if permission}

    async def dependency(
        request: Request,
        credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    ) -> dict:
        settings = get_settings()
        service_key = request.headers.get("X-Service-Key")

        if service_key:
            if settings.service_api_key and hmac.compare_digest(
                service_key, settings.service_api_key
            ):
                return {"authorities": list(required_set)}
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid service key.",
            )

        if not credentials or not credentials.credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing Authorization header.",
            )

        claims = await decode_token(credentials.credentials)
        permissions = _extract_permissions(claims)
        if required_set and not permissions.intersection(required_set):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Missing required permission to access this resource.",
            )
        return claims

    return dependency
