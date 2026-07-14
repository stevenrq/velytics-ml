"""Tests for VelyticsClient."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.clients.velytics_client import VelyticsClient
from app.core.config import Settings


def _settings() -> Settings:
    return Settings(
        velytics_api_url="http://test-velytics-api/api/v1",
        service_api_key="test-service-key",
    )


def _make_response(data: list[dict], page: int, total_pages: int) -> MagicMock:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "data": data,
        "meta": {"page": page, "limit": 100, "totalItems": len(data), "totalPages": total_pages},
    }
    return response


class TestFetchContracts:
    """fetch_contracts()"""

    @pytest.mark.asyncio
    async def test_stops_after_last_page(self) -> None:
        page1 = [{"id": 1, "contractType": "SALE"}]
        page2 = [{"id": 2, "contractType": "SALE"}]

        mock_client = AsyncMock()
        mock_client.get.side_effect = [
            _make_response(page1, page=1, total_pages=2),
            _make_response(page2, page=2, total_pages=2),
        ]
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        with patch("httpx.AsyncClient", return_value=mock_client):
            client = VelyticsClient(_settings())
            results = await client.fetch_contracts()

        assert len(results) == 2
        assert mock_client.get.call_count == 2

    @pytest.mark.asyncio
    async def test_sends_service_key_header(self) -> None:
        mock_client = AsyncMock()
        mock_client.get.return_value = _make_response([], page=1, total_pages=1)
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        with patch("httpx.AsyncClient", return_value=mock_client):
            client = VelyticsClient(_settings())
            await client.fetch_contracts()

        _, kwargs = mock_client.get.call_args
        assert kwargs["headers"]["X-Service-Key"] == "test-service-key"

    @pytest.mark.asyncio
    async def test_stops_on_empty_data(self) -> None:
        mock_client = AsyncMock()
        mock_client.get.return_value = _make_response([], page=1, total_pages=5)
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        with patch("httpx.AsyncClient", return_value=mock_client):
            client = VelyticsClient(_settings())
            results = await client.fetch_contracts()

        assert results == []
        assert mock_client.get.call_count == 1


class TestLoadTransactions:
    """load_transactions()"""

    @pytest.mark.asyncio
    async def test_maps_vehicle_summary_fields(self) -> None:
        contract = {
            "id": 1,
            "contractType": "SALE",
            "contractStatus": "ACTIVE",
            "clientId": 10,
            "userId": 20,
            "vehicleId": 99,
            "purchasePrice": 100.0,
            "salePrice": 150.0,
            "paymentMethod": "CASH",
            "observations": "",
            "createdAt": "2026-01-10T10:00:00",
            "updatedAt": "2026-01-12T11:00:00",
            "vehicleSummary": {
                "id": 99,
                "type": "CAR",
                "brand": "Toyota",
                "model": "Corolla",
                "line": "XEi",
                "status": "AVAILABLE",
            },
        }

        client = VelyticsClient(_settings())
        with patch.object(
            client, "fetch_contracts", AsyncMock(return_value=[contract])
        ):
            df = await client.load_transactions()

        assert len(df) == 1
        row = df.iloc[0]
        assert row["vehicle_type"] == "CAR"
        assert row["brand"] == "Toyota"
        assert row["model"] == "Corolla"
        assert row["line"] == "XEi"
        assert row["vehicle_status"] == "AVAILABLE"
        assert row["sale_price"] == 150.0
        assert row["purchase_price"] == 100.0

    @pytest.mark.asyncio
    async def test_normalizes_missing_vehicle_summary_to_unknown(self) -> None:
        contract = {
            "id": 2,
            "contractType": "SALE",
            "contractStatus": "ACTIVE",
            "clientId": 10,
            "userId": 20,
            "vehicleId": None,
            "purchasePrice": 100.0,
            "salePrice": 150.0,
            "createdAt": "2026-01-10T10:00:00",
            "updatedAt": "2026-01-12T11:00:00",
        }

        client = VelyticsClient(_settings())
        with patch.object(
            client, "fetch_contracts", AsyncMock(return_value=[contract])
        ):
            df = await client.load_transactions()

        row = df.iloc[0]
        assert row["vehicle_type"] == "UNKNOWN"
        assert row["brand"] == "UNKNOWN"
        assert row["line"] == "UNKNOWN"

    @pytest.mark.asyncio
    async def test_returns_empty_dataframe_when_no_contracts(self) -> None:
        client = VelyticsClient(_settings())
        with patch.object(client, "fetch_contracts", AsyncMock(return_value=[])):
            df = await client.load_transactions()

        assert df.empty
