"""HTTP client for the velytics-backend purchase-sales API.

Fetches purchase/sale contracts (with the embedded `vehicleSummary`)
using the shared `X-Service-Key` and builds the raw transactions
DataFrame consumed by `FeatureEngineering.build_feature_table`.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd

from app.core.config import Settings
from app.ml.normalization import normalize_unknown_alias

logger = logging.getLogger(__name__)

_PAGE_LIMIT = 100


class VelyticsClient:
    """Async client for velytics-backend's purchase-sales endpoint."""

    def __init__(self, settings: Settings) -> None:
        self._base_url = settings.velytics_api_url.rstrip("/")
        self._timeout = settings.request_timeout_seconds
        self._service_key = settings.service_api_key

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {"Accept": "application/json"}
        if self._service_key:
            headers["X-Service-Key"] = self._service_key
        return headers

    async def fetch_contracts(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[Dict[str, Any]]:
        """Fetches all purchase-sale contracts, paginated, with vehicle details."""
        results: List[Dict[str, Any]] = []
        page = 1

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            while True:
                params: Dict[str, Any] = {
                    "page": page,
                    "limit": _PAGE_LIMIT,
                    "detailed": "true",
                }
                if start_date:
                    params["startDate"] = start_date.isoformat()
                if end_date:
                    params["endDate"] = end_date.isoformat()

                response = await client.get(
                    f"{self._base_url}/purchase-sales",
                    params=params,
                    headers=self._headers(),
                )
                response.raise_for_status()
                payload = response.json()
                data = payload.get("data") or []
                if not data:
                    break

                results.extend(data)
                total_pages = (payload.get("meta") or {}).get("totalPages")
                if total_pages is None or page >= total_pages:
                    break
                page += 1

        logger.info("Retrieved %s purchase/sale contracts", len(results))
        return results

    async def load_transactions(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """Loads contracts and maps them into a transactions DataFrame."""
        contracts = await self.fetch_contracts(start_date=start_date, end_date=end_date)
        if not contracts:
            return pd.DataFrame()

        rows: List[Dict[str, Any]] = []
        for contract in contracts:
            vehicle_summary = contract.get("vehicleSummary") or {}

            rows.append(
                {
                    "contract_id": contract.get("id"),
                    "contract_type": contract.get("contractType"),
                    "contract_status": contract.get("contractStatus"),
                    "client_id": contract.get("clientId"),
                    "user_id": contract.get("userId"),
                    "vehicle_id": contract.get("vehicleId"),
                    "purchase_price": contract.get("purchasePrice"),
                    "sale_price": contract.get("salePrice"),
                    "payment_method": contract.get("paymentMethod"),
                    "observations": contract.get("observations"),
                    "created_at": contract.get("createdAt"),
                    "updated_at": contract.get("updatedAt"),
                    "vehicle_type": normalize_unknown_alias(
                        vehicle_summary.get("type")
                    ),
                    "brand": normalize_unknown_alias(vehicle_summary.get("brand")),
                    "model": normalize_unknown_alias(vehicle_summary.get("model")),
                    "line": normalize_unknown_alias(vehicle_summary.get("line")),
                    "vehicle_status": normalize_unknown_alias(
                        vehicle_summary.get("status")
                    ),
                }
            )

        return pd.DataFrame(rows)
