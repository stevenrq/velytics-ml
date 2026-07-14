# velytics-ml

Vehicle demand-forecasting FastAPI service for Velytics. Predicts monthly
sales demand per vehicle segment (type/brand/model/line) using historical
purchase-sale contracts pulled from `velytics-backend`.

## Stack

- FastAPI + Pydantic v2, uv-managed (`pyproject.toml` / `uv.lock`)
- SQLAlchemy 2 (async, asyncpg) + Alembic, PostgreSQL
- scikit-learn / XGBoost (RandomForest or XGBoost, selected via
  RandomizedSearchCV + TimeSeriesSplit)
- Auth: RS256 JWT validated against `velytics-backend`'s JWKS endpoint,
  plus a shared `X-Service-Key` for service-to-service calls

## Setup

```bash
uv sync
cp .env.example .env   # fill in SERVICE_API_KEY (same value as velytics-backend)

# One-off: create the ml database in the shared velytics-postgres container
docker exec velytics-postgres createdb -U velytics velytics_ml

uv run fastapi dev app/main.py --port 8000
```

Alembic migrations run automatically on startup
(`DATABASE_RUN_MIGRATIONS=true`).

## API

All endpoints are under `/api/v1/ml`, plus an unprefixed `/health`:

| Method | Path | Permission |
|---|---|---|
| POST | `/api/v1/ml/predict` | `ml:predict` |
| POST | `/api/v1/ml/predict-with-history` | `ml:predict` |
| POST | `/api/v1/ml/retrain` | `ml:retrain` |
| GET | `/api/v1/ml/models/latest` | `ml:models` |
| GET | `/api/v1/ml/models/latest/feature-importance` | `ml:models` |
| POST | `/api/v1/ml/actuals` | `ml:retrain` |
| GET | `/api/v1/ml/drift-report` | `ml:models` |
| GET | `/health` | none |

Auth: `Authorization: Bearer <JWT>` issued by velytics-backend
(`authorities` claim must include the required permission), or header
`X-Service-Key: <SERVICE_API_KEY>` on `/predict`, `/predict-with-history`,
`/retrain` and `/actuals`.

## Tests

```bash
uv run pytest
```
