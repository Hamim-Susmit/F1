# F1 2026 Prediction System

## Overview
This repository contains the data ingestion, model training, prediction API, and dashboard
for the F1 2026 prediction system.

## Installation
```bash
./setup.sh
```

Or install directly:
```bash
pip install -r requirements.txt
```

## Quick Start
1. Copy the environment template:
   ```bash
   cp .env.example .env
   ```
2. Start the services:
   ```bash
   docker-compose up --build
   ```
3. Open the API:
   - http://localhost:8000/health
4. Open the dashboard:
   - http://localhost:8501

## API Documentation
Base URL: `http://localhost:8000`

| Endpoint | Method | Description | Auth |
| --- | --- | --- | --- |
| `/` | GET | Service info | No |
| `/health` | GET | Health check | No |
| `/metrics` | GET | Prometheus metrics | No |
| `/predict` | POST | Generate predictions | `X-API-Key` |
| `/races/upcoming` | GET | Upcoming races | `X-API-Key` |
| `/model/performance` | GET | Model performance | `X-API-Key` |
| `/drivers` | GET | Driver list | `X-API-Key` |
| `/teams` | GET | Team list | `X-API-Key` |
| `/scenarios/simulate` | POST | Scenario comparison | `X-API-Key` |
| `/update/qualifying` | POST | Update grid | `X-API-Key` |

### Example request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${API_KEY}" \
  -d '{"race_id": 202601}'
```

## Configuration
Environment variables (see `.env.example`):
- `DATABASE_URL` - PostgreSQL connection string
- `DB_PASSWORD` - database password
- `MODEL_DIR` - directory containing model artifacts
- `FASTF1_CACHE` - FastF1 cache directory
- `OPENWEATHER_API_KEY` - weather API key
- `WEATHER_BASE_URL` - weather API base URL
- `API_KEY` - API authentication key
- `ALLOWED_ORIGINS` - CORS configuration (must be explicit when `API_KEY` is set)
- `REDIS_URL` - Redis connection string

## Troubleshooting
- **Missing models**: ensure model files are in `MODEL_DIR` (e.g. `xgb_position.joblib`).
- **Database connection errors**: verify `DATABASE_URL` and that PostgreSQL is running.
- **API unauthorized**: set `API_KEY` and include `X-API-Key` in requests.

## Contributing
1. Create a feature branch.
2. Run tests (`pytest`).
3. Submit a pull request with a clear description of changes.
