from __future__ import annotations

import logging
import os
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Gauge, Histogram
from prometheus_client.exposition import generate_latest
from prometheus_client import CONTENT_TYPE_LATEST
from pythonjsonlogger import jsonlogger
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import Response
import sqlalchemy as sa

from f1_prediction_pipeline import (
    F1RacePredictor,
    LivePredictionUpdater,
    ModelEvaluator,
    ScenarioSimulator,
)


app = FastAPI(title="F1 Prediction API", version="1.0")

allowed_origins = [origin.strip() for origin in os.environ.get("ALLOWED_ORIGINS", "*").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("f1_api")
log_handler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
log_handler.setFormatter(formatter)
logger.addHandler(log_handler)
logger.setLevel(logging.INFO)

prediction_requests = Counter(
    "f1_prediction_requests_total",
    "Total prediction requests",
    ["race", "status"],
)
prediction_latency = Histogram(
    "f1_prediction_latency_seconds",
    "Prediction latency",
)
model_accuracy = Gauge(
    "f1_model_accuracy",
    "Current model accuracy",
    ["metric_type"],
)


class PredictionRequest(BaseModel):
    race_id: int
    include_explanations: bool = True
    include_confidence: bool = True
    scenario: Optional[str] = None


class DriverPrediction(BaseModel):
    position: int
    driver: str
    team: str
    confidence: float
    dnf_probability: float
    podium_probability: float
    explanation: Optional[str]


class RacePrediction(BaseModel):
    race_name: str
    circuit: str
    date: str
    predictions: List[DriverPrediction]
    metadata: Dict[str, Any]


def verify_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    required_key = os.environ.get("API_KEY")
    if not required_key:
        return
    if x_api_key != required_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


@lru_cache(maxsize=1)
def get_predictor() -> F1RacePredictor:
    database_url = os.environ.get("DATABASE_URL")
    model_dir = os.environ.get("MODEL_DIR", "models")
    if not database_url:
        raise HTTPException(status_code=500, detail="DATABASE_URL must be set")
    return F1RacePredictor(database_url=database_url, model_dir=model_dir)


def get_engine() -> sa.Engine:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise HTTPException(status_code=500, detail="DATABASE_URL must be set")
    return sa.create_engine(database_url, future=True)


@app.get("/")
async def root() -> Dict[str, str]:
    return {"service": "F1 Prediction API", "status": "ok"}


@app.post("/predict", response_model=RacePrediction, dependencies=[Depends(verify_api_key)])
async def predict_race(request: PredictionRequest) -> RacePrediction:
    start_time = time.time()
    try:
        predictor = get_predictor()
        if request.scenario:
            predictor.set_weather_scenario(request.scenario)
        predictions = predictor.predict_race(race_id=request.race_id, use_latest_data=True)
        if not request.include_explanations:
            for item in predictions["predictions"]:
                item["explanation"] = None
        if not request.include_confidence:
            for item in predictions["predictions"]:
                item["confidence"] = 0.0
        race = predictions["race"]
        prediction_requests.labels(race=str(request.race_id), status="success").inc()
        prediction_latency.observe(time.time() - start_time)
        logger.info(
            "Prediction generated",
            extra={
                "race_id": request.race_id,
                "predicted_winner": predictions["predictions"][0]["driver"],
                "confidence": predictions["predictions"][0]["confidence"],
                "model_version": predictions["metadata"]["model_versions"].get("xgb_position"),
                "latency_ms": int((time.time() - start_time) * 1000),
            },
        )
        return RacePrediction(
            race_name=race["race_name"],
            circuit=race["circuit_name"],
            date=str(race["race_date"]),
            predictions=[DriverPrediction(**item) for item in predictions["predictions"]],
            metadata=predictions["metadata"],
        )
    except Exception as exc:
        prediction_requests.labels(race=str(request.race_id), status="error").inc()
        prediction_latency.observe(time.time() - start_time)
        logger.exception("Prediction failed", extra={"race_id": request.race_id})
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/races/upcoming", dependencies=[Depends(verify_api_key)])
async def get_upcoming_races() -> Dict[str, Any]:
    return {
        "races": [
            {"race_id": 202601, "date": "2026-03-06", "circuit": "Albert Park"},
            {"race_id": 202602, "date": "2026-03-13", "circuit": "Shanghai"},
        ]
    }


@app.get("/model/performance", dependencies=[Depends(verify_api_key)])
async def get_model_performance() -> Dict[str, Any]:
    evaluator = ModelEvaluator()
    report = evaluator.generate_report(season=2026)
    model_accuracy.labels(metric_type="winner").set(report["winner_accuracy"])
    model_accuracy.labels(metric_type="top3").set(report["top3_accuracy"])
    model_accuracy.labels(metric_type="mae").set(report["avg_position_mae"])
    return report


@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/scenarios/simulate", dependencies=[Depends(verify_api_key)])
async def simulate_scenarios(race_id: int, scenarios: List[str]) -> Dict[str, Any]:
    predictor = get_predictor()
    simulator = ScenarioSimulator(predictor)
    return simulator.simulate_weather_scenarios(race_id, scenarios or None)


@app.post("/update/qualifying", dependencies=[Depends(verify_api_key)])
async def update_from_qualifying(
    race_id: int, quali_results: Dict[int, int]
) -> Dict[str, Any]:
    predictor = get_predictor()
    updater = LivePredictionUpdater(predictor)
    return updater.update_from_qualifying(race_id, quali_results)


@app.get("/drivers", dependencies=[Depends(verify_api_key)])
async def list_drivers() -> Dict[str, Any]:
    engine = get_engine()
    query = sa.text("SELECT driver_id, full_name, nationality FROM drivers ORDER BY full_name")
    with engine.begin() as conn:
        rows = conn.execute(query).mappings().all()
    return {"drivers": [dict(row) for row in rows]}


@app.get("/teams", dependencies=[Depends(verify_api_key)])
async def list_teams() -> Dict[str, Any]:
    engine = get_engine()
    query = sa.text("SELECT team_id, team_name, engine_manufacturer FROM teams ORDER BY team_name")
    with engine.begin() as conn:
        rows = conn.execute(query).mappings().all()
    return {"teams": [dict(row) for row in rows]}
