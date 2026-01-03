from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import shap
import sqlalchemy as sa
from tensorflow import keras

from f1_data_ingest import extract_race_features


class FeatureExtractor:
    def __init__(self, engine: sa.Engine) -> None:
        self.engine = engine

    def extract(
        self,
        driver_id: int,
        race_id: int,
        grid_position: int | None = None,
        use_latest_data: bool = True,
    ) -> Dict[str, Any]:
        with self.engine.begin() as conn:
            features = extract_race_features(conn, race_id=race_id, driver_id=driver_id)
        if grid_position is not None:
            features["grid_position"] = grid_position
        return features


class F1RacePredictor:
    def __init__(self, database_url: str, model_dir: str = "models") -> None:
        if not database_url:
            raise ValueError("DATABASE_URL is required")
        self.engine = sa.create_engine(database_url, future=True)
        self.model_dir = model_dir
        self.models = {
            "xgb_position": joblib.load(os.path.join(model_dir, "xgb_position.joblib")),
            "lgb_time": joblib.load(os.path.join(model_dir, "lgb_time.joblib")),
            "rf_position": joblib.load(os.path.join(model_dir, "rf_position.joblib")),
            "nn_position": keras.models.load_model(os.path.join(model_dir, "nn_position.keras")),
            "xgb_dnf": joblib.load(os.path.join(model_dir, "xgb_dnf.joblib")),
            "xgb_podium": joblib.load(os.path.join(model_dir, "xgb_podium.joblib")),
        }
        self.feature_extractor = FeatureExtractor(self.engine)

    def get_race_info(self, race_id: int) -> Dict[str, Any]:
        query = sa.text(
            """
            SELECT race_id, season, round, race_name, circuit_name, race_date
            FROM races
            WHERE race_id = :race_id
            """
        )
        with self.engine.begin() as conn:
            row = conn.execute(query, {"race_id": race_id}).mappings().first()
        if not row:
            raise ValueError(f"Race {race_id} not found")
        return dict(row)

    def get_weather_forecast(self, race_id: int) -> Dict[str, Any]:
        forecast_query = sa.text(
            """
            SELECT air_temp, track_temp, rain_probability, rain_amount, humidity,
                   wind_speed, wind_direction, conditions, forecast_time
            FROM race_weather_forecasts
            WHERE race_id = :race_id AND session_type = 'Race'
            ORDER BY forecast_time DESC
            LIMIT 1
            """
        )
        fallback_query = sa.text(
            """
            SELECT AVG(air_temp) AS air_temp,
                   AVG(track_temp) AS track_temp,
                   AVG(rain_probability) AS rain_probability,
                   AVG(rain_amount) AS rain_amount,
                   AVG(humidity) AS humidity,
                   AVG(wind_speed) AS wind_speed,
                   AVG(wind_direction) AS wind_direction,
                   MAX(conditions) AS conditions
            FROM race_weather
            WHERE race_id = :race_id AND session_type = 'Race'
            """
        )
        with self.engine.begin() as conn:
            row = conn.execute(forecast_query, {"race_id": race_id}).mappings().first()
            if row:
                return dict(row)
            row = conn.execute(fallback_query, {"race_id": race_id}).mappings().first()
            return dict(row) if row else {}

    def get_entry_list(self, race_id: int) -> List[Dict[str, Any]]:
        race_results_query = sa.text(
            """
            SELECT rr.driver_id,
                   d.full_name AS driver_name,
                   t.team_name,
                   rr.grid_position
            FROM race_results rr
            JOIN drivers d ON rr.driver_id = d.driver_id
            JOIN teams t ON rr.team_id = t.team_id
            WHERE rr.race_id = :race_id
            ORDER BY rr.grid_position NULLS LAST
            """
        )
        qualifying_query = sa.text(
            """
            SELECT qr.driver_id,
                   d.full_name AS driver_name,
                   qr.grid_position,
                   (
                       SELECT t.team_name
                       FROM race_results rr
                       JOIN races r ON rr.race_id = r.race_id
                       JOIN teams t ON rr.team_id = t.team_id
                       WHERE rr.driver_id = qr.driver_id
                       ORDER BY r.season DESC, r.round DESC
                       LIMIT 1
                   ) AS team_name
            FROM qualifying_results qr
            JOIN drivers d ON qr.driver_id = d.driver_id
            WHERE qr.race_id = :race_id
            ORDER BY qr.grid_position NULLS LAST
            """
        )
        with self.engine.begin() as conn:
            rows = conn.execute(race_results_query, {"race_id": race_id}).mappings().all()
            if rows:
                return [dict(row) for row in rows]
            rows = conn.execute(qualifying_query, {"race_id": race_id}).mappings().all()
            if rows:
                return [dict(row) for row in rows]
        raise ValueError(f"No entry list found for race {race_id}")

    def predict_race(self, race_id: int, use_latest_data: bool = True) -> Dict[str, Any]:
        race_info = self.get_race_info(race_id)
        weather_forecast = self.get_weather_forecast(race_id)
        entry_list = self.get_entry_list(race_id)

        driver_features: List[Dict[str, Any]] = []
        for driver in entry_list:
            features = self.feature_extractor.extract(
                driver_id=driver["driver_id"],
                race_id=race_id,
                grid_position=driver.get("grid_position"),
                use_latest_data=use_latest_data,
            )
            driver_features.append(features)

        X = pd.DataFrame(driver_features).fillna(0)

        predictions = {
            "position": self.models["xgb_position"].predict(X),
            "time": self.models["lgb_time"].predict(X),
            "dnf_probability": self.models["xgb_dnf"].predict_proba(X)[:, 1],
            "podium_probability": self.models["xgb_podium"].predict_proba(X),
            "rf_position": self.models["rf_position"].predict(X),
            "nn_position": self.models["nn_position"].predict(X, verbose=0).ravel(),
        }

        final_position = (
            predictions["position"] * 0.35
            + predictions["rf_position"] * 0.30
            + predictions["nn_position"] * 0.25
            + predictions["position"] * 0.10
        )

        final_position = self.resolve_position_conflicts(final_position)
        confidence = self.calculate_confidence(predictions)
        explanations = self.explain_predictions(X, entry_list)

        results = []
        for i, driver in enumerate(entry_list):
            results.append(
                {
                    "position": int(final_position[i]),
                    "driver": driver["driver_name"],
                    "team": driver["team_name"],
                    "predicted_time": float(predictions["time"][i]),
                    "dnf_probability": float(predictions["dnf_probability"][i]),
                    "podium_probability": float(predictions["podium_probability"][i][0:3].sum()),
                    "confidence": float(confidence[i]),
                    "explanation": explanations[i],
                }
            )

        results = sorted(results, key=lambda x: x["position"])

        return {
            "race": race_info,
            "predictions": results,
            "metadata": {
                "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
                "weather_forecast": weather_forecast,
                "model_versions": self.get_model_versions(),
            },
        }

    def resolve_position_conflicts(self, predictions: np.ndarray) -> np.ndarray:
        positions = np.round(predictions).astype(int)
        positions = np.clip(positions, 1, 20)
        used = set()
        assigned: Dict[int, int] = {}

        for idx, (predicted, raw) in sorted(
            enumerate(zip(positions, predictions)), key=lambda x: x[1][1]
        ):
            pos = int(predicted)
            if pos in used:
                candidates = [p for p in range(1, 21) if p not in used]
                pos = min(candidates, key=lambda x: abs(x - raw))
            used.add(pos)
            assigned[idx] = pos

        return np.array([assigned[i] for i in range(len(predictions))])

    def calculate_confidence(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        position_preds = np.column_stack(
            [
                predictions["position"],
                predictions["rf_position"],
                predictions["nn_position"],
            ]
        )
        model_agreement = 1 / (1 + np.std(position_preds, axis=1))
        return model_agreement / model_agreement.max() * 100

    def explain_predictions(self, features: pd.DataFrame, drivers: List[Dict[str, Any]]) -> List[str]:
        explainer = shap.TreeExplainer(self.models["xgb_position"])
        shap_values = explainer.shap_values(features)
        explanations = []
        for i, _driver in enumerate(drivers):
            feature_impacts = (
                pd.DataFrame(
                    {"feature": features.columns, "impact": shap_values[i]}
                )
                .sort_values("impact", key=lambda s: s.abs(), ascending=False)
                .head(3)
            )
            explanation = "Key factors: " + ", ".join(
                f"{row.feature} ({row.impact:+.2f})" for row in feature_impacts.itertuples()
            )
            explanations.append(explanation)
        return explanations

    def get_model_versions(self) -> Dict[str, str]:
        versions = {}
        for name, filename in {
            "xgb_position": "xgb_position.joblib",
            "lgb_time": "lgb_time.joblib",
            "rf_position": "rf_position.joblib",
            "nn_position": "nn_position.keras",
            "xgb_dnf": "xgb_dnf.joblib",
            "xgb_podium": "xgb_podium.joblib",
        }.items():
            path = os.path.join(self.model_dir, filename)
            if os.path.exists(path):
                versions[name] = datetime.fromtimestamp(
                    os.path.getmtime(path), tz=timezone.utc
                ).isoformat()
        return versions
