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
        self.weather_override: Dict[str, Any] | None = None
        self.grid_overrides: Dict[int, int] = {}
        self.team_performance_overrides: Dict[str, float] = {}
        self.dnf_overrides: set[int] = set()

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
        if self.weather_override:
            return dict(self.weather_override)
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
            grid_position = self.grid_overrides.get(
                driver["driver_id"], driver.get("grid_position")
            )
            features = self.feature_extractor.extract(
                driver_id=driver["driver_id"],
                race_id=race_id,
                grid_position=grid_position,
                use_latest_data=use_latest_data,
            )
            driver_features.append(features)

        X = pd.DataFrame(driver_features).fillna(0)
        if "wet_race_probability" in X.columns and weather_forecast.get("rain_probability") is not None:
            X["wet_race_probability"] = weather_forecast["rain_probability"]
        if "weather_change_during_race_probability" in X.columns and weather_forecast.get("rain_probability") is not None:
            X["weather_change_during_race_probability"] = min(
                1.0, float(weather_forecast["rain_probability"]) * 1.2
            )
        if self.team_performance_overrides and "team_performance_score" in X.columns:
            for idx, driver in enumerate(entry_list):
                team_name = driver.get("team_name")
                if team_name in self.team_performance_overrides:
                    X.loc[idx, "team_performance_score"] = self.team_performance_overrides[team_name]

        predictions = {
            "position": self.models["xgb_position"].predict(X),
            "time": self.models["lgb_time"].predict(X),
            "dnf_probability": self.models["xgb_dnf"].predict_proba(X)[:, 1],
            "podium_probability": self.models["xgb_podium"].predict_proba(X),
            "rf_position": self.models["rf_position"].predict(X),
            "nn_position": self.models["nn_position"].predict(X, verbose=0).ravel(),
        }
        if self.dnf_overrides:
            for idx, driver in enumerate(entry_list):
                if driver["driver_id"] in self.dnf_overrides:
                    predictions["dnf_probability"][idx] = 1.0

        final_position = (
            predictions["position"] * 0.35
            + predictions["rf_position"] * 0.30
            + predictions["nn_position"] * 0.25
            + predictions["position"] * 0.10
        )

        final_position = self.resolve_position_conflicts(final_position)
        if self.dnf_overrides:
            for idx, driver in enumerate(entry_list):
                if driver["driver_id"] in self.dnf_overrides:
                    final_position[idx] = 20
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


class LivePredictionUpdater:
    def __init__(self, predictor: F1RacePredictor) -> None:
        self.predictor = predictor
        self.cache: Dict[int, Dict[str, Any]] = {}

    def update_from_practice(
        self,
        race_id: int,
        fp1_times: pd.DataFrame,
        fp2_times: pd.DataFrame,
        fp3_times: pd.DataFrame,
    ) -> Dict[str, Any]:
        long_run_pace = self.analyze_practice_pace(fp2_times, fp3_times)
        for team, pace in long_run_pace.items():
            self.update_team_performance(team, pace)
        updated = self.predictor.predict_race(race_id, use_latest_data=True)
        self.cache[race_id] = updated
        return updated

    def update_from_qualifying(
        self, race_id: int, quali_results: Dict[int, int]
    ) -> Dict[str, Any]:
        for driver_id, grid_pos in quali_results.items():
            self.update_driver_grid(driver_id, grid_pos)
        updated = self.predictor.predict_race(race_id, use_latest_data=True)
        self.cache[race_id] = updated
        return updated

    def update_from_weather_change(
        self, race_id: int, new_forecast: Dict[str, Any]
    ) -> Dict[str, Any]:
        current = self.cache.get(race_id)
        if current is None:
            updated = self.predictor.predict_race(race_id)
            self.cache[race_id] = updated
            return updated
        old_rain_prob = current["metadata"].get("weather_forecast", {}).get("rain_probability")
        new_rain_prob = new_forecast.get("rain_probability")
        if old_rain_prob is None or new_rain_prob is None:
            return current
        if abs(new_rain_prob - old_rain_prob) > 0.15:
            updated = self.predictor.predict_race(race_id, use_latest_data=True)
            self.cache[race_id] = updated
            return updated
        return current

    def post_race_learning(
        self, race_id: int, actual_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]] | None:
        predicted = self.cache.get(race_id)
        if predicted is None:
            return None
        errors = []
        for pred, actual in zip(predicted["predictions"], actual_results):
            errors.append(
                {
                    "driver": pred["driver"],
                    "predicted_pos": pred["position"],
                    "actual_pos": actual["position"],
                    "position_error": abs(pred["position"] - actual["position"]),
                    "dnf_predicted": pred["dnf_probability"],
                    "dnf_actual": actual.get("dnf"),
                }
            )
        self.store_prediction_errors(race_id, errors)
        if self.get_races_since_retrain() >= 5:
            self.trigger_retraining()
        return errors

    def analyze_practice_pace(
        self, fp2_times: pd.DataFrame, fp3_times: pd.DataFrame
    ) -> Dict[str, float]:
        combined = pd.concat([fp2_times, fp3_times], ignore_index=True)
        if "team_name" not in combined or "lap_time" not in combined:
            return {}
        pace = combined.groupby("team_name")["lap_time"].median().to_dict()
        return {team: float(value) for team, value in pace.items()}

    def update_team_performance(self, team_name: str, pace: float) -> None:
        query = sa.text(
            """
            UPDATE team_features tf
            SET performance_trend_last_5_races = :pace
            FROM teams t
            WHERE tf.team_id = t.team_id AND t.team_name = :team_name
            """
        )
        with self.predictor.engine.begin() as conn:
            conn.execute(query, {"team_name": team_name, "pace": pace})

    def update_driver_grid(self, driver_id: int, grid_position: int) -> None:
        query = sa.text(
            """
            UPDATE situational_features
            SET grid_position = :grid_position
            WHERE driver_id = :driver_id
            """
        )
        with self.predictor.engine.begin() as conn:
            conn.execute(query, {"driver_id": driver_id, "grid_position": grid_position})

    def store_prediction_errors(self, race_id: int, errors: List[Dict[str, Any]]) -> None:
        payload = pd.DataFrame(errors)
        path = os.path.join(self.predictor.model_dir, f"prediction_errors_race_{race_id}.csv")
        payload.to_csv(path, index=False)

    def get_races_since_retrain(self) -> int:
        marker = os.path.join(self.predictor.model_dir, "last_retrain_marker.txt")
        if not os.path.exists(marker):
            return 999
        try:
            with open(marker, "r", encoding="utf-8") as handle:
                return int(handle.read().strip())
        except ValueError:
            return 999

    def trigger_retraining(self) -> None:
        marker = os.path.join(self.predictor.model_dir, "last_retrain_marker.txt")
        with open(marker, "w", encoding="utf-8") as handle:
            handle.write("0")


class ScenarioSimulator:
    def __init__(self, predictor: F1RacePredictor) -> None:
        self.predictor = predictor

    def simulate_weather_scenarios(self, race_id: int) -> Dict[str, Any]:
        scenarios = {
            "dry": {"rain_probability": 0.0, "temp": 25},
            "mixed": {"rain_probability": 0.3, "temp": 22},
            "wet": {"rain_probability": 0.8, "temp": 18},
            "very_wet": {"rain_probability": 1.0, "temp": 16},
        }
        results = {}
        for scenario_name, weather in scenarios.items():
            self.predictor.weather_override = weather
            pred = self.predictor.predict_race(race_id)
            results[scenario_name] = pred["predictions"]
        self.predictor.weather_override = None
        return self.compare_scenarios(results)

    def simulate_grid_penalty(
        self, race_id: int, driver_id: int, penalty_positions: int
    ) -> Dict[str, Any]:
        base_pred = self.predictor.predict_race(race_id)
        entry_list = self.predictor.get_entry_list(race_id)
        entry = next((driver for driver in entry_list if driver["driver_id"] == driver_id), None)
        if entry and entry.get("grid_position") is not None:
            current_grid = entry["grid_position"]
            self.predictor.grid_overrides[driver_id] = min(
                20, current_grid + penalty_positions
            )
        penalty_pred = self.predictor.predict_race(race_id)
        self.predictor.grid_overrides.pop(driver_id, None)
        return self.compare_predictions(base_pred, penalty_pred)

    def simulate_dnf(self, race_id: int, driver_ids: List[int]) -> Dict[str, Any]:
        base_pred = self.predictor.predict_race(race_id)
        self.predictor.dnf_overrides = set(driver_ids)
        dnf_pred = self.predictor.predict_race(race_id)
        self.predictor.dnf_overrides = set()
        return self.compare_predictions(base_pred, dnf_pred)

    def simulate_team_upgrade(
        self, race_id: int, team_name: str, performance_boost: float
    ) -> Dict[str, Any]:
        base_pred = self.predictor.predict_race(race_id)
        self.predictor.team_performance_overrides[team_name] = performance_boost
        upgraded_pred = self.predictor.predict_race(race_id)
        self.predictor.team_performance_overrides.pop(team_name, None)
        return self.compare_predictions(base_pred, upgraded_pred)

    def monte_carlo_simulation(self, race_id: int, n_simulations: int = 1000) -> Dict[str, Any]:
        base_pred = self.predictor.predict_race(race_id)
        results = {pred["driver"]: [] for pred in base_pred["predictions"]}
        for _ in range(n_simulations):
            sim_result = self.simulate_single_race(base_pred)
            for driver, position in sim_result.items():
                results[driver].append(position)
        predictions = {}
        for driver, positions in results.items():
            positions_arr = np.array(positions)
            predictions[driver] = {
                "mean_position": float(np.mean(positions_arr)),
                "median_position": float(np.median(positions_arr)),
                "p10_position": float(np.percentile(positions_arr, 10)),
                "p90_position": float(np.percentile(positions_arr, 90)),
                "win_probability": float(np.mean(positions_arr == 1)),
                "podium_probability": float(np.mean(positions_arr <= 3)),
                "points_probability": float(np.mean(positions_arr <= 10)),
            }
        return predictions

    def simulate_single_race(self, base_pred: Dict[str, Any]) -> Dict[str, int]:
        predictions = base_pred["predictions"]
        simulated = {}
        for pred in predictions:
            noise = np.random.normal(0, 1.5)
            dnf_roll = np.random.rand() < pred["dnf_probability"]
            if dnf_roll:
                simulated[pred["driver"]] = 20
            else:
                simulated[pred["driver"]] = int(
                    np.clip(round(pred["position"] + noise), 1, 20)
                )
        # Resolve collisions
        final_positions = self.resolve_conflicts(simulated)
        return final_positions

    def resolve_conflicts(self, positions: Dict[str, int]) -> Dict[str, int]:
        assigned = {}
        used = set()
        for driver, pos in sorted(positions.items(), key=lambda x: x[1]):
            final_pos = pos
            if final_pos in used:
                available = [p for p in range(1, 21) if p not in used]
                if available:
                    final_pos = min(available, key=lambda x: abs(x - pos))
            used.add(final_pos)
            assigned[driver] = final_pos
        return assigned

    def compare_predictions(
        self, base_pred: Dict[str, Any], scenario_pred: Dict[str, Any]
    ) -> Dict[str, Any]:
        base_map = {p["driver"]: p["position"] for p in base_pred["predictions"]}
        scenario_map = {p["driver"]: p["position"] for p in scenario_pred["predictions"]}
        changes = {
            driver: scenario_map[driver] - base_map.get(driver, scenario_map[driver])
            for driver in scenario_map
        }
        return {"base": base_pred, "scenario": scenario_pred, "delta": changes}

    def compare_scenarios(self, results: Dict[str, Any]) -> Dict[str, Any]:
        drivers = {pred["driver"] for preds in results.values() for pred in preds}
        comparison = {}
        for driver in drivers:
            comparison[driver] = {
                scenario: next(
                    (pred["position"] for pred in preds if pred["driver"] == driver), None
                )
                for scenario, preds in results.items()
            }
        return comparison
