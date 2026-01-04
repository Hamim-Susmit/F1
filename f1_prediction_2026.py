import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.linear_model import SGDRegressor


LOG = logging.getLogger("f1_2026")


@dataclass
class PredictionResult:
    prediction: np.ndarray
    uncertainty: float
    confidence_interval: Tuple[float, float]
    uncertainty_level: str


def load_pretrained_model(path: str):
    return joblib.load(path)


def train_adaptation_layer(
    base_predictions: np.ndarray,
    testing_features: np.ndarray,
    targets: np.ndarray,
):
    X = np.column_stack([base_predictions, testing_features])
    model = SGDRegressor(random_state=42)
    model.fit(X, targets)
    return model


def estimate_uncertainty(predictions: np.ndarray) -> float:
    if predictions.size == 0:
        return 0.0
    return float(np.std(predictions))


def confidence_interval(prediction: float, uncertainty: float) -> Tuple[float, float]:
    return prediction - 2 * uncertainty, prediction + 2 * uncertainty


class RegulationImpactModel:
    def __init__(self, model: Optional[Any] = None) -> None:
        self.model = model
        self.factors = {
            "engineering_strength": self.load_team_engineering_scores(),
            "budget_efficiency": self.load_budget_cap_data(),
            "aero_expertise": self.load_aero_expertise(),
            "pu_manufacturer_advantage": self.load_pu_data(),
        }

    def load_team_engineering_scores(self) -> Dict[str, float]:
        return {
            "Mercedes": 0.9,
            "Red Bull": 0.92,
            "McLaren": 0.86,
            "Ferrari": 0.85,
            "Aston Martin": 0.82,
        }

    def load_budget_cap_data(self) -> Dict[str, float]:
        return {
            "Mercedes": 0.8,
            "Red Bull": 0.78,
            "McLaren": 0.75,
            "Ferrari": 0.77,
            "Aston Martin": 0.7,
        }

    def load_aero_expertise(self) -> Dict[str, float]:
        return {
            "Mercedes": 0.82,
            "Red Bull": 0.95,
            "McLaren": 0.9,
            "Ferrari": 0.88,
            "Aston Martin": 0.85,
        }

    def load_pu_data(self) -> Dict[str, float]:
        return {
            "Mercedes": 0.85,
            "Red Bull": 0.8,
            "McLaren": 0.82,
            "Ferrari": 0.9,
            "Aston Martin": 0.88,
            "Audi": 0.9,
            "Honda": 0.88,
        }

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.ones(features.shape[0])
        return self.model.predict(features)

    def predict_2026_performance(self, team: str, season_2025_performance: float) -> float:
        aero_advantage = self.factors["aero_expertise"].get(team, 0.5) * 0.3
        engineering_advantage = self.factors["engineering_strength"].get(team, 0.5) * 0.25
        budget_advantage = self.factors["budget_efficiency"].get(team, 0.5) * 0.15
        pu_advantage = self.factors["pu_manufacturer_advantage"].get(team, 0.5) * 0.20
        newey_factor = 1.5 if team == "Aston Martin" else 1.0
        random_factor = float(np.random.normal(1.0, 0.15))
        adaptation_score = (
            aero_advantage + engineering_advantage + budget_advantage + pu_advantage
        ) * newey_factor * random_factor
        predicted_2026 = season_2025_performance * 0.3 + adaptation_score * 0.7
        return float(predicted_2026)

    def identify_dark_horses(self) -> List[Dict[str, Any]]:
        return [
            {
                "team": "Aston Martin",
                "reason": "Adrian Newey + Honda works partnership",
                "upside_potential": 0.85,
            },
            {
                "team": "Audi",
                "reason": "Works power unit manufacturer",
                "upside_potential": 0.6,
            },
            {
                "team": "Williams",
                "reason": "James Vowles leadership + reset",
                "upside_potential": 0.45,
            },
        ]

    def calculate_uncertainty(self, race_number: int) -> float:
        if race_number <= 3:
            return 0.40
        if race_number <= 7:
            return 0.25
        if race_number <= 15:
            return 0.15
        return 0.10


class F1Prediction2026:
    def __init__(self, base_model_path: str) -> None:
        self.base_model = load_pretrained_model(base_model_path)
        self.adaptation_layer = None
        self.regulation_impact = RegulationImpactModel()
        self.online_learner = SGDRegressor(random_state=42)

    def predict_before_testing(self, features: np.ndarray) -> PredictionResult:
        base_pred = self.base_model.predict(features)
        reg_impact = self.regulation_impact.predict(features)
        adjusted = base_pred * reg_impact
        unc = estimate_uncertainty(base_pred)
        interval = confidence_interval(float(np.mean(adjusted)), unc)
        return PredictionResult(
            prediction=adjusted,
            uncertainty=unc,
            confidence_interval=interval,
            uncertainty_level="high",
        )

    def update_after_testing(
        self,
        base_predictions: np.ndarray,
        testing_features: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        self.adaptation_layer = train_adaptation_layer(base_predictions, testing_features, targets)

    def predict_after_testing(
        self,
        features: np.ndarray,
        testing_features: np.ndarray,
    ) -> PredictionResult:
        base_pred = self.base_model.predict(features)
        if self.adaptation_layer is None:
            raise ValueError("adaptation_layer not trained")
        adapted = self.adaptation_layer.predict(np.column_stack([base_pred, testing_features]))
        unc = estimate_uncertainty(adapted)
        interval = confidence_interval(float(np.mean(adapted)), unc)
        return PredictionResult(
            prediction=adapted,
            uncertainty=unc,
            confidence_interval=interval,
            uncertainty_level="medium",
        )

    def predict_mid_season(
        self,
        features: np.ndarray,
        testing_features: np.ndarray,
        race_history_features: np.ndarray,
        race_history_targets: np.ndarray,
    ) -> PredictionResult:
        if self.adaptation_layer is None:
            raise ValueError("adaptation_layer not trained")
        base_pred = self.base_model.predict(features)
        adapted = self.adaptation_layer.predict(np.column_stack([base_pred, testing_features]))
        self.online_learner.partial_fit(race_history_features, race_history_targets)
        online_adjustment = self.online_learner.predict(features)
        final_pred = adapted * 0.8 + online_adjustment * 0.2
        unc = estimate_uncertainty(final_pred)
        interval = confidence_interval(float(np.mean(final_pred)), unc)
        return PredictionResult(
            prediction=final_pred,
            uncertainty=unc,
            confidence_interval=interval,
            uncertainty_level="low",
        )

    def predict_with_uncertainty(self, ensemble_predictions: np.ndarray) -> PredictionResult:
        avg_pred = np.mean(ensemble_predictions, axis=0)
        unc = estimate_uncertainty(ensemble_predictions)
        interval = confidence_interval(float(np.mean(avg_pred)), unc)
        return PredictionResult(
            prediction=avg_pred,
            uncertainty=unc,
            confidence_interval=interval,
            uncertainty_level="medium",
        )


class TestingDataProcessor:
    def __init__(self) -> None:
        self.testing_db: Dict[str, Dict[str, float]] = {}

    def ingest_testing_data(self, session_data: List[Dict[str, Any]]) -> None:
        for team in session_data:
            lap_times = team.get("lap_times", [])
            long_runs = team.get("long_runs", [])
            laps_completed = team.get("laps_completed", 0)
            issues_count = team.get("issues_count", 0)
            metrics = {
                "best_lap_time": min(lap_times) if lap_times else float("nan"),
                "race_pace": self.calculate_race_pace(long_runs),
                "reliability_score": self.calculate_reliability(laps_completed, issues_count),
                "active_aero_efficiency": float(team.get("aero_data", 0.0)),
                "overtake_mode_power": float(team.get("electrical_deployment", 0.0)),
            }
            self.testing_db[team["name"]] = metrics

    def calculate_race_pace(self, long_runs: List[float]) -> float:
        if not long_runs:
            return float("nan")
        return float(np.mean(long_runs))

    def calculate_reliability(self, laps_completed: int, issues_count: int) -> float:
        if laps_completed <= 0:
            return 0.0
        return max(0.0, 1.0 - (issues_count / laps_completed))

    def calculate_performance_order(self) -> List[Tuple[str, Dict[str, float]]]:
        return sorted(
            self.testing_db.items(),
            key=lambda x: x[1].get("race_pace", float("inf")),
        )

    def create_2026_features(self) -> Dict[str, Dict[str, float]]:
        features: Dict[str, Dict[str, float]] = {}
        for team, metrics in self.testing_db.items():
            features[team] = {
                "regulation_adaptation_score": self.calculate_adaptation(team, metrics),
                "active_aero_score": metrics["active_aero_efficiency"],
                "pu_reliability_2026": metrics["reliability_score"],
                "overtake_mode_score": metrics["overtake_mode_power"],
                "development_headroom": self.estimate_development_potential(team),
            }
        return features

    def calculate_adaptation(self, team: str, metrics: Dict[str, float]) -> float:
        pace_score = 0.0 if np.isnan(metrics["race_pace"]) else 1.0 / max(metrics["race_pace"], 1.0)
        return float(0.6 * pace_score + 0.4 * metrics.get("reliability_score", 0.0))

    def estimate_development_potential(self, team: str) -> float:
        return 0.5

    def load_historical_correlation(self) -> float:
        return 0.5

    def update_model_with_testing(
        self,
        base_model: Any,
        base_predictions: np.ndarray,
        targets: np.ndarray,
    ) -> Any:
        testing_features = self.create_2026_features()
        feature_matrix = np.array(list(testing_features.values()))
        if feature_matrix.size == 0:
            raise ValueError("No testing features available")
        adjusted_targets = targets * self.load_historical_correlation()
        adaptation_model = train_adaptation_layer(
            base_predictions=base_predictions,
            testing_features=feature_matrix,
            targets=adjusted_targets,
        )
        return adaptation_model
