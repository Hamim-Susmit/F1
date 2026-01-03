import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

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

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.ones(features.shape[0])
        return self.model.predict(features)


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
