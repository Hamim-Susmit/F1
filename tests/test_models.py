import numpy as np

from f1_prediction_2026 import estimate_uncertainty


def test_estimate_uncertainty() -> None:
    values = np.array([1.0, 2.0, 3.0])
    assert estimate_uncertainty(values) > 0
