"""Probability calibration utilities."""

from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV


def calibrate_model(model, X_val, y_val, method: str):
    """Calibrate probabilities using validation data."""
    method_map = {"platt": "sigmoid", "isotonic": "isotonic"}
    sklearn_method = method_map.get(method, "sigmoid")
    calibrator = CalibratedClassifierCV(model, method=sklearn_method, cv="prefit")
    calibrator.fit(X_val, y_val)
    return calibrator
