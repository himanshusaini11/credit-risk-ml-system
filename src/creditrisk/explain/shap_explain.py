"""Explainability utilities with SHAP fallback."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.inspection import permutation_importance


def _extract_estimator(model):
    if hasattr(model, "named_steps"):
        return model.named_steps.get("model", model)
    if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
        inner = model.calibrated_classifiers_[0].estimator
        if hasattr(inner, "named_steps"):
            return inner.named_steps.get("model", inner)
        return inner
    if hasattr(model, "estimator"):
        inner = model.estimator
        if hasattr(inner, "named_steps"):
            return inner.named_steps.get("model", inner)
        return inner
    return model


def _top_k_importances(feature_names: List[str], values: np.ndarray, top_k: int):
    pairs = list(zip(feature_names, values))
    pairs.sort(key=lambda item: abs(item[1]), reverse=True)
    return [
        {"feature": name, "importance": float(val)} for name, val in pairs[:top_k]
    ]


def compute_global_importance(
    model,
    preprocessor,
    X,
    y,
    feature_names: List[str],
    max_features: int = 20,
) -> Dict[str, Any]:
    """Compute global feature importance summary."""
    try:
        if os.getenv("CREDITRISK_DISABLE_SHAP") == "1":
            raise ImportError("SHAP disabled via env")
        import shap

        estimator = _extract_estimator(model)
        X_trans = preprocessor.transform(X)
        background = shap.sample(X_trans, min(200, X_trans.shape[0]))

        if estimator.__class__.__name__ == "LogisticRegression":
            explainer = shap.LinearExplainer(estimator, background, feature_names=feature_names)
        else:
            explainer = shap.TreeExplainer(estimator, feature_names=feature_names)

        shap_values = explainer(X_trans)
        values = shap_values.values
        if values.ndim == 3:
            values = values[:, :, 1]
        importances = np.mean(np.abs(values), axis=0)
        return {
            "method": "shap",
            "importances": _top_k_importances(feature_names, importances, max_features),
        }
    except Exception:
        result = permutation_importance(
            model,
            X,
            y,
            n_repeats=5,
            random_state=42,
            scoring="roc_auc",
        )
        importances = result.importances_mean
        return {
            "method": "permutation_importance",
            "note": "SHAP unavailable or failed; using permutation importance.",
            "importances": _top_k_importances(feature_names, importances, max_features),
        }


def explain_record(
    model,
    preprocessor,
    record,
    feature_names: List[str],
    background: Optional[Any] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Explain a single record with SHAP if available; fallback to coefficients."""
    X_trans = preprocessor.transform(record)
    estimator = _extract_estimator(model)
    try:
        if os.getenv("CREDITRISK_DISABLE_SHAP") == "1":
            raise ImportError("SHAP disabled via env")
        import shap

        if background is None:
            background = shap.sample(X_trans, min(50, X_trans.shape[0]))
        if estimator.__class__.__name__ == "LogisticRegression":
            explainer = shap.LinearExplainer(estimator, background, feature_names=feature_names)
        else:
            explainer = shap.TreeExplainer(estimator, feature_names=feature_names)
        shap_values = explainer(X_trans)
        values = shap_values.values
        if values.ndim == 3:
            values = values[:, :, 1]
        contributions = values[0]
        return {
            "method": "shap",
            "contributions": _top_k_importances(feature_names, contributions, top_k),
        }
    except Exception:
        if hasattr(estimator, "coef_"):
            contributions = estimator.coef_.reshape(-1) * X_trans[0]
            method = "linear_coefficients"
        elif hasattr(estimator, "feature_importances_"):
            contributions = estimator.feature_importances_ * X_trans[0]
            method = "feature_importance_weighted"
        else:
            contributions = np.zeros(len(feature_names))
            method = "unavailable"
        return {
            "method": method,
            "note": "SHAP unavailable or failed; using approximate contributions.",
            "contributions": _top_k_importances(feature_names, contributions, top_k),
        }
