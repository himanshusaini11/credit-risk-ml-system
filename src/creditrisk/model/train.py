"""Model training orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from creditrisk.config import Config
from creditrisk.data.load import load_csv
from creditrisk.data.preprocess import build_preprocessor, get_feature_names
from creditrisk.data.schema import validate_dataframe
from creditrisk.data.split import split_data
from creditrisk.explain.shap_explain import compute_global_importance
from creditrisk.model.calibrate import calibrate_model
from creditrisk.model.evaluate import (
    evaluate_split,
    predict_proba,
    save_curves,
    select_threshold,
)
from creditrisk.model.registry import make_version, save_artifacts


def _prepare_features(
    df: pd.DataFrame, schema: Dict[str, Any]
) -> Tuple[pd.DataFrame, np.ndarray]:
    feature_cols = schema["feature_columns"]
    X = df[feature_cols].copy()
    y = df[schema["target"]].astype(int).to_numpy()
    return X, y


def _build_estimator(config: Config, scale_pos_weight: float = 1.0):
    seed = config.split.seed
    params = dict(config.model.params)

    if config.model.type == "logistic_regression":
        params.setdefault("max_iter", 300)
        params.setdefault("solver", "liblinear")
        params.setdefault("random_state", seed)
        if config.imbalance.strategy == "class_weight":
            params.setdefault("class_weight", "balanced")
        return LogisticRegression(**params)

    if config.model.type == "lightgbm":
        params.setdefault("n_estimators", 1000)
        params.setdefault("learning_rate", 0.05)
        params.setdefault("num_leaves", 31)
        params.setdefault("min_child_samples", 20)
        params.setdefault("random_state", seed)
        params.setdefault("n_jobs", -1)
        params.setdefault("verbose", -1)
        # scale_pos_weight is always derived from y_train — never from config
        return lgb.LGBMClassifier(scale_pos_weight=scale_pos_weight, **params)

    # Default: random_forest
    params.setdefault("n_estimators", 300)
    params.setdefault("random_state", seed)
    if config.imbalance.strategy == "class_weight":
        params.setdefault("class_weight", "balanced")
    return RandomForestClassifier(**params)


def _build_pipeline(preprocessor, estimator, config: Config):
    if config.imbalance.strategy == "smote":
        return ImbPipeline(
            steps=[
                ("preprocess", preprocessor),
                ("smote", SMOTE(random_state=config.split.seed)),
                ("model", estimator),
            ]
        )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])


def run_training(config: Config) -> Dict[str, Any]:
    """Run end-to-end training, evaluation, and artifact persistence."""
    df = load_csv(config.data.path)
    schema = validate_dataframe(
        df,
        target=config.data.target,
        id_column=config.data.id_column,
        timestamp_column=config.data.timestamp_column,
    )
    train_df, val_df, test_df = split_data(
        df,
        split=config.split,
        target=config.data.target,
        timestamp_column=config.data.timestamp_column,
    )

    numeric_features = schema["features"]["numeric"]
    categorical_features = schema["features"]["categorical"]
    preprocessor = build_preprocessor(numeric_features, categorical_features, config.preprocess)

    X_train, y_train = _prepare_features(train_df, schema)
    X_val, y_val = _prepare_features(val_df, schema)
    X_test, y_test = _prepare_features(test_df, schema)

    # Compute class imbalance ratio from training labels for scale_pos_weight
    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    estimator = _build_estimator(config, scale_pos_weight=scale_pos_weight)
    pipeline = _build_pipeline(preprocessor, estimator, config)

    pipeline.fit(X_train, y_train)

    fitted_preprocessor = pipeline.named_steps["preprocess"]
    feature_names = get_feature_names(fitted_preprocessor)

    model_for_eval = pipeline
    if config.calibration.enabled:
        model_for_eval = calibrate_model(
            pipeline, X_val, y_val, config.calibration.method
        )

    val_proba = predict_proba(model_for_eval, X_val)
    threshold = select_threshold(y_val, val_proba, config.threshold, config.costs)

    val_metrics = evaluate_split("val", y_val, val_proba, threshold)
    test_proba = predict_proba(model_for_eval, X_test)
    test_metrics = evaluate_split("test", y_test, test_proba, threshold)

    metrics = {
        "model_type": config.model.type,
        "imbalance": config.imbalance.strategy,
        "calibration": config.calibration.enabled,
        "threshold": threshold,
        "base_rate": schema["base_rate"],
        "val": val_metrics,
        "test": test_metrics,
    }

    background_raw = X_train.sample(
        n=min(200, len(X_train)), random_state=config.split.seed
    )
    background = fitted_preprocessor.transform(background_raw)
    global_explain = compute_global_importance(
        model_for_eval,
        fitted_preprocessor,
        X_val,
        y_val,
        feature_names,
    )

    version = make_version()
    model_root = save_artifacts(
        model_for_eval,
        fitted_preprocessor,
        metrics,
        schema,
        config,
        config.registry.model_dir,
        version,
        feature_names=feature_names,
        global_explain=global_explain,
        background=background,
    )

    save_curves(y_val, val_proba, Path(model_root) / "val_curves.png")
    save_curves(y_test, test_proba, Path(model_root) / "test_curves.png")

    return {
        "version": version,
        "artifact_dir": str(model_root),
        "threshold": threshold,
        "metrics": metrics,
    }
