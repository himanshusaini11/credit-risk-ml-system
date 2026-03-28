import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from creditrisk.config import load_config
from creditrisk.cli import predict_batch
from creditrisk.model.train import run_training


def _write_config(config_path: Path, data_path: Path, model_dir: Path) -> None:
    payload = {
        "project": {"name": "test", "version": "0.0.0"},
        "data": {
            "path": str(data_path),
            "target": "default",
            "id_column": "ID",
            "timestamp_column": None,
        },
        "split": {
            "method": "stratified",
            "seed": 7,
            "train_frac": 0.7,
            "val_frac": 0.15,
            "test_frac": 0.15,
        },
        "preprocess": {
            "numeric_impute": "median",
            "categorical_impute": "most_frequent",
            "encoding": "onehot",
            "scaling": "standard",
        },
        "model": {"type": "logistic_regression", "params": {"max_iter": 100}},
        "imbalance": {"strategy": "none"},
        "threshold": {"strategy": "f1", "min_precision": 0.8},
        "costs": {"fn_cost": 5.0, "fp_cost": 1.0},
        "calibration": {"enabled": False, "method": "platt"},
        "registry": {"model_dir": str(model_dir)},
        "api": {"model_version_to_load": "latest"},
    }
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle)


def test_predict_batch(tmp_path):
    os.environ["CREDITRISK_DISABLE_SHAP"] = "1"
    rng = np.random.default_rng(123)
    df = pd.DataFrame(
        {
            "ID": np.arange(200),
            "age": rng.integers(21, 70, size=200),
            "income": rng.normal(60000, 15000, size=200).round(2),
            "segment": rng.choice(["A", "B", "C"], size=200),
            "default": rng.integers(0, 2, size=200),
        }
    )
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"

    df.to_csv(train_path, index=False)
    test_df = df.drop(columns=["default"]).copy()
    test_df["default"] = np.nan
    test_df.to_csv(test_path, index=False)

    model_dir = tmp_path / "models"
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, train_path, model_dir)

    config = load_config(config_path)
    run_training(config)

    output_path = tmp_path / "predictions.csv"
    predict_batch(str(config_path), str(test_path), str(output_path))

    result = pd.read_csv(output_path)
    assert len(result) == len(test_df)
    assert "ID" in result.columns
    assert "prob_default" in result.columns
    assert "decision" in result.columns
    assert "threshold" in result.columns
    assert "model_version" in result.columns
