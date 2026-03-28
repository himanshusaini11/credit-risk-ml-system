import importlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from fastapi.testclient import TestClient

from creditrisk.config import load_config
from creditrisk.model.train import run_training


def _write_config(config_path: Path, data_path: Path, model_dir: Path) -> None:
    payload = {
        "project": {"name": "test", "version": "0.0.0"},
        "data": {
            "path": str(data_path),
            "target": "default",
            "id_column": None,
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


def test_api_endpoints(tmp_path, monkeypatch):
    os.environ["CREDITRISK_DISABLE_SHAP"] = "1"
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "age": rng.integers(21, 70, size=200),
            "income": rng.normal(60000, 15000, size=200).round(2),
            "segment": rng.choice(["A", "B", "C"], size=200),
            "default": rng.integers(0, 2, size=200),
        }
    )
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)

    model_dir = tmp_path / "models"
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, data_path, model_dir)

    config = load_config(config_path)
    run_training(config)

    monkeypatch.setenv("CREDITRISK_CONFIG", str(config_path))
    import creditrisk.api.main as main

    importlib.reload(main)
    with TestClient(main.app) as client:
        response = client.get("/health")
        assert response.status_code == 200

        record = df.drop(columns=["default"]).iloc[0].to_dict()
        response = client.post("/predict", json={"record": record})
        assert response.status_code == 200
        payload = response.json()
        assert "prob_default" in payload
        assert "decision" in payload

        response = client.post("/explain", json={"record": record})
        assert response.status_code == 200
        payload = response.json()
        assert "explanation" in payload
