"""Load model artifacts for inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib

from creditrisk.config import load_config


def _latest_version(model_dir: Path) -> str:
    versions = [p.name for p in model_dir.iterdir() if p.is_dir()]
    if not versions:
        raise FileNotFoundError(f"No model versions found in {model_dir}")
    versions.sort()
    return versions[-1]


def load_model_bundle(config_path: str | Path) -> Dict[str, Any]:
    config = load_config(config_path)
    model_dir = Path(config.registry.model_dir)
    version = config.api.model_version_to_load
    if version == "latest":
        version = _latest_version(model_dir)

    model_root = model_dir / version
    if not model_root.exists():
        raise FileNotFoundError(f"Model version not found: {model_root}")

    model = joblib.load(model_root / "model.joblib")
    preprocessor = joblib.load(model_root / "preprocess.joblib")
    background_path = model_root / "background.joblib"
    background = joblib.load(background_path) if background_path.exists() else None

    with (model_root / "metrics.json").open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    with (model_root / "schema.json").open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    with (model_root / "config.yaml").open("r", encoding="utf-8") as handle:
        saved_config = handle.read()

    return {
        "model": model,
        "preprocessor": preprocessor,
        "metrics": metrics,
        "schema": schema,
        "config": config,
        "config_raw": saved_config,
        "model_version": version,
        "background": background,
    }
