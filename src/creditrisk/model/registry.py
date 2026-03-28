"""Model artifact versioning and registry utilities."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import joblib

from creditrisk.config import Config, save_config


def _git_short_hash() -> str:
    try:
        result = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return result.decode("utf-8").strip()
    except Exception:
        return "nogit"


def make_version() -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    git_hash = _git_short_hash()
    return f"{timestamp}-{git_hash}"


def save_artifacts(
    model,
    preprocessor,
    metrics: Dict[str, Any],
    schema: Dict[str, Any],
    config: Config,
    model_dir: str | Path,
    version: str,
    feature_names: Optional[list[str]] = None,
    global_explain: Optional[Dict[str, Any]] = None,
    background: Optional[Any] = None,
) -> Path:
    """Persist model artifacts to a versioned directory."""
    model_root = Path(model_dir) / version
    model_root.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_root / "model.joblib")
    joblib.dump(preprocessor, model_root / "preprocess.joblib")
    if background is not None:
        joblib.dump(background, model_root / "background.joblib")

    if feature_names is not None:
        schema = {**schema, "feature_names": feature_names}

    if global_explain is not None:
        schema = {**schema, "global_explain": global_explain}

    with (model_root / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    with (model_root / "schema.json").open("w", encoding="utf-8") as handle:
        json.dump(schema, handle, indent=2)

    save_config(config, model_root / "config.yaml")
    return model_root
