"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    version: str


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    target: str
    id_column: Optional[str] = None
    timestamp_column: Optional[str] = None


class SplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["stratified", "time_based"]
    seed: int = 42
    train_frac: float = Field(gt=0, lt=1)
    val_frac: float = Field(gt=0, lt=1)
    test_frac: float = Field(gt=0, lt=1)


class PreprocessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    numeric_impute: Literal["mean", "median"] = "median"
    categorical_impute: Literal["most_frequent", "constant"] = "most_frequent"
    encoding: Literal["onehot"] = "onehot"
    scaling: Literal["standard", "none"] = "standard"


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["logistic_regression", "random_forest", "lightgbm"]
    params: Dict[str, Any] = Field(default_factory=dict)


class ImbalanceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["class_weight", "smote", "none"] = "none"


class ThresholdConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: Literal[
        "f1",
        "max_recall_at_precision",
        "min_expected_cost",
    ]
    min_precision: float = Field(default=0.8, ge=0.0, le=1.0)


class CostsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fn_cost: float = Field(gt=0)
    fp_cost: float = Field(gt=0)


class CalibrationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    method: Literal["platt", "isotonic"] = "platt"


class RegistryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_dir: str


class ApiConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_version_to_load: str = "latest"


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project: ProjectConfig
    data: DataConfig
    split: SplitConfig
    preprocess: PreprocessConfig
    model: ModelConfig
    imbalance: ImbalanceConfig
    threshold: ThresholdConfig
    costs: CostsConfig
    calibration: CalibrationConfig
    registry: RegistryConfig
    api: ApiConfig


def load_config(path: str | Path) -> Config:
    """Load configuration from YAML."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return Config.model_validate(payload)


def save_config(config: Config, path: str | Path) -> None:
    """Persist configuration to YAML."""
    path = Path(path)
    payload = config.model_dump()
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
