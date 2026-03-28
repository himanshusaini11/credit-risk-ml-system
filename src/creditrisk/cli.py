"""Command-line interface for training and evaluation."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Tuple

import pandas as pd

from creditrisk.api.model_loader import load_model_bundle
from creditrisk.config import load_config
from creditrisk.data.load import load_csv
from creditrisk.data.schema import validate_dataframe
from creditrisk.data.split import split_data
from creditrisk.model.evaluate import evaluate_split, predict_proba
from creditrisk.model.train import run_training


def _prepare_features(
    df: pd.DataFrame, schema: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.Series]:
    feature_cols = schema["feature_columns"]
    X = df[feature_cols].copy()
    y = df[schema["target"]].astype(int)
    return X, y


def data_summary(config_path: str) -> None:
    config = load_config(config_path)
    df = load_csv(config.data.path)
    schema = validate_dataframe(
        df,
        target=config.data.target,
        id_column=config.data.id_column,
        timestamp_column=config.data.timestamp_column,
    )
    summary = {
        "rows": schema["n_rows"],
        "features": schema["n_features"],
        "base_rate": schema["base_rate"],
        "numeric_features": len(schema["features"]["numeric"]),
        "categorical_features": len(schema["features"]["categorical"]),
    }
    print(json.dumps(summary, indent=2))


def train(config_path: str) -> None:
    config = load_config(config_path)
    result = run_training(config)
    print(json.dumps(result, indent=2))


def evaluate(config_path: str) -> None:
    config = load_config(config_path)
    df = load_csv(config.data.path)
    schema = validate_dataframe(
        df,
        target=config.data.target,
        id_column=config.data.id_column,
        timestamp_column=config.data.timestamp_column,
    )
    _, _, test_df = split_data(
        df,
        split=config.split,
        target=config.data.target,
        timestamp_column=config.data.timestamp_column,
    )
    bundle = load_model_bundle(config_path)
    X_test, y_test = _prepare_features(test_df, schema)
    proba = predict_proba(bundle["model"], X_test)
    threshold = float(bundle["metrics"]["threshold"])
    metrics = evaluate_split("test", y_test.to_numpy(), proba, threshold)
    print(json.dumps(metrics, indent=2))


def predict_batch(config_path: str, input_path: str, output_path: str) -> None:
    config = load_config(config_path)
    df = load_csv(input_path)
    bundle = load_model_bundle(config_path)
    schema = bundle["schema"]
    feature_cols = schema["feature_columns"]

    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required features in input: {missing}")

    if config.data.target in df.columns:
        df = df.drop(columns=[config.data.target])

    X = df[feature_cols].copy()
    proba = bundle["model"].predict_proba(X)[:, 1]
    threshold = float(bundle["metrics"]["threshold"])
    decisions = ["REJECT" if p >= threshold else "APPROVE" for p in proba]

    output = {
        "prob_default": proba,
        "decision": decisions,
        "threshold": threshold,
        "model_version": bundle["model_version"],
    }

    id_column = config.data.id_column
    if id_column and id_column in df.columns:
        output = {id_column: df[id_column].values, **output}

    output_df = pd.DataFrame(output)
    output_df.to_csv(output_path, index=False)
    print(json.dumps({"output_path": output_path, "rows": len(output_df)}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Credit Risk ML CLI")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML configuration.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("data-summary", help="Show dataset summary")
    subparsers.add_parser("train", help="Train and evaluate model")
    subparsers.add_parser("evaluate", help="Evaluate existing model")
    predict_parser = subparsers.add_parser(
        "predict-batch", help="Run batch predictions on a CSV file"
    )
    predict_parser.add_argument("--input", required=True, help="Path to input CSV.")
    predict_parser.add_argument(
        "--output", required=True, help="Path to output predictions CSV."
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "data-summary":
        data_summary(args.config)
    elif args.command == "train":
        train(args.config)
    elif args.command == "evaluate":
        evaluate(args.config)
    elif args.command == "predict-batch":
        predict_batch(args.config, args.input, args.output)


if __name__ == "__main__":
    main()
