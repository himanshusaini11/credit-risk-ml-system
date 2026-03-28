#!/usr/bin/env python3
"""Prepare processed_test.csv using train-fitted encodings."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _build_label_maps(df_train: pd.DataFrame, label_cols: list[str]) -> dict[str, dict[str, int]]:
    return {
        col: {
            val: idx
            for idx, val in enumerate(
                df_train[col].astype(str).fillna("MISSING").unique()
            )
        }
        for col in label_cols
    }


def _apply_transform(
    df: pd.DataFrame,
    grade_order: dict[str, int],
    label_maps: dict[str, dict[str, int]],
    label_cols: list[str],
    freq_map: dict[str, int],
    one_hot_cols: list[str],
) -> pd.DataFrame:
    df = df.copy()

    df["Grade"] = df["Grade"].map(grade_order).fillna(-1)
    for col in label_cols:
        df[col] = (
            df[col]
            .astype(str)
            .fillna("MISSING")
            .map(label_maps[col])
            .fillna(-1)
            .astype(int)
        )

    df["Loan Title"] = df["Loan Title"].map(freq_map).fillna(0)
    df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare processed_test.csv using train-fitted encodings."
    )
    parser.add_argument("--train", default="data/train.csv", help="Path to train.csv")
    parser.add_argument("--test", default="data/test.csv", help="Path to test.csv")
    parser.add_argument(
        "--output", default="data/processed_test.csv", help="Output CSV path"
    )
    parser.add_argument(
        "--keep-id",
        action="store_true",
        help="Keep ID column in the output if present.",
    )
    args = parser.parse_args()

    train_path = Path(args.train)
    test_path = Path(args.test)
    output_path = Path(args.output)

    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    required_cols = [
        "Grade",
        "Sub Grade",
        "Batch Enrolled",
        "Loan Title",
        "Initial List Status",
        "Employment Duration",
        "Verification Status",
    ]
    missing_train = [col for col in required_cols if col not in df_train.columns]
    missing_test = [col for col in required_cols if col not in df_test.columns]
    if missing_train or missing_test:
        raise ValueError(
            f"Missing columns. train={missing_train}, test={missing_test}"
        )

    id_series = None
    if args.keep_id and "ID" in df_test.columns:
        id_series = df_test["ID"].copy()

    for df in (df_train, df_test):
        if "ID" in df.columns:
            df.drop(columns=["ID"], inplace=True)
        if "Loan Status" in df.columns:
            df.drop(columns=["Loan Status"], inplace=True)

    grade_order = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
    label_cols = ["Sub Grade", "Batch Enrolled"]
    one_hot_cols = [
        "Initial List Status",
        "Employment Duration",
        "Verification Status",
    ]

    label_maps = _build_label_maps(df_train, label_cols)
    freq_map = df_train["Loan Title"].value_counts().to_dict()

    train_out = _apply_transform(
        df_train, grade_order, label_maps, label_cols, freq_map, one_hot_cols
    )
    test_out = _apply_transform(
        df_test, grade_order, label_maps, label_cols, freq_map, one_hot_cols
    )

    feature_cols = list(train_out.columns)
    test_out = test_out.reindex(columns=feature_cols, fill_value=0)

    if id_series is not None:
        test_out.insert(0, "ID", id_series.values)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    test_out.to_csv(output_path, index=False)
    print(f"Saved processed test data to {output_path}")


if __name__ == "__main__":
    main()
