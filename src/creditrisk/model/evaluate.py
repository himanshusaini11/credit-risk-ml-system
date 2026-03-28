"""Evaluation and threshold selection utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from creditrisk.config import CostsConfig, ThresholdConfig


def predict_proba(model, X) -> np.ndarray:
    """Predict positive class probabilities."""
    if not hasattr(model, "predict_proba"):
        raise ValueError("Model does not support predict_proba")
    proba = model.predict_proba(X)
    return proba[:, 1]


def _confusion_at_threshold(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float
) -> Tuple[int, int, int, int]:
    preds = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def select_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold_cfg: ThresholdConfig,
    costs: CostsConfig,
) -> float:
    """Select a decision threshold based on validation data."""
    strategy = threshold_cfg.strategy
    if strategy in {"f1", "max_recall_at_precision"}:
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        if len(thresholds) == 0:
            return 0.5

        scores = precision[:-1], recall[:-1], thresholds
        prec, rec, thr = scores
        if strategy == "f1":
            f1_scores = 2 * prec * rec / (prec + rec + 1e-12)
            return float(thr[int(np.argmax(f1_scores))])

        mask = prec >= threshold_cfg.min_precision
        if not mask.any():
            return 0.5
        masked_recalls = rec[mask]
        masked_thresholds = thr[mask]
        return float(masked_thresholds[int(np.argmax(masked_recalls))])

    thresholds = np.unique(np.concatenate([y_proba, [0.0, 1.0]]))
    costs_values = []
    for threshold in thresholds:
        tn, fp, fn, tp = _confusion_at_threshold(y_true, y_proba, threshold)
        cost = fn * costs.fn_cost + fp * costs.fp_cost
        costs_values.append(cost)
    best_idx = int(np.argmin(costs_values))
    return float(thresholds[best_idx])


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """Compute ROC-AUC and PR-AUC."""
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
    }


def evaluate_split(
    name: str,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> Dict[str, float | str | int]:
    tn, fp, fn, tp = _confusion_at_threshold(y_true, y_proba, threshold)
    metrics = compute_metrics(y_true, y_proba)
    metrics.update(
        {
            "split": name,
            "threshold": float(threshold),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        }
    )
    return metrics


def save_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    output_path: str | Path,
) -> None:
    """Save ROC and PR curves to a PNG file."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fpr, tpr, color="#1f77b4")
    axes[0].set_title("ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")

    axes[1].plot(recall, precision, color="#ff7f0e")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
