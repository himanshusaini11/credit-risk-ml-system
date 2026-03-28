"""Simple drift metrics for monitoring."""

from __future__ import annotations

import numpy as np


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Compute PSI between expected and actual distributions."""
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    quantiles = np.linspace(0, 1, bins + 1)
    breakpoints = np.quantile(expected, quantiles)
    expected_hist, _ = np.histogram(expected, bins=breakpoints)
    actual_hist, _ = np.histogram(actual, bins=breakpoints)
    expected_pct = expected_hist / max(expected_hist.sum(), 1)
    actual_pct = actual_hist / max(actual_hist.sum(), 1)
    epsilon = 1e-6
    psi = np.sum((actual_pct - expected_pct) * np.log((actual_pct + epsilon) / (expected_pct + epsilon)))
    return float(psi)
