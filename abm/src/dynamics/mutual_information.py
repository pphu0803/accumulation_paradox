"""Mutual information estimation."""

from __future__ import annotations

import numpy as np


def mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10,
) -> float | None:
    """Estimate mutual information between two signals.

    Args:
        x, y: 1D time series.
        n_bins: Number of bins for histogram-based estimation.

    Returns:
        MI in nats, or None if estimation fails.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 20:
        return None

    hist_xy, _, _ = np.histogram2d(x, y, bins=n_bins)
    hist_xy = hist_xy / hist_xy.sum()
    hist_x = hist_xy.sum(axis=1)
    hist_y = hist_xy.sum(axis=0)

    mi = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if hist_xy[i, j] > 0 and hist_x[i] > 0 and hist_y[j] > 0:
                mi += hist_xy[i, j] * np.log(hist_xy[i, j] / (hist_x[i] * hist_y[j]))
    return float(mi)


def cross_correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    """Pearson correlation between two signals."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 5:
        return None
    return float(np.corrcoef(x, y)[0, 1])
