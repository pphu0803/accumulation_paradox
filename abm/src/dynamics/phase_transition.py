"""Phase transition detection via sliding window."""

from __future__ import annotations

import numpy as np


def detect_phase_transitions(
    series: np.ndarray,
    window: int = 20,
    threshold_sigma: float = 2.0,
) -> list[int]:
    """Detect abrupt shifts in mean using sliding window.

    Args:
        series: 1D time series.
        window: Sliding window size.
        threshold_sigma: Number of std deviations for threshold.

    Returns:
        List of transition indices.
    """
    series = np.asarray(series, dtype=float)
    series = series[np.isfinite(series)]
    if len(series) < window * 3:
        return []

    means = np.array([np.mean(series[i:i + window]) for i in range(len(series) - window)])
    diff = np.abs(np.diff(means))
    threshold = np.mean(diff) + threshold_sigma * np.std(diff)
    return np.where(diff > threshold)[0].tolist()


def pca_intrinsic_dimension(
    data: np.ndarray,
    variance_threshold: float = 0.9,
) -> int:
    """Estimate intrinsic dimension via PCA variance threshold.

    Args:
        data: 2D array (n_samples, n_features).
        variance_threshold: Cumulative variance fraction (e.g. 0.9 for d_90).

    Returns:
        Intrinsic dimension at given threshold.
    """
    from numpy.linalg import svd

    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data).all(axis=1)]
    if len(data) < 10:
        return 0

    # Center
    data = data - data.mean(axis=0)

    # SVD
    _, s, _ = svd(data, full_matrices=False)
    explained = s ** 2
    total = explained.sum()
    if total == 0:
        return 0

    cumvar = np.cumsum(explained) / total
    return int(np.searchsorted(cumvar, variance_threshold) + 1)
