"""Correlation dimension via Grassberger-Procaccia algorithm."""

from __future__ import annotations

import numpy as np


def correlation_dimension(
    series: np.ndarray,
    emb_dim: int = 3,
    lag: int = 1,
    n_radii: int = 20,
    max_pairs: int = 50000,
) -> float | None:
    """Estimate correlation dimension D2 via Grassberger-Procaccia.

    Args:
        series: 1D time series.
        emb_dim: Embedding dimension.
        lag: Time delay for embedding.
        n_radii: Number of radii for correlation sum.
        max_pairs: Max pairwise distances to compute.

    Returns:
        D2 estimate, or None if estimation fails.
    """
    series = np.asarray(series, dtype=float)
    series = series[np.isfinite(series)]
    if len(series) < 50:
        return None

    n_pts = len(series) - (emb_dim - 1) * lag
    if n_pts < 20:
        return None

    embedded = np.column_stack([
        series[i * lag:i * lag + n_pts] for i in range(emb_dim)
    ])

    if len(embedded) > 500:
        idx = np.linspace(0, len(embedded) - 1, 500, dtype=int)
        embedded = embedded[idx]

    N = len(embedded)
    dists = []
    for i in range(min(N, 500)):
        for j in range(i + 1, min(N, 500)):
            dists.append(np.linalg.norm(embedded[i] - embedded[j]))
            if len(dists) >= max_pairs:
                break
        if len(dists) >= max_pairs:
            break

    dists = np.array(dists)
    if len(dists) == 0:
        return None

    r_max = np.percentile(dists, 90)
    r_min = np.percentile(dists, 5)
    if r_max <= r_min:
        return None

    radii = np.linspace(r_min, r_max, n_radii)
    corr_sum = np.array([np.mean(dists <= r) for r in radii])

    valid = (corr_sum > 0) & (radii > 0)
    if np.sum(valid) < 5:
        return None

    log_r = np.log(radii[valid])
    log_c = np.log(corr_sum[valid])
    n_fit = len(log_r)
    start, end = n_fit // 5, 4 * n_fit // 5
    if end - start < 3:
        return None

    coeffs = np.polyfit(log_r[start:end], log_c[start:end], 1)
    return coeffs[0]
