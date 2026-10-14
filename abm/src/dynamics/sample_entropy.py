"""Sample entropy."""

from __future__ import annotations

import numpy as np


def sample_entropy(
    series: np.ndarray,
    m: int = 2,
    r: float | None = None,
) -> float | None:
    """Compute sample entropy. Lower = more regular/predictable.

    Args:
        series: 1D time series.
        m: Embedding dimension for match counting.
        r: Tolerance (fraction of std if None).

    Returns:
        SampEn value, or None if estimation fails.
    """
    series = np.asarray(series, dtype=float)
    series = series[np.isfinite(series)]
    if len(series) < 20:
        return None
    if r is None:
        r = 0.2 * np.std(series)
    if r == 0:
        return None

    def _count_matches(ts, m_val, tol):
        n = len(ts) - m_val
        if n < 1:
            return 0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if np.max(np.abs(ts[i:i + m_val] - ts[j:j + m_val])) <= tol:
                    count += 1
        return count

    A = _count_matches(series, m + 1, r)
    B = _count_matches(series, m, r)
    if B == 0:
        return None
    return -np.log(A / B)


def multiscale_entropy(
    series: np.ndarray,
    max_scale: int = 10,
    m: int = 2,
    r: float | None = None,
) -> list[float | None]:
    """Multiscale Sample Entropy (MSE)."""
    series = np.asarray(series, dtype=float)
    series = series[np.isfinite(series)]
    if r is None:
        r = 0.2 * np.std(series)

    entropies = []
    for scale in range(1, max_scale + 1):
        n = len(series) // scale
        if n < 20:
            entropies.append(None)
            continue
        cg = np.array([np.mean(series[i * scale:(i + 1) * scale]) for i in range(n)])
        entropies.append(sample_entropy(cg, m=m, r=r))
    return entropies
