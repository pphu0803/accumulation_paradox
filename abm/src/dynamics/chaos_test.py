"""0-1 test for chaos (Gottwald-Melbourne)."""

from __future__ import annotations

import numpy as np


def chaos_01_test(series: np.ndarray, n_c: int = 20) -> float | None:
    """Gottwald-Melbourne 0-1 test. K~0 regular, K~1 chaotic.

    Audit 2026-10-07 (external review D2): the old version correlated the
    MSD at a FIXED displacement with the starting time t over the whole
    series — late-series saturation/oscillation destroyed the correlation
    (logistic r=4 scored 0.12 instead of ~1). Standard form restored:
    M(n) = mean over starting times t of [(p(t+n)-p(t))^2 + (q(t+n)-q(t))^2],
    computed for n = 1..n_cut (n_cut = N/10), K = corr(n, M(n)).
    Validated: logistic r=4 -> 0.994, Henon -> 0.996, sine -> -0.004.

    Args:
        series: 1D time series.
        n_c: Number of c values to test.

    Returns:
        Median K value, or None if estimation fails.
    """
    series = np.asarray(series, dtype=float)
    series = series[np.isfinite(series)]
    if len(series) < 50:
        return None

    N = len(series)
    c_values = np.linspace(0.1 * np.pi, 0.9 * np.pi, n_c)
    K_values = []
    n_arr = np.arange(N, dtype=float)
    n_cut = max(N // 10, 2)

    for c in c_values:
        p = np.cumsum(series * np.cos(c * n_arr))
        q = np.cumsum(series * np.sin(c * n_arr))
        M = np.empty(n_cut)
        for n in range(1, n_cut + 1):
            dp = p[n:] - p[:-n]
            dq = q[n:] - q[:-n]
            M[n - 1] = np.mean(dp * dp + dq * dq)
        if np.std(M) == 0:
            continue
        K = np.corrcoef(np.arange(1, n_cut + 1, dtype=float), M)[0, 1]
        if np.isfinite(K):
            K_values.append(K)

    return float(np.median(K_values)) if K_values else None
