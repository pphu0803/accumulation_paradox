"""Detrended Fluctuation Analysis (DFA)."""

from __future__ import annotations

import numpy as np


def dfa_exponent(
    series: np.ndarray,
    min_window: int = 4,
    max_window: int | None = None,
) -> float | None:
    """DFA exponent alpha. alpha~0.5 white noise, ~1 1/f, ~1.5 Brownian.

    Args:
        series: 1D time series.
        min_window: Minimum window size.
        max_window: Maximum window size (N//4 if None).

    Returns:
        DFA alpha, or None if estimation fails.
    """
    series = np.asarray(series, dtype=float)
    series = series[np.isfinite(series)]
    N = len(series)
    if N < 20:
        return None

    y = np.cumsum(series - np.mean(series))
    if max_window is None:
        max_window = N // 4

    windows, flucts = [], []
    for w in range(min_window, max_window + 1):
        n_segments = N // w
        if n_segments < 2:
            continue
        rms = 0.0
        for i in range(n_segments):
            segment = y[i * w:(i + 1) * w]
            t = np.arange(w, dtype=float)
            coeffs = np.polyfit(t, segment, 1)
            trend = np.polyval(coeffs, t)
            rms += np.mean((segment - trend) ** 2)
        rms = np.sqrt(rms / n_segments)
        windows.append(w)
        flucts.append(rms)

    if len(windows) < 3:
        return None

    coeffs = np.polyfit(np.log(windows), np.log(flucts), 1)
    return coeffs[0]


def lag1_autocorr(series: np.ndarray) -> float | None:
    """Lag-1 autocorrelation."""
    series = np.asarray(series, dtype=float)
    series = series[np.isfinite(series)]
    if len(series) < 3 or np.var(series) == 0:
        return None
    m = np.mean(series)
    return np.mean((series[:-1] - m) * (series[1:] - m)) / np.var(series)
