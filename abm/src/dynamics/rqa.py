"""Recurrence Quantification Analysis (RQA)."""

from __future__ import annotations

from collections import Counter

import numpy as np


def rqa_metrics(
    series: np.ndarray,
    emb_dim: int = 3,
    lag: int = 1,
    threshold_pct: float = 0.15,
    max_points: int = 300,
) -> dict[str, float] | None:
    """Compute RQA metrics: RR, DET, L_mean, L_max, ENTR, LAM.

    Args:
        series: 1D time series.
        emb_dim: Embedding dimension.
        lag: Time delay.
        threshold_pct: Threshold percentile for recurrence (0-1).
        max_points: Subsample to this many points if series is long.

    Returns:
        Dict with RQA metrics, or None if estimation fails.
    """
    series = np.asarray(series, dtype=float)
    series = series[np.isfinite(series)]
    if len(series) < 30:
        return None

    n_pts = len(series) - (emb_dim - 1) * lag
    if n_pts < 20:
        return None

    embedded = np.column_stack([
        series[i * lag:i * lag + n_pts] for i in range(emb_dim)
    ])

    N = len(embedded)
    if N > max_points:
        idx = np.linspace(0, N - 1, max_points, dtype=int)
        embedded = embedded[idx]
        N = max_points

    dist_mat = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(embedded[i] - embedded[j])
            dist_mat[i, j] = d
            dist_mat[j, i] = d

    threshold = np.percentile(dist_mat[dist_mat > 0], threshold_pct * 100)
    rec_mat = (dist_mat <= threshold).astype(int)
    np.fill_diagonal(rec_mat, 0)
    rr = np.sum(rec_mat) / (N * (N - 1))

    # Diagonal lines (determinism)
    diag_lengths = []
    for d in range(1, N):
        diag = np.array([rec_mat[i, i + d] for i in range(N - d)])
        in_run, length = False, 0
        for v in diag:
            if v == 1:
                length += 1
                in_run = True
            else:
                if in_run and length >= 2:
                    diag_lengths.append(length)
                length, in_run = 0, False
        if in_run and length >= 2:
            diag_lengths.append(length)

    if not diag_lengths:
        return {"RR": rr, "DET": 0.0, "L_mean": 0.0, "L_max": 0, "ENTR": 0.0, "LAM": 0.0}

    # Audit 2026-10-07 (external review D3): the diagonal-run numerator only
    # scans offsets d >= 1 (upper triangle), so the denominator must be the
    # upper-triangle recurrent-point count too — the full-matrix sum exactly
    # halved DET. LAM scans full columns (same basis as its full-matrix
    # denominator) and is left unchanged.
    upper_points = int(np.sum(np.triu(rec_mat, 1)))
    det = sum(diag_lengths) / max(upper_points, 1)
    l_mean = np.mean(diag_lengths)
    l_max = max(diag_lengths)

    counts = Counter(diag_lengths)
    total = sum(counts.values())
    probs = np.array([c / total for c in counts.values()])
    probs = probs[probs > 0]
    entr = -np.sum(probs * np.log(probs))

    # Vertical lines (laminarity)
    vert_lengths = []
    for j in range(N):
        col = rec_mat[:, j]
        in_run, length = False, 0
        for v in col:
            if v == 1:
                length += 1
                in_run = True
            else:
                if in_run and length >= 2:
                    vert_lengths.append(length)
                length, in_run = 0, False
        if in_run and length >= 2:
            vert_lengths.append(length)

    lam = sum(vert_lengths) / max(np.sum(rec_mat), 1) if vert_lengths else 0.0

    return {"RR": rr, "DET": det, "L_mean": l_mean, "L_max": l_max, "ENTR": entr, "LAM": lam}
