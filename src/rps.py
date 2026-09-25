"""Ranked Probability Score over integer run thresholds 0..300.

RPS(F, y) = sum_{t=0}^{300} (F(t) - 1{y <= t})^2

``F`` is a forecast CDF (P(final score <= t)) sampled at every integer t in
[0, 300]; ``y`` is the realized outcome. A perfect point-mass forecast at the
true value scores 0; every run the point mass is off by adds exactly 1 to the
score (see the unit tests below) — so RPS is directly interpretable "in
runs".
"""

from __future__ import annotations

import numpy as np

MIN_THRESHOLD = 0
MAX_THRESHOLD = 300
THRESHOLDS = np.arange(MIN_THRESHOLD, MAX_THRESHOLD + 1)
N_THRESHOLDS = len(THRESHOLDS)


def point_mass_cdf(y: int) -> np.ndarray:
    """The CDF of a forecast certain the outcome is exactly ``y``."""
    return (THRESHOLDS >= y).astype(float)


def rps_score(F: np.ndarray, y) -> np.ndarray:
    """RPS for one or many (forecast, outcome) pairs.

    ``F``: shape (301,) for a single forecast, or (n, 301) for a batch.
    ``y``: a scalar, or shape (n,) matching the batch.

    Returns a scalar for a single forecast, or shape (n,) for a batch.
    """
    F = np.asarray(F, dtype=float)
    scalar_input = F.ndim == 1
    if scalar_input:
        F = F[None, :]
        y_arr = np.asarray([y], dtype=float)
    else:
        y_arr = np.asarray(y, dtype=float)
        if y_arr.ndim == 0:
            y_arr = np.full(F.shape[0], float(y))

    if F.shape[1] != N_THRESHOLDS:
        raise ValueError(f"F must have {N_THRESHOLDS} columns (thresholds 0..300), got {F.shape[1]}")
    if F.shape[0] != y_arr.shape[0]:
        raise ValueError(f"F has {F.shape[0]} rows but y has {y_arr.shape[0]} entries")

    indicator = (THRESHOLDS[None, :] >= y_arr[:, None]).astype(float)
    rps = np.sum((F - indicator) ** 2, axis=1)
    return rps[0] if scalar_input else rps
