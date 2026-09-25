"""Unit tests for the Ranked Probability Score (src/rps.py)."""

from __future__ import annotations

import numpy as np
import pytest

from src.rps import MAX_THRESHOLD, MIN_THRESHOLD, N_THRESHOLDS, point_mass_cdf, rps_score


@pytest.mark.parametrize("y", [0, 1, 57, 150, 299, 300])
def test_point_mass_at_y_scores_zero(y):
    F = point_mass_cdf(y)
    assert rps_score(F, y) == pytest.approx(0.0)


@pytest.mark.parametrize("y,k", [(0, 1), (0, 10), (50, 1), (50, 25), (150, 50), (100, 200), (0, 300)])
def test_point_mass_at_y_plus_k_scores_k(y, k):
    """A point mass shifted k runs away from the truth scores exactly k."""
    assert y + k <= MAX_THRESHOLD
    F = point_mass_cdf(y + k)
    assert rps_score(F, y) == pytest.approx(float(k))


@pytest.mark.parametrize("y,k", [(50, 1), (50, 30), (150, 5)])
def test_point_mass_below_y_by_k_also_scores_k(y, k):
    """Symmetry: a point mass k runs on the other side of the truth also scores k."""
    assert y - k >= MIN_THRESHOLD
    F = point_mass_cdf(y - k)
    assert rps_score(F, y) == pytest.approx(float(k))


def test_point_mass_cdf_shape_and_monotonic():
    F = point_mass_cdf(150)
    assert F.shape == (N_THRESHOLDS,)
    assert F[149] == 0.0
    assert F[150] == 1.0
    assert np.all(np.diff(F) >= 0)


def test_rps_batch_matches_scalar_loop():
    ys = np.array([10, 50, 150, 290])
    Fs = np.stack([point_mass_cdf(y + 3) for y in ys])
    batch = rps_score(Fs, ys)
    loop = np.array([rps_score(point_mass_cdf(y + 3), y) for y in ys])
    assert batch.shape == (4,)
    np.testing.assert_allclose(batch, loop)
    np.testing.assert_allclose(batch, 3.0)


def test_uniform_forecast_is_worse_than_a_close_point_mass():
    y = 150
    uniform = np.full(N_THRESHOLDS, 0.5)
    close = point_mass_cdf(148)
    assert rps_score(uniform, y) > rps_score(close, y)


def test_wrong_shape_raises():
    with pytest.raises(ValueError):
        rps_score(np.zeros(300), 10)


def test_mismatched_batch_lengths_raise():
    with pytest.raises(ValueError):
        rps_score(np.zeros((3, N_THRESHOLDS)), np.array([1, 2]))
