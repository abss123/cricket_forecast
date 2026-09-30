"""The closed-form Student-t bucket probabilities match the Monte Carlo sampler of report Section 3."""

from __future__ import annotations

import numpy as np

from src.experiments.blr import B0, bucket_cdf, forecast, posterior, sample_buckets


def test_t_predictive_matches_sampler():
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(200), rng.normal(size=(200, 2))])
    r = X @ np.array([60.0, 8.0, -5.0]) + rng.normal(0, 15, 200)
    post = posterior(X, r, tau2=10.0)
    assert np.isclose(post[3], B0 + (((r - X @ post[0]) ** 2).sum() + post[0] @ post[0] / 10.0) / 2)
    score = 97
    # second case has predicted remaining runs near 0, so the clamp at the current score matters
    for x in (np.array([1.0, 0.5, -1.0]), np.array([1.0, -6.0, 0.0])):
        p = forecast(post, x[None, :], np.array([score]))[0][0]
        assert np.all(p[score // 10 :] > 0) and p[: score // 10].sum() == 0
        assert np.abs(p - sample_buckets(post, x, score, 400_000, rng)).max() < 0.005
        np.testing.assert_allclose(np.diff(bucket_cdf(p[None])[0], prepend=0)[:400], np.repeat(p[:40], 10) / 10, atol=1e-15)
