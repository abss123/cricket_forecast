"""The four 1st-innings-total forecasters compared in the q5 experiment.

Every forecaster exposes a ``predict(df) -> np.ndarray`` of shape
(len(df), 301), a CDF over the integer run thresholds 0..300 (see
``src/rps.py``), so all four can be scored with the same RPS function.

1. ``fit_climatology``   — empirical distribution of train totals (no
   per-row information used at all).
2. ``fit_run_rate``      — score + current-run-rate * overs-left, dressed up
   with the empirical distribution of train residuals *at that checkpoint*.
3. ``MainModel``         — LightGBM quantile regression on remaining runs,
   tuned on validation, quantiles sorted then interpolated to a CDF.
4. ``oracle_rps``        — not a deployable forecaster: bins the TEST set
   itself at three coarseness levels and scores each row in-sample against
   its own bin's empirical distribution, as a Bayes-risk lower-bound
   estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.rps import MAX_THRESHOLD, MIN_THRESHOLD, N_THRESHOLDS, THRESHOLDS, rps_score

SCHEDULED_OVERS = 20.0

QUANTILES: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(1, 20))  # 0.05, 0.10, ..., 0.95

NUMERIC_FEATURES: tuple[str, ...] = (
    "legal_balls_bowled",
    "balls_remaining",
    "score",
    "wickets",
    "run_rate",
    "run_rate_last_3",
    "wickets_last_3",
    "run_rate_last_5",
    "wickets_last_5",
    "partnership_runs",
    "partnership_balls",
    "striker_runs",
    "striker_balls",
    "striker_strike_rate",
    "non_striker_runs",
    "non_striker_balls",
    "non_striker_strike_rate",
    "fours",
    "sixes",
    "extras_total",
    "striker_career_sr_t20i",
    "striker_career_avg_t20i",
    "striker_career_innings_t20i",
    "non_striker_career_sr_t20i",
    "non_striker_career_avg_t20i",
    "non_striker_career_innings_t20i",
    "remaining_batters_count",
    "remaining_batting_order_sr",
    "remaining_batting_order_avg",
    "bowlers_with_overs_remaining",
    "remaining_bowling_economy_weighted",
    "elo_batting",
    "elo_bowling",
    "elo_diff",
    "elo_n_batting",
    "elo_n_bowling",
    "team_batting_recent_form",
    "team_batting_recent_form_n",
    "team_bowling_recent_form",
    "team_bowling_recent_form_n",
    "venue_avg_first_innings_score",
    "venue_n_prior_matches",
)
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "batting_team_tier",
    "bowling_team_tier",
    "tier_pairing",
    "home_away",
    "toss_decision",
    "phase",
)
FEATURE_COLUMNS: tuple[str, ...] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

PARAM_GRID: tuple[dict[str, Any], ...] = (
    {"num_leaves": 15, "learning_rate": 0.05, "n_estimators": 200, "min_child_samples": 20},
    {"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 200, "min_child_samples": 20},
    {"num_leaves": 31, "learning_rate": 0.10, "n_estimators": 100, "min_child_samples": 30},
    {"num_leaves": 63, "learning_rate": 0.05, "n_estimators": 300, "min_child_samples": 10},
)


def _prepare_X(df: pd.DataFrame) -> pd.DataFrame:
    X = df[list(FEATURE_COLUMNS)].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")
    return X


# ---------------------------------------------------------------------------
# 1. Climatology
# ---------------------------------------------------------------------------


def fit_climatology(train_y: np.ndarray) -> Callable[[int], np.ndarray]:
    """Empirical CDF of ``train_y`` (final totals), independent of any row features."""
    sorted_y = np.sort(np.asarray(train_y, dtype=float))
    n = len(sorted_y)
    cdf = np.searchsorted(sorted_y, THRESHOLDS, side="right") / n

    def predict(n_rows: int) -> np.ndarray:
        return np.tile(cdf, (n_rows, 1))

    return predict


# ---------------------------------------------------------------------------
# 2. Run-rate projection + empirical residuals
# ---------------------------------------------------------------------------


def _projection(df: pd.DataFrame, global_run_rate: float) -> np.ndarray:
    score = df["score"].to_numpy(dtype=float)
    overs_completed = df["legal_balls_bowled"].to_numpy(dtype=float) / 6.0
    overs_left = SCHEDULED_OVERS - overs_completed
    crr = df["run_rate"].to_numpy(dtype=float)
    crr_filled = np.where(np.isnan(crr), global_run_rate, crr)
    return score + crr_filled * overs_left


def fit_run_rate(train_df: pd.DataFrame) -> tuple[Callable[[pd.DataFrame], np.ndarray], dict[str, Any]]:
    """score + CRR*overs_left, shifted by the empirical train-residual distribution.

    ``CRR`` (current run rate) is undefined before any legal ball is bowled
    (the ``ball1`` checkpoint) — there, it falls back to the train set's
    global mean run rate, making the projection equal the global mean final
    score for every row at that checkpoint (a deliberate, documented
    fallback, not a bug: see results/summary.md).
    """
    global_run_rate = train_df["target_final_score"].mean() / SCHEDULED_OVERS
    projection = _projection(train_df, global_run_rate)
    residuals = train_df["target_final_score"].to_numpy(dtype=float) - projection
    sorted_resid = np.sort(residuals)
    n = len(sorted_resid)

    def predict(df: pd.DataFrame) -> np.ndarray:
        proj = _projection(df, global_run_rate)
        needed = THRESHOLDS[None, :] - proj[:, None]
        idx = np.searchsorted(sorted_resid, needed, side="right")
        return idx / n

    info = {"global_run_rate": global_run_rate, "n_train_residuals": n}
    return predict, info


# ---------------------------------------------------------------------------
# 3. Main model: LightGBM quantile regression on remaining runs
# ---------------------------------------------------------------------------


@dataclass
class MainModel:
    checkpoint: str
    params: dict[str, Any]
    models: dict[float, lgb.LGBMRegressor]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = _prepare_X(df)
        score = df["score"].to_numpy(dtype=float)
        n = len(df)
        remaining_q = np.empty((n, len(QUANTILES)))
        for i, tau in enumerate(QUANTILES):
            pred = self.models[tau].predict(X)
            remaining_q[:, i] = np.clip(pred, 0.0, None)
        remaining_q.sort(axis=1)  # enforce non-crossing quantiles
        # Clip strictly inside (MIN_THRESHOLD, MAX_THRESHOLD) so the anchor points
        # added below stay strictly outside the quantile points even after clipping
        # ties two or more quantiles to the same boundary value.
        final_q = np.clip(score[:, None] + remaining_q, MIN_THRESHOLD, MAX_THRESHOLD - 1e-3)

        F = np.empty((n, N_THRESHOLDS))
        nudge = np.arange(1, final_q.shape[1] + 1) * 1e-6  # force strictly increasing x for interpolation
        for i in range(n):
            x = final_q[i] + nudge
            x_anchored = np.concatenate([[MIN_THRESHOLD], x, [MAX_THRESHOLD]])
            y_anchored = np.concatenate([[0.0], QUANTILES, [1.0]])
            F[i] = np.interp(THRESHOLDS, x_anchored, y_anchored, left=0.0, right=1.0)
        return F

    def feature_importance(self, importance_type: str = "gain") -> pd.Series:
        """Feature importance of the median (tau=0.5) quantile model."""
        model = self.models[0.5]
        imp = pd.Series(model.booster_.feature_importance(importance_type=importance_type), index=FEATURE_COLUMNS)
        return imp.sort_values(ascending=False)


def _fit_quantile_models(
    X_train: pd.DataFrame, y_remaining_train: np.ndarray, params: dict[str, Any]
) -> dict[float, lgb.LGBMRegressor]:
    models: dict[float, lgb.LGBMRegressor] = {}
    for tau in QUANTILES:
        model = lgb.LGBMRegressor(objective="quantile", alpha=tau, random_state=0, verbosity=-1, **params)
        model.fit(X_train, y_remaining_train)
        models[tau] = model
    return models


def fit_main_model(
    checkpoint: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    param_grid: tuple[dict[str, Any], ...] = PARAM_GRID,
) -> tuple[MainModel, list[dict[str, Any]]]:
    """Fit quantile models on TRAIN, select hyperparameters by mean RPS on VALIDATION."""
    X_train = _prepare_X(train_df)
    y_remaining_train = (train_df["target_final_score"] - train_df["score"]).to_numpy(dtype=float)

    tuning_log: list[dict[str, Any]] = []
    best_params: dict[str, Any] | None = None
    best_models: dict[float, lgb.LGBMRegressor] | None = None
    best_val_rps = np.inf

    for params in param_grid:
        models = _fit_quantile_models(X_train, y_remaining_train, params)
        candidate = MainModel(checkpoint=checkpoint, params=params, models=models)
        F_val = candidate.predict(val_df)
        val_rps = float(np.mean(rps_score(F_val, val_df["target_final_score"].to_numpy(dtype=float))))
        tuning_log.append({"checkpoint": checkpoint, **params, "val_mean_rps": val_rps})
        if val_rps < best_val_rps:
            best_val_rps = val_rps
            best_params = params
            best_models = models

    assert best_params is not None and best_models is not None
    return MainModel(checkpoint=checkpoint, params=best_params, models=best_models), tuning_log


# ---------------------------------------------------------------------------
# 4. Oracle (in-sample Bayes-risk estimate on the TEST set)
# ---------------------------------------------------------------------------

ORACLE_LEVELS: dict[int, tuple[str, ...]] = {
    1: ("_over", "wickets"),
    2: ("_over", "wickets", "_score_bucket_20"),
    3: ("_over", "wickets", "_score_bucket_10", "batting_team_tier"),
}


def _empirical_cdf(y_bin: np.ndarray) -> np.ndarray:
    sorted_y = np.sort(y_bin.astype(float))
    n = len(sorted_y)
    return np.searchsorted(sorted_y, THRESHOLDS, side="right") / n


def oracle_rps(test_df: pd.DataFrame, level: int) -> tuple[np.ndarray, int, float]:
    """Per-row RPS for the in-sample binned-empirical oracle at ``level`` coarseness.

    Returns ``(rps_per_row, n_bins, median_bin_size)``. Deliberately
    in-sample: each test row is scored against the empirical outcome
    distribution of the bin it itself belongs to (computed from the TEST set
    only) — an optimistic lower bound on achievable RPS given only that
    binning's information, not a fair out-of-sample estimate.
    """
    df = test_df.reset_index(drop=True).copy()
    df["_over"] = (df["legal_balls_bowled"] // 6).astype(int)
    df["_score_bucket_20"] = (df["score"] // 20 * 20).astype(int)
    df["_score_bucket_10"] = (df["score"] // 10 * 10).astype(int)

    bin_cols = list(ORACLE_LEVELS[level])
    y = df["target_final_score"].to_numpy(dtype=float)
    rps_values = np.empty(len(df))
    bin_sizes: list[int] = []

    for _, idx in df.groupby(bin_cols).groups.items():
        pos = df.index.get_indexer(idx)
        y_bin = y[pos]
        bin_sizes.append(len(y_bin))
        F_bin = _empirical_cdf(y_bin)
        rps_values[pos] = rps_score(np.tile(F_bin, (len(y_bin), 1)), y_bin)

    return rps_values, len(bin_sizes), float(np.median(bin_sizes))
