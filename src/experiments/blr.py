"""Bayesian linear regression forecaster of the 1st-innings total (report Sections 3 and 6).

    python -m src.experiments.blr

Per checkpoint: conjugate normal--inverse-gamma regression of the remaining
runs on the report's Table 2 features, fit on train (<= 2022) with the prior
scale tau^2 picked by validation (2023) RPS, scored on test (2024+); splits
come from the frozen checkpoint in data/checkpoints/first_innings/. Bucket
probabilities come from the closed-form Student-t posterior predictive;
``sample_buckets`` is the Monte Carlo sampler of report Section 3, which
tests/test_blr.py checks against it. The ECDF baseline is the empirical
distribution of training totals, bucketed like the model's forecasts.
Writes results/blr_summary.md and report/figs/blr_demo.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import special, stats

from src.experiments.checkpoints import CHECKPOINT_ORDER, load_splits
from src.rps import rps_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Report Table 2. Balls remaining is constant within a checkpoint, and the in-play
# columns are constant or missing before ball 1; make_design drops such columns.
NUMERIC = (
    "score", "wickets", "balls_remaining", "run_rate_last_5", "partnership_runs",
    "striker_runs", "striker_strike_rate", "non_striker_runs", "non_striker_strike_rate",
    "striker_career_avg_t20i", "striker_career_sr_t20i", "non_striker_career_avg_t20i", "non_striker_career_sr_t20i",
    "remaining_batting_order_sr", "remaining_batting_order_avg", "remaining_bowling_economy_weighted",
    "elo_batting", "elo_bowling", "elo_diff", "elo_n_batting", "elo_n_bowling",
    "team_batting_recent_form", "team_bowling_recent_form",
    "venue_avg_first_innings_score", "venue_n_prior_matches",
)
CATEGORICAL = ("batting_team_tier", "bowling_team_tier", "home_away")
TAU2_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
A0 = B0 = 0.01  # weakly informative Inv-Gamma(a0, b0) prior on sigma^2
N_BUCKETS = 41  # B_j = {10j, ..., 10j+9} for j < 40, and B_40 = {400, 401, ...}
DEMO_MATCH = "1415755"  # 2024 T20 World Cup final, India v South Africa
LABEL = {"ball1": "Before ball 1", "over6": "After over 6", "over10": "After over 10", "over15": "After over 15"}


def bucket_of(total):
    return np.minimum(np.asarray(total) // 10, 40).astype(int)


def make_design(train: pd.DataFrame):
    """Return a map frame -> [1, standardized features], using train means and SDs."""
    def dummies(df):
        return pd.get_dummies(df[list(NUMERIC + CATEGORICAL)], columns=list(CATEGORICAL), dtype=float)

    Z = dummies(train)
    keep = Z.std() > 0  # drops constant and all-missing columns
    mu, sd = Z.mean()[keep], Z.std()[keep]

    def design(df):
        # ponytail: missing values (debutants' career stats) are mean-imputed; add missingness flags if they matter.
        z = ((dummies(df).reindex(columns=mu.index, fill_value=0.0) - mu) / sd).fillna(0.0)
        return np.column_stack([np.ones(len(df)), z.to_numpy()])

    return design


def posterior(X, r, tau2):
    """Normal--inverse-gamma posterior (m_n, V_n, a_n, b_n) under m_0 = 0, V_0 = tau2 * I."""
    Vn = np.linalg.inv(np.eye(X.shape[1]) / tau2 + X.T @ X)
    mn = Vn @ (X.T @ r)
    return mn, Vn, A0 + len(r) / 2, B0 + (r @ r - mn @ (X.T @ r)) / 2  # V_n^{-1} m_n = X'r


def forecast(post, X, score):
    """Bucket probabilities (n, 41), best estimate and predictive variance of the total Y.

    The remaining runs R are Student-t with 2 a_n degrees of freedom, location x'm_n and
    squared scale (b_n / a_n)(1 + x'V_n x). With Y = s + max(0, round(R)),
    P(Y <= t) = T((t + 0.5 - s - x'm_n) / scale) for t >= s, and 0 below the current score s.
    """
    mn, Vn, an, bn = post
    nu = 2 * an
    yhat = score + X @ mn
    scale = np.sqrt(bn / an * (1 + np.einsum("ij,jk,ik->i", X, Vn, X)))
    tops = np.arange(9, 400, 10)  # highest total in buckets 0..39
    # P(Y > t) via sf, not 1 - cdf, so far upper-tail buckets keep positive probability instead of underflowing to 0
    S = stats.t.sf((tops + 0.5 - yhat[:, None]) / scale[:, None], nu)
    S[tops < score[:, None]] = 1.0
    return -np.diff(S, prepend=1.0, append=0.0, axis=1), yhat, scale**2 * nu / (nu - 2)


def sample_buckets(post, x, score, S, rng):
    """Monte Carlo bucket probabilities for one innings (report Section 3, steps 1-3)."""
    mn, Vn, an, bn = post
    sigma2 = stats.invgamma.rvs(an, scale=bn, size=S, random_state=rng)
    beta = mn + np.sqrt(sigma2)[:, None] * rng.multivariate_normal(np.zeros(len(mn)), Vn, size=S)
    total = score + np.maximum(0, np.round(rng.normal(beta @ x, np.sqrt(sigma2))))
    return np.bincount(bucket_of(total), minlength=N_BUCKETS) / S


def bucket_cdf(p):
    """Report Eq. (bucketcdf): spread each bucket uniformly over its ten totals, with F(400) = 1."""
    t = np.arange(400)
    F = (np.cumsum(p, axis=1) - p)[:, t // 10] + (t % 10 + 1) / 10 * p[:, t // 10]
    return np.hstack([F, np.ones((len(p), 1))])


def scores(p, yhat, var, y):
    """Per-innings realized scores, and their expectations if the forecast were the truth."""
    F, i, j, jhat = bucket_cdf(p), np.arange(len(y)), bucket_of(y), bucket_of(np.round(yhat))
    return pd.DataFrame({
        "rps": rps_score(F, y), "rps_opt": (F * (1 - F)).sum(1),
        "log": -np.log(p[i, j]), "log_opt": special.entr(p).sum(1),
        "mse": (y - yhat) ** 2, "mse_opt": var,
        "hit": (jhat == j).astype(float), "hit_opt": p[i, jhat],
        "bias": y - yhat, "cov90": (np.abs(F[i, np.minimum(y, 400).astype(int)] - 0.5) <= 0.45).astype(float),
    })


def ecdf_buckets(y):
    """ECDF baseline: bucket probabilities of the empirical distribution of totals y, bucketed like the model."""
    return np.bincount(bucket_of(y), minlength=N_BUCKETS) / len(y)


def marginal_optimum(y):
    """Optimal expected scores without features: each score's entropy of the bucketed ECDF of y."""
    g = ecdf_buckets(y)
    G = bucket_cdf(g[None])[0]
    return {"rps": (G * (1 - G)).sum(), "log": special.entr(g).sum(), "mse": y.var()}


def run() -> None:
    rows, demo = [], []

    for cp in CHECKPOINT_ORDER:
        train, val, test = load_splits(cp)
        y_train, y_val, y_test = (f["final_total"].to_numpy(float) for f in (train, val, test))
        design = make_design(train)
        X = design(train)
        r = y_train - train["score"].to_numpy(float)

        def predict(post, frame):
            return forecast(post, design(frame), frame["score"].to_numpy(float))

        tau2 = min(TAU2_GRID, key=lambda t: rps_score(bucket_cdf(predict(posterior(X, r, t), val)[0]), y_val).mean())
        post = posterior(X, r, tau2)
        p, yhat, var = predict(post, test)
        per_row = scores(p, yhat, var, y_test)
        ecdf = ecdf_buckets(y_train)
        ecdf_rps = rps_score(bucket_cdf(np.tile(ecdf, (len(test), 1))), y_test).mean()
        rows.append({
            "checkpoint": LABEL[cp], "n_train": len(train), "n_val": len(val), "n_test": len(test),
            "tau2": tau2, "n_features": X.shape[1] - 1, "nu": 2 * post[2],
            "ecdf_rps": ecdf_rps, "skill": 1 - per_row["rps"].mean() / ecdf_rps,
            "ecdf_n_zero_prob": (ecdf[bucket_of(y_test)] == 0).sum(),
            "rps_se": per_row["rps"].std() / np.sqrt(len(test)), **per_row.mean(), "log_max": per_row["log"].max(),
            "n_above_train_max": (y_test > y_train.max()).sum(),
            **{f"marg_{k}": v for k, v in marginal_optimum(y_test).items()},
        })
        k = np.flatnonzero((test["match_id"] == DEMO_MATCH).to_numpy())[0]
        demo.append((cp, p[k], yhat[k], bucket_of(round(yhat[k])), int(test["score"].iloc[k]), int(test["wickets"].iloc[k]), y_test[k]))

    table = pd.DataFrame(rows).set_index("checkpoint").T
    counts = ["n_train", "n_val", "n_test", "n_features", "ecdf_n_zero_prob", "n_above_train_max"]
    table = table.apply(lambda row: row.map((lambda v: f"{int(v):,}") if row.name in counts else (lambda v: f"{v:.4f}")), axis=1)
    out = PROJECT_ROOT / "results" / "blr_summary.md"
    out.write_text(
        "# Bayesian linear regression: test results\n\nReproduce with `python -m src.experiments.blr`. "
        "Splits from data/checkpoints/first_innings/: train <= 2022, validation 2023 (picks tau2), test >= 2024.\n\n"
        "`ecdf_*`: the ECDF baseline, i.e. the empirical distribution of training totals bucketed like the model's forecasts; "
        "`skill` is relative to it, and `ecdf_n_zero_prob` counts test innings whose bucket it gives probability 0. "
        "`*_opt`: model-implied optimum. `marg_*`: no-features optimum, the entropy of the bucketed ECDF of test totals.\n\n"
        + table.to_markdown(colalign=("left",) + ("right",) * table.shape[1]) + "\n\n## Demo: " + f"match {DEMO_MATCH}\n\n"
        + pd.DataFrame(
            [(LABEL[c], f"{s}/{w}", yh, f"{10 * j}-{10 * j + 9}", pr[j], y, pr[bucket_of(y)]) for c, pr, yh, j, s, w, y in demo],
            columns=["checkpoint", "score", "best estimate", "bucket", "P(bucket)", "actual", "P(actual bucket)"],
        ).to_markdown(index=False, floatfmt=".4f") + "\n"
    )
    plot_demo(demo, PROJECT_ROOT / "report" / "figs" / "blr_demo.png")
    print(f"wrote {out}")


def plot_demo(demo, path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 4.4), sharex=True, sharey=True)
    for ax, (cp, p, yhat, jhat, s, w, y) in zip(axes.flat, demo):
        ax.bar(np.arange(N_BUCKETS) * 10 + 4.5, p, width=8, color="#a9c1e8", label="bucket probability")
        ax.bar(jhat * 10 + 4.5, p[jhat], width=8, color="#1f4e9c", label="best-estimate bucket")
        ax.axvline(y, color="black", linestyle="--", linewidth=1, label=f"actual total ({y:.0f})")
        ax.set_title(f"{LABEL[cp]} (India {s}/{w}): estimate {yhat:.0f}, P = {p[jhat]:.2f}", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(fontsize=8, frameon=False)
    for ax in axes[1]:
        ax.set_xlabel("first-innings total (runs)")
    for ax in axes[:, 0]:
        ax.set_ylabel("probability")
    axes[0, 0].set_xlim(80, 270)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    run()
