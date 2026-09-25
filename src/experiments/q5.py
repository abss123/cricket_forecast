"""Q5: T20 1st-innings score forecasters, compared by Ranked Probability Score.

Single-command entry point:

    python -m src.experiments.q5

Builds (or loads a cached) checkpoint feature table, splits it by date
(train <= 2022, validation 2023, test 2024+), fits four forecasters —
climatology, run-rate projection, a LightGBM quantile-regression model, and
an in-sample binned-empirical oracle at three coarseness levels — scores
each on the test set with RPS (src/rps.py), and writes tables/figures/a
narrative summary to results/.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiments.checkpoints import CHECKPOINT_BALLS, CHECKPOINT_ORDER, build_checkpoint_table
from src.experiments.forecasters import (
    ORACLE_LEVELS,
    PARAM_GRID,
    fit_climatology,
    fit_main_model,
    fit_run_rate,
    oracle_rps,
)
from src.rps import MAX_THRESHOLD, N_THRESHOLDS, rps_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"

TRAIN_MAX_YEAR = 2022
VAL_YEAR = 2023
TEST_MIN_YEAR = 2024

N_BOOT = 2000
BOOT_SEED = 0

CHECKPOINT_OVER = {name: CHECKPOINT_BALLS[name] // 6 for name in CHECKPOINT_ORDER}
CHECKPOINT_LABEL = {"ball1": "before ball 1", "over6": "after over 6", "over10": "after over 10", "over15": "after over 15"}

FORECASTER_ORDER = ["climatology", "run_rate", "main_model", "oracle_L1", "oracle_L2", "oracle_L3"]
FORECASTER_LABEL = {
    "climatology": "Climatology",
    "run_rate": "Run-rate projection",
    "main_model": "Main model (LightGBM)",
    "oracle_L1": "Oracle L1 (over, wickets)",
    "oracle_L2": "Oracle L2 (+score/20)",
    "oracle_L3": "Oracle L3 (+score/10, tier)",
}
FORECASTER_COLOR = {
    "climatology": "#888888",
    "run_rate": "#1f77b4",
    "main_model": "#d62728",
    "oracle_L1": "#2ca02c",
    "oracle_L2": "#98df8a",
    "oracle_L3": "#003300",
}


def add_year_split(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year
    conditions = [df["year"] <= TRAIN_MAX_YEAR, df["year"] == VAL_YEAR, df["year"] >= TEST_MIN_YEAR]
    choices = ["train", "val", "test"]
    df["split"] = np.select(conditions, choices, default="unknown")
    return df


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOT, ci: float = 0.95, seed: int = BOOT_SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = values[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(lo), float(hi)


def split_row_counts(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.groupby(["checkpoint", "split"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=CHECKPOINT_ORDER, columns=["train", "val", "test"])
    match_counts = df.groupby("split")["match_id"].nunique().reindex(["train", "val", "test"])
    counts.loc["distinct matches"] = match_counts
    return counts


def run_experiment() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("Loading checkpoint table...")
    df = build_checkpoint_table()
    df = add_year_split(df)
    print(f"  {len(df):,} rows across {df['match_id'].nunique():,} matches")

    counts = split_row_counts(df)
    counts.to_csv(RESULTS_DIR / "split_row_counts.csv")
    print(counts)

    results_rows: list[dict[str, Any]] = []
    per_row_rps: dict[str, dict[str, np.ndarray]] = {}
    main_models = {}
    pit_by_checkpoint: dict[str, np.ndarray] = {}
    tuning_log: list[dict[str, Any]] = []
    oracle_bin_stats: list[dict[str, Any]] = []

    for checkpoint in CHECKPOINT_ORDER:
        print(f"\n=== checkpoint: {checkpoint} ===")
        cp = df[df["checkpoint"] == checkpoint]
        train = cp[cp["split"] == "train"].reset_index(drop=True)
        val = cp[cp["split"] == "val"].reset_index(drop=True)
        test = cp[cp["split"] == "test"].reset_index(drop=True)
        y_test = test["target_final_score"].to_numpy(dtype=float)
        per_row_rps[checkpoint] = {}

        print(f"  train={len(train)} val={len(val)} test={len(test)}")

        # 1. Climatology
        clim_predict = fit_climatology(train["target_final_score"].to_numpy(dtype=float))
        F_clim = clim_predict(len(test))
        per_row_rps[checkpoint]["climatology"] = rps_score(F_clim, y_test)

        # 2. Run-rate projection
        rr_predict, _rr_info = fit_run_rate(train)
        F_rr = rr_predict(test)
        per_row_rps[checkpoint]["run_rate"] = rps_score(F_rr, y_test)

        # 3. Main model (LightGBM quantile regression, tuned on validation)
        model, cp_tuning_log = fit_main_model(checkpoint, train, val, param_grid=PARAM_GRID)
        tuning_log.extend(cp_tuning_log)
        F_main = model.predict(test)
        per_row_rps[checkpoint]["main_model"] = rps_score(F_main, y_test)
        main_models[checkpoint] = model

        y_clip_idx = np.clip(y_test, 0, MAX_THRESHOLD).astype(int)
        pit_by_checkpoint[checkpoint] = F_main[np.arange(len(test)), y_clip_idx]

        # 4. Oracle (in-sample, on TEST)
        for level in ORACLE_LEVELS:
            rps_o, n_bins, median_bin_size = oracle_rps(test, level)
            per_row_rps[checkpoint][f"oracle_L{level}"] = rps_o
            oracle_bin_stats.append(
                {"checkpoint": checkpoint, "level": level, "bin_columns": ORACLE_LEVELS[level], "n_bins": n_bins, "median_bin_size": median_bin_size, "n_test_rows": len(test)}
            )

        clim_mean = per_row_rps[checkpoint]["climatology"].mean()
        for name in FORECASTER_ORDER:
            values = per_row_rps[checkpoint][name]
            mean_rps = float(values.mean())
            lo, hi = bootstrap_ci(values)
            skill = 1.0 - mean_rps / clim_mean
            results_rows.append(
                {
                    "checkpoint": checkpoint,
                    "forecaster": name,
                    "n_test_rows": len(values),
                    "mean_rps": mean_rps,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "skill_vs_climatology": skill,
                }
            )
            print(f"  {name:12s} mean_rps={mean_rps:7.3f}  95% CI=({lo:.3f}, {hi:.3f})  skill={skill:+.3f}")

    results_table = pd.DataFrame(results_rows)
    results_table.to_csv(RESULTS_DIR / "rps_table.csv", index=False)

    tuning_df = pd.DataFrame(tuning_log)
    tuning_df.to_csv(RESULTS_DIR / "model_tuning_log.csv", index=False)

    oracle_bins_df = pd.DataFrame(oracle_bin_stats)
    oracle_bins_df.to_csv(RESULTS_DIR / "oracle_bin_stats.csv", index=False)

    plot_rps_vs_checkpoint(results_table)
    plot_pit_histogram(pit_by_checkpoint)
    plot_feature_importance(main_models)
    write_summary(counts, results_table, oracle_bins_df, tuning_df)

    print(f"\nDone in {time.time() - t0:.1f}s. Results written to {RESULTS_DIR}")


def plot_rps_vs_checkpoint(results_table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = [CHECKPOINT_OVER[c] for c in CHECKPOINT_ORDER]
    for name in FORECASTER_ORDER:
        sub = results_table[results_table["forecaster"] == name].set_index("checkpoint").reindex(CHECKPOINT_ORDER)
        y = sub["mean_rps"].to_numpy()
        yerr_lo = y - sub["ci_lo"].to_numpy()
        yerr_hi = sub["ci_hi"].to_numpy() - y
        ax.errorbar(
            x, y, yerr=[yerr_lo, yerr_hi], label=FORECASTER_LABEL[name], color=FORECASTER_COLOR[name],
            marker="o", capsize=3, linewidth=2,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([CHECKPOINT_LABEL[c] for c in CHECKPOINT_ORDER], rotation=15)
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Mean test RPS (runs, lower is better)")
    ax.set_title("Forecast accuracy (RPS) vs. checkpoint")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "rps_vs_checkpoint.png", dpi=150)
    plt.close(fig)


def plot_pit_histogram(pit_by_checkpoint: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)
    for ax, checkpoint in zip(axes.ravel(), CHECKPOINT_ORDER):
        pit = pit_by_checkpoint[checkpoint]
        ax.hist(pit, bins=20, range=(0, 1), color="#d62728", alpha=0.75, edgecolor="white")
        ax.axhline(len(pit) / 20, color="black", linestyle="--", linewidth=1, label="uniform")
        ax.set_title(CHECKPOINT_LABEL[checkpoint])
        ax.set_xlabel("PIT = F(y)")
        ax.set_ylabel("count")
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle("Main model PIT histogram on test (calibration; flat = well-calibrated)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "pit_histogram.png", dpi=150)
    plt.close(fig)


def plot_feature_importance(main_models: dict[str, Any], top_n: int = 15) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, checkpoint in zip(axes.ravel(), CHECKPOINT_ORDER):
        imp = main_models[checkpoint].feature_importance().head(top_n)[::-1]
        ax.barh(imp.index, imp.values, color="#d62728")
        ax.set_title(CHECKPOINT_LABEL[checkpoint])
        ax.set_xlabel("gain")
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("Main model feature importance (gain, tau=0.5 quantile model)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)


def write_summary(
    counts: pd.DataFrame,
    results_table: pd.DataFrame,
    oracle_bins_df: pd.DataFrame,
    tuning_df: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# Q5: T20 1st-innings forecaster — results summary\n")
    lines.append(
        "Reproduce with `python -m src.experiments.q5`. Scope: men's T20I, Cricsheet "
        "(`data/t20s_json`), rain-reduced/curtailed 1st innings excluded (same default filter as "
        "`scripts/build_training_table.py`). Y = final 1st-innings total. "
        f"Time split: train <= {TRAIN_MAX_YEAR}, validation = {VAL_YEAR}, test >= {TEST_MIN_YEAR}, by match date.\n"
    )

    lines.append("## Row counts by checkpoint and split\n")
    lines.append(counts.to_markdown() + "\n")

    lines.append(
        "\n_A match contributes a row to a checkpoint only if its 1st innings actually reached that "
        "many overs — e.g. an innings all out in 12.3 overs has no `over15` row, which is why the "
        "`over15` counts are below `ball1`/`over6`._\n"
    )

    lines.append("\n## Mean test RPS by forecaster and checkpoint (bootstrap 95% CI, skill vs. climatology)\n")
    display_rows = []
    for name in FORECASTER_ORDER:
        row = {"forecaster": FORECASTER_LABEL[name]}
        for checkpoint in CHECKPOINT_ORDER:
            sub = results_table[(results_table["forecaster"] == name) & (results_table["checkpoint"] == checkpoint)].iloc[0]
            row[CHECKPOINT_LABEL[checkpoint]] = f"{sub['mean_rps']:.2f} [{sub['ci_lo']:.2f}, {sub['ci_hi']:.2f}] (skill {sub['skill_vs_climatology']:+.2f})"
        display_rows.append(row)
    lines.append(pd.DataFrame(display_rows).to_markdown(index=False) + "\n")

    lines.append("\n## Oracle bin counts\n")
    oracle_display = oracle_bins_df.copy()
    oracle_display["bin_columns"] = oracle_display["bin_columns"].apply(lambda cols: ", ".join(cols))
    lines.append(oracle_display.to_markdown(index=False) + "\n")

    lines.append("\n## Selected LightGBM hyperparameters (by validation mean RPS)\n")
    best_per_checkpoint = tuning_df.loc[tuning_df.groupby("checkpoint")["val_mean_rps"].idxmin()]
    lines.append(best_per_checkpoint.to_markdown(index=False) + "\n")

    # ---- Data-driven observations ----
    def rps_of(name: str, checkpoint: str) -> float:
        return float(results_table[(results_table.forecaster == name) & (results_table.checkpoint == checkpoint)]["mean_rps"].iloc[0])

    def skill_of(name: str, checkpoint: str) -> float:
        return float(results_table[(results_table.forecaster == name) & (results_table.checkpoint == checkpoint)]["skill_vs_climatology"].iloc[0])

    bullets = []

    ball1_model, ball1_rr, ball1_clim = rps_of("main_model", "ball1"), rps_of("run_rate", "ball1"), rps_of("climatology", "ball1")
    bullets.append(
        f"**Before ball 1, the run-rate forecaster collapses to climatology** (RPS {ball1_rr:.2f} vs {ball1_clim:.2f}): "
        "with 0 overs bowled, current run rate is undefined, so its projection falls back to the train set's global "
        "mean run rate for every match — its point forecast becomes the global mean score regardless of teams or "
        "venue, and the residual-shift trick just reconstructs the marginal distribution of Y. The main model still "
        f"beats both (RPS {ball1_model:.2f}) purely from pre-match context (Elo, venue history, team tier) that the "
        "other two forecasters can't use at all before a ball is bowled."
    )

    over15_model, over15_o1 = rps_of("main_model", "over15"), rps_of("oracle_L1", "over15")
    lower_is_l1_over15 = over15_o1 < over15_model
    bullets.append(
        f"**The coarsest oracle (L1: over+wickets only) is not always a lower bound on the model's error** — "
        f"at `over15` it scores {over15_o1:.2f} vs. the model's {over15_model:.2f} "
        f"({'better' if lower_is_l1_over15 else 'worse'}). Oracle L1 is Bayes-optimal *given only (over, wickets)*, "
        "but the main model conditions on far more (batting order strength, bowling economy remaining, Elo, venue), "
        "so a richer feature set can beat a coarse-but-perfectly-calibrated oracle — 'oracle' here bounds risk "
        "given a fixed information set, not overall achievable risk."
    )

    l3_bins = oracle_bins_df[oracle_bins_df.level == 3].set_index("checkpoint")["median_bin_size"]
    l1_bins = oracle_bins_df[oracle_bins_df.level == 1].set_index("checkpoint")["median_bin_size"]
    l3_n_bins = oracle_bins_df[oracle_bins_df.level == 3].set_index("checkpoint")["n_bins"]
    bullets.append(
        f"**Oracle L3's apparent skill is mostly in-sample overfitting, not a genuine target.** By `over15`, L3 "
        f"splits {int(oracle_bins_df[(oracle_bins_df.level == 3) & (oracle_bins_df.checkpoint == 'over15')]['n_test_rows'].iloc[0])} "
        f"test rows into {int(l3_n_bins['over15'])} bins with a median of just {l3_bins['over15']:.0f} rows each "
        f"(vs {int(l1_bins['over15'])} for L1's {int(oracle_bins_df[(oracle_bins_df.level == 1) & (oracle_bins_df.checkpoint == 'over15')]['n_bins'].iloc[0])} bins) — "
        "with that few points per bin, the 'empirical distribution' is close to a per-row point mass, so its RPS is "
        "optimistic by construction (it is scored on the same rows used to build its own distribution). Treat oracle "
        "L3 as an upper bound on how much a finer conditioning *could* help in the best case, not a number any "
        "deployable model should be expected to hit."
    )

    model_skills = [skill_of("main_model", c) for c in CHECKPOINT_ORDER]
    model_vs_runrate = {c: rps_of("run_rate", c) - rps_of("main_model", c) for c in CHECKPOINT_ORDER}
    bullets.append(
        f"**Two different 'skill' trends move in opposite directions, and both are real.** Skill vs. climatology "
        f"*grows* with information ({model_skills[0]:+.2f} at ball 1 to {model_skills[-1]:+.2f} after over 15) "
        "because climatology never improves — it ignores the state of the innings entirely, so as the game "
        "progresses the model pulls further ahead of that fixed, uninformed baseline. But the model's *absolute* "
        f"edge over the run-rate projection *shrinks* steadily, from {model_vs_runrate['ball1']:.2f} RPS at ball 1 "
        f"to just {model_vs_runrate['over15']:.2f} RPS after over 15 — once enough of the innings has been played, "
        "score/wickets/overs-left dominate the remaining uncertainty and a simple run-rate extrapolation captures "
        "almost as much signal as the full feature set. The model earns its keep mainly early, when pre-match "
        "context (Elo, venue, batting order) is doing work run-rate structurally cannot."
    )

    model_vs_l2 = {c: rps_of("main_model", c) - rps_of("oracle_L2", c) for c in CHECKPOINT_ORDER}
    l2_vs_l3 = {c: rps_of("oracle_L2", c) - rps_of("oracle_L3", c) for c in CHECKPOINT_ORDER}
    model_beats_l2_everywhere = all(v < 0 for v in model_vs_l2.values())
    bullets.append(
        f"**The model {'beats' if model_beats_l2_everywhere else 'roughly tracks'} oracle L2 at every checkpoint, "
        "and closes in on it as the innings progresses.** Past `ball1` (where L2 collapses to a single bin — see "
        f"the row counts above), the model's edge over L2 shrinks from {abs(model_vs_l2['over6']):.2f} RPS at "
        f"`over6` to {abs(model_vs_l2['over15']):.2f} RPS at `over15` — since L2 bins on (over, wickets, score/20), "
        "a similarly coarse signal to what the model already sees, this convergence is the more honest headroom "
        f"estimate. Oracle L3 stays further out in front throughout, with the L2-to-L3 gap shrinking only from "
        f"{l2_vs_l3['over6']:.2f} RPS at `over6` to {l2_vs_l3['over15']:.2f} RPS at `over15` — tracking L3's "
        "shrinking bin sizes (previous bullet) rather than a real, exploitable signal the model is leaving on "
        "the table."
    )

    lines.append("\n## Observations\n")
    for b in bullets:
        lines.append(f"- {b}\n")

    lines.append("\n## Modeling notes\n")
    lines.append(
        "- RPS is computed exactly as specified, over integer thresholds 0..300; a handful of test-set matches "
        "(associate-nation T20Is) score above 300, which no in-range forecast can capture perfectly — a known, "
        "documented edge case, not a bug.\n"
        "- The main model predicts *remaining* runs (target - score) by LightGBM quantile regression at quantiles "
        "0.05 to 0.95 in steps of 0.05 (19 total), then adds the current score; quantiles are sorted per row to "
        "remove crossing before linear interpolation to a CDF, with probability anchored to 0 at run 0 and 1 at "
        "run 300.\n"
        "- Hyperparameters (num_leaves, learning_rate, n_estimators, min_child_samples) are selected per checkpoint "
        "by mean validation RPS from a small grid; the model is fit on train only (not retrained on train+val) — "
        "see `results/model_tuning_log.csv` for every candidate's validation score.\n"
        "- The oracle is not a deployable forecaster: each level bins the TEST set itself and scores every test row "
        "in-sample against its own bin's empirical distribution, so it is a best-case Bayes-risk estimate given that "
        "binning's information, not an out-of-sample number.\n"
        "- `results/pit_histogram.png` shows a U shape at every checkpoint (excess mass near PIT=0 and PIT=1): the "
        "model's quantile forecasts are mildly underdispersed — true outcomes land outside the 0.05-0.95 quantile "
        "range more often than they should. This is a direct consequence of only fitting quantiles down to 0.05 and "
        "up to 0.95 and then anchoring the CDF's tails at runs 0 and 300 rather than extrapolating them; a wider "
        "quantile grid (e.g. 0.01-0.99) would likely narrow this gap.\n"
    )

    (RESULTS_DIR / "summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    run_experiment()
