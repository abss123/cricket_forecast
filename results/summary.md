# Q5: T20 1st-innings forecaster — results summary

Reproduce with `python -m src.experiments.q5`. Scope: men's T20I, Cricsheet (`data/t20s_json`), rain-reduced/curtailed 1st innings excluded (same default filter as `scripts/build_training_table.py`). Y = final 1st-innings total. Time split: train <= 2022, validation = 2023, test >= 2024, by match date.

## Row counts by checkpoint and split

| checkpoint       |   train |   val |   test |
|:-----------------|--------:|------:|-------:|
| ball1            |    1521 |   331 |   1479 |
| over6            |    1521 |   331 |   1479 |
| over10           |    1521 |   329 |   1477 |
| over15           |    1501 |   324 |   1439 |
| distinct matches |    1521 |   331 |   1479 |


_A match contributes a row to a checkpoint only if its 1st innings actually reached that many overs — e.g. an innings all out in 12.3 overs has no `over15` row, which is why the `over15` counts are below `ball1`/`over6`._


## Mean test RPS by forecaster and checkpoint (bootstrap 95% CI, skill vs. climatology)

| forecaster                  | before ball 1                      | after over 6                       | after over 10                      | after over 15                      |
|:----------------------------|:-----------------------------------|:-----------------------------------|:-----------------------------------|:-----------------------------------|
| Climatology                 | 26.52 [25.38, 27.64] (skill +0.00) | 26.52 [25.38, 27.64] (skill +0.00) | 26.41 [25.30, 27.52] (skill +0.00) | 25.17 [24.14, 26.27] (skill +0.00) |
| Run-rate projection         | 26.52 [25.38, 27.64] (skill +0.00) | 19.06 [18.33, 19.78] (skill +0.28) | 13.55 [13.06, 14.11] (skill +0.49) | 8.17 [7.87, 8.47] (skill +0.68)    |
| Main model (LightGBM)       | 22.35 [21.40, 23.30] (skill +0.16) | 16.20 [15.54, 16.85] (skill +0.39) | 12.47 [12.00, 12.99] (skill +0.53) | 8.08 [7.78, 8.39] (skill +0.68)    |
| Oracle L1 (over, wickets)   | 26.24 [25.19, 27.27] (skill +0.01) | 23.50 [22.57, 24.39] (skill +0.11) | 21.29 [20.48, 22.12] (skill +0.19) | 19.92 [19.10, 20.74] (skill +0.21) |
| Oracle L2 (+score/20)       | 26.24 [25.19, 27.27] (skill +0.01) | 17.33 [16.70, 17.94] (skill +0.35) | 12.93 [12.46, 13.48] (skill +0.51) | 8.36 [8.06, 8.70] (skill +0.67)    |
| Oracle L3 (+score/10, tier) | 25.28 [24.27, 26.30] (skill +0.05) | 15.51 [14.90, 16.14] (skill +0.42) | 11.22 [10.78, 11.68] (skill +0.58) | 6.78 [6.52, 7.07] (skill +0.73)    |


## Oracle bin counts

| checkpoint   |   level | bin_columns                                         |   n_bins |   median_bin_size |   n_test_rows |
|:-------------|--------:|:----------------------------------------------------|---------:|------------------:|--------------:|
| ball1        |       1 | _over, wickets                                      |        1 |            1479   |          1479 |
| ball1        |       2 | _over, wickets, _score_bucket_20                    |        1 |            1479   |          1479 |
| ball1        |       3 | _over, wickets, _score_bucket_10, batting_team_tier |        2 |             739.5 |          1479 |
| over6        |       1 | _over, wickets                                      |        8 |             172.5 |          1479 |
| over6        |       2 | _over, wickets, _score_bucket_20                    |       30 |              16.5 |          1479 |
| over6        |       3 | _over, wickets, _score_bucket_10, batting_team_tier |       87 |               6   |          1479 |
| over10       |       1 | _over, wickets                                      |       11 |              68   |          1477 |
| over10       |       2 | _over, wickets, _score_bucket_20                    |       55 |              10   |          1477 |
| over10       |       3 | _over, wickets, _score_bucket_10, batting_team_tier |      143 |               5   |          1477 |
| over15       |       1 | _over, wickets                                      |       11 |             115   |          1439 |
| over15       |       2 | _over, wickets, _score_bucket_20                    |       79 |               8   |          1439 |
| over15       |       3 | _over, wickets, _score_bucket_10, batting_team_tier |      202 |               3   |          1439 |


## Selected LightGBM hyperparameters (by validation mean RPS)

| checkpoint   |   num_leaves |   learning_rate |   n_estimators |   min_child_samples |   val_mean_rps |
|:-------------|-------------:|----------------:|---------------:|--------------------:|---------------:|
| ball1        |           15 |            0.05 |            200 |                  20 |       21.8017  |
| over10       |           15 |            0.05 |            200 |                  20 |       12.7942  |
| over15       |           15 |            0.05 |            200 |                  20 |        8.18823 |
| over6        |           15 |            0.05 |            200 |                  20 |       16.6747  |


## Observations

- **Before ball 1, the run-rate forecaster collapses to climatology** (RPS 26.52 vs 26.52): with 0 overs bowled, current run rate is undefined, so its projection falls back to the train set's global mean run rate for every match — its point forecast becomes the global mean score regardless of teams or venue, and the residual-shift trick just reconstructs the marginal distribution of Y. The main model still beats both (RPS 22.35) purely from pre-match context (Elo, venue history, team tier) that the other two forecasters can't use at all before a ball is bowled.

- **The coarsest oracle (L1: over+wickets only) is not always a lower bound on the model's error** — at `over15` it scores 19.92 vs. the model's 8.08 (worse). Oracle L1 is Bayes-optimal *given only (over, wickets)*, but the main model conditions on far more (batting order strength, bowling economy remaining, Elo, venue), so a richer feature set can beat a coarse-but-perfectly-calibrated oracle — 'oracle' here bounds risk given a fixed information set, not overall achievable risk.

- **Oracle L3's apparent skill is mostly in-sample overfitting, not a genuine target.** By `over15`, L3 splits 1439 test rows into 202 bins with a median of just 3 rows each (vs 115 for L1's 11 bins) — with that few points per bin, the 'empirical distribution' is close to a per-row point mass, so its RPS is optimistic by construction (it is scored on the same rows used to build its own distribution). Treat oracle L3 as an upper bound on how much a finer conditioning *could* help in the best case, not a number any deployable model should be expected to hit.

- **Two different 'skill' trends move in opposite directions, and both are real.** Skill vs. climatology *grows* with information (+0.16 at ball 1 to +0.68 after over 15) because climatology never improves — it ignores the state of the innings entirely, so as the game progresses the model pulls further ahead of that fixed, uninformed baseline. But the model's *absolute* edge over the run-rate projection *shrinks* steadily, from 4.16 RPS at ball 1 to just 0.09 RPS after over 15 — once enough of the innings has been played, score/wickets/overs-left dominate the remaining uncertainty and a simple run-rate extrapolation captures almost as much signal as the full feature set. The model earns its keep mainly early, when pre-match context (Elo, venue, batting order) is doing work run-rate structurally cannot.

- **The model beats oracle L2 at every checkpoint, and closes in on it as the innings progresses.** Past `ball1` (where L2 collapses to a single bin — see the row counts above), the model's edge over L2 shrinks from 1.13 RPS at `over6` to 0.28 RPS at `over15` — since L2 bins on (over, wickets, score/20), a similarly coarse signal to what the model already sees, this convergence is the more honest headroom estimate. Oracle L3 stays further out in front throughout, with the L2-to-L3 gap shrinking only from 1.83 RPS at `over6` to 1.58 RPS at `over15` — tracking L3's shrinking bin sizes (previous bullet) rather than a real, exploitable signal the model is leaving on the table.


## Modeling notes

- RPS is computed exactly as specified, over integer thresholds 0..300; a handful of test-set matches (associate-nation T20Is) score above 300, which no in-range forecast can capture perfectly — a known, documented edge case, not a bug.
- The main model predicts *remaining* runs (target - score) by LightGBM quantile regression at quantiles 0.05 to 0.95 in steps of 0.05 (19 total), then adds the current score; quantiles are sorted per row to remove crossing before linear interpolation to a CDF, with probability anchored to 0 at run 0 and 1 at run 300.
- Hyperparameters (num_leaves, learning_rate, n_estimators, min_child_samples) are selected per checkpoint by mean validation RPS from a small grid; the model is fit on train only (not retrained on train+val) — see `results/model_tuning_log.csv` for every candidate's validation score.
- The oracle is not a deployable forecaster: each level bins the TEST set itself and scores every test row in-sample against its own bin's empirical distribution, so it is a best-case Bayes-risk estimate given that binning's information, not an out-of-sample number.
- `results/pit_histogram.png` shows a U shape at every checkpoint (excess mass near PIT=0 and PIT=1): the model's quantile forecasts are mildly underdispersed — true outcomes land outside the 0.05-0.95 quantile range more often than they should. This is a direct consequence of only fitting quantiles down to 0.05 and up to 0.95 and then anchoring the CDF's tails at runs 0 and 300 rather than extrapolating them; a wider quantile grid (e.g. 0.01-0.99) would likely narrow this gap.
