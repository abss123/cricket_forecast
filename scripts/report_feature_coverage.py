"""Step 5: feature coverage / season / correlation report over the training table.

Reads ``data/processed/train_1st_innings.parquet`` (built by
``scripts/build_training_table.py``) and writes ``docs/feature_coverage_report.md``.

Usage:
    python scripts/report_feature_coverage.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.features.team_tiers import team_tier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train_1st_innings.parquet"
OUT_PATH = PROJECT_ROOT / "docs" / "feature_coverage_report.md"

OVER_CHECKPOINTS = {6: 36, 10: 60, 15: 90}

NON_FEATURE_COLUMNS = {
    "match_id", "date", "batting_team", "bowling_team", "venue", "striker", "non_striker",
    "target_final_score", "toss_winner",
}


def _missing_pct_overall(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isna().mean().mul(100).round(2).sort_values(ascending=False)
    return missing.rename("pct_missing").to_frame()


def _missing_pct_by_season(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df.groupby("season")[columns].apply(lambda g: g.isna().mean().mul(100).round(1))


def _rows_and_matches_per_season(df: pd.DataFrame) -> pd.DataFrame:
    per_match = df.drop_duplicates("match_id")
    rows = df.groupby("season").size().rename("n_rows")
    matches = per_match.groupby("season").size().rename("n_matches")
    tier_pairs = (
        per_match.groupby(["season", "tier_pairing"]).size().unstack(fill_value=0)
    )
    out = pd.concat([rows, matches], axis=1).join(tier_pairs)
    out["post_2019"] = [int(str(s)[:4]) >= 2019 if str(s)[:4].isdigit() else None for s in out.index]
    return out.sort_index()


def _low_sample_entities(df: pd.DataFrame, threshold: int = 5) -> dict[str, object]:
    """Unique-entity counts use each entity's *highest-ever* point-in-time
    prior-match count in this dataset (i.e. "never accumulates >= threshold
    matches even by its last appearance"), not its first appearance — every
    entity starts at n=0, so "first" or "min" would trivially flag all of
    them."""
    per_match = df.drop_duplicates("match_id")

    venue_n = per_match.groupby("venue")["venue_n_prior_matches"].max()
    # One n-count per (team, role) appearance, deduped at the match level.
    batting_side = per_match[["batting_team", "team_batting_recent_form_n"]].rename(
        columns={"batting_team": "team", "team_batting_recent_form_n": "n"}
    )
    bowling_side = per_match[["bowling_team", "team_bowling_recent_form_n"]].rename(
        columns={"bowling_team": "team", "team_bowling_recent_form_n": "n"}
    )
    team_appearances = pd.concat([batting_side, bowling_side])
    team_max_n = team_appearances.groupby("team")["n"].max()

    return {
        "n_venues_total": venue_n.shape[0],
        "n_venues_never_reach_threshold": int((venue_n < threshold).sum()),
        "pct_rows_venue_lt_threshold": round((df["venue_n_prior_matches"] < threshold).mean() * 100, 1),
        "n_teams_total": team_max_n.shape[0],
        "n_teams_never_reach_threshold": int((team_max_n < threshold).sum()),
        "pct_rows_team_form_lt_threshold": round(
            ((df["team_batting_recent_form_n"] < threshold) | (df["team_bowling_recent_form_n"] < threshold)).mean() * 100, 1
        ),
    }


def _player_rating_coverage(df: pd.DataFrame) -> dict[str, float]:
    has_t20i_history = (df["striker_career_innings_t20i"] >= 1) | (df["non_striker_career_innings_t20i"] >= 1)
    return {
        "pct_rows_striker_has_prior_t20i_innings": round((df["striker_career_innings_t20i"] >= 1).mean() * 100, 1),
        "pct_rows_either_batter_has_prior_t20i_innings": round(has_t20i_history.mean() * 100, 1),
        "pct_rows_all_t20_version_available": 0.0,
    }


def _correlation_at_checkpoints(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number, bool]).columns if c not in NON_FEATURE_COLUMNS
    ]
    out = {}
    for label, balls in OVER_CHECKPOINTS.items():
        subset = df[df["legal_balls_bowled"] == balls]
        if subset.empty:
            continue
        corr = subset[numeric_cols + ["target_final_score"]].corr(numeric_only=True)["target_final_score"]
        out[f"after_over_{label}"] = corr.drop("target_final_score")
    result = pd.DataFrame(out)
    result["max_abs"] = result.abs().max(axis=1)
    return result.sort_values("max_abs", ascending=False).drop(columns="max_abs").round(3)


def _score_distribution_by_tier_pairing(df: pd.DataFrame) -> pd.DataFrame:
    per_match = df.drop_duplicates("match_id")
    return (
        per_match.groupby("tier_pairing")["target_final_score"]
        .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        .round(1)
    )


def main() -> None:
    df = pd.read_parquet(TRAIN_PATH)

    feature_columns = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]

    missing_overall = _missing_pct_overall(df)
    rows_matches_per_season = _rows_and_matches_per_season(df)
    low_sample = _low_sample_entities(df)
    player_coverage = _player_rating_coverage(df)
    correlations = _correlation_at_checkpoints(df)
    score_by_tier = _score_distribution_by_tier_pairing(df)

    lines: list[str] = []
    lines.append("# Feature coverage report\n")
    lines.append(f"Generated from `{TRAIN_PATH.relative_to(PROJECT_ROOT)}`: "
                  f"{len(df):,} rows across {df['match_id'].nunique():,} matches "
                  f"({df['date'].min().date()} .. {df['date'].max().date()}).\n")

    lines.append("## % missing per feature (overall)\n")
    lines.append(missing_overall[missing_overall["pct_missing"] > 0].to_markdown())
    lines.append("")

    lines.append("## Rows and matches per season (by tier pairing)\n")
    lines.append(rows_matches_per_season.to_markdown())
    n_matches_total = df["match_id"].nunique()
    n_matches_post_2019 = rows_matches_per_season.loc[rows_matches_per_season["post_2019"] == True, "n_matches"].sum()
    n_matches_post_2019_associate = rows_matches_per_season.loc[
        rows_matches_per_season["post_2019"] == True,
        [c for c in rows_matches_per_season.columns if "associate" in c],
    ].sum().sum()
    lines.append(
        f"\nPost-2019 matches: {n_matches_post_2019:,}/{n_matches_total:,} "
        f"({n_matches_post_2019 / n_matches_total * 100:.1f}%); of those, "
        f"{n_matches_post_2019_associate:,.0f} involve at least one associate side "
        f"(tier_pairing containing 'associate').\n"
    )

    lines.append("## Venues / teams with < 5 point-in-time prior matches\n")
    for key, value in low_sample.items():
        lines.append(f"- **{key}**: {value}")
    lines.append("")

    lines.append("## Player-rating coverage\n")
    lines.append("- Version (a), T20I-only history — computed:")
    for key, value in player_coverage.items():
        if key != "pct_rows_all_t20_version_available":
            lines.append(f"  - {key}: {value}%")
    lines.append(
        "- Version (b), T20I + franchise leagues — **not computable**: "
        "`data/t20s_json` is international-only (see docs/feature_audit.md); "
        "coverage is 0% until franchise-league Cricsheet data is added."
    )
    lines.append("")

    lines.append("## Correlation with target, at over 6 / 10 / 15\n")
    lines.append(correlations.to_markdown())
    lines.append(
        "\n`NaN` rows (`legal_balls_bowled`, `balls_remaining`, `is_reduced_overs`, "
        "`is_curtailed_first_innings`) are columns with zero variance at a fixed "
        "over-checkpoint in the default (non-flagged) table — correlation is undefined, not missing data.\n"
    )

    lines.append("## 1st-innings score distribution by tier pairing (one row per match)\n")
    lines.append(score_by_tier.to_markdown())
    lines.append("")

    OUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
