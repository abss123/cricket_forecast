"""Checkpoint feature table for the 1st-innings forecasting experiment (q5).

One row per (match, checkpoint) for checkpoints ``ball1`` (before any ball is
bowled), ``over6``, ``over10``, ``over15`` (immediately after that over
completes). A match contributes a row for a checkpoint only if its 1st
innings actually reached that many legal balls — an innings that goes all
out in 12.3 overs has no ``over15`` row.

Reuses ``compute_features`` (src/features/state.py) and the exact scope
filtering + date-grouped point-in-time replay discipline of
``scripts/build_training_table.py`` (rain-reduced/curtailed 1st innings are
excluded, same as that script's default) — see ``tests/test_state_leakage.py``
for why the replay ordering matters. Unlike the ball-by-ball training table,
this only calls ``compute_features`` up to 4 times per match (not once per
legal ball), since only 4 points in time are needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.build_training_table import (
    DATA_DIR,
    DOCS_DIR,
    SCOPE_FILTERS,
    _flatten,
    _load_match,
    _non_super_over_innings,
    _player_match_lines,
    build_match_info,
    load_scope,
)
from src.eda.unit_summary import apply_filters, build_match_table
from src.features.history import PointInTimeHistory
from src.features.state import compute_features, is_dismissal_wicket, is_legal_delivery
from src.features.venue_country import build_venue_country_mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "checkpoint_table.parquet"

# checkpoint name -> number of legal balls bowled at that point ("ball1" = before any ball).
CHECKPOINT_BALLS: dict[str, int] = {"ball1": 0, "over6": 36, "over10": 60, "over15": 90}
CHECKPOINT_ORDER: tuple[str, ...] = tuple(CHECKPOINT_BALLS.keys())


def _prefix_for_legal_count(deliveries: list[dict[str, Any]], n: int) -> list[dict[str, Any]] | None:
    """The delivery prefix ending exactly when the n-th legal ball was bowled.

    Returns ``[]`` for n == 0 (before ball 1), or ``None`` if the innings
    never reached n legal balls (e.g. all out early).
    """
    if n == 0:
        return []
    legal_seen = 0
    for i, d in enumerate(deliveries):
        if is_legal_delivery(d):
            legal_seen += 1
            if legal_seen == n:
                return deliveries[: i + 1]
    return None


def first_innings_exclusion(match: dict[str, Any]) -> str | None:
    """``"reduced_overs"``, ``"curtailed"``, or None if the 1st innings is usable."""
    info = match["info"]
    deliveries = _flatten(_non_super_over_innings(match)[0])
    scheduled_overs = info.get("overs", 20)
    balls_per_over = info.get("balls_per_over", 6)
    max_legal_balls = int(round(scheduled_overs * balls_per_over))
    legal_count = sum(1 for d in deliveries if is_legal_delivery(d))
    wickets_lost = sum(1 for d in deliveries for w in d.get("wickets", []) or [] if is_dismissal_wicket(w))
    if scheduled_overs != 20:
        return "reduced_overs"
    if legal_count < max_legal_balls and wickets_lost < 10:
        return "curtailed"
    return None


def replay_checkpoints(
    scope: pd.DataFrame,
    country_by_venue_raw: dict[str, str | None],
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """Replay ``scope`` in date-grouped chronological order, emitting checkpoint rows.

    Mirrors ``scripts.build_training_table.replay_matches``'s replay loop and
    history-update ordering exactly (so point-in-time correctness is
    identical), but only featurizes each match at the checkpoints in
    ``CHECKPOINT_BALLS`` instead of every legal ball.
    """
    history = PointInTimeHistory()
    rows: list[dict[str, Any]] = []

    for _, day_group in scope.groupby(scope["date"].dt.date, sort=True):
        pending_updates: list[dict[str, Any]] = []

        for _, meta_row in day_group.iterrows():
            match = _load_match(data_dir, meta_row["match_id"])
            info = match["info"]
            non_super = _non_super_over_innings(match)
            first_innings = non_super[0]
            deliveries = _flatten(first_innings)

            venue_country = country_by_venue_raw.get(info.get("venue"))
            match_info = build_match_info(match, meta_row, venue_country)

            if first_innings_exclusion(match) is None:
                for checkpoint_name, n_balls in CHECKPOINT_BALLS.items():
                    prefix = _prefix_for_legal_count(deliveries, n_balls)
                    if prefix is None:
                        continue
                    feats = compute_features(prefix, match_info, history)
                    feats["checkpoint"] = checkpoint_name
                    feats["target_final_score"] = meta_row["first_innings_runs"]
                    rows.append(feats)

            registry = (info.get("registry", {}) or {}).get("people", {}) or {}
            batting_lines, bowling_lines = _player_match_lines(non_super, registry)
            outcome_type, winner = meta_row["outcome_type"], meta_row["winner"]
            pending_updates.append(
                {
                    "batting_team": match_info["batting_team"],
                    "bowling_team": match_info["bowling_team"],
                    "first_innings_runs": meta_row["first_innings_runs"],
                    "venue_clean": match_info["venue_clean"],
                    "winner": winner,
                    "is_tie": outcome_type == "tie",
                    "elo_update_eligible": outcome_type in {"chase_won", "defend_won", "tie"},
                    "batting_lines": batting_lines,
                    "bowling_lines": bowling_lines,
                }
            )

        for update in pending_updates:
            history.apply_match_result(**update)

    return pd.DataFrame(rows)


def build_checkpoint_table(
    data_dir: Path = DATA_DIR,
    out_path: Path = OUT_PATH,
    force: bool = False,
) -> pd.DataFrame:
    """Load the cached checkpoint table, building and caching it if missing."""
    if out_path.exists() and not force:
        return pd.read_parquet(out_path)

    scope = load_scope(data_dir=data_dir)
    venue_country_table = build_venue_country_mapping(scope)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    country_by_venue_raw = dict(zip(venue_country_table["venue_raw"], venue_country_table["country"]))

    df = replay_checkpoints(scope, country_by_venue_raw, data_dir=data_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df


# --------------------------------------------------------------------------
# Frozen train/val/test checkpoint (data/checkpoints/first_innings/)
# --------------------------------------------------------------------------

SPLIT_DIR = PROJECT_ROOT / "data" / "checkpoints" / "first_innings"

# Inclusive match-date ranges; None = open-ended.
SPLIT_RANGES: dict[str, tuple[str | None, str | None]] = {
    "train": (None, "2022-12-31"),
    "val": ("2023-01-01", "2023-12-31"),
    "test": ("2024-01-01", None),
}


def in_split_range(dates: pd.Series, split: str) -> pd.Series:
    start, end = SPLIT_RANGES[split]
    mask = pd.Series(True, index=dates.index)
    if start is not None:
        mask &= dates >= pd.Timestamp(start)
    if end is not None:
        mask &= dates <= pd.Timestamp(end)
    return mask


def _cricsheet_data_date(data_dir: Path) -> str:
    """Latest match date listed in the Cricsheet README shipped with the data."""
    readme = (data_dir / "README.txt").read_text(encoding="utf-8")
    return max(line[:10] for line in readme.splitlines() if line[:4].isdigit() and line[4:5] == "-")


def write_split_checkpoint(data_dir: Path = DATA_DIR, out_dir: Path = SPLIT_DIR) -> dict[str, pd.DataFrame]:
    """Build the checkpoint table, split it by match date, and write parquet + manifest.json."""
    in_scope = apply_filters(build_match_table(data_dir), SCOPE_FILTERS)
    scope = load_scope(data_dir=data_dir)

    # Drop reasons, first match wins. No-results stay in the replay (so point-in-time
    # history is unchanged) and only lose their rows afterwards.
    dropped = {"no_first_innings": len(in_scope) - len(scope), "no_result": 0, "reduced_overs": 0, "curtailed": 0}
    no_result_ids = set()
    for _, meta_row in scope.iterrows():
        exclusion = first_innings_exclusion(_load_match(data_dir, meta_row["match_id"]))
        if meta_row["outcome_type"] == "no_result":
            no_result_ids.add(meta_row["match_id"])
            dropped["no_result"] += 1
        elif exclusion is not None:
            dropped[exclusion] += 1

    venue_country_table = build_venue_country_mapping(scope)
    country_by_venue_raw = dict(zip(venue_country_table["venue_raw"], venue_country_table["country"]))
    df = replay_checkpoints(scope, country_by_venue_raw, data_dir=data_dir)
    feature_columns = [c for c in df.columns if c not in {"match_id", "date", "checkpoint", "target_final_score"}]

    df = df[~df["match_id"].isin(no_result_ids)].rename(columns={"target_final_score": "final_total"})
    df["remaining_runs"] = df["final_total"] - df["score"]
    lead = ["match_id", "date", "checkpoint", "batting_team", "bowling_team", "venue", "score", "wickets"]
    df = df[lead + [c for c in df.columns if c not in lead]].reset_index(drop=True)

    print(f"male T20I matches: {len(in_scope):,}; kept {df['match_id'].nunique():,}; dropped:")
    for reason, n in dropped.items():
        print(f"  {reason:<17} {n:>5}")

    out_dir.mkdir(parents=True, exist_ok=True)
    splits = {name: df[in_split_range(df["date"], name)].reset_index(drop=True) for name in SPLIT_RANGES}
    manifest: dict[str, Any] = {
        "cricsheet_data_date": _cricsheet_data_date(data_dir),
        "splits": {},
        "feature_columns": feature_columns,
    }
    for name, split_df in splits.items():
        split_df.to_parquet(out_dir / f"{name}.parquet", index=False)
        start, end = SPLIT_RANGES[name]
        manifest["splits"][name] = {
            "date_range": {"start": start, "end": end},
            "observed_dates": {"min": str(split_df["date"].min().date()), "max": str(split_df["date"].max().date())},
            "n_matches": int(split_df["match_id"].nunique()),
            "rows_per_checkpoint": {cp: int((split_df["checkpoint"] == cp).sum()) for cp in CHECKPOINT_ORDER},
        }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return splits


def load_splits(
    checkpoint: str | None = None, split_dir: Path = SPLIT_DIR
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(train, val, test) from ``data/checkpoints/first_innings``, optionally one checkpoint only."""
    out = []
    for name in SPLIT_RANGES:
        df = pd.read_parquet(split_dir / f"{name}.parquet")
        if checkpoint is not None:
            df = df[df["checkpoint"] == checkpoint].reset_index(drop=True)
        out.append(df)
    return tuple(out)
