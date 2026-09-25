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

from pathlib import Path
from typing import Any

import pandas as pd

from scripts.build_training_table import (
    DATA_DIR,
    DOCS_DIR,
    _flatten,
    _load_match,
    _non_super_over_innings,
    _player_match_lines,
    build_match_info,
    load_scope,
)
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

            scheduled_overs = info.get("overs", 20)
            balls_per_over = info.get("balls_per_over", 6)
            max_legal_balls = int(round(scheduled_overs * balls_per_over))
            legal_count = sum(1 for d in deliveries if is_legal_delivery(d))
            wickets_lost = sum(
                1 for d in deliveries for w in d.get("wickets", []) or [] if is_dismissal_wicket(w)
            )
            is_reduced_overs = scheduled_overs != 20
            is_curtailed = legal_count < max_legal_balls and wickets_lost < 10

            venue_country = country_by_venue_raw.get(info.get("venue"))
            match_info = build_match_info(match, meta_row, venue_country)

            if not is_reduced_overs and not is_curtailed:
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
