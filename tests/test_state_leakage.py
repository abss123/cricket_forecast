"""Leakage regression test for the point-in-time feature pipeline.

The requirement: features for a match on date D must not change if every
match on/after D is deleted from history.

Two layers are tested:

- ``test_synthetic_*``: drives ``PointInTimeHistory``/``compute_features``
  directly with small hand-built matches (no JSON files, no real data) —
  fast, deterministic, and isolated to ``src/features``. This is the
  primary check, including the trickier same-day tie-break case (two
  matches sharing a date must not see each other's result).
- ``test_real_data_leakage_subset``: runs ``scripts.build_training_table``'s
  actual date-grouped orchestration over a small chronological slice of the
  real Cricsheet data, to catch integration bugs the synthetic test can't
  see (e.g. a mistake in how ``build_training_table.py`` wires up updates).
  Scoped to a slice rather than the full ~3,500 matches to keep the test
  fast (a full replay takes ~90s).
"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.build_training_table import load_scope, replay_matches
from src.features.history import PointInTimeHistory
from src.features.state import compute_features
from src.features.venue_country import build_venue_country_mapping

REAL_DATA_SUBSET_SIZE = 400


# --------------------------------------------------------------------------
# Synthetic fixtures
# --------------------------------------------------------------------------


def _delivery(batter: str, non_striker: str, bowler: str, runs: int, wicket: bool = False) -> dict:
    d = {
        "batter": batter,
        "non_striker": non_striker,
        "bowler": bowler,
        "runs": {"batter": runs, "extras": 0, "total": runs},
    }
    if wicket:
        d["wickets"] = [{"player_out": batter, "kind": "bowled"}]
    return d


def _innings(deliveries: list[dict]) -> list[dict]:
    return deliveries


def _match_info(match_id: str, date: str, batting_team: str, bowling_team: str, batting_players: list[str], bowling_players: list[str]) -> dict:
    registry = {name: f"id_{name}" for name in batting_players + bowling_players}
    return {
        "match_id": match_id,
        "date": date,
        "season": "2024",
        "venue": "Test Ground",
        "venue_clean": "Test Ground",
        "venue_country": None,
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "toss_winner": batting_team,
        "toss_decision": "bat",
        "scheduled_overs": 20,
        "balls_per_over": 6,
        "batting_players": batting_players,
        "bowling_players": bowling_players,
        "registry": registry,
    }


def _replay_synthetic(match_specs: list[tuple[str, dict, list[dict]]]) -> dict[str, list[dict]]:
    """match_specs: list of (date, match_info, deliveries), NOT required to be
    sorted. Replays date-grouped exactly like ``build_training_table.replay_matches``,
    returning {match_id: [feature_row, ...]}."""
    by_date: dict[str, list[tuple[dict, list[dict]]]] = {}
    for date, info, deliveries in match_specs:
        by_date.setdefault(date, []).append((info, deliveries))

    history = PointInTimeHistory()
    results: dict[str, list[dict]] = {}

    for date in sorted(by_date):
        pending = []
        for info, deliveries in by_date[date]:
            running: list[dict] = []
            rows = []
            for delivery in deliveries:
                running.append(delivery)
                rows.append(compute_features(list(running), info, history))
            results[info["match_id"]] = rows
            pending.append(
                {
                    "batting_team": info["batting_team"],
                    "bowling_team": info["bowling_team"],
                    "first_innings_runs": sum(d["runs"]["total"] for d in deliveries),
                    "venue_clean": info["venue_clean"],
                    "winner": info["batting_team"],
                    "is_tie": False,
                    "elo_update_eligible": True,
                    "batting_lines": [],
                    "bowling_lines": [],
                }
            )
        for update in pending:
            history.apply_match_result(**update)

    return results


def test_synthetic_deleting_future_matches_does_not_change_earlier_features() -> None:
    day1 = _match_info("m1", "2024-01-01", "TeamA", "TeamB", ["A1", "A2"], ["B1", "B2"])
    day1_deliveries = [_delivery("A1", "A2", "B1", 4), _delivery("A1", "A2", "B1", 1)]

    day3 = _match_info("m3", "2024-01-03", "TeamA", "TeamD", ["A1", "A2"], ["D1", "D2"])
    day3_deliveries = [_delivery("A1", "A2", "D1", 6), _delivery("A1", "A2", "D1", 0)]

    full = _replay_synthetic([("2024-01-01", day1, day1_deliveries), ("2024-01-03", day3, day3_deliveries)])
    truncated = _replay_synthetic([("2024-01-01", day1, day1_deliveries)])

    assert full["m1"] == truncated["m1"]


def test_synthetic_same_day_matches_do_not_see_each_other() -> None:
    day1 = _match_info("m1", "2024-01-01", "TeamA", "TeamB", ["A1", "A2"], ["B1", "B2"])
    day1_deliveries = [_delivery("A1", "A2", "B1", 4)]

    same_day_a = _match_info("m2a", "2024-01-02", "TeamA", "TeamC", ["A1", "A2"], ["C1", "C2"])
    same_day_a_deliveries = [_delivery("A1", "A2", "C1", 1)]

    same_day_b = _match_info("m2b", "2024-01-02", "TeamB", "TeamD", ["B1", "B2"], ["D1", "D2"])
    same_day_b_deliveries = [_delivery("B1", "B2", "D1", 2)]

    with_sibling = _replay_synthetic(
        [
            ("2024-01-01", day1, day1_deliveries),
            ("2024-01-02", same_day_a, same_day_a_deliveries),
            ("2024-01-02", same_day_b, same_day_b_deliveries),
        ]
    )
    without_sibling = _replay_synthetic(
        [("2024-01-01", day1, day1_deliveries), ("2024-01-02", same_day_a, same_day_a_deliveries)]
    )

    assert with_sibling["m2a"] == without_sibling["m2a"]


# --------------------------------------------------------------------------
# Real-data integration check (small chronological subset for speed)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_scope() -> pd.DataFrame:
    return load_scope().head(REAL_DATA_SUBSET_SIZE)


@pytest.fixture(scope="module")
def real_country_by_venue_raw(real_scope: pd.DataFrame) -> dict[str, str | None]:
    mapping = build_venue_country_mapping(real_scope)
    return dict(zip(mapping["venue_raw"], mapping["country"]))


def test_real_data_leakage_subset(real_scope: pd.DataFrame, real_country_by_venue_raw: dict[str, str | None]) -> None:
    target_match_id = real_scope.iloc[len(real_scope) // 2]["match_id"]
    target_date = real_scope.loc[real_scope["match_id"] == target_match_id, "date"].iloc[0]

    full_rows, _ = replay_matches(real_scope, real_country_by_venue_raw, include_flagged=True)
    truncated_scope = real_scope[real_scope["date"] <= target_date].reset_index(drop=True)
    truncated_rows, _ = replay_matches(truncated_scope, real_country_by_venue_raw, include_flagged=True)

    full_target = full_rows[full_rows["match_id"] == target_match_id].sort_values("legal_balls_bowled").reset_index(drop=True)
    truncated_target = truncated_rows[truncated_rows["match_id"] == target_match_id].sort_values("legal_balls_bowled").reset_index(drop=True)

    assert len(full_target) > 0
    pd.testing.assert_frame_equal(full_target, truncated_target)
