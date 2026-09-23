"""Tests for src.eda.unit_summary.

Builds small synthetic Cricsheet-shaped match JSON files (rather than reading
the real data directory) so each scenario in docs/schema_notes.md's edge-case
list can be exercised in isolation: a plain chase win, a plain defend win, a
tie resolved by super over, a no-result, and a DLS-revised chase.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.eda.unit_summary import (
    apply_filters,
    build_match_table,
    build_player_directory,
    build_venue_mapping,
    classify_outcome,
    summarize_units,
    validate_outcome_counts,
)


# --------------------------------------------------------------------------
# Fixture builders
# --------------------------------------------------------------------------


def make_delivery(total: int) -> dict[str, Any]:
    return {
        "batter": "X",
        "bowler": "Y",
        "non_striker": "Z",
        "runs": {"batter": total, "extras": 0, "total": total},
    }


def make_innings(
    team: str,
    total_runs: int,
    target: dict[str, Any] | None = None,
    super_over: bool = False,
) -> dict[str, Any]:
    innings: dict[str, Any] = {
        "team": team,
        "overs": [{"over": 0, "deliveries": [make_delivery(total_runs)]}],
    }
    if target is not None:
        innings["target"] = target
    if super_over:
        innings["super_over"] = True
    return innings


def make_match(
    match_id: str,
    team1: str,
    team2: str,
    innings: list[dict[str, Any]],
    outcome: dict[str, Any],
    gender: str = "male",
    match_type: str = "T20",
    team_type: str = "international",
    date: str = "2023-01-01",
    season: str = "2023",
    venue: str = "Test Ground",
    city: str = "Test City",
    team1_players: list[str] | None = None,
    team2_players: list[str] | None = None,
    registry_extra: dict[str, str] | None = None,
) -> dict[str, Any]:
    players = {
        team1: team1_players or [f"{team1} P1", f"{team1} P2"],
        team2: team2_players or [f"{team2} P1", f"{team2} P2"],
    }
    registry: dict[str, str] = {}
    for team, names in players.items():
        for i, name in enumerate(names):
            registry.setdefault(name, f"{team[:2].lower()}{i:03d}")
    if registry_extra:
        registry.update(registry_extra)

    return {
        "meta": {"data_version": "1.2.0", "created": date, "revision": 1},
        "info": {
            "teams": [team1, team2],
            "venue": venue,
            "city": city,
            "dates": [date],
            "season": season,
            "match_type": match_type,
            "match_type_number": 1,
            "gender": gender,
            "team_type": team_type,
            "overs": 20,
            "balls_per_over": 6,
            "toss": {"winner": team1, "decision": "bat"},
            "outcome": outcome,
            "players": players,
            "registry": {"people": registry},
            "officials": {},
        },
        "innings": innings,
    }


def write_match(data_dir: Path, match_id: str, match: dict[str, Any]) -> None:
    (data_dir / f"{match_id}.json").write_text(json.dumps(match))


@pytest.fixture()
def synthetic_data_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "matches"
    data_dir.mkdir()

    # m1: plain chase win — B (batting second) beats A (batting first).
    write_match(
        data_dir,
        "m1",
        make_match(
            "m1",
            "TeamA",
            "TeamB",
            innings=[
                make_innings("TeamA", 150),
                make_innings("TeamB", 151, target={"runs": 151, "overs": 20}),
            ],
            outcome={"winner": "TeamB", "by": {"wickets": 5}},
            registry_extra={"Shared Player": "shared01"},
            team1_players=["TeamA P1", "Shared Player"],
        ),
    )

    # m2: plain defend win — A (batting first) beats C (batting second).
    write_match(
        data_dir,
        "m2",
        make_match(
            "m2",
            "TeamA",
            "TeamC",
            innings=[
                make_innings("TeamA", 180),
                make_innings("TeamC", 150, target={"runs": 181, "overs": 20}),
            ],
            outcome={"winner": "TeamA", "by": {"runs": 30}},
        ),
    )

    # m3: tie, resolved by a super over (2 main innings tied at 150, plus
    # 2 super-over innings that must NOT count toward first/second_innings_runs).
    write_match(
        data_dir,
        "m3",
        make_match(
            "m3",
            "TeamA",
            "TeamB",
            innings=[
                make_innings("TeamA", 150),
                make_innings("TeamB", 150, target={"runs": 151, "overs": 20}),
                make_innings("TeamA", 10, super_over=True),
                make_innings("TeamB", 15, super_over=True),
            ],
            outcome={"result": "tie", "eliminator": "TeamB"},
        ),
    )

    # m4: no result — only one (partial) innings ever started.
    write_match(
        data_dir,
        "m4",
        make_match(
            "m4",
            "TeamA",
            "TeamC",
            innings=[make_innings("TeamA", 45)],
            outcome={"result": "no result"},
        ),
    )

    # m5: DLS-revised chase win — target != first_innings_runs + 1, method "D/L".
    write_match(
        data_dir,
        "m5",
        make_match(
            "m5",
            "TeamB",
            "TeamD",
            innings=[
                make_innings("TeamB", 180),
                make_innings("TeamD", 142, target={"runs": 141, "overs": 15}),
            ],
            outcome={"winner": "TeamD", "by": {"wickets": 3}, "method": "D/L"},
        ),
    )

    # m6: Shared Player turns out for TeamE here (batting second), having
    # played for TeamA in m1 (batting first) — tests player aggregation
    # across two different team affiliations.
    write_match(
        data_dir,
        "m6",
        make_match(
            "m6",
            "TeamF",
            "TeamE",
            innings=[
                make_innings("TeamF", 160),
                make_innings("TeamE", 161, target={"runs": 161, "overs": 20}),
            ],
            outcome={"winner": "TeamE", "by": {"wickets": 4}},
            registry_extra={"Shared Player": "shared01"},
            team2_players=["TeamE P1", "Shared Player"],
        ),
    )

    return data_dir


@pytest.fixture()
def match_table(synthetic_data_dir: Path) -> pd.DataFrame:
    cache_path = synthetic_data_dir.parent / "cache" / "match_table.parquet"
    return build_match_table(synthetic_data_dir, cache_path=cache_path)


# --------------------------------------------------------------------------
# classify_outcome
# --------------------------------------------------------------------------


def test_classify_outcome_chase_won() -> None:
    outcome_type, note = classify_outcome({"winner": "B", "by": {"wickets": 5}}, "A", "B")
    assert outcome_type == "chase_won"
    assert note is None


def test_classify_outcome_defend_won() -> None:
    outcome_type, note = classify_outcome({"winner": "A", "by": {"runs": 30}}, "A", "B")
    assert outcome_type == "defend_won"
    assert note is None


def test_classify_outcome_tie_via_eliminator() -> None:
    outcome_type, note = classify_outcome({"result": "tie", "eliminator": "B"}, "A", "B")
    assert outcome_type == "tie"


def test_classify_outcome_no_result() -> None:
    outcome_type, _ = classify_outcome({"result": "no result"}, "A", None)
    assert outcome_type == "no_result"


def test_classify_outcome_draw_and_innings_win() -> None:
    assert classify_outcome({"result": "draw"}, "A", "B")[0] == "draw"
    assert classify_outcome({"winner": "A", "by": {"innings": 1, "runs": 10}}, "A", "B")[0] == "innings_win"


def test_classify_outcome_unrecognized_shape_maps_to_other_with_note() -> None:
    outcome_type, note = classify_outcome({"winner": "Ghana", "method": "Awarded"}, "Ghana", "Rwanda")
    assert outcome_type == "other"
    assert note is not None


# --------------------------------------------------------------------------
# build_match_table
# --------------------------------------------------------------------------


def test_match_table_has_expected_columns(match_table: pd.DataFrame) -> None:
    expected = {
        "match_id", "date", "season", "match_type", "gender", "team_type", "event",
        "venue", "venue_clean", "city", "team1", "team2", "team_batting_first",
        "team_batting_second", "toss_winner", "toss_decision", "first_innings_runs",
        "target_runs", "target_overs", "second_innings_runs", "winner", "outcome_type",
        "method", "dls_affected", "has_super_over", "decided_by_super_over",
        "decided_by_bowl_out", "players_team1", "players_team2",
    }
    assert expected.issubset(set(match_table.columns))
    assert len(match_table) == 6


def test_chase_win_row(match_table: pd.DataFrame) -> None:
    row = match_table.set_index("match_id").loc["m1"]
    assert row["outcome_type"] == "chase_won"
    assert row["team_batting_first"] == "TeamA"
    assert row["team_batting_second"] == "TeamB"
    assert row["dls_affected"] == False  # noqa: E712


def test_defend_win_row(match_table: pd.DataFrame) -> None:
    row = match_table.set_index("match_id").loc["m2"]
    assert row["outcome_type"] == "defend_won"


def test_tie_with_super_over_row(match_table: pd.DataFrame) -> None:
    row = match_table.set_index("match_id").loc["m3"]
    assert row["outcome_type"] == "tie"
    assert row["has_super_over"] == True  # noqa: E712
    assert row["decided_by_super_over"] == True  # noqa: E712
    # Super-over innings must not leak into the main-innings run totals.
    assert row["first_innings_runs"] == 150
    assert row["second_innings_runs"] == 150


def test_no_result_row(match_table: pd.DataFrame) -> None:
    row = match_table.set_index("match_id").loc["m4"]
    assert row["outcome_type"] == "no_result"
    assert row["team_batting_second"] is None
    assert pd.isna(row["second_innings_runs"])


def test_dls_match_row(match_table: pd.DataFrame) -> None:
    row = match_table.set_index("match_id").loc["m5"]
    assert row["outcome_type"] == "chase_won"
    assert row["dls_affected"] == True  # noqa: E712
    assert row["target_runs"] != row["first_innings_runs"] + 1


def test_match_table_is_cached(synthetic_data_dir: Path) -> None:
    cache_path = synthetic_data_dir.parent / "cache2" / "match_table.parquet"
    assert not cache_path.exists()

    first = build_match_table(synthetic_data_dir, cache_path=cache_path)
    assert cache_path.exists()

    # Corrupt the source data to prove the second call reads the cache, not JSON.
    (synthetic_data_dir / "m1.json").write_text("not valid json")
    second = build_match_table(synthetic_data_dir, cache_path=cache_path)
    pd.testing.assert_frame_equal(first, second)

    # force_rebuild should re-parse (and blow up on the corrupted file).
    with pytest.raises(json.JSONDecodeError):
        build_match_table(synthetic_data_dir, cache_path=cache_path, force_rebuild=True)


# --------------------------------------------------------------------------
# build_venue_mapping
# --------------------------------------------------------------------------


def test_venue_mapping_resolves_unambiguous_bare_name() -> None:
    mapping = build_venue_mapping(["Eden Gardens", "Eden Gardens, Kolkata"])
    bare = mapping.set_index("venue_raw").loc["Eden Gardens"]
    assert bare["venue_clean"] == "Eden Gardens, Kolkata"
    assert bare["needs_review"] == False  # noqa: E712


def test_venue_mapping_flags_ambiguous_bare_name() -> None:
    mapping = build_venue_mapping(
        ["County Ground", "County Ground, Bristol", "County Ground, Derby"]
    )
    bare = mapping.set_index("venue_raw").loc["County Ground"]
    assert bare["venue_clean"] == "County Ground"  # left unresolved
    assert bare["needs_review"] == True  # noqa: E712


def test_venue_mapping_flags_distinct_grounds_sharing_a_base_name() -> None:
    mapping = build_venue_mapping(
        ["Gymkhana Club Ground, Nairobi", "Gymkhana Club Ground, Dar-es-Salaam"]
    )
    assert mapping["needs_review"].all()
    # Each qualified venue still maps to itself, not merged together.
    assert set(mapping["venue_clean"]) == {
        "Gymkhana Club Ground, Nairobi",
        "Gymkhana Club Ground, Dar-es-Salaam",
    }


# --------------------------------------------------------------------------
# apply_filters
# --------------------------------------------------------------------------


def test_apply_filters_by_gender_and_date_range(match_table: pd.DataFrame) -> None:
    df = match_table.copy()
    df.loc[df["match_id"] == "m1", "gender"] = "female"
    filtered = apply_filters(df, {"gender": "female"})
    assert set(filtered["match_id"]) == {"m1"}

    filtered = apply_filters(match_table, {"start_date": "2023-01-01", "end_date": "2023-01-01"})
    assert len(filtered) == 6  # all fixtures share this date


def test_apply_filters_exclude_super_over(match_table: pd.DataFrame) -> None:
    filtered = apply_filters(match_table, {"exclude_super_over_innings": True})
    assert "m3" not in set(filtered["match_id"])
    assert len(filtered) == 5


def test_apply_filters_rejects_unknown_key(match_table: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        apply_filters(match_table, {"nonsense_key": True})


# --------------------------------------------------------------------------
# summarize_units
# --------------------------------------------------------------------------


def test_summarize_units_venue(match_table: pd.DataFrame) -> None:
    summary, header = summarize_units(match_table, unit="venue")
    validate_outcome_counts(summary)
    assert header["n_units"] == 1  # all fixtures share "Test Ground"
    row = summary.iloc[0]
    assert row["n_matches"] == 6
    assert row["n_chase_won"] == 3  # m1, m5, m6
    assert row["n_defend_won"] == 1  # m2
    assert row["n_tie"] == 1  # m3
    assert row["n_no_result"] == 1  # m4


def test_summarize_units_team_counts_and_role_split(match_table: pd.DataFrame) -> None:
    summary, header = summarize_units(match_table, unit="team")
    validate_outcome_counts(summary)
    assert header["n_units"] == 6  # A, B, C, D, E, F

    team_a = summary.set_index("unit_id").loc["TeamA"]
    # TeamA appears in m1 (bat first), m2 (bat first), m3 (bat first), m4 (bat first).
    assert team_a["n_matches"] == 4
    assert team_a["n_matches_batting_first"] == 4
    assert team_a["n_matches_batting_second"] == 0


def test_summarize_units_player_appears_for_two_teams(match_table: pd.DataFrame) -> None:
    summary, header = summarize_units(match_table, unit="player")
    validate_outcome_counts(summary)

    shared = summary.set_index("unit_id").loc["shared01"]
    assert shared["n_matches"] == 2  # m1 (TeamA) and m6 (TeamE)
    assert shared["n_matches_batting_first"] == 1  # TeamA in m1
    assert shared["n_matches_batting_second"] == 1  # TeamE in m6


def test_summarize_units_player_names_from_directory(
    match_table: pd.DataFrame, synthetic_data_dir: Path
) -> None:
    directory = build_player_directory(
        synthetic_data_dir, cache_path=synthetic_data_dir.parent / "cache" / "players.parquet"
    )
    summary, _ = summarize_units(match_table, unit="player", player_directory=directory)
    assert summary.set_index("unit_id").loc["shared01", "unit_name"] == "Shared Player"


def test_summarize_units_invalid_unit_raises(match_table: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        summarize_units(match_table, unit="ground")  # type: ignore[arg-type]


def test_outcome_counts_sum_to_n_matches_for_all_units(match_table: pd.DataFrame) -> None:
    for unit in ("team", "venue", "player"):
        summary, _ = summarize_units(match_table, unit=unit)  # type: ignore[arg-type]
        validate_outcome_counts(summary)  # should not raise


def test_summarize_units_player_after_parquet_round_trip(synthetic_data_dir: Path) -> None:
    """Regression test: pyarrow deserializes list columns as numpy arrays,
    not Python lists, so `players_teamN or []`-style code breaks only after
    a real save/reload cycle — building fresh in-memory doesn't catch it."""
    cache_path = synthetic_data_dir.parent / "cache_roundtrip" / "match_table.parquet"
    build_match_table(synthetic_data_dir, cache_path=cache_path)  # populate cache
    reloaded = build_match_table(synthetic_data_dir, cache_path=cache_path)  # read from parquet

    summary, _ = summarize_units(reloaded, unit="player")
    validate_outcome_counts(summary)
    assert summary.set_index("unit_id").loc["shared01", "n_matches"] == 2
