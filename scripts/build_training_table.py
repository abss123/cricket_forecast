"""Build the ball-by-ball 1st-innings training table (Step 3 of the forecasting pipeline).

One row per legal ball of the 1st innings of every male T20I match in
``data/t20s_json``, target = the innings' final total. Matches are replayed
in chronological, date-grouped order so that ``compute_features`` never
sees a future match's outcome (see ``tests/test_state_leakage.py``).

Usage:
    python scripts/build_training_table.py [--include-flagged] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.eda.unit_summary import apply_filters, build_match_table
from src.features.history import PointInTimeHistory
from src.features.state import compute_features, is_dismissal_wicket, is_legal_delivery
from src.features.venue_country import build_venue_country_mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "t20s_json"
DOCS_DIR = PROJECT_ROOT / "docs"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "train_1st_innings.parquet"

SCOPE_FILTERS = {"match_type": "T20", "gender": "male", "team_type": "international"}

# Credited to the bowler per standard scoring convention (run outs, retired,
# obstructing the field, timed out, and handled-the-ball are not).
BOWLER_CREDITED_DISMISSALS = {"bowled", "caught", "lbw", "stumped", "caught and bowled", "hit wicket"}


def _load_match(data_dir: Path, match_id: str) -> dict[str, Any]:
    with (data_dir / f"{match_id}.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def _flatten(innings: dict[str, Any]) -> list[dict[str, Any]]:
    return [d for over in innings.get("overs", []) or [] for d in over.get("deliveries", []) or []]


def _non_super_over_innings(match: dict[str, Any]) -> list[dict[str, Any]]:
    return [i for i in match.get("innings", []) or [] if not i.get("super_over")]


def _player_match_lines(
    non_super_innings: list[dict[str, Any]], registry: dict[str, str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Career-store update payload for every player's batting/bowling in this match.

    Built from every non-super-over innings (not just the 1st), so a
    player's point-in-time career stats reflect their whole match history —
    training rows still only ever come from 1st innings (see ``main``).
    """
    batting: dict[str, dict[str, Any]] = {}
    bowling: dict[str, dict[str, Any]] = {}

    for innings in non_super_innings:
        dismissed_names: set[str] = set()
        for delivery in _flatten(innings):
            for wicket in delivery.get("wickets", []) or []:
                if is_dismissal_wicket(wicket) and wicket.get("player_out"):
                    dismissed_names.add(wicket["player_out"])

        for delivery in _flatten(innings):
            extras = delivery.get("extras") or {}
            runs = delivery.get("runs") or {}
            total = runs.get("total", 0) or 0
            batter_runs = runs.get("batter", 0) or 0
            legal = is_legal_delivery(delivery)

            batter_name = delivery.get("batter")
            if batter_name:
                row = batting.setdefault(batter_name, {"runs": 0, "balls": 0, "dismissed": False})
                if legal:
                    row["runs"] += batter_runs
                    row["balls"] += 1

            bowler_name = delivery.get("bowler")
            if bowler_name:
                row = bowling.setdefault(bowler_name, {"balls": 0, "runs_conceded": 0, "wickets": 0})
                if legal:
                    row["balls"] += 1
                bowler_runs = total - (extras.get("byes", 0) or 0) - (extras.get("legbyes", 0) or 0)
                row["runs_conceded"] += bowler_runs
                for wicket in delivery.get("wickets", []) or []:
                    if is_dismissal_wicket(wicket) and wicket.get("kind") in BOWLER_CREDITED_DISMISSALS:
                        row["wickets"] += 1

        for name in dismissed_names:
            if name in batting:
                batting[name]["dismissed"] = True

    batting_lines = [
        {"player_id": registry.get(name), "runs": v["runs"], "balls": v["balls"], "dismissed": v["dismissed"]}
        for name, v in batting.items()
    ]
    bowling_lines = [
        {"player_id": registry.get(name), "balls": v["balls"], "runs_conceded": v["runs_conceded"], "wickets": v["wickets"]}
        for name, v in bowling.items()
    ]
    return batting_lines, bowling_lines


def build_match_info(
    match: dict[str, Any],
    meta_row: pd.Series,
    venue_country: str | None,
) -> dict[str, Any]:
    info = match["info"]
    registry = (info.get("registry", {}) or {}).get("people", {}) or {}
    batting_team, bowling_team = meta_row["team_batting_first"], meta_row["team_batting_second"]
    players = info.get("players", {}) or {}
    return {
        "match_id": meta_row["match_id"],
        "date": meta_row["date"],
        "season": meta_row["season"],
        "venue": info.get("venue"),
        "venue_clean": meta_row["venue_clean"],
        "venue_country": venue_country,
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "toss_winner": info.get("toss", {}).get("winner"),
        "toss_decision": info.get("toss", {}).get("decision"),
        "scheduled_overs": info.get("overs", 20),
        "balls_per_over": info.get("balls_per_over", 6),
        "batting_players": players.get(batting_team, []) or [],
        "bowling_players": players.get(bowling_team, []) or [],
        "registry": registry,
    }


def load_scope(data_dir: Path = DATA_DIR, filters: dict[str, Any] = SCOPE_FILTERS) -> pd.DataFrame:
    """The match-level metadata table this pipeline runs over: filtered, has a
    usable 1st innings, sorted chronologically (date, then match_id as a
    determinism tie-break within a date)."""
    matches = build_match_table(data_dir)
    scope = apply_filters(matches, filters)
    scope = scope[scope["first_innings_runs"].notna()].copy()
    return scope.sort_values(["date", "match_id"]).reset_index(drop=True)


def replay_matches(
    scope: pd.DataFrame,
    country_by_venue_raw: dict[str, str | None],
    include_flagged: bool = True,
    data_dir: Path = DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay ``scope`` in date-grouped chronological order.

    Returns ``(rows, match_flags)``: ``rows`` has one row per legal
    1st-innings ball (only for flagged matches too if ``include_flagged``),
    ``match_flags`` has one row per match in ``scope`` with its
    ``is_reduced_overs``/``is_curtailed_first_innings`` status regardless of
    ``include_flagged`` (for reporting).

    ``compute_features`` for any match in a date group only ever sees
    history from *strictly earlier* dates — see
    ``tests/test_state_leakage.py``, which relies on exactly this property.
    """
    history = PointInTimeHistory()
    rows: list[dict[str, Any]] = []
    match_flags: list[dict[str, Any]] = []

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
            match_flags.append(
                {
                    "match_id": meta_row["match_id"],
                    "is_reduced_overs": is_reduced_overs,
                    "is_curtailed_first_innings": is_curtailed,
                }
            )

            venue_country = country_by_venue_raw.get(info.get("venue"))
            match_info = build_match_info(match, meta_row, venue_country)

            if (not is_reduced_overs and not is_curtailed) or include_flagged:
                running: list[dict[str, Any]] = []
                for delivery in deliveries:
                    running.append(delivery)
                    if is_legal_delivery(delivery):
                        feats = compute_features(running, match_info, history)
                        feats["target_final_score"] = meta_row["first_innings_runs"]
                        feats["is_reduced_overs"] = is_reduced_overs
                        feats["is_curtailed_first_innings"] = is_curtailed
                        feats["dls_affected"] = bool(meta_row["dls_affected"])
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

    return pd.DataFrame(rows), pd.DataFrame(match_flags)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-flagged", action="store_true", help="Also emit rows for reduced-overs/curtailed 1st innings")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    scope = load_scope()

    venue_country_table = build_venue_country_mapping(scope)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    venue_country_table.to_csv(DOCS_DIR / "venue_country_mapping.csv", index=False)
    country_by_venue_raw = dict(zip(venue_country_table["venue_raw"], venue_country_table["country"]))

    df, match_flags = replay_matches(scope, country_by_venue_raw, include_flagged=args.include_flagged)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    n_reduced_overs = int(match_flags["is_reduced_overs"].sum())
    n_curtailed = int(match_flags["is_curtailed_first_innings"].sum())
    print(f"matches processed (male T20I, usable 1st innings): {len(match_flags):,}")
    print(f"  reduced-overs matches: {n_reduced_overs:,}; curtailed 1st innings: {n_curtailed:,}")
    print(f"training rows written: {len(df):,} -> {args.out}")
    if len(df):
        print(f"date range: {df['date'].min()} .. {df['date'].max()}")


if __name__ == "__main__":
    main()
