from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl


PROJECT_MARKERS = ("data/t20s_json", "requirements.txt", ".git")


def find_project_root(start: Path | None = None) -> Path:
    """Return the project root whether called from the root or code directory."""
    current = (start or Path.cwd()).resolve()
    candidates = [current, *current.parents]

    for candidate in candidates:
        if (candidate / "data" / "t20s_json").is_dir():
            return candidate

    for candidate in candidates:
        if any((candidate / marker).exists() for marker in PROJECT_MARKERS):
            return candidate

    raise FileNotFoundError(
        f"Could not find project root from {current}. Expected data/t20s_json nearby."
    )


def load_match_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _registry(match: dict[str, Any]) -> dict[str, str]:
    return match.get("info", {}).get("registry", {}).get("people", {}) or {}


def _player_id(match: dict[str, Any], player_name: str | None) -> str | None:
    if player_name is None:
        return None
    return _registry(match).get(player_name)


def _first_or_none(value: Any) -> Any:
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _empty_to_none(values: list[str | None]) -> list[str] | None:
    cleaned = [value for value in values if value is not None]
    return cleaned or None


def parse_match(path: Path, match: dict[str, Any]) -> dict[str, Any]:
    info = match.get("info", {}) or {}
    event = info.get("event", {}) or {}
    toss = info.get("toss", {}) or {}
    outcome = info.get("outcome", {}) or {}
    outcome_by = outcome.get("by", {}) or {}
    teams = info.get("teams", []) or []

    return {
        "match_id": path.stem,
        "start_date": _as_text(_first_or_none(info.get("dates"))),
        "season": _as_text(info.get("season")),
        "balls_per_over": _to_int(info.get("balls_per_over")),
        "scheduled_overs": _to_float(info.get("overs")),
        "match_type": info.get("match_type"),
        "match_type_number": _to_int(info.get("match_type_number")),
        "gender": info.get("gender"),
        "team_type": info.get("team_type"),
        "city": info.get("city"),
        "venue": info.get("venue"),
        "event_name": event.get("name"),
        "event_match_number": _as_text(event.get("match_number")),
        "team1": teams[0] if len(teams) > 0 else None,
        "team2": teams[1] if len(teams) > 1 else None,
        "toss_winner": toss.get("winner"),
        "toss_decision": toss.get("decision"),
        "outcome_winner": outcome.get("winner"),
        "outcome_result": outcome.get("result"),
        "outcome_method": outcome.get("method"),
        "win_by_runs": _to_int(outcome_by.get("runs")),
        "win_by_wickets": _to_int(outcome_by.get("wickets")),
        "player_of_match": _first_or_none(info.get("player_of_match")),
    }


def parse_innings(path: Path, match: dict[str, Any]) -> list[dict[str, Any]]:
    info = match.get("info", {}) or {}
    balls_per_over = _to_int(info.get("balls_per_over")) or 6
    rows: list[dict[str, Any]] = []

    for innings_number, innings in enumerate(match.get("innings", []) or [], start=1):
        deliveries = [
            delivery
            for over in innings.get("overs", []) or []
            for delivery in over.get("deliveries", []) or []
        ]
        legal_balls = sum(1 for delivery in deliveries if _is_legal_delivery(delivery))
        wickets_lost = sum(
            1
            for delivery in deliveries
            for wicket in delivery.get("wickets", []) or []
            if _is_dismissal(wicket)
        )
        total_runs = sum(
            _to_int((delivery.get("runs", {}) or {}).get("total")) or 0
            for delivery in deliveries
        )
        target = innings.get("target", {}) or {}

        rows.append(
            {
                "match_id": path.stem,
                "innings_number": innings_number,
                "batting_team": innings.get("team"),
                "target_runs": _to_int(target.get("runs")),
                "target_overs": _to_float(target.get("overs")),
                "total_runs": total_runs,
                "wickets_lost": wickets_lost,
                "total_delivery_records": len(deliveries),
                "legal_balls": legal_balls,
                "overs_bowled": legal_balls / balls_per_over if balls_per_over else None,
            }
        )

    return rows


def _is_legal_delivery(delivery: dict[str, Any]) -> bool:
    extras = delivery.get("extras", {}) or {}
    return not (extras.get("wides") is not None or extras.get("noballs") is not None)


def _is_dismissal(wicket: dict[str, Any]) -> bool:
    return wicket.get("kind") not in {"retired hurt", "retired not out"}


def parse_deliveries(path: Path, match: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for innings_number, innings in enumerate(match.get("innings", []) or [], start=1):
        delivery_sequence = 0
        for over in innings.get("overs", []) or []:
            over_number = _to_int(over.get("over"))
            for delivery in over.get("deliveries", []) or []:
                delivery_sequence += 1
                runs = delivery.get("runs", {}) or {}
                extras = delivery.get("extras", {}) or {}
                wickets = delivery.get("wickets", []) or []

                rows.append(
                    {
                        "match_id": path.stem,
                        "innings_number": innings_number,
                        "batting_team": innings.get("team"),
                        "over_number": over_number,
                        "actual_delivery": _as_text(delivery.get("actual_delivery")),
                        "delivery_sequence": delivery_sequence,
                        "batter": delivery.get("batter"),
                        "batter_id": _player_id(match, delivery.get("batter")),
                        "non_striker": delivery.get("non_striker"),
                        "non_striker_id": _player_id(match, delivery.get("non_striker")),
                        "bowler": delivery.get("bowler"),
                        "bowler_id": _player_id(match, delivery.get("bowler")),
                        "batter_runs": _to_int(runs.get("batter")) or 0,
                        "extras_runs": _to_int(runs.get("extras")) or 0,
                        "total_runs": _to_int(runs.get("total")) or 0,
                        "wides": _to_int(extras.get("wides")) or 0,
                        "noballs": _to_int(extras.get("noballs")) or 0,
                        "byes": _to_int(extras.get("byes")) or 0,
                        "legbyes": _to_int(extras.get("legbyes")) or 0,
                        "penalty": _to_int(extras.get("penalty")) or 0,
                        "is_legal_delivery": _is_legal_delivery(delivery),
                        "is_wicket": bool(wickets),
                        "wickets_on_delivery": len(wickets),
                    }
                )

    return rows


def parse_wickets(path: Path, match: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for innings_number, innings in enumerate(match.get("innings", []) or [], start=1):
        delivery_sequence = 0
        for over in innings.get("overs", []) or []:
            over_number = _to_int(over.get("over"))
            for delivery in over.get("deliveries", []) or []:
                delivery_sequence += 1
                for wicket in delivery.get("wickets", []) or []:
                    fielders = wicket.get("fielders", []) or []
                    fielder_names = [
                        fielder.get("name") if isinstance(fielder, dict) else str(fielder)
                        for fielder in fielders
                    ]
                    fielder_ids = [_player_id(match, name) for name in fielder_names]

                    rows.append(
                        {
                            "match_id": path.stem,
                            "innings_number": innings_number,
                            "batting_team": innings.get("team"),
                            "over_number": over_number,
                            "actual_delivery": _as_text(delivery.get("actual_delivery")),
                            "delivery_sequence": delivery_sequence,
                            "player_out": wicket.get("player_out"),
                            "player_out_id": _player_id(match, wicket.get("player_out")),
                            "wicket_kind": wicket.get("kind"),
                            "fielder_names": _empty_to_none(fielder_names),
                            "fielder_ids": _empty_to_none(fielder_ids),
                        }
                    )

    return rows


def parse_players(path: Path, match: dict[str, Any]) -> list[dict[str, Any]]:
    players = match.get("info", {}).get("players", {}) or {}
    registry = _registry(match)
    rows: list[dict[str, Any]] = []

    for team, player_names in players.items():
        for player_name in player_names or []:
            rows.append(
                {
                    "match_id": path.stem,
                    "team": team,
                    "player_name": player_name,
                    "player_id": registry.get(player_name),
                }
            )

    return rows


def parse_powerplays(path: Path, match: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for innings_number, innings in enumerate(match.get("innings", []) or [], start=1):
        for powerplay in innings.get("powerplays", []) or []:
            rows.append(
                {
                    "match_id": path.stem,
                    "innings_number": innings_number,
                    "batting_team": innings.get("team"),
                    "powerplay_type": powerplay.get("type"),
                    "from_ball": _to_float(powerplay.get("from")),
                    "to_ball": _to_float(powerplay.get("to")),
                }
            )

    return rows


def parse_officials(path: Path, match: dict[str, Any]) -> list[dict[str, Any]]:
    officials = match.get("info", {}).get("officials", {}) or {}
    role_names = {
        "umpires": "umpire",
        "reserve_umpires": "reserve umpire",
        "tv_umpires": "TV umpire",
        "match_referees": "match referee",
    }
    rows: list[dict[str, Any]] = []

    for source_role, output_role in role_names.items():
        for official_name in officials.get(source_role, []) or []:
            rows.append(
                {
                    "match_id": path.stem,
                    "official_name": official_name,
                    "official_id": _player_id(match, official_name),
                    "role": output_role,
                }
            )

    return rows


SCHEMAS: dict[str, dict[str, Any]] = {
    "matches": {
        "match_id": pl.String,
        "start_date": pl.String,
        "season": pl.String,
        "balls_per_over": pl.Int64,
        "scheduled_overs": pl.Float64,
        "match_type": pl.String,
        "match_type_number": pl.Int64,
        "gender": pl.String,
        "team_type": pl.String,
        "city": pl.String,
        "venue": pl.String,
        "event_name": pl.String,
        "event_match_number": pl.String,
        "team1": pl.String,
        "team2": pl.String,
        "toss_winner": pl.String,
        "toss_decision": pl.String,
        "outcome_winner": pl.String,
        "outcome_result": pl.String,
        "outcome_method": pl.String,
        "win_by_runs": pl.Int64,
        "win_by_wickets": pl.Int64,
        "player_of_match": pl.String,
    },
    "innings": {
        "match_id": pl.String,
        "innings_number": pl.Int64,
        "batting_team": pl.String,
        "target_runs": pl.Int64,
        "target_overs": pl.Float64,
        "total_runs": pl.Int64,
        "wickets_lost": pl.Int64,
        "total_delivery_records": pl.Int64,
        "legal_balls": pl.Int64,
        "overs_bowled": pl.Float64,
    },
    "deliveries": {
        "match_id": pl.String,
        "innings_number": pl.Int64,
        "batting_team": pl.String,
        "over_number": pl.Int64,
        "actual_delivery": pl.String,
        "delivery_sequence": pl.Int64,
        "batter": pl.String,
        "batter_id": pl.String,
        "non_striker": pl.String,
        "non_striker_id": pl.String,
        "bowler": pl.String,
        "bowler_id": pl.String,
        "batter_runs": pl.Int64,
        "extras_runs": pl.Int64,
        "total_runs": pl.Int64,
        "wides": pl.Int64,
        "noballs": pl.Int64,
        "byes": pl.Int64,
        "legbyes": pl.Int64,
        "penalty": pl.Int64,
        "is_legal_delivery": pl.Boolean,
        "is_wicket": pl.Boolean,
        "wickets_on_delivery": pl.Int64,
    },
    "wickets": {
        "match_id": pl.String,
        "innings_number": pl.Int64,
        "batting_team": pl.String,
        "over_number": pl.Int64,
        "actual_delivery": pl.String,
        "delivery_sequence": pl.Int64,
        "player_out": pl.String,
        "player_out_id": pl.String,
        "wicket_kind": pl.String,
        "fielder_names": pl.List(pl.String),
        "fielder_ids": pl.List(pl.String),
    },
    "players": {
        "match_id": pl.String,
        "team": pl.String,
        "player_name": pl.String,
        "player_id": pl.String,
    },
    "powerplays": {
        "match_id": pl.String,
        "innings_number": pl.Int64,
        "batting_team": pl.String,
        "powerplay_type": pl.String,
        "from_ball": pl.Float64,
        "to_ball": pl.Float64,
    },
    "officials": {
        "match_id": pl.String,
        "official_name": pl.String,
        "official_id": pl.String,
        "role": pl.String,
    },
}


def _frame(rows: list[dict[str, Any]], table_name: str) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=SCHEMAS[table_name])


def build_cricket_tables(data_dir: Path | str | None = None) -> dict[str, pl.DataFrame]:
    if data_dir is None:
        data_path = find_project_root() / "data" / "t20s_json"
    else:
        data_path = Path(data_dir).expanduser().resolve()

    json_paths = sorted(data_path.glob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No .json files found in {data_path}")

    rows: dict[str, list[dict[str, Any]]] = {
        "matches": [],
        "innings": [],
        "deliveries": [],
        "wickets": [],
        "players": [],
        "powerplays": [],
        "officials": [],
    }

    for path in json_paths:
        match = load_match_json(path)
        rows["matches"].append(parse_match(path, match))
        rows["innings"].extend(parse_innings(path, match))
        rows["deliveries"].extend(parse_deliveries(path, match))
        rows["wickets"].extend(parse_wickets(path, match))
        rows["players"].extend(parse_players(path, match))
        rows["powerplays"].extend(parse_powerplays(path, match))
        rows["officials"].extend(parse_officials(path, match))

    return {table_name: _frame(table_rows, table_name) for table_name, table_rows in rows.items()}


if __name__ == "__main__":
    tables = build_cricket_tables()
    for name, table in tables.items():
        print(f"{name}: {table.shape[0]:,} rows x {table.shape[1]} columns")
