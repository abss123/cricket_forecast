"""Unit-level summaries (team / venue / player) over Cricsheet T20 match data.

Built on the schema findings in ``docs/schema_notes.md``. The pipeline is:

1. ``build_match_table`` — parse every match JSON file into one tidy row per
   match (cached to parquet so it is not re-parsed on every call).
2. ``summarize_units`` — group that match table by team, venue, or player and
   report outcome-count breakdowns plus a header summary of the unit
   distribution.

Design notes / deliberate additions beyond the literal column list requested:

- ``team1`` / ``team2`` (the two teams named in ``info.teams``, in Cricsheet's
  own order) are kept alongside ``players_team1`` / ``players_team2`` —
  without them the two player-ID lists would be unattributable to a team.
- ``dls_affected`` is derived per docs/schema_notes.md §3.3: true when
  ``outcome.method == "D/L"`` or the 2nd-innings target differs from the
  1st-innings total + 1 (a revised target without a recorded method).
- ``decided_by_super_over`` / ``decided_by_bowl_out`` make the two distinct
  tie-resolution mechanics found in the corpus (§3.1) explicit, instead of
  collapsing every tie into one undifferentiated bucket.
- ``outcome.method == "Awarded"`` matches are bucketed as ``"other"`` rather
  than inferred from the scoreboard: inspecting the 4 such matches showed the
  scoreboard does not reliably reflect the ruling (one match was awarded to
  the team that scored *fewer* runs while chasing), so guessing chase/defend
  from runs would silently fabricate a result.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

UnitType = Literal["team", "venue", "player"]

OUTCOME_TYPES: tuple[str, ...] = (
    "chase_won",
    "defend_won",
    "tie",
    "no_result",
    "draw",
    "innings_win",
    "other",
)

MATCH_TABLE_COLUMNS: tuple[str, ...] = (
    "match_id",
    "date",
    "season",
    "match_type",
    "gender",
    "team_type",
    "event",
    "venue",
    "venue_clean",
    "city",
    "team1",
    "team2",
    "team_batting_first",
    "team_batting_second",
    "toss_winner",
    "toss_decision",
    "first_innings_runs",
    "target_runs",
    "target_overs",
    "second_innings_runs",
    "winner",
    "outcome_type",
    "method",
    "dls_affected",
    "has_super_over",
    "decided_by_super_over",
    "decided_by_bowl_out",
    "players_team1",
    "players_team2",
)

_VALID_FILTER_KEYS = {
    "match_type",
    "gender",
    "team_type",
    "start_date",
    "end_date",
    "exclude_super_over_innings",
}


# --------------------------------------------------------------------------
# JSON loading helpers
# --------------------------------------------------------------------------


def _as_str(value: Any) -> str | None:
    """Coerce to str, preserving None. ``info.season`` is a str in most files
    (e.g. "2019/20") but a bare int in others (e.g. 2016) — normalize so the
    column has one dtype for parquet and downstream comparisons."""
    return None if value is None else str(value)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_match_files(data_dir: Path | str) -> list[Path]:
    """Return every match JSON file in ``data_dir``, sorted for determinism."""
    return sorted(Path(data_dir).glob("*.json"))


def _default_cache_path(data_dir: Path, filename: str) -> Path:
    return Path(data_dir) / ".cache" / filename


# --------------------------------------------------------------------------
# Innings helpers
# --------------------------------------------------------------------------


def _innings_total_runs(innings: dict[str, Any]) -> int:
    """Sum ``runs.total`` across every delivery in an innings."""
    return sum(
        (delivery.get("runs", {}) or {}).get("total", 0) or 0
        for over in innings.get("overs", []) or []
        for delivery in over.get("deliveries", []) or []
    )


def _non_super_over_innings(innings_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter out super-over innings, preserving order.

    Super-over matches have 4 (or, after repeated ties, 8) innings entries;
    the "real" 1st/2nd innings are always the first two non-super-over
    entries (docs/schema_notes.md §3.2).
    """
    return [inn for inn in innings_list if not inn.get("super_over")]


# --------------------------------------------------------------------------
# Venue name cleaning
# --------------------------------------------------------------------------


def build_venue_mapping(raw_venues: Iterable[str]) -> pd.DataFrame:
    """Build a raw-venue -> canonical-venue mapping table for manual review.

    Heuristic (per docs/schema_notes.md §3.5): many grounds appear both as a
    bare name (``"Eden Gardens"``) and a city-qualified name
    (``"Eden Gardens, Kolkata"``). Where a bare name has exactly one
    city-qualified counterpart, the bare name is mapped to it. Where a bare
    name has zero or more-than-one qualified counterpart (e.g. ``"County
    Ground"``, which is genuinely ambiguous across 6 UK cities), or where
    multiple *distinct* qualified venues share the same base name (e.g. two
    different "Gymkhana Club Ground"s), the row is flagged ``needs_review``
    and left unresolved (mapped to itself) rather than guessed.

    Returns
    -------
    pd.DataFrame with columns:
        venue_raw, venue_clean, base_name, n_variants_sharing_base,
        resolved_from_bare, needs_review
    """
    venues = sorted({v for v in raw_venues if v})
    base_to_variants: dict[str, list[str]] = defaultdict(list)
    for v in venues:
        base_to_variants[v.split(",")[0].strip()].append(v)

    rows: list[dict[str, Any]] = []
    for base, variants in base_to_variants.items():
        qualified = sorted({v for v in variants if "," in v})
        multiple_distinct_grounds = len(qualified) > 1

        for v in variants:
            if "," in v:
                clean, resolved = v, False
                needs_review = multiple_distinct_grounds
            elif len(qualified) == 1:
                clean, resolved = qualified[0], True
                needs_review = False
            else:
                clean, resolved = v, False
                needs_review = True

            rows.append(
                {
                    "venue_raw": v,
                    "venue_clean": clean,
                    "base_name": base,
                    "n_variants_sharing_base": len(variants),
                    "resolved_from_bare": resolved,
                    "needs_review": needs_review,
                }
            )

    return (
        pd.DataFrame(
            rows,
            columns=[
                "venue_raw",
                "venue_clean",
                "base_name",
                "n_variants_sharing_base",
                "resolved_from_bare",
                "needs_review",
            ],
        )
        .sort_values(["base_name", "venue_raw"])
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------
# Outcome classification
# --------------------------------------------------------------------------


def classify_outcome(
    outcome: dict[str, Any],
    team_batting_first: str | None,
    team_batting_second: str | None,
) -> tuple[str, str | None]:
    """Map a raw ``info.outcome`` dict to one of ``OUTCOME_TYPES``.

    Returns ``(outcome_type, note)`` — ``note`` is non-None only for
    ``"other"``, explaining why the shape didn't fit a known bucket (callers
    should log it rather than drop the match silently).
    """
    result = outcome.get("result")
    by = outcome.get("by", {}) or {}
    winner = outcome.get("winner")

    if result == "no result":
        return "no_result", None
    if result == "draw":
        return "draw", None
    if "innings" in by:
        return "innings_win", None
    if result == "tie":
        return "tie", None
    if winner is not None and ("runs" in by or "wickets" in by):
        if winner == team_batting_second:
            return "chase_won", None
        if winner == team_batting_first:
            return "defend_won", None
        return (
            "other",
            f"winner {winner!r} matches neither batting_first "
            f"({team_batting_first!r}) nor batting_second ({team_batting_second!r})",
        )

    return "other", f"unrecognized/administrative outcome shape: {outcome!r}"


# --------------------------------------------------------------------------
# Match table construction
# --------------------------------------------------------------------------


def _parse_match_row(path: Path, match: dict[str, Any]) -> dict[str, Any]:
    info = match.get("info", {}) or {}
    outcome = info.get("outcome", {}) or {}
    toss = info.get("toss", {}) or {}
    event = info.get("event", {}) or {}
    teams = info.get("teams", []) or []
    team1 = teams[0] if len(teams) > 0 else None
    team2 = teams[1] if len(teams) > 1 else None

    innings = match.get("innings", []) or []
    has_super_over = any(inn.get("super_over") for inn in innings)
    main_innings = _non_super_over_innings(innings)
    first = main_innings[0] if len(main_innings) >= 1 else None
    second = main_innings[1] if len(main_innings) >= 2 else None

    team_batting_first = first.get("team") if first else None
    team_batting_second = second.get("team") if second else None
    first_innings_runs = _innings_total_runs(first) if first else None
    second_innings_runs = _innings_total_runs(second) if second else None

    target = (second or {}).get("target") or {}
    target_runs = target.get("runs")
    target_overs = target.get("overs")

    outcome_type, note = classify_outcome(outcome, team_batting_first, team_batting_second)
    if outcome_type == "other":
        logger.warning("match %s: outcome bucketed as 'other' (%s)", path.stem, note)

    method = outcome.get("method")
    dls_affected = method == "D/L" or (
        target_runs is not None
        and first_innings_runs is not None
        and target_runs != first_innings_runs + 1
    )

    registry = (info.get("registry", {}) or {}).get("people", {}) or {}
    players = info.get("players", {}) or {}
    players_team1 = [registry[n] for n in players.get(team1, []) or [] if n in registry]
    players_team2 = [registry[n] for n in players.get(team2, []) or [] if n in registry]

    return {
        "match_id": path.stem,
        "date": info.get("dates", [None])[0],
        "season": _as_str(info.get("season")),
        "match_type": info.get("match_type"),
        "gender": info.get("gender"),
        "team_type": info.get("team_type"),
        "event": event.get("name"),
        "venue": info.get("venue"),
        "city": info.get("city"),
        "team1": team1,
        "team2": team2,
        "team_batting_first": team_batting_first,
        "team_batting_second": team_batting_second,
        "toss_winner": toss.get("winner"),
        "toss_decision": toss.get("decision"),
        "first_innings_runs": first_innings_runs,
        "target_runs": target_runs,
        "target_overs": target_overs,
        "second_innings_runs": second_innings_runs,
        "winner": outcome.get("winner"),
        "outcome_type": outcome_type,
        "method": method,
        "dls_affected": dls_affected,
        "has_super_over": has_super_over,
        "decided_by_super_over": outcome.get("eliminator") is not None,
        "decided_by_bowl_out": outcome.get("bowl_out") is not None,
        "players_team1": players_team1,
        "players_team2": players_team2,
    }


def build_match_table(
    data_dir: Path | str,
    cache_path: Path | str | None = None,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Parse every match JSON in ``data_dir`` into one tidy row per match.

    Results are cached to ``cache_path`` (parquet). If the cache file already
    exists and ``force_rebuild`` is False, it is loaded directly instead of
    re-parsing the raw JSON. A companion venue-mapping review table is
    written as CSV next to the cache (see ``build_venue_mapping``).

    Parameters
    ----------
    data_dir:
        Directory containing Cricsheet match ``*.json`` files.
    cache_path:
        Parquet path to read/write. Defaults to ``<data_dir>/.cache/match_table.parquet``.
    force_rebuild:
        If True, ignore any existing cache and re-parse from scratch.
    """
    data_dir = Path(data_dir)
    resolved_cache_path = Path(cache_path) if cache_path is not None else _default_cache_path(
        data_dir, "match_table.parquet"
    )

    if resolved_cache_path.exists() and not force_rebuild:
        return pd.read_parquet(resolved_cache_path)

    files = list_match_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No .json match files found in {data_dir}")

    rows = []
    for path in files:
        match = _load_json(path)
        rows.append(_parse_match_row(path, match))

    df = pd.DataFrame(rows)

    venue_mapping = build_venue_mapping(df["venue"].dropna().unique())
    venue_clean_lookup = dict(zip(venue_mapping["venue_raw"], venue_mapping["venue_clean"]))
    df["venue_clean"] = df["venue"].map(venue_clean_lookup)

    df["date"] = pd.to_datetime(df["date"])
    df = df[list(MATCH_TABLE_COLUMNS)]

    resolved_cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(resolved_cache_path, index=False)
    venue_mapping.to_csv(resolved_cache_path.parent / "venue_mapping.csv", index=False)

    return df


def build_player_directory(
    data_dir: Path | str,
    cache_path: Path | str | None = None,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Build a player_id -> canonical display name lookup.

    A single registry ID can be recorded under more than one display name
    over time (docs/schema_notes.md §3.6, e.g. a marriage name change). The
    canonical ``display_name`` is the most frequently observed name for that
    ID; every observed name is kept in ``all_names`` for review.
    """
    data_dir = Path(data_dir)
    resolved_cache_path = Path(cache_path) if cache_path is not None else _default_cache_path(
        data_dir, "player_directory.parquet"
    )

    if resolved_cache_path.exists() and not force_rebuild:
        return pd.read_parquet(resolved_cache_path)

    files = list_match_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No .json match files found in {data_dir}")

    name_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for path in files:
        match = _load_json(path)
        registry = (match.get("info", {}).get("registry", {}) or {}).get("people", {}) or {}
        for name, player_id in registry.items():
            name_counts[player_id][name] += 1

    rows = []
    for player_id, counts in name_counts.items():
        display_name, _ = counts.most_common(1)[0]
        rows.append(
            {
                "player_id": player_id,
                "display_name": display_name,
                "all_names": sorted(counts),
                "n_name_appearances": int(sum(counts.values())),
            }
        )

    directory = pd.DataFrame(rows).sort_values("player_id").reset_index(drop=True)
    resolved_cache_path.parent.mkdir(parents=True, exist_ok=True)
    directory.to_parquet(resolved_cache_path, index=False)
    return directory


# --------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------


def apply_filters(matches: pd.DataFrame, filters: dict[str, Any] | None) -> pd.DataFrame:
    """Apply the supported filter set to a match table.

    Supported keys:
        match_type, gender, team_type: str or list[str], exact-match on the column.
        start_date, end_date: inclusive date bounds (anything ``pd.Timestamp`` accepts).
        exclude_super_over_innings: bool — drop matches where ``has_super_over`` is True.
    """
    if not filters:
        return matches.copy()

    unknown = set(filters) - _VALID_FILTER_KEYS
    if unknown:
        raise ValueError(
            f"Unknown filter keys: {sorted(unknown)}. Valid keys: {sorted(_VALID_FILTER_KEYS)}"
        )

    df = matches
    for key in ("match_type", "gender", "team_type"):
        value = filters.get(key)
        if value is not None:
            values = [value] if isinstance(value, str) else list(value)
            df = df[df[key].isin(values)]

    if filters.get("start_date") is not None:
        df = df[df["date"] >= pd.Timestamp(filters["start_date"])]
    if filters.get("end_date") is not None:
        df = df[df["date"] <= pd.Timestamp(filters["end_date"])]
    if filters.get("exclude_super_over_innings"):
        df = df[~df["has_super_over"]]

    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# Long-format construction per unit
# --------------------------------------------------------------------------


def _role_for_team(
    team: str | None, batting_first: str | None, batting_second: str | None
) -> str | None:
    if team is not None and team == batting_first:
        return "batting_first"
    if team is not None and team == batting_second:
        return "batting_second"
    return None


def _venue_long(matches: pd.DataFrame) -> pd.DataFrame:
    long = matches[["venue_clean", "match_id", "date", "outcome_type", "dls_affected"]].copy()
    long = long.rename(columns={"venue_clean": "unit_id"}).dropna(subset=["unit_id"])
    long["role"] = None
    return long


def _team_long(matches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in matches.itertuples(index=False):
        for team in (r.team1, r.team2):
            if team is None:
                continue
            rows.append(
                {
                    "unit_id": team,
                    "match_id": r.match_id,
                    "date": r.date,
                    "outcome_type": r.outcome_type,
                    "dls_affected": r.dls_affected,
                    "role": _role_for_team(team, r.team_batting_first, r.team_batting_second),
                }
            )
    return pd.DataFrame(rows, columns=["unit_id", "match_id", "date", "outcome_type", "dls_affected", "role"])


def _as_list(value: Any) -> list[Any]:
    """Normalize a players_teamN cell to a plain list.

    A freshly built match table stores Python lists, but after a parquet
    round-trip pyarrow deserializes list columns as numpy arrays, whose
    truthiness (``arr or []``) is ambiguous for length > 1 — so this always
    goes through ``list()`` rather than relying on truthiness.
    """
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    return list(value)


def _player_long(matches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in matches.itertuples(index=False):
        role1 = _role_for_team(r.team1, r.team_batting_first, r.team_batting_second)
        role2 = _role_for_team(r.team2, r.team_batting_first, r.team_batting_second)
        for player_id in _as_list(r.players_team1):
            rows.append(
                {
                    "unit_id": player_id,
                    "match_id": r.match_id,
                    "date": r.date,
                    "outcome_type": r.outcome_type,
                    "dls_affected": r.dls_affected,
                    "role": role1,
                }
            )
        for player_id in _as_list(r.players_team2):
            rows.append(
                {
                    "unit_id": player_id,
                    "match_id": r.match_id,
                    "date": r.date,
                    "outcome_type": r.outcome_type,
                    "dls_affected": r.dls_affected,
                    "role": role2,
                }
            )
    return pd.DataFrame(rows, columns=["unit_id", "match_id", "date", "outcome_type", "dls_affected", "role"])


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def _aggregate_unit(long: pd.DataFrame, unit: UnitType) -> pd.DataFrame:
    base_columns = (
        ["unit_id", "unit_name", "n_matches"]
        + [f"n_{o}" for o in OUTCOME_TYPES]
        + ["n_dls_affected", "chase_win_pct", "first_date", "last_date"]
    )
    if unit in ("team", "player"):
        base_columns += ["n_matches_batting_first", "n_matches_batting_second"]

    if long.empty:
        return pd.DataFrame(columns=base_columns)

    summary = (
        long.groupby("unit_id")
        .agg(n_matches=("match_id", "count"), first_date=("date", "min"), last_date=("date", "max"))
        .reset_index()
    )

    for outcome in OUTCOME_TYPES:
        counts = long[long["outcome_type"] == outcome].groupby("unit_id")["match_id"].count()
        summary[f"n_{outcome}"] = summary["unit_id"].map(counts).fillna(0).astype(int)

    dls_counts = long[long["dls_affected"]].groupby("unit_id")["match_id"].count()
    summary["n_dls_affected"] = summary["unit_id"].map(dls_counts).fillna(0).astype(int)

    decided = summary["n_matches"] - summary["n_no_result"] - summary["n_draw"]
    with np.errstate(invalid="ignore", divide="ignore"):
        summary["chase_win_pct"] = (summary["n_chase_won"] / decided.replace(0, np.nan) * 100).round(1)

    if unit in ("team", "player"):
        role_counts = long.groupby(["unit_id", "role"])["match_id"].count().unstack(fill_value=0)
        for role_col, out_col in (
            ("batting_first", "n_matches_batting_first"),
            ("batting_second", "n_matches_batting_second"),
        ):
            if role_col in role_counts.columns:
                summary[out_col] = summary["unit_id"].map(role_counts[role_col]).fillna(0).astype(int)
            else:
                summary[out_col] = 0

    summary["unit_name"] = summary["unit_id"]
    return summary[base_columns]


def _attach_player_names(summary: pd.DataFrame, player_directory: pd.DataFrame | None) -> pd.DataFrame:
    if player_directory is None:
        return summary
    name_lookup = dict(zip(player_directory["player_id"], player_directory["display_name"]))
    summary = summary.copy()
    summary["unit_name"] = summary["unit_id"].map(name_lookup).fillna(summary["unit_id"])
    return summary


def validate_outcome_counts(summary: pd.DataFrame) -> None:
    """Raise ValueError if any unit's outcome-type counts don't sum to n_matches."""
    outcome_sum = summary[[f"n_{o}" for o in OUTCOME_TYPES]].sum(axis=1)
    mismatched = summary.loc[outcome_sum != summary["n_matches"], "unit_id"]
    if not mismatched.empty:
        raise ValueError(
            f"Outcome-type counts do not sum to n_matches for units: {mismatched.tolist()}"
        )


def _build_header_summary(summary: pd.DataFrame, unit: UnitType) -> dict[str, Any]:
    if summary.empty:
        return {
            "unit": unit,
            "n_units": 0,
            "n_matches_min": 0,
            "n_matches_median": 0.0,
            "n_matches_max": 0,
            "n_units_lt_10_matches": 0,
        }
    n_matches = summary["n_matches"]
    return {
        "unit": unit,
        "n_units": int(len(summary)),
        "n_matches_min": int(n_matches.min()),
        "n_matches_median": float(n_matches.median()),
        "n_matches_max": int(n_matches.max()),
        "n_units_lt_10_matches": int((n_matches < 10).sum()),
    }


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def summarize_units(
    matches: pd.DataFrame,
    unit: UnitType,
    filters: dict[str, Any] | None = None,
    player_directory: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Summarize a match table at the level of team, venue, or player.

    Parameters
    ----------
    matches:
        A match table as produced by ``build_match_table``.
    unit:
        One of ``"team"``, ``"venue"``, ``"player"``.
    filters:
        Optional filter dict, see ``apply_filters``.
    player_directory:
        Optional lookup from ``build_player_directory``, used to attach a
        display name to each player ID when ``unit == "player"``. If
        omitted, ``unit_name`` falls back to the raw player ID.

    Returns
    -------
    (summary, header): a one-row-per-unit DataFrame with columns
        unit_id, unit_name, n_matches, n_chase_won, n_defend_won, n_tie,
        n_no_result, n_draw, n_innings_win, n_other, n_dls_affected,
        chase_win_pct, first_date, last_date
        (plus n_matches_batting_first / n_matches_batting_second for
        "team" and "player"), and a header dict with n_units and the
        n_matches distribution (min/median/max, count of units with < 10
        matches).

    ``chase_win_pct`` is ``n_chase_won / (n_matches - n_no_result - n_draw)``
    for every unit type, i.e. it is *not* conditioned on batting role — for a
    team/player that mostly batted first, this will understate their actual
    chase-success rate. Use ``n_matches_batting_second`` (team/player only)
    to compute a role-conditioned rate if needed.
    """
    if unit not in ("team", "venue", "player"):
        raise ValueError(f"unit must be one of 'team', 'venue', 'player'; got {unit!r}")

    filtered = apply_filters(matches, filters)

    if unit == "venue":
        long = _venue_long(filtered)
    elif unit == "team":
        long = _team_long(filtered)
    else:
        long = _player_long(filtered)

    summary = _aggregate_unit(long, unit)
    if unit == "player":
        summary = _attach_player_names(summary, player_directory)

    validate_outcome_counts(summary)
    header = _build_header_summary(summary, unit)
    return summary, header
