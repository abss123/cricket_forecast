"""The single point-in-time feature function, for training replay AND live inference.

``compute_features`` is a pure function of its three arguments — it holds no
state of its own — so the exact same call produces the exact same row
whether it's invoked while replaying ball 47 of a 2019 match from
``data/t20s_json``, or live during an in-progress match. All of the
point-in-time bookkeeping (team form, Elo, venue/player history) lives in
``PointInTimeHistory`` (see ``history.py``) and is the caller's
responsibility to advance correctly between matches.

``match_info`` contract
------------------------
A dict with (at least) these keys — see ``scripts/build_training_table.py``
for how it's built from a Cricsheet match file:

    match_id, date, season, venue, venue_clean, venue_country,
    batting_team, bowling_team, toss_winner, toss_decision,
    scheduled_overs, balls_per_over, batting_players, bowling_players,
    registry (dict[player name -> registry id])

``deliveries_so_far``
----------------------
A list of raw Cricsheet delivery dicts (``{"batter", "bowler",
"non_striker", "runs": {...}, "extras": {...}, "wickets": [...]}``) for the
*current 1st innings only*, in order, ending at the ball this feature row
describes.
"""

from __future__ import annotations

from typing import Any

from src.features.history import PointInTimeHistory
from src.features.team_tiers import classify_home_away, team_tier, tier_pairing

RUN_RATE_WINDOWS = (3, 5)
MAX_OVERS_PER_BOWLER_DIVISOR = 5  # T20 rule: no bowler may bowl more than overs/5


def is_legal_delivery(delivery: dict[str, Any]) -> bool:
    extras = delivery.get("extras") or {}
    return extras.get("wides") is None and extras.get("noballs") is None


def is_dismissal_wicket(wicket: dict[str, Any]) -> bool:
    return wicket.get("kind") not in {"retired hurt", "retired not out"}


def _delivery_total(delivery: dict[str, Any]) -> int:
    return (delivery.get("runs") or {}).get("total", 0) or 0


def _delivery_batter_runs(delivery: dict[str, Any]) -> int:
    return (delivery.get("runs") or {}).get("batter", 0) or 0


def _delivery_wickets(delivery: dict[str, Any]) -> list[dict[str, Any]]:
    return [w for w in (delivery.get("wickets") or []) if is_dismissal_wicket(w)]


def _trailing_window(deliveries: list[dict[str, Any]], n_overs: int, balls_per_over: int) -> list[dict[str, Any]]:
    """Deliveries from the tail of the innings covering >= n_overs legal overs.

    Truncated at the start of the innings if fewer than n_overs have been
    bowled yet (rather than returning None) — documented in
    docs/feature_audit.md as "run_rate_last_n uses whatever overs are
    available before over n".
    """
    target = n_overs * balls_per_over
    legal_seen = 0
    start = 0
    for i in range(len(deliveries) - 1, -1, -1):
        if is_legal_delivery(deliveries[i]):
            legal_seen += 1
        start = i
        if legal_seen >= target:
            break
    return deliveries[start:]


def _ball_state_features(deliveries_so_far: list[dict[str, Any]], match_info: dict[str, Any]) -> dict[str, Any]:
    balls_per_over = match_info.get("balls_per_over", 6)
    scheduled_overs = match_info.get("scheduled_overs", 20)
    max_legal_balls = int(round(scheduled_overs * balls_per_over))

    legal_balls = sum(1 for d in deliveries_so_far if is_legal_delivery(d))
    score = sum(_delivery_total(d) for d in deliveries_so_far)
    wickets = sum(len(_delivery_wickets(d)) for d in deliveries_so_far)
    overs_completed = legal_balls / balls_per_over if balls_per_over else 0.0
    run_rate = score / overs_completed if overs_completed > 0 else None

    features: dict[str, Any] = {
        "legal_balls_bowled": legal_balls,
        "balls_remaining": max_legal_balls - legal_balls,
        "score": score,
        "wickets": wickets,
        "run_rate": run_rate,
    }

    for n in RUN_RATE_WINDOWS:
        window = _trailing_window(deliveries_so_far, n, balls_per_over)
        window_legal = [d for d in window if is_legal_delivery(d)]
        window_overs = len(window_legal) / balls_per_over if balls_per_over else 0.0
        window_runs = sum(_delivery_total(d) for d in window_legal)
        features[f"run_rate_last_{n}"] = (window_runs / window_overs) if window_overs > 0 else None
        features[f"wickets_last_{n}"] = sum(len(_delivery_wickets(d)) for d in window)

    current_over_1indexed = legal_balls // balls_per_over + 1
    if current_over_1indexed <= 6:
        phase = "powerplay"
    elif current_over_1indexed <= 15:
        phase = "middle"
    else:
        phase = "death"
    features["phase"] = phase

    last_wicket_idx = -1
    for i, d in enumerate(deliveries_so_far):
        if _delivery_wickets(d):
            last_wicket_idx = i
    partnership_deliveries = deliveries_so_far[last_wicket_idx + 1 :]
    partnership_legal = [d for d in partnership_deliveries if is_legal_delivery(d)]
    features["partnership_runs"] = sum(_delivery_total(d) for d in partnership_deliveries)
    features["partnership_balls"] = len(partnership_legal)

    if deliveries_so_far:
        last = deliveries_so_far[-1]
        striker, non_striker = last.get("batter"), last.get("non_striker")
    else:
        striker, non_striker = None, None
    features["striker"] = striker
    features["non_striker"] = non_striker

    for role, name in (("striker", striker), ("non_striker", non_striker)):
        faced = [d for d in deliveries_so_far if is_legal_delivery(d) and d.get("batter") == name]
        runs = sum(_delivery_batter_runs(d) for d in faced)
        balls = len(faced)
        features[f"{role}_runs"] = runs
        features[f"{role}_balls"] = balls
        features[f"{role}_strike_rate"] = (runs / balls * 100) if balls > 0 else None

    fours = sum(1 for d in deliveries_so_far if is_legal_delivery(d) and _delivery_batter_runs(d) == 4)
    sixes = sum(1 for d in deliveries_so_far if is_legal_delivery(d) and _delivery_batter_runs(d) == 6)
    features["fours"], features["sixes"] = fours, sixes

    extras_totals = {"wides": 0, "noballs": 0, "byes": 0, "legbyes": 0, "penalty": 0}
    for d in deliveries_so_far:
        extras = d.get("extras") or {}
        for key in extras_totals:
            extras_totals[key] += extras.get(key, 0) or 0
    for key, value in extras_totals.items():
        features[f"extras_{key}"] = value
    features["extras_total"] = sum(extras_totals.values())

    return features


def _in_match_names(deliveries_so_far: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for d in deliveries_so_far:
        if d.get("batter"):
            names.add(d["batter"])
        if d.get("non_striker"):
            names.add(d["non_striker"])
    return names


def _player_features(
    deliveries_so_far: list[dict[str, Any]],
    match_info: dict[str, Any],
    history: PointInTimeHistory,
    striker: str | None,
    non_striker: str | None,
) -> dict[str, Any]:
    registry = match_info.get("registry", {}) or {}
    features: dict[str, Any] = {}

    for role, name in (("striker", striker), ("non_striker", non_striker)):
        player_id = registry.get(name) if name else None
        career_a = history.player_career_t20i.batting_stats(player_id)
        features[f"{role}_career_sr_t20i"] = career_a["career_strike_rate"] if career_a else None
        features[f"{role}_career_avg_t20i"] = career_a["career_average"] if career_a else None
        features[f"{role}_career_innings_t20i"] = career_a["career_innings"] if career_a else 0
        if history.player_career_all_t20 is not None:
            career_b = history.player_career_all_t20.batting_stats(player_id)
            features[f"{role}_career_sr_all_t20"] = career_b["career_strike_rate"] if career_b else None
            features[f"{role}_career_avg_all_t20"] = career_b["career_average"] if career_b else None
        else:
            features[f"{role}_career_sr_all_t20"] = None
            features[f"{role}_career_avg_all_t20"] = None

    already_appeared = _in_match_names(deliveries_so_far)
    remaining = [n for n in match_info.get("batting_players", []) if n not in already_appeared]
    remaining_stats = [history.player_career_t20i.batting_stats(registry.get(n)) for n in remaining]
    remaining_srs = [s["career_strike_rate"] for s in remaining_stats if s and s["career_strike_rate"] is not None]
    remaining_avgs = [s["career_average"] for s in remaining_stats if s and s["career_average"] is not None]
    features["remaining_batters_count"] = len(remaining)
    features["remaining_batting_order_sr"] = sum(remaining_srs) / len(remaining_srs) if remaining_srs else None
    features["remaining_batting_order_avg"] = sum(remaining_avgs) / len(remaining_avgs) if remaining_avgs else None

    scheduled_overs = match_info.get("scheduled_overs", 20)
    max_overs_per_bowler = scheduled_overs / MAX_OVERS_PER_BOWLER_DIVISOR
    balls_per_over = match_info.get("balls_per_over", 6)
    weighted_economy_sum, weight_sum, n_bowlers_remaining = 0.0, 0.0, 0
    for name in match_info.get("bowling_players", []):
        overs_bowled = sum(1 for d in deliveries_so_far if is_legal_delivery(d) and d.get("bowler") == name) / balls_per_over
        overs_remaining = max(max_overs_per_bowler - overs_bowled, 0.0)
        if overs_remaining <= 0:
            continue
        n_bowlers_remaining += 1
        bowling_stats = history.player_career_t20i.bowling_stats(registry.get(name))
        if bowling_stats and bowling_stats["career_economy"] is not None:
            weighted_economy_sum += bowling_stats["career_economy"] * overs_remaining
            weight_sum += overs_remaining
    features["bowlers_with_overs_remaining"] = n_bowlers_remaining
    features["remaining_bowling_economy_weighted"] = (
        weighted_economy_sum / weight_sum if weight_sum > 0 else None
    )

    return features


def _team_context_features(match_info: dict[str, Any], history: PointInTimeHistory) -> dict[str, Any]:
    batting_team = match_info["batting_team"]
    bowling_team = match_info["bowling_team"]

    venue_mean, venue_n = history.shrunk_venue_mean(match_info.get("venue_clean"))
    bat_form, bat_n = history.shrunk_team_batting_form(batting_team)
    bowl_form, bowl_n = history.shrunk_team_bowling_form(bowling_team)

    return {
        "season": match_info.get("season"),
        "toss_winner": match_info.get("toss_winner"),
        "toss_decision": match_info.get("toss_decision"),
        "batting_team_tier": team_tier(batting_team),
        "bowling_team_tier": team_tier(bowling_team),
        "tier_pairing": tier_pairing(batting_team, bowling_team),
        "home_away": classify_home_away(batting_team, bowling_team, match_info.get("venue_country")),
        "elo_batting": history.elo.rating(batting_team),
        "elo_bowling": history.elo.rating(bowling_team),
        "elo_diff": history.elo.rating(batting_team) - history.elo.rating(bowling_team),
        "elo_n_batting": history.elo.n(batting_team),
        "elo_n_bowling": history.elo.n(bowling_team),
        "team_batting_recent_form": bat_form,
        "team_batting_recent_form_n": bat_n,
        "team_bowling_recent_form": bowl_form,
        "team_bowling_recent_form_n": bowl_n,
        "venue_avg_first_innings_score": venue_mean,
        "venue_n_prior_matches": venue_n,
    }


def compute_features(
    deliveries_so_far: list[dict[str, Any]],
    match_info: dict[str, Any],
    history: PointInTimeHistory,
) -> dict[str, Any]:
    """One row of point-in-time features for the current ball of a 1st innings.

    Used identically by the training replay (``scripts/build_training_table.py``)
    and by live inference — this function never mutates ``history``.
    """
    ball_state = _ball_state_features(deliveries_so_far, match_info)
    player = _player_features(
        deliveries_so_far, match_info, history, ball_state["striker"], ball_state["non_striker"]
    )
    context = _team_context_features(match_info, history)

    return {
        "match_id": match_info.get("match_id"),
        "date": match_info.get("date"),
        "batting_team": match_info.get("batting_team"),
        "bowling_team": match_info.get("bowling_team"),
        "venue": match_info.get("venue"),
        **ball_state,
        **player,
        **context,
    }
