"""Point-in-time state stores for team/venue/player features.

Every store here only ever answers "what do we know as of right now", and is
advanced one match (or one day's worth of matches) at a time via ``update``/
``update_from_match``. Point-in-time correctness is a *usage* guarantee, not
something the classes enforce themselves: callers (``scripts/build_training_table.py``,
``tests/test_state_leakage.py``) must call the read methods for every match
in a date group *before* calling the update methods for that same group —
see ``PointInTimeHistory.apply_match_result`` for the single call that does
the latter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def shrink(estimate: float | None, n: int, global_mean: float | None, k: float) -> float | None:
    """Empirical-Bayes shrinkage of ``estimate`` (from ``n`` observations) toward ``global_mean``.

    Larger ``k`` pulls harder toward the global mean for the same ``n`` —
    this is the "stronger shrinkage for venue/team features" knob.
    """
    if global_mean is None:
        return estimate
    if estimate is None or n == 0:
        return global_mean
    return (n * estimate + k * global_mean) / (n + k)


@dataclass
class TeamEloStore:
    """Standard logistic Elo, updated only on matches with a clear winner or tie."""

    k: float = 24.0
    default_rating: float = 1500.0
    ratings: dict[str, float] = field(default_factory=dict)
    n_matches: dict[str, int] = field(default_factory=dict)

    def rating(self, team: str) -> float:
        return self.ratings.get(team, self.default_rating)

    def n(self, team: str) -> int:
        return self.n_matches.get(team, 0)

    def update(self, team1: str, team2: str, winner: str | None, is_tie: bool) -> None:
        r1, r2 = self.rating(team1), self.rating(team2)
        expected1 = 1.0 / (1.0 + 10 ** ((r2 - r1) / 400.0))
        if is_tie:
            score1 = 0.5
        elif winner == team1:
            score1 = 1.0
        elif winner == team2:
            score1 = 0.0
        else:
            return  # no result / no decisive outcome: rating is left unchanged
        self.ratings[team1] = r1 + self.k * (score1 - expected1)
        self.ratings[team2] = r2 + self.k * ((1 - score1) - (1 - expected1))
        self.n_matches[team1] = self.n(team1) + 1
        self.n_matches[team2] = self.n(team2) + 1


@dataclass
class _EwmaSeries:
    halflife: float
    values: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def get(self, key: str) -> tuple[float | None, int]:
        return self.values.get(key), self.counts.get(key, 0)

    def update(self, key: str, value: float) -> None:
        n = self.counts.get(key, 0)
        if n == 0:
            self.values[key] = value
        else:
            alpha = 1 - 0.5 ** (1 / self.halflife)
            self.values[key] = alpha * value + (1 - alpha) * self.values[key]
        self.counts[key] = n + 1


@dataclass
class TeamFormStore:
    """Exponentially-decayed recent 1st-innings scoring/conceding, per team.

    Batting and bowling series are separate: a team's runs scored batting
    first and runs conceded bowling first are different distributions.
    """

    halflife: float = 5.0
    _batting: _EwmaSeries = field(init=False)
    _bowling: _EwmaSeries = field(init=False)

    def __post_init__(self) -> None:
        self._batting = _EwmaSeries(self.halflife)
        self._bowling = _EwmaSeries(self.halflife)

    def batting_form(self, team: str) -> tuple[float | None, int]:
        return self._batting.get(team)

    def bowling_form(self, team: str) -> tuple[float | None, int]:
        return self._bowling.get(team)

    def update(self, batting_team: str, bowling_team: str, first_innings_runs: float) -> None:
        self._batting.update(batting_team, first_innings_runs)
        self._bowling.update(bowling_team, first_innings_runs)


@dataclass
class RunningMean:
    n: int = 0
    mean: float = 0.0

    def value(self) -> float | None:
        return self.mean if self.n > 0 else None

    def update(self, x: float) -> None:
        self.n += 1
        self.mean += (x - self.mean) / self.n


@dataclass
class VenueStatsStore:
    """Point-in-time mean 1st-innings score per canonical venue."""

    stats: dict[str, RunningMean] = field(default_factory=dict)

    def get(self, venue_clean: str) -> tuple[float | None, int]:
        rm = self.stats.get(venue_clean)
        return (rm.value(), rm.n) if rm else (None, 0)

    def update(self, venue_clean: str, first_innings_runs: float) -> None:
        self.stats.setdefault(venue_clean, RunningMean()).update(first_innings_runs)


@dataclass
class PlayerCareerStore:
    """Point-in-time career batting/bowling aggregates, keyed by registry player_id.

    ``source`` is a free-text label (e.g. "t20i" or "t20i+leagues") purely
    for reporting/debugging which corpus fed this instance — it has no
    effect on the arithmetic.
    """

    source: str
    batting: dict[str, dict[str, int]] = field(default_factory=dict)
    bowling: dict[str, dict[str, int]] = field(default_factory=dict)

    def batting_stats(self, player_id: str | None) -> dict[str, float] | None:
        if player_id is None or player_id not in self.batting:
            return None
        row = self.batting[player_id]
        strike_rate = (row["runs"] / row["balls"] * 100) if row["balls"] > 0 else None
        average = (row["runs"] / row["dismissals"]) if row["dismissals"] > 0 else None
        return {
            "career_innings": row["innings"],
            "career_runs": row["runs"],
            "career_balls": row["balls"],
            "career_strike_rate": strike_rate,
            "career_average": average,
        }

    def bowling_stats(self, player_id: str | None) -> dict[str, float] | None:
        if player_id is None or player_id not in self.bowling:
            return None
        row = self.bowling[player_id]
        overs = row["balls"] / 6
        economy = (row["runs"] / overs) if overs > 0 else None
        return {
            "career_balls_bowled": row["balls"],
            "career_runs_conceded": row["runs"],
            "career_wickets": row["wickets"],
            "career_economy": economy,
        }

    def update_from_innings(
        self,
        batting_lines: list[dict[str, Any]],
        bowling_lines: list[dict[str, Any]],
    ) -> None:
        """Apply one completed innings' worth of per-player deltas.

        ``batting_lines``: list of {"player_id", "runs", "balls", "dismissed": bool}.
        ``bowling_lines``: list of {"player_id", "balls", "runs_conceded", "wickets"}.
        """
        for line in batting_lines:
            player_id = line["player_id"]
            if player_id is None:
                continue
            row = self.batting.setdefault(
                player_id, {"runs": 0, "balls": 0, "dismissals": 0, "innings": 0}
            )
            row["runs"] += line["runs"]
            row["balls"] += line["balls"]
            row["dismissals"] += int(line["dismissed"])
            row["innings"] += 1

        for line in bowling_lines:
            player_id = line["player_id"]
            if player_id is None:
                continue
            row = self.bowling.setdefault(player_id, {"balls": 0, "runs": 0, "wickets": 0})
            row["balls"] += line["balls"]
            row["runs"] += line["runs_conceded"]
            row["wickets"] += line["wickets"]


@dataclass
class PointInTimeHistory:
    """Bundles every point-in-time store ``compute_features`` reads from."""

    elo: TeamEloStore = field(default_factory=TeamEloStore)
    team_form: TeamFormStore = field(default_factory=TeamFormStore)
    venue_stats: VenueStatsStore = field(default_factory=VenueStatsStore)
    global_first_innings: RunningMean = field(default_factory=RunningMean)
    player_career_t20i: PlayerCareerStore = field(default_factory=lambda: PlayerCareerStore("t20i"))
    # Version (b) from the spec ("T20I + franchise leagues") is not
    # computable with the data currently on disk (data/t20s_json is
    # international-only — see docs/feature_audit.md). Left as None so
    # compute_features reports it as missing rather than silently reusing
    # the T20I-only numbers.
    player_career_all_t20: PlayerCareerStore | None = None
    venue_shrinkage_k: float = 15.0
    team_shrinkage_k: float = 10.0

    def shrunk_venue_mean(self, venue_clean: str | None) -> tuple[float | None, int]:
        if venue_clean is None:
            return None, 0
        mean, n = self.venue_stats.get(venue_clean)
        return shrink(mean, n, self.global_first_innings.value(), self.venue_shrinkage_k), n

    def shrunk_team_batting_form(self, team: str) -> tuple[float | None, int]:
        mean, n = self.team_form.batting_form(team)
        return shrink(mean, n, self.global_first_innings.value(), self.team_shrinkage_k), n

    def shrunk_team_bowling_form(self, team: str) -> tuple[float | None, int]:
        mean, n = self.team_form.bowling_form(team)
        return shrink(mean, n, self.global_first_innings.value(), self.team_shrinkage_k), n

    def apply_match_result(
        self,
        batting_team: str,
        bowling_team: str,
        first_innings_runs: float,
        venue_clean: str | None,
        winner: str | None,
        is_tie: bool,
        elo_update_eligible: bool,
        batting_lines: list[dict[str, Any]],
        bowling_lines: list[dict[str, Any]],
    ) -> None:
        """Advance every store by one match's worth of 1st-innings results.

        Call this exactly once per match, after computing features for
        every ball of every match in the match's date group (see the
        module docstring) — never before, or later matches on the same
        date would leak into this match's own features.
        """
        if elo_update_eligible:
            self.elo.update(batting_team, bowling_team, winner, is_tie)
        self.team_form.update(batting_team, bowling_team, first_innings_runs)
        if venue_clean is not None:
            self.venue_stats.update(venue_clean, first_innings_runs)
        self.global_first_innings.update(first_innings_runs)
        self.player_career_t20i.update_from_innings(batting_lines, bowling_lines)
