"""ICC full-member vs associate team tiering.

Static membership list (there is no reliable per-match "tier" field in
Cricsheet, and membership changes rarely enough that a hardcoded set is
easier to audit than a fuzzy join against an external source). Current as
of 2026; update ``FULL_MEMBERS`` if the ICC admits/expels a board.
"""

from __future__ import annotations

from typing import Literal

TeamTier = Literal["full", "associate"]

# The 12 current ICC full members. Afghanistan is included for correctness
# even though it never appears in data/t20s_json: Cricsheet withholds all
# Afghanistan-involved matches from this download (docs/schema_notes.md).
FULL_MEMBERS: frozenset[str] = frozenset(
    {
        "Afghanistan",
        "Australia",
        "Bangladesh",
        "England",
        "India",
        "Ireland",
        "New Zealand",
        "Pakistan",
        "South Africa",
        "Sri Lanka",
        "West Indies",
        "Zimbabwe",
    }
)

# Not a national board — an exhibition/invitational side. Tagged "associate"
# so it doesn't inflate full-member statistics, but callers doing anything
# tier-sensitive should consider excluding it outright.
NOT_A_BOARD: frozenset[str] = frozenset({"ICC World XI"})


def team_tier(team: str | None) -> TeamTier | None:
    """Return "full" or "associate" for a Cricsheet team name, or None if unset."""
    if team is None:
        return None
    return "full" if team in FULL_MEMBERS else "associate"


def tier_pairing(team1: str | None, team2: str | None) -> str | None:
    """Return an order-independent tier pairing label, e.g. "full_vs_associate"."""
    tier1, tier2 = team_tier(team1), team_tier(team2)
    if tier1 is None or tier2 is None:
        return None
    return "_vs_".join(sorted((tier1, tier2)))


# West Indies is a composite of several sovereign Caribbean nations/
# territories; every other team's "home" country is just its own name (see
# src/features/venue_country.py for how venues resolve to these same
# country strings). Only West Indies needs an explicit override.
WEST_INDIES_COUNTRIES: frozenset[str] = frozenset(
    {
        "Antigua and Barbuda",
        "Barbados",
        "Dominica",
        "Grenada",
        "Guyana",
        "Jamaica",
        "Saint Kitts and Nevis",
        "Saint Lucia",
        "Saint Vincent and the Grenadines",
        "Trinidad and Tobago",
    }
)

TEAM_HOME_COUNTRIES: dict[str, frozenset[str]] = {"West Indies": WEST_INDIES_COUNTRIES}


def team_home_countries(team: str | None) -> frozenset[str]:
    """Countries where ``team`` is considered to be playing at home.

    Defaults to ``{team}`` (the team name doubles as its home country name
    for the large majority of Cricsheet teams, by construction of the
    venue->country aliasing in ``venue_country.py``), with explicit
    overrides in ``TEAM_HOME_COUNTRIES`` for composite teams.
    """
    if team is None:
        return frozenset()
    return TEAM_HOME_COUNTRIES.get(team, frozenset({team}))


HomeAway = Literal["home", "away", "neutral"]


def classify_home_away(batting_team: str | None, bowling_team: str | None, venue_country: str | None) -> HomeAway | None:
    """Home/away/neutral from the batting team's perspective.

    "home" if the venue's country is one of the batting team's home
    countries, "away" if it's one of the bowling team's, else "neutral".
    A venue with a home country for *both* teams is not possible with the
    current team-to-country mapping other than a West Indies-vs-West-Indies
    fixture, which doesn't occur.
    """
    if venue_country is None:
        return None
    if venue_country in team_home_countries(batting_team):
        return "home"
    if venue_country in team_home_countries(bowling_team):
        return "away"
    return "neutral"
