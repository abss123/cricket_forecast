"""Venue -> country mapping, for the home/away/neutral feature.

Cricsheet gives a raw ``venue`` string and an optional ``city``, but no
country. This module resolves a "cricket board country" (not strictly a
sovereign-state lookup — see the deliberate exceptions below) for every
venue observed in the male-international scope, and writes the result to
``docs/venue_country_mapping.csv`` for manual review.

Approach
--------
1. Canonicalize raw venue strings with ``src.eda.unit_summary.build_venue_mapping``
   (already used for the 2nd-innings win-probability table) so
   "Eden Gardens" and "Eden Gardens, Kolkata" resolve to one venue.
2. For each canonical venue, take the most frequently recorded non-empty
   ``city`` across all matches at that venue (some grounds are recorded
   with an empty city in a subset of files, e.g. "Harare Sports Club").
3. Look the city up in ``CITY_ALIASES`` (also contains a handful of
   ``city`` values that are themselves country/territory names, e.g.
   Cricsheet sometimes puts "Guyana" or "Barbados" in the city field).
4. If no city is ever recorded for a venue, fall back to ``GROUND_OVERRIDES``
   keyed on the venue string itself.
5. Anything still unresolved is left as ``country=None, needs_review=True``
   rather than guessed.

Deliberate exceptions to "country" meaning "sovereign state"
--------------------------------------------------------------
- Northern Ireland venues (Belfast, Derry/Londonderry, Bready) map to
  ``"Ireland"``, not ``"United Kingdom"`` — Cricket Ireland fields one team
  for the whole island, so these grounds are cricketing "home" for Ireland.
- Welsh venues (Cardiff/Sophia Gardens) map to ``"England"`` — Wales has no
  separate team; the England & Wales Cricket Board covers both.
- West Indies is a composite of many sovereign Caribbean nations/territories
  (Guyana, Jamaica, Trinidad and Tobago, Barbados, ...). This module maps
  each venue to its actual country (so e.g. Sabina Park -> "Jamaica"); the
  team-side home/away resolution in ``team_tiers.TEAM_HOME_COUNTRIES``
  special-cases "West Indies" to match against the whole set of those
  countries, rather than a single one.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.eda.unit_summary import build_venue_mapping

# city (or, in a handful of Cricsheet rows, a territory name recorded in the
# city field) -> cricket-board country. Built by hand against the 352
# distinct (venue, city) pairs observed in the male/international scope of
# data/t20s_json — see docs/feature_audit.md for the full derivation note.
CITY_ALIASES: dict[str, str] = {
    # Middle East
    "Abu Dhabi": "United Arab Emirates", "Dubai": "United Arab Emirates",
    "Sharjah": "United Arab Emirates", "Doha": "Qatar", "Al Amarat": "Oman",
    "Marsa": "Malta",
    # South Asia
    "Delhi": "India", "New Delhi": "India", "Kolkata": "India", "Mumbai": "India",
    "Bangalore": "India", "Bengaluru": "India", "Chennai": "India", "Ahmedabad": "India",
    "Chandigarh": "India", "New Chandigarh": "India", "Cuttack": "India",
    "Guwahati": "India", "Dharamsala": "India", "Dharmasala": "India",
    "Hyderabad": "India", "Indore": "India", "Jaipur": "India", "Kanpur": "India",
    "Lucknow": "India", "Nagpur": "India", "Pune": "India", "Raipur": "India",
    "Rajkot": "India", "Ranchi": "India", "Thiruvananthapuram": "India",
    "Visakhapatnam": "India", "Gwalior": "India",
    "Lahore": "Pakistan", "Karachi": "Pakistan", "Rawalpindi": "Pakistan",
    "Colombo": "Sri Lanka", "Kandy": "Sri Lanka", "Dambulla": "Sri Lanka",
    "Hambantota": "Sri Lanka",
    "Dhaka": "Bangladesh", "Mirpur": "Bangladesh", "Chattogram": "Bangladesh",
    "Chittagong": "Bangladesh", "Sylhet": "Bangladesh", "Khulna": "Bangladesh",
    "Fatullah": "Bangladesh",
    "Kathmandu": "Nepal", "Kirtipur": "Nepal",
    "Gelephu": "Bhutan",
    # Australasia
    "Adelaide": "Australia", "Brisbane": "Australia", "Canberra": "Australia",
    "Cairns": "Australia", "Carrara": "Australia", "Darwin": "Australia",
    "Geelong": "Australia", "Victoria": "Australia", "Hobart": "Australia",
    "Melbourne": "Australia", "Perth": "Australia", "Sydney": "Australia",
    "Townsville": "Australia",
    "Auckland": "New Zealand", "Christchurch": "New Zealand", "Dunedin": "New Zealand",
    "Hamilton": "New Zealand", "Mount Maunganui": "New Zealand", "Napier": "New Zealand",
    "Nelson": "New Zealand", "Queenstown": "New Zealand", "Wellington": "New Zealand",
    "Port Moresby": "Papua New Guinea", "Apia": "Samoa", "Port Vila": "Vanuatu",
    # Southern / East / West Africa
    "Bloemfontein": "South Africa", "Benoni": "South Africa", "Cape Town": "South Africa",
    "Centurion": "South Africa", "Durban": "South Africa", "East London": "South Africa",
    "Gqeberha": "South Africa", "Port Elizabeth": "South Africa",
    "Johannesburg": "South Africa", "Kimberley": "South Africa", "Paarl": "South Africa",
    "Potchefstroom": "South Africa",
    "Harare": "Zimbabwe", "Bulawayo": "Zimbabwe",
    "Windhoek": "Namibia", "Gaborone": "Botswana",
    "Nairobi": "Kenya", "Mombasa": "Kenya",
    "Entebbe": "Uganda", "Jinja": "Uganda", "Kampala": "Uganda",
    "Dar-es-Salaam": "Tanzania", "Kigali City": "Rwanda",
    "Abuja": "Nigeria", "Lagos": "Nigeria", "Accra": "Ghana", "Blantyre": "Malawi",
    # Caribbean (West Indies is a composite team — see module docstring)
    "Antigua": "Antigua and Barbuda", "Coolidge": "Antigua and Barbuda",
    "Barbados": "Barbados", "Bridgetown": "Barbados",
    "Basseterre": "Saint Kitts and Nevis", "St Kitts": "Saint Kitts and Nevis",
    "Dominica": "Dominica", "Roseau": "Dominica",
    "Gros Islet": "Saint Lucia", "St Lucia": "Saint Lucia", "North Sound": "Antigua and Barbuda",
    "Guyana": "Guyana", "Providence": "Guyana",
    "Jamaica": "Jamaica", "Kingston": "Jamaica",
    "St Vincent": "Saint Vincent and the Grenadines", "Kingstown": "Saint Vincent and the Grenadines",
    "Trinidad": "Trinidad and Tobago", "Tarouba": "Trinidad and Tobago",
    "St George's": "Grenada",
    "George Town": "Cayman Islands",
    # North America
    "Lauderhill": "United States of America", "Dallas": "United States of America",
    "Houston": "United States of America", "New York": "United States of America",
    "Naucalpan": "Mexico", "King City": "Canada",
    # Europe
    "London": "England", "Manchester": "England", "Birmingham": "England",
    "Nottingham": "England", "Southampton": "England", "Cardiff": "England",
    "Leeds": "England", "Chester-le-Street": "England", "Bristol": "England",
    "Taunton": "England",
    "Edinburgh": "Scotland", "Glasgow": "Scotland",
    "Belfast": "Ireland", "Derry": "Ireland", "Londonderry": "Ireland",
    "Bready": "Ireland", "Dublin": "Ireland",
    "St Peter Port": "Guernsey", "Port  Soif": "Guernsey", "Castel": "Guernsey",
    "St Saviour": "Jersey",
    "Gibraltar": "Gibraltar",
    "Rotterdam": "Netherlands", "Deventer": "Netherlands", "Utrecht": "Netherlands",
    "The Hague": "Netherlands", "Amstelveen": "Netherlands",
    "Stockholm": "Sweden", "Koge": "Denmark", "Brondby": "Denmark",
    "Copenhagen": "Denmark", "Ishoj": "Denmark",
    "Oslo": "Norway", "Tallinn": "Estonia", "Kerava": "Finland", "Vantaa": "Finland",
    "Szodliget": "Hungary", "Sofia": "Bulgaria", "Zagreb": "Croatia",
    "Belgrade": "Serbia", "Prague": "Czech Republic", "Graz": "Austria",
    "Latschach": "Austria", "Krefeld": "Germany",
    "Ghent": "Belgium", "Waterloo": "Belgium", "Zemst": "Belgium",
    "Walferdange": "Luxembourg", "Almeria": "Spain", "Murcia": "Spain",
    "Rome": "Italy", "Spinaceto": "Italy", "Ilfov County": "Romania", "Dreux": "France",
    "Albergaria": "Portugal",
    # Asia (other)
    "Kuala Lumpur": "Malaysia", "Bangi": "Malaysia", "Singapore": "Singapore",
    "Bangkok": "Thailand", "Hong Kong": "Hong Kong", "Mong Kok": "Hong Kong",
    "Incheon": "South Korea", "Nisshin": "Japan", "Sano": "Japan",
    "Bali": "Indonesia", "Hangzhou": "China",
    # Americas (other)
    "Buenos Aires": "Argentina", "Bogota": "Colombia", "Guacima": "Costa Rica",
    "Panama City": "Panama", "Seropedica": "Brazil",
    # Southern/Eastern Africa (islands)
    "Episkopi": "Cyprus", "Malkerns": "Eswatini",
}

# Grounds that never carry a resolvable city anywhere in this dataset.
GROUND_OVERRIDES: dict[str, str] = {
    "Mombasa Sports Club Ground": "Kenya",
    "St Georges Quilmes": "Argentina",
    "Sylhet Stadium": "Bangladesh",
}


def _representative_city(cities: pd.Series) -> str | None:
    non_empty = cities[cities.notna() & (cities != "")]
    if non_empty.empty:
        return None
    return non_empty.mode().iloc[0]


def _resolve_country(venue_clean: str, representative_city: str | None) -> tuple[str | None, bool]:
    if representative_city and representative_city in CITY_ALIASES:
        return CITY_ALIASES[representative_city], False
    if venue_clean in GROUND_OVERRIDES:
        return GROUND_OVERRIDES[venue_clean], False
    return None, True


def build_venue_country_mapping(matches: pd.DataFrame) -> pd.DataFrame:
    """Build a venue -> country review table from a match table.

    Parameters
    ----------
    matches:
        Must have ``venue`` and ``city`` columns, one row per match (e.g.
        the output of ``src.eda.unit_summary.build_match_table``, already
        filtered to the scope you want the mapping built over).

    Returns
    -------
    One row per distinct raw ``venue`` string, with columns:
    ``venue_raw, venue_clean, venue_name_needs_review, representative_city,
    country, country_needs_review``.
    """
    raw_venues = matches["venue"].dropna().unique()
    venue_name_mapping = build_venue_mapping(raw_venues)

    raw_to_clean = dict(zip(venue_name_mapping["venue_raw"], venue_name_mapping["venue_clean"]))
    working = matches[["venue", "city"]].dropna(subset=["venue"]).copy()
    working["venue_clean"] = working["venue"].map(raw_to_clean)

    city_by_clean = working.groupby("venue_clean")["city"].apply(_representative_city)

    rows: list[dict[str, Any]] = []
    for _, row in venue_name_mapping.iterrows():
        venue_clean = row["venue_clean"]
        representative_city = city_by_clean.get(venue_clean)
        country, country_needs_review = _resolve_country(venue_clean, representative_city)
        rows.append(
            {
                "venue_raw": row["venue_raw"],
                "venue_clean": venue_clean,
                "venue_name_needs_review": row["needs_review"],
                "representative_city": representative_city,
                "country": country,
                "country_needs_review": country_needs_review,
            }
        )

    return pd.DataFrame(rows).sort_values(["country_needs_review", "venue_clean", "venue_raw"], ascending=[False, True, True]).reset_index(drop=True)
