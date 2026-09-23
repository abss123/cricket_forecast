# Feature audit — 1st-innings T20I score forecasting

Scope: forecast the **final 1st-innings score** of a men's T20 international, at any ball, from `data/t20s_json`. This audit covers every feature in the spec: whether it's derivable from Cricsheet, how, and its edge cases. Implementation lives in `src/features/` (`state.py::compute_features` is the single function used for both training replay and live inference); the training table is built by `scripts/build_training_table.py`.

## 0. Scope confirmation

Checked empirically against all 5,700 files in `data/t20s_json` (not assumed):

| Check | Result |
|---|---|
| Distinct `match_type` values | **`"T20"` only** — 5,700/5,700 files. There is no `"IT20"` value in this download; Cricsheet's `t20s_json.zip` archive is already T20-international-only, so the "T20 vs IT20" distinction the brief warned about doesn't arise here. |
| Distinct `team_type` values | `"international"` only — 100%. This download contains **no franchise-league data** (no IPL/BBL/PSL/CPL/T20 Blast/etc. anywhere in the repo). This directly limits player-rating version (b) — see §4. |
| `gender` breakdown | male: 3,539 matches; female: 2,161 |
| **Training scope** (`gender=="male" & team_type=="international"`) | **3,539 matches**, all `match_type=="T20"` |
| Legal 1st-innings deliveries in scope | 409,796 (raw); 393,379 after excluding reduced-overs/curtailed matches (see §3) |
| Distinct raw venue strings in scope | 336 (334 after canonicalization) |
| Distinct teams in scope | 109 (Afghanistan appears 0 times — Cricsheet withholds all Afghanistan matches per `docs/schema_notes.md`) |
| Distinct seasons | 44 |
| Matches with `overs != 20` | 8 |
| 1st innings ending early without being all-out (curtailment candidates) | 207 |

## 1. Ball-state features (`src/features/state.py::_ball_state_features`)

All derivable directly from `deliveries_so_far` (the current innings' Cricsheet delivery list up to and including the ball being featurized). "Legal" = not a wide and not a no-ball (`is_legal_delivery`), matching the convention already used in `code/cricket_tables.py`.

| Feature | Derivation | Edge cases |
|---|---|---|
| `balls_remaining` | `scheduled_overs * balls_per_over - legal_balls_bowled` | Uses the match's own `info.overs`/`info.balls_per_over`, not a hardcoded 20/6 — correct for the 8 reduced-overs matches too (though those are excluded from training rows by default, see §3). |
| `score` | Sum of `runs.total` over all deliveries so far | Includes extras (byes/legbyes/wides/noballs), which is correct for "final score" framing. |
| `wickets` | Count of dismissal-type wickets, excluding `retired hurt`/`retired not out` | Reuses the exclusion rule already established in `code/cricket_tables.py::_is_dismissal`; `retired out` **is** counted (genuine dismissal). A same-ball double dismissal (rare, e.g. run-out off a no-ball) correctly counts as 2. |
| `run_rate` | `score / (legal_balls_bowled / balls_per_over)` | `None` at ball 0 (no overs bowled yet — avoids a divide-by-zero silently becoming 0). |
| `run_rate_last_n` (n=3,5) | Run rate over the trailing window of the last n\*6 legal balls | **Simplification, documented in code**: if fewer than n overs have been bowled, the window truncates at the start of the innings rather than returning `None` — so `run_rate_last_5` at ball 12 is really "run rate over the 2 overs bowled so far." Flagged as a modeling choice, not a bug. |
| `wickets_last_n` | Same trailing window, wicket count | Same truncation behavior. |
| `phase` | `powerplay` (overs 1–6) / `middle` (7–15) / `death` (16–20), from the over the *current* ball belongs to | Computed from `legal_balls_bowled`, so it's correct even mid-over. Not adjusted for the 8 reduced-overs matches (those still use the standard 6/15/20 boundaries) — a genuine simplification worth revisiting if reduced-overs matches are ever included via `--include-flagged`. |
| `partnership_runs` / `partnership_balls` | Runs/legal-balls since the most recent dismissal (or innings start if 0 wickets) | A same-ball double-wicket delivery resets the partnership window to *after* that ball, same as normal. |
| `striker_runs`/`balls`/`strike_rate`, `non_striker_*` | The striker/non-striker are read off the **last delivery's** `batter`/`non_striker` fields (Cricsheet doesn't give a separate "who's currently on strike" field) | This is "who faced the last completed ball," not "who will face the next ball" — those differ only when strike rotates (odd runs, end of over), which isn't knowable without seeing the next ball. Standard simplification for ball-by-ball state features. |
| `fours`, `sixes` | Count of legal deliveries with `runs.batter` == 4 or 6 | Byes/leg-byes that happen to total 4 aren't counted (not off the bat) — deliberate. |
| `extras_{wides,noballs,byes,legbyes,penalty}`, `extras_total` | Summed straight from each delivery's `extras` dict | Cricsheet only includes an extras key when it's nonzero (schema_notes.md), so missing keys are treated as 0 — correct. |

## 2. Player features, point-in-time (`src/features/state.py::_player_features`, `src/features/history.py::PlayerCareerStore`)

"Point-in-time" means: only matches strictly before the current match's date (see §5, the date-grouped replay design) ever contribute to these numbers — enforced by construction, and checked by `tests/test_state_leakage.py`.

| Feature | Derivation | Edge cases |
|---|---|---|
| Striker/non-striker career SR & average | `PlayerCareerStore`, keyed by **`registry` player ID, never raw name** | Per `docs/schema_notes.md` §3.6, 69 display names map to >1 player ID and 17 IDs map to >1 display name — joining on ID (as `cricket_tables.py` and `unit_summary.py` already do) avoids both collision directions. `career_average` is `None` when `dismissals==0` (not-out every innings so far) rather than a fabricated "undefined ÷ 0". |
| Strength of remaining batting order | Mean career SR/avg (T20I version) of names in `match_info["batting_players"]` that haven't yet appeared as a batter/non-striker in `deliveries_so_far` | **Known limitation**: Cricsheet's `info.players[team]` list order is the squad list, not a guaranteed batting order — a promoted pinch-hitter or a demoted tail-ender won't be reflected. Players with no career history yet are silently excluded from the mean rather than treated as 0 (so `remaining_batting_order_sr` can be `None` if literally none of the remaining names have prior T20I innings — rare, but happens for associate-nation debut XIs). |
| Per-bowler overs remaining / economy of remaining overs | For each name in `match_info["bowling_players"]`, `overs_remaining = max(scheduled_overs/5 - overs_bowled_so_far, 0)`; `remaining_bowling_economy_weighted` = economy-weighted mean over bowlers who still have overs left | `overs/5` (e.g. 4 overs in a 20-over match) is the standard T20 per-bowler cap; a bowler already at their cap contributes 0 weight. Bowlers with no career economy on file are counted in `bowlers_with_overs_remaining` but excluded from the weighted economy average (not treated as league-average, to avoid fabricating a number) — this is why `remaining_bowling_economy_weighted` is `None` for ~2% of rows (early-tournament debut sides where *no* remaining bowler has a prior economy figure). |

### Two player-rating corpora (a) vs (b)

The spec asks for two versions: **(a)** T20I history only, **(b)** all men's T20 history (T20I + franchise leagues), point-in-time, with training rows staying T20I 1st-innings-only regardless.

- **(a) is fully implemented**: `PointInTimeHistory.player_career_t20i`, fed from the same male-T20I match stream used for training. Coverage (`docs/feature_coverage_report.md`): 93.8% of rows have a striker with ≥1 prior T20I innings; 97.6% have at least one of the two batters with prior T20I history. The ~6% gap is mostly the very first T20I appearance of a player's career, which no amount of point-in-time data can fill.
- **(b) is not computable today**: as confirmed in §0, `data/t20s_json` contains **zero franchise-league matches** — there is no IPL/BBL/PSL/CPL/T20 Blast/etc. data anywhere in this repo. Building it would require downloading additional Cricsheet archives. Per your direction, the code is built so this is a **swappable data source**: `PointInTimeHistory.player_career_all_t20` is a second, independent `PlayerCareerStore` slot (currently `None`); `compute_features` already emits `*_career_sr_all_t20`/`*_career_avg_all_t20` columns and reports them as 100% missing (`docs/feature_coverage_report.md`) rather than silently reusing the T20I-only numbers. Once franchise-league JSON is added, populating that slot (feeding it the same way `player_career_t20i` is fed, just from a larger match stream) requires no change to `compute_features` or the training rows.

## 3. Team/context features (`src/features/state.py::_team_context_features`, `history.py`)

| Feature | Derivation | Edge cases |
|---|---|---|
| `team_batting_recent_form`, `team_bowling_recent_form` | Exponentially-decayed (halflife=5 matches) point-in-time mean of the team's own 1st-innings runs (batting) / runs conceded (bowling), then **shrunk toward the point-in-time global mean** with `k_team=10` | Replaces the removed head-to-head features per your instruction — this is decay + shrinkage on a team's *own* recent scoring, not opponent-specific. |
| `elo_batting`, `elo_bowling`, `elo_diff` | Standard logistic Elo (`TeamEloStore`, K=24, initial rating 1500) | Updates only on decisive results (`chase_won`/`defend_won`/outright `tie`); `no result`/`Awarded`/`bowl_out`-decided matches (see `docs/schema_notes.md` §3.1) leave ratings unchanged rather than guessing a winner. Multiple same-day matches for one team use that team's pre-day rating for both (a known, documented approximation — true simultaneous-round Elo is rarely used in practice and the effect is negligible at T20I scheduling density). |
| `venue_avg_first_innings_score` | Point-in-time mean 1st-innings score at the canonical venue, shrunk toward the point-in-time global mean with **`k_venue=15`** (stronger shrinkage than the team-level `k=10`, per your instruction) | Canonicalization reuses `src/eda/unit_summary.py::build_venue_mapping` (bare-name/city-suffix grouping). Coverage report: 151/334 venues (45%) never accumulate 5 prior matches even by their last appearance in the dataset; 34.3% of *rows* are computed with a venue below that threshold — this is exactly why the stronger shrinkage matters. |
| `home_away` | `"home"`/`"away"`/`"neutral"` from the batting team's perspective, via `venue → country` (§3a) compared against each team's home countries | Every venue in scope now resolves to a country (0 rows flagged `needs_review` in `docs/venue_country_mapping.csv`), so `home_away` has 0% missing. |
| `batting_team_tier`, `bowling_team_tier`, `tier_pairing` | Static ICC full-member set (`src/features/team_tiers.py`) — see §3b | 0.15% of rows have a missing team (a handful of scraped-team-name edge cases in `build_match_table`); tier fields are `None` there rather than guessed. |
| `toss_decision`, `season` | Passed straight through from `info.toss.decision` / `info.season` | `season` normalized to `str` per the existing `_as_str` convention in `unit_summary.py` (mixed `"2019/20"` string vs. bare-int seasons). |

### 3a. Venue → country (for `home_away`)

Built by `src/features/venue_country.py::build_venue_country_mapping`, written to **`docs/venue_country_mapping.csv`** for review (352 raw venue/city pairs inspected by hand for this dataset's actual scope; all 334 canonical venues currently resolve, 0 flagged `country_needs_review` — re-run `scripts/build_training_table.py` after any future data refresh to confirm that still holds for any newly-appearing venue). Method: canonicalize the venue (reusing `build_venue_mapping`), take the most frequent recorded city for that venue, resolve city → country via a hand-built alias table, with a 3-entry ground-level override for venues that never carry a usable city (`Mombasa Sports Club Ground`, `St Georges Quilmes`, `Sylhet Stadium`). Anything that fails to resolve is left `None` + `needs_review=True` rather than guessed — review the CSV after any data refresh in case new venues appear.

Deliberate departures from strict sovereign-state geography (documented in the module docstring, called out here because they directly affect `home_away` correctness):
- Northern Ireland grounds (Belfast, Derry/Londonderry, Bready) → `"Ireland"` (Cricket Ireland fields one team for the whole island).
- Welsh grounds (Cardiff) → `"England"` (no separate Wales team).
- **West Indies** is a composite of ten Caribbean nations/territories — `venue_country` resolves each ground to its actual country (e.g. Sabina Park → `"Jamaica"`), and `team_tiers.TEAM_HOME_COUNTRIES["West Indies"]` is the only team with a multi-country home set, so a West Indies match at any of those ten venues still correctly resolves to `"home"`.

### 3b. Team tier

Static 12-member ICC full-member set in `src/features/team_tiers.py`. Afghanistan is included for correctness even though it has 0 matches in this download. `"ICC World XI"` (4 matches, an exhibition side, not a national board) is tagged `"associate"` with a documented caveat — it isn't a real tier, but excluding it from statistics is a modeling call left to you.

**1st-innings score distribution by tier pairing** (see `docs/feature_coverage_report.md` for the full table): full-vs-full averages 164.5 runs (n=1,047 matches), associate-vs-full 153.7 (n=257), associate-vs-associate 143.1 (n=2,022) — the ordering is intuitive, and the associate-vs-associate bucket is by far the largest, which is the post-2019 volume growth flagged next.

### 3c. Matches per season / post-2019 associate volume

Full season-by-season, tier-pairing-cross-tabbed breakdown is in `docs/feature_coverage_report.md`. Headline: **2,723 of 3,331 matches in the final training table (81.7%) are from 2019 onward**, and of those, 2,156 involve at least one associate side. In other words, the large majority of this dataset's *volume* is recent associate-nation cricket (driven by ICC's post-2018 T20I-status expansion to all member boards) — worth keeping in mind for any train/test split or recency weighting decision.

## 4. Rain-affected 1st innings (exclusion/flagging)

Two boolean columns, computed at the match level in `scripts/build_training_table.py` before ball-level replay (they require knowing the whole innings, so they can't be point-in-time ball features):

- `is_reduced_overs`: `info.overs != 20` (agreed in advance, e.g. a rain delay before the toss) — 8 matches.
- `is_curtailed_first_innings`: the 1st innings ended before the scheduled ball count **and** the batting side wasn't all out — 207 matches. This is a proxy (Cricsheet has no explicit "curtailed" flag); it will also catch the very rare non-weather case of an innings ending early for an unrelated administrative reason, which is an acceptable false-positive rate for a v1 exclusion rule.

Both are **excluded from the saved training table by default** (`scripts/build_training_table.py`, opt back in with `--include-flagged`) and kept as columns either way, per your "exclude or flag" instruction. Note this is orthogonal to `dls_affected` (also carried through as a column): D/L only ever revises the **2nd-innings chase target**, never the 1st-innings total itself, so a `dls_affected` match's 1st innings is still valid training data unless it's *also* flagged reduced/curtailed.

## 5. Point-in-time replay design (leakage prevention)

`scripts/build_training_table.py::replay_matches` processes matches in **date-grouped** chronological order: every match sharing a calendar date has its features computed against the history frozen as of the *previous* date, and only after every match in that date group has been featurized are all of that day's results applied to the stores together. This is what makes `tests/test_state_leakage.py`'s property hold even for the rare same-day tripleheader in the T20I calendar — a per-match (not per-date-group) processing order would leak one same-day match's result into another's features. See the leakage test for both a fast synthetic proof of this mechanism and a real-data integration check.

## 6. Not in Cricsheet

- **Weather** (temperature, humidity, precipitation) — not present anywhere in the schema. Would need an external join keyed on `(venue, date)`, e.g. a historical weather API — no such source is wired up.
- **Day/night** — not an explicit field. `info.dates` gives the calendar date only, no start time. A very rough proxy might be inferable from `officials`/`event` metadata for some tournaments, but nothing in the schema reliably distinguishes a day match from a day/night match — flagged as a genuine external-join candidate, not attempted here.

## 7. Where everything is

| Deliverable | File |
|---|---|
| This audit | `docs/feature_audit.md` |
| Venue → country review table | `docs/venue_country_mapping.csv` |
| Feature function (training + live) | `src/features/state.py::compute_features` |
| Point-in-time stores | `src/features/history.py` |
| Team tier / home-away | `src/features/team_tiers.py` |
| Venue → country builder | `src/features/venue_country.py` |
| Training table builder | `scripts/build_training_table.py` → `data/processed/train_1st_innings.parquet` |
| Leakage test | `tests/test_state_leakage.py` |
| Coverage/correlation report | `scripts/report_feature_coverage.py` → `docs/feature_coverage_report.md` |
