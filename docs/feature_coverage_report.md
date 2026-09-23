# Feature coverage report

Generated from `data/processed/train_1st_innings.parquet`: 393,379 rows across 3,331 matches (2005-02-17 .. 2026-09-09).

## % missing per feature (overall)

|                                    |   pct_missing |
|:-----------------------------------|--------------:|
| striker_career_avg_all_t20         |        100    |
| striker_career_sr_all_t20          |        100    |
| non_striker_career_sr_all_t20      |        100    |
| non_striker_career_avg_all_t20     |        100    |
| non_striker_career_avg_t20i        |          7.93 |
| striker_career_avg_t20i            |          7.82 |
| non_striker_strike_rate            |          6.58 |
| non_striker_career_sr_t20i         |          6.27 |
| striker_career_sr_t20i             |          6.25 |
| remaining_batting_order_avg        |          5.07 |
| remaining_batting_order_sr         |          4.01 |
| remaining_bowling_economy_weighted |          1.94 |
| tier_pairing                       |          0.15 |
| bowling_team_tier                  |          0.15 |
| bowling_team                       |          0.15 |
| team_bowling_recent_form           |          0.03 |
| venue_avg_first_innings_score      |          0.03 |
| team_batting_recent_form           |          0.03 |

## Rows and matches per season (by tier pairing)

| season   |   n_rows |   n_matches |   associate_vs_associate |   associate_vs_full |   full_vs_full | post_2019   |
|:---------|---------:|------------:|-------------------------:|--------------------:|---------------:|:------------|
| 2004/05  |      120 |           1 |                        0 |                   0 |              1 | False       |
| 2005     |      120 |           1 |                        0 |                   0 |              1 | False       |
| 2005/06  |      477 |           4 |                        0 |                   0 |              4 | False       |
| 2006     |      240 |           2 |                        0 |                   0 |              2 | False       |
| 2006/07  |      470 |           4 |                        0 |                   0 |              4 | False       |
| 2007     |      240 |           2 |                        0 |                   0 |              2 | False       |
| 2007/08  |     4398 |          37 |                        0 |                   5 |             32 | False       |
| 2008     |      464 |           4 |                        2 |                   1 |              1 | False       |
| 2008/09  |     1555 |          13 |                        0 |                   0 |             13 | False       |
| 2009     |     3709 |          31 |                        0 |                   3 |             28 | False       |
| 2009/10  |     2256 |          19 |                        2 |                   2 |             15 | False       |
| 2010     |     4143 |          35 |                        0 |                   0 |             35 | False       |
| 2010/11  |     1319 |          11 |                        0 |                   0 |             11 | False       |
| 2011     |     1196 |          10 |                        0 |                   0 |             10 | False       |
| 2011/12  |     3467 |          29 |                        2 |                   7 |             20 | False       |
| 2012     |     1677 |          14 |                        0 |                   2 |             12 | False       |
| 2012/13  |     5025 |          42 |                        2 |                   0 |             40 | False       |
| 2013     |     1552 |          13 |                        2 |                   0 |             11 | False       |
| 2013/14  |     7116 |          60 |                        7 |                  11 |             42 | False       |
| 2014     |      360 |           3 |                        0 |                   0 |              3 | False       |
| 2014/15  |     1200 |          10 |                        1 |                   0 |              9 | False       |
| 2015     |     3071 |          26 |                        8 |                   5 |             13 | False       |
| 2015/16  |     9302 |          78 |                       11 |                  12 |             55 | False       |
| 2016     |     1198 |          10 |                        0 |                   1 |              9 | False       |
| 2016/17  |     2987 |          25 |                        6 |                   2 |             17 | False       |
| 2017     |     1550 |          13 |                        3 |                   0 |             10 | False       |
| 2017/18  |     4429 |          37 |                        0 |                   3 |             34 | False       |
| 2018     |     3458 |          29 |                        2 |                   7 |             20 | False       |
| 2018/19  |     5341 |          45 |                       10 |                   4 |             31 | False       |
| 2019     |     7850 |          66 |                       53 |                   4 |              9 | True        |
| 2019/20  |    16298 |         137 |                       83 |                  15 |             39 | True        |
| 2020     |     1080 |           9 |                        4 |                   0 |              5 | True        |
| 2020/21  |     3807 |          32 |                        2 |                   0 |             30 | True        |
| 2021     |    13560 |         114 |                       60 |                   3 |             51 | True        |
| 2021/22  |    22792 |         194 |                      120 |                  26 |             48 | True        |
| 2022     |    27043 |         228 |                      163 |                  18 |             47 | True        |
| 2022/23  |    20482 |         175 |                      105 |                  11 |             58 | True        |
| 2023     |    20127 |         171 |                      144 |                  11 |             16 | True        |
| 2023/24  |    25169 |         213 |                      162 |                  16 |             35 | True        |
| 2024     |    36856 |         316 |                      241 |                  29 |             45 | True        |
| 2024/25  |    24103 |         204 |                      158 |                   5 |             41 | True        |
| 2025     |    34725 |         294 |                      234 |                  16 |             44 | True        |
| 2025/26  |    33098 |         280 |                      173 |                  34 |             71 | True        |
| 2026     |    33949 |         290 |                      262 |                   4 |             23 | True        |

Post-2019 matches: 2,723/3,331 (81.7%); of those, 2,156 involve at least one associate side (tier_pairing containing 'associate').

## Venues / teams with < 5 point-in-time prior matches

- **n_venues_total**: 334
- **n_venues_never_reach_threshold**: 151
- **pct_rows_venue_lt_threshold**: 34.3
- **n_teams_total**: 109
- **n_teams_never_reach_threshold**: 8
- **pct_rows_team_form_lt_threshold**: 21.4

## Player-rating coverage

- Version (a), T20I-only history — computed:
  - pct_rows_striker_has_prior_t20i_innings: 93.8%
  - pct_rows_either_batter_has_prior_t20i_innings: 97.6%
- Version (b), T20I + franchise leagues — **not computable**: `data/t20s_json` is international-only (see docs/feature_audit.md); coverage is 0% until franchise-league Cricsheet data is added.

## Correlation with target, at over 6 / 10 / 15

|                                    |   after_over_6 |   after_over_10 |   after_over_15 |
|:-----------------------------------|---------------:|----------------:|----------------:|
| run_rate                           |          0.727 |           0.839 |           0.936 |
| score                              |          0.727 |           0.839 |           0.936 |
| run_rate_last_5                    |          0.706 |           0.744 |           0.754 |
| fours                              |          0.542 |           0.616 |           0.695 |
| sixes                              |          0.464 |           0.577 |           0.685 |
| run_rate_last_3                    |          0.626 |           0.665 |           0.661 |
| wickets                            |         -0.453 |          -0.57  |          -0.6   |
| remaining_batters_count            |          0.447 |           0.566 |           0.599 |
| striker_strike_rate                |          0.45  |           0.512 |           0.46  |
| non_striker_strike_rate            |          0.435 |           0.503 |           0.505 |
| striker_runs                       |          0.464 |           0.47  |           0.385 |
| partnership_runs                   |          0.461 |           0.433 |           0.341 |
| non_striker_runs                   |          0.4   |           0.432 |           0.356 |
| remaining_batting_order_sr         |          0.429 |           0.42  |           0.356 |
| wickets_last_5                     |         -0.412 |          -0.403 |          -0.266 |
| non_striker_career_sr_t20i         |          0.368 |           0.377 |           0.341 |
| remaining_batting_order_avg        |          0.364 |           0.373 |           0.34  |
| striker_career_sr_t20i             |          0.372 |           0.37  |           0.363 |
| team_batting_recent_form           |          0.37  |           0.365 |           0.35  |
| striker_career_avg_t20i            |          0.259 |           0.298 |           0.326 |
| elo_batting                        |          0.323 |           0.321 |           0.308 |
| elo_n_batting                      |          0.319 |           0.318 |           0.312 |
| wickets_last_3                     |         -0.313 |          -0.304 |          -0.185 |
| team_batting_recent_form_n         |          0.309 |           0.308 |           0.302 |
| non_striker_career_avg_t20i        |          0.258 |           0.286 |           0.3   |
| partnership_balls                  |          0.284 |           0.28  |           0.177 |
| elo_diff                           |          0.267 |           0.265 |           0.259 |
| striker_balls                      |          0.259 |           0.256 |           0.199 |
| striker_career_innings_t20i        |          0.211 |           0.235 |           0.257 |
| non_striker_balls                  |          0.198 |           0.251 |           0.16  |
| non_striker_career_innings_t20i    |          0.191 |           0.23  |           0.247 |
| team_bowling_recent_form           |          0.246 |           0.243 |           0.243 |
| remaining_bowling_economy_weighted |          0.244 |           0.231 |           0.212 |
| venue_avg_first_innings_score      |          0.221 |           0.216 |           0.204 |
| bowlers_with_overs_remaining       |         -0.015 |           0.117 |           0.166 |
| team_bowling_recent_form_n         |          0.139 |           0.138 |           0.144 |
| elo_n_bowling                      |          0.135 |           0.134 |           0.141 |
| extras_total                       |          0.064 |           0.081 |           0.132 |
| extras_noballs                     |          0.074 |           0.092 |           0.13  |
| extras_wides                       |          0.041 |           0.057 |           0.118 |
| extras_penalty                     |        nan     |           0.066 |           0.06  |
| extras_legbyes                     |          0.035 |           0.028 |           0.027 |
| venue_n_prior_matches              |         -0.03  |          -0.031 |          -0.034 |
| extras_byes                        |          0.009 |           0.023 |           0.016 |
| dls_affected                       |          0.012 |           0.011 |           0.016 |
| elo_bowling                        |          0.013 |           0.013 |           0.007 |
| legal_balls_bowled                 |        nan     |         nan     |         nan     |
| balls_remaining                    |        nan     |         nan     |         nan     |
| is_reduced_overs                   |        nan     |         nan     |         nan     |
| is_curtailed_first_innings         |        nan     |         nan     |         nan     |

`NaN` rows (`legal_balls_bowled`, `balls_remaining`, `is_reduced_overs`, `is_curtailed_first_innings`) are columns with zero variance at a fixed over-checkpoint in the default (non-flagged) table — correlation is undefined, not missing data.

## 1st-innings score distribution by tier pairing (one row per match)

| tier_pairing           |   count |   mean |   std |   min |   10% |   25% |   50% |   75% |   90% |   max |
|:-----------------------|--------:|-------:|------:|------:|------:|------:|------:|------:|------:|------:|
| associate_vs_associate |    2022 |  143.1 |  44.7 |    10 |    87 |   113 |   143 | 171.8 | 199.9 |   322 |
| associate_vs_full      |     257 |  153.7 |  43.4 |    39 |   104 |   128 |   154 | 180   | 204.4 |   344 |
| full_vs_full           |    1047 |  164.5 |  35.1 |    55 |   122 |   141 |   165 | 187.5 | 207   |   304 |

