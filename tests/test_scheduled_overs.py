"""Men's T20Is are featurized as 20 overs with 4 overs per bowler, even when Cricsheet records info.overs = 50."""

import pandas as pd
import pytest

from scripts.build_training_table import DATA_DIR, _flatten, _load_match, _non_super_over_innings, build_match_info
from src.experiments.checkpoints import CHECKPOINT_BALLS, SPLIT_DIR, _prefix_for_legal_count, load_splits
from src.features.history import PointInTimeHistory
from src.features.state import bowler_overs_remaining, compute_features

needs_checkpoint = pytest.mark.skipif(
    not (SPLIT_DIR / "train.parquet").exists() or not DATA_DIR.exists(),
    reason="needs data/t20s_json and scripts/build_first_innings_checkpoint.py output",
)


def _match_info(match, match_id="0"):
    teams = match["info"]["teams"]
    batting = match["innings"][0]["team"] if match.get("innings") else teams[0]
    meta = pd.Series({
        "match_id": match_id, "date": pd.Timestamp("2025-01-01"), "season": "2025", "venue_clean": "X",
        "team_batting_first": batting, "team_batting_second": next(t for t in teams if t != batting),
    })
    return build_match_info(match, meta, None)


def test_mislabelled_50_over_t20i_uses_20_overs():
    match = {"info": {
        "match_type": "T20", "gender": "male", "team_type": "international", "overs": 50, "balls_per_over": 6,
        "teams": ["A", "B"], "players": {"A": ["a1"], "B": ["b1", "b2"]},
    }}
    info = _match_info(match)
    assert info["scheduled_overs"] == 20
    assert compute_features([], info, PointInTimeHistory())["balls_remaining"] == 120
    assert bowler_overs_remaining([], info) == {"b1": 4.0, "b2": 4.0}
    six_balls = [{"bowler": "b1", "batter": "a1", "non_striker": "a1", "runs": {"total": 0}}] * 6
    assert bowler_overs_remaining(six_balls, info) == {"b1": 3.0, "b2": 4.0}


@needs_checkpoint
def test_checkpoint_balls_remaining():
    df = pd.concat(load_splits())
    expected = 120 - df["checkpoint"].map(CHECKPOINT_BALLS)
    assert (df.loc[df["checkpoint"] == "ball1", "balls_remaining"] == 120).all()
    assert (df["balls_remaining"] == expected).all(), df.loc[df["balls_remaining"] != expected, ["match_id", "checkpoint"]]


@needs_checkpoint
def test_checkpoint_bowlers_have_at_most_4_overs_remaining():
    worst = {}
    for match_id in pd.concat(load_splits())["match_id"].unique():
        match = _load_match(DATA_DIR, match_id)
        info = _match_info(match, match_id)
        deliveries = _flatten(_non_super_over_innings(match)[0])
        for n_balls in CHECKPOINT_BALLS.values():
            prefix = _prefix_for_legal_count(deliveries, n_balls)
            if prefix is not None:
                remaining = bowler_overs_remaining(prefix, info)
                worst[match_id] = max(worst.get(match_id, 0.0), *remaining.values(), 0.0)
    over = {m: v for m, v in worst.items() if v > 4}
    assert not over, f"bowlers with > 4 overs remaining: {over}"
