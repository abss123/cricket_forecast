"""The written 1st-innings checkpoint splits are disjoint by match and respect their date ranges."""

import pytest

from src.experiments.checkpoints import SPLIT_DIR, SPLIT_RANGES, in_split_range, load_splits


@pytest.mark.skipif(not (SPLIT_DIR / "train.parquet").exists(), reason="run scripts/build_first_innings_checkpoint.py first")
def test_splits_are_disjoint_and_within_date_ranges():
    splits = dict(zip(SPLIT_RANGES, load_splits()))
    ids = {name: set(df["match_id"]) for name, df in splits.items()}
    assert not ids["train"] & ids["val"]
    assert not ids["train"] & ids["test"]
    assert not ids["val"] & ids["test"]
    for name, df in splits.items():
        assert in_split_range(df["date"], name).all(), f"{name} has rows outside {SPLIT_RANGES[name]}"
