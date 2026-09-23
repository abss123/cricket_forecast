"""Build the match table and print team/venue/player unit summaries.

Usage:
    python scripts/run_unit_summary.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.eda.unit_summary import build_match_table, build_player_directory, summarize_units

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "t20s_json"

# Per docs/schema_notes.md, the raw download is already T20 + international-only;
# the one meaningful subset choice left is gender.
FILTERS = {"match_type": "T20", "gender": "male", "team_type": "international"}


def main() -> None:
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)

    matches = build_match_table(DATA_DIR)
    print(f"match table: {len(matches):,} rows\n")

    player_directory = build_player_directory(DATA_DIR)

    for unit in ("team", "venue", "player"):
        kwargs = {"player_directory": player_directory} if unit == "player" else {}
        summary, header = summarize_units(matches, unit=unit, filters=FILTERS, **kwargs)
        print(f"=== {unit} summary (filters={FILTERS}) ===")
        print("header:", header)
        print(summary.sort_values("n_matches", ascending=False).head(10).to_string(index=False))
        print()


if __name__ == "__main__":
    main()
