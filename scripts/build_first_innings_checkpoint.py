"""Write the frozen 1st-innings checkpoint splits to data/checkpoints/first_innings/.

Usage:
    python scripts/build_first_innings_checkpoint.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.checkpoints import CHECKPOINT_ORDER, SPLIT_DIR, write_split_checkpoint

if __name__ == "__main__":
    splits = write_split_checkpoint()
    print(f"\n{'checkpoint':<11}" + "".join(f"{name:>8}" for name in splits))
    for cp in CHECKPOINT_ORDER:
        print(f"{cp:<11}" + "".join(f"{(df['checkpoint'] == cp).sum():>8,}" for df in splits.values()))
    print(f"\nwrote {SPLIT_DIR}")
