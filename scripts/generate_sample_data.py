"""Generate the reproducible SYNTHETIC demo dataset.

Usage:
    python scripts/generate_sample_data.py
    python scripts/generate_sample_data.py --rows 8000 --seed 7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from weather_impact.config.settings import RANDOM_STATE, SAMPLE_DATA_PATH
from weather_impact.data.synthetic import generate_synthetic_dataset
from weather_impact.utils.logger import get_logger

logger = get_logger("generate_sample_data")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic crop/weather data.")
    parser.add_argument("--rows", type=int, default=5000, help="Number of rows (default: 5000)")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE, help="Random seed for reproducibility")
    parser.add_argument("--output", type=Path, default=SAMPLE_DATA_PATH, help="Output CSV path")
    args = parser.parse_args(argv)

    try:
        df = generate_synthetic_dataset(n_rows=args.rows, seed=args.seed)
    except ValueError as exc:
        logger.error("Invalid arguments: %s", exc)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    logger.info("Wrote %d synthetic rows to %s", len(df), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
