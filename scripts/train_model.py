"""Train the yield prediction model and save it to models/saved/.

Usage:
    python scripts/train_model.py                          # best model on the default dataset
    python scripts/train_model.py --algorithm xgboost
    python scripts/train_model.py --data path/to/records.csv --algorithm random_forest
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from weather_impact.config.settings import (
    ACCURACY_TARGET_PCT,
    DEFAULT_MODEL_NAME,
    RAW_DATA_PATH,
    SAMPLE_DATA_PATH,
)
from weather_impact.data.cleaning import clean_dataset
from weather_impact.data.loader import load_dataset
from weather_impact.models.registry import save_model
from weather_impact.models.train import ALGORITHMS, TrainResult, train_best, train_model
from weather_impact.utils.exceptions import WeatherImpactError
from weather_impact.utils.logger import get_logger

logger = get_logger("train_model")


def _describe(result: TrainResult) -> str:
    m = result.metrics
    return (
        f"{result.algorithm:<14} R2={m['r2']:.4f}  accuracy={m['accuracy_pct']:.2f}%  "
        f"MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  MAPE={m['mape_pct']:.2f}%"
    )


def _resolve_data_path(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    if RAW_DATA_PATH.exists():
        return RAW_DATA_PATH
    return SAMPLE_DATA_PATH


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the crop yield model.")
    parser.add_argument("--data", type=Path, default=None, help="CSV dataset path")
    parser.add_argument(
        "--algorithm",
        choices=(*ALGORITHMS, "best"),
        default="best",
        help="Algorithm to train, or 'best' to compare all and keep the top R²",
    )
    parser.add_argument("--name", default=DEFAULT_MODEL_NAME, help="Name for the saved model")
    parser.add_argument("--fast", action="store_true", help="Fewer trees (for quick checks)")
    args = parser.parse_args(argv)

    try:
        data_path = _resolve_data_path(args.data)
        logger.info("Loading dataset: %s", data_path)
        raw = load_dataset(data_path)

        df, report = clean_dataset(raw)
        logger.info("Cleaning report: %s", report.as_dict())

        if args.algorithm == "best":
            chosen, all_results = train_best(df, fast=args.fast)
            for result in all_results.values():
                logger.info(_describe(result))
        else:
            chosen = train_model(df, args.algorithm, fast=args.fast)
            logger.info(_describe(chosen))

        model_path = save_model(chosen.pipeline, args.name, chosen.metrics, chosen.algorithm)
        logger.info("Saved %s model to %s", chosen.algorithm, model_path)

        accuracy = chosen.metrics["accuracy_pct"]
        if accuracy < ACCURACY_TARGET_PCT:
            logger.warning("Accuracy %.2f%% is below the %.0f%% target.", accuracy, ACCURACY_TARGET_PCT)
        else:
            logger.info("Accuracy %.2f%% meets the %.0f%% target.", accuracy, ACCURACY_TARGET_PCT)
        return 0
    except WeatherImpactError as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
