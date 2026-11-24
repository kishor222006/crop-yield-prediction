"""Schema and value validation for training datasets and prediction inputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Iterable

import pandas as pd

from weather_impact.config.crops import CROP_PROFILES
from weather_impact.config.settings import (
    INPUT_COLUMNS,
    INPUT_NUMERIC_COLUMNS,
    REQUIRED_COLUMNS,
    VALID_RANGES,
)
from weather_impact.utils.exceptions import DataValidationError, InvalidInputError


def validate_dataset(df: pd.DataFrame) -> None:
    """Check a training dataset. Raises DataValidationError listing every problem found."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError("Missing required columns: " + ", ".join(missing))
    if df.empty:
        raise DataValidationError("Dataset is empty.")

    problems: list[str] = []

    null_counts = df[REQUIRED_COLUMNS].isna().sum()
    for column, count in null_counts.items():
        if count:
            problems.append(f"{column} has {int(count)} missing value(s)")

    for column, (low, high) in VALID_RANGES.items():
        if column not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[column]):
            problems.append(f"{column} must be numeric")
            continue
        out_of_range = int(((df[column] < low) | (df[column] > high)).sum())
        if out_of_range:
            problems.append(f"{column} has {out_of_range} value(s) outside [{low}, {high}]")

    invalid_pairs = {
        (crop, season)
        for crop, season in zip(df["crop_type"], df["season"], strict=True)
        if (crop, season) not in CROP_PROFILES
    }
    if invalid_pairs:
        examples = ", ".join(f"{c}/{s}" for c, s in sorted(invalid_pairs, key=str)[:5])
        problems.append(f"unsupported crop/season combinations: {examples}")

    if problems:
        raise DataValidationError("; ".join(problems))


def validate_inputs(inputs: Mapping[str, Any] | Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Validate prediction input(s) and return a clean DataFrame with INPUT_COLUMNS.

    Accepts one mapping or an iterable of mappings. Extra keys (such as year or
    district) are ignored. Raises InvalidInputError with a clear message on failure.
    """
    if isinstance(inputs, Mapping):
        df = pd.DataFrame([dict(inputs)])
    else:
        df = pd.DataFrame(list(inputs))

    missing = [c for c in INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise InvalidInputError("Missing input field(s): " + ", ".join(missing))
    if df.empty:
        raise InvalidInputError("No input values were provided.")

    df = df[list(INPUT_COLUMNS)].copy()
    for column in INPUT_NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["crop_type"] = df["crop_type"].astype(str).str.strip()
    df["season"] = df["season"].astype(str).str.strip()

    if df[list(INPUT_NUMERIC_COLUMNS)].isna().any().any():
        raise InvalidInputError("Temperature, rainfall, and humidity must be valid numbers.")

    for column in INPUT_NUMERIC_COLUMNS:
        low, high = VALID_RANGES[column]
        if ((df[column] < low) | (df[column] > high)).any():
            raise InvalidInputError(f"{column} must be between {low} and {high}.")

    for crop, season in zip(df["crop_type"], df["season"], strict=True):
        if (crop, season) not in CROP_PROFILES:
            raise InvalidInputError(
                f"'{season}' is not a valid season for '{crop}' in this model."
            )

    return df.reset_index(drop=True)
