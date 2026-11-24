"""Cleaning pipeline: turns raw crop and weather records into model-ready data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Tuple

import pandas as pd

from weather_impact.config.crops import CROP_PROFILES
from weather_impact.config.settings import (
    INPUT_NUMERIC_COLUMNS,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    VALID_RANGES,
)
from weather_impact.data.validation import validate_dataset
from weather_impact.utils.exceptions import DataValidationError

_NUMERIC_COLUMNS = ["year", *INPUT_NUMERIC_COLUMNS, TARGET_COLUMN]
_TEXT_COLUMNS = ["district", "crop_type", "season", "state"]


@dataclass
class CleaningReport:
    """Counts of what the cleaning step changed. Useful for logs and audits."""

    input_rows: int = 0
    missing_key_removed: int = 0
    invalid_category_removed: int = 0
    missing_target_removed: int = 0
    out_of_range_removed: int = 0
    values_imputed: int = 0
    duplicates_removed: int = 0
    output_rows: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def clean_dataset(raw: pd.DataFrame) -> Tuple[pd.DataFrame, CleaningReport]:
    """Clean a raw dataset and return (clean_df, report).

    Steps, in order:
      1. Normalise column names and trim text values.
      2. Convert numeric columns, turning bad entries into missing values.
      3. Remove rows with an unknown crop/season, a missing key, or a missing target.
      4. Remove rows with physically impossible values.
      5. Fill missing weather values with the median of the same crop and season.
      6. Remove exact duplicate rows.
      7. Validate the result. Raises DataValidationError if the data is still unusable.
    """
    report = CleaningReport(input_rows=len(raw))
    df = raw.copy()

    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DataValidationError("Missing required columns: " + ", ".join(missing_cols))
    if df.empty:
        raise DataValidationError("Dataset is empty.")

    for column in _TEXT_COLUMNS:
        df[column] = df[column].apply(lambda v: v.strip() if isinstance(v, str) else v)
    for column in _NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    before = len(df)
    pairs_valid = [
        isinstance(c, str) and isinstance(s, str) and (c, s) in CROP_PROFILES
        for c, s in zip(df["crop_type"], df["season"], strict=True)
    ]
    df = df[pairs_valid]
    report.invalid_category_removed = before - len(df)

    before = len(df)
    df = df[df["year"].notna() & df["district"].notna()]
    report.missing_key_removed = before - len(df)

    before = len(df)
    df = df[df[TARGET_COLUMN].notna()]
    report.missing_target_removed = before - len(df)

    before = len(df)
    in_range = pd.Series(True, index=df.index)
    for column, (low, high) in VALID_RANGES.items():
        if column in df.columns:
            in_range &= df[column].between(low, high) | df[column].isna()
    df = df[in_range]
    report.out_of_range_removed = before - len(df)

    df = df.copy()
    for column in INPUT_NUMERIC_COLUMNS:
        missing_mask = df[column].isna()
        report.values_imputed += int(missing_mask.sum())
        if missing_mask.any():
            group_median = df.groupby(["crop_type", "season"])[column].transform("median")
            df[column] = df[column].fillna(group_median).fillna(df[column].median())
        if df[column].isna().any():
            raise DataValidationError(f"Column '{column}' has no usable values after cleaning.")

    before = len(df)
    df = df.drop_duplicates()
    report.duplicates_removed = before - len(df)

    df = df.reset_index(drop=True)
    validate_dataset(df)
    report.output_rows = len(df)
    return df, report
