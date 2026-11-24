"""Load crop and weather datasets from disk."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

from weather_impact.utils.exceptions import DataValidationError

PathLike = Union[str, Path]


def load_dataset(path: PathLike) -> pd.DataFrame:
    """Read a CSV dataset. Raises DataValidationError for any readable problem."""
    file_path = Path(path)
    if not file_path.exists():
        raise DataValidationError(f"Dataset not found: {file_path}")
    if file_path.suffix.lower() != ".csv":
        raise DataValidationError(f"Only CSV files are supported, got: {file_path.name}")
    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError(f"CSV file is empty: {file_path}") from exc
    except pd.errors.ParserError as exc:
        raise DataValidationError(f"Could not parse CSV {file_path}: {exc}") from exc
    if df.empty:
        raise DataValidationError(f"CSV file has no data rows: {file_path}")
    return df
