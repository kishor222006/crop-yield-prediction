import numpy as np
import pandas as pd
import pytest

from weather_impact.data.cleaning import clean_dataset
from weather_impact.utils.exceptions import DataValidationError


def _base_rows():
    return pd.DataFrame(
        [
            {"year": 2020, "district": "Mysuru", "crop_type": "Rice", "season": "Kharif",
             "temperature_c": 27.0, "rainfall_mm": 1000.0, "humidity_pct": 70.0, "yield_t_per_ha": 5.0},
            {"year": 2021, "district": "Mysuru", "crop_type": "Rice", "season": "Kharif",
             "temperature_c": 28.0, "rainfall_mm": 1100.0, "humidity_pct": 72.0, "yield_t_per_ha": 5.2},
            {"year": 2022, "district": "Hassan", "crop_type": "Wheat", "season": "Rabi",
             "temperature_c": 18.0, "rainfall_mm": 200.0, "humidity_pct": 50.0, "yield_t_per_ha": 3.1},
        ]
    )


def test_normalises_column_names_and_trims_text():
    raw = _base_rows()
    raw.columns = [" Year ", "District", "Crop Type", "Season", "Temperature_C",
                   "Rainfall_MM", "Humidity_PCT", "Yield_T_Per_Ha"]
    raw["Crop Type"] = raw["Crop Type"].map(lambda v: f"  {v}  ")
    df, _ = clean_dataset(raw)
    assert "crop_type" in df.columns
    assert set(df["crop_type"]) == {"Rice", "Wheat"}


def test_removes_duplicates():
    raw = pd.concat([_base_rows(), _base_rows().iloc[[0]]], ignore_index=True)
    df, report = clean_dataset(raw)
    assert report.duplicates_removed == 1
    assert len(df) == 3


def test_drops_rows_with_missing_target():
    raw = _base_rows()
    raw.loc[0, "yield_t_per_ha"] = np.nan
    df, report = clean_dataset(raw)
    assert report.missing_target_removed == 1
    assert df["yield_t_per_ha"].notna().all()


def test_drops_out_of_range_values():
    raw = _base_rows()
    raw.loc[1, "humidity_pct"] = 150.0
    raw.loc[2, "temperature_c"] = 99.0
    df, report = clean_dataset(raw)
    assert report.out_of_range_removed == 2
    assert len(df) == 1


def test_drops_unsupported_crop_season_combination():
    raw = _base_rows()
    raw.loc[0, "season"] = "Zaid"  # Rice is not grown in Zaid in this model
    df, report = clean_dataset(raw)
    assert report.invalid_category_removed == 1
    assert len(df) == 2


def test_imputes_missing_weather_with_crop_season_median():
    raw = pd.concat([_base_rows(), _base_rows().iloc[[1]]], ignore_index=True)
    raw.loc[3, "rainfall_mm"] = np.nan
    raw.loc[3, "year"] = 2023  # avoid being removed as a duplicate
    df, report = clean_dataset(raw)
    assert report.values_imputed == 1
    imputed = df.loc[df["year"] == 2023, "rainfall_mm"].iloc[0]
    # Rice/Kharif rainfall values present: 1000 and 1100 -> median 1050 (NaN ignored)
    assert imputed == pytest.approx(1050.0)


def test_non_numeric_text_becomes_missing_and_is_imputed():
    raw = _base_rows()
    raw["temperature_c"] = raw["temperature_c"].astype(object)
    raw.loc[0, "temperature_c"] = "hot"
    df, report = clean_dataset(raw)
    assert report.values_imputed == 1
    assert pd.api.types.is_numeric_dtype(df["temperature_c"])


def test_missing_required_column_raises():
    raw = _base_rows().drop(columns=["humidity_pct"])
    with pytest.raises(DataValidationError, match="humidity_pct"):
        clean_dataset(raw)


def test_empty_dataset_raises():
    with pytest.raises(DataValidationError):
        clean_dataset(_base_rows().iloc[0:0])


def test_report_counts_are_consistent():
    raw = pd.concat([_base_rows(), _base_rows()], ignore_index=True)
    raw.loc[0, "yield_t_per_ha"] = np.nan
    df, report = clean_dataset(raw)
    removed = (
        report.missing_key_removed
        + report.invalid_category_removed
        + report.missing_target_removed
        + report.out_of_range_removed
        + report.duplicates_removed
    )
    assert report.input_rows - removed == report.output_rows == len(df)
