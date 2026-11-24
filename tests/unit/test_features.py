import pandas as pd
import pytest

from weather_impact.config.settings import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from weather_impact.features.engineering import add_derived_features, prepare_features


def _frame():
    return pd.DataFrame(
        [
            {"temperature_c": 30.0, "rainfall_mm": 800.0, "humidity_pct": 50.0,
             "crop_type": "Maize", "season": "Kharif", "extra": 1},
        ]
    )


def test_temp_humidity_index_formula():
    out = add_derived_features(_frame())
    assert out.loc[0, "temp_humidity_index"] == pytest.approx(15.0)


def test_add_derived_features_does_not_modify_input():
    original = _frame()
    add_derived_features(original)
    assert "temp_humidity_index" not in original.columns


def test_prepare_features_returns_fixed_column_order_and_drops_extras():
    X = prepare_features(_frame())
    assert list(X.columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert "extra" not in X.columns
