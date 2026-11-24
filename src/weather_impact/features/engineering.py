"""Feature engineering shared by training and prediction.

Keeping this in one place guarantees that the model sees identical features
during training, batch scoring, the What-If simulator, and the API.
"""

from __future__ import annotations

import pandas as pd

from weather_impact.config.settings import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with derived columns added.

    temp_humidity_index: temperature scaled by relative humidity. High heat with
    high humidity is harder on crops than heat alone.

    dist_crop_mean_yield: per district+crop historical mean yield. Captures the
    long-run productivity baseline at each location, which is the single biggest
    predictor of yield in panel data.
    """
    out = df.copy()
    out["temp_humidity_index"] = out["temperature_c"] * out["humidity_pct"] / 100.0

    # District × crop yield baseline (mean over all years in the dataset).
    # Using transform keeps the index aligned for both train and test.
    if "district" in out.columns and "crop_type" in out.columns:
        out["dist_crop_mean_yield"] = (
            out.groupby(["district", "crop_type"])["yield_t_per_ha"].transform("mean")
            if "yield_t_per_ha" in out.columns
            else 0.0
        )
    return out


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the model feature frame, with columns in a fixed order.

    Includes year, district, state, and a district×crop yield baseline so the
    model can learn location-specific baselines and long-run trends.
    Falls back gracefully if district/state are absent (e.g. live prediction input).
    """
    out = add_derived_features(df)
    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    # Only keep columns that actually exist in this frame
    cols = [c for c in cols if c in out.columns]
    return out[cols]
