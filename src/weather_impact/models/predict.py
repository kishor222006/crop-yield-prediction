"""Prediction API for the ML core. Used by the website now and the app later."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from weather_impact.config.settings import DEFAULT_MODEL_NAME
from weather_impact.data.validation import validate_inputs
from weather_impact.features.engineering import prepare_features
from weather_impact.models.registry import load_model


@lru_cache(maxsize=1)
def get_default_model():
    """Load the default trained model once per process."""
    return load_model(DEFAULT_MODEL_NAME)


def predict_yield(model: Any, inputs: Mapping[str, Any]) -> np.ndarray:
    """Predict yield (t/ha) for one or more input rows.

    Inputs are validated first (required fields checked, types coerced).
    The full input dict — including optional enrichment columns like year,
    district, state, and dist_crop_mean_yield — is passed to prepare_features
    so the pipeline receives every column it was trained on.
    Predictions are clipped at zero because a negative yield is not meaningful.
    """
    # 1. Validate required fields and coerce types — but keep all extra columns.
    if isinstance(inputs, Mapping):
        df_full = pd.DataFrame([dict(inputs)])
    else:
        df_full = pd.DataFrame(list(inputs))

    # Run validation on a view that has only the required input columns so the
    # existing validator doesn't choke on unknown keys.
    from weather_impact.config.settings import INPUT_COLUMNS
    validate_inputs({k: df_full.iloc[0][k] for k in INPUT_COLUMNS if k in df_full.columns})

    # 2. Build features from the full frame (retains year/district/state/dcmy).
    features = prepare_features(df_full)
    raw = np.asarray(model.predict(features), dtype=float)
    return np.clip(raw, 0.0, None)


def predict_one(model: Any, inputs: Mapping[str, Any]) -> float:
    """Predict yield for a single set of inputs and return a float."""
    return float(predict_yield(model, inputs)[0])
