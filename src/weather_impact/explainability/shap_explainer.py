"""SHAP explanations: how much each weather factor pushed a prediction up or down.

One-hot encoded columns (for example cat__crop_type_Rice) are summed back into
their original feature (crop_type) so that users see one line per factor.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from weather_impact.config.settings import CATEGORICAL_FEATURES
from weather_impact.features.engineering import prepare_features


def _original_feature(encoded_name: str) -> str:
    """Map a transformed column name back to the original feature name."""
    transformer, _, rest = encoded_name.partition("__")
    if transformer == "cat":
        for feature in CATEGORICAL_FEATURES:
            if rest.startswith(f"{feature}_"):
                return feature
    return rest


def _explain_frame(pipeline: Pipeline, X: pd.DataFrame) -> Tuple[np.ndarray, float, list]:
    """Return (per-encoded-column SHAP values, base value, encoded column names)."""
    preprocessor = pipeline.named_steps["preprocessor"]
    regressor = pipeline.named_steps["regressor"]
    transformed = preprocessor.transform(X)
    encoded_names = list(preprocessor.get_feature_names_out())
    explainer = shap.TreeExplainer(regressor)
    values = np.asarray(explainer.shap_values(transformed), dtype=float)
    base_value = float(np.asarray(explainer.expected_value).ravel()[0])
    return values, base_value, encoded_names


def explain_prediction(pipeline: Pipeline, inputs: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """Explain one prediction.

    Returns (contributions, base_value). contributions has one row per original
    feature with columns feature, value, and shap_value, sorted by impact size.
    base_value + sum(shap_value) equals the model's prediction.
    """
    if len(inputs) != 1:
        raise ValueError("explain_prediction expects exactly one row.")
    X = prepare_features(inputs)
    values, base_value, encoded_names = _explain_frame(pipeline, X)

    grouped: dict[str, float] = {}
    for name, value in zip(encoded_names, values[0], strict=True):
        feature = _original_feature(name)
        grouped[feature] = grouped.get(feature, 0.0) + float(value)

    contributions = pd.DataFrame(
        [
            {"feature": feature, "value": X.iloc[0][feature], "shap_value": shap_value}
            for feature, shap_value in grouped.items()
        ]
    )
    contributions["abs_impact"] = contributions["shap_value"].abs()
    contributions = contributions.sort_values("abs_impact", ascending=False).drop(columns="abs_impact")
    return contributions.reset_index(drop=True), base_value


def global_importance(pipeline: Pipeline, X: pd.DataFrame, max_rows: int = 500) -> pd.DataFrame:
    """Mean absolute SHAP value per original feature across a sample of rows."""
    sample = X.sample(n=min(max_rows, len(X)), random_state=0) if len(X) > max_rows else X
    values, _, encoded_names = _explain_frame(pipeline, sample)

    # Sum encoded columns per row first, then take the mean absolute impact per feature.
    features = [_original_feature(name) for name in encoded_names]
    result = pd.DataFrame(
        [
            {"feature": feature, "mean_abs_shap": float(np.mean(np.abs(group.sum(axis=1))))}
            for feature, group in (
                (feature, values[:, [i for i, f in enumerate(features) if f == feature]])
                for feature in dict.fromkeys(features)
            )
        ]
    )
    return result.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
