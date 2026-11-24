"""Model training: Random Forest and XGBoost regressors inside one scikit-learn pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from weather_impact.config.settings import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from weather_impact.data.validation import validate_dataset
from weather_impact.features.engineering import prepare_features
from weather_impact.models.evaluate import regression_metrics

ALGORITHMS: Tuple[str, ...] = ("random_forest", "xgboost")


@dataclass
class TrainResult:
    """Output of a training run."""

    algorithm: str
    pipeline: Pipeline
    metrics: Dict[str, float]
    X_test: pd.DataFrame
    y_test: pd.Series


def build_preprocessor() -> ColumnTransformer:
    """Scale numeric features and one-hot encode categorical features."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_estimator(algorithm: str, fast: bool = False):
    """Create an untrained regressor. fast=True uses fewer trees for tests."""
    if algorithm == "random_forest":
        return RandomForestRegressor(
            n_estimators=100 if fast else 500,
            max_depth=None,
            min_samples_leaf=1,
            max_features=0.6,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if algorithm == "xgboost":
        return XGBRegressor(
            n_estimators=200 if fast else 1000,
            learning_rate=0.05,
            max_depth=8,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown algorithm '{algorithm}'. Choose from: {', '.join(ALGORITHMS)}.")


def build_pipeline(algorithm: str, fast: bool = False) -> Pipeline:
    """Full pipeline: preprocessing followed by the regressor."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", build_estimator(algorithm, fast=fast)),
        ]
    )


def split_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Validate a cleaned dataset and return (X, y)."""
    validate_dataset(df)
    return prepare_features(df), df[TARGET_COLUMN]


def train_model(df: pd.DataFrame, algorithm: str, fast: bool = False) -> TrainResult:
    """Train one algorithm on an 80/20 split and score it on the held-out 20%."""
    X, y = split_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    pipeline = build_pipeline(algorithm, fast=fast)
    pipeline.fit(X_train, y_train)
    metrics = regression_metrics(y_test, pipeline.predict(X_test))
    return TrainResult(
        algorithm=algorithm,
        pipeline=pipeline,
        metrics=metrics,
        X_test=X_test,
        y_test=y_test,
    )


def train_best(df: pd.DataFrame, fast: bool = False) -> Tuple[TrainResult, Dict[str, TrainResult]]:
    """Train every supported algorithm and return (best_by_r2, all_results)."""
    results = {algorithm: train_model(df, algorithm, fast=fast) for algorithm in ALGORITHMS}
    best = max(results.values(), key=lambda result: result.metrics["r2"])
    return best, results
