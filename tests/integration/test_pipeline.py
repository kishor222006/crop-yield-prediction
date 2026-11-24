"""End-to-end checks: data -> training -> saving -> prediction -> explanation."""

import pytest

from weather_impact.config.settings import ACCURACY_TARGET_PCT
from weather_impact.data.cleaning import clean_dataset
from weather_impact.explainability.shap_explainer import explain_prediction, global_importance
from weather_impact.features.engineering import prepare_features
from weather_impact.models.predict import predict_one
from weather_impact.models.registry import load_metadata, load_model, save_model
from weather_impact.models.train import train_best, train_model
from weather_impact.utils.exceptions import ModelNotFoundError


@pytest.fixture(scope="module")
def trained(synthetic_df):
    df, _ = clean_dataset(synthetic_df)
    best, results = train_best(df, fast=True)
    return df, best, results


def test_both_algorithms_meet_accuracy_target(trained):
    _, _, results = trained
    for name, result in results.items():
        assert result.metrics["accuracy_pct"] >= ACCURACY_TARGET_PCT, name
        assert result.metrics["r2"] >= 0.85, name


def test_best_model_is_selected_by_r2(trained):
    _, best, results = trained
    assert best.metrics["r2"] == max(r.metrics["r2"] for r in results.values())


def test_save_and_load_round_trip(trained, tmp_path, good_conditions):
    _, best, _ = trained
    save_model(best.pipeline, "test_model", best.metrics, best.algorithm, model_dir=tmp_path)
    reloaded = load_model("test_model", model_dir=tmp_path)
    original = predict_one(best.pipeline, good_conditions)
    assert predict_one(reloaded, good_conditions) == pytest.approx(original)
    meta = load_metadata("test_model", model_dir=tmp_path)
    assert meta["algorithm"] == best.algorithm
    assert "scikit-learn" in meta["library_versions"]


def test_missing_model_raises(tmp_path):
    with pytest.raises(ModelNotFoundError, match="train_model"):
        load_model("does_not_exist", model_dir=tmp_path)


def test_prediction_is_never_negative(trained, good_conditions):
    _, best, _ = trained
    worst = {**good_conditions, "temperature_c": 55.0, "rainfall_mm": 0.0, "humidity_pct": 0.0}
    assert predict_one(best.pipeline, worst) >= 0.0


def test_explanation_is_additive(trained, good_conditions):
    _, best, _ = trained
    import pandas as pd

    row = pd.DataFrame([good_conditions])
    contributions, base_value = explain_prediction(best.pipeline, row)
    assert set(contributions["feature"]) == {
        "temperature_c", "rainfall_mm", "humidity_pct", "temp_humidity_index", "crop_type", "season",
    }
    prediction = best.pipeline.predict(prepare_features(row))[0]
    assert base_value + contributions["shap_value"].sum() == pytest.approx(prediction, rel=1e-2, abs=1e-2)


def test_global_importance_ranks_all_features(trained):
    df, best, _ = trained
    X = prepare_features(df)
    importance = global_importance(best.pipeline, X, max_rows=200)
    assert len(importance) == 6
    assert (importance["mean_abs_shap"] >= 0).all()


def test_train_model_rejects_unknown_algorithm(synthetic_df):
    df, _ = clean_dataset(synthetic_df)
    with pytest.raises(ValueError):
        train_model(df, "svm", fast=True)
