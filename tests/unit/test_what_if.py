import numpy as np
import pytest

from weather_impact.simulation.what_if import ScenarioChange, apply_change, run_what_if
from weather_impact.utils.exceptions import ScenarioError


class _StubModel:
    """Predicts yield directly from temperature so scenario arithmetic is easy to check."""

    def predict(self, X):
        return np.asarray(X["rainfall_mm"], dtype=float) / 100.0


def test_apply_change_adjusts_all_factors(good_conditions):
    change = ScenarioChange(temperature_delta_c=2.0, rainfall_change_pct=10.0, humidity_delta_pct=-5.0)
    row = apply_change(good_conditions, change)
    assert row["temperature_c"] == pytest.approx(29.0)
    assert row["rainfall_mm"] == pytest.approx(1210.0)
    assert row["humidity_pct"] == pytest.approx(67.0)


def test_rainfall_never_goes_negative_and_humidity_stays_in_range(good_conditions):
    row = apply_change(good_conditions, ScenarioChange(rainfall_change_pct=-90.0, humidity_delta_pct=50.0))
    assert row["rainfall_mm"] >= 0
    assert 0 <= row["humidity_pct"] <= 100


def test_run_what_if_reports_yield_change(good_conditions):
    result = run_what_if(_StubModel(), good_conditions, ScenarioChange(rainfall_change_pct=-50.0))
    assert result.scenario_yield == pytest.approx(result.baseline_yield * 0.5)
    assert result.yield_change_pct == pytest.approx(-50.0)


def test_no_change_gives_identical_results(good_conditions):
    result = run_what_if(_StubModel(), good_conditions, ScenarioChange())
    assert result.yield_change_pct == 0.0
    assert result.risk_change == 0.0


def test_scenario_out_of_limits_is_rejected():
    with pytest.raises(ScenarioError):
        ScenarioChange(temperature_delta_c=50.0)


def test_scenario_producing_impossible_temperature_is_rejected(good_conditions):
    hot = {**good_conditions, "temperature_c": 50.0}
    with pytest.raises(ScenarioError):
        apply_change(hot, ScenarioChange(temperature_delta_c=10.0))


def test_non_finite_scenario_value_is_rejected():
    with pytest.raises(ScenarioError):
        ScenarioChange(humidity_delta_pct=float("nan"))
