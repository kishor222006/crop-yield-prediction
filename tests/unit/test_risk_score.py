import pytest

from weather_impact.risk.recommendations import NO_ACTION_MESSAGE, recommendations
from weather_impact.risk.risk_score import compute_risk
from weather_impact.risk.thresholds import RISK_WEIGHTS, classify_risk
from weather_impact.utils.exceptions import InvalidInputError


def test_weights_sum_to_one():
    assert sum(RISK_WEIGHTS.values()) == pytest.approx(1.0)


def test_conditions_inside_optimal_range_score_zero(good_conditions):
    result = compute_risk(good_conditions)
    assert result.score == 0.0
    assert result.level == "Low"
    assert all(d == "ok" for d in result.direction.values())
    assert recommendations(result) == [NO_ACTION_MESSAGE]


def test_score_reaches_100_only_when_every_factor_is_fully_out_of_band(good_conditions):
    # Humidity 0 is far below the 60-85 band (severity 1.0), so all three factors are at full severity.
    extreme = {**good_conditions, "temperature_c": 55.0, "rainfall_mm": 3000.0, "humidity_pct": 0.0}
    result = compute_risk(extreme)
    assert result.score == 100.0
    assert result.level == "Severe"


def test_partial_extremes_score_below_100(good_conditions):
    # Humidity 100 is 15 points above an 85 ceiling in a 25-point band: severity 0.6.
    extreme = {**good_conditions, "temperature_c": 55.0, "rainfall_mm": 3000.0, "humidity_pct": 100.0}
    result = compute_risk(extreme)
    assert result.severity["humidity_pct"] == pytest.approx(0.6)
    assert result.score == pytest.approx(90.0)


def test_heat_only_stress_matches_weight(good_conditions):
    # Rice Kharif band is 22-32 C (width 10). 42 C is 10 C above it, so severity = 1.0.
    result = compute_risk({**good_conditions, "temperature_c": 42.0})
    assert result.direction["temperature_c"] == "high"
    assert result.severity["temperature_c"] == pytest.approx(1.0)
    assert result.score == pytest.approx(40.0)


def test_drought_direction_is_low(good_conditions):
    result = compute_risk({**good_conditions, "rainfall_mm": 400.0})
    assert result.direction["rainfall_mm"] == "low"
    assert result.severity["rainfall_mm"] > 0


def test_score_is_within_bounds_for_random_grid(good_conditions):
    for t in range(-10, 56, 5):
        for r in range(0, 3001, 300):
            for h in range(0, 101, 20):
                conditions = {**good_conditions, "temperature_c": t, "rainfall_mm": r, "humidity_pct": h}
                score = compute_risk(conditions).score
                assert 0.0 <= score <= 100.0


@pytest.mark.parametrize(
    "score, level",
    [
        (0, "Low"),
        (24.9, "Low"),
        (25, "Moderate"),
        (49.9, "Moderate"),
        (50, "High"),
        (75, "Severe"),
        (100, "Severe"),
    ],
)
def test_classify_risk_boundaries(score, level):
    assert classify_risk(score) == level


def test_recommendations_mention_stressed_factors_most_severe_first(good_conditions):
    result = compute_risk({**good_conditions, "temperature_c": 40.0, "humidity_pct": 95.0})
    advice = recommendations(result)
    assert advice[0].startswith("Heat stress")
    assert any(a.startswith("High humidity") for a in advice)


def test_unknown_crop_season_raises(good_conditions):
    with pytest.raises(InvalidInputError):
        compute_risk({**good_conditions, "season": "Zaid"})
