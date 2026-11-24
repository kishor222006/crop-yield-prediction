import pytest

from weather_impact.optimization.resource_optimizer import MitigationAction, optimize_mitigation
from weather_impact.utils.exceptions import InvalidInputError


def _actions():
    return [
        # value density (reduction per unit cost): mulching 0.4 > irrigation 0.2 > shade nets 0.15
        MitigationAction("irrigation", unit_cost=10.0, risk_reduction_per_unit=2.0, max_units=20),
        MitigationAction("mulching", unit_cost=5.0, risk_reduction_per_unit=2.0, max_units=10),
        MitigationAction("shade_nets", unit_cost=20.0, risk_reduction_per_unit=3.0, max_units=5),
    ]


def test_budget_is_never_exceeded():
    result = optimize_mitigation(_actions(), budget=100.0)
    assert result.total_cost <= 100.0 + 1e-9
    assert result.remaining_budget == pytest.approx(100.0 - result.total_cost)


def test_most_efficient_action_is_filled_first():
    result = optimize_mitigation(_actions(), budget=50.0)
    assert result.allocations[0].name == "mulching"
    assert result.allocations[0].units == pytest.approx(10.0)  # capped by max_units


def test_fractional_units_fill_remaining_budget_exactly():
    result = optimize_mitigation(_actions(), budget=60.0)
    # mulching 10 units = 50, leaving 10 for irrigation at 10 per unit = 1 unit
    names = {a.name: a.units for a in result.allocations}
    assert names["mulching"] == pytest.approx(10.0)
    assert names["irrigation"] == pytest.approx(1.0)
    assert result.total_risk_reduction == pytest.approx(10 * 2.0 + 1 * 2.0)


def test_zero_budget_allocates_nothing():
    result = optimize_mitigation(_actions(), budget=0.0)
    assert result.allocations == []
    assert result.total_risk_reduction == 0.0


def test_optimal_against_brute_force_on_small_instance():
    actions = _actions()
    budget = 75.0
    result = optimize_mitigation(actions, budget)
    # Brute-force grid search over fractions of each action (step 0.5 unit)
    best = 0.0
    steps = [i * 0.5 for i in range(0, 41)]
    for u1 in steps:
        for u2 in [i * 0.5 for i in range(0, 21)]:
            for u3 in [i * 0.5 for i in range(0, 11)]:
                cost = u1 * 10 + u2 * 5 + u3 * 20
                if cost <= budget + 1e-9:
                    reduction = min(u1, 20) * 2 + min(u2, 10) * 2 + min(u3, 5) * 3
                    best = max(best, reduction)
    assert result.total_risk_reduction >= best - 1e-6


def test_zero_reduction_actions_are_ignored():
    actions = [MitigationAction("noop", 1.0, 0.0, 100.0), MitigationAction("real", 1.0, 1.0, 5.0)]
    result = optimize_mitigation(actions, budget=10.0)
    assert [a.name for a in result.allocations] == ["real"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "", "unit_cost": 1.0, "risk_reduction_per_unit": 1.0, "max_units": 1.0},
        {"name": "x", "unit_cost": 0.0, "risk_reduction_per_unit": 1.0, "max_units": 1.0},
        {"name": "x", "unit_cost": 1.0, "risk_reduction_per_unit": -1.0, "max_units": 1.0},
        {"name": "x", "unit_cost": 1.0, "risk_reduction_per_unit": 1.0, "max_units": -1.0},
    ],
)
def test_invalid_actions_are_rejected(kwargs):
    with pytest.raises(InvalidInputError):
        MitigationAction(**kwargs)


def test_negative_budget_is_rejected():
    with pytest.raises(InvalidInputError):
        optimize_mitigation(_actions(), budget=-1.0)


def test_duplicate_action_names_are_rejected():
    dup = [MitigationAction("a", 1, 1, 1), MitigationAction("a", 2, 1, 1)]
    with pytest.raises(InvalidInputError):
        optimize_mitigation(dup, budget=10)
