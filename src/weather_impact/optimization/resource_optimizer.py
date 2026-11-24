"""Mitigation resource optimiser: spend a limited budget to cut risk the most.

Model: each action has a unit cost, a risk reduction per unit, and a maximum
number of units. Units may be fractional (for example, litres of water or kg of
fertiliser). This is a fractional knapsack problem. Taking actions in order of
risk reduction per unit cost is optimal for it, so the result is the best
possible allocation for the budget, not just a heuristic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List

from weather_impact.utils.exceptions import InvalidInputError


@dataclass(frozen=True)
class MitigationAction:
    name: str
    unit_cost: float
    risk_reduction_per_unit: float
    max_units: float

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise InvalidInputError("Action name must not be empty.")
        if not math.isfinite(self.unit_cost) or self.unit_cost <= 0:
            raise InvalidInputError(f"'{self.name}': unit_cost must be greater than 0.")
        if not math.isfinite(self.risk_reduction_per_unit) or self.risk_reduction_per_unit < 0:
            raise InvalidInputError(f"'{self.name}': risk_reduction_per_unit must be 0 or greater.")
        if not math.isfinite(self.max_units) or self.max_units < 0:
            raise InvalidInputError(f"'{self.name}': max_units must be 0 or greater.")


@dataclass(frozen=True)
class Allocation:
    name: str
    units: float
    cost: float
    risk_reduction: float


@dataclass(frozen=True)
class OptimizationResult:
    allocations: List[Allocation]
    total_cost: float
    total_risk_reduction: float
    remaining_budget: float


def optimize_mitigation(actions: Iterable[MitigationAction], budget: float) -> OptimizationResult:
    """Allocate the budget across actions to maximise total risk reduction."""
    if not math.isfinite(budget) or budget < 0:
        raise InvalidInputError("Budget must be 0 or greater.")

    actions = list(actions)
    names = [a.name for a in actions]
    if len(names) != len(set(names)):
        raise InvalidInputError("Action names must be unique.")

    # Highest risk reduction per unit of money first. Ties are broken by name for stable results.
    ranked = sorted(
        actions,
        key=lambda a: (-(a.risk_reduction_per_unit / a.unit_cost), a.name),
    )

    remaining = float(budget)
    allocations: List[Allocation] = []
    for action in ranked:
        if action.risk_reduction_per_unit == 0 or action.max_units == 0:
            continue
        affordable_units = remaining / action.unit_cost
        units = min(action.max_units, affordable_units)
        if units <= 0:
            continue
        cost = units * action.unit_cost
        allocations.append(
            Allocation(
                name=action.name,
                units=units,
                cost=cost,
                risk_reduction=units * action.risk_reduction_per_unit,
            )
        )
        remaining -= cost

    remaining = max(0.0, remaining)
    return OptimizationResult(
        allocations=allocations,
        total_cost=sum(a.cost for a in allocations),
        total_risk_reduction=sum(a.risk_reduction for a in allocations),
        remaining_budget=remaining,
    )
