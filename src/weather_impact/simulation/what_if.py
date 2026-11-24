"""What-If scenario simulator: change weather inputs and compare yield and risk."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping

from weather_impact.data.validation import validate_inputs
from weather_impact.models.predict import predict_one
from weather_impact.risk.risk_score import compute_risk
from weather_impact.utils.exceptions import InvalidInputError, ScenarioError

# Allowed changes. These limits keep scenarios realistic.
SCENARIO_LIMITS: Dict[str, tuple[float, float]] = {
    "temperature_delta_c": (-10.0, 10.0),
    "rainfall_change_pct": (-90.0, 200.0),
    "humidity_delta_pct": (-50.0, 50.0),
}


@dataclass(frozen=True)
class ScenarioChange:
    """Relative changes applied to a baseline set of conditions."""

    temperature_delta_c: float = 0.0
    rainfall_change_pct: float = 0.0
    humidity_delta_pct: float = 0.0

    def __post_init__(self) -> None:
        for name, (low, high) in SCENARIO_LIMITS.items():
            value = getattr(self, name)
            if not math.isfinite(value) or not low <= value <= high:
                raise ScenarioError(f"{name} must be between {low} and {high}.")


@dataclass(frozen=True)
class WhatIfResult:
    """Comparison of a baseline and a modified scenario."""

    baseline_yield: float
    scenario_yield: float
    yield_change_pct: float
    baseline_risk: float
    scenario_risk: float
    risk_change: float
    baseline_inputs: Dict[str, Any]
    scenario_inputs: Dict[str, Any]


def apply_change(inputs: Mapping[str, Any], change: ScenarioChange) -> Dict[str, Any]:
    """Return a new input dict with the scenario changes applied.

    Extra enrichment columns are preserved; only the three weather values change.
    """
    validated_row = validate_inputs(inputs).iloc[0].to_dict()
    row = {**dict(inputs), **validated_row}   # keep extra cols, use validated vals
    row["temperature_c"] = float(row["temperature_c"]) + change.temperature_delta_c
    rain_factor = 1.0 + change.rainfall_change_pct / 100.0
    row["rainfall_mm"] = max(0.0, float(row["rainfall_mm"]) * rain_factor)
    row["humidity_pct"] = min(100.0, max(0.0, float(row["humidity_pct"]) + change.humidity_delta_pct))
    try:
        validate_inputs(row)
    except InvalidInputError as exc:
        raise ScenarioError(f"This scenario produces unrealistic conditions: {exc}") from exc
    return row


def run_what_if(model: Any, inputs: Mapping[str, Any], change: ScenarioChange) -> WhatIfResult:
    """Predict yield and risk before and after the scenario changes.

    Extra enrichment columns (year, district, state, dist_crop_mean_yield) are
    preserved so that the ML pipeline receives the full feature set it was
    trained on.
    """
    # Validate required fields, then merge with full input to keep extra columns.
    validated_row = validate_inputs(inputs).iloc[0].to_dict()
    baseline = {**dict(inputs), **validated_row}   # extra cols first, validated vals override

    scenario = apply_change(baseline, change)

    baseline_yield = predict_one(model, baseline)
    scenario_yield = predict_one(model, scenario)
    baseline_risk = compute_risk(baseline).score
    scenario_risk = compute_risk(scenario).score

    if baseline_yield == 0:
        yield_change_pct = 0.0
    else:
        yield_change_pct = (scenario_yield - baseline_yield) / baseline_yield * 100.0

    return WhatIfResult(
        baseline_yield=baseline_yield,
        scenario_yield=scenario_yield,
        yield_change_pct=yield_change_pct,
        baseline_risk=baseline_risk,
        scenario_risk=scenario_risk,
        risk_change=round(scenario_risk - baseline_risk, 1),
        baseline_inputs=baseline,
        scenario_inputs=scenario,
    )
