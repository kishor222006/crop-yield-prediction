"""Environmental Risk Score: how far current weather is from the crop's optimal range.

Method, for each factor (temperature, rainfall, humidity):
  - distance = how far the value lies outside the optimal (low, high) band, else 0
  - severity = min(1, distance / band width). 1.0 means fully outside the band.
  - score    = 100 * sum(weight * severity). Weights sum to 1, so 0 <= score <= 100.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping

from weather_impact.config.crops import get_profile
from weather_impact.data.validation import validate_inputs
from weather_impact.risk.thresholds import RISK_WEIGHTS, classify_risk


@dataclass(frozen=True)
class RiskResult:
    """Risk score with per-factor detail."""

    crop_type: str
    season: str
    score: float
    level: str
    values: Dict[str, float] = field(default_factory=dict)
    severity: Dict[str, float] = field(default_factory=dict)
    direction: Dict[str, str] = field(default_factory=dict)  # "low", "high", or "ok"


def compute_risk(inputs: Mapping[str, Any]) -> RiskResult:
    """Compute the 0–100 Environmental Risk Score for one set of conditions."""
    row = validate_inputs(inputs).iloc[0]
    crop_type = str(row["crop_type"])
    season = str(row["season"])
    profile = get_profile(crop_type, season)

    bands = {
        "temperature_c": profile.temperature_c,
        "rainfall_mm": profile.rainfall_mm,
        "humidity_pct": profile.humidity_pct,
    }

    values: Dict[str, float] = {}
    severity: Dict[str, float] = {}
    direction: Dict[str, str] = {}
    for factor, (low, high) in bands.items():
        value = float(row[factor])
        values[factor] = value
        if value < low:
            distance, direction[factor] = low - value, "low"
        elif value > high:
            distance, direction[factor] = value - high, "high"
        else:
            distance, direction[factor] = 0.0, "ok"
        severity[factor] = min(1.0, distance / (high - low))

    raw_score = 100.0 * sum(RISK_WEIGHTS[f] * severity[f] for f in RISK_WEIGHTS)
    score = round(min(100.0, max(0.0, raw_score)), 1)

    return RiskResult(
        crop_type=crop_type,
        season=season,
        score=score,
        level=classify_risk(score),
        values=values,
        severity=severity,
        direction=direction,
    )
