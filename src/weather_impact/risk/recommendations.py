"""Plain-language mitigation advice generated from a RiskResult."""

from __future__ import annotations

from typing import List, Tuple

from weather_impact.risk.risk_score import RiskResult

_MESSAGES: dict[Tuple[str, str], str] = {
    ("temperature_c", "high"): (
        "Heat stress: shift sowing to cooler dates, use shade nets or mulch, "
        "and irrigate early morning or evening."
    ),
    ("temperature_c", "low"): (
        "Cold stress: delay sowing until temperatures rise, and use row covers "
        "during cold spells."
    ),
    ("rainfall_mm", "high"): (
        "Excess rainfall: improve field drainage, use raised beds, and apply "
        "preventive fungicide on a fixed schedule."
    ),
    ("rainfall_mm", "low"): (
        "Water deficit: start supplementary irrigation (drip if possible), apply "
        "mulch, and consider drought-tolerant varieties."
    ),
    ("humidity_pct", "high"): (
        "High humidity: widen plant spacing for airflow and schedule preventive "
        "fungicide sprays against fungal disease."
    ),
    ("humidity_pct", "low"): (
        "Dry air: mulch to reduce evaporation and keep soil moisture up with light, "
        "frequent irrigation."
    ),
}

NO_ACTION_MESSAGE = (
    "Conditions are within the optimal range for this crop and season. "
    "Continue routine monitoring."
)


def recommendations(result: RiskResult) -> List[str]:
    """Return advice for each stressed factor, most severe first."""
    stressed = [
        factor
        for factor, severity in sorted(result.severity.items(), key=lambda item: item[1], reverse=True)
        if severity > 0
    ]
    if not stressed:
        return [NO_ACTION_MESSAGE]
    return [_MESSAGES[(factor, result.direction[factor])] for factor in stressed]
