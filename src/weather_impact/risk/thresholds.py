"""Weights and level boundaries for the Environmental Risk Score (0–100)."""

from __future__ import annotations

from typing import Dict, Tuple

# Relative influence of each weather factor. Must sum to 1.0 (checked by tests).
RISK_WEIGHTS: Dict[str, float] = {
    "temperature_c": 0.40,
    "rainfall_mm": 0.35,
    "humidity_pct": 0.25,
}

# (exclusive upper bound, label). Scores at or above the last bound are "Severe".
RISK_LEVEL_BOUNDS: Tuple[Tuple[float, str], ...] = (
    (25.0, "Low"),
    (50.0, "Moderate"),
    (75.0, "High"),
)
SEVERE_LABEL = "Severe"


def classify_risk(score: float) -> str:
    """Map a 0–100 score to a risk level label."""
    for bound, label in RISK_LEVEL_BOUNDS:
        if score < bound:
            return label
    return SEVERE_LABEL
