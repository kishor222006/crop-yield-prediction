"""Crop profiles: optimal weather ranges and reference yields per crop and season.

Each (crop, season) pair has its own optimal ranges. Rainfall is the total for
the growing season in millimetres. Indian seasons are used:
Kharif (monsoon), Rabi (winter), and Zaid (summer).

These are agronomic reference values. Have them verified by an agronomist
before any production or farmer-facing use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from weather_impact.utils.exceptions import InvalidInputError

SEASONS: Tuple[str, ...] = ("Kharif", "Rabi", "Zaid")


@dataclass(frozen=True)
class CropProfile:
    """Optimal (low, high) ranges and the reference yield under good conditions."""

    temperature_c: Tuple[float, float]
    rainfall_mm: Tuple[float, float]
    humidity_pct: Tuple[float, float]
    base_yield_t_per_ha: float


CROP_PROFILES: Dict[Tuple[str, str], CropProfile] = {
    ("Rice", "Kharif"): CropProfile((22.0, 32.0), (800.0, 1500.0), (60.0, 85.0), 5.5),
    ("Maize", "Kharif"): CropProfile((18.0, 30.0), (500.0, 900.0), (50.0, 75.0), 6.0),
    ("Maize", "Rabi"): CropProfile((16.0, 28.0), (150.0, 350.0), (40.0, 65.0), 5.0),
    ("Ragi", "Kharif"): CropProfile((20.0, 30.0), (400.0, 800.0), (50.0, 70.0), 2.5),
    ("Groundnut", "Kharif"): CropProfile((22.0, 32.0), (450.0, 900.0), (50.0, 75.0), 2.0),
    ("Groundnut", "Zaid"): CropProfile((25.0, 35.0), (100.0, 300.0), (40.0, 65.0), 1.8),
    ("Wheat", "Rabi"): CropProfile((12.0, 25.0), (100.0, 300.0), (40.0, 70.0), 3.5),
}

CROPS: Tuple[str, ...] = tuple(sorted({crop for crop, _ in CROP_PROFILES}))


def get_profile(crop_type: str, season: str) -> CropProfile:
    """Return the profile for a crop in a season, or raise InvalidInputError."""
    key = (crop_type, season)
    if key not in CROP_PROFILES:
        valid = ", ".join(f"{c} ({s})" for c, s in sorted(CROP_PROFILES))
        raise InvalidInputError(
            f"'{crop_type}' is not grown in '{season}' in this model. Valid combinations: {valid}."
        )
    return CROP_PROFILES[key]


def seasons_for_crop(crop_type: str) -> Tuple[str, ...]:
    """Return the seasons in which a crop is supported."""
    return tuple(sorted(s for c, s in CROP_PROFILES if c == crop_type))
