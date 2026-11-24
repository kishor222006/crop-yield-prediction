"""Synthetic crop and weather dataset generator.

Used for development, tests, and demos until real records are available.
The yield response is an agronomic-style model: each weather factor has an
optimal band, drought and waterlogging are penalised, extreme heat applies a
shock penalty, and multiplicative noise is added. The data is NOT real
field data and must never be presented as such.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from weather_impact.config.crops import CROP_PROFILES, CropProfile
from weather_impact.config.settings import RANDOM_STATE

DISTRICTS = (
    "Bengaluru Rural",
    "Mysuru",
    "Tumakuru",
    "Belagavi",
    "Kalaburagi",
    "Hassan",
    "Shivamogga",
    "Dharwad",
)

# (mean, standard deviation) of seasonal weather
SEASON_CLIMATE = {
    "Kharif": {"temperature_c": (26.0, 2.5), "rainfall_mm": (650.0, 250.0), "humidity_pct": (75.0, 8.0)},
    "Rabi": {"temperature_c": (22.0, 3.0), "rainfall_mm": (200.0, 120.0), "humidity_pct": (55.0, 10.0)},
    "Zaid": {"temperature_c": (32.0, 2.5), "rainfall_mm": (150.0, 100.0), "humidity_pct": (45.0, 10.0)},
}

HEAT_SHOCK_C = 38.0


def _band_factor(value: float, low: float, high: float, tolerance: float) -> float:
    """Gaussian-shaped factor that is 1.0 at the middle of the band and falls off outside it."""
    centre = (low + high) / 2.0
    half_width = (high - low) / 2.0 + tolerance
    return math.exp(-0.5 * ((value - centre) / half_width) ** 2)


def yield_response(profile: CropProfile, temperature: float, rainfall: float, humidity: float) -> float:
    """Expected yield (t/ha) for given weather under the synthetic response model."""
    t_low, t_high = profile.temperature_c
    factor_temp = _band_factor(temperature, t_low, t_high, tolerance=3.0)
    if temperature > HEAT_SHOCK_C:
        factor_temp *= 0.7

    r_low, r_high = profile.rainfall_mm
    if rainfall < r_low:
        factor_rain = (max(rainfall, 1.0) / r_low) ** 1.2
    elif rainfall > r_high:
        factor_rain = max(0.4, 1.0 - (rainfall - r_high) / r_high)
    else:
        factor_rain = 1.0

    h_low, h_high = profile.humidity_pct
    factor_hum = _band_factor(humidity, h_low, h_high, tolerance=10.0)

    return profile.base_yield_t_per_ha * factor_temp * factor_rain * factor_hum


def generate_synthetic_dataset(
    n_rows: int = 5000,
    seed: int = RANDOM_STATE,
    start_year: int = 2010,
    end_year: int = 2024,
) -> pd.DataFrame:
    """Create a reproducible synthetic dataset with the project's column layout."""
    if n_rows < 1:
        raise ValueError("n_rows must be at least 1.")
    if start_year > end_year:
        raise ValueError("start_year must not be after end_year.")

    rng = np.random.default_rng(seed)
    combinations = list(CROP_PROFILES.keys())
    rows = []

    for _ in range(n_rows):
        crop, season = combinations[rng.integers(len(combinations))]
        profile = CROP_PROFILES[(crop, season)]
        climate = SEASON_CLIMATE[season]

        temperature = rng.normal(*climate["temperature_c"])
        rainfall = float(np.clip(rng.normal(*climate["rainfall_mm"]), 0.0, 2000.0))
        humidity = float(np.clip(rng.normal(*climate["humidity_pct"]), 10.0, 100.0))
        noise = rng.normal(1.0, 0.05)
        expected = yield_response(profile, temperature, rainfall, humidity)

        rows.append(
            {
                "year": int(rng.integers(start_year, end_year + 1)),
                "district": DISTRICTS[rng.integers(len(DISTRICTS))],
                "crop_type": crop,
                "season": season,
                "temperature_c": round(float(temperature), 1),
                "rainfall_mm": round(rainfall, 1),
                "humidity_pct": round(humidity, 1),
                "yield_t_per_ha": round(max(0.05, expected * noise), 3),
            }
        )

    return pd.DataFrame(rows)
