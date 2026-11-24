"""Shared pytest fixtures."""

import pytest

from weather_impact.data.synthetic import generate_synthetic_dataset


@pytest.fixture(scope="session")
def synthetic_df():
    """A small, reproducible synthetic dataset for fast tests."""
    return generate_synthetic_dataset(n_rows=1500, seed=123)


@pytest.fixture()
def good_conditions():
    """Rice in Kharif with every factor inside its optimal band."""
    return {
        "crop_type": "Rice",
        "season": "Kharif",
        "temperature_c": 27.0,
        "rainfall_mm": 1100.0,
        "humidity_pct": 72.0,
    }
