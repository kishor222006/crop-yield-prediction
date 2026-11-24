"""Central configuration for the weather impact ML core.

Every tunable constant lives here so that training, prediction, risk scoring,
and the future API and app share a single source of truth.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "crop_weather_records.csv"
SAMPLE_DATA_PATH = DATA_DIR / "sample" / "crop_weather_synthetic.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "saved"
DEFAULT_MODEL_NAME = "yield_model"

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "yield_t_per_ha"
ACCURACY_TARGET_PCT = 85.0  # Target stated in the project synopsis

NUMERIC_FEATURES = [
    "year",
    "temperature_c",
    "rainfall_mm",
    "humidity_pct",
    "temp_humidity_index",
    "dist_crop_mean_yield",
]
CATEGORICAL_FEATURES = ["crop_type", "season", "district", "state"]

# Columns a prediction request must provide (year/district are optional metadata)
INPUT_COLUMNS = ("crop_type", "season", "temperature_c", "rainfall_mm", "humidity_pct")
INPUT_NUMERIC_COLUMNS = ("temperature_c", "rainfall_mm", "humidity_pct")

# Columns a training dataset must contain
REQUIRED_COLUMNS = [
    "year",
    "district",
    "state",
    *INPUT_COLUMNS,
    TARGET_COLUMN,
]

# Physically plausible bounds used by validation and cleaning
VALID_RANGES = {
    "year": (1900, 2100),
    "temperature_c": (-10.0, 55.0),
    "rainfall_mm": (0.0, 3000.0),
    "humidity_pct": (0.0, 100.0),
    TARGET_COLUMN: (0.0, 50.0),
}

# Website settings
SCENARIO_DB_PATH = DATA_DIR / "weather_impact.db"
CURRENCY_SYMBOL = "₹"
MAX_SCENARIO_NAME_LENGTH = 80
