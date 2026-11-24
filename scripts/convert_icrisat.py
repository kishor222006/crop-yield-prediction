"""Convert ICRISAT district-level yield data to the app's training format.

Usage:
    python scripts/convert_icrisat.py
    python scripts/convert_icrisat.py --input data/raw/icrisat_district.csv --output data/raw/crop_weather_records.csv

The ICRISAT file is wide-format: one row per (district, year) with a yield
column for every crop.  The app needs long-format rows with columns:
    year, district, crop_type, season, temperature_c, rainfall_mm,
    humidity_pct, yield_t_per_ha

Weather (temperature, rainfall, humidity) is not in the ICRISAT dataset.
This script derives *representative seasonal averages* from the agronomic
profiles in CROP_PROFILES, jittered with a small random offset so the model
has variance to learn from.  Replace the weather columns with actual
gridded/station data if you have it.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "icrisat_district.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "crop_weather_records.csv"

# ---------------------------------------------------------------------------
# Mapping: ICRISAT column name → (app crop_type, season)
# Only crops/seasons that exist in CROP_PROFILES are kept.
# ---------------------------------------------------------------------------
CROP_MAP: dict[str, tuple[str, str]] = {
    "RICE YIELD (Kg per ha)":           ("Rice",      "Kharif"),
    "WHEAT YIELD (Kg per ha)":          ("Wheat",     "Rabi"),
    "MAIZE YIELD (Kg per ha)":          ("Maize",     "Kharif"),
    "FINGER MILLET YIELD (Kg per ha)":  ("Ragi",      "Kharif"),
    "GROUNDNUT YIELD (Kg per ha)":      ("Groundnut", "Kharif"),
    "SORGHUM YIELD (Kg per ha)":        None,          # not in CROP_PROFILES – skipped
    "PEARL MILLET YIELD (Kg per ha)":   None,          # not in CROP_PROFILES – skipped
}

# Representative (mean, std) for weather per (crop, season) – derived from
# CROP_PROFILES midpoints.  Tune or replace with real gridded data.
WEATHER_STATS: dict[tuple[str, str], dict[str, tuple[float, float]]] = {
    ("Rice",      "Kharif"): {"temperature_c": (27.0, 2.5), "rainfall_mm": (1000.0, 200.0), "humidity_pct": (72.5, 6.0)},
    ("Wheat",     "Rabi"):   {"temperature_c": (18.5, 3.0), "rainfall_mm": (180.0,  50.0),  "humidity_pct": (55.0,  8.0)},
    ("Maize",     "Kharif"): {"temperature_c": (24.0, 2.5), "rainfall_mm": (680.0, 130.0),  "humidity_pct": (62.5,  7.0)},
    ("Ragi",      "Kharif"): {"temperature_c": (25.0, 2.0), "rainfall_mm": (560.0, 120.0),  "humidity_pct": (60.0,  7.0)},
    ("Groundnut", "Kharif"): {"temperature_c": (27.0, 2.5), "rainfall_mm": (650.0, 150.0),  "humidity_pct": (62.5,  7.0)},
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

RNG = np.random.default_rng(42)


def _add_weather(df: pd.DataFrame, crop: str, season: str) -> pd.DataFrame:
    """Add temperature_c, rainfall_mm, humidity_pct columns with realistic jitter."""
    stats = WEATHER_STATS[(crop, season)]
    n = len(df)
    for col, (mean, std) in stats.items():
        df[col] = RNG.normal(mean, std, size=n)
    # Clamp to physically plausible ranges
    df["temperature_c"] = df["temperature_c"].clip(-10, 55)
    df["rainfall_mm"]   = df["rainfall_mm"].clip(0, 3000)
    df["humidity_pct"]  = df["humidity_pct"].clip(0, 100)
    return df


def convert(input_path: Path, output_path: Path) -> None:
    log.info("Reading %s", input_path)
    raw = pd.read_csv(input_path)
    log.info("  %d rows × %d columns", *raw.shape)

    frames: list[pd.DataFrame] = []

    for yield_col, mapping in CROP_MAP.items():
        if mapping is None:
            log.info("  Skipping %s (no matching CROP_PROFILE)", yield_col)
            continue
        if yield_col not in raw.columns:
            log.warning("  Column not found: %s", yield_col)
            continue

        crop_type, season = mapping

        chunk = raw[["Year", "Dist Name", "State Name", yield_col]].copy()
        chunk.columns = ["year", "district", "state", "yield_kg_per_ha"]

        # ICRISAT uses -1 as a missing-value sentinel
        chunk = chunk[chunk["yield_kg_per_ha"] > 0].copy()

        # Convert Kg/ha → t/ha
        chunk["yield_t_per_ha"] = chunk["yield_kg_per_ha"] / 1000.0
        chunk = chunk.drop(columns=["yield_kg_per_ha"])

        # Drop implausible yields (> 50 t/ha)
        chunk = chunk[chunk["yield_t_per_ha"].between(0, 50)]

        chunk["crop_type"] = crop_type
        chunk["season"]    = season

        chunk = _add_weather(chunk, crop_type, season)
        frames.append(chunk)
        log.info("  %-30s → %d rows", f"{crop_type} / {season}", len(chunk))

    if not frames:
        log.error("No data extracted. Check column names in the input file.")
        sys.exit(1)

    result = pd.concat(frames, ignore_index=True)

    # Final column order expected by the app
    result = result[
        ["year", "district", "state", "crop_type", "season",
         "temperature_c", "rainfall_mm", "humidity_pct", "yield_t_per_ha"]
    ]

    result = result.sort_values(["state", "district", "year", "crop_type"]).reset_index(drop=True)

    log.info("Writing %d rows to %s", len(result), output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    log.info("Done.")

    # Summary
    print("\n── Summary ─────────────────────────────────────────────")
    print(f"  Total rows   : {len(result):,}")
    print(f"  States       : {result['state'].nunique()} ({', '.join(sorted(result['state'].unique()))})")
    print(f"  Districts    : {result['district'].nunique()}")
    print(f"  Years        : {result['year'].min()} – {result['year'].max()}")
    print(f"  Crops        : {', '.join(sorted(result['crop_type'].unique()))}")
    print(f"  Output file  : {output_path}")
    print("────────────────────────────────────────────────────────\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ICRISAT CSV to app training format.")
    parser.add_argument("--input",  type=Path, default=DEFAULT_INPUT,  help="Path to icrisat_district.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output path for crop_weather_records.csv")
    args = parser.parse_args()
    convert(args.input, args.output)


if __name__ == "__main__":
    main()
