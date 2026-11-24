"""Fetch real weather data from NASA POWER API and join it to ICRISAT yield data.

For each district we pull monthly T2M (temperature), PRECTOTCORR (precipitation),
and RH2M (humidity) from 1966-2017.  Seasonal averages are then computed per
crop's growing season and merged with the ICRISAT yield records.

Usage:
    python scripts/fetch_nasa_weather.py                  # full run
    python scripts/fetch_nasa_weather.py --fast           # 5 districts only (quick test)
    python scripts/fetch_nasa_weather.py --resume         # skip already-fetched districts

Output:
    data/raw/crop_weather_records.csv   (replaces the synthetic-weather version)
    data/raw/nasa_weather_cache.csv     (raw monthly weather; safe to keep for reruns)

NASA POWER API docs: https://power.larc.nasa.gov/docs/
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT          = Path(__file__).resolve().parents[1]
ICRISAT_PATH  = ROOT / "data" / "raw" / "icrisat_district.csv"
COORDS_PATH   = ROOT / "data" / "raw" / "district_coords.csv"
CACHE_PATH    = ROOT / "data" / "raw" / "nasa_weather_cache.csv"
OUTPUT_PATH   = ROOT / "data" / "raw" / "crop_weather_records.csv"

# ---------------------------------------------------------------------------
# NASA POWER config
# ---------------------------------------------------------------------------
NASA_URL      = "https://power.larc.nasa.gov/api/temporal/monthly/point"
PARAMETERS    = "T2M,PRECTOTCORR,RH2M"
START_YEAR    = 1981   # NASA POWER monthly data starts at 1981-01-01
END_YEAR      = 2017
# Polite delay between API calls (seconds).  NASA asks for ≤ 1 req/s.
REQUEST_DELAY = 1.1

# ---------------------------------------------------------------------------
# Season → months mapping (Indian agricultural seasons)
#   Kharif  = monsoon sown, Jun–Oct harvest window → use Jun-Sep growing months
#   Rabi    = winter sown, Nov–Mar harvest window  → use Oct-Feb growing months
# ---------------------------------------------------------------------------
SEASON_MONTHS = {
    "Kharif": [6, 7, 8, 9],       # Jun Jul Aug Sep
    "Rabi":   [10, 11, 12, 1, 2], # Oct Nov Dec Jan Feb
}

# Crops we extract from ICRISAT and their seasons
CROP_MAP: dict[str, tuple[str, str]] = {
    "RICE YIELD (Kg per ha)":          ("Rice",      "Kharif"),
    "WHEAT YIELD (Kg per ha)":         ("Wheat",     "Rabi"),
    "MAIZE YIELD (Kg per ha)":         ("Maize",     "Kharif"),
    "FINGER MILLET YIELD (Kg per ha)": ("Ragi",      "Kharif"),
    "GROUNDNUT YIELD (Kg per ha)":     ("Groundnut", "Kharif"),
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_one(lat: float, lon: float, retries: int = 3) -> pd.DataFrame | None:
    """Fetch monthly weather for (lat, lon) from START_YEAR to END_YEAR.

    Returns a DataFrame with columns [year, month, temp_c, rain_mm_day, humidity_pct]
    or None on failure.
    """
    params = {
        "parameters": PARAMETERS,
        "community":  "AG",
        "longitude":  lon,
        "latitude":   lat,
        "start":      START_YEAR,
        "end":        END_YEAR,
        "format":     "JSON",
    }
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(NASA_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            props = data["properties"]["parameter"]
            fill  = data["header"]["fill_value"]  # typically -999

            rows = []
            for key_str, temp in props["T2M"].items():
                year  = int(key_str[:4])
                month = int(key_str[4:6])
                if month == 13:          # NASA month 13 = annual average; skip
                    continue
                rain = props["PRECTOTCORR"].get(key_str, fill)
                rh   = props["RH2M"].get(key_str, fill)
                rows.append({
                    "year":         year,
                    "month":        month,
                    "temp_c":       None if temp == fill else temp,
                    "rain_mm_day":  None if rain == fill else rain,
                    "humidity_pct": None if rh   == fill else rh,
                })
            return pd.DataFrame(rows)
        except Exception as exc:
            log.warning("Attempt %d/%d failed for (%.3f, %.3f): %s", attempt, retries, lat, lon, exc)
            if attempt < retries:
                time.sleep(5 * attempt)
    return None


def _seasonal_avg(monthly: pd.DataFrame, season: str) -> pd.DataFrame:
    """Aggregate monthly rows to one row per year with seasonal averages.

    For Rabi the season straddles two calendar years (Oct year N → Feb year N+1).
    We attribute the season to year N (the sowing year) so it lines up with ICRISAT
    which records harvest year N+1 — we therefore shift Rabi months Oct/Nov/Dec
    back one year when grouping.
    """
    df = monthly.copy()
    months = SEASON_MONTHS[season]

    if season == "Rabi":
        # Oct(10) Nov(11) Dec(12) belong to sowing year; Jan(1) Feb(2) to next year
        # Shift Jan/Feb forward so they group with the previous Oct-Dec
        def _adj_year(row: pd.Series) -> int:
            return row["year"] - 1 if row["month"] in (1, 2) else row["year"]
        df["group_year"] = df.apply(_adj_year, axis=1)
    else:
        df["group_year"] = df["year"]

    season_df = df[df["month"].isin(months)].copy()
    agg = (
        season_df
        .groupby("group_year")[["temp_c", "rain_mm_day", "humidity_pct"]]
        .mean()
        .reset_index()
        .rename(columns={
            "group_year":   "year",
            "temp_c":       "temperature_c",
            "rain_mm_day":  "rainfall_mm_day",
            "humidity_pct": "humidity_pct",
        })
    )
    # Convert daily rainfall → seasonal total (approx days in season)
    n_days = {4: 122, 5: 153}.get(len(months), len(months) * 30)
    agg["rainfall_mm"] = agg["rainfall_mm_day"] * n_days
    agg = agg.drop(columns=["rainfall_mm_day"])
    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast",   action="store_true", help="Fetch only first 5 districts (test mode)")
    parser.add_argument("--resume", action="store_true", help="Skip districts already in the cache")
    args = parser.parse_args()

    # ── load coordinates ────────────────────────────────────────────────────
    coords = pd.read_csv(COORDS_PATH)
    log.info("Loaded %d district coordinates", len(coords))

    # ── load ICRISAT ────────────────────────────────────────────────────────
    icrisat = pd.read_csv(ICRISAT_PATH)
    log.info("Loaded ICRISAT: %d rows", len(icrisat))

    # Normalise district name for joining
    icrisat["dist_name_norm"] = icrisat["Dist Name"].str.strip().str.lower()
    coords["dist_name_norm"]  = coords["dist_name"].str.strip().str.lower()

    # ── load or init cache ──────────────────────────────────────────────────
    if args.resume and CACHE_PATH.exists():
        cache = pd.read_csv(CACHE_PATH)
        already = set(zip(cache["dist_name_norm"], cache["lat"], cache["lon"]))
        log.info("Resuming: cache has %d rows for %d entries", len(cache), len(already))
    else:
        cache = pd.DataFrame()
        already = set()

    # ── fetch loop ──────────────────────────────────────────────────────────
    districts = coords.copy()
    if args.fast:
        districts = districts.head(5)
        log.info("FAST mode: fetching only %d districts", len(districts))

    new_rows = []
    total = len(districts)
    for idx, row in districts.iterrows():
        key = (row["dist_name_norm"], row["latitude"], row["longitude"])
        if key in already:
            log.info("[%d/%d] SKIP (cached): %s", idx + 1, total, row["dist_name"])
            continue

        log.info("[%d/%d] Fetching: %s, %s  (%.3f, %.3f)",
                 idx + 1, total, row["dist_name"], row["state_name"],
                 row["latitude"], row["longitude"])

        monthly = _fetch_one(row["latitude"], row["longitude"])
        if monthly is None:
            log.warning("  → Failed, skipping %s", row["dist_name"])
            time.sleep(REQUEST_DELAY)
            continue

        monthly["dist_name"]      = row["dist_name"]
        monthly["dist_name_norm"] = row["dist_name_norm"]
        monthly["state_name"]     = row["state_name"]
        monthly["lat"]            = row["latitude"]
        monthly["lon"]            = row["longitude"]
        new_rows.append(monthly)

        time.sleep(REQUEST_DELAY)

    # Append new rows to cache and save
    if new_rows:
        new_cache = pd.concat(new_rows, ignore_index=True)
        cache = pd.concat([cache, new_cache], ignore_index=True) if not cache.empty else new_cache
        cache.to_csv(CACHE_PATH, index=False)
        log.info("Cache saved: %d total rows → %s", len(cache), CACHE_PATH)
    else:
        log.info("No new data fetched; using existing cache.")

    if cache.empty:
        log.error("Cache is empty. Run without --resume or check network.")
        return

    # ── build seasonal weather per district per year ─────────────────────
    log.info("Computing seasonal averages …")
    season_frames = []
    for dist_norm, grp in cache.groupby("dist_name_norm"):
        dist_name  = grp["dist_name"].iloc[0]
        state_name = grp["state_name"].iloc[0]
        for season in ("Kharif", "Rabi"):
            seas = _seasonal_avg(grp, season)
            seas["district"]       = dist_name
            seas["dist_name_norm"] = dist_norm
            seas["state"]          = state_name
            seas["season_key"]     = season
            season_frames.append(seas)

    weather = pd.concat(season_frames, ignore_index=True)
    log.info("Seasonal weather rows: %d", len(weather))

    # ── melt ICRISAT to long format and join ────────────────────────────────
    log.info("Merging with ICRISAT …")
    frames = []
    for yield_col, (crop_type, season) in CROP_MAP.items():
        if yield_col not in icrisat.columns:
            log.warning("Column not found in ICRISAT: %s", yield_col)
            continue

        chunk = icrisat[["Year", "Dist Name", "State Name", yield_col]].copy()
        chunk.columns = ["year", "dist_name_orig", "state_icrisat", "yield_kg_per_ha"]
        chunk = chunk[chunk["yield_kg_per_ha"] > 0].copy()
        chunk["yield_t_per_ha"] = chunk["yield_kg_per_ha"] / 1000.0
        chunk = chunk[chunk["yield_t_per_ha"].between(0, 50)]
        chunk["crop_type"]      = crop_type
        chunk["dist_name_norm"] = chunk["dist_name_orig"].str.strip().str.lower()
        chunk["season_key"]     = season

        # Join weather on district + year + season
        merged = chunk.merge(
            weather[["dist_name_norm", "season_key", "year",
                     "temperature_c", "rainfall_mm", "humidity_pct", "state"]],
            on=["dist_name_norm", "season_key", "year"],
            how="inner",
        )
        merged["district"] = merged["dist_name_orig"]
        merged["season"]   = merged["season_key"]
        frames.append(merged)
        log.info("  %-30s → %d rows after join", f"{crop_type} / {season}", len(merged))

    if not frames:
        log.error("Nothing merged. Check district name alignment between coords and ICRISAT.")
        return

    result = pd.concat(frames, ignore_index=True)
    result = result[[
        "year", "district", "state", "crop_type", "season",
        "temperature_c", "rainfall_mm", "humidity_pct", "yield_t_per_ha",
    ]].sort_values(["state", "district", "year", "crop_type"]).reset_index(drop=True)

    result.to_csv(OUTPUT_PATH, index=False)
    log.info("Output saved: %d rows → %s", len(result), OUTPUT_PATH)

    # ── quick correlation check ──────────────────────────────────────────────
    print("\n── Weather-Yield Correlations ─────────────────────────────────")
    corr = result[["temperature_c", "rainfall_mm", "humidity_pct", "yield_t_per_ha"]].corr()
    print(corr["yield_t_per_ha"].drop("yield_t_per_ha").to_string())
    print()
    print(f"  Total rows : {len(result):,}")
    print(f"  Districts  : {result['district'].nunique()}")
    print(f"  States     : {result['state'].nunique()}")
    print(f"  Years      : {result['year'].min()} – {result['year'].max()}")
    print("───────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
