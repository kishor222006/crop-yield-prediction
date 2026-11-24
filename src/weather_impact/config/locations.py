"""Reference locations (district headquarters) for the districts in the dataset."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import pandas as pd

from weather_impact.config.settings import DATA_DIR

DISTRICT_COORDS_PATH = DATA_DIR / "raw" / "district_coords.csv"

# name -> (latitude, longitude). Fallback used when district_coords.csv is absent.
DISTRICT_LOCATIONS: Dict[str, Tuple[float, float]] = {
    "Bengaluru Rural": (13.3409, 77.5421),
    "Mysuru": (12.2958, 76.6394),
    "Tumakuru": (13.3392, 77.1010),
    "Belagavi": (15.8497, 74.4977),
    "Kalaburagi": (17.3297, 76.8343),
    "Hassan": (13.0072, 76.1003),
    "Shivamogga": (13.9299, 75.5681),
    "Dharwad": (15.4589, 75.0078),
}
DEFAULT_LOCATION_NAME = "Bengaluru Rural"


@lru_cache(maxsize=1)
def _coords_table() -> Dict[Tuple[str, str], Tuple[float, float]]:
    """Map (district, state) -> (lat, lon) from district_coords.csv. Empty if the file is missing."""
    if not DISTRICT_COORDS_PATH.exists():
        return {}
    df = pd.read_csv(DISTRICT_COORDS_PATH).dropna(subset=["latitude", "longitude"])
    return {
        (str(r.dist_name), str(r.state_name)): (float(r.latitude), float(r.longitude))
        for r in df.itertuples(index=False)
    }


def get_district_coords(district: str, state: Optional[str] = None) -> Optional[Tuple[float, float]]:
    """Return (latitude, longitude) for a district, or None when it is unknown.

    The state disambiguates district names that exist in more than one state.
    """
    table = _coords_table()
    if state is not None and (district, state) in table:
        return table[(district, state)]
    matches = [coords for (name, _), coords in table.items() if name == district]
    if matches:
        return matches[0]
    return DISTRICT_LOCATIONS.get(district)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in kilometres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * 6371.0088 * math.asin(math.sqrt(a))


def nearest_district(
    latitude: float, longitude: float, candidates: List[Tuple[str, str]]
) -> Optional[Tuple[str, str, float]]:
    """Closest (district, state) from `candidates` to a point, with the distance in km."""
    best: Optional[Tuple[str, str, float]] = None
    for district, state in candidates:
        coords = get_district_coords(district, state)
        if coords is None:
            continue
        km = haversine_km(latitude, longitude, coords[0], coords[1])
        if best is None or km < best[2]:
            best = (district, state, km)
    return best
