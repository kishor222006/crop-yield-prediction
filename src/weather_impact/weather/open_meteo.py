"""Live weather from the Open-Meteo API (free, no API key required).

Docs: https://open-meteo.com/en/docs and https://open-meteo.com/en/docs/geocoding-api

The model uses seasonal averages, so this module summarises recent weather
(the last 7 days) into the same units the model expects, and returns a short
forecast for context. Seasonal rainfall totals are not available from a live
feed, so the 7-day rainfall total is reported separately for the user to judge.
"""

from __future__ import annotations

import json
import math
import ssl
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from weather_impact.utils.exceptions import WeatherServiceError

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
DEFAULT_TIMEOUT_S = 10.0
PAST_DAYS = 7
FORECAST_DAYS = 7

JsonFetcher = Callable[[str], Dict[str, Any]]


@dataclass(frozen=True)
class Location:
    name: str
    admin1: str
    country: str
    latitude: float
    longitude: float

    @property
    def label(self) -> str:
        parts = [self.name, self.admin1, self.country]
        return ", ".join(p for p in parts if p)


@dataclass(frozen=True)
class ForecastDay:
    date: str
    temp_max_c: float
    temp_min_c: float
    precipitation_mm: float
    precipitation_probability_pct: Optional[float]


@dataclass(frozen=True)
class WeatherSnapshot:
    """Live conditions plus a 7-day summary in the model's units."""

    location: Location
    observed_at: str
    current_temperature_c: float
    current_humidity_pct: float
    current_precipitation_mm: float
    avg_temperature_7d_c: float
    avg_humidity_7d_pct: float
    rainfall_7d_mm: float
    forecast: List[ForecastDay] = field(default_factory=list)


def _ssl_context() -> ssl.SSLContext:
    """TLS context that trusts the certifi CA bundle.

    Python installed from python.org on macOS ships without system root
    certificates, which makes every HTTPS call fail with
    CERTIFICATE_VERIFY_FAILED. certifi provides a maintained CA bundle, so we
    use it when available and fall back to the platform default otherwise.
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # certifi missing or unreadable: use the platform default
        return ssl.create_default_context()


def _default_fetch(url: str) -> Dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "weather-impact/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=DEFAULT_TIMEOUT_S, context=_ssl_context()) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise WeatherServiceError(
                "Open-Meteo rate limit reached. Wait a minute and try again."
            ) from exc
        raise WeatherServiceError(f"Live weather service returned HTTP {exc.code}.") from exc
    except urllib.error.URLError as exc:
        reason = str(exc.reason)
        if "CERTIFICATE_VERIFY_FAILED" in reason:
            raise WeatherServiceError(
                "Live weather service is unreachable: SSL certificate check failed. "
                "Run `pip install --upgrade certifi` (on macOS also run "
                "'Install Certificates.command' from your Python folder)."
            ) from exc
        raise WeatherServiceError(f"Live weather service is unreachable: {reason}") from exc
    except (TimeoutError, json.JSONDecodeError) as exc:
        raise WeatherServiceError("Live weather service returned an unexpected response.") from exc


def _validate_coordinates(latitude: float, longitude: float) -> None:
    if not (math.isfinite(latitude) and -90.0 <= latitude <= 90.0):
        raise WeatherServiceError("Latitude must be between -90 and 90.")
    if not (math.isfinite(longitude) and -180.0 <= longitude <= 180.0):
        raise WeatherServiceError("Longitude must be between -180 and 180.")


def search_locations(query: str, count: int = 5, fetch: JsonFetcher = _default_fetch) -> List[Location]:
    """Find places by name. Returns an empty list when nothing matches."""
    query = query.strip()
    if len(query) < 2:
        raise WeatherServiceError("Type at least 2 characters to search for a place.")
    params = urllib.parse.urlencode({"name": query, "count": count, "language": "en", "format": "json"})
    payload = fetch(f"{GEOCODING_URL}?{params}")
    results = payload.get("results") or []
    return [
        Location(
            name=str(r.get("name", "")),
            admin1=str(r.get("admin1", "") or ""),
            country=str(r.get("country", "") or ""),
            latitude=float(r["latitude"]),
            longitude=float(r["longitude"]),
        )
        for r in results
    ]


def _average(values: List[Optional[float]]) -> float:
    clean = [v for v in values if v is not None]
    if not clean:
        raise WeatherServiceError("The weather service returned no usable hourly values.")
    return sum(clean) / len(clean)


def summarise(payload: Dict[str, Any], location: Location) -> WeatherSnapshot:
    """Turn a raw forecast payload into a WeatherSnapshot. Separated from the network call for testing."""
    try:
        current = payload["current"]
        hourly = payload["hourly"]
        daily = payload["daily"]
        now = current["time"]
    except KeyError as exc:
        raise WeatherServiceError(f"Weather response is missing the field {exc}.") from exc

    times: List[str] = hourly["time"]
    past_idx = [i for i, t in enumerate(times) if t < now][-PAST_DAYS * 24:]
    if not past_idx:
        raise WeatherServiceError("No past hourly data was returned for the last 7 days.")

    temps = [hourly["temperature_2m"][i] for i in past_idx]
    humid = [hourly["relative_humidity_2m"][i] for i in past_idx]
    rain = [hourly["precipitation"][i] or 0.0 for i in past_idx]

    forecast = []
    for i, day in enumerate(daily["time"]):
        if day < now[:10]:
            continue
        prob = (daily.get("precipitation_probability_max") or [None] * len(daily["time"]))[i]
        forecast.append(
            ForecastDay(
                date=day,
                temp_max_c=float(daily["temperature_2m_max"][i]),
                temp_min_c=float(daily["temperature_2m_min"][i]),
                precipitation_mm=float(daily["precipitation_sum"][i] or 0.0),
                precipitation_probability_pct=None if prob is None else float(prob),
            )
        )

    return WeatherSnapshot(
        location=location,
        observed_at=now,
        current_temperature_c=float(current["temperature_2m"]),
        current_humidity_pct=float(current["relative_humidity_2m"]),
        current_precipitation_mm=float(current.get("precipitation") or 0.0),
        avg_temperature_7d_c=round(_average(temps), 1),
        avg_humidity_7d_pct=round(_average(humid), 1),
        rainfall_7d_mm=round(sum(rain), 1),
        forecast=forecast[:FORECAST_DAYS],
    )


def fetch_weather(location: Location, fetch: JsonFetcher = _default_fetch) -> WeatherSnapshot:
    """Fetch live conditions, the last 7 days, and a 7-day forecast for a location."""
    _validate_coordinates(location.latitude, location.longitude)
    params = urllib.parse.urlencode(
        {
            "latitude": f"{location.latitude:.4f}",
            "longitude": f"{location.longitude:.4f}",
            "current": "temperature_2m,relative_humidity_2m,precipitation",
            "hourly": "temperature_2m,relative_humidity_2m,precipitation",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max",
            "past_days": PAST_DAYS,
            "forecast_days": FORECAST_DAYS,
            "timezone": "auto",
        }
    )
    payload = fetch(f"{FORECAST_URL}?{params}")
    return summarise(payload, location)

