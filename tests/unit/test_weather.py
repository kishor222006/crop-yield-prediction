import urllib.error
from datetime import datetime, timedelta

import pytest

from weather_impact.utils.exceptions import WeatherServiceError
from weather_impact.weather import open_meteo
from weather_impact.weather.open_meteo import Location, fetch_weather, search_locations, summarise

LOC = Location(name="Mysuru", admin1="Karnataka", country="India", latitude=12.3, longitude=76.6)
NOW = "2026-10-08T12:00"


def _payload(temp=25.0, humidity=70.0, rain=1.0):
    start = datetime.fromisoformat(NOW) - timedelta(days=7)
    times = [(start + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M") for h in range(14 * 24)]
    return {
        "current": {"time": NOW, "temperature_2m": 26.5, "relative_humidity_2m": 60.0, "precipitation": 0.0},
        "hourly": {
            "time": times,
            "temperature_2m": [temp] * len(times),
            "relative_humidity_2m": [humidity] * len(times),
            "precipitation": [rain] * len(times),
        },
        "daily": {
            "time": ["2026-10-07", "2026-10-08", "2026-10-09"],
            "temperature_2m_max": [30.0, 31.0, 32.0],
            "temperature_2m_min": [20.0, 21.0, 22.0],
            "precipitation_sum": [0.0, 2.0, None],
            "precipitation_probability_max": [10.0, 40.0, 70.0],
        },
    }


def test_summary_averages_last_seven_days_only():
    snap = summarise(_payload(temp=25.0, humidity=70.0), LOC)
    assert snap.avg_temperature_7d_c == pytest.approx(25.0)
    assert snap.avg_humidity_7d_pct == pytest.approx(70.0)
    # 168 past hours at 1 mm each; future hours must not be counted
    assert snap.rainfall_7d_mm == pytest.approx(168.0)


def test_current_values_come_from_current_block():
    snap = summarise(_payload(), LOC)
    assert snap.current_temperature_c == 26.5
    assert snap.current_humidity_pct == 60.0
    assert snap.observed_at == NOW


def test_forecast_excludes_past_days_and_handles_missing_values():
    snap = summarise(_payload(), LOC)
    assert [d.date for d in snap.forecast] == ["2026-10-08", "2026-10-09"]
    assert snap.forecast[1].precipitation_probability_pct == 70.0
    assert snap.forecast[0].precipitation_mm == 2.0
    assert snap.forecast[1].precipitation_mm == 0.0  # missing value treated as zero


def test_missing_field_raises_clear_error():
    payload = _payload()
    del payload["hourly"]
    with pytest.raises(WeatherServiceError, match="hourly"):
        summarise(payload, LOC)


def test_invalid_coordinates_rejected_before_any_request():
    bad = Location(name="x", admin1="", country="", latitude=95.0, longitude=10.0)
    with pytest.raises(WeatherServiceError, match="Latitude"):
        fetch_weather(bad, fetch=lambda url: pytest.fail("should not call the network"))


def test_fetch_uses_expected_parameters():
    seen = {}

    def fake(url):
        seen["url"] = url
        return _payload()

    fetch_weather(LOC, fetch=fake)
    assert seen["url"].startswith(open_meteo.FORECAST_URL)
    assert "past_days=7" in seen["url"] and "timezone=auto" in seen["url"]


def test_search_parses_results_and_handles_no_match():
    found = search_locations(
        "Mysuru",
        fetch=lambda url: {"results": [{"name": "Mysuru", "admin1": "Karnataka", "country": "India",
                                        "latitude": 12.3, "longitude": 76.6}]},
    )
    assert found[0].label == "Mysuru, Karnataka, India"
    assert search_locations("Nowhere", fetch=lambda url: {}) == []


def test_search_needs_two_characters():
    with pytest.raises(WeatherServiceError):
        search_locations("a", fetch=lambda url: {})


def test_network_failure_becomes_weather_service_error(monkeypatch):
    def boom(*args, **kwargs):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(open_meteo.urllib.request, "urlopen", boom)
    with pytest.raises(WeatherServiceError, match="unreachable"):
        search_locations("Pune")


def test_tls_context_uses_certifi_bundle():
    # Regression: python.org builds of Python on macOS have no system CA certificates,
    # which made every request fail with CERTIFICATE_VERIFY_FAILED.
    import certifi

    ctx = open_meteo._ssl_context()
    assert ctx.verify_mode.name == "CERT_REQUIRED"
    assert len(ctx.get_ca_certs()) > 0
    assert certifi.where()


def test_certificate_failure_gets_an_actionable_message(monkeypatch):
    import ssl

    def boom(*args, **kwargs):
        raise urllib.error.URLError(ssl.SSLCertVerificationError("CERTIFICATE_VERIFY_FAILED"))

    monkeypatch.setattr(open_meteo.urllib.request, "urlopen", boom)
    with pytest.raises(WeatherServiceError, match="certifi"):
        search_locations("Pune")


def test_rate_limit_is_reported_clearly(monkeypatch):
    def limited(*args, **kwargs):
        raise urllib.error.HTTPError("http://x", 429, "Too Many Requests", {}, None)

    monkeypatch.setattr(open_meteo.urllib.request, "urlopen", limited)
    with pytest.raises(WeatherServiceError, match="rate limit"):
        search_locations("Pune")
