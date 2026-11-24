"""Drives the Live Weather view of app.py through Streamlit's AppTest, with the network faked."""

import io
import json
import ssl
import urllib.error
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from weather_impact.config.settings import DEFAULT_MODEL_NAME, MODEL_DIR

APP = str(Path(__file__).resolve().parents[2] / "app.py")

pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / f"{DEFAULT_MODEL_NAME}.joblib").exists(),
    reason="Train the model first: python scripts/train_model.py",
)

NOW = datetime(2026, 11, 22, 14, 0)


def _forecast_payload():
    start = NOW - timedelta(days=7)
    times = [(start + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M") for h in range(14 * 24)]
    days = [(NOW + timedelta(days=d)).strftime("%Y-%m-%d") for d in range(-7, 7)]
    return {
        "current": {"time": NOW.strftime("%Y-%m-%dT%H:%M"), "temperature_2m": 26.0,
                    "relative_humidity_2m": 61.0, "precipitation": 0.2},
        "hourly": {"time": times, "temperature_2m": [24.0] * len(times),
                   "relative_humidity_2m": [68.0] * len(times), "precipitation": [0.05] * len(times)},
        "daily": {"time": days, "temperature_2m_max": [31.0] * len(days), "temperature_2m_min": [19.0] * len(days),
                  "precipitation_sum": [1.0] * len(days), "precipitation_probability_max": [30.0] * len(days)},
    }


def _geocode_payload():
    return {"results": [
        {"name": "Chitradurga", "admin1": "Karnataka", "country": "India", "latitude": 14.23, "longitude": 76.4},
        {"name": "Chitradurga", "admin1": "Odisha", "country": "India", "latitude": 20.1, "longitude": 85.1},
    ]}


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture(autouse=True)
def _fresh_cache():
    """st.cache_data lives for the whole process, so clear it to keep tests independent."""
    st.cache_data.clear()
    yield
    st.cache_data.clear()


@pytest.fixture()
def fake_network(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout=None, context=None):
        url = request.full_url
        calls.append(url)
        body = _geocode_payload() if "geocoding-api" in url else _forecast_payload()
        return _Resp(json.dumps(body).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return calls


def _open_live_tab():
    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["view"] = "live weather"
    return at.run()


def _problems(at):
    return [e.value for e in at.exception] + [e.value for e in at.error]


def test_tab_shows_live_values_and_prediction(fake_network):
    at = _open_live_tab()
    assert _problems(at) == []
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Temperature now"] == "26.0 °C"
    assert metrics["Humidity now"] == "61%"
    assert metrics["Rain, last 7 days"] == "8.4 mm"  # 168 hours x 0.05 mm
    assert "Predicted Yield" in metrics and "Risk Level" in metrics
    assert len(fake_network) == 1  # one forecast call, no search


def test_repeat_runs_use_the_cache(fake_network):
    at = _open_live_tab()
    at.run()
    assert len(fake_network) == 1


def test_search_picks_a_place_and_fetches_it(fake_network):
    at = _open_live_tab()
    at.text_input(key="lw_search").set_value("Chitradurga, Karnataka").run()
    assert _problems(at) == []
    place = at.selectbox(key="lw_place_chitradurga, karnataka")
    assert len(place.options) == 1 and "Karnataka" in place.options[0]  # region hint filtered the Odisha match
    assert any("latitude=14.2300" in url for url in fake_network)


def test_ssl_failure_is_reported_not_raised(monkeypatch):
    def boom(*args, **kwargs):
        raise urllib.error.URLError(ssl.SSLCertVerificationError("CERTIFICATE_VERIFY_FAILED"))

    monkeypatch.setattr("urllib.request.urlopen", boom)
    at = _open_live_tab()
    assert at.exception == []
    assert any("unavailable" in e.value and "certifi" in e.value for e in at.error)


def test_save_scenario_from_live_tab(fake_network):
    at = _open_live_tab()
    at.text_input(key="lw_save_name").set_value("Live test").run()
    at.button(key="lw_save").click().run()
    assert _problems(at) == []
    saved = at.session_state["saved"]
    assert saved and saved[0]["name"] == "Live test"
    assert saved[0]["temperature_c"] == 24.0 and saved[0]["humidity_pct"] == 68.0


def test_open_in_yield_prediction_carries_values_over(fake_network):
    at = _open_live_tab()
    at.button(key="lw_open").click().run()
    assert _problems(at) == []
    assert at.session_state["view"] == "yield prediction"
    assert at.slider(key="field_temp").value == 24.0
    assert at.slider(key="field_hum").value == 68.0
