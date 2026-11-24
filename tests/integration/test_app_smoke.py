"""End-to-end checks for the Streamlit website, driven through Streamlit's AppTest harness.

These tests need a trained model in models/saved/. They are skipped when it is absent.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
APP_DIR = ROOT / "app"
sys.path.insert(0, str(APP_DIR))

from streamlit.testing.v1 import AppTest  # noqa: E402

from weather_impact.config.settings import DEFAULT_MODEL_NAME, MODEL_DIR  # noqa: E402
from weather_impact.database import db as db_module  # noqa: E402
from weather_impact.database.repository import list_scenarios  # noqa: E402

PAGES = [
    "Home.py",
    "pages/1_Dashboard.py",
    "pages/2_Yield_Prediction.py",
    "pages/3_What_If_Simulator.py",
    "pages/4_Risk_Score.py",
    "pages/5_Explainability.py",
    "pages/6_Saved_Scenarios.py",
]

pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / f"{DEFAULT_MODEL_NAME}.joblib").exists(),
    reason="Train the model first: python scripts/train_model.py",
)


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    monkeypatch.setattr(db_module, "SCENARIO_DB_PATH", path)
    return path


def _run(page):
    return AppTest.from_file(str(APP_DIR / page), default_timeout=180).run()


def _errors(at):
    return [e.value for e in at.exception] + [e.value for e in at.error]


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_exceptions(page, temp_db):
    at = _run(page)
    assert _errors(at) == []


def test_prediction_and_save_flow(temp_db):
    at = _run("pages/2_Yield_Prediction.py")
    at.selectbox(key="crop_choice").set_value("Wheat").run()
    assert at.selectbox(key="season_for_Wheat").value == "Rabi"
    at.button(key="pred_save").click().run()
    assert _errors(at) == []
    saved = list_scenarios(temp_db)
    assert len(saved) == 1 and saved[0].source == "prediction" and saved[0].crop_type == "Wheat"


def test_what_if_reset_and_save_flow(temp_db):
    at = _run("pages/3_What_If_Simulator.py")
    at.slider(key="wi_temp").set_value(8.0).run()
    assert at.slider(key="wi_temp").value == 8.0
    at.button(key="wi_reset").click().run()
    assert at.slider(key="wi_temp").value == 0.0
    at.slider(key="wi_temp").set_value(4.0).run()
    at.button(key="wi_save").click().run()
    assert _errors(at) == []
    assert list_scenarios(temp_db)[0].source == "what_if"


def test_extreme_scenario_shows_message_not_crash(temp_db):
    at = _run("pages/3_What_If_Simulator.py")
    at.selectbox(key="crop_choice").set_value("Rice").run()
    at.slider(key="temperature_Rice_Kharif").set_value(50.0).run()
    at.slider(key="wi_temp").set_value(10.0).run()
    assert [e.value for e in at.exception] == []
    assert any("unrealistic" in e.value for e in at.error)


def test_compare_and_delete_flow(temp_db):
    from weather_impact.database.repository import Scenario, save_scenario

    base = dict(crop_type="Rice", season="Kharif", temperature_c=27.0, rainfall_mm=1100.0,
                humidity_pct=72.0, predicted_yield=5.0, risk_score=0.0, risk_level="Low")
    first = save_scenario(Scenario(name="A", source="prediction", **base), temp_db)
    save_scenario(Scenario(name="B", source="what_if", **base), temp_db)

    at = _run("pages/6_Saved_Scenarios.py")
    at.multiselect(key="compare_ids").set_value([first]).run()
    assert _errors(at) == []
    at.multiselect(key="delete_ids").set_value([first]).run()
    at.button(key="delete_btn").click().run()
    assert _errors(at) == []
    assert [s.name for s in list_scenarios(temp_db)] == ["B"]


def test_empty_dashboard_filter_shows_info(temp_db):
    at = _run("pages/1_Dashboard.py")
    at.multiselect(key="dash_crop").set_value([]).run()
    assert _errors(at) == []
    assert any("No records match" in i.value for i in at.info)


def test_risk_planner_caps_reduction_at_score(temp_db):
    at = _run("pages/4_Risk_Score.py")
    assert _errors(at) == []
    assert any(m.label == "Estimated risk after" for m in at.metric)


def test_live_page_renders_with_stubbed_service(temp_db, monkeypatch):
    import components.loaders as loaders
    from tests.unit.test_weather import LOC, _payload

    from weather_impact.weather.open_meteo import summarise

    snapshot = summarise(_payload(temp=28.4, humidity=66.0), LOC)
    monkeypatch.setattr(loaders, "live_snapshot", lambda location: snapshot)
    at = _run("pages/7_Live_Weather.py")
    assert _errors(at) == []
    assert any(m.label == "Avg temp, last 7 days" for m in at.metric)


def test_apply_live_averages_updates_sliders(temp_db, monkeypatch):
    import components.loaders as loaders
    from tests.unit.test_weather import LOC, _payload

    from weather_impact.weather.open_meteo import summarise

    snapshot = summarise(_payload(temp=28.4, humidity=66.0), LOC)
    monkeypatch.setattr(loaders, "live_snapshot", lambda location: snapshot)
    at = _run("pages/7_Live_Weather.py")
    at.selectbox(key="crop_choice").set_value("Rice").run()
    at.button(key="apply_live").click().run()
    assert _errors(at) == []
    assert at.slider(key="temperature_Rice_Kharif").value == 28.5
    assert at.slider(key="humidity_Rice_Kharif").value == 66.0
