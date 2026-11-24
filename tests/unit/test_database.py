import pytest

from weather_impact.database.repository import (
    Scenario,
    delete_scenarios,
    list_scenarios,
    save_scenario,
)
from weather_impact.utils.exceptions import InvalidInputError


def _scenario(name="Test", source="prediction"):
    return Scenario(
        name=name,
        source=source,
        crop_type="Rice",
        season="Kharif",
        temperature_c=27.0,
        rainfall_mm=1100.0,
        humidity_pct=72.0,
        predicted_yield=5.1,
        risk_score=0.0,
        risk_level="Low",
    )


@pytest.fixture()
def db_path(tmp_path):
    return tmp_path / "scenarios.db"


def test_save_and_list_round_trip(db_path):
    new_id = save_scenario(_scenario("First"), db_path)
    saved = list_scenarios(db_path)
    assert len(saved) == 1
    assert saved[0].id == new_id
    assert saved[0].name == "First"
    assert saved[0].created_at.endswith("+00:00")


def test_list_is_newest_first(db_path):
    save_scenario(_scenario("Old"), db_path)
    save_scenario(_scenario("New"), db_path)
    assert [s.name for s in list_scenarios(db_path)] == ["New", "Old"]


def test_name_is_trimmed_and_required(db_path):
    save_scenario(_scenario("  Padded  "), db_path)
    assert list_scenarios(db_path)[0].name == "Padded"
    with pytest.raises(InvalidInputError):
        save_scenario(_scenario("   "), db_path)


def test_overlong_name_rejected(db_path):
    with pytest.raises(InvalidInputError):
        save_scenario(_scenario("x" * 200), db_path)


def test_invalid_source_rejected(db_path):
    with pytest.raises(InvalidInputError):
        save_scenario(_scenario(source="bogus"), db_path)


def test_delete_removes_only_selected(db_path):
    a = save_scenario(_scenario("A"), db_path)
    b = save_scenario(_scenario("B"), db_path)
    removed = delete_scenarios([a], db_path)
    assert removed == 1
    assert [s.id for s in list_scenarios(db_path)] == [b]


def test_delete_with_no_ids_is_a_noop(db_path):
    assert delete_scenarios([], db_path) == 0
