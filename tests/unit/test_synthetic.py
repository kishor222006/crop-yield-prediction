import pytest

from weather_impact.config.crops import CROP_PROFILES, get_profile
from weather_impact.data.synthetic import generate_synthetic_dataset, yield_response
from weather_impact.data.validation import validate_dataset


def test_generated_dataset_is_valid(synthetic_df):
    validate_dataset(synthetic_df)


def test_generation_is_reproducible():
    a = generate_synthetic_dataset(n_rows=200, seed=5)
    b = generate_synthetic_dataset(n_rows=200, seed=5)
    assert a.equals(b)


def test_yield_is_highest_at_optimal_conditions():
    profile = get_profile("Rice", "Kharif")
    optimal = yield_response(profile, 27.0, 1100.0, 72.0)
    drought = yield_response(profile, 27.0, 200.0, 72.0)
    heat = yield_response(profile, 45.0, 1100.0, 72.0)
    assert optimal > drought
    assert optimal > heat
    assert optimal == pytest.approx(profile.base_yield_t_per_ha, rel=0.05)


def test_all_crop_seasons_are_generated_when_enough_rows():
    df = generate_synthetic_dataset(n_rows=2000, seed=1)
    assert set(zip(df["crop_type"], df["season"], strict=True)) == set(CROP_PROFILES)


@pytest.mark.parametrize("n_rows", [0, -5])
def test_invalid_row_count_rejected(n_rows):
    with pytest.raises(ValueError):
        generate_synthetic_dataset(n_rows=n_rows)
