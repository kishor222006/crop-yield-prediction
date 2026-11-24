import pandas as pd
import plotly.graph_objects as go
import pytest

from weather_impact.data.cleaning import clean_dataset
from weather_impact.risk.risk_score import compute_risk
from weather_impact.visualization import plots


@pytest.fixture(scope="module")
def clean_df(synthetic_df):
    df, _ = clean_dataset(synthetic_df)
    return df


def test_dashboard_figures_build(clean_df):
    builders = (plots.yield_trend, plots.rainfall_vs_yield, plots.season_yield_box, plots.crop_season_heatmap)
    for builder in builders:
        fig = builder(clean_df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


def test_risk_figures_build(good_conditions):
    risk = compute_risk({**good_conditions, "temperature_c": 40.0})
    assert isinstance(plots.risk_gauge(risk.score, risk.level), go.Figure)
    severity = plots.severity_bars(risk.severity, risk.direction)
    assert list(severity.data[0].x) == pytest.approx([v * 100 for v in risk.severity.values()])


def test_contribution_bars_colour_by_sign():
    contributions = pd.DataFrame(
        [{"feature": "rainfall_mm", "value": 1.0, "shap_value": 0.5},
         {"feature": "temperature_c", "value": 2.0, "shap_value": -0.3}]
    )
    fig = plots.contribution_bars(contributions)
    colours = list(fig.data[0].marker.color)
    assert colours == [plots.NEGATIVE_COLOR, plots.POSITIVE_COLOR]


def test_friendly_names_fallback():
    assert plots.friendly_name("unknown_factor") == "Unknown factor"
    assert plots.friendly_name("rainfall_mm") == "Rainfall"


def test_sensitivity_and_comparison_figures_build():
    assert isinstance(plots.sensitivity_line([0, 1], [1.0, 2.0], 0.5, "x", "y"), go.Figure)
    assert isinstance(
        plots.comparison_bars(["Yield"], {"A": [1.0], "B": [2.0]}, "t/ha", "t"), go.Figure
    )
    assert isinstance(plots.scenario_comparison(["#1"], [2.0], [30.0]), go.Figure)
