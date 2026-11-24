"""Plotly figure builders. They return figures only, so they are reusable by any UI."""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from weather_impact.config.settings import CURRENCY_SYMBOL, TARGET_COLUMN

PALETTE = ["#2E7D32", "#F9A825", "#1565C0", "#8E24AA", "#E65100", "#00897B", "#6D4C41", "#AD1457"]
RISK_COLORS = {"Low": "#2E7D32", "Moderate": "#F9A825", "High": "#EF6C00", "Severe": "#C62828"}
POSITIVE_COLOR = "#2E7D32"
NEGATIVE_COLOR = "#C62828"
NEUTRAL_COLOR = "#607D8B"
TEMPLATE = "plotly_white"

FRIENDLY_NAMES = {
    "temperature_c": "Temperature",
    "rainfall_mm": "Rainfall",
    "humidity_pct": "Humidity",
    "temp_humidity_index": "Heat–humidity index",
    "crop_type": "Crop",
    "season": "Season",
    "temperature": "Temperature",
    "rainfall": "Rainfall",
    "humidity": "Humidity",
}


def friendly_name(feature: str) -> str:
    return FRIENDLY_NAMES.get(feature, feature.replace("_", " ").capitalize())


def _style(fig: go.Figure, title: str, height: int = 380) -> go.Figure:
    fig.update_layout(
        template=TEMPLATE,
        title=dict(text=title, x=0.0, xanchor="left", font=dict(size=16)),
        height=height,
        margin=dict(l=16, r=16, t=56, b=64),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0),
        font=dict(family="Inter, system-ui, -apple-system, Segoe UI, sans-serif", size=13),
        hoverlabel=dict(font_size=13),
    )
    return fig


def yield_trend(df: pd.DataFrame) -> go.Figure:
    """Average yield per year, one line per crop."""
    agg = df.groupby(["year", "crop_type"], as_index=False)[TARGET_COLUMN].mean()
    fig = go.Figure()
    for index, (crop, group) in enumerate(agg.groupby("crop_type")):
        fig.add_trace(
            go.Scatter(
                x=group["year"],
                y=group[TARGET_COLUMN],
                mode="lines+markers",
                name=crop,
                line=dict(color=PALETTE[index % len(PALETTE)], width=2.5),
                marker=dict(size=7),
                hovertemplate=f"{crop}<br>Year %{{x}}<br>%{{y:.2f}} t/ha<extra></extra>",
            )
        )
    fig.update_xaxes(title="Year")
    fig.update_yaxes(title="Average yield (t/ha)")
    return _style(fig, "Yield trend by crop")


def rainfall_vs_yield(df: pd.DataFrame, max_points: int = 1500) -> go.Figure:
    """Scatter of rainfall against yield, coloured by crop."""
    sample = df.sample(n=min(max_points, len(df)), random_state=0) if len(df) > max_points else df
    fig = go.Figure()
    for index, (crop, group) in enumerate(sample.groupby("crop_type")):
        fig.add_trace(
            go.Scatter(
                x=group["rainfall_mm"],
                y=group[TARGET_COLUMN],
                mode="markers",
                name=crop,
                marker=dict(color=PALETTE[index % len(PALETTE)], size=7, opacity=0.6),
                hovertemplate="Rainfall %{x:.0f} mm<br>Yield %{y:.2f} t/ha<extra>" + crop + "</extra>",
            )
        )
    fig.update_xaxes(title="Seasonal rainfall (mm)")
    fig.update_yaxes(title="Yield (t/ha)")
    return _style(fig, "Rainfall and yield")


def season_yield_box(df: pd.DataFrame) -> go.Figure:
    """Yield distribution by season, split by crop."""
    fig = go.Figure()
    for index, (crop, group) in enumerate(df.groupby("crop_type")):
        fig.add_trace(
            go.Box(
                x=group["season"],
                y=group[TARGET_COLUMN],
                name=crop,
                marker_color=PALETTE[index % len(PALETTE)],
                boxmean=True,
            )
        )
    fig.update_layout(boxmode="group")
    fig.update_yaxes(title="Yield (t/ha)")
    return _style(fig, "Yield spread by season")


def crop_season_heatmap(df: pd.DataFrame) -> go.Figure:
    """Average yield for each crop and season combination."""
    pivot = df.pivot_table(index="crop_type", columns="season", values=TARGET_COLUMN, aggfunc="mean")
    text = [[f"{v:.2f}" if pd.notna(v) else "–" for v in row] for row in pivot.values]
    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale="Greens",
            text=text,
            texttemplate="%{text}",
            colorbar=dict(title="t/ha"),
            hovertemplate="%{y} · %{x}<br>%{z:.2f} t/ha<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    return _style(fig, "Average yield: crop × season")


def risk_gauge(score: float, level: str) -> go.Figure:
    """Speedometer-style gauge for the 0–100 Environmental Risk Score."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number=dict(suffix="", font=dict(size=40)),
            title=dict(text=f"<b>{level}</b> risk", font=dict(size=16)),
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=1, tickcolor="#90A4AE"),
                bar=dict(color=RISK_COLORS.get(level, NEUTRAL_COLOR), thickness=0.28),
                bgcolor="white",
                borderwidth=0,
                steps=[
                    dict(range=[0, 25], color="#E8F5E9"),
                    dict(range=[25, 50], color="#FFF8E1"),
                    dict(range=[50, 75], color="#FFF3E0"),
                    dict(range=[75, 100], color="#FFEBEE"),
                ],
            ),
        )
    )
    fig.update_layout(
        template=TEMPLATE,
        height=280,
        margin=dict(l=24, r=24, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif"),
    )
    return fig


def severity_bars(severity: Mapping[str, float], direction: Mapping[str, str]) -> go.Figure:
    """How strongly each factor contributes to stress (0 = optimal, 100% = fully outside the band)."""
    factors = list(severity.keys())
    values = [severity[f] * 100 for f in factors]
    labels = []
    colors = []
    for factor, value in zip(factors, values, strict=True):
        status = direction.get(factor, "ok")
        labels.append(f"{friendly_name(factor)} ({'optimal' if status == 'ok' else status})")
        if value == 0:
            colors.append(POSITIVE_COLOR)
        elif value < 50:
            colors.append("#EF6C00")
        else:
            colors.append(NEGATIVE_COLOR)
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.0f}%" for v in values],
            textposition="outside",
            hovertemplate="%{y}<br>Stress %{x:.0f}%<extra></extra>",
        )
    )
    fig.update_xaxes(range=[0, 115], title="Stress level (% outside optimal band)")
    fig.update_yaxes(autorange="reversed")
    return _style(fig, "Stress by factor", height=300)


def contribution_bars(contributions: pd.DataFrame, unit: str = "t/ha") -> go.Figure:
    """Horizontal bars: how each factor pushed the predicted yield up or down."""
    data = contributions.iloc[::-1]
    labels = [friendly_name(f) for f in data["feature"]]
    values = data["shap_value"].astype(float).tolist()
    colors = [POSITIVE_COLOR if v >= 0 else NEGATIVE_COLOR for v in values]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.2f}" for v in values],
            textposition="outside",
            hovertemplate="%{y}<br>Effect %{x:+.3f} " + unit + "<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_width=1, line_color="#90A4AE")
    # Pad the axis so value labels at the bar ends are never clipped.
    limit = max(abs(v) for v in values) * 1.35 if values and max(abs(v) for v in values) > 0 else 1.0
    fig.update_xaxes(
        title=f"Effect on predicted yield ({unit})", zeroline=False, range=[-limit, limit]
    )
    return _style(fig, "What drives this prediction", height=320)


def sensitivity_line(
    x: Sequence[float], y: Sequence[float], current_x: float, xlabel: str, ylabel: str
) -> go.Figure:
    """Predicted yield as one factor changes, with the current value marked."""
    fig = go.Figure(
        go.Scatter(
            x=list(x),
            y=list(y),
            mode="lines+markers",
            line=dict(color=POSITIVE_COLOR, width=3),
            marker=dict(size=7),
            hovertemplate=f"{xlabel} %{{x:+.1f}}<br>Yield %{{y:.2f}} t/ha<extra></extra>",
            name="Predicted yield",
        )
    )
    fig.add_vline(x=current_x, line_dash="dot", line_color=NEUTRAL_COLOR, annotation_text="Now")
    fig.update_xaxes(title=xlabel)
    fig.update_yaxes(title=ylabel)
    return _style(fig, "Sensitivity: yield across the range", height=340)


def comparison_bars(
    categories: Sequence[str], series: Mapping[str, Sequence[float]], y_title: str, title: str
) -> go.Figure:
    """Grouped bars comparing several series across categories."""
    fig = go.Figure()
    for index, (name, values) in enumerate(series.items()):
        fig.add_trace(
            go.Bar(
                name=name,
                x=list(categories),
                y=list(values),
                marker_color=[PALETTE[0], PALETTE[1], PALETTE[2]][index % 3],
                text=[f"{v:.2f}" for v in values],
                textposition="outside",
            )
        )
    fig.update_layout(barmode="group")
    fig.update_yaxes(title=y_title)
    return _style(fig, title, height=340)


def importance_bars(importance: pd.DataFrame) -> go.Figure:
    """Mean absolute SHAP value per feature: overall influence on yield predictions."""
    data = importance.iloc[::-1]
    labels = [friendly_name(f) for f in data["feature"]]
    fig = go.Figure(
        go.Bar(
            x=data["mean_abs_shap"].astype(float),
            y=labels,
            orientation="h",
            marker_color=PALETTE[0],
            text=[f"{v:.2f}" for v in data["mean_abs_shap"]],
            textposition="outside",
            hovertemplate="%{y}<br>Average impact %{x:.3f} t/ha<extra></extra>",
        )
    )
    fig.update_xaxes(title="Average absolute impact (t/ha)")
    return _style(fig, "Overall factor importance", height=320)


def allocation_bars(allocations: pd.DataFrame) -> go.Figure:
    """Money spent on each mitigation action."""
    data = allocations.sort_values("cost")
    fig = go.Figure(
        go.Bar(
            x=data["cost"].astype(float),
            y=data["action"],
            orientation="h",
            marker_color=PALETTE[2],
            text=[f"{CURRENCY_SYMBOL}{v:,.0f}" for v in data["cost"]],
            textposition="outside",
            hovertemplate="%{y}<br>Spend " + CURRENCY_SYMBOL + "%{x:,.0f}<extra></extra>",
        )
    )
    fig.update_xaxes(title=f"Spend ({CURRENCY_SYMBOL})")
    return _style(fig, "Where the budget goes", height=300)


def scenario_comparison(names: Sequence[str], yields: Sequence[float], risks: Sequence[float]) -> go.Figure:
    """Side-by-side yield and risk for saved scenarios."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Predicted yield (t/ha)", "Risk score (0–100)"))
    fig.add_trace(
        go.Bar(x=list(names), y=list(yields), marker_color=PALETTE[0], text=[f"{v:.2f}" for v in yields],
               textposition="outside", showlegend=False),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=list(names), y=list(risks), marker_color=PALETTE[4], text=[f"{v:.1f}" for v in risks],
               textposition="outside", showlegend=False),
        row=1,
        col=2,
    )
    fig.update_yaxes(range=[0, 115], row=1, col=2)
    return _style(fig, "Scenario comparison", height=360)


def forecast_chart(days: Sequence[Mapping[str, float]]) -> go.Figure:
    """Daily high and low temperature (lines) with rainfall (bars) on a second axis."""
    dates = [d["date"] for d in days]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=dates,
            y=[d["precipitation_mm"] for d in days],
            name="Rainfall (mm)",
            marker_color="#90CAF9",
            opacity=0.85,
            hovertemplate="%{x}<br>Rain %{y:.1f} mm<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=[d["temp_max_c"] for d in days],
            name="High (°C)",
            mode="lines+markers",
            line=dict(color="#EF6C00", width=3),
            hovertemplate="%{x}<br>High %{y:.1f} °C<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=[d["temp_min_c"] for d in days],
            name="Low (°C)",
            mode="lines+markers",
            line=dict(color="#1565C0", width=3),
            hovertemplate="%{x}<br>Low %{y:.1f} °C<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.update_yaxes(title_text="Temperature (°C)", secondary_y=False)
    fig.update_yaxes(title_text="Rainfall (mm)", secondary_y=True, showgrid=False)
    return _style(fig, "7-day forecast", height=360)
