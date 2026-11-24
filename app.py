"""
Weather Impact Analysis for Agriculture
Single-Page Application — session-state driven navigation.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from weather_impact.config.crops import CROP_PROFILES, get_profile, seasons_for_crop
from weather_impact.config.locations import DEFAULT_LOCATION_NAME, get_district_coords, nearest_district
from weather_impact.config.settings import (
    CURRENCY_SYMBOL, DEFAULT_MODEL_NAME,
    RAW_DATA_PATH, SAMPLE_DATA_PATH, TARGET_COLUMN,
)
from weather_impact.models.predict import predict_one
from weather_impact.models.registry import load_model, load_metadata
from weather_impact.risk.recommendations import recommendations
from weather_impact.risk.risk_score import compute_risk
from weather_impact.simulation.what_if import SCENARIO_LIMITS, ScenarioChange, run_what_if
from weather_impact.utils.exceptions import ModelNotFoundError, WeatherImpactError, WeatherServiceError
from weather_impact.weather.open_meteo import Location, WeatherSnapshot, fetch_weather, search_locations

try:
    from weather_impact.explainability.shap_explainer import explain_prediction
    _SHAP_OK = True
except Exception:
    _SHAP_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Weather Impact · Agriculture",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
  --p900:#0F172A; --p800:#1E293B; --p700:#334155; --p600:#475569;
  --p500:#64748B; --p400:#94A3B8; --p300:#CBD5E1; --p200:#E2E8F0;
  --p100:#F1F5F9; --p50:#F8FAFC;
  --accent:#667eea; --accent2:#764ba2; --sky:#0284C7;
  --ok:#10B981; --warn:#F59E0B; --danger:#EF4444;
  --surface:#FFFFFF;
}

html,body,[class*="css"]{
  font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  -webkit-font-smoothing:antialiased;
}
.block-container{padding-top:1.8rem;padding-bottom:4rem;max-width:1380px;}
#MainMenu,footer{visibility:hidden;}
[data-testid="stSidebarNav"]{display:none!important;}

/* ── metrics ── */
[data-testid="stMetric"]{
  background:var(--surface);border:1px solid var(--p200);border-radius:16px;
  padding:1.1rem 1.2rem;box-shadow:0 1px 4px rgba(0,0,0,.06);
  transition:border-color .2s,box-shadow .2s,transform .2s;overflow:visible;
}
[data-testid="stMetric"]:hover{
  border-color:var(--sky);box-shadow:0 4px 18px rgba(2,132,199,.12);
  transform:translateY(-2px);
}
[data-testid="stMetricLabel"] p{
  color:var(--p500);font-weight:600;font-size:.72rem;text-transform:uppercase;
  letter-spacing:.06em;white-space:normal!important;overflow:visible!important;
}
[data-testid="stMetricValue"],[data-testid="stMetricValue"]>div{
  color:var(--p900);font-weight:700;overflow:visible!important;
  white-space:normal!important;text-overflow:unset!important;
  font-size:clamp(1rem,2vw,1.5rem)!important;line-height:1.3!important;
}

/* ── sidebar ── */
section[data-testid="stSidebar"]>div:first-child{
  background:#FFFFFF!important;border-right:1px solid var(--p200)!important;
}
.sb-brand{padding:1.6rem 1.25rem 1rem;}
.sb-brand-title{display:block;font-size:1.2rem;font-weight:700;color:var(--accent);letter-spacing:-.02em;}
.sb-brand-sub{display:block;font-size:.75rem;color:var(--p500);margin-top:.2rem;}
.sb-divider{height:1px;background:var(--p200);margin:0 1.25rem .8rem;}
.sb-section{font-size:.7rem;font-weight:700;color:var(--p400);padding:0 1.25rem .5rem;
  text-transform:uppercase;letter-spacing:.08em;}
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]>div[data-testid="element-container"]{
  margin:0!important;padding:0!important;}
section[data-testid="stSidebar"] .stButton{
  margin:0!important;padding:0!important;background:none!important;border:none!important;}
section[data-testid="stSidebar"] .stButton>div,
section[data-testid="stSidebar"] .stButton>div>div{
  background:none!important;border:none!important;padding:0!important;margin:0!important;}
section[data-testid="stSidebar"] .stButton button{
  all:unset!important;display:block!important;width:100%!important;
  box-sizing:border-box!important;padding:.52rem .85rem!important;
  border-radius:10px!important;font-size:.875rem!important;font-weight:400!important;
  color:var(--p600)!important;cursor:pointer!important;text-align:left!important;
  transition:background .15s,color .15s,transform .12s!important;
}
section[data-testid="stSidebar"] .stButton button:hover{
  background:rgba(102,126,234,.09)!important;color:var(--p900)!important;
  transform:translateX(3px)!important;
}

/* ── hero ── */
@keyframes heroIn{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:none}}
.hero{
  background:linear-gradient(135deg,#14532d 0%,#15803d 55%,#4ade80 130%);
  border-radius:20px;padding:2.6rem 3rem;color:#fff;margin-bottom:2rem;
  position:relative;overflow:hidden;animation:heroIn .5s ease both;
}
.hero::after{
  content:'';position:absolute;inset:0;pointer-events:none;
  background:radial-gradient(circle at 10% 70%,rgba(255,255,255,.13) 0%,transparent 42%),
             radial-gradient(circle at 90% 15%,rgba(255,255,255,.10) 0%,transparent 40%);
}
.hero-badge{
  display:inline-block;background:rgba(255,255,255,.18);backdrop-filter:blur(10px);
  border:1px solid rgba(255,255,255,.28);padding:.28rem .9rem;border-radius:50px;
  font-size:.72rem;font-weight:600;letter-spacing:.07em;text-transform:uppercase;
  margin-bottom:.9rem;position:relative;z-index:1;
}
.hero h1{
  position:relative;z-index:1;margin:0 0 .6rem;font-size:2.3rem;
  font-weight:800;letter-spacing:-.03em;line-height:1.12;
}
.hero p{
  position:relative;z-index:1;margin:0;max-width:680px;
  font-size:1rem;line-height:1.7;color:rgba(255,255,255,.9);
}

/* ── section title ── */
.section-title{
  font-size:1.3rem;font-weight:700;color:var(--p900);
  margin:2rem 0 1.1rem;letter-spacing:-.02em;
  display:flex;align-items:center;gap:.6rem;
}
.section-title::after{content:'';flex:1;height:1px;background:var(--p200);}

/* ── explore cards ── */
@keyframes cardIn{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:none}}
.explore-card{
  background:var(--surface);border:1px solid var(--p200);border-radius:18px;
  padding:1.4rem;min-height:140px;box-shadow:0 1px 3px rgba(15,23,42,.05);
  transition:border-color .22s,box-shadow .22s,transform .22s;
  position:relative;overflow:hidden;animation:cardIn .4s ease both;
}
.explore-card::before{
  content:'';position:absolute;top:0;left:0;right:0;height:3px;
  background:linear-gradient(90deg,var(--accent),var(--sky));
  opacity:0;transition:opacity .22s;
}
.explore-card:hover{border-color:var(--sky);box-shadow:0 6px 22px rgba(2,132,199,.10);transform:translateY(-3px);}
.explore-card:hover::before{opacity:1;}
.explore-card-icon{font-size:1.6rem;margin-bottom:.5rem;display:block;line-height:1;}
.explore-card h3{margin:0 0 .4rem;font-size:1rem;font-weight:600;color:var(--p900);}
.explore-card p{margin:0;color:var(--p600);font-size:.84rem;line-height:1.55;}

/* ── stat cards (home KPI row) ── */
.stat-card{
  background:var(--surface);border:1px solid var(--p200);border-radius:16px;
  padding:1.3rem 1.4rem;box-shadow:0 1px 4px rgba(0,0,0,.06);
  transition:border-color .2s,box-shadow .2s,transform .2s;
}
.stat-card:hover{border-color:var(--sky);box-shadow:0 4px 18px rgba(2,132,199,.10);transform:translateY(-2px);}
.stat-card-label{font-size:.72rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:var(--p500);margin-bottom:.5rem;}
.stat-card-value{font-size:1.7rem;font-weight:800;color:var(--p900);line-height:1;letter-spacing:-.02em;}
.stat-card-sub{font-size:.78rem;color:var(--p400);margin-top:.3rem;}

/* ── step cards ── */
.step-card{
  background:var(--p50);border:1px solid var(--p200);border-radius:16px;
  padding:1.3rem;text-align:center;height:100%;
}
.step-num{
  display:inline-flex;align-items:center;justify-content:center;
  width:34px;height:34px;border-radius:50%;background:#15803d;
  color:#fff;font-size:.85rem;font-weight:700;margin-bottom:.75rem;
}
.step-card h4{margin:0 0 .45rem;font-size:.95rem;font-weight:600;color:var(--p900);}
.step-card p{margin:0;font-size:.82rem;color:var(--p600);line-height:1.55;}

/* ── page lead ── */
.page-lead{color:var(--p500);font-size:.94rem;line-height:1.65;margin:0 0 1.5rem;}

/* ── rec box ── */
.rec-box{background:#f0fdf4;border:1px solid #bbf7d0;border-radius:14px;padding:1rem 1.2rem;}
.rec-box p{margin:.3rem 0;color:#14532d;font-size:.88rem;line-height:1.65;}

/* ── result box ── */
.result-box{background:var(--p50);border:1px solid var(--p200);border-radius:14px;padding:1.1rem 1.3rem;}
.result-box h4{margin:0 0 .7rem;color:var(--p900);font-size:1rem;font-weight:600;}
.result-box p{margin:.25rem 0;color:var(--p700);font-size:.875rem;}

/* ── risk badge ── */
.risk-badge{display:inline-block;padding:.28rem .85rem;border-radius:50px;font-size:.8rem;font-weight:700;letter-spacing:.03em;}
.risk-Low{background:rgba(16,185,129,.14);color:#059669;}
.risk-Moderate{background:rgba(245,158,11,.14);color:#B45309;}
.risk-High{background:rgba(239,68,68,.13);color:#DC2626;}
.risk-Severe{background:rgba(127,29,29,.16);color:#7F1D1D;}

/* ── saved scenario card ── */
.sc-card{
  background:var(--surface);border:1px solid var(--p200);border-radius:14px;
  padding:1rem 1.2rem;transition:border-color .2s,box-shadow .2s;
}
.sc-card:hover{border-color:var(--sky);box-shadow:0 3px 12px rgba(2,132,199,.08);}
.sc-card-name{font-weight:600;font-size:.95rem;color:var(--p900);margin-bottom:.4rem;}
.sc-card-row{display:flex;justify-content:space-between;font-size:.82rem;color:var(--p600);margin:.15rem 0;}
.sc-card-val{font-weight:600;color:var(--p800);}

/* ── filter container ── */
[data-testid="stVerticalBlockBorderWrapper"]{border-radius:14px!important;}

/* ── buttons ── */
[data-testid="stButton"] button[kind="primary"]{
  background:var(--accent);border-color:var(--accent);
  border-radius:10px;font-weight:600;
}
[data-testid="stButton"] button[kind="secondary"]{border-radius:10px;font-weight:500;}

/* ── expander ── */
[data-testid="stExpander"]{border-radius:12px!important;border-color:var(--p200)!important;}

/* ── dashboard subtitle ── */
.dash-sub{color:var(--p500);font-size:.92rem;line-height:1.65;margin:0 0 1.4rem;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────────────────────
_PALETTE = ["#667eea","#10B981","#F59E0B","#0EA5E9","#EF4444","#8B5CF6","#14B8A6","#F97316"]
_RISK_COLORS = {"Low":"#10B981","Moderate":"#F59E0B","High":"#EF4444","Severe":"#7F1D1D"}
_CFG = {"displayModeBar": False, "responsive": True}

def _layout(height: int = 380, legend_below: bool = True) -> dict:
    base = dict(
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=60, r=20, t=52, b=60),
        font=dict(family="Inter, sans-serif", size=12, color="#334155"),
        xaxis=dict(
            showline=True, linewidth=1, linecolor="#E2E8F0",
            showgrid=False, zeroline=False,
            tickfont=dict(size=11, color="#64748B"),
            title_font=dict(size=12, color="#64748B"),
        ),
        yaxis=dict(
            showline=True, linewidth=1, linecolor="#E2E8F0",
            showgrid=True, gridcolor="#F1F5F9", gridwidth=1,
            zeroline=False,
            tickfont=dict(size=11, color="#64748B"),
            title_font=dict(size=12, color="#64748B"),
        ),
    )
    if legend_below:
        base["legend"] = dict(
            orientation="h", yanchor="top", y=-0.22,
            xanchor="left", x=0,
            font=dict(size=11),
        )
    return base

def _show(fig: go.Figure) -> None:
    st.plotly_chart(fig, use_container_width=True, theme=None, config=_CFG)

# ─────────────────────────────────────────────────────────────────────────────
# CACHED LOADERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def _load_df() -> pd.DataFrame:
    path = RAW_DATA_PATH if RAW_DATA_PATH.exists() else SAMPLE_DATA_PATH
    df = pd.read_csv(path)
    for col in ["temperature_c","rainfall_mm","humidity_pct","yield_t_per_ha","year"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["yield_t_per_ha"]).reset_index(drop=True)

@st.cache_resource(show_spinner="Loading model…")
def _load_model():
    try:
        return load_model(DEFAULT_MODEL_NAME)
    except ModelNotFoundError:
        return None

@st.cache_data(show_spinner=False)
def _load_meta() -> dict | None:
    try:
        return load_metadata(DEFAULT_MODEL_NAME)
    except ModelNotFoundError:
        return None

@st.cache_data(show_spinner=False)
def _dcmy_table(df: pd.DataFrame) -> pd.Series:
    return df.groupby(["district","crop_type"])[TARGET_COLUMN].mean()

@st.cache_data(ttl=600, show_spinner=False)
def _live_snapshot(name: str, admin1: str, country: str, lat: float, lon: float) -> WeatherSnapshot:
    """Live conditions for a point. Cached for 10 minutes so reruns don't hit the API."""
    return fetch_weather(Location(name=name, admin1=admin1, country=country, latitude=lat, longitude=lon))

@st.cache_data(ttl=3600, show_spinner=False)
def _live_search(query: str) -> list[Location]:
    """Geocode a place name. Accepts "Place" or "Place, Region" (the region narrows the matches)."""
    name, _, hint = query.partition(",")
    found = search_locations(name.strip(), count=10)
    hint = hint.strip().lower()
    if hint:
        found = [p for p in found if hint in p.admin1.lower() or hint in p.country.lower()] or found
    return found[:5]

# ── SHAP helpers ──────────────────────────────────────────────────────────────
_FNAME = {
    "temperature_c":        "Temperature",
    "rainfall_mm":          "Rainfall",
    "humidity_pct":         "Humidity",
    "temp_humidity_index": "Heat-Humidity Index",
    "dist_crop_mean_yield":"Location Baseline",
    "year":                "Year",
    "crop_type":           "Crop",
    "season":              "Season",
    "district":            "District",
    "state":               "State",
}
_WEATHER_FEATURES = ["temperature_c", "rainfall_mm", "humidity_pct"]

def _shap_one(inputs: dict) -> tuple[dict, float] | None:
    """Compute SHAP values for a single row instantly."""
    if not _SHAP_OK:
        return None
    try:
        m = _load_model()
        if m is None:
            return None
        c, base_val = explain_prediction(m, pd.DataFrame([inputs]))
        return {str(row["feature"]): float(row["shap_value"]) for _, row in c.iterrows()}, base_val
    except Exception:
        return None

def _nlg_summary(shap_map: dict, predicted: float, base_val: float) -> str:
    weather = {k: v for k, v in shap_map.items() if k in _WEATHER_FEATURES}
    if not weather:
        return f"The predicted yield is {predicted:.2f} t/ha."

    sorted_w = sorted(weather.items(), key=lambda x: abs(x[1]), reverse=True)
    top2 = sorted_w[:2]

    def _describe(feat: str, val: float) -> str:
        name = _FNAME.get(feat, feat)
        direction = "supported by" if val >= 0 else "reduced by"
        return f"{direction} {name} ({val:+.3f} t/ha)"

    primary   = _describe(*top2[0])
    secondary = f", and {_describe(*top2[1])}" if len(top2) > 1 else ""

    overall = "above" if predicted >= base_val else "below"
    return (
        f"The predicted yield of {predicted:.2f} t/ha is {overall} the crop baseline "
        f"({base_val:.2f} t/ha), primarily {primary}{secondary}."
    )

# ─────────────────────────────────────────────────────────────────────────────
# ENRICHMENT
# ─────────────────────────────────────────────────────────────────────────────
def _enrich(inputs: dict, df: pd.DataFrame) -> dict:
    dcmy   = _dcmy_table(df)
    out    = dict(inputs)
    mode_d = str(df["district"].mode()[0])
    out.setdefault("year",     int(df["year"].max()))
    out.setdefault("district", mode_d)
    out.setdefault("state",
        str(df[df["district"] == mode_d]["state"].iloc[0])
        if "state" in df.columns else "Unknown"
    )
    district, crop = str(out["district"]), str(out["crop_type"])
    for key in [(district, crop), (mode_d, crop)]:
        if key in dcmy.index and dcmy[key] > 0:
            out["dist_crop_mean_yield"] = float(dcmy[key])
            break
    else:
        m = df[df["crop_type"] == crop][TARGET_COLUMN].mean()
        out["dist_crop_mean_yield"] = float(m) if not pd.isna(m) else 3.0
    return out

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
_VIEWS = [
    ("home",                   "Home"),
    ("dashboard",              "Dashboard"),
    ("yield prediction",       "Yield Prediction"),
    ("what if",                "What-If Simulator"),
    ("risk score",             "Risk Score"),
    ("saved scenarios",        "Saved Scenarios"),
    ("live weather",           "Live Weather"),
]
for k, default in [
    ("view",         "home"),
    ("saved",        []),
    ("field_crop",   "Rice"),
    ("field_season", "Kharif"),
    ("field_temp",   27.0),
    ("field_rain",   900.0),
    ("field_hum",    70.0),
    ("wi_temp_d",    0.0),
    ("wi_rain_d",    0.0),
    ("wi_hum_d",     0.0),
]:
    st.session_state.setdefault(k, default)

# Values handed over from the Live Weather view. Applied here, before the sidebar
# widgets are created, because widget state can't be changed after instantiation.
_pending = st.session_state.pop("_pending_field", None)
if _pending:
    st.session_state.update(_pending)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD RESOURCES
# ─────────────────────────────────────────────────────────────────────────────
df_global = _load_df()
model     = _load_model()
meta      = _load_meta()

_ALL_CROPS   = sorted(df_global["crop_type"].unique().tolist())
_ALL_SEASONS = sorted(df_global["season"].unique().tolist())

def _midpoint(lo: float, hi: float, step: float) -> float:
    return round(((lo + hi) / 2) / step) * step

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <span class="sb-brand-title">Weather Impact</span>
        <span class="sb-brand-sub">Agriculture Analytics Platform</span>
    </div>
    <div class="sb-divider"></div>
    <div class="sb-section">Navigation</div>
    """, unsafe_allow_html=True)

    for vk, vl in _VIEWS:
        active = st.session_state.view == vk
        label  = f"› {vl}" if active else f"  {vl}"
        if st.button(label, key=f"nav_{vk}", use_container_width=True):
            st.session_state.view = vk
            st.rerun()

    if st.session_state.view in ("yield prediction", "what if", "risk score"):
        st.markdown('<div class="sb-divider" style="margin-top:.9rem;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-section">Field Conditions</div>', unsafe_allow_html=True)

        crop = st.selectbox("Crop", _ALL_CROPS, key="field_crop")
        avail = sorted(df_global[df_global["crop_type"] == crop]["season"].unique().tolist())
        if st.session_state.field_season not in avail:
            st.session_state.field_season = avail[0]
        st.selectbox("Season", avail, key="field_season")

        try:
            p  = get_profile(crop, st.session_state.field_season)
            td = _midpoint(p.temperature_c[0], p.temperature_c[1], 0.5)
            rd = _midpoint(p.rainfall_mm[0],   p.rainfall_mm[1],   10.0)
            hd = _midpoint(p.humidity_pct[0],  p.humidity_pct[1],  1.0)
            th = f"Optimal {p.temperature_c[0]}–{p.temperature_c[1]} °C"
            rh = f"Optimal {p.rainfall_mm[0]}–{p.rainfall_mm[1]} mm"
            hh = f"Optimal {p.humidity_pct[0]}–{p.humidity_pct[1]} %"
        except Exception:
            td, rd, hd = 27.0, 800.0, 65.0
            th = rh = hh = ""

        st.slider("Temperature (°C)",      5.0,   50.0,   td,  0.5,  key="field_temp", help=th)
        st.slider("Seasonal Rainfall (mm)", 0.0,   2000.0, rd,  10.0, key="field_rain", help=rh)
        st.slider("Relative Humidity (%)",  10.0,  100.0,  hd,  1.0,  key="field_hum",  help=hh)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _field() -> dict:
    return {
        "crop_type":    st.session_state.field_crop,
        "season":       st.session_state.field_season,
        "temperature_c": float(st.session_state.field_temp),
        "rainfall_mm":  float(st.session_state.field_rain),
        "humidity_pct": float(st.session_state.field_hum),
    }

def _save_scenario(name: str, inputs: dict, predicted: float,
                   risk_score: float, risk_level: str) -> None:
    if not name.strip():
        st.warning("Enter a name before saving.")
        return
    st.session_state.saved.append({
        "name":          name.strip(),
        "crop_type":     inputs["crop_type"],
        "season":        inputs["season"],
        "temperature_c": round(float(inputs["temperature_c"]), 2),
        "rainfall_mm":   round(float(inputs["rainfall_mm"]), 1),
        "humidity_pct":  round(float(inputs["humidity_pct"]), 1),
        "yield_t_ha":    round(float(predicted), 3),
        "risk_score":    round(float(risk_score), 1),
        "risk_level":    risk_level,
    })
    st.success(f"Saved \"{name.strip()}\"")

# ─────────────────────────────────────────────────────────────────────────────
# VIEW: HOME
# ─────────────────────────────────────────────────────────────────────────────
def render_home() -> None:
    st.markdown("""
    <div class="hero">
        <span class="hero-badge">Machine Learning · Agriculture · Climate</span>
        <h1>Weather Impact Analysis Platform</h1>
        <p>Predict crop yields from real weather conditions, assess environmental risk,
        and stress-test field decisions before they reach the farm.</p>
    </div>
    """, unsafe_allow_html=True)

    n_crops    = len(_ALL_CROPS)
    n_states   = df_global["state"].nunique() if "state" in df_global.columns else "—"
    n_dist     = df_global["district"].nunique()
    yr_min     = int(df_global["year"].min())
    yr_max     = int(df_global["year"].max())
    accuracy   = f"{meta['metrics']['accuracy_pct']:.1f}%" if meta else "—"
    r2         = f"{meta['metrics']['r2']:.3f}"            if meta else "—"
    n_records  = f"{len(df_global):,}"
    n_saved    = len(st.session_state.saved)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model Accuracy",  accuracy,  help="100 × (1 − WAPE)")
    c2.metric("R² Score",        r2,        help="Yield variance explained by the model")
    c3.metric("Records Analysed", n_records)
    c4.metric("Saved Scenarios",  str(n_saved))

    st.markdown("""<div class="section-title">Platform Overview</div>""", unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    for col, label, val, sub in [
        (s1, "Crops",    str(n_crops),          "crop varieties"),
        (s2, "States",   str(n_states),          "Indian states"),
        (s3, "Districts",str(n_dist),            "distinct locations"),
        (s4, "Years",    f"{yr_min}–{yr_max}",   "data coverage"),
    ]:
        with col:
            st.markdown(
                f'<div class="stat-card">'
                f'<div class="stat-card-label">{label}</div>'
                f'<div class="stat-card-value">{val}</div>'
                f'<div class="stat-card-sub">{sub}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("""<div class="section-title">Explore</div>""", unsafe_allow_html=True)

    _CARDS = [
        ("Dashboard",
         "Filter 31k+ historical records by state, district, crop, season, and year. "
         "Four live Plotly charts update as you filter.",
         "dashboard"),
        ("Yield Prediction",
         "Set field conditions in the sidebar. Get a predicted yield, risk score, "
         "SHAP waterfall, and plain-language recommendations.",
         "yield prediction"),
        ("What-If Simulator",
         "Apply temperature, rainfall, and humidity deltas to your baseline and "
         "compare yield and risk side-by-side with a sensitivity chart.",
         "what if"),
        ("Risk Score",
         "0-100 environmental risk gauge, per-factor severity breakdown, "
         "and a budget planner with editable mitigation actions.",
         "risk score"),
        ("Saved Scenarios",
         "Compare, manage, and export prediction and simulation results "
         "you have saved across the session.",
         "saved scenarios"),
        ("Live Weather",
         "Current conditions and a 7-day forecast from Open-Meteo fed "
         "directly into the prediction engine.",
         "live weather"),
    ]

    for row_start in range(0, len(_CARDS), 3):
        row_cards = _CARDS[row_start: row_start + 3]
        cols = st.columns(len(row_cards), gap="large")
        for col, (title, blurb, target) in zip(cols, row_cards):
            with col:
                st.markdown(
                    f'<div class="explore-card">'
                    f'<h3>{title}</h3><p>{blurb}</p></div>',
                    unsafe_allow_html=True,
                )
                if st.button(f"Open {title}", key=f"home_btn_{target}",
                             use_container_width=True):
                    st.session_state.view = target
                    st.rerun()

    st.markdown("""<div class="section-title">How it works</div>""", unsafe_allow_html=True)
    steps = [
        ("1", "Set Conditions",
         "Choose crop and season, then dial in temperature, rainfall, and humidity from the sidebar."),
        ("2", "Get Prediction",
         "The Random Forest model estimates yield (t/ha) and how close it is to the crop's potential."),
        ("3", "Stress-Test",
         "Use the What-If Simulator to shift weather parameters and see how yield and risk respond."),
        ("4", "Act Early",
         "Use the risk level, recommendations, and budget planner to decide on irrigation or drainage."),
    ]
    cols = st.columns(4, gap="medium")
    for col, (num, title, body) in zip(cols, steps):
        with col:
            st.markdown(
                f'<div class="step-card">'
                f'<div class="step-num">{num}</div>'
                f'<h4>{title}</h4><p>{body}</p></div>',
                unsafe_allow_html=True,
            )

# ─────────────────────────────────────────────────────────────────────────────
# VIEW: DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
def render_dashboard() -> None:
    st.markdown('<div class="section-title">Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="dash-sub">Explore historical crop yield and weather records. '
        'State selection cascades into the district list. All charts and KPIs '
        'update live as you change filters.</p>',
        unsafe_allow_html=True,
    )

    df = df_global.copy()
    all_states  = sorted(df["state"].unique().tolist()) if "state" in df.columns else []
    all_dists   = sorted(df["district"].unique().tolist())
    all_crops   = sorted(df["crop_type"].unique().tolist())
    all_seasons = sorted(df["season"].unique().tolist())
    yr_min, yr_max = int(df["year"].min()), int(df["year"].max())

    with st.container(border=True):
        if all_states:
            r1a, r1b = st.columns([2, 3])
            sel_states = r1a.multiselect("State", all_states, default=all_states, key="db_state")
            avail_dist = sorted(
                df[df["state"].isin(sel_states)]["district"].unique().tolist()
                if sel_states else all_dists
            )
        else:
            sel_states = []
            avail_dist = all_dists
            r1b = st.container()

        prev = st.session_state.get("db_district", avail_dist)
        valid = [d for d in prev if d in avail_dist]
        sel_dist = r1b.multiselect(
            "District", avail_dist,
            default=valid if valid else avail_dist,
            key="db_district",
        )

        r2a, r2b, r2c = st.columns([2, 1.6, 2.4])
        sel_crops   = r2a.multiselect("Crop",      all_crops,   default=all_crops,   key="db_crop")
        sel_seasons = r2b.multiselect("Season",    all_seasons, default=all_seasons, key="db_season")
        sel_years   = r2c.slider("Year range", yr_min, yr_max, (yr_min, yr_max),       key="db_years")

    flt = df[
        df["district"].isin(sel_dist)
        & df["crop_type"].isin(sel_crops)
        & df["season"].isin(sel_seasons)
        & df["year"].between(sel_years[0], sel_years[1])
    ]
    if sel_states:
        flt = flt[flt["state"].isin(sel_states)]

    if flt.empty:
        st.info("No records match these filters — try widening the selection.")
        return

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Records",          f"{len(flt):,}")
    k2.metric("Avg Yield (t/ha)",  f"{flt[TARGET_COLUMN].mean():.2f}")
    k3.metric("Avg Rainfall (mm)", f"{flt['rainfall_mm'].mean():.0f}")
    k4.metric("Avg Temp (°C)",     f"{flt['temperature_c'].mean():.1f}")
    k5.metric("Avg Humidity (%)",  f"{flt['humidity_pct'].mean():.0f}")

    trend = (
        flt.groupby(["year","crop_type"], as_index=False)[TARGET_COLUMN]
           .mean()
           .rename(columns={TARGET_COLUMN: "avg_yield"})
    )
    fig_trend = go.Figure()
    for i, (crop, grp) in enumerate(trend.groupby("crop_type")):
        fig_trend.add_trace(go.Scatter(
            x=grp["year"], y=grp["avg_yield"], mode="lines+markers", name=crop,
            line=dict(color=_PALETTE[i % len(_PALETTE)], width=2.5),
            marker=dict(size=6),
            hovertemplate=f"{crop}  Year %{{x}}  %{{y:.2f}} t/ha<extra></extra>",
        ))
    fig_trend.update_layout(title="Yield trend by crop", xaxis_title="Year", yaxis_title="Avg yield (t/ha)", **_layout())

    samp = flt.sample(n=min(2000, len(flt)), random_state=0) if len(flt) > 2000 else flt
    fig_scatter = go.Figure()
    for i, (crop, grp) in enumerate(samp.groupby("crop_type")):
        fig_scatter.add_trace(go.Scatter(
            x=grp["rainfall_mm"], y=grp[TARGET_COLUMN], mode="markers", name=crop,
            marker=dict(color=_PALETTE[i % len(_PALETTE)], size=6, opacity=0.55),
            hovertemplate="Rain %{x:.0f} mm  Yield %{y:.2f} t/ha<extra>" + crop + "</extra>",
        ))
    fig_scatter.update_layout(title="Rainfall and yield", xaxis_title="Seasonal rainfall (mm)", yaxis_title="Yield (t/ha)", **_layout())

    fig_box = go.Figure()
    for i, (crop, grp) in enumerate(flt.groupby("crop_type")):
        fig_box.add_trace(go.Box(
            x=grp["season"], y=grp[TARGET_COLUMN], name=crop,
            marker_color=_PALETTE[i % len(_PALETTE)], boxmean=True,
        ))
    fig_box.update_layout(boxmode="group", title="Yield spread by season", yaxis_title="Yield (t/ha)", **_layout())

    pivot = flt.pivot_table(index="crop_type", columns="season", values=TARGET_COLUMN, aggfunc="mean")
    fig_heat = go.Figure(go.Heatmap(
        z=pivot.values, x=list(pivot.columns), y=list(pivot.index),
        colorscale="Greens",
        text=[[f"{v:.2f}" if pd.notna(v) else "—" for v in row] for row in pivot.values],
        texttemplate="%{text}", colorbar=dict(title="t/ha"),
        hovertemplate="%{y} · %{x}  %{z:.2f} t/ha<extra></extra>",
    ))
    fig_heat.update_yaxes(autorange="reversed")
    fig_heat.update_layout(title="Avg yield: crop × season", xaxis_title="Season", yaxis_title="Crop", **_layout(legend_below=False))

    cl, cr = st.columns(2)
    with cl, st.container(border=True): _show(fig_trend)
    with cr, st.container(border=True): _show(fig_scatter)
    cl, cr = st.columns(2)
    with cl, st.container(border=True): _show(fig_box)
    with cr, st.container(border=True): _show(fig_heat)

    with st.expander(f"View filtered records ({len(flt):,})"):
        st.dataframe(flt.reset_index(drop=True), use_container_width=True, hide_index=True)
        st.download_button("Download filtered CSV", data=flt.to_csv(index=False).encode("utf-8"), file_name="filtered_crop_records.csv", mime="text/csv")

# ─────────────────────────────────────────────────────────────────────────────
# VIEW: YIELD PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
def render_yield_prediction() -> None:
    st.markdown('<div class="section-title">Yield Prediction</div>', unsafe_allow_html=True)
    st.markdown('<p class="page-lead">Set field conditions in the sidebar. The model predicts yield and risk instantly.</p>', unsafe_allow_html=True)
    if model is None:
        st.error("No trained model found.")
        return

    inputs    = _enrich(_field(), df_global)
    predicted = predict_one(model, inputs)
    risk      = compute_risk(inputs)
    recs      = recommendations(risk)
    rc        = _RISK_COLORS.get(risk.level, "#667eea")

    try:
        profile = get_profile(inputs["crop_type"], inputs["season"])
        share   = min(999.0, predicted / profile.base_yield_t_per_ha * 100) if profile.base_yield_t_per_ha > 0 else 0.0
    except Exception:
        share = 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Predicted Yield",    f"{predicted:.2f} t/ha")
    k2.metric("Share of Potential", f"{share:.0f}%", help="Percentage of genetic/benchmark base yield achieved under current weather conditions")
    k3.metric("Risk Score (0–100)", f"{risk.score:.0f}")
    k4.metric("Risk Level",         risk.level)

    gauge_col, wf_col = st.columns([1, 1.4])

    with gauge_col, st.container(border=True):
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk.score,
            number={"suffix": " / 100", "font": {"size": 28, "color": rc}},
            title={"text": f"<b>{risk.level}</b> Risk", "font": {"size": 15}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar":  {"color": rc, "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
                "steps": [
                    {"range": [0,  25],  "color": "rgba(16,185,129,.12)"},
                    {"range": [25, 50],  "color": "rgba(245,158,11,.12)"},
                    {"range": [50, 75],  "color": "rgba(239,68,68,.10)"},
                    {"range": [75, 100], "color": "rgba(127,29,29,.13)"},
                ],
                "threshold": {"line": {"color": rc, "width": 3}, "thickness": .8, "value": risk.score},
            },
        ))
        fig_g.update_layout(height=280, paper_bgcolor="#FFFFFF", margin=dict(l=30, r=30, t=48, b=24), font=dict(family="Inter, sans-serif", color="#334155"))
        _show(fig_g)

    with wf_col, st.container(border=True):
        shap_one = _shap_one(inputs)
        if shap_one is not None:
            shap_map, base_val = shap_one
            weather_shap = {k: shap_map[k] for k in _WEATHER_FEATURES if k in shap_map}
            sorted_w = sorted(weather_shap.items(), key=lambda x: abs(x[1]), reverse=True)
            labels   = ["Baseline"] + [_FNAME[k] for k, _ in sorted_w] + ["Predicted"]
            measures = ["absolute"] + ["relative"] * len(sorted_w) + ["total"]
            values   = [float(base_val)] + [v for _, v in sorted_w] + [float(predicted)]
        else:
            base_val = float(df_global[df_global["crop_type"] == inputs["crop_type"]][TARGET_COLUMN].mean())
            if pd.isna(base_val):
                base_val = float(predicted)
            raw = [
                ("Temperature", -(risk.severity["temperature_c"]) * base_val * 0.40),
                ("Rainfall",    -(risk.severity["rainfall_mm"])   * base_val * 0.35),
                ("Humidity",    -(risk.severity["humidity_pct"])  * base_val * 0.25),
            ]
            labels   = ["Baseline"] + [n for n, _ in raw] + ["Predicted"]
            measures = ["absolute"] + ["relative"] * len(raw) + ["total"]
            values   = [base_val] + [v for _, v in raw] + [float(predicted)]
            shap_map = None

        fig_wf = go.Figure(go.Waterfall(
            orientation="v", measure=measures, x=labels, y=values,
            connector={"line": {"color": "#CBD5E1", "width": 1.5}},
            increasing={"marker": {"color": "#10B981", "line": {"width": 0}}},
            decreasing={"marker": {"color": "#EF4444", "line": {"width": 0}}},
            totals={"marker":    {"color": "#667eea", "line": {"width": 0}}},
            text=[f"{v:+.3f}" if m == "relative" else f"{v:.3f}" for v, m in zip(values, measures)],
            textposition="outside", textfont=dict(size=12, color="#334155"),
            hovertemplate="%{x}  %{y:+.3f} t/ha<extra></extra>",
        ))
        fig_wf.update_layout(
            title=dict(text="What drives this prediction (SHAP)", font=dict(size=15, color="#0F172A")),
            xaxis=dict(showgrid=False, showline=False, zeroline=False, tickfont=dict(size=12, color="#334155"), automargin=True),
            yaxis=dict(title=dict(text="Yield (t/ha)", font=dict(size=12, color="#64748B")), showgrid=True, gridcolor="#F1F5F9", zeroline=True, zerolinecolor="#CBD5E1", zerolinewidth=1, showline=True, linecolor="#E2E8F0", tickfont=dict(size=11, color="#64748B"), automargin=True),
            showlegend=False, height=420, margin=dict(l=20, r=40, t=52, b=20), paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", font=dict(family="Inter, sans-serif", size=12, color="#334155"),
        )
        _show(fig_wf)

    if shap_one is not None:
        summary = _nlg_summary(shap_map, predicted, base_val)
        st.markdown(f'<div class="rec-box" style="background:#EFF6FF;border-color:#BFDBFE;"><p style="color:#1E40AF;font-weight:600;margin-bottom:.3rem;">Plain English Summary</p><p style="color:#1E3A8A;">{summary}</p></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("**Recommendations**")
    st.markdown('<div class="rec-box">' + "".join(f"<p>{r}</p>" for r in recs) + "</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title" style="font-size:1.05rem;margin-top:1.8rem;">Save this prediction</div>', unsafe_allow_html=True)
    nc, bc = st.columns([4, 1])
    name = nc.text_input("Name", placeholder="e.g. Rice Kharif — current week", label_visibility="collapsed", key="pred_name")
    with bc:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Save", key="save_pred", type="primary", use_container_width=True):
            _save_scenario(name, inputs, predicted, risk.score, risk.level)

# ─────────────────────────────────────────────────────────────────────────────
# VIEW: WHAT-IF SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────
def render_what_if() -> None:
    st.markdown('<div class="section-title">What-If Simulator</div>', unsafe_allow_html=True)
    st.markdown('<p class="page-lead">Start from your current field conditions and apply weather shifts to compare baseline vs scenario yield and risk.</p>', unsafe_allow_html=True)
    if model is None:
        st.error("No trained model found.")
        return

    t_lo, t_hi = SCENARIO_LIMITS["temperature_delta_c"]
    r_lo, r_hi = SCENARIO_LIMITS["rainfall_change_pct"]
    h_lo, h_hi = SCENARIO_LIMITS["humidity_delta_pct"]

    def _reset():
        st.session_state.wi_temp_d = 0.0
        st.session_state.wi_rain_d = 0.0
        st.session_state.wi_hum_d  = 0.0

    with st.container(border=True):
        s1, s2, s3, s4 = st.columns([1.3, 1.3, 1.3, 0.7])
        temp_d = s1.slider("Temperature change (°C)", float(t_lo), float(t_hi), float(st.session_state.wi_temp_d), 0.5, key="wi_temp_d")
        rain_d = s2.slider("Rainfall change (%)", float(r_lo), float(r_hi), float(st.session_state.wi_rain_d), 5.0, key="wi_rain_d")
        hum_d  = s3.slider("Humidity change (pp)", float(h_lo), float(h_hi), float(st.session_state.wi_hum_d), 1.0, key="wi_hum_d")
        s4.markdown("<br>", unsafe_allow_html=True)
        s4.button("Reset", key="wi_reset", use_container_width=True, on_click=_reset)

    baseline = _enrich(_field(), df_global)
    change   = ScenarioChange(temperature_delta_c=temp_d, rainfall_change_pct=rain_d, humidity_delta_pct=hum_d)
    try:
        result = run_what_if(model, baseline, change)
    except Exception as exc:
        st.error(f"Simulation error: {exc}")
        return

    all_zero = temp_d == 0.0 and rain_d == 0.0 and hum_d == 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Baseline Yield",  f"{result.baseline_yield:.2f} t/ha")
    k2.metric("Scenario Yield",  f"{result.scenario_yield:.2f} t/ha", delta=f"{result.yield_change_pct:+.1f}%" if not all_zero else None)
    k3.metric("Baseline Risk",   f"{result.baseline_risk:.0f}")
    k4.metric("Scenario Risk",   f"{result.scenario_risk:.0f}", delta=f"{result.risk_change:+.1f} pts" if not all_zero else None, delta_color="inverse")

    left, right = st.columns(2)
    with left, st.container(border=True):
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(name="Baseline", x=["Yield (t/ha)", "Risk (0–100)"], y=[result.baseline_yield, result.baseline_risk], marker_color="#667eea", marker_line_width=0, width=0.34))
        fig_bar.add_trace(go.Bar(name="Scenario", x=["Yield (t/ha)", "Risk (0–100)"], y=[result.scenario_yield, result.scenario_risk], marker_color="#10B981", marker_line_width=0, width=0.34))
        fig_bar.update_layout(barmode="group", title="Baseline vs Scenario", xaxis_title="", yaxis_title="Value", **_layout())
        _show(fig_bar)

    with right, st.container(border=True):
        t_range = np.linspace(float(t_lo), float(t_hi), 21)
        s_yields = []
        for dt in t_range:
            try:
                r = run_what_if(model, baseline, ScenarioChange(temperature_delta_c=float(dt), rainfall_change_pct=rain_d, humidity_delta_pct=hum_d))
                s_yields.append(r.scenario_yield)
            except Exception:
                s_yields.append(np.nan)

        fig_sens = go.Figure()
        fig_sens.add_trace(go.Scatter(x=t_range, y=s_yields, mode="lines+markers", line=dict(color="#667eea", width=2.5), marker=dict(size=6), hovertemplate="Δ Temp %{x:+.1f} °C  Yield %{y:.2f} t/ha<extra></extra>"))
        fig_sens.add_vline(x=temp_d, line_dash="dot", line_color="#94A3B8", annotation_text=" Now", annotation_font_color="#64748B")
        fig_sens.update_layout(title="Sensitivity: yield across temperature range", xaxis_title="Temperature change (°C)", yaxis_title="Yield (t/ha)", showlegend=False, **_layout())
        _show(fig_sens)

    sc      = result.scenario_inputs
    risk_sc = compute_risk(sc)
    st.markdown(
        f'<div class="result-box"><h4>Scenario Result</h4>'
        f'<p><strong>Temperature:</strong> {sc["temperature_c"]:.1f} °C ({temp_d:+.1f} °C vs baseline)</p>'
        f'<p><strong>Rainfall:</strong> {sc["rainfall_mm"]:.0f} mm ({rain_d:+.0f}% vs baseline)</p>'
        f'<p><strong>Humidity:</strong> {sc["humidity_pct"]:.0f}% ({hum_d:+.0f} pp vs baseline)</p>'
        f'<p><strong>Yield:</strong> {result.scenario_yield:.2f} t/ha ({result.yield_change_pct:+.1f}%)</p>'
        f'<p><strong>Risk:</strong> {risk_sc.score:.0f}/100 — <span class="risk-badge risk-{risk_sc.level}">{risk_sc.level}</span></p>'
        f'</div>', unsafe_allow_html=True,
    )

    if not all_zero:
        shap_base = _shap_one(baseline)
        shap_sc   = _shap_one(result.scenario_inputs)
        if shap_base is not None and shap_sc is not None:
            base_map, _ = shap_base
            sc_map,   _ = shap_sc
            deltas = {f: sc_map.get(f, 0.0) - base_map.get(f, 0.0) for f in _WEATHER_FEATURES}
            sorted_d = sorted(deltas.items(), key=lambda x: abs(x[1]), reverse=True)

            st.markdown('<div class="section-title" style="font-size:1.05rem;margin-top:1.6rem;">Scenario Impact Breakdown (Delta SHAP)</div>', unsafe_allow_html=True)
            with st.container(border=True):
                d_labels   = ["Baseline Yield"] + [_FNAME[k] for k, _ in sorted_d] + ["Scenario Yield"]
                d_measures = ["absolute"] + ["relative"] * len(sorted_d) + ["total"]
                d_values   = [result.baseline_yield] + [v for _, v in sorted_d] + [result.scenario_yield]

                fig_delta = go.Figure(go.Waterfall(
                    orientation="v", measure=d_measures, x=d_labels, y=d_values,
                    connector={"line": {"color": "#CBD5E1", "width": 1.5}},
                    increasing={"marker": {"color": "#10B981", "line": {"width": 0}}},
                    decreasing={"marker": {"color": "#EF4444", "line": {"width": 0}}},
                    totals={"marker":    {"color": "#667eea", "line": {"width": 0}}},
                    text=[f"{v:+.3f}" if m == "relative" else f"{v:.3f}" for v, m in zip(d_values, d_measures)],
                    textposition="outside", textfont=dict(size=12, color="#334155"),
                    hovertemplate="%{x}  %{y:+.3f} t/ha<extra></extra>",
                ))
                fig_delta.update_layout(title=dict(text="Which factor caused the yield change?", font=dict(size=15, color="#0F172A")), xaxis=dict(showgrid=False, showline=False, zeroline=False, tickfont=dict(size=12, color="#334155"), automargin=True), yaxis=dict(title=dict(text="Yield (t/ha)", font=dict(size=12, color="#64748B")), showgrid=True, gridcolor="#F1F5F9", zeroline=True, zerolinecolor="#CBD5E1", zerolinewidth=1, showline=True, linecolor="#E2E8F0", tickfont=dict(size=11, color="#64748B"), automargin=True), showlegend=False, height=380, margin=dict(l=20, r=40, t=52, b=20), paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", font=dict(family="Inter, sans-serif", size=12, color="#334155"))
                _show(fig_delta)

    st.markdown('<div class="section-title" style="font-size:1.05rem;margin-top:1.8rem;">Save this scenario</div>', unsafe_allow_html=True)
    nc, bc = st.columns([4, 1])
    wi_name = nc.text_input("Name", placeholder="e.g. Heatwave +3°C, −20% rain", label_visibility="collapsed", key="wi_name")
    with bc:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Save", key="save_wi", type="primary", use_container_width=True):
            _save_scenario(wi_name, sc, result.scenario_yield, risk_sc.score, risk_sc.level)

# ─────────────────────────────────────────────────────────────────────────────
# VIEW: RISK SCORE
# ─────────────────────────────────────────────────────────────────────────────
def render_risk_score() -> None:
    st.markdown('<div class="section-title">Risk Score Analytics</div>', unsafe_allow_html=True)
    st.markdown('<p class="page-lead">Adjust field conditions in the sidebar. The 0–100 Environmental Risk Score updates instantly.</p>', unsafe_allow_html=True)
    inputs = _enrich(_field(), df_global)
    risk   = compute_risk(inputs)
    recs   = recommendations(risk)
    rc     = _RISK_COLORS.get(risk.level, "#667eea")

    gauge_col, detail_col = st.columns([1, 1.4])

    with gauge_col, st.container(border=True):
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk.score,
            delta={"reference": 50, "valueformat": ".1f"},
            number={"suffix": " / 100", "font": {"size": 30, "color": rc}},
            title={"text": f"<b>{risk.level}</b> Risk", "font": {"size": 15}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar":  {"color": rc, "thickness": 0.3},
                "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
                "steps": [
                    {"range": [0,  25],  "color": "rgba(16,185,129,.12)"},
                    {"range": [25, 50],  "color": "rgba(245,158,11,.12)"},
                    {"range": [50, 75],  "color": "rgba(239,68,68,.10)"},
                    {"range": [75, 100], "color": "rgba(127,29,29,.13)"},
                ],
                "threshold": {"line": {"color": rc, "width": 3}, "thickness": .8, "value": risk.score},
            },
        ))
        fig_g.update_layout(height=290, paper_bgcolor="#FFFFFF", margin=dict(l=30, r=30, t=48, b=24), font=dict(family="Inter, sans-serif", color="#334155"))
        _show(fig_g)
        st.markdown(f'<div style="text-align:center;margin-top:.4rem;"><span class="risk-badge risk-{risk.level}">{risk.level} Risk — {risk.score:.0f}/100</span></div>', unsafe_allow_html=True)

    with detail_col, st.container(border=True):
        st.markdown("**Factor Detail**")
        _FL = {"temperature_c": "Temperature", "rainfall_mm": "Rainfall", "humidity_pct": "Humidity"}
        _FU = {"temperature_c": "°C", "rainfall_mm": "mm", "humidity_pct": "%"}
        rows = []
        for f in ["temperature_c", "rainfall_mm", "humidity_pct"]:
            try:
                p = get_profile(inputs["crop_type"], inputs["season"])
                lo, hi = getattr(p, f)
            except Exception:
                lo, hi = 0.0, 100.0
            u = _FU[f]
            rows.append({
                "Factor":        _FL[f],
                "Current":       f"{risk.values[f]:.1f} {u}",
                "Optimal range": f"{lo:g}–{hi:g} {u}",
                "Status":        risk.direction[f].upper(),
                "Severity":      f"{risk.severity[f]*100:.0f}%",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        fig_sev = go.Figure(go.Bar(
            x=[risk.severity[f] * 100 for f in ["temperature_c","rainfall_mm","humidity_pct"]],
            y=[_FL[f] for f in ["temperature_c","rainfall_mm","humidity_pct"]],
            orientation="h",
            marker_color=["#10B981" if risk.severity[f] == 0 else "#F59E0B" if risk.severity[f] < 0.5 else "#EF4444" for f in ["temperature_c","rainfall_mm","humidity_pct"]],
            text=[f"{risk.severity[f]*100:.0f}%" for f in ["temperature_c","rainfall_mm","humidity_pct"]],
            textposition="outside",
        ))
        fig_sev.update_layout(
            title="Severity by factor",
            xaxis=dict(range=[0, 130], title="Stress (%)", showgrid=False, showline=True, linecolor="#E2E8F0", tickfont=dict(size=11, color="#64748B"), title_font=dict(size=12, color="#64748B")),
            yaxis=dict(autorange="reversed", showline=True, linecolor="#E2E8F0", tickfont=dict(size=12, color="#334155"), automargin=True),
            showlegend=False, height=220, margin=dict(l=10, r=60, t=40, b=40), paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", font=dict(family="Inter, sans-serif", size=12, color="#334155"),
        )
        _show(fig_sev)

    st.markdown("**Recommendations**")
    st.markdown('<div class="rec-box">' + "".join(f"<p>{r}</p>" for r in recs) + "</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title" style="font-size:1.05rem;margin-top:2rem;">Mitigation Budget Planner</div>', unsafe_allow_html=True)
    budget = st.number_input(f"Total budget ({CURRENCY_SYMBOL})", min_value=0.0, max_value=10_000_000.0, value=50_000.0, step=1000.0, key="risk_budget", format="%.0f")
    _ACTIONS = {
        "Low":      [("Routine monitoring", 500, 2.0), ("Soil testing", 1200, 3.0)],
        "Moderate": [("Supplemental irrigation", 3000, 8.0), ("Mulching", 1500, 5.0), ("Fungicide spray", 2000, 6.0)],
        "High":     [("Drip irrigation setup", 8000, 15.0), ("Shade nets", 4000, 10.0), ("Fungicide × 2", 3500, 9.0), ("Crop insurance", 5000, 12.0)],
        "Severe":   [("Emergency irrigation", 10000, 18.0), ("Shade nets + covers", 6000, 14.0), ("Fungicide × 3", 5000, 11.0), ("Crop insurance", 7000, 15.0), ("Advisory service", 2000, 5.0)],
    }
    cost_col = f"Cost/unit ({CURRENCY_SYMBOL})"
    init_df  = pd.DataFrame([{"Action": a, cost_col: c, "Risk reduction/unit (pts)": r, "Units": 0} for a, c, r in _ACTIONS.get(risk.level, _ACTIONS["Moderate"])])
    edited = st.data_editor(
        init_df, use_container_width=True, hide_index=True,
        column_config={
            "Action": st.column_config.TextColumn(disabled=True),
            cost_col: st.column_config.NumberColumn(disabled=True, format=f"{CURRENCY_SYMBOL}%.0f"),
            "Risk reduction/unit (pts)": st.column_config.NumberColumn(disabled=True, format="%.1f pts"),
            "Units": st.column_config.NumberColumn(min_value=0, max_value=100, step=1),
        },
        key="risk_editor",
    )
    total_spend  = float((edited[cost_col] * edited["Units"]).sum())
    total_reduce = float((edited["Risk reduction/unit (pts)"] * edited["Units"]).sum())
    remaining    = budget - total_spend
    new_risk     = max(0.0, risk.score - total_reduce)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"Total Spend ({CURRENCY_SYMBOL})",       f"{total_spend:,.0f}")
    m2.metric(f"Remaining ({CURRENCY_SYMBOL})",         f"{remaining:,.0f}", delta=f"{remaining-budget:+,.0f}" if total_spend > 0 else None, delta_color="inverse")
    m3.metric("Risk Reduction",   f"−{total_reduce:.1f} pts")
    m4.metric("Projected Risk",   f"{new_risk:.0f}/100", delta=f"{new_risk-risk.score:+.1f}" if total_spend > 0 else None, delta_color="inverse")
    if total_spend > budget:
        st.warning(f"Over budget by {CURRENCY_SYMBOL}{total_spend - budget:,.0f}.")

# ─────────────────────────────────────────────────────────────────────────────
# VIEW: SAVED SCENARIOS
# ─────────────────────────────────────────────────────────────────────────────
def render_saved_scenarios() -> None:
    st.markdown('<div class="section-title">Saved Scenarios</div>', unsafe_allow_html=True)
    saved = st.session_state.saved
    if not saved:
        st.info("No scenarios saved yet. Use **Yield Prediction** or **What-If Simulator** to save your first scenario.")
        c1, c2, _ = st.columns([1.5, 1.5, 4])
        if c1.button("Go to Yield Prediction", use_container_width=True):
            st.session_state.view = "yield prediction"; st.rerun()
        if c2.button("Go to What-If Simulator", use_container_width=True):
            st.session_state.view = "what if"; st.rerun()
        return

    df_saved = pd.DataFrame(saved)

    k1, k2, k3 = st.columns(3)
    k1.metric("Saved Scenarios", str(len(saved)))
    k2.metric("Best Yield",      f"{df_saved['yield_t_ha'].max():.3f} t/ha")
    k3.metric("Lowest Risk",     f"{df_saved['risk_score'].min():.1f} / 100")

    st.markdown('<div class="section-title">All Scenarios</div>', unsafe_allow_html=True)
    cols_per_row = 3
    for row_start in range(0, len(saved), cols_per_row):
        row = saved[row_start: row_start + cols_per_row]
        cols = st.columns(cols_per_row, gap="medium")
        for col, sc in zip(cols, row):
            with col:
                badge = f'<span class="risk-badge risk-{sc["risk_level"]}">{sc["risk_level"]}</span>'
                st.markdown(
                    f'<div class="sc-card">'
                    f'<div class="sc-card-name">{sc["name"]}</div>'
                    f'<div class="sc-card-row"><span>{sc["crop_type"]} · {sc["season"]}</span>{badge}</div>'
                    f'<div class="sc-card-row"><span>Yield</span><span class="sc-card-val">{sc["yield_t_ha"]:.3f} t/ha</span></div>'
                    f'<div class="sc-card-row"><span>Risk score</span><span class="sc-card-val">{sc["risk_score"]:.1f}/100</span></div>'
                    f'<div class="sc-card-row"><span>Temp / Rain / Hum</span>'
                    f'<span class="sc-card-val">{sc["temperature_c"]}°C · {sc["rainfall_mm"]} mm · {sc["humidity_pct"]}%</span></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.markdown('<div class="section-title">Compare</div>', unsafe_allow_html=True)
    names      = df_saved["name"].tolist()
    sel_cmp    = st.multiselect("Select up to 4 scenarios to compare", names, max_selections=4, key="ss_compare")
    if sel_cmp:
        cmp = df_saved[df_saved["name"].isin(sel_cmp)]
        fig_cmp = make_subplots(rows=1, cols=2, subplot_titles=("Predicted Yield (t/ha)", "Risk Score (0–100)"))
        fig_cmp.add_trace(go.Bar(x=cmp["name"], y=cmp["yield_t_ha"], marker_color="#667eea", text=[f"{v:.3f}" for v in cmp["yield_t_ha"]], textposition="outside", showlegend=False), row=1, col=1)
        fig_cmp.add_trace(go.Bar(x=cmp["name"], y=cmp["risk_score"], marker_color="#F59E0B", text=[f"{v:.1f}" for v in cmp["risk_score"]], textposition="outside", showlegend=False), row=1, col=2)
        fig_cmp.update_yaxes(range=[0, 115], row=1, col=2)
        fig_cmp.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", height=340, margin=dict(l=56, r=20, t=52, b=60), font=dict(family="Inter, sans-serif", size=12, color="#334155"))
        fig_cmp.update_xaxes(showline=True, linecolor="#E2E8F0", showgrid=False, tickfont=dict(size=11, color="#64748B"), automargin=True)
        fig_cmp.update_yaxes(showline=True, linecolor="#E2E8F0", showgrid=True, gridcolor="#F1F5F9", tickfont=dict(size=11, color="#64748B"))
        _show(fig_cmp)

    st.markdown('<div class="section-title">Data Table</div>', unsafe_allow_html=True)
    st.dataframe(df_saved, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Manage</div>', unsafe_allow_html=True)
    sel_del = st.multiselect("Select scenarios to delete", names, key="ss_delete")
    dc, dl, _ = st.columns([1.4, 1.6, 4])
    with dc:
        if st.button("Delete selected", key="btn_del", type="secondary", disabled=not sel_del, use_container_width=True):
            st.session_state.saved = [s for s in saved if s["name"] not in sel_del]
            st.rerun()
    with dl:
        st.download_button("Download all as CSV", data=df_saved.to_csv(index=False).encode("utf-8"), file_name="saved_scenarios.csv", mime="text/csv", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# VIEW: LIVE WEATHER
# ─────────────────────────────────────────────────────────────────────────────
def _fmt_day(iso: str) -> str:
    return pd.to_datetime(iso).strftime("%a %d %b")

def _live_forecast_chart(snap: WeatherSnapshot) -> go.Figure:
    days  = [_fmt_day(d.date) for d in snap.forecast]
    probs = [("n/a" if d.precipitation_probability_pct is None else f"{d.precipitation_probability_pct:.0f}%") for d in snap.forecast]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(
        x=days, y=[d.precipitation_mm for d in snap.forecast], name="Rain (mm)",
        marker_color="#0EA5E9", opacity=0.75, customdata=probs,
        hovertemplate="%{x}<br>Rain %{y:.1f} mm<br>Chance %{customdata}<extra></extra>",
        secondary_y=False,
    )
    fig.add_scatter(
        x=days, y=[d.temp_max_c for d in snap.forecast], name="Max (°C)", mode="lines+markers",
        line=dict(color="#EF4444", width=2.5), marker=dict(size=7),
        hovertemplate="%{x}<br>Max %{y:.1f} °C<extra></extra>", secondary_y=True,
    )
    fig.add_scatter(
        x=days, y=[d.temp_min_c for d in snap.forecast], name="Min (°C)", mode="lines+markers",
        line=dict(color="#667eea", width=2.5), marker=dict(size=7),
        hovertemplate="%{x}<br>Min %{y:.1f} °C<extra></extra>", secondary_y=True,
    )
    fig.update_layout(
        title=dict(text="7-day forecast", font=dict(size=15, color="#0F172A")),
        height=380, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        margin=dict(l=60, r=60, t=52, b=60),
        font=dict(family="Inter, sans-serif", size=12, color="#334155"),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0, font=dict(size=11)),
        bargap=0.45,
    )
    fig.update_xaxes(showline=True, linecolor="#E2E8F0", showgrid=False, tickfont=dict(size=11, color="#64748B"))
    fig.update_yaxes(title_text="Rain (mm)", showgrid=True, gridcolor="#F1F5F9", rangemode="tozero",
                     showline=True, linecolor="#E2E8F0", tickfont=dict(size=11, color="#64748B"), secondary_y=False)
    fig.update_yaxes(title_text="Temperature (°C)", showgrid=False, showline=True, linecolor="#E2E8F0",
                     tickfont=dict(size=11, color="#64748B"), secondary_y=True)
    return fig

def _live_baseline_district(crop: str, location: Location, from_dataset: bool,
                            sel_district: str, sel_state: str) -> tuple[str, str, str]:
    """Pick the dataset district whose history anchors the prediction, with a note explaining the choice."""
    grows = df_global[df_global["crop_type"] == crop][["district", "state"]].drop_duplicates()
    candidates = [(str(d), str(s)) for d, s in grows.itertuples(index=False)]
    if from_dataset and (sel_district, sel_state) in candidates:
        return sel_district, sel_state, f"Baseline: {sel_district} historical records for {crop}."
    near = nearest_district(location.latitude, location.longitude, candidates)
    if near is None:
        d, s = candidates[0]
        return d, s, f"Baseline: {d}, {s} (no coordinates available to find a closer district)."
    d, s, km = near
    why = f"{sel_district} has no {crop} records" if from_dataset else f"{location.name} is not in the dataset"
    return d, s, f"Baseline: {d}, {s}, the nearest district with {crop} records ({km:.0f} km away). {why}."

def _live_prediction(snap: WeatherSnapshot, location: Location, from_dataset: bool,
                     sel_district: str, sel_state: str) -> None:
    st.markdown('<div class="section-title" style="font-size:1.05rem;margin-top:1.8rem;">Predict yield from live conditions</div>', unsafe_allow_html=True)
    if model is None:
        st.info("No trained model found, so live conditions can't be turned into a prediction.")
        return

    c1, c2 = st.columns(2)
    crop = c1.selectbox("Crop", _ALL_CROPS, key="lw_crop")
    avail = sorted(df_global[df_global["crop_type"] == crop]["season"].unique().tolist())
    if st.session_state.get("lw_season") not in avail:
        st.session_state["lw_season"] = avail[0]
    season = c2.selectbox("Season", avail, key="lw_season")

    b_district, b_state, b_note = _live_baseline_district(crop, location, from_dataset, sel_district, sel_state)

    hist = df_global[(df_global["district"] == b_district) & (df_global["crop_type"] == crop) & (df_global["season"] == season)]
    if hist.empty:
        hist = df_global[(df_global["crop_type"] == crop) & (df_global["season"] == season)]
    rain_default = float(np.clip(round(hist["rainfall_mm"].mean() / 10) * 10, 0, 2000)) if not hist.empty else 800.0
    hist_yield = float(hist[TARGET_COLUMN].mean()) if not hist.empty else float("nan")

    sig = f"{location.latitude:.3f}_{location.longitude:.3f}_{snap.observed_at}_{crop}_{season}"
    t_col, h_col, r_col = st.columns(3)
    temp = t_col.number_input("Temperature (°C)", -10.0, 55.0, float(np.clip(snap.avg_temperature_7d_c, -10, 55)), 0.1,
                              key=f"lw_t_{sig}", help="Default: average of the last 7 days.")
    hum  = h_col.number_input("Relative humidity (%)", 0.0, 100.0, float(np.clip(snap.avg_humidity_7d_pct, 0, 100)), 1.0,
                              key=f"lw_h_{sig}", help="Default: average of the last 7 days.")
    rain = r_col.number_input("Seasonal rainfall (mm)", 0.0, 2000.0, rain_default, 10.0,
                              key=f"lw_r_{sig}",
                              help="The model expects a full-season total, which a live feed can't provide. "
                                   "Defaults to the historical average for this crop, season and district; adjust it as needed.")
    st.caption(f"{b_note} Temperature and humidity are the last 7 days' averages. "
               f"Rainfall is a seasonal assumption (last 7 days: {snap.rainfall_7d_mm:.1f} mm).")

    try:
        inputs    = _enrich({"crop_type": crop, "season": season, "temperature_c": float(temp),
                             "rainfall_mm": float(rain), "humidity_pct": float(hum),
                             "district": b_district, "state": b_state}, df_global)
        predicted = predict_one(model, inputs)
        risk      = compute_risk(inputs)
        recs      = recommendations(risk)
    except WeatherImpactError as exc:
        st.error(f"Couldn't compute a prediction for these conditions: {exc}")
        return

    k1, k2, k3, k4 = st.columns(4)
    delta = None if pd.isna(hist_yield) else f"{predicted - hist_yield:+.2f} vs historical"
    k1.metric("Predicted Yield", f"{predicted:.2f} t/ha", delta=delta,
              help="Compared with the historical average for this crop and season.")
    k2.metric("Risk Score (0–100)", f"{risk.score:.0f}")
    k3.metric("Risk Level", risk.level)
    k4.metric("Historical Average", "n/a" if pd.isna(hist_yield) else f"{hist_yield:.2f} t/ha")

    st.markdown("**Recommendations**")
    st.markdown('<div class="rec-box">' + "".join(f"<p>{r}</p>" for r in recs) + "</div>", unsafe_allow_html=True)

    nc, sc, oc = st.columns([3, 1, 1.4])
    name = nc.text_input("Scenario name", placeholder=f"e.g. {crop} {season} — {location.name} live",
                         label_visibility="collapsed", key="lw_save_name")
    with sc:
        if st.button("Save", key="lw_save", type="primary", width="stretch"):
            _save_scenario(name, inputs, predicted, risk.score, risk.level)
    with oc:
        if st.button("Open in Yield Prediction", key="lw_open", width="stretch"):
            st.session_state["_pending_field"] = {
                "field_crop": crop, "field_season": season,
                "field_temp": float(np.clip(round(temp * 2) / 2, 5.0, 50.0)),
                "field_rain": float(np.clip(round(rain / 10) * 10, 0.0, 2000.0)),
                "field_hum":  float(np.clip(round(hum), 10.0, 100.0)),
            }
            st.session_state.view = "yield prediction"
            st.rerun()

def render_live_weather() -> None:
    st.markdown('<div class="section-title">Live Weather</div>', unsafe_allow_html=True)
    st.markdown('<p class="page-lead">Current conditions and a 7-day forecast from Open-Meteo, fed directly into the prediction engine.</p>', unsafe_allow_html=True)

    pair_list = [(str(d), str(s)) for d, s in
                 df_global[["district", "state"]].drop_duplicates().sort_values(["district", "state"]).itertuples(index=False)]
    labels = [f"{d}, {s}" for d, s in pair_list]
    default_idx = next((i for i, (d, _) in enumerate(pair_list) if d == DEFAULT_LOCATION_NAME), 0)

    col_dist, col_search = st.columns(2)
    district_label = col_dist.selectbox("District", labels, index=default_idx, key="lw_district")
    query = col_search.text_input("Search location", placeholder="e.g. Chitradurga, Karnataka", key="lw_search").strip()
    sel_district, sel_state = pair_list[labels.index(district_label)]

    location: Location | None = None
    if query:
        if len(query) < 2:
            col_search.caption("Type at least 2 characters to search.")
        else:
            try:
                places = _live_search(query)
            except WeatherServiceError as exc:
                st.error(f"Place search failed. {exc}")
                return
            if not places:
                col_search.warning(f'No places found for "{query}". Showing the selected district instead.')
            else:
                idx = col_search.selectbox(
                    "Matching places", range(len(places)), key=f"lw_place_{query.lower()}",
                    format_func=lambda i: f"{places[i].label} ({places[i].latitude:.2f}, {places[i].longitude:.2f})",
                )
                location = places[idx]
    from_dataset = location is None
    if location is None:
        coords = get_district_coords(sel_district, sel_state)
        if coords is None:
            st.error(f"No coordinates are known for {sel_district}. Use the search box to pick a place instead.")
            return
        location = Location(name=sel_district, admin1=sel_state, country="India", latitude=coords[0], longitude=coords[1])

    head, refresh = st.columns([4, 1])
    head.markdown(f"**{location.label}** · {location.latitude:.2f}°, {location.longitude:.2f}°")
    if refresh.button("Refresh", key="lw_refresh", width="stretch"):
        _live_snapshot.clear()

    try:
        with st.spinner("Fetching live weather…"):
            snap = _live_snapshot(location.name, location.admin1, location.country, location.latitude, location.longitude)
    except WeatherServiceError as exc:
        st.error(f"Live weather is unavailable right now. {exc}")
        st.button("Try again", key="lw_retry")
        return

    head.caption(f"Observed {snap.observed_at.replace('T', ' ')} (local time at the location). Updates are cached for up to 10 minutes.")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Temperature now", f"{snap.current_temperature_c:.1f} °C", help=f"Last 7 days average: {snap.avg_temperature_7d_c:.1f} °C")
    k2.metric("Humidity now", f"{snap.current_humidity_pct:.0f}%", help=f"Last 7 days average: {snap.avg_humidity_7d_pct:.0f}%")
    k3.metric("Rain now", f"{snap.current_precipitation_mm:.1f} mm")
    k4.metric("Rain, last 7 days", f"{snap.rainfall_7d_mm:.1f} mm")
    st.caption(f"Last 7 days: average temperature {snap.avg_temperature_7d_c:.1f} °C, average humidity {snap.avg_humidity_7d_pct:.0f}%.")

    if snap.forecast:
        chart_col, table_col = st.columns([1.5, 1])
        with chart_col, st.container(border=True):
            _show(_live_forecast_chart(snap))
        with table_col:
            table = pd.DataFrame({
                "Day": [_fmt_day(d.date) for d in snap.forecast],
                "Max °C": [d.temp_max_c for d in snap.forecast],
                "Min °C": [d.temp_min_c for d in snap.forecast],
                "Rain mm": [d.precipitation_mm for d in snap.forecast],
                "Rain chance": [None if d.precipitation_probability_pct is None else d.precipitation_probability_pct / 100
                                for d in snap.forecast],
            })
            st.dataframe(
                table, hide_index=True, width="stretch",
                column_config={
                    "Max °C": st.column_config.NumberColumn(format="%.1f"),
                    "Min °C": st.column_config.NumberColumn(format="%.1f"),
                    "Rain mm": st.column_config.NumberColumn(format="%.1f"),
                    "Rain chance": st.column_config.NumberColumn(format="percent"),
                },
            )
    else:
        st.info("The service returned no forecast days for this location.")

    _live_prediction(snap, location, from_dataset, sel_district, sel_state)

# ─────────────────────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────────────────────
_v = st.session_state.view
if   _v == "home":                  render_home()
elif _v == "dashboard":             render_dashboard()
elif _v == "yield prediction":      render_yield_prediction()
elif _v == "what if":               render_what_if()
elif _v == "risk score":            render_risk_score()
elif _v == "saved scenarios":       render_saved_scenarios()
elif _v == "live weather":          render_live_weather()
else:
    st.session_state.view = "home"
    st.rerun()