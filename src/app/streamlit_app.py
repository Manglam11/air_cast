"""AirCast Streamlit app: next-day AQI forecast by city.

Renders a sidebar with city and base-date selectors and displays the next-day
AQI value, category, and health advisory, plus an interactive chart of recent
actual AQI with the forecast point shown against CPCB severity bands.
"""
from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run` puts src/app/ on the path, not the project root, so `src`
# isn't importable by default. Add the project root (two levels up) first.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.models.predictor import Predictor

# Display-only severity colors per AQI category (CPCB convention).
# App-layer presentation — deliberately kept out of config (data/model truth).
CATEGORY_COLORS: dict[str, str] = {
    "Good":         "#4caf50",
    "Satisfactory": "#9ccc65",
    "Moderate":     "#ffb300",
    "Poor":         "#fb8c00",
    "Very Poor":    "#e53935",
    "Severe":       "#b71c1c",
}

# How many days of real history to show behind the forecast point.
HISTORY_DAYS: int = 30

# Custom styling for the forecast hero card. Scoped to our own class names so it
# doesn't depend on Streamlit's internal DOM (which changes between versions).
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

.stApp { font-family: 'Inter', sans-serif; }
footer { visibility: hidden; }

.ac-title {
    font-size: 2.6rem; font-weight: 800; letter-spacing: -0.02em;
    background: linear-gradient(90deg, #7c8cf0, #a5b4fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.ac-tagline { color: rgba(255,255,255,0.5); font-size: 0.95rem; margin-top: -0.2rem; }

.ac-card {
    --cat: #888;
    padding: 1.5rem 1.75rem; margin-top: 0.5rem;
    border-radius: 18px;
    background: linear-gradient(145deg, rgba(255,255,255,0.05), rgba(255,255,255,0.01));
    border: 1px solid rgba(255,255,255,0.08);
    border-left: 4px solid var(--cat);
    box-shadow: 0 10px 34px -14px var(--cat);
    animation: ac-rise 0.5s cubic-bezier(0.22, 1, 0.36, 1);
}
.ac-card__date {
    font-size: 0.75rem; letter-spacing: 0.09em; text-transform: uppercase;
    color: rgba(255,255,255,0.5); margin-bottom: 0.9rem;
}
.ac-card__row { display: flex; align-items: center; justify-content: space-between; gap: 1rem; }
.ac-card__aqi { display: flex; align-items: baseline; gap: 0.45rem; }
.ac-card__value { font-size: 3.4rem; font-weight: 800; line-height: 1; color: #fff; }
.ac-card__unit { font-size: 1rem; font-weight: 600; color: rgba(255,255,255,0.5); }
.ac-card__badge {
    padding: 0.55rem 1.2rem; border-radius: 999px;
    background: var(--cat); color: #fff; font-weight: 700; font-size: 1.05rem;
    box-shadow: 0 4px 18px -4px var(--cat); text-shadow: 0 1px 2px rgba(0,0,0,0.35);
}
.ac-card__advisory {
    margin-top: 1.1rem; color: rgba(255,255,255,0.82);
    line-height: 1.55; font-size: 0.95rem;
}
@keyframes ac-rise {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
"""


@st.cache_resource
def load_predictor() -> Predictor:
    """Load the Predictor once per session (models + feature table are heavy)."""
    return Predictor()


def get_city_options(predictor: Predictor) -> list[str]:
    """Return the sorted list of city_ids available for selection."""
    return sorted(predictor.features[config.CITY_ID_COL].unique())


def get_date_range(
    predictor: Predictor, city_id: str
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the (min_date, max_date) of feature rows for one city."""
    city_rows = predictor.features.loc[
        predictor.features[config.CITY_ID_COL] == city_id
    ]
    return city_rows[config.DATE_COL].min(), city_rows[config.DATE_COL].max()


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a #rrggbb hex color to an rgba() string with the given alpha."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def _category_bands() -> list[tuple[str, int, int]]:
    """Return (category, lower_bound, upper_bound) for each AQI band."""
    bands, lower = [], 0
    for name, upper in config.AQI_BANDS:
        bands.append((name, lower, upper))
        lower = upper
    return bands


def render_forecast_card(result: dict) -> None:
    """Render the forecast as a styled hero card (value, category, advisory).

    Args:
        result: Output of ``Predictor.predict`` — keys ``aqi``, ``category``,
            ``advisory``, and ``forecast_date``.
    """
    color = CATEGORY_COLORS[result["category"]]
    st.markdown(
        f"""
        <div class="ac-card" style="--cat: {color};">
            <div class="ac-card__date">Forecast &middot; {result['forecast_date']}</div>
            <div class="ac-card__row">
                <div class="ac-card__aqi">
                    <span class="ac-card__value">{result['aqi']}</span>
                    <span class="ac-card__unit">AQI</span>
                </div>
                <div class="ac-card__badge">{result['category']}</div>
            </div>
            <div class="ac-card__advisory">{result['advisory']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_history_figure(
    predictor: Predictor, city_id: str, base_date, result: dict
) -> go.Figure | None:
    """Build a Plotly chart: recent actual AQI + forecast on severity bands.

    Args:
        predictor: Loaded Predictor holding the feature table.
        city_id: Canonical "city, state" identity.
        base_date: The day the forecast is made from (history ends here).
        result: Output of ``Predictor.predict`` (supplies the forecast point).

    Returns:
        A Plotly figure, or None if there is no history to plot.
    """
    base_date = pd.Timestamp(base_date)
    window_start = base_date - pd.Timedelta(days=HISTORY_DAYS)
    feats = predictor.features
    mask = (
        (feats[config.CITY_ID_COL] == city_id)
        & (feats[config.DATE_COL] > window_start)
        & (feats[config.DATE_COL] <= base_date)
    )
    history = feats.loc[mask, [config.DATE_COL, config.AQI_COL]].sort_values(
        config.DATE_COL
    )
    if history.empty:
        return None

    forecast_date = pd.Timestamp(result["forecast_date"])
    forecast_aqi = result["aqi"]
    color = CATEGORY_COLORS[result["category"]]

    y_max = max(history[config.AQI_COL].max(), forecast_aqi) * 1.15
    y_max = max(y_max, 60)  # always show at least Good + Satisfactory bands

    fig = go.Figure()

    # Faint severity bands behind everything, so the line's health zone is visible.
    for name, lower, upper in _category_bands():
        if lower >= y_max:
            break
        fig.add_hrect(
            y0=lower, y1=min(upper, y_max),
            fillcolor=_hex_to_rgba(CATEGORY_COLORS[name], 0.10),
            line_width=0, layer="below",
            annotation_text=name, annotation_position="right",
            annotation_font_size=9,
            annotation_font_color=_hex_to_rgba(CATEGORY_COLORS[name], 0.9),
        )

    # Actual AQI — straight segments (no spline; we don't invent curvature).
    fig.add_trace(go.Scatter(
        x=history[config.DATE_COL], y=history[config.AQI_COL],
        mode="lines+markers", name="Actual AQI",
        line=dict(color="#7c8cf0", width=2.5),
        marker=dict(size=5, color="#7c8cf0"),
        hovertemplate="%{x|%b %d}<br>AQI %{y:.0f}<extra></extra>",
    ))

    # Dotted connector from the last real day into tomorrow's forecast.
    last = history.iloc[-1]
    fig.add_trace(go.Scatter(
        x=[last[config.DATE_COL], forecast_date],
        y=[last[config.AQI_COL], forecast_aqi],
        mode="lines", line=dict(color=color, width=2, dash="dot"),
        showlegend=False, hoverinfo="skip",
    ))

    # The forecast point, colored by its category.
    fig.add_trace(go.Scatter(
        x=[forecast_date], y=[forecast_aqi],
        mode="markers", name="Forecast",
        marker=dict(size=15, color=color, line=dict(color="white", width=2)),
        hovertemplate="%{x|%b %d} (forecast)<br>AQI %{y:.1f}<extra></extra>",
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10), height=380,
        yaxis=dict(title="AQI", range=[0, y_max],
                   gridcolor="rgba(255,255,255,0.06)", zeroline=False),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        hovermode="closest",
        transition=dict(duration=400, easing="cubic-in-out"),
    )
    return fig


def main() -> None:
    """Render the AirCast app: header, sidebar selectors, forecast, and chart."""
    st.set_page_config(
        page_title="AirCast — Next-Day AQI Forecast", page_icon="📊", layout="centered"
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown('<div class="ac-title">📊 AirCast</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ac-tagline">Next-day air quality forecasting for Indian cities</div>',
        unsafe_allow_html=True,
    )

    predictor = load_predictor()

    st.sidebar.header("Forecast inputs")
    city_id = st.sidebar.selectbox("City", get_city_options(predictor))
    min_date, max_date = get_date_range(predictor, city_id)
    selected_date = st.sidebar.date_input(
        "Base date (forecast is for the day after)",
        value=max_date.date(),
        min_value=min_date.date(),
        max_value=max_date.date(),
    )

    try:
        result = predictor.predict(city_id, selected_date)
    except ValueError:
        st.warning(
            f"No data available for {city_id} on {selected_date}. "
            "This day was dropped in cleaning (a gap in the records). "
            "Try a nearby date."
        )
        return

    render_forecast_card(result)

    fig = build_history_figure(predictor, city_id, selected_date, result)
    if fig is None:
        st.caption("No recent history to plot for this city and date.")
    else:
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})


if __name__ == "__main__":
    main()