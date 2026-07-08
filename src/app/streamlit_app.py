"""AirCast Streamlit app: next-day AQI forecast by city.

Renders a sidebar with city and base-date selectors and displays the next-day
AQI value, category, and health advisory for the chosen city.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src import config
from src.models.predictor import Predictor


# Display-only: severity color per AQI category, matching CPCB's convention.
# App-layer presentation, deliberately kept out of config (which is data/model truth).
CATEGORY_COLORS: dict[str, str] = {
    "Good":         "#4caf50",
    "Satisfactory": "#8bc34a",
    "Moderate":     "#ffb300",
    "Poor":         "#fb8c00",
    "Very Poor":    "#e53935",
    "Severe":       "#b71c1c",
}
# How many days of real history to show behind the forecast point.
HISTORY_DAYS: int = 30

@st.cache_resource
def load_predictor() -> Predictor:
    """Load the Predictor once per app session (models + feature table are heavy)."""
    return Predictor()


def get_city_options(predictor: Predictor) -> list[str]:
    """Return sorted list of city_ids available for selection."""
    return sorted(predictor.features[config.CITY_ID_COL].unique())


def get_date_range(predictor: Predictor, city_id: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (min_date, max_date) of available feature rows for one city."""
    city_rows = predictor.features.loc[predictor.features[config.CITY_ID_COL] == city_id]
    return city_rows[config.DATE_COL].min(), city_rows[config.DATE_COL].max()

def render_forecast(result: dict) -> None:
    """Display a forecast result as a colored metric card with advisory.

    Args:
        result: Output of ``Predictor.predict`` — keys ``aqi``, ``category``,
            ``advisory``, and ``forecast_date``.
    """
    color = CATEGORY_COLORS[result["category"]]
    st.markdown(f"### Forecast for {result['forecast_date']}")
    left, right = st.columns([1, 2])
    left.metric("Predicted AQI", result["aqi"])
    right.markdown(
        f"<div style='padding:0.75rem 1rem;border-radius:0.5rem;"
        f"background:{color};color:white;font-weight:600;font-size:1.1rem;'>"
        f"{result['category']}</div>",
        unsafe_allow_html=True,
    )
    st.write(result["advisory"])

def render_history_chart(
    predictor: Predictor, city_id: str, base_date, result: dict
) -> None:
    """Plot recent actual AQI for a city with the forecast point appended.

    Shows the ``HISTORY_DAYS`` days of measured AQI ending on ``base_date``,
    then marks the next-day forecast so it can be read against recent history.

    Args:
        predictor: Loaded Predictor holding the feature table.
        city_id: Canonical "city, state" identity.
        base_date: The day the forecast is made from (history ends here).
        result: Output of ``Predictor.predict`` (supplies the forecast point).
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

    forecast_date = pd.Timestamp(result["forecast_date"])
    color = CATEGORY_COLORS[result["category"]]

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(
        history[config.DATE_COL], history[config.AQI_COL],
        marker="o", markersize=3, linewidth=1.5, color="#5c6bc0", label="Actual AQI",
    )
    ax.scatter(
        forecast_date, result["aqi"],
        color=color, s=90, zorder=5, edgecolor="white", label="Forecast",
    )
    ax.set_title(f"{city_id} — last {HISTORY_DAYS} days")
    ax.set_ylabel("AQI")
    ax.legend(loc="upper right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.autofmt_xdate()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def main() -> None:
    """Render the AirCast app: sidebar selectors + a placeholder body for now."""
    st.set_page_config(
        page_title="AirCast — Next-Day AQI Forecast", page_icon="🌫️", layout="centered"
    )
    st.title("🌫️ AirCast")
    st.caption("Next-day air quality forecasting for Indian cities")

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

    result = predictor.predict(city_id, selected_date)
    render_forecast(result)
    render_history_chart(predictor, city_id, selected_date, result)


if __name__ == "__main__":
    main()