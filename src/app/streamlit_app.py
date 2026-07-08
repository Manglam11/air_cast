"""AirCast Streamlit app: next-day AQI forecast by city.

Bucket 1 — skeleton: page setup, cached Predictor load, city/date selectors.
Forecast display and charts land in later buckets.
"""
from __future__ import annotations

import sys
from pathlib import Path

# streamlit run adds src/app/ to sys.path, not the project root, so `src`
# isn't importable by default. Put the project root (two levels up) on the
# path before any src import so `from src import ...` resolves.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src import config
from src.models.predictor import Predictor


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

    st.write(f"City: **{city_id}**")
    st.write(
        f"Base date: **{selected_date}** → forecasting "
        f"**{selected_date + pd.Timedelta(days=1)}**"
    )
    st.info("Forecast card lands in Bucket 2.")


if __name__ == "__main__":
    main()