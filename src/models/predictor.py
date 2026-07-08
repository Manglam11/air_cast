"""Predictor: turn a city + date into a next-day AQI forecast with advisory.

Loads the fitted champion pipelines (regression + classification) persisted in
Session 8 and serves next-day forecasts. The regression pipeline gives the
numeric AQI; the classification pipeline gives the category (it catches rare
hazardous days better than bucketing the regression output would), and the
advisory follows the predicted category.
"""
from __future__ import annotations

import joblib
import pandas as pd

from src import config


class Predictor:
    """Serve next-day AQI forecasts from the persisted champion pipelines.

    Loads the fitted pipelines once and keeps the feature table in memory so any
    historical day can be looked up and fed to the models. Since v1 has no live
    feed, a forecast reads a real day's feature row and predicts the day after.

    Attributes:
        reg: Fitted regression pipeline (Ridge) -> next-day AQI value.
        clf: Fitted classification pipeline (LightGBM) -> next-day category.
        features: The full feature table used for city/date lookups.
    """

    def __init__(self, features: pd.DataFrame | None = None) -> None:
        """Load the saved pipelines and the feature table.

        Args:
            features: Optional pre-loaded feature frame; read from disk if None.
        """
        self.reg = joblib.load(config.MODELS_DIR / config.REG_MODEL_FILE)
        self.clf = joblib.load(config.MODELS_DIR / config.CLF_MODEL_FILE)
        if features is None:
            features = pd.read_parquet(config.FEATURES_PATH)
        self.features = features

    def _feature_row(self, city_id: str, date) -> pd.DataFrame:
        """Fetch the single feature row for one city on one date.

        Args:
            city_id: Canonical "city, state" identity.
            date: The day whose features drive the T+1 forecast.

        Returns:
            A one-row DataFrame of ``config.MODEL_FEATURES``.

        Raises:
            ValueError: If no row exists for that city and date.
        """
        date = pd.Timestamp(date)
        mask = (
            (self.features[config.CITY_ID_COL] == city_id)
            & (self.features[config.DATE_COL] == date)
        )
        row = self.features.loc[mask]
        if row.empty:
            raise ValueError(f"no feature row for {city_id!r} on {date.date()}")
        return row[config.MODEL_FEATURES]

    def predict(self, city_id: str, date) -> dict:
        """Forecast next-day AQI value, category, and health advisory.

        Runs the regression pipeline for the numeric next-day AQI and the
        classification pipeline for the next-day category; the advisory follows
        the predicted category. The forecast is for the day after ``date``.

        Args:
            city_id: Canonical "city, state" identity.
            date: The day whose features drive the T+1 forecast.

        Returns:
            A dict with ``city_id``, ``forecast_date`` (D+1), ``aqi`` (float),
            ``category`` (str), and ``advisory`` (str).
        """
        row = self._feature_row(city_id, date)
        aqi_value = float(self.reg.predict(row)[0])
        cat_id = int(self.clf.predict(row)[0])
        category = config.AQI_CATEGORIES[cat_id]
        return {
            "city_id": city_id,
            "forecast_date": (pd.Timestamp(date) + pd.Timedelta(days=1)).date(),
            "aqi": round(aqi_value, 1),
            "category": category,
            "advisory": config.ADVISORY[category],
        }