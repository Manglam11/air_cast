import numpy as np
import pandas as pd

from src import config


class AQICalculator:
    """Compute CPCB AQI values and category labels for daily station data."""

    def compute(self, daily: pd.DataFrame) -> pd.DataFrame:
        """Return the daily frame with ``aqi`` and ``aqi_bucket`` columns added.

        Args:
            daily: One row per station-day with daily pollutant concentrations.

        Returns:
            The frame plus a numeric ``aqi`` column and a categorical
            ``aqi_bucket`` column; both are NaN for days with insufficient data.
        """
        result = daily.copy()
        result[config.AQI_COL] = result.apply(self._row_aqi, axis=1)
        result[config.AQI_BUCKET_COL] = result[config.AQI_COL].apply(self._bucket)
        self._validate(result)
        return result

    def _row_aqi(self, row: pd.Series) -> float:
        """Compute one station-day's AQI as the max of its valid sub-indices.

        Returns NaN unless at least ``AQI_MIN_POLLUTANTS`` sub-indices exist
        and at least one belongs to a pollutant in ``AQI_REQUIRED_ANY``.
        """
        sub_indices = {}
        for pollutant in config.CPCB_AQI_POLLUTANTS:
            value = row.get(pollutant, np.nan)
            sub = self._sub_index(value, config.AQI_BREAKPOINTS[pollutant])
            if not pd.isna(sub):
                sub_indices[pollutant] = sub

        if len(sub_indices) < config.AQI_MIN_POLLUTANTS:
            return np.nan
        if not any(p in sub_indices for p in config.AQI_REQUIRED_ANY):
            return np.nan

        return max(sub_indices.values())

    def _sub_index(self, concentration: float, breakpoints: list) -> float:
        """Convert one pollutant concentration to its CPCB sub-index.

        Args:
            concentration: The pollutant's daily concentration value.
            breakpoints: Ordered ``(conc_low, conc_high, aqi_low, aqi_high)``
                bands for the pollutant, lowest band first.

        Returns:
            The interpolated sub-index clamped to ``0``-``500``, or ``NaN``
            when the concentration is missing.
        """
        if pd.isna(concentration):
            return np.nan

        for conc_low, conc_high, aqi_low, aqi_high in breakpoints:
            if concentration <= conc_high:
                sub_index = (
                    (aqi_high - aqi_low) / (conc_high - conc_low)
                    * (concentration - conc_low)
                    + aqi_low
                )
                return float(min(max(sub_index, 0.0), 500.0))

        return 500.0

    def _bucket(self, aqi: float) -> str:
        """Map a numeric AQI value to its CPCB category label, or NaN if missing."""
        if pd.isna(aqi):
            return np.nan
        for category, upper_bound in config.AQI_BANDS:
            if aqi <= upper_bound:
                return category
        return config.AQI_BANDS[-1][0]

    def _validate(self, frame: pd.DataFrame) -> None:
        """Check that AQI columns exist and all values fall in the 0-500 range."""
        assert config.AQI_COL in frame, "aqi column missing."
        assert config.AQI_BUCKET_COL in frame, "aqi_bucket column missing."
        valid_aqi = frame[config.AQI_COL].dropna()
        assert valid_aqi.between(0, 500).all(), "AQI values fell outside 0-500."