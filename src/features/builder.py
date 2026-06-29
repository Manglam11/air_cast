"""FeatureBuilder: turn the clean daily AQI series into a model-ready table.

Takes the gap-free city-day frame from Cleaner and produces one row per
(city_id, date) carrying leakage-safe features that describe day T or earlier,
paired with day T+1's AQI value and category as targets. Rows whose structural
features or targets cannot exist (each city's warm-up days and its final day)
are dropped; ordinary pollutant missingness is left for the modeling Pipeline's
imputer.
"""
from __future__ import annotations

import pandas as pd

from src import config


class FeatureBuilder:
    """Build leakage-safe forecasting features from a clean city-day frame.

    Input is the gap-free daily frame from ``Cleaner`` (one row per
    ``(city_id, date)`` on a complete daily grid per city). Output is a
    model-ready frame: metadata, the feature matrix, and both targets, with
    structurally-unusable rows removed.

    All features describe day T or earlier; the targets describe day T+1. Lag
    and rolling features are built per city after a chronological sort, so a
    city never borrows another's history, and they are correct only because the
    input grid is gap-free.

    Attributes:
        lag_days: Day offsets used to build AQI lag features.
        roll_windows: Window sizes, in days, for the rolling mean and std.
    """

    def __init__(self) -> None:
        self.lag_days: list[int] = config.LAG_DAYS
        self.roll_windows: list[int] = config.ROLL_WINDOWS

    def build(self, clean: pd.DataFrame) -> pd.DataFrame:
        """Build the model-ready feature table from a clean city-day frame.

        Args:
            clean: Gap-free daily frame from Cleaner, one row per
                ``(city_id, date)``.

        Returns:
            A model-ready frame with metadata, feature, and target columns,
            sorted by ``(city_id, date)`` and free of structural NaNs.
        """
        df = clean.sort_values(
            [config.CITY_ID_COL, config.DATE_COL]
        ).reset_index(drop=True)
        self._check_ordering(df)

        df = self._add_targets(df)
        df = self._add_lags(df)
        df = self._add_rollings(df)
        df = self._add_calendar(df)

        model_df = self._assemble_and_drop(df)
        self._validate(model_df)
        return model_df

    def _check_ordering(self, df: pd.DataFrame) -> None:
        """Assert dates climb in order inside every city.

        Lag and rolling features depend entirely on this ordering; if it is
        wrong, the leakage guarantees break with no error to warn us.

        Raises:
            AssertionError: If any city's dates are not monotonically
                increasing.
        """
        ordered = (
            df.groupby(config.CITY_ID_COL)[config.DATE_COL]
            .apply(lambda s: s.is_monotonic_increasing)
            .all()
        )
        assert ordered, "Dates are not ordered within every city."

    def _add_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add next-day AQI value and category targets, per city.

        Each city's AQI and bucket columns are slid up one row with shift(-1),
        so day T+1's values land on day T. Each city's final row gets a NaN
        target by design and is dropped later.
        """
        out = df.copy()
        by_city = out.groupby(config.CITY_ID_COL)
        out[config.TARGET_AQI_COL] = by_city[config.AQI_COL].shift(-1)
        out[config.TARGET_BUCKET_COL] = by_city[config.AQI_BUCKET_COL].shift(-1)
        return out

    def _add_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add AQI lag features (yesterday through last week), per city.

        For each offset k, shift(k) slides the AQI column down k rows, so the
        value from k days ago lands on day T. Built per city so no city borrows
        another's history; correct only because the grid is gap-free.
        """
        out = df.copy()
        aqi_by_city = out.groupby(config.CITY_ID_COL)[config.AQI_COL]
        for k in self.lag_days:
            out[f"aqi_lag_{k}"] = aqi_by_city.shift(k)
        return out

    def _add_rollings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling mean and std of AQI over each window, per city.

        Each window looks backward and ends on day T, so it uses only data at
        T or earlier. ``min_periods`` equals the window, so a value is emitted
        only once the window holds its full count of real days.
        """
        out = df.copy()
        aqi_by_city = out.groupby(config.CITY_ID_COL)[config.AQI_COL]
        for w in self.roll_windows:
            roll = aqi_by_city.rolling(window=w, min_periods=w)
            out[f"aqi_roll_mean_{w}"] = roll.mean().reset_index(level=0, drop=True)
            out[f"aqi_roll_std_{w}"] = roll.std().reset_index(level=0, drop=True)
        return out

    def _add_calendar(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cyclical calendar features from the date column.

        Extracts month, day of year, day of week, and a weekend flag, and maps
        month to an India-appropriate season bucket. These describe day T
        itself, so they add no NaNs and carry no leakage risk.
        """
        out = df.copy()
        date_parts = out[config.DATE_COL].dt
        out[config.MONTH_COL] = date_parts.month
        out[config.DAY_OF_YEAR_COL] = date_parts.dayofyear
        out[config.DAY_OF_WEEK_COL] = date_parts.dayofweek
        out[config.IS_WEEKEND_COL] = (date_parts.dayofweek >= 5).astype("int8")
        out[config.SEASON_COL] = out[config.MONTH_COL].map(config.MONTH_TO_SEASON)
        return out

    def _assemble_and_drop(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select final columns and drop structurally-unusable rows.

        Keeps metadata, feature, and target columns. Drops any row whose
        structural columns (lags, rollings, targets) are NaN — these cannot be
        filled. Pollutant level NaNs are left in deliberately, to be imputed by
        the modeling Pipeline on training folds only.
        """
        keep = (
            config.META_COLS
            + config.FEATURE_COLS
            + [config.TARGET_AQI_COL, config.TARGET_BUCKET_COL]
        )
        model_df = df[keep].copy()
        model_df = model_df.dropna(subset=config.STRUCTURAL_COLS)
        return model_df.reset_index(drop=True)

    def _validate(self, model_df: pd.DataFrame) -> None:
        """Check the model-ready frame meets the guarantees modeling relies on.

        Raises:
            AssertionError: If the frame is empty, has duplicate city-days,
                retains structural NaNs, or has lost all pollutant missingness
                (which would mean imputation was silently skipped upstream).
        """
        assert len(model_df) > 0, "Feature frame has no rows."

        duplicate_days = model_df.duplicated(
            subset=[config.CITY_ID_COL, config.DATE_COL]
        ).sum()
        assert duplicate_days == 0, f"Found {duplicate_days} duplicate city-days."

        structural_nans = model_df[config.STRUCTURAL_COLS].isna().sum().sum()
        assert structural_nans == 0, f"{structural_nans} structural NaNs remain."

        pollutant_nans = model_df[config.MODEL_POLLUTANT_COLS].isna().sum().sum()
        assert pollutant_nans > 0, (
            "Expected pollutant NaNs to remain for downstream imputation."
        )