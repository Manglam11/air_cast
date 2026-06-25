"""Cleaner: turn the raw station-day AQI frame into a modeling-ready city-day frame.

Executes the banked EDA decisions in dependency order and emits one clean
city-day series per city. This is the single input the feature-engineering
stage (Session 4) will ever see.
"""
from __future__ import annotations

import pandas as pd

from src import config
from src.data.aqi import AQICalculator


class Cleaner:
    """Clean the raw daily AQI frame into a modeling-ready city-day frame.

    Pipeline: drop sparse VOC columns, restrict to the modeling window, collapse
    multi-station cities to one city-day series (recomputing CPCB AQI on the
    pooled concentrations), drop rows with no AQI label, keep only cities with
    enough labelled history, reindex each city to a gap-free daily grid, and
    forward-fill short predictor gaps in a leakage-safe, capped way.
    """

    def __init__(self) -> None:
        self.calculator = AQICalculator()

    def clean(self) -> pd.DataFrame:
        """Run the full cleaning pipeline and return the city-day frame."""
        df = pd.read_parquet(config.DAILY_AQI_PATH)

        df = self._drop_voc_columns(df)
        df = self._apply_modeling_window(df)
        city_day = self._collapse_stations(df)
        city_day = self._add_city_id(city_day)
        city_day = self._drop_unlabelled_rows(city_day)
        city_day = self._filter_thin_cities(city_day)
        city_day = self._reindex_daily_grid(city_day)
        city_day = self._fill_short_gaps(city_day)
        city_day = self._order_columns(city_day)

        self._validate(city_day)
        return city_day

    def _drop_voc_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop the sparse VOC pollutant columns (benzene, toluene, xylene)."""
        return df.drop(columns=config.VOC_COLS)

    def _apply_modeling_window(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only rows from MODELING_START_YEAR onward."""
        return df[df[config.DATE_COL].dt.year >= config.MODELING_START_YEAR]

    def _collapse_stations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pool stations within a city-day and recompute one CPCB AQI.

        Averages raw pollutant concentrations across a city's stations, then
        recomputes the AQI once on the pooled values, rather than averaging the
        nonlinear per-station AQI numbers directly. Selecting only the pollutant
        columns here also drops the stale per-station aqi, bucket, and _n columns.
        """
        pooled = (
            df.groupby(
                [config.CITY_COL, config.STATE_COL, config.DATE_COL],
                as_index=False,
            )[config.MODEL_POLLUTANT_COLS]
            .mean()
        )
        return self.calculator.compute(pooled)

    def _add_city_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add the canonical city_id identity column ("city, state").

        Aurangabad exists in both Bihar and Maharashtra, so city name alone is
        not a unique key; city_id is the identity used for every per-city step.
        """
        out = df.copy()
        out[config.CITY_ID_COL] = (
            out[config.CITY_COL] + ", " + out[config.STATE_COL]
        )
        return out

    def _drop_unlabelled_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows with no AQI label — useless for forecasting."""
        return df[df[config.AQI_COL].notna()]

    def _filter_thin_cities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only cities with at least MIN_CITY_DAYS labelled days.

        transform broadcasts each city's day-count back onto its own rows, so
        the result is a full-length boolean mask over the frame.
        """
        day_counts = df.groupby(config.CITY_ID_COL)[config.DATE_COL].transform("size")
        return df[day_counts >= config.MIN_CITY_DAYS]

    def _reindex_daily_grid(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reindex every city to a gap-free daily date range.

        Inserts a NaN row for every calendar day missing between a city's first
        and last labelled day, so 'previous row' always means 'yesterday' for
        the lag features built later. Identity columns, blanked by reindex on the
        inserted rows, are restored from the city's own values.
        """
        pieces = []
        for _, group in df.groupby(config.CITY_ID_COL):
            full_range = pd.date_range(
                group[config.DATE_COL].min(),
                group[config.DATE_COL].max(),
                freq="D",
            )
            gridded = group.set_index(config.DATE_COL).reindex(full_range)
            gridded.index.name = config.DATE_COL

            identity = group.iloc[0]
            for col in config.IDENTITY_COLS:
                gridded[col] = identity[col]

            pieces.append(gridded.reset_index())
        return pd.concat(pieces, ignore_index=True)

    def _fill_short_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill short predictor gaps per city, leakage-safe and capped.

        Fills only the pollutant predictors, never the AQI label, using only
        past readings (forward-fill) capped at FILL_LIMIT days so long silences
        stay NaN rather than being fabricated.
        """
        out = df.sort_values(
            [config.CITY_ID_COL, config.DATE_COL]
        ).reset_index(drop=True)
        out[config.MODEL_POLLUTANT_COLS] = (
            out.groupby(config.CITY_ID_COL)[config.MODEL_POLLUTANT_COLS]
            .ffill(limit=config.FILL_LIMIT)
        )
        return out

    def _order_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enforce a tidy, predictable output column order and row order."""
        ordered = (
            [config.CITY_ID_COL, config.CITY_COL, config.STATE_COL, config.DATE_COL]
            + config.MODEL_POLLUTANT_COLS
            + [config.AQI_COL, config.AQI_BUCKET_COL]
        )
        return (
            df[ordered]
            .sort_values([config.CITY_ID_COL, config.DATE_COL])
            .reset_index(drop=True)
        )

    def _validate(self, df: pd.DataFrame) -> None:
        """Fail loudly if the cleaned frame breaks its downstream guarantees."""
        assert len(df) > 0, "Cleaned frame is empty."

        dupes = df.duplicated(
            subset=[config.CITY_ID_COL, config.DATE_COL]
        ).sum()
        assert dupes == 0, f"Found {dupes} duplicate city-day rows."

        # every surviving city has at least the minimum labelled history
        labelled = df[df[config.AQI_COL].notna()]
        days_per_city = labelled.groupby(config.CITY_ID_COL)[config.DATE_COL].nunique()
        assert (days_per_city >= config.MIN_CITY_DAYS).all(), (
            "A surviving city has fewer than MIN_CITY_DAYS labelled days."
        )

        # the grid is gap-free: each city's row count equals its full day-span
        spans = df.groupby(config.CITY_ID_COL)[config.DATE_COL].agg(
            ["min", "max", "nunique"]
        )
        expected = (spans["max"] - spans["min"]).dt.days + 1
        assert (expected == spans["nunique"]).all(), (
            "A city still has missing calendar dates after gridding."
        )

        # label integrity: aqi was never filled, so aqi and aqi_bucket must agree
        # on which rows are missing
        assert (
            df[config.AQI_COL].isna() == df[config.AQI_BUCKET_COL].isna()
        ).all(), "aqi and aqi_bucket missingness disagree — label integrity broken."