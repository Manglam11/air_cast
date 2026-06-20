"""DailyAggregator: collapse the hourly station frame into daily station rows.

For each station and calendar day, it produces the daily statistic the CPCB AQI
rules require: a 24-hour mean for most pollutants and the worst rolling 8-hour
average for CO and O3, plus the number of valid hours behind each pollutant.
Computing the AQI value itself is done separately by AQICalculator.

"""
from __future__ import annotations

import pandas as pd

from src import config


class DailyAggregator:
    """Aggregate an hourly station frame into a tidy daily station frame.

    Input is the tidy hourly frame from ``DataLoader`` (one row per
    station-hour). Output is one row per ``(station, date)``, carrying each
    pollutant summarised by its CPCB-correct daily statistic, plus the number
    of valid hours that backed each CPCB pollutant.

    Attributes:
        mean_pollutants: Pollutants summarised by the 24-hour mean.
        max8h_pollutants: Pollutants summarised by the worst 8-hour average.
        cpcb_pollutants: The 7 pollutants whose valid-hour counts we record.
    """

    def __init__(self) -> None:
        self.mean_pollutants: list[str] = config.AQI_MEAN_POLLUTANTS
        self.max8h_pollutants: list[str] = config.AQI_MAX8H_POLLUTANTS
        self.cpcb_pollutants: list[str] = config.CPCB_AQI_POLLUTANTS

    def aggregate(self, hourly: pd.DataFrame) -> pd.DataFrame:
        """Collapse an hourly frame into one row per station-day.

        Args:
            hourly: Tidy hourly frame from DataLoader, sorted by
                ``[city, station, datetime]``.

        Returns:
            A daily frame with one row per ``(station, date)``: identity
            columns ``[city, state, station, date]``, each pollutant's daily
            statistic, and ``<pollutant>_n`` valid-hour counts for the CPCB
            pollutants.
        """
        dated = self._add_date_column(hourly)

        keys = [
            config.CITY_COL,
            config.STATE_COL,
            config.STATION_COL,
            config.DATE_COL,
        ]

        daily = (
            self._daily_mean(dated)
            .merge(self._rolling_8h_max(dated), on=keys, how="outer")
            .merge(self._valid_hour_counts(dated), on=keys, how="outer")
        )

        ordered_cols = (
            keys
            + config.POLLUTANT_COLS
            + [f"{p}_n" for p in self.cpcb_pollutants]
        )
        daily = (
            daily[ordered_cols]
            .sort_values([config.CITY_COL, config.STATION_COL, config.DATE_COL])
            .reset_index(drop=True)
        )

        self._validate(daily)
        return daily

    def _add_date_column(self, hourly: pd.DataFrame) -> pd.DataFrame:
        """Add a calendar-day column derived from the hourly timestamp.

        Args:
            hourly: Hourly frame containing the datetime column.

        Returns:
            A copy of the frame with a ``date`` column set to midnight of each
            row's day, keeping the datetime dtype for later sorting.
        """
        out = hourly.copy()
        out[config.DATE_COL] = out[config.DATETIME_COL].dt.normalize()
        return out

    def _daily_mean(self, hourly: pd.DataFrame) -> pd.DataFrame:
        """Average each 24-hour pollutant to one value per station-day.

        Args:
            hourly: Hourly frame that already has a ``date`` column.

        Returns:
            One row per station-day with the daily mean of every mean-pollutant
            and the identity columns ``[city, state, station, date]``.
        """
        keys = [
            config.CITY_COL,
            config.STATE_COL,
            config.STATION_COL,
            config.DATE_COL,
        ]
        return (
            hourly
            .groupby(keys, as_index=False)[self.mean_pollutants]
            .mean()
        )

    def _rolling_8h_max(self, hourly: pd.DataFrame) -> pd.DataFrame:
        """Take the worst 8-hour average of CO and O3 per station-day.

        For each station, a rolling 8-hour mean is computed over the hourly
        readings in chronological order, then the maximum value within each
        calendar day is kept, following the CPCB rule for CO and ozone.

        Args:
            hourly: Hourly frame, sorted by ``[city, station, datetime]`` and
                already carrying a ``date`` column.

        Returns:
            One row per station-day with the worst 8-hour average of each
            8-hour pollutant and the identity columns
            ``[city, state, station, date]``.
        """
        keys = [
            config.CITY_COL,
            config.STATE_COL,
            config.STATION_COL,
            config.DATE_COL,
        ]

        # Rolling 8-hour mean within each station, in chronological order.
        rolled = (
            hourly
            .groupby(config.STATION_COL)[self.max8h_pollutants]
            .rolling(
                window=config.ROLLING_WINDOW_HOURS,
                min_periods=config.MIN_HOURS_8H_WINDOW,
            )
            .mean()
            .reset_index(level=0, drop=True)
        )

        # Re-attach identity and date, then keep the day's worst window.
        rolled[keys] = hourly[keys]
        return (
            rolled
            .groupby(keys, as_index=False)[self.max8h_pollutants]
            .max()
        )

    def _valid_hour_counts(self, hourly: pd.DataFrame) -> pd.DataFrame:
        """Count the valid hourly readings behind each CPCB pollutant per day.

        Args:
            hourly: Hourly frame that already has a ``date`` column.

        Returns:
            One row per station-day with a ``<pollutant>_n`` column for each
            CPCB pollutant, giving the number of non-missing hourly readings.
        """
        keys = [
            config.CITY_COL,
            config.STATE_COL,
            config.STATION_COL,
            config.DATE_COL,
        ]
        counts = (
            hourly
            .groupby(keys, as_index=False)[self.cpcb_pollutants]
            .count()
        )
        rename_map = {p: f"{p}_n" for p in self.cpcb_pollutants}
        return counts.rename(columns=rename_map)

    def _validate(self, daily: pd.DataFrame) -> None:
        """Check the daily frame meets the guarantees later stages rely on.

        Raises:
            AssertionError: If the frame is empty, has repeated station-days,
                lacks a real date column, or has rows with no city.
        """
        assert len(daily) > 0, "Daily frame has no rows."

        repeated_days = daily.duplicated(
            subset=[config.STATION_COL, config.DATE_COL]
        ).sum()
        assert repeated_days == 0, f"Found {repeated_days} repeated station-days."

        assert daily[config.DATE_COL].dtype == "datetime64[ns]", (
            "The date column is not a real date type."
        )

        assert daily[config.CITY_COL].notna().all(), "Some rows have no city."


