"""DataLoader: read raw CPCB station files into one tidy hourly DataFrame.

Single responsibility: load, merge, and sort. No daily aggregation, no AQI
computation, and no value-level cleaning — those belong to later pipeline
stages, each in its own class.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config


class DataLoader:
    """Load per-station CPCB CSVs and stack them into one hourly frame.

    The raw dataset stores one CSV per monitoring station (e.g. ``AP001.csv``)
    plus a metadata file (``stations_info.csv``) mapping each station code to
    its city and state. This class reads the station files, attaches
    city/state, standardises column names, parses timestamps, and returns a
    single tidy DataFrame sorted by ``[city, station, datetime]``.

    Attributes:
        stations_dir: Folder containing the per-station CSV files.
        metadata_path: Path to the station metadata CSV.
        cities: Optional list of city names to load. If None, load every city.
    """

    def __init__(
        self,
        stations_dir: Path = config.STATIONS_DIR,
        metadata_path: Path = config.METADATA_PATH,
        cities: list[str] | None = None,
    ) -> None:
        self.stations_dir = Path(stations_dir)
        self.metadata_path = Path(metadata_path)
        self.cities = cities

    def load(self) -> pd.DataFrame:
        """Read and merge all (or selected) station files into one frame.

        Returns:
            A tidy hourly DataFrame with columns
            ``[datetime, station, city, state, <pollutants>]``, sorted by
            ``[city, station, datetime]`` with a fresh integer index.

        Raises:
            FileNotFoundError: If the metadata file is missing.
            ValueError: If no station files were loaded (e.g. a bad city filter).
        """
        station_meta = self._load_metadata()

        frames: list[pd.DataFrame] = []
        for station_code, (city, state) in station_meta.items():
            file_path = self.stations_dir / f"{station_code}.csv"
            if not file_path.exists():
                # Metadata lists a station with no CSV file — skip, don't crash.
                continue
            frames.append(
                self._load_one_station(file_path, station_code, city, state)
            )

        if not frames:
            raise ValueError(
                "No station files loaded. Check the `cities` filter and "
                f"that CSVs exist in {self.stations_dir}."
            )

        combined = pd.concat(frames, ignore_index=True)

        # Enforce a tidy, predictable column order.
        ordered_cols = [
            config.DATETIME_COL,
            config.STATION_COL,
            config.CITY_COL,
            config.STATE_COL,
            *config.POLLUTANT_COLS,
        ]
        combined = combined[ordered_cols]

        # Chronological order *within* each city/station — never a random shuffle.
        combined = combined.sort_values(
            by=[config.CITY_COL, config.STATION_COL, config.DATETIME_COL]
        ).reset_index(drop=True)

        # A station should never have two readings for the same hour.
        combined = combined.drop_duplicates(
            subset=[config.STATION_COL, config.DATETIME_COL], keep="first"
        ).reset_index(drop=True)

        self._validate(combined)
        return combined

    def _load_metadata(self) -> dict[str, tuple[str, str]]:
        """Build a ``{station_code: (city, state)}`` lookup from the metadata.

        Applies the optional ``cities`` filter so we only load relevant files.

        Returns:
            Mapping from station code (e.g. ``"AP001"``) to ``(city, state)``.

        Raises:
            FileNotFoundError: If the metadata file does not exist.
        """
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        meta = pd.read_csv(self.metadata_path)

        if self.cities is not None:
            meta = meta[meta[config.META_CITY_COL].isin(self.cities)]

        return {
            row[config.META_FILENAME_COL]: (
                row[config.META_CITY_COL],
                row[config.META_STATE_COL],
            )
            for _, row in meta.iterrows()
        }

    def _load_one_station(
        self, file_path: Path, station_code: str, city: str, state: str
    ) -> pd.DataFrame:
        """Read and standardise a single station's CSV.

        Args:
            file_path: Path to the station CSV.
            station_code: The station's code, e.g. ``"AP001"``.
            city: City this station belongs to.
            state: State this station belongs to.

        Returns:
            A clean per-station DataFrame: parsed ``datetime``, the kept
            pollutant columns (missing ones filled with NaN), and identity
            columns ``station``, ``city``, ``state``.
        """
        raw = pd.read_csv(file_path)
        raw = raw.rename(columns=config.RAW_TO_CLEAN_COLS)

        # Keep only datetime + pollutants. `reindex` makes the column set
        # uniform across stations: any pollutant a station didn't measure
        # becomes a NaN column instead of a KeyError. This also drops the
        # "To Date" and weather columns we don't use in v1.
        keep_cols = [config.DATETIME_COL, *config.POLLUTANT_COLS]
        station = raw.reindex(columns=keep_cols)

        # Parse timestamps; unparseable values become NaT (not a crash).
        station[config.DATETIME_COL] = pd.to_datetime(
            station[config.DATETIME_COL], errors="coerce"
        )
        # A row with no usable timestamp can't be placed in time — drop it.
        station = station.dropna(subset=[config.DATETIME_COL])

        # Attach identity.
        station[config.STATION_COL] = station_code
        station[config.CITY_COL] = city
        station[config.STATE_COL] = state
        return station

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        """Fail loudly if the merged frame breaks its guarantees.

        Args:
            df: The combined DataFrame produced by ``load``.

        Raises:
            AssertionError: If any structural guarantee is violated.
        """
        assert not df.empty, "Loaded DataFrame is empty."
        assert pd.api.types.is_datetime64_any_dtype(df[config.DATETIME_COL]), (
            "datetime column is not a real datetime dtype."
        )
        assert df[config.CITY_COL].notna().all(), "Some rows have no city."
        dup_count = df.duplicated(
            subset=[config.STATION_COL, config.DATETIME_COL]
        ).sum()
        assert dup_count == 0, f"Found {dup_count} duplicate station-hour rows."