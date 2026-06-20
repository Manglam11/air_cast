"""Project-wide configuration: paths, constants, AQI bands.

Single source of truth. Never hard-code these values elsewhere.
"""
from pathlib import Path

# ----- Paths (resolved from this file's location, so they work
# whether code runs from the project root or a notebook) -----
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

# ----- Raw data layout -----
STATIONS_DIR: Path = RAW_DATA_DIR / "stations"
METADATA_PATH: Path = RAW_DATA_DIR / "stations_info.csv"

# ----- Raw column names (exactly as they appear in station files) -----
RAW_DATE_FROM_COL: str = "From Date"
RAW_DATE_TO_COL: str = "To Date"


# ----- Reproducibility -----
RANDOM_STATE: int = 42

# Map messy raw column names -> clean snake_case names we'll use everywhere.
# (Units in the raw names are documented here in config; code uses the clean side.)
RAW_TO_CLEAN_COLS: dict[str, str] = {
    "From Date":        "datetime",
    "PM2.5 (ug/m3)":    "pm25",
    "PM10 (ug/m3)":     "pm10",
    "NO (ug/m3)":       "no",
    "NO2 (ug/m3)":      "no2",
    "NOx (ppb)":        "nox",
    "NH3 (ug/m3)":      "nh3",
    "SO2 (ug/m3)":      "so2",
    "CO (mg/m3)":       "co",
    "Ozone (ug/m3)":    "o3",
    "Benzene (ug/m3)":  "benzene",
    "Toluene (ug/m3)":  "toluene",
    "Xylene (ug/m3)":   "xylene",
}

POLLUTANT_COLS: list[str] = [
    "pm25", "pm10", "no", "no2", "nox", "nh3",
    "so2", "co", "o3", "benzene", "toluene", "xylene",
]

# The 7 CPCB pollutants we can use to compute AQI (Pb is absent from this data).
CPCB_AQI_POLLUTANTS: list[str] = ["pm25", "pm10", "so2", "no2", "nh3", "co", "o3"]

# ----- CPCB AQI bands: (category_name, inclusive_upper_bound) -----
AQI_BANDS: list[tuple[str, int]] = [
    ("Good",         50),
    ("Satisfactory", 100),
    ("Moderate",     200),
    ("Poor",         300),
    ("Very Poor",    400),
    ("Severe",       500),
]
AQI_CATEGORIES: list[str] = [name for name, _ in AQI_BANDS]

# Columns we CREATE later (not present in raw data — computed by AQICalculator).
CITY_COL: str = "city"
DATE_COL: str = "date"
AQI_COL: str = "aqi"
AQI_BUCKET_COL: str = "aqi_bucket"

# ----- Clean column names we create / standardise on -----
DATETIME_COL: str = "datetime"   # hourly timestamp (from "From Date")
STATION_COL: str = "station"     # station code, e.g. "AP001"
STATE_COL: str = "state"

# ----- Metadata (stations_info.csv) column names -----
META_FILENAME_COL: str = "file_name"
META_CITY_COL: str = "city"
META_STATE_COL: str = "state"

# ----- Daily aggregation rules (hourly -> daily, CPCB-aware) -----
# Pollutants summarised by the day's WORST rolling 8-hour average (CPCB rule).
AQI_MAX8H_POLLUTANTS: list[str] = ["co", "o3"]

# Everything else uses the day's 24-hour MEAN. Derived from POLLUTANT_COLS so
# there is one source of truth (same pattern as AQI_CATEGORIES from AQI_BANDS).
AQI_MEAN_POLLUTANTS: list[str] = [
    p for p in POLLUTANT_COLS if p not in AQI_MAX8H_POLLUTANTS
]

ROLLING_WINDOW_HOURS: int = 8   # window size for CO / O3
MIN_VALID_HOURS: int = 16       # a 24-hr pollutant needs >=16 valid hours to trust
MIN_HOURS_8H_WINDOW: int = 6   # an 8-hour average needs >= 6 valid hours