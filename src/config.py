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

# CPCB AQI sub-index breakpoints.
# Each band is (conc_low, conc_high, aqi_low, aqi_high).
# Units: ug/m3 for all pollutants except co (mg/m3).
# Top band upper edges are conventional (CPCB leaves them open-ended);
# AQICalculator clamps any resulting sub-index at 500.
AQI_BREAKPOINTS = {
    "pm25": [
        (0, 30, 0, 50),
        (31, 60, 51, 100),
        (61, 90, 101, 200),
        (91, 120, 201, 300),
        (121, 250, 301, 400),
        (251, 380, 401, 500),
    ],
    "pm10": [
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 250, 101, 200),
        (251, 350, 201, 300),
        (351, 430, 301, 400),
        (431, 510, 401, 500),
    ],
    "no2": [
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 180, 101, 200),
        (181, 280, 201, 300),
        (281, 400, 301, 400),
        (401, 520, 401, 500),
    ],
    "o3": [
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 168, 101, 200),
        (169, 208, 201, 300),
        (209, 748, 301, 400),
        (749, 940, 401, 500),
    ],
    "so2": [
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 380, 101, 200),
        (381, 800, 201, 300),
        (801, 1600, 301, 400),
        (1601, 2620, 401, 500),
    ],
    "nh3": [
        (0, 200, 0, 50),
        (201, 400, 51, 100),
        (401, 800, 101, 200),
        (801, 1200, 201, 300),
        (1201, 1800, 301, 400),
        (1801, 2400, 401, 500),
    ],
    "co": [
        (0, 1.0, 0, 50),
        (1.1, 2.0, 51, 100),
        (2.1, 10, 101, 200),
        (10.1, 17, 201, 300),
        (17.1, 34, 301, 400),
        (34.1, 50, 401, 500),
    ],
}

# Minimum pollutants required for a valid CPCB AQI on a given day.
AQI_MIN_POLLUTANTS = 3
# At least one of these must be present for a valid AQI.
AQI_REQUIRED_ANY = ("pm25", "pm10")

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DAILY_AQI_PATH = PROCESSED_DIR / "daily_aqi.parquet"

# ----- Cleaning (Session 3) -----
VOC_COLS: list[str] = ["benzene", "toluene", "xylene"]   # sparse — dropped before modeling

# The 9 pollutants kept for modeling, derived so there's one source of truth
# (same pattern as AQI_CATEGORIES from AQI_BANDS).
MODEL_POLLUTANT_COLS: list[str] = [p for p in POLLUTANT_COLS if p not in VOC_COLS]

MODELING_START_YEAR: int = 2018   # coverage stabilises here (EDA bucket 4)
MIN_CITY_DAYS: int = 730          # two seasonal cycles of LABELLED days
FILL_LIMIT: int = 3               # bridge gaps up to 3 days; longer stays NaN

CITY_ID_COL: str = "city_id"      # canonical identity = "city, state"
IDENTITY_COLS: list[str] = [CITY_ID_COL, CITY_COL, STATE_COL]

CLEAN_CITY_DAY_PATH = PROCESSED_DIR / "clean_city_day.parquet"

# ----- Feature engineering (Session 4) -----
LAG_DAYS: list[int] = [1, 2, 3, 7]
ROLL_WINDOWS: list[int] = [3, 7]
ROLL_STATS: tuple[str, ...] = ("mean", "std")

# India-appropriate season buckets, mapped from month.
MONTH_TO_SEASON: dict[int, str] = {
    12: "Winter",       1: "Winter",        2: "Winter",
     3: "Summer",       4: "Summer",        5: "Summer",
     6: "Monsoon",      7: "Monsoon",       8: "Monsoon",      9: "Monsoon",
    10: "Post-Monsoon", 11: "Post-Monsoon",
}

# Created feature / target column names.
SEASON_COL: str        = "season"
MONTH_COL: str         = "month"
DAY_OF_YEAR_COL: str   = "day_of_year"
DAY_OF_WEEK_COL: str   = "day_of_week"
IS_WEEKEND_COL: str    = "is_weekend"
TARGET_AQI_COL: str    = "target_aqi"      # day T+1 AQI value
TARGET_BUCKET_COL: str = "target_bucket"   # day T+1 AQI category

# Derived feature-name lists — single source of truth for the model sessions.
LAG_COLS: list[str] = [f"aqi_lag_{k}" for k in LAG_DAYS]
ROLL_COLS: list[str] = [
    f"aqi_roll_{stat}_{w}" for w in ROLL_WINDOWS for stat in ROLL_STATS
]
CALENDAR_NUMERIC_COLS: list[str] = [
    MONTH_COL, DAY_OF_YEAR_COL, DAY_OF_WEEK_COL, IS_WEEKEND_COL,
]

# The model's X: today's pollutant levels + lags + rollings + calendar.
FEATURE_COLS: list[str] = (
    MODEL_POLLUTANT_COLS + LAG_COLS + ROLL_COLS + CALENDAR_NUMERIC_COLS
)

# Structural columns: NaNs here are unfixable -> their rows are dropped.
# Pollutant NaNs are deliberately excluded (imputed later in the Pipeline).
STRUCTURAL_COLS: list[str] = (
    LAG_COLS + ROLL_COLS + [TARGET_AQI_COL, TARGET_BUCKET_COL]
)

# Metadata: kept for the app + chronological splitting, never fed as features.
META_COLS: list[str] = [
    CITY_ID_COL, CITY_COL, STATE_COL, DATE_COL,
    AQI_COL, AQI_BUCKET_COL, SEASON_COL,
]

FEATURES_PATH = PROCESSED_DIR / "features.parquet"

# ----- Modeling: split, CV, preprocessing (Session 5) -----
TEST_CUTOFF: str = "2022-04-01"    # chronological hold-out wall (ISO date)
N_CV_SPLITS: int = 5               # TimeSeriesSplit expanding-window folds
IMPUTER_STRATEGY: str = "median"   # robust to pollutant spike days

# The model's feature contract: numeric features + the one categorical (season).
# season lives in META_COLS but is pulled in HERE as the sole categorical feature.
MODEL_NUMERIC_FEATURES: list[str] = FEATURE_COLS
MODEL_CATEGORICAL_FEATURES: list[str] = [SEASON_COL]
MODEL_FEATURES: list[str] = MODEL_NUMERIC_FEATURES + MODEL_CATEGORICAL_FEATURES

# ----- Regression bake-off scoring (Session 6) -----
# sklearn reports error metrics negated (higher = better); the trainer's
# run_bakeoff flips the neg_ ones back so the leaderboard reads in natural units.
REGRESSION_SCORING: dict[str, str] = {
    "mae":  "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2":   "r2",
}