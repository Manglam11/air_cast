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

# Confirm exact filename after Kaggle download; common name is "city_day.csv".
RAW_CSV_PATH: Path = RAW_DATA_DIR / "city_day.csv"

# ----- Reproducibility -----
RANDOM_STATE: int = 42

# ----- Domain columns (match Kaggle CPCB schema) -----
DATE_COL: str = "Date"
CITY_COL: str = "City"
AQI_COL: str = "AQI"
AQI_BUCKET_COL: str = "AQI_Bucket"

POLLUTANT_COLS: list[str] = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
    "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene",
]

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