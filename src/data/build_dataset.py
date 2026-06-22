"""Build the labeled daily AQI dataset from raw station files.

Runs the full pipeline (load -> aggregate -> compute AQI) across every
station and writes the result to ``config.DAILY_AQI_PATH`` as Parquet.
Run once whenever the raw data changes:

    python -m src.data.build_dataset
"""

from src import config
from src.data.loader import DataLoader
from src.data.aggregator import DailyAggregator
from src.data.aqi import AQICalculator


def build_daily_aqi() -> None:
    """Run the full raw-to-labeled pipeline and persist the result."""
    hourly = DataLoader().load()
    print(f"Loaded hourly frame: {hourly.shape}")

    daily = DailyAggregator().aggregate(hourly)
    print(f"Aggregated to daily:  {daily.shape}")

    labeled = AQICalculator().compute(daily)
    print(f"Computed AQI:         {labeled.shape}")

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    labeled.to_parquet(config.DAILY_AQI_PATH, index=False)
    print(f"Saved -> {config.DAILY_AQI_PATH}")

    valid = labeled[config.AQI_COL].notna().sum()
    print(f"Valid-AQI rows: {valid} / {len(labeled)}")


if __name__ == "__main__":
    build_daily_aqi()