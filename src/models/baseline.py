"""PersistenceBaseline: the naive 'tomorrow = today' forecast, scored honestly.

The falsifiable bar for every model in the regression and classification
bake-offs. It ignores the feature matrix entirely: it carries each day's own AQI
(and category) forward as the prediction for the next day, then scores that
against the true next-day targets. A model that cannot beat this baseline adds no
value, because copying today's value forward is already strong on autocorrelated
air-quality data.
"""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

from src import config


class PersistenceBaseline:
    """Score the naive next-day forecast that copies today's value forward.

    For regression the prediction is each row's own ``aqi`` (today) and the truth
    is ``target_aqi`` (tomorrow). For classification the prediction is today's
    ``aqi_bucket`` and the truth is ``target_bucket``. Scoring is done on a
    caller-supplied frame, so the exact held-out test frame used to judge real
    models judges the baseline too — an apples-to-apples comparison.
    """

    def regression_scores(self, frame: pd.DataFrame) -> dict[str, float]:
        """Score the naive 'tomorrow's AQI = today's AQI' forecast.

        Args:
            frame: Frame carrying ``aqi`` (today) and ``target_aqi`` (tomorrow).

        Returns:
            Mapping with ``mae``, ``rmse`` (AQI points) and ``r2`` (unitless).
        """
        self._require_no_nan(frame, config.AQI_COL)
        y_true = frame[config.TARGET_AQI_COL]
        y_pred = frame[config.AQI_COL]
        return {
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": root_mean_squared_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
        }

    def classification_scores(self, frame: pd.DataFrame) -> dict[str, float]:
        """Score the naive 'tomorrow's category = today's category' forecast.

        Args:
            frame: Frame carrying ``aqi_bucket`` (today) and ``target_bucket``.

        Returns:
            Mapping with ``accuracy``, ``weighted_f1`` and ``macro_f1``.
        """
        self._require_no_nan(frame, config.AQI_BUCKET_COL)
        y_true = frame[config.TARGET_BUCKET_COL]
        y_pred = frame[config.AQI_BUCKET_COL]
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        }

    def confusion(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Build the confusion matrix in CPCB severity order.

        Rows are the true next-day category; columns are today's category used as
        the prediction. Ordering labels by severity (not alphabetically) keeps
        the diagonal readable and collects dangerous underestimates in the
        lower-left corner.

        Args:
            frame: Frame carrying ``aqi_bucket`` and ``target_bucket``.

        Returns:
            A labelled confusion-matrix frame, ``actual`` (rows) x ``predicted``.
        """
        y_true = frame[config.TARGET_BUCKET_COL]
        y_pred = frame[config.AQI_BUCKET_COL]
        matrix = confusion_matrix(y_true, y_pred, labels=config.AQI_CATEGORIES)
        cm = pd.DataFrame(
            matrix, index=config.AQI_CATEGORIES, columns=config.AQI_CATEGORIES
        )
        cm.index.name = "actual"
        cm.columns.name = "predicted"
        return cm

    def _require_no_nan(self, frame: pd.DataFrame, column: str) -> None:
        """Fail loudly if the persistence prediction column carries NaNs.

        Raises:
            AssertionError: If ``column`` has any missing values, which would
                silently corrupt the baseline score.
        """
        n_nan = frame[column].isna().sum()
        assert n_nan == 0, f"'{column}' has {n_nan} NaNs; baseline cannot score."