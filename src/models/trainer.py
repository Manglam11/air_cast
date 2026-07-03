"""ModelTrainer: the leakage-safe preprocessing and validation machinery.

Provides the shared scaffold the regression and classification bake-offs
(Sessions 6-7) plug every model into: a chronological train/test split, a
ColumnTransformer that median-imputes and scales the numeric features and
one-hot-encodes season, a TimeSeriesSplit cross-validator, and a factory that
wraps any estimator in a Pipeline so preprocessing is fit on training folds only.
The model roster and bake-off loop arrive in Session 6; this module ships the
proven, model-agnostic parts.
"""
from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


class ModelTrainer:
    """Build the leakage-safe split, preprocessor, and CV for the bake-offs.

    Preprocessing is assembled fresh per model and only ever fit inside a
    Pipeline on training data, so validation and test statistics never leak into
    training. The chronological split and TimeSeriesSplit enforce time-aware
    validation: models always train on the past and are judged on the future.

    Attributes:
        numeric_features: Numeric feature columns (imputed and scaled).
        categorical_features: Categorical feature columns (one-hot encoded).
        n_splits: Number of TimeSeriesSplit folds.
    """

    def __init__(self) -> None:
        self.numeric_features: list[str] = config.MODEL_NUMERIC_FEATURES
        self.categorical_features: list[str] = config.MODEL_CATEGORICAL_FEATURES
        self.n_splits: int = config.N_CV_SPLITS

    def split(self, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the feature frame chronologically at the test cutoff.

        Everything strictly before ``config.TEST_CUTOFF`` is training data;
        everything on or after it is the held-out test set. The training frame is
        returned sorted by date so that TimeSeriesSplit — which splits by row
        position — splits by time, not by city.

        Args:
            features: The full model-ready feature frame.

        Returns:
            A ``(train, test)`` tuple with no shared dates; ``train`` is sorted
            chronologically.
        """
        cutoff = pd.Timestamp(config.TEST_CUTOFF)
        is_train = features[config.DATE_COL] < cutoff

        train = (
            features.loc[is_train]
            .sort_values(config.DATE_COL)
            .reset_index(drop=True)
        )
        test = features.loc[~is_train].reset_index(drop=True)

        assert train[config.DATE_COL].max() < cutoff, "train leaks past cutoff"
        assert test[config.DATE_COL].min() >= cutoff, "test precedes cutoff"
        return train, test

    def build_preprocessor(self) -> ColumnTransformer:
        """Build the ColumnTransformer that prepares features for modeling.

        Numeric features are median-imputed (robust to pollutant spike days) then
        standardised; season is one-hot encoded. Assembled fresh so it can be fit
        independently inside each model's Pipeline on training folds only.

        Returns:
            The unfitted ColumnTransformer.
        """
        numeric_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy=config.IMPUTER_STRATEGY)),
                ("scale", StandardScaler()),
            ]
        )
        categorical_pipe = Pipeline(
            [("encode", OneHotEncoder(handle_unknown="ignore"))]
        )
        return ColumnTransformer(
            [
                ("num", numeric_pipe, self.numeric_features),
                ("cat", categorical_pipe, self.categorical_features),
            ]
        )

    def get_cv(self) -> TimeSeriesSplit:
        """Return the time-aware cross-validator for model selection.

        Returns:
            A TimeSeriesSplit with ``config.N_CV_SPLITS`` expanding-window folds;
            correct only on a date-sorted training frame (see ``split``).
        """
        return TimeSeriesSplit(n_splits=self.n_splits)

    def make_pipeline(self, model: BaseEstimator) -> Pipeline:
        """Wrap an estimator behind the shared preprocessor.

        Every model in the bake-offs is wrapped this way, so preprocessing is
        re-fit on each training fold and never sees validation or test data.

        Args:
            model: Any scikit-learn compatible estimator.

        Returns:
            A Pipeline of ``preprocessor -> model``.
        """
        return Pipeline([("prep", self.build_preprocessor()), ("model", model)])

    def run_bakeoff(self) -> None:
        """Run the model roster under cross-validation (Session 6).

        Placeholder: the regression and classification rosters and the CV-scoring
        loop are crystallized here once Session 6 proves them in the notebook.
        Kept as an explicit stub so the skeleton's intent is visible in the code.

        Raises:
            NotImplementedError: Always, until Session 6.
        """
        raise NotImplementedError("Model roster is added in Session 6.")