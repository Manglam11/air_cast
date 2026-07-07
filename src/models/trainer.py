"""ModelTrainer: the leakage-safe preprocessing, validation, and bake-off machinery.

Provides the shared scaffold the regression and classification bake-offs plug
every model into: a chronological train/test split, a ColumnTransformer that
median-imputes and scales the numeric features and one-hot-encodes season, a
TimeSeriesSplit cross-validator, a factory that wraps any estimator in a Pipeline
so preprocessing is fit on training folds only, and the cross-validated bake-off
loop that ranks a roster of models. Both the regression and classification
rosters live here, alongside the custom Severe-recall scorer the classification
bake-off ranks on.
"""
from __future__ import annotations

import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from src import config


def severe_recall_score(y_true, y_pred) -> float:
    """Recall for the Severe class only: of truly-Severe days, the fraction caught.

    sklearn ships no built-in string for single-class recall, so this is the
    metric the classification bake-off cares about most, made explicit. A missed
    Severe day is the dangerous error (a person told 'all clear' on a hazardous
    day), so this isolates how well a model catches that rare, critical class.

    Args:
        y_true: Integer-encoded true next-day categories (Good=0 .. Severe=5).
        y_pred: Integer-encoded predicted categories.

    Returns:
        Recall for the single Severe class (0.0-1.0). ``zero_division=0`` makes a
        fold with no Severe days score 0.0 rather than raise.
    """
    severe_id = config.AQI_CATEGORIES.index("Severe")
    return recall_score(
        y_true, y_pred, labels=[severe_id], average="macro", zero_division=0
    )


# Scorer wrapper so cross_validate can call the metric per fold (greater = better).
severe_recall_scorer = make_scorer(severe_recall_score)


class ModelTrainer:
    """Build the leakage-safe split, preprocessor, CV, and bake-off for the roster.

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

    def regression_roster(self) -> list[tuple[str, BaseEstimator]]:
        """Return the 9-model regression roster for the bake-off.

        Nine estimators across four families — linear, distance, tree, and
        boosting — so the bake-off compares fundamentally different ways of
        reading the features. Only the stochastic models (trees and boosting)
        take a random_state; the linear models and KNN are deterministic. SVR is
        deliberately omitted: its RBF kernel scales poorly on this many rows.

        Returns:
            ``(name, estimator)`` pairs, each ready to wrap in the pipeline.
        """
        return [
            ("LinearRegression", LinearRegression()),
            ("Ridge",            Ridge()),
            ("Lasso",            Lasso()),
            ("ElasticNet",       ElasticNet()),
            ("KNN",              KNeighborsRegressor(n_jobs=-1)),
            ("DecisionTree",     DecisionTreeRegressor(random_state=config.RANDOM_STATE)),
            ("RandomForest",     RandomForestRegressor(random_state=config.RANDOM_STATE, n_jobs=-1)),
            ("XGBoost",          XGBRegressor(random_state=config.RANDOM_STATE, n_jobs=-1)),
            ("LightGBM",         LGBMRegressor(random_state=config.RANDOM_STATE, n_jobs=-1, verbose=-1)),
        ]

    def classification_roster(
        self, balanced: bool = False
    ) -> list[tuple[str, BaseEstimator]]:
        """Return the 7-model classification roster for the bake-off.

        Seven estimators across four families — linear, probabilistic, tree, and
        boosting — so the bake-off compares fundamentally different ways of
        drawing class boundaries. The SVM family is represented by LinearSVC, not
        an RBF SVC: a kernel SVC is O(n^2)+ and will not finish on this many rows
        across TimeSeriesSplit folds (the same reason SVR was cut from
        regression).

        When ``balanced`` is True, ``class_weight="balanced"`` is applied to every
        model that exposes it, up-weighting rare classes (Severe is ~1.5% of days)
        so a missed Severe day is penalised heavily. GaussianNB has no such knob
        (it is generative), and XGBoost balances via per-row ``sample_weight``
        rather than ``class_weight``; both therefore ride unbalanced regardless of
        the flag, and are documented as such.

        Args:
            balanced: If True, apply ``class_weight="balanced"`` where supported.

        Returns:
            ``(name, estimator)`` pairs, each ready to wrap in the pipeline.
        """
        weight = "balanced" if balanced else None
        return [
            ("LogisticRegression", LogisticRegression(max_iter=1000, class_weight=weight)),
            ("GaussianNB",         GaussianNB()),  # generative — no class_weight
            ("LinearSVM",          LinearSVC(dual="auto", max_iter=2000, class_weight=weight)),
            ("DecisionTree",       DecisionTreeClassifier(random_state=config.RANDOM_STATE, class_weight=weight)),
            ("RandomForest",       RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=-1, class_weight=weight)),
            ("XGBoost",            XGBClassifier(random_state=config.RANDOM_STATE, n_jobs=-1)),  # balances via sample_weight
            ("LightGBM",           LGBMClassifier(random_state=config.RANDOM_STATE, n_jobs=-1, verbose=-1, class_weight=weight)),
        ]

    def run_bakeoff(
        self,
        roster: list[tuple[str, BaseEstimator]],
        X: pd.DataFrame,
        y: pd.Series,
        scoring: dict[str, object],
    ) -> pd.DataFrame:
        """Cross-validate every model in the roster and rank them.

        Each model is wrapped in the leakage-safe pipeline and scored with the
        trainer's TimeSeriesSplit, so preprocessing is re-fit per fold and the
        comparison is time-aware. sklearn reports error metrics negated (higher
        is better); those are flipped back so the leaderboard reads in natural
        units. The leaderboard is sorted by the first metric — ascending when it
        is a negated error metric, descending otherwise.

        The ``scoring`` values may be sklearn scorer strings (e.g. ``"f1_macro"``,
        ``"neg_mean_absolute_error"``) or callables (e.g. ``severe_recall_scorer``);
        only string scorers are checked for the ``neg_`` error convention.

        Args:
            roster: ``(name, estimator)`` pairs to compare.
            X: Feature frame (raw columns; the pipeline preprocesses them).
            y: The target aligned to ``X``.
            scoring: Mapping of ``metric_name -> sklearn scorer string or callable``.

        Returns:
            One row per model: a ``model`` column plus a ``cv_<metric>`` column
            for each metric, sorted best first.
        """
        cv = self.get_cv()
        rows = []
        for name, model in roster:
            pipe = self.make_pipeline(model)
            result = cross_validate(
                pipe, X, y,
                cv=cv,
                scoring=scoring,
                n_jobs=1,             # models parallelize internally; avoid nesting
                error_score="raise",  # fail loudly, never silently score NaN
            )
            row = {"model": name}
            for metric, scorer in scoring.items():
                mean_score = result[f"test_{metric}"].mean()
                # Only string scorers follow sklearn's negated-error convention;
                # callables (e.g. the Severe-recall scorer) are already natural.
                if isinstance(scorer, str) and scorer.startswith("neg_"):
                    mean_score = -mean_score
                row[f"cv_{metric}"] = mean_score
            rows.append(row)

        first_metric = next(iter(scoring))
        first_scorer = scoring[first_metric]
        best_is_low = isinstance(first_scorer, str) and first_scorer.startswith("neg_")
        return (
            pd.DataFrame(rows)
            .sort_values(f"cv_{first_metric}", ascending=best_is_low)
            .reset_index(drop=True)
        )