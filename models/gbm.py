"""GBM baseline via scikit-learn -- CPU-only."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from models.statistical import _feature_matrix


class GBMModel:
    """Gradient-boosted trees (scikit-learn) on the example's ``context_features``.

    Missing features are imputed with training-set medians. The seed is passed
    to the underlying estimator as ``random_state``.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        random_state: int = 0,
    ):
        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        self.feature_names: list[str] = []
        self._impute: Any = None
        self._model: Any = None

    def fit(self, train: list) -> None:
        """Fit on ``train``; raises ``ValueError`` if it is empty."""
        if not train:
            raise ValueError("GBMModel.fit() called with no training examples")
        # Union across all training examples -- see statistical.py's
        # RidgeRegressionModel.fit() for why train[0]-only was a bug.
        names = sorted({k for e in train for k in e.context_features})
        self.feature_names = names
        X_raw = _feature_matrix(train, names)
        self._impute = np.nanmedian(X_raw, axis=0)
        self._impute = np.where(np.isnan(self._impute), 0.0, self._impute)
        X = np.where(np.isnan(X_raw), self._impute, X_raw)
        y = np.array([e.target_y for e in train])

        self._model = GradientBoostingRegressor(**self.params)
        self._model.fit(X, y)

    def predict(self, examples: list) -> np.ndarray:
        """Predict ``target_y`` for each example; raises ``RuntimeError`` before ``fit``."""
        if self._model is None:
            raise RuntimeError("GBMModel.predict() called before fit()")
        X_raw = _feature_matrix(examples, self.feature_names)
        X = np.where(np.isnan(X_raw), self._impute, X_raw)
        return self._model.predict(X)
