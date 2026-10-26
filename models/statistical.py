"""Ridge regression from scratch (numpy closed-form)."""

from __future__ import annotations

from typing import Any

import numpy as np


def _feature_matrix(examples: list, feature_names: list[str]) -> np.ndarray:
    X = np.zeros((len(examples), len(feature_names)))
    for i, e in enumerate(examples):
        for j, name in enumerate(feature_names):
            X[i, j] = e.context_features.get(name, np.nan)
    return X


class RidgeRegressionModel:
    """Ridge regression solved in closed form with numpy.

    Features are median-imputed and standardized; the intercept is not
    penalized.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.feature_names: list[str] = []
        self._impute: Any = None
        self._mean: Any = None
        self._std: Any = None
        self._w: Any = None

    def fit(self, train: list) -> None:
        """Fit on ``train``; raises ``ValueError`` if it is empty."""
        if not train:
            raise ValueError("RidgeRegressionModel.fit() called with no training examples")
        # Union across all training examples, not just train[0] -- a
        # feature present for only some examples (e.g. CrossCheck's
        # hourly_* columns, only available for subjects with an
        # unambiguous hourly mapping) must not be silently dropped just
        # because the first example happens to lack it.
        names = sorted({k for e in train for k in e.context_features})
        self.feature_names = names
        X_raw = _feature_matrix(train, names)

        self._impute = np.nanmedian(X_raw, axis=0)
        self._impute = np.where(np.isnan(self._impute), 0.0, self._impute)
        X = np.where(np.isnan(X_raw), self._impute, X_raw)

        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0)
        self._std = np.where(self._std < 1e-8, 1.0, self._std)
        Xs = (X - self._mean) / self._std
        Xs = np.hstack([np.ones((Xs.shape[0], 1)), Xs])

        y = np.array([e.target_y for e in train])
        n_feat = Xs.shape[1]
        reg = self.alpha * np.eye(n_feat)
        reg[0, 0] = 0.0
        self._w = np.linalg.solve(Xs.T @ Xs + reg, Xs.T @ y)

    def predict(self, examples: list) -> np.ndarray:
        """Predict ``target_y`` for each example; raises ``RuntimeError`` before ``fit``."""
        if self._w is None:
            raise RuntimeError("RidgeRegressionModel.predict() called before fit()")
        X_raw = _feature_matrix(examples, self.feature_names)
        X = np.where(np.isnan(X_raw), self._impute, X_raw)
        Xs = (X - self._mean) / self._std
        Xs = np.hstack([np.ones((Xs.shape[0], 1)), Xs])
        return Xs @ self._w
