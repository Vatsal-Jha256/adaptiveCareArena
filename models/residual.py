"""Persistence-offset residual models: `y_hat = last_known_ema + f(x)`.

Why this shape, specifically. Every from-scratch regressor in the ladder
(ridge, GBM) has to spend its capacity re-learning the single dominant
fact about this data -- that tomorrow's EMA score is close to today's --
before it can model anything else, and it does that from noisy features
while competing against a baseline that gets that fact for free. An
independent review of Stage 1 named this as the single best next
experiment, and the paired result that followed the leakage fix supports
it: persistence is not beaten by any learned model, and GBM is only
statistically indistinguishable from it.

A residual model removes that handicap by construction:

  * It nests persistence exactly. If the learned correction predicts 0
    everywhere, the model IS persistence. So it cannot do structurally
    worse for the reason the from-scratch models did -- any loss is a
    real failure to find signal, not a failure to rediscover carry-forward.
  * It changes the question from "can a model beat persistence?" (answer
    so far: no) to "is there any learnable structure in the residual?",
    which is the scientifically interesting version and is directly
    falsifiable.
  * A near-zero learned correction is itself a clean, reportable result:
    it says the residual is noise at this sample size, which is a much
    stronger statement than "our regressor lost."

The fallback rule matters and is deliberate: when `last_known_ema_score`
is missing there is no offset to correct, so the model falls back to the
training-mean anchor plus the correction rather than silently treating a
missing anchor as 0.0."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

from models.statistical import _feature_matrix

ANCHOR_FEATURE = "last_known_ema_score"


class _PersistenceResidualModel:
    """Shared machinery; subclasses supply the residual regressor."""

    def __init__(self, **params):
        self.params = params
        self.feature_names: list[str] = []
        self._impute: Any = None
        self._model: Any = None
        self._anchor_fallback = 0.0

    def _make_regressor(self):
        raise NotImplementedError

    def _anchors(self, examples: list) -> np.ndarray:
        out = np.empty(len(examples))
        for i, e in enumerate(examples):
            v = e.context_features.get(ANCHOR_FEATURE, float("nan"))
            out[i] = v if v == v else self._anchor_fallback
        return out

    def fit(self, train: list) -> None:
        """Fit the residual head on ``target - anchor``.

        The anchor feature is excluded from the head's own inputs so the head
        cannot re-learn the identity mapping and undo the nesting.
        """
        if not train:
            raise ValueError(f"{type(self).__name__}.fit() called with no training examples")

        ys = np.array([e.target_y for e in train])
        self._anchor_fallback = float(np.mean(ys))

        # Union across all training examples -- see statistical.py for why
        # train[0]-only was a bug.
        names = sorted({k for e in train for k in e.context_features})
        # The anchor is excluded from the residual's own features: it is
        # already fully used as the offset, and leaving it in lets the
        # regressor re-derive the identity mapping and undo the nesting
        # this model exists to guarantee.
        self.feature_names = [n for n in names if n != ANCHOR_FEATURE]

        X_raw = _feature_matrix(train, self.feature_names)
        self._impute = np.nanmedian(X_raw, axis=0)
        self._impute = np.where(np.isnan(self._impute), 0.0, self._impute)
        X = np.where(np.isnan(X_raw), self._impute, X_raw)

        residual = ys - self._anchors(train)
        self._model = self._make_regressor()
        self._model.fit(X, residual)

    def predict(self, examples: list) -> np.ndarray:
        """Return the anchor plus the residual head's correction."""
        if self._model is None:
            raise RuntimeError(f"{type(self).__name__}.predict() called before fit()")
        X_raw = _feature_matrix(examples, self.feature_names)
        X = np.where(np.isnan(X_raw), self._impute, X_raw)
        return self._anchors(examples) + self._model.predict(X)

    def learned_correction_magnitude(self, examples: list) -> float:
        """Mean |correction| the residual head applies. Near zero means the
        model has collapsed onto persistence and found no residual signal
        -- report this alongside MAE, because a residual model that ties
        persistence is uninformative unless you know why: because it found
        nothing, or because it found something that cancels out."""
        X_raw = _feature_matrix(examples, self.feature_names)
        X = np.where(np.isnan(X_raw), self._impute, X_raw)
        return float(np.mean(np.abs(self._model.predict(X))))


class ResidualRidgeModel(_PersistenceResidualModel):
    """Persistence-anchored model whose residual head is ridge regression."""

    def __init__(self, alpha: float = 1.0):
        super().__init__(alpha=alpha)

    def _make_regressor(self):
        return Ridge(**self.params)


class ResidualGBMModel(_PersistenceResidualModel):
    """Persistence-anchored model whose residual head is gradient-boosted trees."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        random_state: int = 0,
    ):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
        )

    def _make_regressor(self):
        return GradientBoostingRegressor(**self.params)
