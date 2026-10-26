"""One model that predicts ema_score from any subset of asked items plus
the free passive context.

With 10 items there are 1024 possible question subsets, so
training a separate model per subset is not an option. Instead one model
is trained with random item-dropout augmentation, so it has
seen partial questionnaires during training rather than being asked to
extrapolate to them at test time.

An unasked item is NaN, not zero. Zero is a real, meaningful response on
this 0-3 scale ("not at all"), so imputing unasked items as 0 would tell
the model the patient denied the symptom -- the single most damaging
thing you could do in a benchmark about not asking. HistGradientBoosting
routes NaN natively at split time, which is exactly the "never asked"
semantics required."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


class MaskAwareEMAModel:
    """One regressor that predicts ``ema_score`` from any subset of asked items.

    With ``k`` items there are ``2**k`` possible questionnaires, so a model per
    subset is impractical. Training adds randomly masked copies of every
    event (item dropout), so the model has seen partial questionnaires. Items
    not asked are NaN, which the booster handles natively.

    Args:
        items: names of the instrument's items.
        n_dropout_augmentations: masked copies added per training event.
        random_state: seeds both the masks and the booster.
        max_train_rows: subsample the augmented set above this size.
    """

    DTYPE = np.float32

    def __init__(
        self,
        items,
        n_dropout_augmentations: int = 4,
        random_state: int = 0,
        max_train_rows: int = 250_000,
    ):
        # `items` is passed in, not imported: CrossCheck has 10 signed EMA
        # items and CES has 4 PHQ-4 items, and the model should not know
        # which instrument it is looking at.
        self.items = tuple(items)
        self.n_dropout_augmentations = n_dropout_augmentations
        self.max_train_rows = max_train_rows
        self.rng = np.random.default_rng(random_state)
        self.context_names: list[str] = []
        self.model = HistGradientBoostingRegressor(
            max_iter=200, max_depth=6, random_state=random_state
        )

    def _columns(self) -> list[str]:
        return list(self.items) + self.context_names

    def _to_matrix(
        self, events: list, asked: list[set[str]] | None = None, out: np.ndarray | None = None
    ) -> np.ndarray:
        cols = self._columns()
        idx = {c: j for j, c in enumerate(cols)}
        X = out if out is not None else np.full((len(events), len(cols)), np.nan, dtype=self.DTYPE)
        if out is not None:
            X[:] = np.nan
        for i, e in enumerate(events):
            allowed = set(self.items) if asked is None else asked[i]
            for item in self.items:
                if item in allowed:
                    X[i, idx[item]] = e.item_values[item]
            for c, v in e.context_features.items():
                if c in idx:
                    X[i, idx[c]] = v
        return X

    def fit(self, events: list) -> None:
        """Fit on ``events`` plus their randomly masked copies."""
        # Union across all events, never one event's key set.
        self.context_names = sorted({k for e in events for k in e.context_features})

        y = np.array([e.target_y for e in events])
        n, n_cols = len(events), len(self._columns())
        n_blocks = 1 + self.n_dropout_augmentations
        X_train = np.empty((n * n_blocks, n_cols), dtype=self.DTYPE)

        self._to_matrix(events, asked=None, out=X_train[:n])
        for b in range(1, n_blocks):
            asked = []
            for _ in events:
                k = int(self.rng.integers(0, len(self.items) + 1))
                asked.append(set(self.rng.choice(self.items, size=k, replace=False)))
            self._to_matrix(events, asked=asked, out=X_train[b * n : (b + 1) * n])

        y_train = np.tile(y, n_blocks)
        if self.max_train_rows and len(X_train) > self.max_train_rows:
            keep = np.sort(self.rng.choice(len(X_train), size=self.max_train_rows, replace=False))
            X_train, y_train = X_train[keep], y_train[keep]
        self.model.fit(X_train, y_train)

    def predict(self, events: list, asked: list[set[str]]) -> np.ndarray:
        """Predict each event's target given only the items in ``asked[i]``."""
        return self.model.predict(self._to_matrix(events, asked=asked))
