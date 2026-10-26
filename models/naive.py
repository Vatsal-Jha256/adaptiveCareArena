"""Naive baselines -- the bar Stage 1 must clear. Includes one
baseline specific to this domain's structure: `LastKnownEMAModel` directly
exploits the `last_known_ema_score` feature crosscheck.py already builds
into every example -- pure persistence, the simplest possible forecast of
"the next EMA will look like the last one did."""

from __future__ import annotations

import numpy as np


class PopulationMeanModel:
    """Predicts the mean training target for everyone."""

    def __init__(self):
        self._mean = 0.0

    def fit(self, train: list) -> None:
        """Store the mean of the training targets (0.0 if there are none)."""
        ys = [e.target_y for e in train]
        self._mean = float(np.mean(ys)) if ys else 0.0

    def predict(self, examples: list) -> np.ndarray:
        """Return the stored population mean for every example."""
        return np.full(len(examples), self._mean)


class PersonalMeanModel:
    """Predicts each subject's own mean training target, falling back to the population mean for unseen subjects."""

    def __init__(self):
        self._by_subject: dict[str, float] = {}
        self._population_mean = 0.0

    def fit(self, train: list) -> None:
        """Store each training subject's mean target and the population mean."""
        by_subject: dict[str, list[float]] = {}
        for e in train:
            by_subject.setdefault(e.subject_id, []).append(e.target_y)
        self._by_subject = {sid: float(np.mean(v)) for sid, v in by_subject.items()}
        ys = [e.target_y for e in train]
        self._population_mean = float(np.mean(ys)) if ys else 0.0

    def predict(self, examples: list) -> np.ndarray:
        """Return each subject's stored mean, or the population mean if unseen."""
        return np.array(
            [self._by_subject.get(e.subject_id, self._population_mean) for e in examples]
        )


class LastKnownEMAModel:
    """Predicts the target as exactly the `last_known_ema_score` feature
    (carrying the most recent genuine EMA response forward) -- the
    strongest pure-persistence baseline this domain offers for free."""

    def __init__(self):
        self._fallback = 0.0

    def fit(self, train: list) -> None:
        """Store the mean training target, used only when an example has no last-known score."""
        ys = [e.target_y for e in train]
        self._fallback = float(np.mean(ys)) if ys else 0.0

    def predict(self, examples: list) -> np.ndarray:
        """Return ``last_known_ema_score`` for each example, or the stored fallback if it is NaN or absent."""
        out = []
        for e in examples:
            v = e.context_features.get("last_known_ema_score", float("nan"))
            out.append(v if v == v else self._fallback)  # v==v is a NaN check
        return np.array(out)
