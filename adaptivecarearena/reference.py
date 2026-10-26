"""Small reference estimators implemented only through the public contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


class Persistence:
    """Rank by the need implied by the most recently observed report."""

    def reset(self) -> None:
        """Forget every subject."""
        self._need: dict[int, float] = {}

    def observe(self, subject_id: int, observed_at: int, value: float) -> None:
        """Store the need implied by this report (need is the negated score)."""
        self._need[subject_id] = -value

    def score(self, subject_id: int, prediction_time: int) -> float:
        """Return the need implied by the subject's latest report."""
        return self._need[subject_id]


class ExponentialMovingAverage:
    """Reference exponentially smoothed need estimator."""

    def __init__(self, alpha: float = 0.3):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = float(alpha)
        self.reset()

    def reset(self) -> None:
        """Forget every subject."""
        self._need: dict[int, float] = {}

    def observe(self, subject_id: int, observed_at: int, value: float) -> None:
        """Blend this report's need into the subject's running average."""
        need = -value
        previous = self._need.get(subject_id, need)
        self._need[subject_id] = self.alpha * need + (1.0 - self.alpha) * previous

    def score(self, subject_id: int, prediction_time: int) -> float:
        """Return the subject's smoothed need."""
        return self._need[subject_id]


class FixedOrderMeasurementPolicy:
    """Reveal affordable items in one declared, answer-independent order."""

    def __init__(self, order: Sequence[str] | None = None):
        self._declared_order = tuple(order) if order is not None else None

    def reset(self, items: Sequence[str], budget: float) -> None:
        """Nothing to clear; the order is fixed at construction."""
        self._order = self._declared_order or tuple(items)

    def choose(
        self,
        revealed: Mapping[str, float],
        available: tuple[str, ...],
        remaining_budget: float,
    ) -> str | None:
        """Return the first available item in the declared order, or ``None`` to stop."""
        return next((item for item in self._order if item in available), None)
