"""Contracts for externally supplied patient-state estimators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class StateEstimator(Protocol):
    """A longitudinal estimator that never receives future outcomes.

    ``observe`` is called in timestamp order for every recorded report at or
    before a decision. ``score`` must return a finite scalar where larger means
    greater review need. The evaluator owns eligibility, targets, queues, and
    metrics, preventing a submitted method from reading outcome labels.
    """

    def reset(self) -> None:
        """Clear all state before a benchmark evaluation."""

    def observe(self, subject_id: int, observed_at: int, value: float) -> None:
        """Update from one legally available self-report observation."""

    def score(self, subject_id: int, prediction_time: int) -> float:
        """Return decision priority at ``prediction_time`` (larger is worse)."""


@runtime_checkable
class MeasurementPolicy(Protocol):
    """Choose the next unrevealed recorded item, or stop with ``None``."""

    def reset(self, items: Sequence[str], budget: float) -> None:
        """Clear policy state for one recorded questionnaire event."""

    def choose(
        self,
        revealed: Mapping[str, float],
        available: tuple[str, ...],
        remaining_budget: float,
    ) -> str | None:
        """Return one name from ``available`` without seeing hidden answers."""


@runtime_checkable
class ReviewPolicy(Protocol):
    """Convert a legal state estimate into a review-priority score."""

    def reset(self) -> None:
        """Clear policy state before evaluation."""

    def priority(
        self,
        subject_id: int,
        prediction_time: int,
        estimated_need: float,
    ) -> float:
        """Return finite priority; larger values are allocated first."""
