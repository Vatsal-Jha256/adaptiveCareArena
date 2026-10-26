"""Universal task contract: one example per (subject, prediction time), with
the leakage invariant that every feature timestamp is <= prediction time
and the target is observed no earlier than prediction time + horizon."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CareExample:
    """One forecasting example for one subject at one prediction time.

    Leakage invariant: every entry of ``context_features`` is known at or
    before ``prediction_time_t``, and the target was observed at
    ``target_observed_at >= prediction_time_t + horizon``.

    Attributes:
        subject_id: participant identifier (never re-identified).
        prediction_time_t: time at which the forecast is made.
        horizon: how far ahead the target lies.
        context_features: feature name to value, all available at prediction time.
        target_y: the value being forecast.
        target_observed_at: when ``target_y`` was actually recorded.
        domain_id: dataset the example came from.
        cycle_or_state_id: optional grouping tag for examples sharing a state.
        sample_weight: weight for training.
        provenance: free-text note on how the example was built.
    """

    subject_id: str
    prediction_time_t: float
    horizon: float
    context_features: dict[str, float]
    target_y: float
    target_observed_at: float
    domain_id: str
    cycle_or_state_id: str | None = None
    sample_weight: float = 1.0
    provenance: str = ""


def assert_no_future_leakage(examples: list[CareExample]) -> None:
    """Raise ``ValueError`` if any target was observed before ``t + horizon``."""
    for ex in examples:
        if ex.target_observed_at < ex.prediction_time_t + ex.horizon:
            raise ValueError(
                f"leakage: example for subject={ex.subject_id} at "
                f"t={ex.prediction_time_t} has target_observed_at="
                f"{ex.target_observed_at}, before prediction_time_t+horizon="
                f"{ex.prediction_time_t + ex.horizon}"
            )
