"""Public API for retrospective AdaptiveCareArena evaluation.

The package evaluates recorded self-report proxies.  It does not validate
diagnosis, treatment, clinical decision support, or deployment benefit.
"""

from adaptivecarearena.arena import AdaptiveCareArena, EvaluationResult, MethodArm
from adaptivecarearena.interfaces import MeasurementPolicy, ReviewPolicy, StateEstimator
from adaptivecarearena.measurement import (
    MeasurementObservation,
    MeasurementTransition,
    RecordedQuestionnaireEnv,
)
from adaptivecarearena.reference import (
    ExponentialMovingAverage,
    FixedOrderMeasurementPolicy,
    Persistence,
)

__all__ = [
    "AdaptiveCareArena",
    "EvaluationResult",
    "ExponentialMovingAverage",
    "FixedOrderMeasurementPolicy",
    "MeasurementObservation",
    "MeasurementPolicy",
    "MeasurementTransition",
    "MethodArm",
    "Persistence",
    "RecordedQuestionnaireEnv",
    "ReviewPolicy",
    "StateEstimator",
]
