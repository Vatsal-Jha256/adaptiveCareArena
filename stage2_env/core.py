"""Core types + plugin registry for the budgeted acquisition-timing
environment. State is continuous (CrossCheck's composite EMA score,
`ema_pos_score - ema_neg_score`, is a real-valued mood/symptom composite,
not a probability).

Status: internal validation machinery, not a result. The synthetic
environment is kept because comparing a policy ranking here against the
same ranking on recorded trajectories (`replay_world.py`) is a useful
methodological check, but no result computed against this simulator is
evidence about a real cohort. It is a pipeline/harness calibrated to
real data statistics (see scripts/calibrate_from_crosscheck.py). Generic module: must not import from world/,
policies/, estimators/, or metrics/."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class SubjectSpec:
    """Static identity of one subject: which severity-tercile group they
    belong to (see world.py's GroupSpec) and their own calibrated baseline
    offset from the group mean (real per-subject heterogeneity, not shared
    group-level noise)."""

    subject_id: int
    group: str
    baseline_offset: float


@dataclass
class SubjectState:
    """Mutable per-round latent state (a centered version of the composite
    EMA score -- see world.py). Never exposed directly outside world/ --
    only SubjectFeature (no true_state) and, for checked subjects only,
    Observation are exposed."""

    subject_id: int
    true_state: float
    rounds_since_check: int = 0
    last_observed_round: int | None = None
    last_observed_state: float | None = None


@dataclass(frozen=True)
class SubjectFeature:
    """Everything a policy or estimator may see for one subject, available
    before this round's check decision. Deliberately has no true_state
    field -- ground-truth leakage guard, mirrors allocarena's ArmFeature."""

    group_onehot: tuple[float, ...]
    rounds_since_check: int
    last_observed_state: float
    round_idx_norm: float
    check_count_so_far: int


@dataclass(frozen=True)
class Observation:
    """Revealed only for a checked subject. true_state_at_check is kept
    for diagnostics; estimators must train on `observed_state` (identical
    here -- see world.py's documented simplification that this
    environment has no separable self-report noise term) not on any
    oracle field."""

    round_idx: int
    subject_id: int
    group: str
    feature: SubjectFeature
    observed_state: float
    true_state_at_check: float


class Registry:
    """A name-to-class registry used to select policies and estimators from configs."""

    def __init__(self, kind: str):
        self._kind = kind
        self._items: dict[str, Callable[..., object]] = {}

    def register(self, name: str):
        """Decorator that registers a class under ``name``; duplicate names raise ``ValueError``."""

        def deco(cls_or_fn):
            """Register ``cls_or_fn`` and return it unchanged."""
            if name in self._items:
                raise ValueError(f"{self._kind} '{name}' already registered")
            self._items[name] = cls_or_fn
            return cls_or_fn

        return deco

    def get(self, name: str):
        """Look up a registered item, raising ``KeyError`` that lists the valid names."""
        if name not in self._items:
            raise KeyError(f"unknown {self._kind} '{name}'; registered: {sorted(self._items)}")
        return self._items[name]

    def names(self) -> list[str]:
        """Return the registered names, sorted."""
        return sorted(self._items)


POLICIES = Registry("policy")
ESTIMATORS = Registry("estimator")
