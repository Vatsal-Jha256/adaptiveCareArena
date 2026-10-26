"""Sequential replay of one fully recorded questionnaire event.

This software environment reveals historical answers one at a time. Actions do
not alter later participant states, reporting, or outcomes and therefore are
not interventions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from adaptivecarearena.interfaces import MeasurementPolicy


@dataclass(frozen=True)
class MeasurementObservation:
    """What a measurement policy sees at one step.

    Attributes:
        revealed: items already revealed, mapped to their recorded values.
        available: unrevealed items whose cost still fits in the budget.
        spent: total cost of the items revealed so far.
        remaining_budget: budget left (never negative).
    """

    revealed: Mapping[str, float]
    available: tuple[str, ...]
    spent: float
    remaining_budget: float


@dataclass(frozen=True)
class MeasurementTransition:
    """Result of one ``step`` call.

    Attributes:
        observation: the observation after the step.
        revealed_item: item revealed by this step, or ``None`` if the policy stopped.
        revealed_value: recorded value of that item, or ``None``.
        terminated: whether the episode is over.
        reason: ``"policy_stop"``, ``"all_items_revealed"``, ``"budget_exhausted"``,
            or ``None`` while the episode continues.
    """

    observation: MeasurementObservation
    revealed_item: str | None
    revealed_value: float | None
    terminated: bool
    reason: str | None


class RecordedQuestionnaireEnv:
    """A deterministic, retrospective item-reveal environment.

    ``answers`` must come from one complete recorded questionnaire. The policy
    sees only values already revealed by ``step``. Item costs default to one.
    """

    def __init__(
        self,
        answers: Mapping[str, float],
        *,
        budget: float,
        costs: Mapping[str, float] | None = None,
    ):
        if not answers:
            raise ValueError("answers must contain at least one recorded item")
        self._answers = {str(item): float(value) for item, value in answers.items()}
        self._items = tuple(self._answers)
        self._costs = (
            {item: 1.0 for item in self._items}
            if costs is None
            else {str(item): float(cost) for item, cost in costs.items()}
        )
        if set(self._costs) != set(self._answers):
            raise ValueError("costs must name exactly the recorded items")
        if any(cost <= 0 for cost in self._costs.values()):
            raise ValueError("every item cost must be positive")
        if budget < 0:
            raise ValueError("budget must be non-negative")
        self._budget = float(budget)
        self._revealed: dict[str, float] = {}
        self._spent = 0.0
        self._terminated = False

    def reset(self) -> MeasurementObservation:
        """Clear all revealed items and spend, and return the opening observation."""
        self._revealed = {}
        self._spent = 0.0
        self._terminated = False
        return self._observation()

    def _observation(self) -> MeasurementObservation:
        available = tuple(
            item
            for item in self._items
            if item not in self._revealed
            and self._costs[item] <= self._budget - self._spent + 1e-12
        )
        return MeasurementObservation(
            revealed=MappingProxyType(dict(self._revealed)),
            available=available,
            spent=self._spent,
            remaining_budget=max(0.0, self._budget - self._spent),
        )

    def step(self, item: str | None) -> MeasurementTransition:
        """Reveal one item, or stop the episode if ``item`` is ``None``.

        Raises:
            RuntimeError: if the episode has already terminated.
            ValueError: if ``item`` is not currently available (unknown, already
                revealed, or unaffordable).
        """
        if self._terminated:
            raise RuntimeError("measurement episode has terminated; call reset()")
        if item is None:
            self._terminated = True
            return MeasurementTransition(self._observation(), None, None, True, "policy_stop")
        observation = self._observation()
        if item not in observation.available:
            raise ValueError(f"item {item!r} is not currently available")
        value = self._answers[item]
        self._revealed[item] = value
        self._spent += self._costs[item]
        next_observation = self._observation()
        reason = None
        if len(self._revealed) == len(self._items):
            reason = "all_items_revealed"
        elif not next_observation.available:
            reason = "budget_exhausted"
        self._terminated = reason is not None
        return MeasurementTransition(
            next_observation,
            item,
            value,
            self._terminated,
            reason,
        )

    def run(self, policy: MeasurementPolicy) -> tuple[MeasurementTransition, ...]:
        """Play one full episode of ``policy`` and return every transition in order.

        The policy is reset with the item list and budget, then asked to choose
        until the environment terminates.
        """
        observation = self.reset()
        policy.reset(self._items, self._budget)
        transitions = []
        while True:
            item = policy.choose(
                observation.revealed,
                observation.available,
                observation.remaining_budget,
            )
            transition = self.step(item)
            transitions.append(transition)
            if transition.terminated:
                return tuple(transitions)
            observation = transition.observation
