"""Item-acquisition policies. Every policy implements
`run(events, model, budget, cost_profile, rng, y_true=None) -> (asked, mean_cost)`
where `asked[i]` is the set of items actually put to the patient for
event i, costing at most `budget` in burden.

`y_true` is accepted everywhere for interface uniformity and used only by
the adaptive policy, as a bandit reward observed after the questionnaire
for that event is complete. That reward updates beliefs applied to later
events; it never changes what was asked or predicted for the event it
came from. That is the ordinary decide-observe-update loop, not the
ground-truth leakage this project forbids at decision time."""

from __future__ import annotations

import numpy as np

# Note: no module-level item list. `cost_profile` is keyed by item and
# therefore already defines what is acquirable, so every policy derives
# the item set from it. That is what makes these policies instrument-
# agnostic: CrossCheck has 10 signed EMA items, CES has 4 PHQ-4 items,
# and neither the policies nor the model should need to know which.


class AskAllPolicy:
    """Upper bound: the full questionnaire, ignoring burden."""

    name = "ask_all"

    def run(self, events, model, budget, cost_profile, rng, y_true=None):
        """Ask every item at every event."""
        return [set(cost_profile) for _ in events], sum(cost_profile.values())


class AskNonePolicy:
    """Zero-burden floor: predict from passive sensing and EMA history
    alone. The baseline any item-selection policy must beat to
    justify asking a patient anything at all."""

    name = "ask_none"

    def run(self, events, model, budget, cost_profile, rng, y_true=None):
        """Ask nothing at any event."""
        return [set() for _ in events], 0.0


class FixedSubsetPolicy:
    """The same precomputed item set for everyone, chosen on validation
    under the burden budget -- the short-form questionnaire baseline, and
    the one an adaptive policy actually has to beat."""

    name = "fixed_subset"

    def __init__(self, subset: set[str]):
        self.subset = set(subset)

    def run(self, events, model, budget, cost_profile, rng, y_true=None):
        """Ask the same subset at every event; the cost is that subset's cost."""
        cost = sum(cost_profile[i] for i in self.subset)
        return [set(self.subset) for _ in events], cost


class RandomMatchedPolicy:
    """Random items until the budget is spent -- controls for "asking
    more questions helps" independent of which questions."""

    name = "random_matched"

    def run(self, events, model, budget, cost_profile, rng, y_true=None):
        """Ask a fresh random item order per event, keeping items while the budget allows."""
        asked, costs = [], []
        items = list(cost_profile)
        for _ in events:
            order = rng.permutation(items)
            chosen, spent = set(), 0.0
            for it in order:
                if spent + cost_profile[it] <= budget:
                    chosen.add(it)
                    spent += cost_profile[it]
            asked.append(chosen)
            costs.append(spent)
        return asked, float(np.mean(costs))


class GreedyVOIPolicy:
    """Asks items in a fixed rank order (by standalone marginal value on
    validation) until the budget is spent. Static: the order does not
    adapt per patient or over time, which is what separates it from
    ThompsonItemPolicy."""

    name = "greedy_voi"

    def __init__(self, rank_order: list[str]):
        self.rank_order = list(rank_order)

    def run(self, events, model, budget, cost_profile, rng, y_true=None):
        """Ask items in rank order until the budget is spent; the same set for every event."""
        chosen, spent = set(), 0.0
        for it in self.rank_order:
            if spent + cost_profile[it] <= budget:
                chosen.add(it)
                spent += cost_profile[it]
        return [set(chosen) for _ in events], spent


class ThompsonItemPolicy:
    """The one adaptive policy: replays events as a stream and
    Thompson-samples which item to ask next from a per-item Beta
    posterior, stopping when the burden budget is spent.

    Reward is outcome-based: after the questionnaire for an event is
    complete, every item asked is credited 1 if the resulting prediction
    landed within `tol` of the realized target, else 0. An earlier
    confidence-swing proxy never checked correctness and underperformed
    every fixed baseline -- a reward blind to whether the swing was toward the right answer
    rewards being confidently wrong exactly as much as being right. Equal
    credit across asked items is a documented simplification, not a
    counterfactual/Shapley attribution."""

    name = "thompson_items"

    def __init__(self, tol: float = 2.0):
        self.tol = tol
        self.alpha: dict[str, float] = {}
        self.beta: dict[str, float] = {}

    def run(self, events, model, budget, cost_profile, rng, y_true=None):
        """Replay events in random order, Thompson-sampling items from Beta posteriors.

        After each event the model predicts from the chosen items and each asked
        item is credited 1 if the prediction landed within ``tol`` of ``y_true``.
        The reward only updates choices for later events. Raises ``ValueError`` if
        ``y_true`` is missing.
        """
        if y_true is None:
            raise ValueError("ThompsonItemPolicy.run() requires y_true (outcome-based reward)")
        for it in cost_profile:
            self.alpha.setdefault(it, 1.0)
            self.beta.setdefault(it, 1.0)

        order = rng.permutation(len(events))
        asked = [set() for _ in events]
        costs = np.zeros(len(events))

        for idx in order:
            e = events[idx]
            chosen, spent = set(), 0.0
            while True:
                affordable = [
                    i for i in cost_profile if i not in chosen and spent + cost_profile[i] <= budget
                ]
                if not affordable:
                    break
                samples = {i: rng.beta(self.alpha[i], self.beta[i]) for i in affordable}
                pick = max(samples, key=lambda k: samples[k])
                chosen.add(pick)
                spent += cost_profile[pick]

            pred = model.predict([e], [chosen])[0]
            correct = abs(pred - y_true[idx]) <= self.tol
            for i in chosen:
                if correct:
                    self.alpha[i] += 1
                else:
                    self.beta[i] += 1

            asked[idx] = chosen
            costs[idx] = spent

        return asked, float(np.mean(costs))
