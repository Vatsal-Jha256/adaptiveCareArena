"""Check-in (acquisition) policies. Every policy implements
`select(features, predicted_need, budget, round_idx, rng) -> set[int]` and
receives only features/predicted_need -- never true_state. `predicted_need`
is higher for subjects the estimator believes are doing worse (more
negative composite EMA state) -- see estimators.py.

Every policy that ranks by a score breaks exact ties with rng jitter
before argmax/sort, not plain sort/max. This is not a stylistic choice. A deterministic `max()`/`sort()` silently
defaults to "always the same low-id subject" whenever scores are tied,
freezing what should be an active policy into a static one."""

from __future__ import annotations

import numpy as np

from stage2_env.core import POLICIES


@POLICIES.register("random")
class RandomPolicy:
    """Reviews ``budget`` subjects chosen uniformly at random each round."""

    def select(self, features, predicted_need, budget, round_idx, rng):
        """Return ``budget`` subject ids sampled without replacement."""
        ids = list(features.keys())
        chosen = rng.choice(ids, size=min(budget, len(ids)), replace=False)
        return {int(x) for x in chosen}


@POLICIES.register("round_robin")
class RoundRobinPolicy:
    """Deterministic full-coverage baseline -- matches CrossCheck's own
    real fixed every-2-3-days schedule most closely of any policy here,
    so it is the natural comparator for "does a smarter policy beat the
    schedule actually used," once Stage 1 earns the right to ask that."""

    def __init__(self):
        self._cursor = 0

    def select(self, features, predicted_need, budget, round_idx, rng):
        """Return the next ``budget`` subjects in id order, wrapping around."""
        ids = sorted(features.keys())
        n = len(ids)
        b = min(budget, n)
        chosen = {ids[(self._cursor + k) % n] for k in range(b)}
        self._cursor = (self._cursor + b) % n
        return chosen


@POLICIES.register("round_robin_shuffled")
class ShuffledRoundRobinPolicy:
    """round_robin over a fixed random permutation of subject ids (drawn once,
    on first call). `round_robin` walks ids in sorted order, and panels built
    by the bootstrap number ids contiguously by severity tercile, so its cursor
    sweeps one tercile at a time: its attention disparity is an artifact of id
    ordering, not a property of cyclic review (measured: relabelled-id
    round_robin matches `random` on disparity; outputs/review_extras_*.json).
    This is the id-order-agnostic cyclic baseline. `round_robin` is kept
    unchanged so earlier outputs stay reproducible."""

    def __init__(self):
        self._cursor = 0
        self._order = None

    def select(self, features, predicted_need, budget, round_idx, rng):
        """Return the next ``budget`` subjects in the fixed random order, wrapping around."""
        if self._order is None:
            self._order = [int(x) for x in rng.permutation(sorted(features.keys()))]
        pos = {sid: i for i, sid in enumerate(self._order)}
        ids = sorted(features.keys(), key=lambda s: pos.get(s, len(pos)))
        n = len(ids)
        b = min(budget, n)
        chosen = {ids[(self._cursor + k) % n] for k in range(b)}
        self._cursor = (self._cursor + b) % n
        return chosen


@POLICIES.register("greedy_need")
class GreedyNeedPolicy:
    """Checks the top-`budget` subjects by current predicted need (worse
    predicted mood first)."""

    def select(self, features, predicted_need, budget, round_idx, rng):
        """Return the ``budget`` subjects with the highest predicted need; ties are broken at random."""
        jitter = rng.uniform(-1e-9, 1e-9, size=len(predicted_need))
        scored = {sid: v + j for (sid, v), j in zip(predicted_need.items(), jitter, strict=False)}
        ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
        return {sid for sid, _ in ranked[:budget]}


@POLICIES.register("ucb_explore")
class UCBExplorePolicy:
    """Predicted need plus an exploration bonus that shrinks with check
    count -- isolates how much of any closed-loop effect a classic
    optimism bonus fixes on its own."""

    def __init__(self, c: float = 0.5):
        self.c = c

    def select(self, features, predicted_need, budget, round_idx, rng):
        """Return the ``budget`` subjects with the highest need plus exploration bonus."""
        scores = {}
        for sid, feat in features.items():
            bonus = self.c * np.sqrt(np.log(round_idx + 2) / (feat.check_count_so_far + 1))
            scores[sid] = predicted_need.get(sid, 0.0) + bonus
        jitter = rng.uniform(-1e-9, 1e-9, size=len(scores))
        jittered = {sid: v + j for (sid, v), j in zip(scores.items(), jitter, strict=False)}
        ranked = sorted(jittered.items(), key=lambda kv: kv[1], reverse=True)
        return {sid for sid, _ in ranked[:budget]}


@POLICIES.register("fair_floor")
class FairFloorPolicy:
    """Reserves a fixed fraction of the budget each round for the
    least-recently-checked subjects, greedy-need on the remainder."""

    def __init__(self, floor_frac: float = 0.3):
        self.floor_frac = floor_frac

    def select(self, features, predicted_need, budget, round_idx, rng):
        """Return the stalest subjects for the reserved share, then the highest-need subjects."""
        floor_n = max(1, int(budget * self.floor_frac))
        jitter1 = rng.uniform(-1e-9, 1e-9, size=len(features))
        by_staleness = sorted(
            zip(features.items(), jitter1, strict=False),
            key=lambda pair: pair[0][1].rounds_since_check + pair[1],
            reverse=True,
        )
        floor_ids = {pair[0][0] for pair in by_staleness[:floor_n]}
        remaining = budget - len(floor_ids)
        candidates = [(sid, r) for sid, r in predicted_need.items() if sid not in floor_ids]
        jitter2 = rng.uniform(-1e-9, 1e-9, size=len(candidates))
        ranked = sorted(
            zip(candidates, jitter2, strict=False),
            key=lambda pair: pair[0][1] + pair[1],
            reverse=True,
        )
        return floor_ids | {sid for (sid, _r), _j in ranked[:remaining]}


@POLICIES.register("whittle_index")
class WhittleIndexPolicy:
    """A Whittle-index-inspired policy (not a solved per-subject MDP):
    predicted need plus
    a bonus linear in rounds-since-check. Since this world's own
    unobserved-drift mechanism is a variance increment rather than a
    directional push (world.py), the staleness bonus here represents
    growing uncertainty about a stale subject's state, not growing
    predicted harm."""

    def __init__(self, staleness_belief: float = 0.05):
        self.staleness_belief = staleness_belief

    def select(self, features, predicted_need, budget, round_idx, rng):
        """Return the ``budget`` subjects with the highest need plus staleness bonus."""
        scores = {
            sid: predicted_need.get(sid, 0.0) + self.staleness_belief * feat.rounds_since_check
            for sid, feat in features.items()
        }
        jitter = rng.uniform(-1e-9, 1e-9, size=len(scores))
        jittered = {sid: v + j for (sid, v), j in zip(scores.items(), jitter, strict=False)}
        ranked = sorted(jittered.items(), key=lambda kv: kv[1], reverse=True)
        return {sid for sid, _ in ranked[:budget]}
