"""round_robin_shuffled must be id-order-agnostic (see outputs/review_extras_*.json: id-ordered
round_robin's attention disparity was an artifact of tercile-contiguous ids)."""

import numpy as np

from stage2_env import policies  # noqa: F401  (registry side effects)
from stage2_env.core import POLICIES


def _run(ids, rounds=60, budget=3, seed=0):
    pol = POLICIES.get("round_robin_shuffled")()
    rng = np.random.default_rng(seed)
    feats = {i: None for i in ids}
    counts = {i: 0 for i in ids}
    first_cycle = []
    for r in range(rounds):
        chosen = pol.select(feats, {}, budget, r, rng)
        assert len(chosen) == budget
        for c in chosen:
            counts[c] += 1
        if r < len(ids) // budget:
            first_cycle.extend(sorted(chosen))
    return counts, first_cycle


def test_equal_coverage_over_full_cycles():
    ids = list(range(12))
    counts, _ = _run(ids, rounds=12 * 5 // 3 * 3 // 3 * 1 + 0)  # 20 rounds*3 = 60 picks = 5 per id
    assert max(counts.values()) - min(counts.values()) <= 1


def test_not_sorted_id_order():
    ids = list(range(30))
    _, first = _run(ids, rounds=10, budget=3)
    assert first != sorted(first), "cursor must not sweep ids in sorted order"


def test_registry_has_both_variants():
    assert "round_robin" in POLICIES.names() and "round_robin_shuffled" in POLICIES.names()
