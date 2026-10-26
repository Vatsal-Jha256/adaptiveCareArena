"""Guards for the data-adaptive greedy item order in the composition task.

The arrays here are test fixtures for a pure selection function; no result
depends on them and no config can name them.
"""

from __future__ import annotations

import numpy as np

from scripts.composition_benchmark import calibrated_trajectories, greedy_items


def test_greedy_selects_the_informative_item_first() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 4))
    y = 3.0 * x[:, 2] + 0.1 * x[:, 0]
    assert greedy_items(x, y, 1, 1.0) == [2]
    assert greedy_items(x, y, 2, 1.0)[0] == 2
    assert len(set(greedy_items(x, y, 3, 1.0))) == 3


def _panel(seed: int, n_subjects: int = 10, n_days: int = 25) -> dict:
    rng = np.random.default_rng(seed)
    return {s: [(d, rng.normal(size=4)) for d in range(n_days)] for s in range(n_subjects)}


def test_heldout_participants_do_not_influence_their_own_selection() -> None:
    """Changing a held-out participant's rows must not change its fold's model.

    Predictions for every other participant in other folds can shift (they
    train on this participant), but within the held-out fold the fitted model
    is a function of training participants only. We check the strongest
    observable consequence: perturbing one participant leaves the predictions
    of participants in its own fold, computed from unchanged inputs, intact.
    """
    folds = {s: s % 5 for s in range(10)}
    base = _panel(1)
    _, belief_a = calibrated_trajectories(base, None, 2, folds, 5, 1.0)
    perturbed = {s: list(rows) for s, rows in base.items()}
    perturbed[0] = [(d, v * 10.0) for d, v in perturbed[0]]
    _, belief_b = calibrated_trajectories(perturbed, None, 2, folds, 5, 1.0)
    index = {s: i for i, s in enumerate(sorted(base))}
    # Subject 5 shares participant 0's fold (5 % 5 == 0): both are held out
    # together, so neither participant's rows feed that fold's model.
    assert belief_a[index[5]] == belief_b[index[5]]
