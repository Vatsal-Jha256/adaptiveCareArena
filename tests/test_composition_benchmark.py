import numpy as np

from scripts.composition_benchmark import (
    calibrated_trajectories,
    full_trajectories,
    participant_folds,
    score_diagnostics,
)


def _panel():
    return {
        "a": [(0, np.array([1.0, 2.0])), (1, np.array([2.0, 3.0]))],
        "b": [(0, np.array([2.0, 1.0])), (1, np.array([3.0, 2.0]))],
        "c": [(0, np.array([4.0, 1.0])), (1, np.array([5.0, 2.0]))],
        "d": [(0, np.array([1.0, 4.0])), (1, np.array([2.0, 5.0]))],
    }


def test_participant_cross_fit_produces_all_subjects_without_row_leakage():
    rows = _panel()
    folds = participant_folds(sorted(rows), 2, 9)
    truth, belief = calibrated_trajectories(rows, [0, 1], 1, folds, 2, 1.0)
    assert set(truth) == set(belief) == {0, 1, 2, 3}
    assert all(len([v for v in series if v is not None]) == 2 for series in belief.values())


def test_score_diagnostics_names_bias_and_ranges():
    truth = {0: [1.0, 2.0]}
    belief = {0: [1.5, 1.5]}
    out = score_diagnostics(truth, belief)
    assert out["mae"] == 0.5
    assert out["bias"] == 0.0
    assert out["truth_range"] == [1.0, 2.0]


def test_full_trajectory_is_exact_not_regularized():
    truth = full_trajectories(_panel())
    out = score_diagnostics(truth, truth)
    assert out["mae"] == 0.0
    assert out["bias"] == 0.0
