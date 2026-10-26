from __future__ import annotations

import numpy as np

import adaptivecarearena.arena as arena_module
from adaptivecarearena import AdaptiveCareArena, FixedOrderMeasurementPolicy
from adaptivecarearena.composed import sequential_calibrated_trajectories
from scripts.composition_benchmark import (
    calibrated_trajectories,
    participant_folds,
)


def _panel() -> dict:
    return {
        subject: [(day, np.array([subject - day, 2 * subject], dtype=float)) for day in range(7)]
        for subject in range(6)
    }


def _config() -> dict:
    return {
        "seed": 7,
        "data": {"source": "ces", "local_root": "unused"},
        "outcome": {"threshold": -5.5, "horizon_days": 1},
        "measurement_budgets": [1, "full"],
        "review_budgets": [1],
        "calibration": {"n_folds": 3, "ridge_alpha": 1.0},
        "inference": {"n_boot": 20},
    }


def test_fixed_sequential_policy_matches_existing_fixed_subset_calibration():
    panel = _panel()
    folds = participant_folds(sorted(panel), 3, 7)
    expected_truth, expected_belief = calibrated_trajectories(panel, [0, 1], 1, folds, 3, 1.0)
    truth, belief, diagnostics = sequential_calibrated_trajectories(
        panel,
        ("item_a", "item_b"),
        FixedOrderMeasurementPolicy(),
        1,
        folds,
        3,
        1.0,
    )
    assert truth == expected_truth
    for subject in belief:
        np.testing.assert_allclose(belief[subject], expected_belief[subject])
    assert diagnostics["mean_items_revealed"] == 1.0


def test_public_composed_evaluation_keeps_eligibility_inside_benchmark(monkeypatch):
    monkeypatch.setattr(
        arena_module,
        "load_questionnaire_panel",
        lambda dataset, align, local_root: (_panel(), ("item_a", "item_b"), -5.5),
    )
    result = AdaptiveCareArena(_config()).evaluate_measurement_policies(
        {"fixed_external": FixedOrderMeasurementPolicy()}, measurement_budget=1
    )
    assert result.summary["subjects"] == 6
    assert result.summary["measurement_budget"] == 1
    assert result.summary["candidate_rows"] > 0
    assert result.summary["policy_diagnostics"]["fixed_external"]["mean_items_revealed"] == 1.0
    assert "fixed_external-full" in result.summary["by_review_budget"]["1"]["contrasts"]
