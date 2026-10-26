"""Cross-fitted composition of sequential item replay and review evaluation."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from sklearn.linear_model import Ridge

from adaptivecarearena.interfaces import MeasurementPolicy
from adaptivecarearena.measurement import RecordedQuestionnaireEnv


def _revealed_vector(
    values: np.ndarray,
    items: tuple[str, ...],
    policy: MeasurementPolicy,
    budget: float,
) -> np.ndarray:
    answers = {item: float(values[index]) for index, item in enumerate(items)}
    transitions = RecordedQuestionnaireEnv(answers, budget=budget).run(policy)
    revealed = transitions[-1].observation.revealed
    value_block = [revealed.get(item, 0.0) for item in items]
    flag_block = [1.0 if item in revealed else 0.0 for item in items]
    return np.asarray([*value_block, *flag_block], dtype=float)


def sequential_calibrated_trajectories(
    subject_rows: dict,
    items: tuple[str, ...],
    policy: MeasurementPolicy,
    budget: float,
    folds: Mapping,
    n_folds: int,
    ridge_alpha: float,
) -> tuple[dict[int, list], dict[int, list], dict[str, float | int]]:
    """Run one policy on train and held-out participants, then calibrate.

    Calibration uses revealed values plus observed flags and is fitted only on
    other participants. Full composites are targets for calibration and later
    outcomes, never inputs to the measurement policy.
    """
    max_day = max(day for rows in subject_rows.values() for day, _ in rows)
    truth: dict[int, list] = {}
    belief: dict[int, list] = {}
    subject_index = {subject: index for index, subject in enumerate(sorted(subject_rows))}
    item_counts: list[int] = []
    for held_out in range(n_folds):
        train_examples = [
            (subject, day, values)
            for subject, rows in subject_rows.items()
            if folds[subject] != held_out
            for day, values in rows
        ]
        train_x = np.vstack(
            [_revealed_vector(values, items, policy, budget) for _, _, values in train_examples]
        )
        train_y = np.asarray([values.sum() for _, _, values in train_examples], dtype=float)
        model = Ridge(alpha=ridge_alpha).fit(train_x, train_y)
        for subject, rows in subject_rows.items():
            if folds[subject] != held_out:
                continue
            index = subject_index[subject]
            truth[index] = [None] * (max_day + 1)
            belief[index] = [None] * (max_day + 1)
            for day, values in rows:
                features = _revealed_vector(values, items, policy, budget)
                item_counts.append(int(features[len(items) :].sum()))
                truth[index][day] = float(values.sum())
                belief[index][day] = float(model.predict(features.reshape(1, -1))[0])
    diagnostics: dict[str, float | int] = {
        "reports": len(item_counts),
        "mean_items_revealed": float(np.mean(item_counts)) if item_counts else 0.0,
        "minimum_items_revealed": min(item_counts) if item_counts else 0,
        "maximum_items_revealed": max(item_counts) if item_counts else 0,
    }
    return truth, belief, diagnostics
