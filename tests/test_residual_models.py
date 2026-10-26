"""Tests for the persistence-offset residual models. The properties that
matter are structural (does it really nest persistence?), not numeric."""

from dataclasses import dataclass, field

import numpy as np
import pytest

from models.residual import ANCHOR_FEATURE, ResidualGBMModel, ResidualRidgeModel


@dataclass
class _Ex:
    subject_id: str
    prediction_time_t: float
    target_y: float
    context_features: dict = field(default_factory=dict)
    horizon: float = 1.0
    target_observed_at: float = 0.0


def _pure_persistence_data(n=300, seed=0):
    """Target IS the anchor plus pure noise -- there is no residual signal
    to find, so a residual model must collapse toward persistence."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        anchor = rng.uniform(0, 10)
        out.append(
            _Ex(
                f"s{i % 10}",
                float(i),
                anchor + rng.normal(0, 0.05),
                {ANCHOR_FEATURE: anchor, "noise_a": rng.normal(), "noise_b": rng.normal()},
            )
        )
    return out


def _learnable_residual_data(n=400, seed=0):
    """Target = anchor + 3*drive. A real, learnable residual exists."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        anchor = rng.uniform(0, 10)
        drive = rng.normal()
        out.append(
            _Ex(
                f"s{i % 10}",
                float(i),
                anchor + 3.0 * drive,
                {ANCHOR_FEATURE: anchor, "drive": drive, "noise": rng.normal()},
            )
        )
    return out


@pytest.mark.parametrize("cls", [ResidualRidgeModel, ResidualGBMModel])
def test_collapses_to_persistence_when_no_residual_signal_exists(cls):
    data = _pure_persistence_data()
    model = cls()
    model.fit(data)
    preds = model.predict(data)
    anchors = np.array([e.context_features[ANCHOR_FEATURE] for e in data])
    # Must stay close to the anchor: the whole point of nesting is that a
    # signal-free residual cannot drag predictions away from persistence.
    assert np.mean(np.abs(preds - anchors)) < 0.5
    assert model.learned_correction_magnitude(data) < 0.5


@pytest.mark.parametrize("cls", [ResidualRidgeModel, ResidualGBMModel])
def test_recovers_a_real_residual_and_beats_plain_persistence(cls):
    data = _learnable_residual_data()
    model = cls()
    model.fit(data)
    y = np.array([e.target_y for e in data])
    anchors = np.array([e.context_features[ANCHOR_FEATURE] for e in data])
    mae_residual = np.mean(np.abs(model.predict(data) - y))
    mae_persistence = np.mean(np.abs(anchors - y))
    assert mae_residual < mae_persistence * 0.5, "must exploit a learnable residual"
    assert model.learned_correction_magnitude(data) > 1.0


@pytest.mark.parametrize("cls", [ResidualRidgeModel, ResidualGBMModel])
def test_anchor_is_excluded_from_the_residual_features(cls):
    # Leaving the anchor in the residual's own feature set lets the
    # regressor re-derive the identity mapping and undo the nesting the
    # model exists to guarantee.
    model = cls()
    model.fit(_learnable_residual_data())
    assert ANCHOR_FEATURE not in model.feature_names


@pytest.mark.parametrize("cls", [ResidualRidgeModel, ResidualGBMModel])
def test_missing_anchor_falls_back_to_training_mean_not_zero(cls):
    data = _learnable_residual_data()
    model = cls()
    model.fit(data)
    # An example with NO anchor feature at all.
    orphan = _Ex("new", 1.0, 0.0, {"drive": 0.0, "noise": 0.0})
    pred = model.predict([orphan])[0]
    train_mean = float(np.mean([e.target_y for e in data]))
    # Should sit near the training-mean anchor, not near 0.0 (which would
    # be a wild out-of-range prediction for data centred around ~5).
    assert abs(pred - train_mean) < abs(pred - 0.0)


@pytest.mark.parametrize("cls", [ResidualRidgeModel, ResidualGBMModel])
def test_fit_rejects_empty_training_data(cls):
    with pytest.raises(ValueError):
        cls().fit([])


@pytest.mark.parametrize("cls", [ResidualRidgeModel, ResidualGBMModel])
def test_predict_before_fit_raises(cls):
    with pytest.raises(RuntimeError):
        cls().predict(_learnable_residual_data(n=5))
