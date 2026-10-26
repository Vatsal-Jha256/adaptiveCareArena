"""Reproducibility contract for every fitted model, estimator and stochastic
policy in the package.

Three properties are checked, each for every component that has the
relevant behaviour:

1. **Determinism**: two independent instances built with the same
   arguments and seed, fitted on the same data, give bit-identical
   predictions.
2. **Seed sensitivity** (stochastic components only): a different seed
   changes the output. Without this, property 1 could pass for a
   component that silently ignores its seed. A seed-invariant component
   would also make a multi-seed study report a standard deviation of zero
   that means "ignores the seed", not "stable".
3. **No hidden state**: re-running ``fit`` on the same instance, and
   predicting twice, does not change the answer.

The data here are synthetic test fixtures. They exercise the code path;
they are not evidence about any cohort.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from adaptivecarearena.reference import ExponentialMovingAverage, Persistence
from emaselect.model import MaskAwareEMAModel
from models.gbm import GBMModel
from models.naive import LastKnownEMAModel, PersonalMeanModel, PopulationMeanModel
from models.residual import ANCHOR_FEATURE, ResidualGBMModel, ResidualRidgeModel
from models.statistical import RidgeRegressionModel
from stage2_env.core import Observation, SubjectFeature
from stage2_env.estimators import KalmanNeedEstimator, NaiveEMAEstimator
from stage2_env.policies import RandomPolicy, ShuffledRoundRobinPolicy, UCBExplorePolicy


@dataclass
class _Example:
    subject_id: str
    prediction_time_t: float
    target_y: float
    context_features: dict = field(default_factory=dict)
    horizon: float = 1.0
    target_observed_at: float = 0.0


def _tabular_examples(n: int = 200, seed: int = 0) -> list[_Example]:
    """Subjects with a persistent level plus a feature-driven component."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        anchor = rng.uniform(0, 10)
        drive = rng.normal()
        out.append(
            _Example(
                f"s{i % 8}",
                float(i),
                anchor + 2.0 * drive + rng.normal(0, 0.1),
                {
                    ANCHOR_FEATURE: anchor,
                    "last_known_ema_score": anchor,
                    "drive": drive,
                    "noise": rng.normal(),
                },
            )
        )
    return out


# --------------------------------------------------------------------------
# Supervised models: fit(train) / predict(examples)
# --------------------------------------------------------------------------

# name -> (factory taking a seed, whether the model is stochastic). The
# flag documents intent; only the stochastic models are seed-sensitive.
_SUPERVISED = {
    "population_mean": (lambda seed: PopulationMeanModel(), False),
    "personal_mean": (lambda seed: PersonalMeanModel(), False),
    "last_known_ema": (lambda seed: LastKnownEMAModel(), False),
    "ridge": (lambda seed: RidgeRegressionModel(alpha=1.0), False),
    "residual_ridge": (lambda seed: ResidualRidgeModel(), False),
    "gbm": (lambda seed: GBMModel(n_estimators=40, random_state=seed), False),
}


@pytest.mark.parametrize("name", sorted(_SUPERVISED))
def test_supervised_model_is_deterministic_for_a_fixed_seed(name):
    factory, _ = _SUPERVISED[name]
    train, test = _tabular_examples(seed=1), _tabular_examples(60, seed=2)
    a, b = factory(0), factory(0)
    a.fit(train)
    b.fit(train)
    np.testing.assert_array_equal(a.predict(test), b.predict(test))


@pytest.mark.parametrize("name", sorted(_SUPERVISED))
def test_supervised_model_refit_and_repredict_do_not_drift(name):
    factory, _ = _SUPERVISED[name]
    train, test = _tabular_examples(seed=1), _tabular_examples(60, seed=2)
    model = factory(0)
    model.fit(train)
    first = model.predict(test)
    np.testing.assert_array_equal(first, model.predict(test))
    model.fit(train)
    np.testing.assert_array_equal(first, model.predict(test))


def test_residual_gbm_is_deterministic_and_has_a_default_seed():
    train, test = _tabular_examples(seed=1), _tabular_examples(60, seed=2)
    a, b = ResidualGBMModel(), ResidualGBMModel()
    a.fit(train)
    b.fit(train)
    np.testing.assert_array_equal(a.predict(test), b.predict(test))


def test_supervised_fit_does_not_mutate_its_training_data():
    train = _tabular_examples(seed=1)
    snapshot = [(e.subject_id, e.target_y, dict(e.context_features)) for e in train]
    for factory, _ in _SUPERVISED.values():
        factory(0).fit(train)
    assert snapshot == [(e.subject_id, e.target_y, dict(e.context_features)) for e in train]


def test_gbm_forwards_its_seed_to_the_underlying_estimator():
    """``GBMModel`` uses default (non-subsampled) boosting, so its output is
    seed-invariant by construction. The seed must still reach the
    estimator, so enabling subsampling later cannot silently ignore it."""
    train = _tabular_examples(seed=1)
    for seed in (3, 4):
        model = GBMModel(n_estimators=10, random_state=seed)
        model.fit(train)
        assert model._model.get_params()["random_state"] == seed


# --------------------------------------------------------------------------
# Mask-aware item-dropout model (measurement environment)
# --------------------------------------------------------------------------

_ITEMS = ("q1", "q2", "q3", "q4")


@dataclass
class _Event:
    subject_id: str
    day_ordinal: float
    item_values: dict
    context_features: dict = field(default_factory=dict)
    target_y: float = 0.0


def _events(n: int = 120, seed: int = 0) -> list[_Event]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        values = {q: float(rng.integers(0, 4)) for q in _ITEMS}
        out.append(
            _Event(
                f"s{i % 6}",
                float(i),
                values,
                {"ctx": rng.normal()},
                sum(values.values()) + rng.normal(0, 0.2),
            )
        )
    return out


def _mask_aware(seed: int) -> MaskAwareEMAModel:
    return MaskAwareEMAModel(_ITEMS, n_dropout_augmentations=2, random_state=seed)


def test_mask_aware_model_is_deterministic_for_a_fixed_seed():
    train, test = _events(seed=1), _events(40, seed=2)
    asked = [set(_ITEMS[:2])] * len(test)
    a, b = _mask_aware(0), _mask_aware(0)
    a.fit(train)
    b.fit(train)
    np.testing.assert_array_equal(a.predict(test, asked), b.predict(test, asked))


def test_mask_aware_model_uses_its_seed_for_item_dropout():
    """The dropout masks are drawn from the seeded generator, so the
    augmented training set differs between seeds."""
    train = _events(seed=1)
    a, b = _mask_aware(0), _mask_aware(1)
    a.fit(train)
    b.fit(train)
    test = _events(40, seed=2)
    asked = [set(_ITEMS[:1])] * len(test)
    assert not np.array_equal(a.predict(test, asked), b.predict(test, asked))


# --------------------------------------------------------------------------
# Online belief estimators: update(observations) / predict(features)
# --------------------------------------------------------------------------

_FEATURE = SubjectFeature(
    group_onehot=(),
    rounds_since_check=0,
    last_observed_state=0.0,
    round_idx_norm=0.0,
    check_count_so_far=0,
)


def _observation_stream(seed: int = 0, rounds: int = 30, subjects: int = 5):
    rng = np.random.default_rng(seed)
    stream = []
    for t in range(rounds):
        stream.append(
            [
                Observation(
                    round_idx=t,
                    subject_id=int(s),
                    group="g",
                    feature=_FEATURE,
                    observed_state=float(rng.normal()),
                    true_state_at_check=0.0,
                )
                for s in rng.choice(subjects, size=2, replace=False)
            ]
        )
    return stream


@pytest.mark.parametrize(
    "factory",
    [
        lambda: NaiveEMAEstimator(alpha=0.3),
        lambda: KalmanNeedEstimator(q=0.05, r=1.0),
    ],
)
def test_estimator_is_a_pure_function_of_its_observation_history(factory):
    features = {s: _FEATURE for s in range(5)}
    a, b = factory(), factory()
    for batch in _observation_stream():
        a.update(batch)
        b.update(batch)
        assert a.predict(features) == b.predict(features)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: NaiveEMAEstimator(alpha=0.3),
        lambda: KalmanNeedEstimator(q=0.05, r=1.0),
    ],
)
def test_estimator_predict_does_not_change_state(factory):
    features = {s: _FEATURE for s in range(5)}
    est = factory()
    for batch in _observation_stream():
        est.update(batch)
    assert est.predict(features) == est.predict(features)


@pytest.mark.parametrize("factory", [Persistence, ExponentialMovingAverage])
def test_public_reference_estimator_resets_to_an_identical_state(factory):
    stream = [
        (s, t, v)
        for t, batch in enumerate(_observation_stream())
        for s, v in ((o.subject_id, o.observed_state) for o in batch)
    ]
    est = factory()
    est.reset()
    for s, t, v in stream:
        est.observe(s, t, v)
    first = {s: est.score(s, 0) for s in {s for s, _, _ in stream}}
    est.reset()
    for s, t, v in stream:
        est.observe(s, t, v)
    assert first == {s: est.score(s, 0) for s in first}


# --------------------------------------------------------------------------
# Stochastic review policies: select(..., rng)
# --------------------------------------------------------------------------


def _select(policy, seed: int):
    features = {s: _FEATURE for s in range(12)}
    need = {s: float(s % 4) for s in features}  # many exact ties on purpose
    rng = np.random.default_rng(seed)
    return [tuple(policy.select(features, need, 3, r, rng)) for r in range(6)]


@pytest.mark.parametrize("policy_cls", [RandomPolicy, ShuffledRoundRobinPolicy, UCBExplorePolicy])
def test_stochastic_policy_is_deterministic_given_the_generator_seed(policy_cls):
    assert _select(policy_cls(), 7) == _select(policy_cls(), 7)


@pytest.mark.parametrize("policy_cls", [RandomPolicy, ShuffledRoundRobinPolicy, UCBExplorePolicy])
def test_stochastic_policy_depends_on_the_generator_seed(policy_cls):
    assert _select(policy_cls(), 7) != _select(policy_cls(), 8)
