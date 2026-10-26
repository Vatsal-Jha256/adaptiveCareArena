"""Tests for stage2_env -- the budgeted acquisition-timing environment.
status: infrastructure only (see stage2_env/core.py); these tests check
the mechanics are correct, not that any result here is a scientific
claim about CrossCheck's real acquisition-timing tradeoff."""

import numpy as np

from stage2_env.core import ESTIMATORS, POLICIES, Observation, SubjectFeature
from stage2_env.estimators import NaiveEMAEstimator
from stage2_env.policies import GreedyNeedPolicy, RandomPolicy, RoundRobinPolicy
from tests.fixtures.toy_restless_world import GroupSpec, RestlessSubjectWorld, build_subjects
from tests.fixtures.toy_stage2_runner import run_experiment


def _simple_groups():
    return [
        GroupSpec(
            name="low",
            weight=0.5,
            mu=-5.0,
            reversion=0.2,
            noise_std=0.5,
            unobserved_extra_noise_std=0.1,
            subject_std=0.0,
        ),
        GroupSpec(
            name="high",
            weight=0.5,
            mu=5.0,
            reversion=0.2,
            noise_std=0.5,
            unobserved_extra_noise_std=0.1,
            subject_std=0.0,
        ),
    ]


def test_world_never_leaks_true_state_into_features():
    rng = np.random.default_rng(0)
    groups = _simple_groups()
    specs = build_subjects(groups, 10, rng)
    world = RestlessSubjectWorld(specs, {g.name: g for g in groups}, rng)
    feats = world.current_features(0, 10, {s.subject_id: 0 for s in specs})
    for f in feats.values():
        assert not hasattr(f, "true_state")


def test_checked_subject_reveals_exact_current_state():
    rng = np.random.default_rng(1)
    groups = _simple_groups()
    specs = build_subjects(groups, 5, rng)
    world = RestlessSubjectWorld(specs, {g.name: g for g in groups}, rng)
    feats = world.current_features(0, 10, {s.subject_id: 0 for s in specs})
    before = world.true_state_snapshot()
    checked = {specs[0].subject_id}
    observations, _true_state_now = world.pull_and_advance(0, checked, feats)
    assert len(observations) == 1
    obs = observations[0]
    assert obs.observed_state == obs.true_state_at_check == before[specs[0].subject_id]


def test_unchecked_subjects_still_evolve_restless():
    rng = np.random.default_rng(2)
    groups = _simple_groups()
    specs = build_subjects(groups, 5, rng)
    world = RestlessSubjectWorld(specs, {g.name: g for g in groups}, rng)
    before = dict(world.true_state_snapshot())
    for r in range(10):
        feats = world.current_features(r, 10, {s.subject_id: 0 for s in specs})
        world.pull_and_advance(r, set(), feats)  # never check anyone
    after = world.true_state_snapshot()
    # At least one unchecked subject's state must have moved (restless).
    assert any(abs(after[sid] - before[sid]) > 1e-9 for sid in before)


def test_greedy_need_policy_prioritizes_lower_predicted_state():
    rng = np.random.default_rng(3)
    features = {
        1: SubjectFeature(
            group_onehot=(1.0,),
            rounds_since_check=0,
            last_observed_state=0.0,
            round_idx_norm=0.0,
            check_count_so_far=0,
        ),
        2: SubjectFeature(
            group_onehot=(1.0,),
            rounds_since_check=0,
            last_observed_state=0.0,
            round_idx_norm=0.0,
            check_count_so_far=0,
        ),
    }
    predicted_need = {1: 10.0, 2: 1.0}  # subject 1 predicted much worse off
    policy = GreedyNeedPolicy()
    selected = policy.select(features, predicted_need, budget=1, round_idx=0, rng=rng)
    assert selected == {1}


def test_random_and_round_robin_respect_budget():
    rng = np.random.default_rng(4)
    features = {
        i: SubjectFeature(
            group_onehot=(1.0,),
            rounds_since_check=0,
            last_observed_state=0.0,
            round_idx_norm=0.0,
            check_count_so_far=0,
        )
        for i in range(10)
    }
    for policy in (RandomPolicy(), RoundRobinPolicy()):
        selected = policy.select(features, {}, budget=3, round_idx=0, rng=rng)
        assert len(selected) == 3
        assert selected.issubset(set(features))


def test_naive_ema_estimator_predicts_higher_need_for_lower_observed_state():
    est = NaiveEMAEstimator(alpha=0.5, prior_need=0.0)
    obs_low = Observation(
        round_idx=0,
        subject_id=1,
        group="g",
        feature=SubjectFeature((1.0,), 0, 0.0, 0.0, 0),
        observed_state=-10.0,
        true_state_at_check=-10.0,
    )
    obs_high = Observation(
        round_idx=0,
        subject_id=2,
        group="g",
        feature=SubjectFeature((1.0,), 0, 0.0, 0.0, 0),
        observed_state=10.0,
        true_state_at_check=10.0,
    )
    est.update([obs_low, obs_high])
    preds = est.predict({1: None, 2: None})
    assert preds[1] > preds[2]


def test_registries_have_the_expected_policies_and_estimators():
    assert {
        "random",
        "round_robin",
        "greedy_need",
        "ucb_explore",
        "fair_floor",
        "whittle_index",
    } <= set(POLICIES.names())
    assert "naive_ema" in ESTIMATORS.names()


def test_run_experiment_end_to_end_smoke():
    cfg = {
        "seed": 0,
        "world": {
            "n_subjects": 20,
            "n_rounds": 15,
            "groups": [
                {
                    "name": "low",
                    "weight": 0.5,
                    "mu": -5.0,
                    "reversion": 0.2,
                    "noise_std": 0.5,
                    "unobserved_extra_noise_std": 0.1,
                    "subject_std": 1.0,
                },
                {
                    "name": "high",
                    "weight": 0.5,
                    "mu": 5.0,
                    "reversion": 0.2,
                    "noise_std": 0.5,
                    "unobserved_extra_noise_std": 0.1,
                    "subject_std": 1.0,
                },
            ],
        },
        "budget": {"b_per_round": 5},
        "estimator": {"name": "naive_ema", "params": {"alpha": 0.3, "prior_need": 0.0}},
        "policy": {"name": "greedy_need", "params": {}},
    }
    metrics = run_experiment(cfg)
    assert metrics["n_subjects"] == 20
    assert metrics["n_rounds"] == 15
    assert "cumulative_regret_vs_oracle" in metrics
    assert "allocation_fairness" in metrics
    assert "information_fairness" in metrics


def test_all_registered_policies_run_without_error():
    cfg_base = {
        "seed": 0,
        "world": {
            "n_subjects": 15,
            "n_rounds": 10,
            "groups": [
                {
                    "name": "low",
                    "weight": 0.5,
                    "mu": -5.0,
                    "reversion": 0.2,
                    "noise_std": 0.5,
                    "unobserved_extra_noise_std": 0.1,
                    "subject_std": 1.0,
                },
                {
                    "name": "high",
                    "weight": 0.5,
                    "mu": 5.0,
                    "reversion": 0.2,
                    "noise_std": 0.5,
                    "unobserved_extra_noise_std": 0.1,
                    "subject_std": 1.0,
                },
            ],
        },
        "budget": {"b_per_round": 4},
        "estimator": {"name": "naive_ema", "params": {}},
    }
    for policy_name in POLICIES.names():
        cfg = dict(cfg_base, policy={"name": policy_name, "params": {}})
        metrics = run_experiment(cfg)
        assert metrics["n_subjects"] == 15
