"""Tests for KalmanNeedEstimator (local-level Kalman filter over
observed need, stage2_env/estimators.py). The properties that matter:
uncertainty grows for an unobserved subject and shrinks on observation,
the gain therefore adapts with elapsed time (unlike NaiveEMAEstimator's
fixed alpha), and a never-observed subject falls back to the prior."""

from __future__ import annotations

from stage2_env.core import Observation, SubjectFeature
from stage2_env.estimators import KalmanNeedEstimator

FEAT = SubjectFeature(
    group_onehot=(),
    rounds_since_check=0,
    last_observed_state=0.0,
    round_idx_norm=0.0,
    check_count_so_far=0,
)


def _obs(round_idx: int, subject_id: int, value: float) -> Observation:
    return Observation(
        round_idx=round_idx,
        subject_id=subject_id,
        group="g",
        feature=FEAT,
        observed_state=value,
        true_state_at_check=value,
    )


def test_never_observed_subject_predicts_the_prior():
    est = KalmanNeedEstimator(prior_need=0.0)
    assert est.predict({1: FEAT}) == {1: 0.0}


def test_variance_grows_without_observation_and_shrinks_on_observation():
    est = KalmanNeedEstimator(q=0.1, r=1.0, prior_var=1.0)
    est.update([_obs(0, 1, -2.0)])
    var_after_obs = est._var[1]
    # Three rounds pass with no new observation for subject 1.
    est.update([])
    est.update([])
    est.update([])
    assert est._var[1] > var_after_obs, "uncertainty must grow while unobserved"


def test_gain_adapts_with_elapsed_time_unlike_fixed_alpha_ema():
    """A subject observed every round vs. one observed only after a long
    gap should not get the same correction weight on their next real
    observation -- that adaptive gain is the point of the Kalman filter over
    NaiveEMAEstimator's constant alpha."""
    frequent = KalmanNeedEstimator(q=0.1, r=1.0)
    rare = KalmanNeedEstimator(q=0.1, r=1.0)
    for r in range(5):
        frequent.update([_obs(r, 1, -1.0)])
    rare.update([_obs(0, 1, -1.0)])
    for _ in range(1, 5):
        rare.update([])  # no observation: uncertainty accumulates

    var_frequent = frequent._var[1]
    var_rare = rare._var[1]
    assert var_rare > var_frequent, "a long-unobserved subject must carry more uncertainty"

    mean0_frequent, mean0_rare = frequent._mean[1], rare._mean[1]
    # Same new observation to both: the rarely-observed (higher-variance)
    # filter must move further from where it stood than the frequently-
    # observed one -- that displacement, not the raw sign, is the gain.
    frequent.update([_obs(5, 1, 3.0)])
    rare.update([_obs(5, 1, 3.0)])
    moved_frequent = abs(frequent._mean[1] - mean0_frequent)
    moved_rare = abs(rare._mean[1] - mean0_rare)
    assert moved_rare > moved_frequent


def test_predict_before_update_matches_replay_experiment_call_order():
    """replay_experiment.run_one_policy calls predict() then update() each
    round; predict() must use the PRE-update state, never leak this
    round's own observation into its own prediction."""
    est = KalmanNeedEstimator(q=0.1, r=1.0)
    est.update([_obs(0, 1, 5.0)])  # observed_state=5.0 -> need=-5.0
    pred_before = est.predict({1: FEAT})[1]
    assert abs(pred_before - (-5.0)) < abs(pred_before - 0.0), (
        "should move toward the observed need, not stay at the prior"
    )
    est.update([_obs(1, 1, -5.0)])  # observed_state=-5.0 -> need=+5.0
    pred_after = est.predict({1: FEAT})[1]
    assert pred_before != pred_after
    assert pred_after > pred_before, (
        "a strongly opposite new observation must pull the mean back up"
    )
