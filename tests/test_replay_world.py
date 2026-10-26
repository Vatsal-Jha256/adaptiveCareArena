"""Tests for the real-data replay world. The properties that matter are
that it never invents data and that restlessness is genuine."""

import numpy as np

from stage2_env.core import SubjectSpec
from stage2_env.replay_world import ReplaySubjectWorld

GROUPS = {"low_baseline": {}, "mid_baseline": {}, "high_baseline": {}}


def _world(carry=True):
    specs = [SubjectSpec(subject_id=i, group="mid_baseline", baseline_offset=0.0) for i in range(3)]
    # Sparse availability, exactly like real EMA every 2-3 days.
    traj = {
        0: [1.0, None, None, 4.0, None],
        1: [None, 2.0, None, None, 5.0],
        2: [0.0, None, 3.0, None, None],
    }
    return ReplaySubjectWorld(
        specs, GROUPS, traj, np.random.default_rng(0), carry_forward_state=carry
    )


def test_available_reports_only_real_recorded_days():
    w = _world()
    assert w.available(0) == {0, 2}
    assert w.available(1) == {1}
    assert w.available(2) == {2}
    assert w.available(4) == {1}


def test_checking_a_subject_with_no_ema_yields_no_observation():
    # Nothing was recorded that day, so there is nothing to reveal.
    # A reviewer who opens that chart learned nothing, and the
    # wasted slot must not be papered over with an invented value.
    w = _world()
    feats = w.current_features(1, 5, {})
    obs, _ = w.pull_and_advance(1, {0, 1, 2}, feats)
    assert [o.subject_id for o in obs] == [1]


def test_observed_values_are_exactly_the_recorded_ones():
    w = _world()
    obs, _ = w.pull_and_advance(0, {0, 2}, w.current_features(0, 5, {}))
    got = {o.subject_id: o.observed_state for o in obs}
    assert got == {0: 1.0, 2: 0.0}


def test_state_moves_while_unobserved_without_any_simulation():
    # Subject 0 is 1.0 at round 0 and 4.0 at round 3. If we only look at
    # round 0, we never see the move -- that IS the restless property,
    # and it comes from the recording, not from a noise model.
    w = _world()
    w.pull_and_advance(0, {0}, w.current_features(0, 5, {}))
    assert w.last_observed_state[0] == 1.0
    for r in (1, 2):
        w.pull_and_advance(r, set(), w.current_features(r, 5, {}))
    assert w.last_observed_state[0] == 1.0, "no real EMA arrived at rounds 1-2"
    assert w.true_state(0, 3) == 4.0, "reality moved on"


def test_belief_updates_without_review_but_attention_clock_does_not():
    # Regression test for the observation/review conflation: subject 0
    # has a real recorded EMA at round 3 (value 4.0) but is never
    # reviewed (checked_subject_ids is always empty). The automated
    # belief must still update -- CrossCheck's EMA is digital and
    # reaches the record whether or not a human opens the chart -- while
    # rounds_since_check (human attention recency) must keep climbing
    # regardless, because nobody actually looked. Before this fix,
    # `last_observed_state` only updated on review, so an unreviewed
    # subject looked "stale" purely because the simulator withheld data
    # from the model whenever a policy declined to check them.
    w = _world()
    for r in range(4):
        w.pull_and_advance(r, set(), w.current_features(r, 5, {}))  # never review anyone
    assert w.last_observed_state[0] == 4.0, "real EMA must be auto-ingested without review"
    assert w.rounds_since_check[0] == 4, "attention recency must not be reset by auto-ingestion"


def test_rounds_since_check_tracks_staleness():
    w = _world()
    w.pull_and_advance(0, {0}, w.current_features(0, 5, {}))
    assert w.rounds_since_check[0] == 0
    for r in (1, 2, 3):
        w.pull_and_advance(r, set(), w.current_features(r, 5, {}))
    assert w.rounds_since_check[0] == 3


def test_no_carry_forward_returns_none_between_recordings():
    w = _world(carry=False)
    w.true_state(0, 0)
    assert w.true_state(0, 1) is None, "must not invent a value for an unmeasured day"


def test_carry_forward_is_persistence_not_interpolation():
    # Carrying the last value forward is an assumption; it must never
    # peek at the next recorded value (that would be interpolation, and
    # would leak the future into the present).
    w = _world(carry=True)
    w.true_state(0, 0)
    assert w.true_state(0, 1) == 1.0  # last known, not 4.0 and not a blend
    assert w.true_state(0, 2) == 1.0
