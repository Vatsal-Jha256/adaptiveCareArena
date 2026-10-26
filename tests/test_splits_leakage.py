"""Regression tests for the target-event leakage an independent review
found in `future_time_split` (47 events / 4.26% of test rows shared
between train and test on the real CrossCheck Stage 1 split)."""

from dataclasses import dataclass

from splits.splits import count_shared_target_events, future_time_split


@dataclass
class _Ex:
    subject_id: str
    prediction_time_t: float
    target_observed_at: float
    horizon: float = 1.0
    target_y: float = 0.0


def _multi_horizon_examples(n_events=10, horizons=(1.0, 2.0, 3.0), subjects=("a", "b")):
    """The exact shape that produced the bug: one target event reached
    from several prediction times at different horizons, so ordering rows
    by prediction_time_t splits an event across the cut."""
    out = []
    for sid in subjects:
        for k in range(n_events):
            observed = float(k)
            for h in horizons:
                out.append(
                    _Ex(sid, prediction_time_t=observed - h, target_observed_at=observed, horizon=h)
                )
    return out


def test_the_old_ungrouped_split_really_does_leak():
    # Guards the guard: if this ever stops leaking, the regression test
    # below is no longer testing anything.
    ex = _multi_horizon_examples()
    train, test = future_time_split(ex, train_fraction=0.7, group_by_target_event=False)
    assert count_shared_target_events(train, test) > 0


def test_event_grouped_split_shares_no_target_event():
    ex = _multi_horizon_examples()
    train, test = future_time_split(ex, train_fraction=0.7)
    assert count_shared_target_events(train, test) == 0
    assert train and test


def test_every_row_of_a_shared_event_lands_on_one_side():
    ex = _multi_horizon_examples()
    train, test = future_time_split(ex, train_fraction=0.7)
    train_events = {(e.subject_id, e.target_observed_at) for e in train}
    for e in test:
        assert (e.subject_id, e.target_observed_at) not in train_events


def test_split_stays_temporal_every_train_target_precedes_every_test_target():
    ex = _multi_horizon_examples()
    train, test = future_time_split(ex, train_fraction=0.7)
    for sid in {e.subject_id for e in ex}:
        tr = [e.target_observed_at for e in train if e.subject_id == sid]
        te = [e.target_observed_at for e in test if e.subject_id == sid]
        assert max(tr) < min(te), "a training target must never be observed after a test target"


def test_embargo_removes_training_rows_adjacent_to_the_test_period():
    ex = _multi_horizon_examples()
    no_embargo, test_a = future_time_split(ex, train_fraction=0.7, embargo=0.0)
    embargoed, test_b = future_time_split(ex, train_fraction=0.7, embargo=2.0)
    # The test side is unchanged; only training rows too close to it go.
    assert len(test_a) == len(test_b)
    assert len(embargoed) < len(no_embargo)
    test_start = min(e.target_observed_at for e in test_b if e.subject_id == "a")
    for e in embargoed:
        if e.subject_id == "a":
            assert e.target_observed_at < test_start - 2.0


def test_single_event_subject_is_not_split_across_sides():
    # One event cannot be split without putting the same label on both
    # sides -- it must go wholly to train, never silently into both.
    ex = [_Ex("solo", prediction_time_t=-h, target_observed_at=0.0, horizon=h) for h in (1.0, 2.0)]
    train, test = future_time_split(ex, train_fraction=0.7)
    assert len(train) == 2 and test == []
    assert count_shared_target_events(train, test) == 0
