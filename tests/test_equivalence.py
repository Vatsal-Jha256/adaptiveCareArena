"""Tests for the TOST equivalence test. It exists because overlapping
error bars were being used to argue equivalence, which they do not
show."""

import pytest

from eval.metrics import equivalence_test


def test_identical_arms_are_equivalent_within_margin():
    r = equivalence_test([10, 10.1, 9.9, 10, 10], [10, 10, 10.1, 9.9, 10], margin=2.0)
    assert r["verdict"] == "equivalent"
    assert -2.0 < r["ci_lo"] and r["ci_hi"] < 2.0


def test_constant_separation_is_reported_as_deterministic():
    r = equivalence_test([10, 10, 10, 10, 10.1], [20, 20, 20, 20, 20.1], margin=2.0)
    assert r["verdict"] == "deterministic"
    assert r["deterministic_relation"] == "outside_margin"


def test_noisy_underpowered_data_is_inconclusive_not_equivalent():
    # The point of the test. Wide, overlapping intervals must not be
    # reported as equivalence -- that is the mistake this replaces.
    r = equivalence_test([10, 20, 0, 15, 5], [12, 20, 4, 13, 1], margin=1.0)
    assert r["verdict"] == "inconclusive"


def test_zero_variance_is_reported_as_deterministic_not_equivalent():
    # A degenerate denominator must not manufacture certainty -- the same
    # trap that made a 10-event difference look "significant" under
    # calendar alignment, and that made CV explode near zero.
    close = equivalence_test([5] * 5, [5] * 5, margin=1.0)
    far = equivalence_test([5] * 5, [50] * 5, margin=1.0)
    assert close["verdict"] == "deterministic"
    assert far["verdict"] == "deterministic"
    assert close["deterministic_relation"] == "identical"
    assert far["deterministic_relation"] == "outside_margin"
    assert "zero variance" in close["note"]


def test_margin_is_required_and_must_be_positive():
    # A margin chosen after seeing the result can make anything
    # "equivalent", so there is deliberately no default.
    with pytest.raises(TypeError):
        equivalence_test([1, 2], [1, 2])  # margin not supplied
    with pytest.raises(ValueError):
        equivalence_test([1, 2], [1, 2], margin=0)


def test_too_few_observations_raises_rather_than_guessing():
    with pytest.raises(ValueError):
        equivalence_test([1.0], [2.0], margin=1.0)


def test_pairing_is_required_and_used():
    # Marginal arm variance is large, but every paired difference is small.
    r = equivalence_test([0, 100, 200, 300], [0.1, 100.1, 200.1, 299.9], margin=0.5)
    assert r["verdict"] == "equivalent"
    with pytest.raises(ValueError):
        equivalence_test([1, 2, 3], [1, 2], margin=1.0)
