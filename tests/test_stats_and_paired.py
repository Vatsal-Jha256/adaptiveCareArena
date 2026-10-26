"""Tests for eval/paired.py and eval/stats.py.

The properties that matter are the ones a summary can get wrong silently:
the direction of a verdict on an error metric, the degenerate zero-variance
case, and the three-way TOST outcome. Each of these has to be able to say
"inconclusive" or "deterministic" distinctly from a substantive result.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from eval.paired import paired_policy_delta, pairing_gain, seeds_needed
from eval.stats import (
    _t_cdf,
    _t_ppf,
    benjamini_hochberg,
    bootstrap_paired_ci,
    holm_bonferroni,
    summarise_family,
    tost_equivalence,
)

# --------------------------------------------------------------------------
# paired_policy_delta
# --------------------------------------------------------------------------


def test_paired_rejects_mismatched_or_too_short_input():
    with pytest.raises(ValueError):
        paired_policy_delta([1.0, 2.0], [1.0], higher_is_better=True)
    with pytest.raises(ValueError):
        paired_policy_delta([1.0], [2.0], higher_is_better=True)


def test_paired_noisy_difference_around_zero_is_inconclusive():
    a = [0.50, 0.40, 0.60, 0.45, 0.55]
    b = [0.45, 0.50, 0.55, 0.50, 0.50]
    assert paired_policy_delta(a, b, higher_is_better=True)["verdict"] == "inconclusive"


def test_paired_verdict_respects_metric_direction():
    a = [0.60, 0.62, 0.59, 0.63]
    b = [0.50, 0.51, 0.49, 0.51]
    assert paired_policy_delta(a, b, True, "A", "B")["verdict"] == "A better"
    # The same numbers on an error metric mean A is worse.
    assert paired_policy_delta(a, b, False, "A", "B")["verdict"] == "B better"


def test_paired_constant_gap_is_deterministic_and_direction_aware():
    a = [0.7, 0.8, 0.9, 1.0]
    b = [x - 0.5 for x in a]
    higher = paired_policy_delta(a, b, True, "A", "B")
    assert higher["effect_sd"] == float("inf")
    assert higher["verdict"] == "A better (deterministic)"
    lower = paired_policy_delta(a, b, False, "A", "B")
    assert lower["verdict"] == "B better (deterministic)"


def test_paired_identical_series_are_reported_as_identical():
    a = [0.1, 0.2, 0.3]
    assert paired_policy_delta(a, list(a), True)["verdict"] == "identical"


def test_pairing_gain_is_large_when_arms_share_a_nuisance_term():
    rng = np.random.default_rng(0)
    shared = rng.normal(0, 1.0, 12)  # "which subjects landed in test"
    a = list(shared + 0.30 + rng.normal(0, 0.02, 12))
    b = list(shared + rng.normal(0, 0.02, 12))
    result = paired_policy_delta(a, b, True)
    assert pairing_gain(result) > 5.0


def test_seeds_needed_scales_with_the_inverse_square_of_the_effect():
    result = {"effect_sd": 1.0, "n_seeds": 5}
    assert seeds_needed(result, target_effect_sd=2.0) == 20
    # Already past the target: never asks for fewer seeds than were run.
    assert seeds_needed({"effect_sd": 4.0, "n_seeds": 5}, 2.0) == 5
    # A deterministic or zero effect cannot be priced in seeds.
    assert seeds_needed({"effect_sd": float("inf"), "n_seeds": 5}) == 5
    assert seeds_needed({"effect_sd": 0, "n_seeds": 5}) == 5


# --------------------------------------------------------------------------
# multiple-comparison corrections
# --------------------------------------------------------------------------

_P = {"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005}


def test_holm_stops_at_the_first_failure():
    out = holm_bonferroni(_P, alpha=0.05)
    assert [k for k, v in out.items() if v["significant"]] == ["d", "a"]
    # b has p=0.04 <= 0.05 but comes after the first failure (c), so it stays accepted.
    assert not out["b"]["significant"]


def test_benjamini_hochberg_rejects_everything_up_to_the_largest_passing_rank():
    out = benjamini_hochberg(_P, alpha=0.05)
    assert all(v["significant"] for v in out.values())


def test_holm_is_never_less_conservative_than_benjamini_hochberg():
    rng = np.random.default_rng(1)
    p = {f"c{i}": float(x) for i, x in enumerate(rng.uniform(0, 0.2, 15))}
    holm = sum(v["significant"] for v in holm_bonferroni(p).values())
    bh = sum(v["significant"] for v in benjamini_hochberg(p).values())
    assert holm <= bh


# --------------------------------------------------------------------------
# bootstrap_paired_ci / summarise_family
# --------------------------------------------------------------------------


def test_bootstrap_ci_needs_three_seeds_and_matched_lengths():
    with pytest.raises(ValueError):
        bootstrap_paired_ci([1.0, 2.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        bootstrap_paired_ci([1.0, 2.0, 3.0], [1.0, 2.0])


def test_bootstrap_ci_is_reproducible_for_a_seed_and_never_reports_p_zero():
    a = [1.0, 1.1, 0.9, 1.05, 0.95, 1.0]
    b = [0.0, 0.1, -0.1, 0.05, -0.05, 0.0]
    first = bootstrap_paired_ci(a, b, n_boot=500, seed=3)
    assert first == bootstrap_paired_ci(a, b, n_boot=500, seed=3)
    assert first["p_value"] >= 1.0 / 500
    assert first["excludes_zero"]
    assert first["ci_lo"] <= first["mean_diff"] <= first["ci_hi"]


def test_summarise_family_counts_what_correction_removed():
    rng = np.random.default_rng(0)
    comparisons = {
        "real": ([1.0 + x for x in rng.normal(0, 0.05, 10)], [0.0] * 10),
        **{f"null{i}": (list(rng.normal(0, 1, 10)), list(rng.normal(0, 1, 10))) for i in range(8)},
    }
    out = summarise_family(comparisons, higher_is_better=True, n_boot=400)
    assert out["n_comparisons"] == 9
    assert (
        out["n_lost_to_correction"] == out["n_significant_naive"] - out["n_significant_corrected"]
    )
    assert out["comparisons"]["real"]["verdict"] == "A better"


def test_summarise_family_rejects_an_unknown_method():
    with pytest.raises(ValueError):
        summarise_family({"x": ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])}, True, method="bonferroni")


# --------------------------------------------------------------------------
# tost_equivalence
# --------------------------------------------------------------------------


def test_tost_requires_a_positive_margin_and_three_pairs():
    with pytest.raises(ValueError):
        tost_equivalence([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], margin=0.0)
    with pytest.raises(ValueError):
        tost_equivalence([1.0, 2.0], [1.0, 2.0], margin=0.1)


def test_tost_declares_equivalence_for_a_tight_difference_inside_the_margin():
    rng = np.random.default_rng(0)
    a = list(rng.normal(0, 0.01, 30))
    out = tost_equivalence(a, [0.0] * 30, margin=0.5)
    assert out["equivalent"]
    assert out["outcome"].startswith("EQUIVALENT")


def test_tost_is_inconclusive_when_underpowered():
    out = tost_equivalence([0.1, -0.3, 0.4, -0.2], [0.0] * 4, margin=0.1)
    assert not out["equivalent"] and not out["different"]
    assert out["outcome"].startswith("INCONCLUSIVE")


def test_tost_declares_difference_for_a_large_consistent_gap():
    rng = np.random.default_rng(0)
    a = list(1.0 + rng.normal(0, 0.05, 20))
    out = tost_equivalence(a, [0.0] * 20, margin=0.1)
    assert out["different"] and not out["equivalent"]


def test_tost_zero_variance_cases_are_distinct_outcomes():
    same = tost_equivalence([0.5, 0.6, 0.7], [0.5, 0.6, 0.7], margin=0.1)
    assert same["outcome"].startswith("IDENTICAL")
    inside = tost_equivalence([0.55, 0.65, 0.75], [0.5, 0.6, 0.7], margin=0.1)
    assert inside["outcome"].startswith("EQUIVALENT exactly")
    outside = tost_equivalence([1.5, 1.6, 1.7], [0.5, 0.6, 0.7], margin=0.1)
    assert outside["outcome"].startswith("DIFFERENT exactly")


# --------------------------------------------------------------------------
# hand-rolled Student-t against scipy
# --------------------------------------------------------------------------


@pytest.mark.parametrize("df", [2, 5, 19, 100])
def test_t_cdf_and_ppf_match_scipy(df):
    scipy_stats = pytest.importorskip("scipy.stats")
    for t in (-4.0, -1.3, 0.0, 0.7, 2.5):
        assert math.isclose(_t_cdf(t, df), scipy_stats.t.cdf(t, df), abs_tol=1e-10)
    for p in (0.01, 0.5, 0.95):
        assert math.isclose(_t_ppf(p, df), scipy_stats.t.ppf(p, df), abs_tol=1e-6)
