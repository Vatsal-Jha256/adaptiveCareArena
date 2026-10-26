"""Offline tests for the item-selection environment. Synthetic events --
no CrossCheck download needed."""

import numpy as np
import pytest

from emaselect.dataset import EMAEvent, assert_no_item_leakage
from emaselect.items import (
    BURDEN_PROFILES,
    EMA_ITEMS,
    ITEM_SIGN,
    NEGATIVE_ITEMS,
    POSITIVE_ITEMS,
    ema_score_from_items,
)
from emaselect.model import MaskAwareEMAModel
from emaselect.policies import (
    AskAllPolicy,
    AskNonePolicy,
    FixedSubsetPolicy,
    GreedyVOIPolicy,
    RandomMatchedPolicy,
    ThompsonItemPolicy,
)


def _events(n=300, seed=0):
    """ema_STRESSED carries real signal about the target; ema_SLEEPING is
    pure noise. A working policy should prefer the former."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        vals = {it: float(rng.integers(0, 4)) for it in EMA_ITEMS}
        target = 3.0 * vals["ema_STRESSED"] + rng.normal(0, 0.3)
        out.append(
            EMAEvent(
                f"s{i % 12}",
                float(i),
                vals,
                {"passive_a": rng.normal(), "passive_b": rng.normal()},
                target,
            )
        )
    return out


def test_item_registry_is_complete_and_signed():
    assert len(EMA_ITEMS) == 10
    assert set(POSITIVE_ITEMS) | set(NEGATIVE_ITEMS) == set(EMA_ITEMS)
    assert not set(POSITIVE_ITEMS) & set(NEGATIVE_ITEMS)
    assert all(ITEM_SIGN[i] in (+1, -1) for i in EMA_ITEMS)


def test_scoring_rule_matches_the_released_definition():
    # Best possible state: every positive item maxed, every negative at 0.
    best = {**{i: 3.0 for i in POSITIVE_ITEMS}, **{i: 0.0 for i in NEGATIVE_ITEMS}}
    worst = {**{i: 0.0 for i in POSITIVE_ITEMS}, **{i: 3.0 for i in NEGATIVE_ITEMS}}
    assert ema_score_from_items(best) == 15.0
    assert ema_score_from_items(worst) == -15.0
    assert ema_score_from_items({i: 0.0 for i in EMA_ITEMS}) == 0.0


def test_every_burden_profile_prices_every_item():
    for name, profile in BURDEN_PROFILES.items():
        assert set(profile) == set(EMA_ITEMS), f"{name} has missing/extra items"
        assert all(c > 0 for c in profile.values()), f"{name} has a free item"


def test_inverted_profile_really_does_reverse_the_sensitivity_ordering():
    # It is an adversarial control; if it accidentally agreed with the
    # sensitivity ordering it would control for nothing.
    sens = BURDEN_PROFILES["sensitivity_weighted"]
    inv = BURDEN_PROFILES["inverted_sensitivity"]
    # Compare the order relation, not a sorted list: several items share a
    # cost (think and stressed are both 1.0), and reversing a sorted list
    # reorders ties, which is an artifact rather than a property.
    for a in EMA_ITEMS:
        for b in EMA_ITEMS:
            if sens[a] < sens[b]:
                assert inv[a] > inv[b], f"{a} cheaper than {b} in both profiles"
            elif sens[a] == sens[b]:
                assert inv[a] == inv[b]


def test_leakage_guard_catches_an_item_in_the_free_context():
    bad = EMAEvent("s1", 0.0, {i: 1.0 for i in EMA_ITEMS}, {"ema_VOICES": 2.0}, 0.0)
    with pytest.raises(ValueError, match="item leakage"):
        assert_no_item_leakage([bad])
    ok = EMAEvent("s1", 0.0, {i: 1.0 for i in EMA_ITEMS}, {"passive_a": 2.0}, 0.0)
    assert_no_item_leakage([ok])  # must not raise


def test_unasked_items_are_missing_not_zero():
    # Zero is a real response on the 0-3 scale ("not at all"), so an
    # unasked item imputed as 0 would tell the model the patient denied
    # the symptom -- the worst possible bug in a not-asking benchmark.
    ev = _events(n=20)
    m = MaskAwareEMAModel(EMA_ITEMS, random_state=0)
    m.fit(ev)
    X = m._to_matrix(ev, asked=[{"ema_CALM"}] * len(ev))
    cols = m._columns()
    for item in EMA_ITEMS:
        col = X[:, cols.index(item)]
        if item == "ema_CALM":
            assert not np.isnan(col).any()
        else:
            assert np.isnan(col).all(), f"{item} was not asked but is not NaN"


def test_asking_the_informative_item_beats_asking_the_noise_item():
    ev = _events()
    m = MaskAwareEMAModel(EMA_ITEMS, random_state=0)
    m.fit(ev)
    y = np.array([e.target_y for e in ev])

    def mae(asked):
        return float(np.mean(np.abs(m.predict(ev, [asked] * len(ev)) - y)))

    assert mae({"ema_STRESSED"}) < mae({"ema_SLEEPING"})


def test_ask_none_and_ask_all_are_the_two_burden_endpoints():
    ev = _events(n=50)
    m = MaskAwareEMAModel(EMA_ITEMS, random_state=0)
    m.fit(ev)
    profile = BURDEN_PROFILES["uniform"]
    rng = np.random.default_rng(0)
    none_asked, none_cost = AskNonePolicy().run(ev, m, 5.0, profile, rng)
    all_asked, all_cost = AskAllPolicy().run(ev, m, 5.0, profile, rng)
    assert none_cost == 0.0 and all(a == set() for a in none_asked)
    assert all_cost == sum(profile.values()) and all(a == set(EMA_ITEMS) for a in all_asked)


@pytest.mark.parametrize("budget", [1.0, 3.0, 7.0])
def test_no_policy_ever_exceeds_the_burden_budget(budget):
    ev = _events(n=60)
    m = MaskAwareEMAModel(EMA_ITEMS, random_state=0)
    m.fit(ev)
    profile = BURDEN_PROFILES["sensitivity_weighted"]
    y = np.array([e.target_y for e in ev])
    for pol in (RandomMatchedPolicy(), GreedyVOIPolicy(list(EMA_ITEMS)), ThompsonItemPolicy()):
        asked, _ = pol.run(ev, m, budget, profile, np.random.default_rng(0), y_true=y)
        for a in asked:
            assert sum(profile[i] for i in a) <= budget + 1e-9, f"{pol.name} overspent"


def test_fixed_subset_policy_asks_everyone_the_same_questions():
    ev = _events(n=30)
    m = MaskAwareEMAModel(EMA_ITEMS, random_state=0)
    m.fit(ev)
    pol = FixedSubsetPolicy({"ema_CALM", "ema_VOICES"})
    asked, cost = pol.run(ev, m, 5.0, BURDEN_PROFILES["uniform"], np.random.default_rng(0))
    assert all(a == {"ema_CALM", "ema_VOICES"} for a in asked)
    assert cost == 2.0


def test_thompson_requires_outcomes_and_learns_from_them():
    ev = _events(n=400)
    m = MaskAwareEMAModel(EMA_ITEMS, random_state=0)
    m.fit(ev)
    profile = BURDEN_PROFILES["uniform"]
    y = np.array([e.target_y for e in ev])

    with pytest.raises(ValueError):
        ThompsonItemPolicy().run(ev, m, 3.0, profile, np.random.default_rng(0))

    pol = ThompsonItemPolicy(tol=1.5)
    pol.run(ev, m, 3.0, profile, np.random.default_rng(0), y_true=y)
    posterior = {i: pol.alpha[i] / (pol.alpha[i] + pol.beta[i]) for i in EMA_ITEMS}
    # The informative item should end with a higher posterior mean than
    # the pure-noise one after learning from real outcomes.
    assert posterior["ema_STRESSED"] > posterior["ema_SLEEPING"]


def test_thompson_posterior_state_does_not_leak_between_runs():
    """ThompsonItemPolicy accumulates Beta posterior state across run()
    calls. A calibration pass that reuses an instance already run on test
    would let test labels shape which items get asked on the calibration
    split -- and therefore shape the uncertainty estimates the test
    predictions are judged by. A fresh instance must start from a flat
    prior."""
    ev = _events(n=200)
    m = MaskAwareEMAModel(EMA_ITEMS, random_state=0)
    m.fit(ev)
    profile = BURDEN_PROFILES["uniform"]
    y = np.array([e.target_y for e in ev])

    pol = ThompsonItemPolicy()
    pol.run(ev, m, 3.0, profile, np.random.default_rng(0), y_true=y)
    trained = dict(pol.alpha)
    # Running again continues from the trained posterior, by design.
    pol.run(ev, m, 3.0, profile, np.random.default_rng(0), y_true=y)
    assert pol.alpha != trained, "state should accumulate within one instance"

    fresh = ThompsonItemPolicy()
    assert fresh.alpha == {} and fresh.beta == {}
    fresh.run(ev, m, 3.0, profile, np.random.default_rng(0), y_true=y)
    # A fresh instance reproduces the first run exactly, proving the
    # calibration pass is uncontaminated by the earlier test-label run.
    assert fresh.alpha == trained
