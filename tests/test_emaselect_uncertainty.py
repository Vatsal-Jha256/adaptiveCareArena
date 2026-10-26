"""Tests for the selective-prediction layer. The properties that matter
are coverage validity and whether the uncertainty signal is actually
useful for abstention -- not any particular MAE."""

import numpy as np
import pytest

from emaselect.dataset import EMAEvent
from emaselect.items import EMA_ITEMS
from emaselect.model import MaskAwareEMAModel
from emaselect.uncertainty import LocallyAdaptiveConformal, aurc, risk_coverage_curve


def _heteroscedastic_events(n=900, seed=0):
    """Noise scale depends on an observed context feature, so difficulty
    is predictable -- which is exactly what a locally adaptive
    method should exploit and a global-width one cannot."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        vals = {it: float(rng.integers(0, 4)) for it in EMA_ITEMS}
        hard = float(rng.integers(0, 2))  # 1 = noisy regime
        noise = rng.normal(0, 0.2 + 3.0 * hard)
        out.append(
            EMAEvent(
                f"s{i % 15}",
                float(i),
                vals,
                {"hard": hard, "extra": rng.normal()},
                3.0 * vals["ema_STRESSED"] + noise,
            )
        )
    return out


def _fit(events, seed=0):
    n = len(events)
    train, cal, test = events[: n // 2], events[n // 2 : 3 * n // 4], events[3 * n // 4 :]
    m = MaskAwareEMAModel(EMA_ITEMS, random_state=seed)
    m.fit(train)

    def asked(evs):
        return [set(EMA_ITEMS)] * len(evs)

    conf = LocallyAdaptiveConformal(alpha=0.1, random_state=seed)
    conf.calibrate(m, cal, asked(cal), np.array([e.target_y for e in cal]))
    return m, conf, test, asked


def test_intervals_achieve_approximately_nominal_coverage():
    m, conf, test, asked = _fit(_heteroscedastic_events())
    y = np.array([e.target_y for e in test])
    pred = m.predict(test, asked(test))
    hw = conf.interval_halfwidth(m, test, asked(test))
    covered = np.mean(np.abs(y - pred) <= hw)
    # Split conformal targets 90%; allow real finite-sample slack.
    assert 0.78 <= covered <= 0.99, f"coverage {covered:.3f} far from nominal 0.90"


def test_interval_widths_actually_vary_with_difficulty():
    # The whole point of the locally adaptive construction: plain split
    # conformal gives one global width, which cannot support abstention.
    m, conf, test, asked = _fit(_heteroscedastic_events())
    hw = conf.interval_halfwidth(m, test, asked(test))
    assert hw.std() > 1e-3, "widths are constant -- abstention has nothing to rank on"
    hard = np.array([e.context_features["hard"] for e in test])
    assert hw[hard == 1].mean() > hw[hard == 0].mean(), (
        "wider intervals must land on the noisy regime"
    )


def test_abstaining_on_the_widest_intervals_reduces_error():
    m, conf, test, asked = _fit(_heteroscedastic_events())
    y = np.array([e.target_y for e in test])
    pred = m.predict(test, asked(test))
    hw = conf.interval_halfwidth(m, test, asked(test))
    curve = risk_coverage_curve(y, pred, hw)
    full = next(c for c in curve if c["coverage"] == 1.0)["mae"]
    half = min(curve, key=lambda c: c["coverage"])["mae"]
    assert half < full, "uncertainty signal is not usable for abstention"


def test_random_uncertainty_is_worse_than_the_real_signal_by_aurc():
    # Guards the guard: if a random ranking scored as well, the test
    # above would be measuring nothing.
    m, conf, test, asked = _fit(_heteroscedastic_events())
    y = np.array([e.target_y for e in test])
    pred = m.predict(test, asked(test))
    hw = conf.interval_halfwidth(m, test, asked(test))
    rng = np.random.default_rng(0)
    assert aurc(risk_coverage_curve(y, pred, hw)) < aurc(
        risk_coverage_curve(y, pred, rng.permutation(hw))
    )


def test_risk_coverage_curve_is_well_formed():
    curve = risk_coverage_curve(np.arange(100.0), np.zeros(100), np.arange(100.0))
    assert [c["coverage"] for c in curve] == sorted([c["coverage"] for c in curve], reverse=True)
    assert all(0 < c["coverage"] <= 1.0 for c in curve)
    assert all(c["n_retained"] >= 1 for c in curve)


def test_halfwidth_before_calibration_raises():
    m = MaskAwareEMAModel(EMA_ITEMS, random_state=0)
    ev = _heteroscedastic_events(n=40)
    m.fit(ev)
    with pytest.raises(RuntimeError):
        LocallyAdaptiveConformal().interval_halfwidth(m, ev, [set(EMA_ITEMS)] * len(ev))
