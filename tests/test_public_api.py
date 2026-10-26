from __future__ import annotations

import pytest

import adaptivecarearena.arena as arena_module
from adaptivecarearena import (
    AdaptiveCareArena,
    ExponentialMovingAverage,
    MethodArm,
    Persistence,
)
from scripts.forward_benchmark import queue_bootstrap_budgets
from scripts.report_landmarks import build_report_landmarks


def _trajectories() -> dict[int, list[float | None]]:
    # Six legal history reports followed by one future threshold crossing.
    return {
        0: [0.0, 0.0, 0.0, 0.0, 0.0, -0.5, -2.0, None],
        1: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None],
        2: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, None],
    }


def _config() -> dict:
    return {
        "seed": 7,
        "data": {"source": "ces", "local_root": "unused"},
        "outcome": {"threshold": -1.0, "horizon_days": 1},
        "budgets": [1],
        "inference": {"n_boot": 40},
    }


def test_public_reference_estimators_reproduce_internal_scores(monkeypatch):
    trajectories = _trajectories()
    monkeypatch.setattr(
        arena_module,
        "load_panel",
        lambda cfg: ([], {}, trajectories, {}),
    )
    result = AdaptiveCareArena(_config()).evaluate(
        {
            "persistence": Persistence(),
            "ema_03": ExponentialMovingAverage(0.3),
        }
    )

    rows, _ = build_report_landmarks(
        trajectories,
        -1.0,
        1,
        None,
        risk="report",
        at_risk=True,
    )
    expected = queue_bootstrap_budgets(
        rows,
        sorted(trajectories),
        (1,),
        ("last", "ema3"),
        (("ema3", "last"),),
        40,
        7,
    )
    actual = result.summary["by_budget"]["1"]
    assert actual["episode_recall"]["persistence"] == expected["1"]["episode_recall"]["last"]
    assert actual["episode_recall"]["ema_03"] == expected["1"]["episode_recall"]["ema3"]
    assert actual["contrasts"]["ema_03-persistence"] == expected["1"]["contrasts"]["ema3-last"]
    assert result.summary["scope"].startswith("not diagnostic")


class _FutureSeekingEstimator:
    def reset(self):
        self.latest = {}

    def observe(self, subject_id, observed_at, value):
        self.latest[subject_id] = observed_at

    def score(self, subject_id, prediction_time):
        # The interface supplies no trajectory or target accessor. Verify the
        # latest observation never lies after the declared decision time.
        assert self.latest[subject_id] <= prediction_time
        return 0.0


def test_public_estimator_receives_only_decision_time_inputs(monkeypatch):
    monkeypatch.setattr(
        arena_module,
        "load_panel",
        lambda cfg: ([], {}, _trajectories(), {}),
    )
    result = AdaptiveCareArena(_config()).evaluate(
        {
            "safe_a": _FutureSeekingEstimator(),
            "safe_b": _FutureSeekingEstimator(),
        }
    )
    assert result.summary["status"] == "evaluated"


class _ReversePriority:
    def reset(self):
        pass

    def priority(self, subject_id, prediction_time, estimated_need):
        return -estimated_need


def test_external_review_policy_changes_priority_without_owning_capacity(monkeypatch):
    monkeypatch.setattr(
        arena_module,
        "load_panel",
        lambda cfg: ([], {}, _trajectories(), {}),
    )
    result = AdaptiveCareArena(_config()).evaluate(
        {
            "normal": MethodArm(Persistence()),
            "reversed": MethodArm(Persistence(), _ReversePriority()),
        }
    )
    by_budget = result.summary["by_budget"]["1"]
    assert by_budget["episode_recall"]["normal"] == 1.0
    assert by_budget["episode_recall"]["reversed"] == 0.0
    assert by_budget["burden"]["selected_landmarks"] == 2


def test_public_api_rejects_invalid_or_nonfinite_estimators(monkeypatch):
    monkeypatch.setattr(
        arena_module,
        "load_panel",
        lambda cfg: ([], {}, _trajectories(), {}),
    )

    class Bad(_FutureSeekingEstimator):
        def score(self, subject_id, prediction_time):
            return float("nan")

    with pytest.raises(ValueError, match="non-finite"):
        AdaptiveCareArena(_config()).evaluate({"good": Persistence(), "bad": Bad()})
    with pytest.raises(ValueError, match="at least two"):
        AdaptiveCareArena(_config()).evaluate({"only": Persistence()})
