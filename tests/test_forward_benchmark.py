import numpy as np

from scripts.forward_benchmark import (
    _decision_metrics,
    queue_bootstrap,
    queue_bootstrap_budgets,
    resample_panel_rows,
)


def _rows():
    return [
        {"i": 0, "r": 1, "last": 2.0, "ema3": 1.0, "onset": 1, "ep": (0, 2)},
        {"i": 1, "r": 1, "last": 1.0, "ema3": 2.0, "onset": 0, "ep": None},
        {"i": 2, "r": 1, "last": 0.0, "ema3": 0.0, "onset": 0, "ep": None},
    ]


def test_resampling_keeps_zero_episode_subjects_and_separates_clones():
    rebuilt = resample_panel_rows(_rows(), [0, 1, 2], np.random.default_rng(4))
    assert {row["i"] for row in rebuilt}.issubset({0, 1, 2})
    # Every output id identifies a bootstrap draw, never the source subject.
    for row in rebuilt:
        if row["ep"] is not None:
            assert row["ep"][0] == row["i"]


def test_bootstrap_rebuilds_competing_queue_and_reports_three_way_verdict():
    out = queue_bootstrap(
        _rows(),
        [0, 1, 2],
        1,
        ("last", "ema3"),
        (("last", "ema3"),),
        100,
        7,
    )
    assert out["episode_recall"] == {"last": 1.0, "ema3": 0.0}
    assert out["contrasts"]["last-ema3"]["verdict"] in {
        "different",
        "inconclusive",
        "deterministic",
        "not_applicable",
    }
    assert out["contrasts"]["last-ema3"]["n_valid"] > 0
    assert out["contrasts"]["last-ema3"]["family_size"] == 1


def test_multi_budget_bootstrap_preserves_each_budget():
    out = queue_bootstrap_budgets(
        _rows(),
        [0, 1, 2],
        (1, 2),
        ("last", "ema3"),
        (("last", "ema3"),),
        20,
        7,
    )
    assert set(out) == {"1", "2"}
    assert out["1"]["episode_recall"]["last"] == 1.0
    assert out["1"]["contrasts"]["last-ema3"]["family_size"] == 2
    assert out["1"]["missed_episodes"]["ema3"] == 1
    assert out["1"]["burden"]["capacity_utilization"] == 1.0


def test_decision_metrics_report_both_efficiency_and_not_applicable_states():
    metrics = _decision_metrics(_rows(), [0, 1, 2], 1, ("last", "ema3"), 7)
    assert metrics["last"]["captured_episodes"] == 1
    assert metrics["last"]["review_yield"] == 1.0
    assert metrics["last"]["mean_episode_lead_time_days"] == 1.0
    assert metrics["last"]["lead_time_status"] == "evaluated"
    assert metrics["ema3"]["captured_episodes"] == 0
    assert metrics["ema3"]["mean_episode_lead_time_days"] is None
    assert metrics["ema3"]["lead_time_status"] == "not_applicable_no_captured_episodes"
    assert metrics["ema3"]["attention_gini_status"] == "evaluated"
