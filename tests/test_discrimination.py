import scripts.discrimination_benchmark as discrimination_runner
from eval.discrimination import oracle_cutoff_degradation


def _rows():
    return [
        {"i": 0, "r": 1, "last": 4.0, "onset": 1, "ep": (0, 2)},
        {"i": 1, "r": 1, "last": 3.0, "onset": 0, "ep": None},
        {"i": 2, "r": 1, "last": 2.0, "onset": 0, "ep": None},
        {"i": 3, "r": 1, "last": 1.0, "onset": 0, "ep": None},
    ]


def test_oracle_degradation_swaps_selected_positive_below_cutoff():
    rows, names = oracle_cutoff_degradation(
        _rows(),
        source_score="last",
        budget=2,
        swaps_per_day=(1, 2),
        seed=0,
    )
    assert names == ("oracle_degraded_1", "oracle_degraded_2")
    for name in names:
        ranked = sorted(rows, key=lambda row: row[name], reverse=True)
        assert all(row["i"] != 0 for row in ranked[:2])


def test_oracle_degradation_is_explicitly_parameter_validated():
    try:
        oracle_cutoff_degradation(
            _rows(),
            source_score="last",
            budget=0,
            swaps_per_day=(1,),
            seed=0,
        )
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("zero budget was accepted")


def test_runner_reports_degenerate_curve_as_distinct_outcome(monkeypatch):
    monkeypatch.setattr(
        discrimination_runner,
        "load_panel",
        lambda cfg: (
            [],
            {},
            {
                0: [0.0, 0.0, 0.0, 0.0, 0.0, -0.5, -2.0],
                1: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                2: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            },
            {},
        ),
    )
    cfg = {
        "seed": 7,
        "data": {"source": "ces", "local_root": "unused"},
        "outcome": {"threshold": -1.0, "horizon_days": 1},
        "control": {"source_score": "last", "budget": 1, "swaps_per_day": [1, 2]},
        "inference": {"n_boot": 20},
    }
    result = discrimination_runner.run(cfg)
    assert result["curve_status"] == "degenerate_ungraded"
    assert "cannot distinguish" in result["curve_interpretation"]
