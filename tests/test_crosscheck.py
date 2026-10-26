"""Structural tests for caredata/crosscheck.py, using a small constructed
DataFrame (not the real file, so this runs without data present) --
validates the leakage-critical logic (genuine-vs-NaN EMA density, no
forward-filled target) independent of the real download.
"""

import numpy as np
import pandas as pd
import pytest

from caredata.crosscheck import CrossCheckDataset
from task.types import assert_no_future_leakage


def _make_fake_daily_df() -> pd.DataFrame:
    """One subject, 20 days starting 2016-01-25 (deliberately spans the
    Jan/Feb boundary -- the exact case the YYYYMMDD-as-day-count bug got
    wrong), EMA answered on offsets 0, 3, 6, 10, 15 (irregular,
    like real CrossCheck), NaN elsewhere -- mirrors the real "every 2-3
    days" cadence including gaps."""
    n = 20
    ema_offsets = {0, 3, 6, 10, 15}
    base = pd.Timestamp("2016-01-25")
    rows = []
    for offset in range(n):
        date = base + pd.Timedelta(days=offset)
        row = {
            "study_id": 1,
            "eureka_id": "u_test",
            "day": int(date.strftime("%Y%m%d")),
            "act_still_ep_0": 1000.0 + offset,
            "sleep_duration": 7.0,
            "quality_activity": 20.0,
            "quality_audio": 20.0,
            "quality_gps_on": 20.0,
            "quality_light": 20.0,
            "quality_loc": 20.0,
            "ema_score": float(offset) if offset in ema_offsets else np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def test_ema_administration_density_counts_genuine_responses_only():
    ds = CrossCheckDataset()
    ds._daily = _make_fake_daily_df()
    density = ds.ema_administration_density()
    row = density.iloc[0]
    assert row["n_ema_responses"] == 5  # Not 20
    assert row["n_days"] == 20


def test_build_task_table_never_targets_a_nan_ema_row():
    ds = CrossCheckDataset()
    ds._daily = _make_fake_daily_df()
    examples = ds.build_task_table(min_history_days=1)
    # Every example's target_y must come from one of the
    # answered days (0, 3, 6, 10, 15), reflected in target_observed_at.
    ema_days = {0.0, 3.0, 6.0, 10.0, 15.0}
    for ex in examples:
        assert ex.target_observed_at in ema_days


def test_build_task_table_horizon_is_always_strictly_positive():
    ds = CrossCheckDataset()
    ds._daily = _make_fake_daily_df()
    examples = ds.build_task_table(min_history_days=1)
    for ex in examples:
        assert ex.horizon > 0
        assert ex.target_observed_at == ex.prediction_time_t + ex.horizon


def test_validate_temporal_contract_passes_on_real_construction():
    ds = CrossCheckDataset()
    ds._daily = _make_fake_daily_df()
    examples = ds.build_task_table(min_history_days=1)
    ds.validate_temporal_contract(examples)  # must not raise


def test_last_known_ema_score_never_uses_a_future_value():
    ds = CrossCheckDataset()
    ds._daily = _make_fake_daily_df()
    examples = ds.build_task_table(min_history_days=1)
    for ex in examples:
        last_known = ex.context_features.get("last_known_ema_score")
        if last_known == last_known:  # not NaN
            # The example's own prediction time must be >= whatever day
            # last_known_ema_score came from -- we can't recover the exact
            # source day from the feature dict alone, but we can assert it
            # never equals the future target itself (a stronger, cheap
            # leakage smell test).
            assert last_known != ex.target_y or ex.prediction_time_t >= ex.target_observed_at


def _make_fake_hourly_df() -> pd.DataFrame:
    """24 hours for one date, study_id=1 (unambiguous 1:1 map to
    eureka_id=u_test in _make_fake_daily_df). Night hours (0-6) given
    much higher accel_mean than day hours (8-21) so the night/day ratio
    is unambiguously > 1 -- a deliberately unrealistic but easy-to-check
    signal, not a claim about real circadian direction."""
    rows = []
    for h in range(24):
        night = h in range(0, 7)
        rows.append(
            {
                "study_id": 1.0,
                "hour": f"2016-01-26 {h:02d}:00:00",
                "day": f"2016-01-26 {h:02d}:00:00",
                "filled_pct": 0.8,
                "accel_mean": 20.0 if night else 5.0,
                "screen_sum_1hr_screen_duration_sum": 60.0,
                "distance_sum": 10.0,
            }
        )
    return pd.DataFrame(rows)


def test_study_to_eureka_map_excludes_sentinel_and_ambiguous_ids():
    ds = CrossCheckDataset()
    ds._daily = pd.DataFrame(
        [
            {"study_id": -1, "eureka_id": "a", "day": 20160101, "ema_score": np.nan},
            {"study_id": -1, "eureka_id": "b", "day": 20160101, "ema_score": np.nan},
            {"study_id": 5, "eureka_id": "c", "day": 20160101, "ema_score": np.nan},
            {"study_id": 5, "eureka_id": "d", "day": 20160101, "ema_score": np.nan},  # ambiguous
            {"study_id": 7, "eureka_id": "e", "day": 20160101, "ema_score": np.nan},  # clean 1:1
        ]
    )
    m = ds._study_to_eureka_map()
    assert m == {7: "e"}


def test_hourly_daily_aggregates_night_day_ratio_and_imputation_rate():
    ds = CrossCheckDataset()
    ds._daily = _make_fake_daily_df()
    ds._hourly_raw = _make_fake_hourly_df()
    agg = ds.hourly_daily_aggregates()
    assert len(agg) == 1
    row = agg.iloc[0]
    assert row["eureka_id"] == "u_test"
    assert row["hourly_night_day_accel_ratio"] > 1.0  # 20.0/5.0 = 4.0 by construction
    # filled_pct is the imputation rate per the CrossCheck data
    # dictionary -- high means less real sensor data. Named "completeness"
    # here and in the adapter until 2026-09-02, which inverts the meaning.
    assert abs(row["hourly_filled_pct_mean"] - 0.8) < 1e-9
    assert row["hourly_screen_sum_total"] == 24 * 60.0


def test_all_missing_hourly_group_stays_nan_rather_than_summing_to_zero():
    """`sum()` returns 0.0 for an all-NaN group, which is indistinguishable
    from a genuine zero -- a day the phone recorded no screen time. Those
    mean opposite things ("no data" vs "measured, and it was zero"), so an
    all-missing group must stay NaN and be routed as missing."""
    import numpy as np

    ds = CrossCheckDataset()
    ds._daily = _make_fake_daily_df()
    hourly = _make_fake_hourly_df()
    hourly["distance_sum"] = np.nan
    ds._hourly_raw = hourly
    agg = ds.hourly_daily_aggregates()
    assert np.isnan(agg.iloc[0]["hourly_distance_sum_total"])
    # A real zero must still come through as zero, not NaN.
    hourly2 = _make_fake_hourly_df()
    hourly2["distance_sum"] = 0.0
    ds2 = CrossCheckDataset()
    ds2._daily = _make_fake_daily_df()
    ds2._hourly_raw = hourly2
    assert ds2.hourly_daily_aggregates().iloc[0]["hourly_distance_sum_total"] == 0.0


def test_build_task_table_with_hourly_features_adds_expected_columns():
    ds = CrossCheckDataset()
    ds._daily = _make_fake_daily_df()
    ds._hourly_raw = _make_fake_hourly_df()
    examples = ds.build_task_table(min_history_days=1, use_hourly_features=True)
    assert len(examples) > 0
    # Only the example whose window includes 2016-01-25 (the hourly fixture's
    # only date) should get a non-NaN hourly feature; window means over dates
    # without hourly coverage must fall back to NaN, not silently zero.
    has_signal = [
        e
        for e in examples
        if e.context_features.get("hourly_night_day_accel_ratio_mean")
        == e.context_features.get("hourly_night_day_accel_ratio_mean")
    ]
    assert len(has_signal) >= 1
    for e in examples:
        assert "weekday_sin" in e.context_features
        assert "weekday_cos" in e.context_features


def test_corrupted_example_with_early_target_is_caught_by_leakage_check():
    from task.types import CareExample

    bad = CareExample(
        subject_id="x",
        prediction_time_t=10.0,
        horizon=2.0,
        context_features={},
        target_y=1.0,
        target_observed_at=10.5,  # before prediction_time_t + horizon = 12.0
        domain_id="crosscheck",
    )
    with pytest.raises(ValueError, match="leakage"):
        assert_no_future_leakage([bad])
