"""Tests for the CES adapter. Marked `real_data` -- they need the
extracted dataset and are skipped when it is absent, so the suite still
runs on a machine without it."""

import numpy as np
import pytest

from caredata.ces import PHQ4_ITEMS, CESDataset

pytestmark = pytest.mark.skipif(
    not __import__("pathlib").Path("data/raw/ces/EMA/general_ema.csv").exists(),
    reason="CES dataset not present",
)


@pytest.fixture(scope="module")
def ds():
    return CESDataset()


def test_scale_matches_the_published_description(ds):
    d = ds.describe()
    # Nepal et al. 2024 describe ~200+ Dartmouth undergraduates.
    assert 200 <= d["n_participants"] <= 240
    assert d["n_complete_phq4_events"] > 30000
    assert d["median_events_per_participant"] > 100
    assert d["span_days"] > 1700  # ~4.8 years


def test_score_is_recomputed_from_items_not_taken_from_the_file(ds):
    ema = ds.load_ema()
    assert np.allclose(ema[list(PHQ4_ITEMS)].sum(axis=1), ema["phq4_recomputed"])
    # And the released column really does disagree, which is why.
    mm = ds.score_mismatches()
    assert mm["mismatched"] > 0
    assert 0 < mm["fraction"] < 0.02, "expected a small but non-zero inconsistency"


def test_items_are_on_the_documented_0_to_3_scale(ds):
    ema = ds.load_ema()
    for item in PHQ4_ITEMS:
        assert ema[item].min() >= 0 and ema[item].max() <= 3
    assert ema["phq4_recomputed"].between(0, 12).all()


def test_day_ordinal_is_monotone_within_a_participant(ds):
    ema = ds.load_ema()
    for _uid, g in list(ema.groupby("uid"))[:20]:
        assert g["day_ordinal"].is_monotonic_increasing


def test_sensing_epochs_are_summed_with_min_count(ds):
    # An all-missing day must stay NaN, never become a spurious 0.0 --
    # "no data" and "measured, and it was zero" mean opposite things.
    s = ds.load_sensing()
    daily = [c for c in s.columns if c.endswith("_daily")]
    assert daily, "no daily features built"
    assert s[daily].isna().any().any(), "min_count=1 should leave some NaNs"


def test_ema_and_sensing_cover_the_same_participants(ds):
    ema, s = ds.load_ema(), ds.load_sensing()
    common = set(ema["uid"]) & set(s["uid"])
    assert len(common) >= 200
    merged = ema.merge(s[["uid", "date"]].drop_duplicates(), on=["uid", "date"], how="inner")
    # Near-total same-day coverage is what makes passive context usable.
    assert len(merged) / len(ema) > 0.95


def test_list_subjects_respects_the_minimum(ds):
    many = ds.list_subjects(min_ema=20)
    few = ds.list_subjects(min_ema=200)
    assert len(many) > len(few) and set(few) <= set(many)
