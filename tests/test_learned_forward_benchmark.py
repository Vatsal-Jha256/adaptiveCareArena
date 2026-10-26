import numpy as np

from scripts.learned_forward_benchmark import split_masks


def test_time_split_embargoes_training_horizons():
    days = np.arange(20)
    cutoff, train, test = split_masks(days, horizon=3, quantile=0.6)
    assert np.all(days[train] + 3 < cutoff)
    assert np.all(days[test] >= cutoff)
    assert not np.any(train & test)
