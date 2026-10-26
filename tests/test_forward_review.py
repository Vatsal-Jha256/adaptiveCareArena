"""Invariants of scripts/forward_review_v2.py on a tiny hand-built panel (a test fixture, never an experiment)."""

from scripts import forward_review_v2 as fv

TAU = -5.0


def panel():
    # subject 0: stays above tau then crosses; subject 1: always severe; subject 2: mild, never crosses
    n = 40
    t0 = [None] * n
    t1 = [None] * n
    t2 = [None] * n
    for d in range(0, 30, 2):
        t0[d] = -1.0 if d < 20 else -7.0
        t1[d] = -8.0
        t2[d] = -2.0
    return {0: t0, 1: t1, 2: t2}


def test_at_risk_excludes_already_severe():
    rows, _ = fv.build(panel(), TAU, 3, None, risk="active", at_risk=True)
    assert all(x["lastraw"] > TAU for x in rows)
    assert 1 not in {x["i"] for x in rows}
    rows_all, _ = fv.build(panel(), TAU, 3, None, risk="active", at_risk=False)
    assert 1 in {x["i"] for x in rows_all}


def test_episode_deduplicated_and_recall_in_unit_interval():
    rows, raw = fv.build(panel(), TAU, 3, None, risk="active")
    eps = sorted({x["ep"] for x in rows if x["onset"]})
    assert eps == sorted(raw) and len(eps) == 1  # one crossing, many landmark rows, one episode
    assert sum(x["onset"] for x in rows) > 1
    for B in (1, 2, 0.5):
        res, _, rnd = fv.queues(rows, B, n_random=5)
        for m in res:
            assert 0 <= len(res[m] & set(eps)) / len(eps) <= 1
        for s in rnd:
            assert len(s & set(eps)) <= len(eps)


def test_no_future_information_in_features():
    base = panel()
    rows_a, _ = fv.build(base, TAU, 3, None, risk="active")
    changed = {i: list(t) for i, t in base.items()}
    changed[0][38 - 8] = None  # alter a late report of subject 0 (day 22 -> already crossed region)
    rows_b, _ = fv.build(changed, TAU, 3, None, risk="active")
    early_a = {(x["i"], x["r"]): x["ema3"] for x in rows_a if x["r"] < 20}
    early_b = {(x["i"], x["r"]): x["ema3"] for x in rows_b if x["r"] < 20}
    assert early_a == early_b  # features before day 20 are unaffected by later reports


def test_tie_jitter_shared_and_tiny():
    j = [fv.jitter(i, r) for i in range(5) for r in range(5)]
    assert max(j) < 1e-8 and len(set(j)) > 20
    assert fv.jitter(3, 7) == fv.jitter(3, 7)
