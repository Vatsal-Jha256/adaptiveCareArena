"""Robustness sweep for the real-data allocation findings.

Two claims came out of the default configuration (budget=5,
deterioration_threshold=-5.0, horizon=270, relative alignment):

  C1. round_robin achieves the lowest belief staleness yet misses more
      deterioration days than every need-aware policy -- i.e. "keep
      everyone uniformly fresh" and "catch who is deteriorating" are
      different objectives.
  C2. The four need-aware policies (fair_floor, whittle_index,
      ucb_explore, greedy_need) are statistically indistinguishable from
      each other on detection -- the sophisticated RMAB index buys
      nothing over simple greedy-need.

Both were measured at one point in configuration space. This sweeps one
factor at a time around that default and reports, per condition, whether
each claim still holds -- rather than dumping numbers and eyeballing
them. A claim that only holds at budget=5 and threshold=-5.0 is an
artifact of those settings and needs to be reported as one.

python -m scripts.replay_robustness --seeds 5
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from stage2_env.replay_experiment import run_experiment

NEED_AWARE = ("fair_floor", "whittle_index", "ucb_explore", "greedy_need")
NAIVE = ("round_robin", "random")

BASE = {
    "experiment_id": "replay_robustness",
    "seed": 0,
    "data": {
        "local_root": "data/raw/crosscheck",
        "min_ema": 20,
        "align": "relative",
        "horizon": 270,
    },
    "budget": {"b_per_round": 5},
    "estimator": {"name": "naive_ema"},
    "restrict_to_available": True,
    "deterioration_threshold": -5.0,
}

# One factor at a time around the default -- a full grid would be 48
# cells and would mostly test interactions nobody claimed.
CONDITIONS: list[tuple[str, dict]] = [
    ("budget=2", {"budget": {"b_per_round": 2}}),
    ("budget=5 (default)", {}),
    ("budget=10", {"budget": {"b_per_round": 10}}),
    ("budget=20", {"budget": {"b_per_round": 20}}),
    ("threshold=-3", {"deterioration_threshold": -3.0}),
    ("threshold=-8", {"deterioration_threshold": -8.0}),
    ("horizon=180", {"data": {**BASE["data"], "horizon": 180}}),
    ("horizon=365", {"data": {**BASE["data"], "horizon": 365}}),
    ("align=calendar", {"data": {**BASE["data"], "align": "calendar"}}),
    ("min_ema=40", {"data": {**BASE["data"], "min_ema": 40}}),
]


def _mean_std(xs):
    return st.mean(xs), (st.stdev(xs) if len(xs) > 1 else 0.0)


def evaluate(agg: dict) -> dict:
    """Test the two claims against one condition's aggregated results."""
    miss = {p: _mean_std(v["miss"]) for p, v in agg.items()}
    stale = {p: _mean_std(v["stale"]) for p, v in agg.items()}

    # C1: round_robin lowest staleness and worse detection than every
    # need-aware policy (comparing means; sd reported alongside).
    rr_freshest = all(stale["round_robin"][0] <= stale[p][0] for p in agg if p != "round_robin")
    rr_misses_more = all(miss["round_robin"][0] > miss[p][0] for p in NEED_AWARE if p in miss)

    # C2: are the need-aware policies distinguishable from each other?
    #
    # Two criteria, because they answer different questions and the
    # statistical one alone was misleading. A pooled-sd overlap test
    # reported "distinguishable" in conditions where the intervals
    # overlap heavily and the gap is ~1% of the level -- statistically
    # separable, practically identical. Reporting only that understated a
    # claim that is near-universal on the numbers.
    present = [p for p in NEED_AWARE if p in miss]
    means = [miss[p][0] for p in present]
    spread = max(means) - min(means)
    level = st.mean(means) if means else 0.0

    pairs_overlap = []
    for i, a in enumerate(present):
        for b in present[i + 1 :]:
            pooled = (miss[a][1] ** 2 + miss[b][1] ** 2) ** 0.5
            pairs_overlap.append(abs(miss[a][0] - miss[b][0]) <= max(pooled, 1e-9))
    need_indistinguishable = all(pairs_overlap) if pairs_overlap else False

    # Practical equivalence: the whole spread across need-aware policies
    # is under 2% of the level. This is the criterion that matters for
    # "does it matter which one you deploy?".
    practically_equivalent = bool(level) and (spread / abs(level)) < 0.02

    # The specific comparison the environment was built around.
    whittle_vs_greedy = None
    if "whittle_index" in miss and "greedy_need" in miss:
        diff = miss["whittle_index"][0] - miss["greedy_need"][0]
        pooled = (miss["whittle_index"][1] ** 2 + miss["greedy_need"][1] ** 2) ** 0.5
        whittle_vs_greedy = {
            "diff_missed_days": diff,
            "pooled_std": pooled,
            "whittle_better_beyond_noise": bool(diff < -max(pooled, 1e-9)),
        }

    # Supporting: do need-aware policies beat the naive ones at all?
    best_need = min(miss[p][0] for p in NEED_AWARE if p in miss)
    worst_naive = max(miss[p][0] for p in NAIVE if p in miss)
    need_beats_naive = best_need < worst_naive

    return {
        "C2_practically_equivalent": practically_equivalent,
        "need_aware_spread_frac": (spread / abs(level)) if level else None,
        "whittle_vs_greedy": whittle_vs_greedy,
        "C1_round_robin_freshest": rr_freshest,
        "C1_round_robin_misses_more": rr_misses_more,
        "C1_holds": rr_freshest and rr_misses_more,
        "C2_need_aware_indistinguishable": need_indistinguishable,
        "need_beats_naive_detection": need_beats_naive,
        "miss_by_policy": {p: {"mean": m, "std": s} for p, (m, s) in miss.items()},
        "stale_by_policy": {p: {"mean": m, "std": s} for p, (m, s) in stale.items()},
        # Raw per-seed values. Storing only mean/sd forced a later
        # equivalence test to reconstruct the sample by interpolating
        # between mean+-sd, which has the right centre and roughly the
        # right spread but is not the real draw. Any test more careful
        # than a t-test needs the actual observations, so they are kept.
        "raw_miss_by_policy": {p: list(v["miss"]) for p, v in agg.items()},
        "raw_stale_by_policy": {p: list(v["stale"]) for p, v in agg.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="outputs/replay_robustness.json")
    args = ap.parse_args()

    report = {}
    for name, override in CONDITIONS:
        cfg = {**BASE, **override}
        agg: dict[str, dict[str, list]] = {}
        for seed in range(args.seeds):
            c = dict(cfg)
            c["seed"] = seed
            res = run_experiment(c)
            for p, m in res["by_policy"].items():
                slot = agg.setdefault(p, {"miss": [], "stale": [], "gap": []})
                slot["miss"].append(m["missed_deterioration_days"])
                slot["stale"].append(m["mean_belief_staleness"])
                slot["gap"].append(m["allocation_max_gap"])
        report[name] = evaluate(agg)
        v = report[name]
        print(
            f"{name:22s} C1={'HOLDS' if v['C1_holds'] else 'FAILS':5s} "
            f"C2stat={'Y' if v['C2_need_aware_indistinguishable'] else 'n'} "
            f"C2prac={'Y' if v['C2_practically_equivalent'] else 'n'} "
            f"spread={v['need_aware_spread_frac']:.3f} "
            f"need>naive={'yes' if v['need_beats_naive_detection'] else 'NO'}",
            flush=True,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    c1 = sum(v["C1_holds"] for v in report.values())
    c2 = sum(v["C2_need_aware_indistinguishable"] for v in report.values())
    nb = sum(v["need_beats_naive_detection"] for v in report.values())
    n = len(report)
    print(f"\nC1 (staleness/detection dissociation) holds in {c1}/{n} conditions")
    c2p = sum(v["C2_practically_equivalent"] for v in report.values())
    wg = sum(
        1
        for v in report.values()
        if v["whittle_vs_greedy"] and v["whittle_vs_greedy"]["whittle_better_beyond_noise"]
    )
    print(f"C2 statistical (pooled-sd overlap) holds in {c2}/{n} conditions")
    print(f"C2 practical (spread < 2% of level) holds in {c2p}/{n} conditions")
    print(f"whittle_index beats greedy_need beyond noise in {wg}/{n} conditions")
    print(f"need-aware beats naive on detection in {nb}/{n} conditions")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
