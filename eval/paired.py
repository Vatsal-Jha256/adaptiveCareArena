"""Paired comparison of policies across seeds.

Policies run inside the same seed share a subject split, a fitted model and a
test set, so most seed-to-seed variance ("which subjects landed in test") is
common to both arms and cancels in the per-seed difference. Comparing
A's mean +- sd against B's mean +- sd throws that cancellation away and
loses most of the available power. The unpaired analysis can also err in
either direction, so it is not a conservative fallback.

How much pairing helps depends on how much the arms share. It is large for
policies that share one split and one model, and small for components that
are each fitted separately per seed (their errors are independent). Report
the gain with `pairing_gain` rather than assuming it.

Use `paired_policy_delta` for any A-vs-B comparison where both policies
were run on the same seed. Use the unpaired form only when the arms
came from different runs."""

from __future__ import annotations

import statistics as st


def paired_policy_delta(
    per_seed_a: list[float],
    per_seed_b: list[float],
    higher_is_better: bool,
    label_a: str = "A",
    label_b: str = "B",
) -> dict:
    """Per-seed difference A-B, with the sd of the difference.

    `per_seed_a[i]` and `per_seed_b[i]` must come from the same seed --
    that is what makes the pairing valid. Passing shuffled or
    independently-ordered lists silently reverts this to the unpaired
    analysis it exists to replace.

    Returns `effect_sd` (|mean| / sd of the difference) and a three-way
    verdict, never a bare "A wins", because every summary needs a category for "this tells you nothing".
    """
    if len(per_seed_a) != len(per_seed_b):
        raise ValueError("paired comparison needs one value per arm per seed")
    if len(per_seed_a) < 2:
        raise ValueError("need at least 2 seeds to estimate the paired sd")

    diffs = [a - b for a, b in zip(per_seed_a, per_seed_b, strict=False)]
    mean = st.mean(diffs)
    sd = st.stdev(diffs)

    # Degenerate-sd test with a tolerance, not `sd == 0.0`. Subtracting
    # floats that differ by a constant leaves sd ~1e-16 rather than
    # exactly zero, so an equality check misses the deterministic case
    # and then divides by that residue, manufacturing an effect size of
    # ~1e14.
    degenerate = sd <= 1e-12 * max(1.0, abs(mean))
    effect = float("inf") if degenerate else abs(mean) / sd

    if degenerate:
        # The winner still depends on the metric's direction: on an error
        # metric a positive difference means A is worse.
        if mean == 0:
            verdict = "identical"
        else:
            a_wins = (mean > 0) if higher_is_better else (mean < 0)
            verdict = f"{label_a if a_wins else label_b} better (deterministic)"
    elif effect < 1.0:
        verdict = "inconclusive"
    else:
        a_wins = (mean > 0) if higher_is_better else (mean < 0)
        verdict = f"{label_a} better" if a_wins else f"{label_b} better"

    return {
        "mean_diff": mean,
        "paired_sd": sd,
        "effect_sd": effect,
        "n_seeds": len(diffs),
        "per_seed_diffs": diffs,
        "verdict": verdict,
        # How much power the unpaired analysis would have thrown away.
        "unpaired_sd": (st.stdev(per_seed_a) ** 2 + st.stdev(per_seed_b) ** 2) ** 0.5,
    }


def pairing_gain(result: dict) -> float:
    """Ratio of unpaired to paired sd -- how much the pairing bought."""
    return result["unpaired_sd"] / result["paired_sd"] if result["paired_sd"] else float("inf")


def seeds_needed(result: dict, target_effect_sd: float = 2.0) -> int:
    """Seeds required to reach `target_effect_sd`, if the effect is real
    and the paired sd holds. sd of the mean shrinks as 1/sqrt(n), so this
    is n * (target / current)^2. Returns the current count when already
    there. This is a planning estimate from a small sample, not a power
    guarantee."""
    import math

    cur = result["effect_sd"]
    if cur == 0 or cur == float("inf"):
        return result["n_seeds"]
    need = result["n_seeds"] * (target_effect_sd / cur) ** 2
    return max(result["n_seeds"], int(math.ceil(need)))
