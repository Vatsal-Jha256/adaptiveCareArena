"""Controlled benchmark-discrimination diagnostics.

These controls deliberately use future proxy labels to damage a ranking. They
are evaluation instruments, never admissible policies or evidence of clinical
utility.
"""

from __future__ import annotations

from copy import deepcopy

from scripts.report_landmarks import deterministic_jitter


def oracle_cutoff_degradation(
    rows: list[dict],
    *,
    source_score: str,
    budget: int,
    swaps_per_day: tuple[int, ...],
    seed: int,
) -> tuple[list[dict], tuple[str, ...]]:
    """Add score columns with controlled positive-to-negative cutoff swaps.

    For each day, the nearest selected positive landmark is exchanged with the
    nearest unselected negative landmark. Repeating this operation creates a
    monotone *attempted* degradation ladder; days without both kinds of row are
    unchanged. The use of ``onset`` is explicit and makes every generated arm
    an oracle diagnostic rather than a realizable method.
    """
    if budget <= 0:
        raise ValueError("budget must be positive")
    if not swaps_per_day or any(value <= 0 for value in swaps_per_day):
        raise ValueError("swaps_per_day must contain positive integers")
    output = deepcopy(rows)
    names = tuple(f"oracle_degraded_{value}" for value in swaps_per_day)
    by_day: dict[int, list[int]] = {}
    for index, row in enumerate(output):
        if source_score not in row:
            raise KeyError(f"missing source score {source_score!r}")
        by_day.setdefault(row["r"], []).append(index)

    for day, indices in by_day.items():
        order = sorted(
            indices,
            key=lambda index: (
                output[index][source_score]
                + deterministic_jitter(output[index]["i"], day, seed=seed)
            ),
            reverse=True,
        )
        for swaps, name in zip(swaps_per_day, names, strict=True):
            degraded = list(order)
            for _ in range(swaps):
                selected_positive = next(
                    (
                        position
                        for position in range(min(budget, len(degraded)) - 1, -1, -1)
                        if output[degraded[position]]["onset"]
                    ),
                    None,
                )
                unselected_negative = next(
                    (
                        position
                        for position in range(budget, len(degraded))
                        if not output[degraded[position]]["onset"]
                    ),
                    None,
                )
                if selected_positive is None or unselected_negative is None:
                    break
                degraded[selected_positive], degraded[unselected_negative] = (
                    degraded[unselected_negative],
                    degraded[selected_positive],
                )
            # Integer gaps dominate the evaluator's 1e-9 deterministic jitter.
            for rank, index in enumerate(degraded):
                output[index][name] = float(len(degraded) - rank)
    return output, names
