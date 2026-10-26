"""Does the EMA item-selection effect differ between CrossCheck and CES?

The two cohorts are run through deliberately matched adapters (same 10
PHQ-adjacent items, same 4 passive epochs, same event construction), so
a difference here is a difference between populations rather than
between pipelines. That is the whole reason to build the CES arm.

The pairing structure IS the point OF this file, and it is the opposite
of every other analysis in this repo:

  * within a cohort, arms are paired. ask_all and fixed_subset share the
    seed, the split and the fitted model, so the per-seed difference
    removes that shared variance.

  * between cohorts, nothing is paired. CrossCheck seed 3 and CES seed 3
    have no relationship whatsoever -- different people, different
    devices, different study. Subtracting them elementwise would invent
    a correspondence that does not exist and would report a precision
    the data cannot support.

So: form the paired within-cohort effect per seed, then compare the two
resulting distributions unpaired. Anything else is wrong here, and the
mistake would be invisible in the output.

python -m scripts.cross_cohort
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics as st
from pathlib import Path

from eval.stats import _t_cdf, tost_equivalence

MARGIN = 0.10  # as in ces_replication.py; PHQ items are 0-3


def _leaves(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _leaves(v, f"{p}.{k}" if p else k)
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        yield p, float(o)


def _load(pattern: str) -> list[dict]:
    fs = sorted(glob.glob(pattern), key=lambda f: int(f.split("seed")[-1].split(".")[0]))
    return [dict(_leaves(json.load(open(f)))) for f in fs]


def _within_cohort_effects(per_seed, a, b, metric) -> dict[str, list[float]]:
    """Paired within cohort: one effect per seed per cell."""
    out: dict[str, list[float]] = {}
    for k in per_seed[0]:
        if not k.endswith(f".{a}.{metric}"):
            continue
        cell = k[len("by_burden_profile.") : -len(f".{a}.{metric}")]
        kb = f"by_burden_profile.{cell}.{b}.{metric}"
        if all(k in s and kb in s for s in per_seed):
            out[cell.replace(".by_budget.", " @ ")] = [s[k] - s[kb] for s in per_seed]
    return out


def _welch(x: list[float], y: list[float]) -> tuple[float, float, float]:
    """UNPAIRED Welch t-test -- unequal variances, no assumption that the
    two cohorts are the same size or equally noisy. Returns
    (mean difference, t, two-sided p)."""
    nx, ny = len(x), len(y)
    mx, my = st.mean(x), st.mean(y)
    vx, vy = st.variance(x), st.variance(y)
    se = math.sqrt(vx / nx + vy / ny)
    if se <= 1e-12 * max(1.0, abs(mx - my)):
        return (
            mx - my,
            float("inf") if abs(mx - my) > 1e-12 else 0.0,
            0.0 if abs(mx - my) > 1e-12 else 1.0,
        )
    t = (mx - my) / se
    df = (vx / nx + vy / ny) ** 2 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1))
    p = 2.0 * (1.0 - _t_cdf(abs(t), max(1, int(round(df)))))
    return mx - my, t, p


def compare(cc, ces, a, b, metric, label):
    ecc = _within_cohort_effects(cc, a, b, metric)
    ece = _within_cohort_effects(ces, a, b, metric)
    shared = sorted(set(ecc) & set(ece))
    print(f"\n{label}: {a} minus {b} on {metric}")
    print(f"  cells in CrossCheck {len(ecc)}, in CES {len(ece)}, comparable {len(shared)}")
    if len(shared) < min(len(ecc), len(ece)):
        print(
            f"  NOTE: only {len(shared)} of {min(len(ecc), len(ece))} cells overlap. The cause is "
            f"STRUCTURAL, not a\n        config oversight: CrossCheck offers 10 EMA items and CES "
            f"offers 4 PHQ-4\n        items, so an absolute budget of 10 means 'ask everything' "
            f"in one cohort\n        and is unreachable in the other. Absolute budgets are simply "
            f"not the same\n        quantity across the two instruments.\n"
            f"        The fix is the matched-FRACTION grid in configs/dev/emaselect_ces_matched.yaml,"
            f"\n        which prices budgets as a share of asking everything so the axes "
            f"correspond."
        )
    if not shared:
        print("  no shared cells -- the two configs do not line up")
        return {}

    # An effect of zero has NO direction. In these runs a cell where both
    # policies were identical on every seed yields exactly 0.0, and a
    # naive sign comparison reads that as "positive" and reports a
    # direction flip -- which would be the single most misleading
    # sentence this analysis could produce ("the finding reverses between
    # cohorts!") for a cell where one cohort ran no comparison at all.
    # Such cells are reported as NO effect and excluded from the
    # replication denominator.
    ZERO = 1e-9

    rows = {}
    n_agree = n_flip = n_null = 0
    for cell in shared:
        x, y = ecc[cell], ece[cell]
        mx, my = st.mean(x), st.mean(y)
        diff, t, p = _welch(x, y)
        if abs(mx) <= ZERO or abs(my) <= ZERO:
            direction, flag = "no effect", "no effect in one cohort -- not comparable"
            n_null += 1
        elif (mx < 0) == (my < 0):
            direction, flag = "same", "same dir"
            n_agree += 1
        else:
            direction, flag = "flip", "*** DIRECTION FLIP ***"
            n_flip += 1
        rows[cell] = {
            "crosscheck_mean": mx,
            "ces_mean": my,
            "cohort_diff": diff,
            "welch_t": t,
            "welch_p": p,
            "direction": direction,
        }
        print(
            f"   {cell:>34} CC={mx:>+7.4f}  CES={my:>+7.4f}  delta={diff:>+7.4f} p={p:.4f}  {flag}"
        )

    denom = n_agree + n_flip
    print(
        f"  -> effect points the same way in {n_agree}/{denom} cells with an effect "
        f"in both cohorts"
        + (f", FLIPS in {n_flip}" if n_flip else "")
        + (f"; {n_null} cell(s) had no effect in one cohort and are excluded" if n_null else "")
    )
    if denom <= 3:
        print(
            f"  CAUTION: {denom} comparable cells is a thin basis for a replication "
            f"claim. See the cell-overlap note below."
        )

    # Is the cohort difference itself big enough to matter? Unpaired TOST
    # on the pooled effects, so "the two cohorts agree" can be a positive
    # claim rather than a failure to reject.
    pooled_cc = [v for c in shared for v in ecc[c]]
    pooled_ces = [v for c in shared for v in ece[c]]
    n = min(len(pooled_cc), len(pooled_ces))
    eq = tost_equivalence(pooled_cc[:n], pooled_ces[:n], MARGIN)
    print(
        f"  pooled effect: CrossCheck {st.mean(pooled_cc):+.4f} vs CES {st.mean(pooled_ces):+.4f}"
    )
    print("  NOTE: the pooled TOST below indexes two unrelated cohorts against each")
    print("        other, so read it as a magnitude check only, never as a paired test.")
    print(f"  cohort magnitude difference within +/-{MARGIN}? {eq['outcome']}")
    return {
        "cells": rows,
        "n_same_direction": n_agree,
        "n_flipped": n_flip,
        "n_no_effect": n_null,
        "n_comparable": denom,
        "pooled_tost": eq,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/cross_cohort.json")
    args = ap.parse_args()

    pairs = [
        (
            "forecast",
            "outputs/_emaselect_robust_ckpt/default_forecast_seed*.json",
            "outputs/_ces_ckpt/forecast_seed*.json",
        ),
        (
            "reconstruct",
            "outputs/_emaselect_robust_ckpt/targetreconstruct_seed*.json",
            "outputs/_ces_ckpt/reconstruct_seed*.json",
        ),
    ]
    out: dict = {"margin": MARGIN, "by_target": {}}
    for target, cc_glob, ces_glob in pairs:
        cc, ces = _load(cc_glob), _load(ces_glob)
        if not cc or not ces:
            print(f"{target}: missing checkpoints (CC={len(cc)}, CES={len(ces)})")
            continue
        print(
            f"\n{'=' * 74}\nCROSS-COHORT -- target={target}  "
            f"CrossCheck n={len(cc)} seeds, CES n={len(ces)} seeds\n"
            f"paired WITHIN cohort, unpaired BETWEEN cohorts\n{'=' * 74}"
        )
        out["by_target"][target] = {
            "n_crosscheck": len(cc),
            "n_ces": len(ces),
            "raw_mae": compare(cc, ces, "ask_all", "fixed_subset", "mae", "RAW MAE"),
            "cost_matched": compare(
                cc, ces, "ask_all", "fixed_subset", "frontier_gap", "COST-MATCHED"
            ),
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
