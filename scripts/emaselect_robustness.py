"""Robustness sweep for the emaselect item-selection findings.

The reported claims were measured at one configuration (forecast target,
n_train=48, default model). Findings that hold at one configuration can be artifacts of it, so these
are swept before being trusted.

Claims under test, evaluated in every (burden profile x budget) cell
rather than at a headline number:

  E1. Adaptive item selection (thompson_items) is worse than the best
      fixed questionnaire at matched burden -- frontier_gap > 0.
  E2. Static greedy-VOI ties the val-selected fixed subset -- i.e. a
      fixed rank order is as good as greedy subset search.
  E3. A fixed short form recovers most of the full questionnaire's value.

`run_experiment` already sweeps profiles x budgets internally, so each
condition yields 9-12 cells; the outer sweep varies what that grid does
not cover (target, train size, model capacity, calibration level).

python -m scripts.emaselect_robustness --seeds 5
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from emaselect.run import run_experiment

BASE = {
    "experiment_id": "emaselect_robustness",
    "seed": 0,
    "target_mode": "forecast",
    "dataset": {"local_root": "data/raw/crosscheck"},
    "split": {"n_train": 48, "n_val": 12},
    "budgets": [2.0, 4.0, 6.0, 8.0],
    "burden_profiles": ["uniform", "sensitivity_weighted", "inverted_sensitivity"],
    "selective_prediction": False,  # not under test here; saves a lot of time
}

CONDITIONS: list[tuple[str, dict]] = [
    ("default (forecast)", {}),
    ("target=reconstruct", {"target_mode": "reconstruct"}),
    ("train=36 (smaller)", {"split": {"n_train": 36, "n_val": 12}}),
    ("train=54 (larger)", {"split": {"n_train": 54, "n_val": 10}}),
]


def evaluate(cells: dict) -> dict:
    """cells[(profile, budget)][policy] = list of frontier_gap / mae."""
    e1_hold = e1_total = e1_better = e1_inconclusive = 0
    e2_hold = e2_total = 0
    gaps_thompson, gaps_greedy = [], []

    for (_prof, _b), pol in cells.items():
        if "thompson_items" in pol:
            g = st.mean(pol["thompson_items"]["gap"])
            sd = (
                st.stdev(pol["thompson_items"]["gap"])
                if len(pol["thompson_items"]["gap"]) > 1
                else 0.0
            )
            gaps_thompson.append(g)
            e1_total += 1
            # Three-way, deliberately. Counting only "adaptive worse"
            # cells is an asymmetric summary, and it misleads: on the
            # reconstruct target 0/12 cells showed adaptive worse, which
            # reads like "adaptive is fine" but actually meant 0/12
            # better as well -- every cell inconclusive, with per-cell sd
            # (0.05-0.32) dwarfing the means. A one-sided counter cannot
            # distinguish "no effect" from "no power", so it reports
            # both categories and the inconclusive remainder.
            if g - sd > 0:
                e1_hold += 1  # adaptive demonstrably worse
            elif g + sd < 0:
                e1_better += 1  # adaptive demonstrably better
            else:
                e1_inconclusive += 1
        if "greedy_voi" in pol:
            g = st.mean(pol["greedy_voi"]["gap"])
            sd = st.stdev(pol["greedy_voi"]["gap"]) if len(pol["greedy_voi"]["gap"]) > 1 else 0.0
            gaps_greedy.append(g)
            e2_total += 1
            if abs(g) <= max(sd, 1e-9):  # indistinguishable from the fixed subset
                e2_hold += 1

    return {
        "E1_adaptive_worse_cells": f"{e1_hold}/{e1_total}",
        "E1_adaptive_better_cells": f"{e1_better}/{e1_total}",
        "E1_inconclusive_cells": f"{e1_inconclusive}/{e1_total}",
        "E1_frac": e1_hold / max(1, e1_total),
        "E1_mean_gap": st.mean(gaps_thompson) if gaps_thompson else None,
        "E2_greedy_ties_fixed_cells": f"{e2_hold}/{e2_total}",
        "E2_frac": e2_hold / max(1, e2_total),
        "E2_mean_gap": st.mean(gaps_greedy) if gaps_greedy else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="outputs/emaselect_robustness.json")
    ap.add_argument("--only", default=None, help="run a single condition by name and exit")
    ap.add_argument(
        "--ckpt-dir",
        default="outputs/_emaselect_robust_ckpt",
        help="per-(condition,seed) checkpoints -- these runs are long enough "
        "that they have been interrupted mid-sweep",
    )
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    conditions = [c for c in CONDITIONS if c[0] == args.only] if args.only else CONDITIONS

    report = {}
    for name, override in conditions:
        cfg = {**BASE, **override}
        cells: dict = {}
        recovery: list[float] = []
        for seed in range(args.seeds):
            slug = name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
            ck = ckpt_dir / f"{slug}_seed{seed}.json"
            if ck.exists():
                res = json.loads(ck.read_text())
                print(f"  [{name}] seed {seed} from checkpoint", flush=True)
            else:
                c = dict(cfg)
                c["seed"] = seed
                res = run_experiment(c)
                ck.parent.mkdir(parents=True, exist_ok=True)
                ck.write_text(json.dumps(res))
                print(f"  [{name}] seed {seed} done", flush=True)
            for prof, block in res["by_burden_profile"].items():
                for b, by_pol in block["by_budget"].items():
                    slot = cells.setdefault((prof, b), {})
                    for p, m in by_pol.items():
                        slot.setdefault(p, {"gap": [], "mae": []})
                        slot[p]["gap"].append(m["frontier_gap"])
                        slot[p]["mae"].append(m["mae"])
                    # E3: fraction of the full-questionnaire benefit a
                    # fixed short form recovers at this budget.
                    none_mae = by_pol["ask_none"]["mae"]
                    all_mae = by_pol["ask_all"]["mae"]
                    fx_mae = by_pol["fixed_subset"]["mae"]
                    if none_mae != all_mae:
                        recovery.append((none_mae - fx_mae) / (none_mae - all_mae))
        ev = evaluate(cells)
        ev["E3_mean_short_form_recovery"] = st.mean(recovery) if recovery else None
        report[name] = ev
        print(
            f"{name:22s} E1 worse={ev['E1_adaptive_worse_cells']:>6} "
            f"better={ev['E1_adaptive_better_cells']:>6} "
            f"inconcl={ev['E1_inconclusive_cells']:>6} "
            f"meangap={ev['E1_mean_gap']:+.4f}   "
            f"E2(greedy==fixed) {ev['E2_greedy_ties_fixed_cells']:>7}   "
            f"E3 recovery={ev['E3_mean_short_form_recovery']:.2f}",
            flush=True,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    n = len(report)
    print(
        f"\nE1 held in >=80% of cells in "
        f"{sum(1 for v in report.values() if v['E1_frac'] >= 0.8)}/{n} conditions"
    )
    print(
        f"E2 held in >=80% of cells in "
        f"{sum(1 for v in report.values() if v['E2_frac'] >= 0.8)}/{n} conditions"
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
