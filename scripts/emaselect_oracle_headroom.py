"""Per-instance oracle headroom for EMASelect.

The negative result (adaptive per-patient selection barely beats a fixed
short form) has an obvious objection: maybe the adaptive algorithm is
just weak, not that adaptivity has no headroom. This settles it directly.

For each (profile, budget) cell, enumerate every feasible item subset
(there are only 10 items for CrossCheck, 4 for CES, so full enumeration
is cheap) and, using the held-out test labels, pick the best subset for
each test event individually. That is an upper bound no real policy can
reach (it uses labels no deployed system has) -- an oracle, not a
baseline. If FIXED_SUBSET is already close to oracle, there is little
headroom for any adaptive method, Thompson sampling included. If oracle
is far ahead of both, the negative result is about Thompson specifically,
not about adaptivity in general.

Uses fewer seeds than the main sweep (5, not 20/35): this is a
structural headroom check, not an inferential claim requiring tight
CIs, and full enumeration per seed is the added cost.

python -m scripts.emaselect_oracle_headroom --config configs/dev/emaselect_smoke.yaml --dataset crosscheck
python -m scripts.emaselect_oracle_headroom --config configs/dev/emaselect_ces.yaml --dataset ces
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import yaml

from emaselect.model import MaskAwareEMAModel
from emaselect.run import _best_fixed_subset, _build_events, _mae, _split

N_SEEDS = 5


def _all_feasible_subsets(items: tuple, cost_profile: dict, budget: float) -> list[set]:
    out = []
    for r in range(len(items) + 1):
        for combo in itertools.combinations(items, r):
            cost = sum(cost_profile[i] for i in combo)
            if cost <= budget:
                out.append(set(combo))
    return out


def oracle_mae(model, items, test, y_test, cost_profile, budget) -> tuple[float, float]:
    """Returns (oracle_mae, mean_subset_size) using per-event best subset."""
    subsets = _all_feasible_subsets(items, cost_profile, budget)
    if not subsets:
        return float("nan"), 0.0
    # preds[j] = predictions of subset j across all test events
    preds = np.stack(
        [model.predict(test, [s] * len(test)) for s in subsets]
    )  # (n_subsets, n_events)
    errs = np.abs(preds - y_test[None, :])  # (n_subsets, n_events)
    best_j = np.argmin(errs, axis=0)
    best_err = errs[best_j, np.arange(len(test))]
    sizes = np.array([len(s) for s in subsets])
    return float(np.mean(best_err)), float(np.mean(sizes[best_j]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base_cfg = yaml.safe_load(open(args.config))
    out_path = args.out or f"outputs/emaselect_oracle_{args.dataset}.json"

    results: dict = {"dataset": args.dataset, "n_seeds": N_SEEDS, "by_seed": []}

    for seed in range(N_SEEDS):
        cfg = dict(base_cfg)
        cfg["seed"] = seed
        rng = np.random.default_rng(seed)
        events, items, burden_profiles = _build_events(cfg)
        train, val, test = _split(events, rng, cfg["split"]["n_train"], cfg["split"]["n_val"])
        model = MaskAwareEMAModel(items, random_state=seed)
        model.fit(train)
        y_val = np.array([e.target_y for e in val])
        y_test = np.array([e.target_y for e in test])

        seed_out = {"seed": seed, "by_burden_profile": {}}
        for pname in cfg.get("burden_profiles", list(burden_profiles)):
            cost_profile = burden_profiles[pname]
            seed_out["by_burden_profile"][pname] = {}
            for budget in cfg["budgets"]:
                if budget <= 0:
                    continue
                fixed = _best_fixed_subset(model, val, y_val, cost_profile, budget)
                fixed_mae = _mae(model.predict(test, [fixed] * len(test)), y_test)
                o_mae, o_size = oracle_mae(model, items, test, y_test, cost_profile, budget)
                seed_out["by_burden_profile"][pname][str(budget)] = {
                    "fixed_subset_mae": fixed_mae,
                    "oracle_mae": o_mae,
                    "oracle_mean_items": o_size,
                    "headroom": fixed_mae - o_mae,
                }
                print(
                    f"seed {seed} {pname} b={budget}: fixed={fixed_mae:.4f} "
                    f"oracle={o_mae:.4f} headroom={fixed_mae - o_mae:.4f}",
                    flush=True,
                )
        results["by_seed"].append(seed_out)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(results, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
