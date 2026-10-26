"""Threshold sensitivity for the review-axis headline: does "need-aware
policies miss fewer high-severity events" survive alternative
severity-threshold choices, not just the single pre-specified one?

Deliberately not another 40-cell hypothesis-testing pass -- this is a
point-estimate robustness check on the real panel (no bootstrap), per
(threshold, budget, policy), reported as a heatmap of relative reduction
in missed events vs round_robin. CrossCheck's -5.0 threshold has no
independent clinical validation (unlike CES's PHQ-4 severe-distress
cutoff), so it is the one that most needs this.

python -m scripts.stage2_threshold_sensitivity --dataset crosscheck --thresholds -3,-5,-7,-9
python -m scripts.stage2_threshold_sensitivity --dataset ces --thresholds -6,-9,-12
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from stage2_env.core import POLICIES
from stage2_env.replay_experiment import load_panel, run_one_policy

BUDGETS = (3, 5, 10)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["crosscheck", "ces"], required=True)
    ap.add_argument("--thresholds", required=True, help="comma-separated")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(",")]
    cfg_path = f"configs/dev/replay_{args.dataset}.yaml"
    base_cfg = yaml.safe_load(open(cfg_path))
    specs, gspecs, traj, _ = load_panel(base_cfg["data"])
    policies = sorted(POLICIES.names())
    baseline = "round_robin"

    out: dict = {
        "dataset": args.dataset,
        "thresholds": thresholds,
        "budgets": list(BUDGETS),
        "n_subjects": len(specs),
        "cells": [],
    }

    for thr in thresholds:
        for budget in BUDGETS:
            by_policy = {}
            for pname in policies:
                m = run_one_policy(
                    pname,
                    specs,
                    gspecs,
                    traj,
                    budget=budget,
                    estimator_name=base_cfg["estimator"]["name"],
                    estimator_params=base_cfg["estimator"].get("params", {}),
                    policy_params=base_cfg.get("policy_params", {}).get(pname, {}),
                    restrict=base_cfg.get("restrict_to_available", True),
                    deterioration_threshold=thr,
                    seed=0,
                )
                by_policy[pname] = m["missed_deterioration_days"]
            base_missed = by_policy[baseline]
            for pname in policies:
                if pname == baseline:
                    continue
                rel = (
                    (by_policy[pname] - base_missed) / base_missed if base_missed else float("nan")
                )
                out["cells"].append(
                    {
                        "threshold": thr,
                        "budget": budget,
                        "policy": pname,
                        "missed": by_policy[pname],
                        "missed_baseline": base_missed,
                        "relative_change": rel,
                    }
                )
            print(
                f"threshold={thr} budget={budget}: "
                + ", ".join(f"{p}={by_policy[p]}" for p in policies),
                flush=True,
            )

    out_path = args.out or f"outputs/stage2_threshold_sensitivity_{args.dataset}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
