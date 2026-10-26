"""Run an oracle rank-degradation diagnostic on real report landmarks.

The generated arms use future proxy labels and are never admissible policies.
They test benchmark resolution only; they do not support diagnostic, treatment,
intervention, triage, or deployment claims.
"""

from __future__ import annotations

import argparse
import time

from eval.discrimination import oracle_cutoff_degradation
from runtime.manifest import load_config, make_run_dir, write_manifest, write_results
from scripts.forward_benchmark import queue_bootstrap_budgets
from scripts.report_landmarks import KALMAN_PARAMETERS, build_report_landmarks
from stage2_env.replay_experiment import load_panel


def run(cfg: dict) -> dict:
    data_cfg = dict(cfg["data"], align="calendar", horizon=None)
    _, _, trajectories, _ = load_panel(data_cfg)
    threshold = float(cfg["outcome"]["threshold"])
    horizon = int(cfg["outcome"]["horizon_days"])
    rows, raw_episodes = build_report_landmarks(
        trajectories,
        threshold,
        horizon,
        KALMAN_PARAMETERS.get(cfg["data"]["source"]),
        risk="report",
        at_risk=True,
    )
    source = str(cfg["control"]["source_score"])
    budget = int(cfg["control"]["budget"])
    rows, degraded = oracle_cutoff_degradation(
        rows,
        source_score=source,
        budget=budget,
        swaps_per_day=tuple(int(value) for value in cfg["control"]["swaps_per_day"]),
        seed=int(cfg["seed"]),
    )
    models = (source, *degraded)
    comparisons = tuple((source, name) for name in degraded)
    eligible = len({row["ep"] for row in rows if row["onset"]})
    result = {
        "evaluation_type": "oracle-labeled controlled rank degradation",
        "admissible_policy": False,
        "control_uses_future_proxy_labels": True,
        "scope": "benchmark discrimination diagnostic only",
        "dataset": cfg["data"]["source"],
        "source_score": source,
        "budget": budget,
        "swaps_per_day": list(cfg["control"]["swaps_per_day"]),
        "subjects": len(trajectories),
        "candidate_rows": len(rows),
        "raw_episodes": len(raw_episodes),
        "eligible_episodes": eligible,
        "by_budget": {},
    }
    if eligible == 0:
        result["status"] = "not_applicable_no_eligible_episodes"
    else:
        result["status"] = "evaluated"
        result["by_budget"] = queue_bootstrap_budgets(
            rows,
            sorted(trajectories),
            (budget,),
            models,
            comparisons,
            int(cfg["inference"]["n_boot"]),
            int(cfg["seed"]),
        )
        recalls = result["by_budget"][str(budget)]["episode_recall"]
        degraded_recalls = [float(recalls[name]) for name in degraded]
        if len(set(degraded_recalls)) == 1:
            result["curve_status"] = "degenerate_ungraded"
            result["curve_interpretation"] = (
                "the control detects a ranking cliff but cannot distinguish the requested "
                "degradation levels"
            )
        elif all(
            right <= left
            for left, right in zip(degraded_recalls, degraded_recalls[1:], strict=False)
        ):
            result["curve_status"] = "graded_monotone"
            result["curve_interpretation"] = (
                "observed recall decreases monotonically across requested degradation levels"
            )
        else:
            result["curve_status"] = "inconclusive_nonmonotone"
            result["curve_interpretation"] = (
                "requested degradation levels do not induce a monotone observed-recall curve"
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-out", default="outputs")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg["_source_path"] = args.config
    started = time.perf_counter()
    result = run(cfg)
    wall_seconds = time.perf_counter() - started
    run_dir = make_run_dir(cfg, args.base_out)
    write_manifest(run_dir, cfg, data_roots={cfg["data"]["source"]: cfg["data"]["local_root"]})
    digest = write_results(run_dir, result, wall_seconds=wall_seconds)
    print(f"run: {run_dir}\nresults digest: {digest}")


if __name__ == "__main__":
    main()
