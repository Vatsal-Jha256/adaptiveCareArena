"""Test fixture -- the driver loop for the toy restless-arm world.

This was stage2_env/experiment.py. It is a fixture now, not a
production runner: the real Stage 2 runner is
`stage2_env/replay_experiment.py`, which replays actual recorded
CrossCheck trajectories. Kept only so the environment and policy
mechanics tests can run cheaply without touching restricted data.

Nothing here may be imported by production code and no config can
name it. Numbers it produces are not results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from runtime.manifest import load_config, make_run_dir, write_manifest, write_results

# Import for @registry.register(...) side effects -- the only place these
# need to be imported by name.
from stage2_env import estimators as _estimators  # noqa: F401
from stage2_env import policies as _policies  # noqa: F401
from stage2_env.core import ESTIMATORS, POLICIES
from stage2_env.metrics import compute_all_metrics, save_history_and_metrics
from tests.fixtures.toy_restless_world import GroupSpec, RestlessSubjectWorld, build_subjects


def run_experiment(cfg: dict, run_dir: Path | None = None) -> dict:
    seed = cfg["seed"]
    rng = np.random.default_rng(seed)

    world_cfg = cfg["world"]
    groups = [GroupSpec(**g) for g in world_cfg["groups"]]
    group_map = {g.name: g for g in groups}
    subject_specs = build_subjects(groups, world_cfg["n_subjects"], rng)
    world = RestlessSubjectWorld(subject_specs, group_map, rng)

    budget = cfg["budget"]["b_per_round"]
    n_rounds = world_cfg["n_rounds"]

    est_cls = ESTIMATORS.get(cfg["estimator"]["name"])
    estimator = est_cls(**cfg["estimator"].get("params", {}))

    policy_cls = POLICIES.get(cfg["policy"]["name"])
    policy = policy_cls(**cfg["policy"].get("params", {}))

    check_counts: dict[int, int] = {spec.subject_id: 0 for spec in subject_specs}
    history: list[dict] = []

    for r in range(n_rounds):
        features = world.current_features(r, n_rounds, check_counts)
        predicted_need = estimator.predict(features)
        checked = policy.select(features, predicted_need, budget, r, rng)

        observations, true_state_now = world.pull_and_advance(r, checked, features)
        estimator.update(observations)

        for sid in checked:
            check_counts[sid] = check_counts.get(sid, 0) + 1

        history.append(
            {
                "round": r,
                "checked": sorted(checked),
                "true_state": true_state_now,
                "predicted_need": predicted_need,
            }
        )

    metrics = compute_all_metrics(history, subject_specs, budget)

    if run_dir is not None:
        save_history_and_metrics(run_dir, history, metrics)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default=None, help="defaults to outputs/<experiment_id>.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["_source_path"] = args.config

    if args.out:
        # Explicit --out keeps the old flat-file behaviour for scripted
        # sweeps that manage their own layout; no manifest is written in
        # that case, and the path is not immutability-guarded.
        metrics = run_experiment(cfg)
        out_path = args.out
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
    else:
        run_dir = make_run_dir(cfg, base="outputs")
        write_manifest(run_dir, cfg)
        metrics = run_experiment(cfg)
        digest = write_results(run_dir, metrics)
        out_path = run_dir / "metrics.json"
        print(f"results digest: {digest}")

    print(f"run: {out_path}")
    print(
        f"n_subjects={metrics['n_subjects']}  n_rounds={metrics['n_rounds']}  budget={cfg['budget']['b_per_round']}"
    )
    print(f"cumulative_regret_vs_oracle={metrics['cumulative_regret_vs_oracle']:.2f}")
    af = metrics["allocation_fairness"]
    print(
        f"allocation_fairness: max_gap={af['max_pairwise_gap']:.4f}  by_group={af['final_check_rate_by_group']}"
    )
    inf = metrics["information_fairness"]
    print(
        f"information_fairness: max_gap={inf['max_pairwise_gap_overall']:.4f}  by_group={inf['mean_error_by_group']}"
    )


if __name__ == "__main__":
    main()
