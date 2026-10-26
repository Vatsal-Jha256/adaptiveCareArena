"""Time-forward supervised rung for the leakage-safe report-landmark task.

Training landmarks and their complete horizon end before a fixed calendar
cutoff. Evaluation uses later report landmarks from the same panel. This is a
within-participant future-time task, not participant-generalization. All queue
inference is delegated to the full-panel, multiplicity-aware benchmark runner.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from runtime.manifest import load_config, make_run_dir, write_manifest, write_results
from scripts.forward_benchmark import _recalls, queue_bootstrap_budgets
from scripts.report_landmarks import KALMAN_PARAMETERS, build_report_landmarks
from stage2_env.replay_experiment import load_panel

FEATURES = ("last", "ema3", "ema6", "ema15", "z", "pool")


def split_masks(
    days: np.ndarray, horizon: int, quantile: float
) -> tuple[int, np.ndarray, np.ndarray]:
    cutoff = int(np.quantile(days, quantile))
    train = days + horizon < cutoff
    test = days >= cutoff
    return cutoff, train, test


def run(cfg: dict) -> dict:
    data_cfg = dict(cfg["data"], align="calendar", horizon=None)
    _, _, trajectories, _ = load_panel(data_cfg)
    threshold = float(cfg["outcome"]["threshold"])
    horizon = int(cfg["outcome"]["horizon_days"])
    rows, _ = build_report_landmarks(
        trajectories,
        threshold,
        horizon,
        KALMAN_PARAMETERS.get(cfg["data"]["source"]),
        risk="report",
        at_risk=True,
    )
    X = np.array([[row[name] for name in FEATURES] for row in rows], dtype=float)
    y = np.array([row["onset"] for row in rows], dtype=int)
    days = np.array([row["r"] for row in rows], dtype=int)
    cutoff, train_mask, test_mask = split_masks(
        days,
        horizon,
        float(cfg["split"]["train_quantile"]),
    )
    if len(np.unique(y[train_mask])) < 2:
        raise ValueError("training period does not contain both onset classes")
    model_cfg = cfg["model"]
    model = HistGradientBoostingClassifier(
        max_depth=int(model_cfg["max_depth"]),
        max_iter=int(model_cfg["max_iter"]),
        learning_rate=float(model_cfg["learning_rate"]),
        min_samples_leaf=int(model_cfg["min_samples_leaf"]),
        l2_regularization=float(model_cfg["l2_regularization"]),
        class_weight="balanced",
        random_state=int(cfg["seed"]),
    )
    model.fit(X[train_mask], y[train_mask])
    probabilities = model.predict_proba(X)[:, 1]
    test_rows = [
        dict(row, learned=float(score))
        for row, score, keep in zip(rows, probabilities, test_mask, strict=True)
        if keep
    ]
    models = tuple(cfg["models"])
    comparisons = tuple(tuple(x) for x in cfg["comparisons"])
    subjects = sorted(trajectories)
    by_budget = queue_bootstrap_budgets(
        test_rows,
        subjects,
        tuple(cfg["budgets"]),
        models,
        comparisons,
        int(cfg["inference"]["n_boot"]),
        int(cfg["seed"]),
    )
    tie_salts = int(cfg["inference"].get("tie_salts", 0))
    tie = {}
    for budget in cfg["budgets"]:
        observed = [_recalls(test_rows, budget, models, salt) for salt in range(tie_salts)]
        tie[str(budget)] = {
            f"{left}-{right}": {
                "min": float(min(x[left] - x[right] for x in observed)),
                "max": float(max(x[left] - x[right] for x in observed)),
                "n_salts": tie_salts,
            }
            for left, right in comparisons
        }
    return {
        "evaluation_type": "within-participant future-time report-landmark replay",
        "outcome": "observed self-report threshold-crossing proxy",
        "eligibility": "report observed at decision time; no future-retention conditioning",
        "features": list(FEATURES),
        "cutoff_day": cutoff,
        "train_rule": "landmark day plus horizon strictly before cutoff",
        "test_rule": "landmark day at or after cutoff",
        "train_rows": int(train_mask.sum()),
        "train_onsets": int(y[train_mask].sum()),
        "test_rows": len(test_rows),
        "test_episodes": len({row["ep"] for row in test_rows if row["onset"]}),
        "subjects": len(subjects),
        "bootstrap": "full participant-panel resampling with daily queues rebuilt",
        "multiplicity": "Bonferroni simultaneous percentile intervals over all configured model-by-budget contrasts",
        "by_budget": by_budget,
        "tie_sensitivity": tie,
    }


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
