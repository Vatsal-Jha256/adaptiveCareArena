"""Cross-fitted measurement-to-review benchmark with fixed eligibility.

Partial item vectors are mapped to the concurrently observed full composite by
Ridge calibration trained on other participants. The calibrated value updates
the belief used for ranking. Eligibility and episode labels are built once from
the full recorded trajectory and are therefore identical across measurement
arms. This is retrospective masking of complete reports, not evidence about a
prospective questionnaire policy or unobserved outcomes.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from sklearn.linear_model import Ridge

from runtime.manifest import load_config, make_run_dir, write_manifest, write_results
from scripts.forward_benchmark import queue_bootstrap_budgets
from scripts.questionnaire_panel import load_questionnaire_panel
from scripts.report_landmarks import build_report_landmarks


def participant_folds(subjects: list, n_folds: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(subjects))
    return {subject: index % n_folds for index, subject in enumerate(shuffled)}


def greedy_items(x: np.ndarray, y: np.ndarray, k: int, ridge_alpha: float) -> list[int]:
    """Forward-select k items minimizing training reconstruction error of y.

    Uses only the rows it is given, so called per fold on training participants
    it never sees the held-out participants or any outcome label."""
    chosen: list[int] = []
    for _ in range(k):
        best, best_err = None, np.inf
        for j in range(x.shape[1]):
            if j in chosen:
                continue
            cols = chosen + [j]
            fit = Ridge(alpha=ridge_alpha).fit(x[:, cols], y)
            err = float(np.mean((fit.predict(x[:, cols]) - y) ** 2))
            if err < best_err:
                best, best_err = j, err
        chosen.append(best)
    return chosen


def calibrated_trajectories(
    subject_rows: dict,
    order: list[int] | None,
    k: int,
    folds: dict,
    n_folds: int,
    ridge_alpha: float,
) -> tuple[dict, dict]:
    """Return full truth and participant-cross-fitted partial-score predictions."""
    max_day = max(day for rows in subject_rows.values() for day, _ in rows)
    truth: dict[int, list] = {}
    belief: dict[int, list] = {}
    subject_index = {subject: index for index, subject in enumerate(sorted(subject_rows))}
    for held_out in range(n_folds):
        full_x = np.vstack(
            [
                values
                for subject, rows in subject_rows.items()
                if folds[subject] != held_out
                for _, values in rows
            ]
        )
        train_y = np.array(
            [
                values.sum()
                for subject, rows in subject_rows.items()
                if folds[subject] != held_out
                for _, values in rows
            ]
        )
        selected = (
            list(order[:k]) if order is not None else greedy_items(full_x, train_y, k, ridge_alpha)
        )
        model = Ridge(alpha=ridge_alpha).fit(full_x[:, selected], train_y)
        for subject, rows in subject_rows.items():
            if folds[subject] != held_out:
                continue
            idx = subject_index[subject]
            truth[idx] = [None] * (max_day + 1)
            belief[idx] = [None] * (max_day + 1)
            x = np.vstack([values[selected] for _, values in rows])
            predicted = model.predict(x)
            for (day, values), estimate in zip(rows, predicted, strict=True):
                truth[idx][day] = float(values.sum())
                belief[idx][day] = float(estimate)
    return truth, belief


def score_diagnostics(truth: dict, belief: dict) -> dict:
    observed = [
        (truth[i][d], belief[i][d])
        for i in truth
        for d in range(len(truth[i]))
        if truth[i][d] is not None
    ]
    y = np.array([x[0] for x in observed])
    p = np.array([x[1] for x in observed])
    return {
        "mae": float(np.mean(np.abs(y - p))),
        "bias": float(np.mean(p - y)),
        "truth_range": [float(y.min()), float(y.max())],
        "prediction_range": [float(p.min()), float(p.max())],
        "n_reports": len(observed),
    }


def full_trajectories(subject_rows: dict) -> dict[int, list]:
    max_day = max(day for rows in subject_rows.values() for day, _ in rows)
    output = {}
    for index, subject in enumerate(sorted(subject_rows)):
        output[index] = [None] * (max_day + 1)
        for day, values in subject_rows[subject]:
            output[index][day] = float(values.sum())
    return output


def run(cfg: dict) -> dict:
    dataset = cfg["data"]["source"]
    subject_rows, items, default_threshold = load_questionnaire_panel(
        dataset, align="calendar", local_root=cfg["data"].get("local_root")
    )
    threshold = float(cfg["outcome"].get("threshold", default_threshold))
    horizon = int(cfg["outcome"]["horizon_days"])
    subjects = sorted(subject_rows)
    folds = participant_folds(subjects, int(cfg["calibration"]["n_folds"]), int(cfg["seed"]))
    fixed_order = list(range(len(items)))
    random_order = list(np.random.default_rng(int(cfg["seed"]) + 1).permutation(len(items)))
    ks = [len(items) if k == "full" else int(k) for k in cfg["measurement_budgets"]]

    # Full information supplies the one shared eligibility/outcome table.
    truth = full_trajectories(subject_rows)
    full_belief = {subject: list(values) for subject, values in truth.items()}
    base_rows, _ = build_report_landmarks(
        full_belief,
        threshold,
        horizon,
        None,
        truth=truth,
        risk="report",
        at_risk=False,
    )
    keyed = {(row["i"], row["r"]): row for row in base_rows}
    for row in base_rows:
        row["full"] = row["ema3"]

    diagnostics = {"full": score_diagnostics(truth, full_belief)}
    models = ["full"]
    for name, order in (("fixed", fixed_order), ("random", random_order), ("greedy", None)):
        for k in sorted(set(ks)):
            if k == len(items):
                continue
            _, belief = calibrated_trajectories(
                subject_rows,
                order,
                k,
                folds,
                int(cfg["calibration"]["n_folds"]),
                float(cfg["calibration"]["ridge_alpha"]),
            )
            rows, _ = build_report_landmarks(
                belief,
                threshold,
                horizon,
                None,
                truth=truth,
                risk="report",
                at_risk=False,
            )
            assert {(row["i"], row["r"]) for row in rows} == set(keyed), (
                "measurement arm changed eligibility"
            )
            model_name = f"{name}_k{k}"
            models.append(model_name)
            diagnostics[model_name] = score_diagnostics(truth, belief)
            for row in rows:
                keyed[(row["i"], row["r"])][model_name] = row["ema3"]

    comparisons = []
    for k in sorted(set(ks)):
        if k == len(items):
            continue
        comparisons.extend(
            (
                (f"fixed_k{k}", f"random_k{k}"),
                (f"fixed_k{k}", "full"),
                (f"greedy_k{k}", f"fixed_k{k}"),
                (f"greedy_k{k}", "full"),
            )
        )
    result = {
        "evaluation_type": "retrospective complete-report masking and report-landmark replay",
        "outcome": "observed self-report threshold-crossing proxy",
        "eligibility": "shared full-trajectory report-day landmarks; independent of partial measurement arm",
        "calibration": "participant-cross-fitted Ridge from partial items to concurrent full composite",
        "selection": {
            "fixed": "declared questionnaire order; not optimized on outcomes",
            "random": "one seed-fixed order shared across participants",
            "greedy": "per-fold forward selection minimizing training reconstruction error of the composite; label-free, held-out participants never seen",
            "full": "all recorded items; descriptive ceiling",
        },
        "dataset": dataset,
        "items": list(items),
        "subjects": len(subjects),
        "candidate_rows": len(base_rows),
        "eligible_episodes": len({x["ep"] for x in base_rows if x["onset"]}),
        "score_diagnostics": diagnostics,
        "by_review_budget": queue_bootstrap_budgets(
            base_rows,
            sorted(truth),
            tuple(cfg["review_budgets"]),
            tuple(models),
            tuple(comparisons),
            int(cfg["inference"]["n_boot"]),
            int(cfg["seed"]),
        ),
    }
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
