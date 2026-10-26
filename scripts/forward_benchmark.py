"""Leakage-safe, config-driven forward benchmark runner.

The primary estimand uses report-day landmarks: candidate membership is known
at the decision time and does not use future attendance or a participant's
eventual final report. Participant bootstrap replicates rebuild the complete
capacity-coupled queue, including participants with no onset episodes.

This is retrospective replay against an observed self-report proxy. It is not
diagnostic, causal, treatment, or deployment validation.
"""

from __future__ import annotations

import argparse
import time
from copy import deepcopy

import numpy as np

from runtime.manifest import load_config, make_run_dir, write_manifest, write_results
from scripts.report_landmarks import (
    KALMAN_PARAMETERS,
    build_report_landmarks,
    deterministic_jitter,
)
from stage2_env.replay_experiment import load_panel


def _queue(
    rows: list[dict], budget: int | float, models: tuple[str, ...], seed: int
) -> dict[str, set]:
    """Return captured episode ids after rebuilding every daily queue."""
    by_day: dict[int, list[dict]] = {}
    for row in rows:
        by_day.setdefault(row["r"], []).append(row)
    captured = {model: set() for model in models}
    for day, candidates in by_day.items():
        take = budget if isinstance(budget, int) else max(1, int(np.ceil(budget * len(candidates))))
        tie = np.array([deterministic_jitter(x["i"], day, seed=seed) for x in candidates])
        for model in models:
            score = np.array([x[model] for x in candidates], dtype=float) + tie
            for index in np.argsort(-score)[:take]:
                row = candidates[int(index)]
                if row["onset"]:
                    captured[model].add(row["ep"])
    return captured


def _recalls(
    rows: list[dict], budget: int | float, models: tuple[str, ...], seed: int
) -> dict[str, float]:
    episodes = {x["ep"] for x in rows if x["onset"]}
    if not episodes:
        return {model: float("nan") for model in models}
    captured = _queue(rows, budget, models, seed)
    return {model: len(captured[model]) / len(episodes) for model in models}


def _burden(rows: list[dict], budget: int | float) -> dict[str, float | int]:
    counts: dict[int, int] = {}
    for row in rows:
        counts[row["r"]] = counts.get(row["r"], 0) + 1
    nominal = 0
    selected = 0
    for count in counts.values():
        take = budget if isinstance(budget, int) else max(1, int(np.ceil(budget * count)))
        nominal += take
        selected += min(take, count)
    return {
        "selected_landmarks": selected,
        "nominal_capacity": nominal,
        "capacity_utilization": selected / nominal if nominal else float("nan"),
        "allocation_days": len(counts),
    }


def _gini(values: list[int]) -> float | None:
    """Return concentration of non-negative counts, or NA for zero total."""
    array = np.sort(np.asarray(values, dtype=float))
    total = float(array.sum())
    if len(array) == 0 or total == 0.0:
        return None
    weights = np.arange(1, len(array) + 1, dtype=float)
    return float(
        (2.0 * np.dot(weights, array) / (len(array) * total)) - (len(array) + 1) / len(array)
    )


def _decision_metrics(
    rows: list[dict],
    subjects: list[int],
    budget: int | float,
    models: tuple[str, ...],
    seed: int,
) -> dict[str, dict[str, float | int | None | str]]:
    """Complement recall with burden efficiency, lead time, and concentration."""
    by_day: dict[int, list[dict]] = {}
    for row in rows:
        by_day.setdefault(row["r"], []).append(row)
    selected = {model: [] for model in models}
    for day, candidates in by_day.items():
        take = budget if isinstance(budget, int) else max(1, int(np.ceil(budget * len(candidates))))
        tie = np.array([deterministic_jitter(row["i"], day, seed=seed) for row in candidates])
        for model in models:
            score = np.array([row[model] for row in candidates], dtype=float) + tie
            selected[model].extend(candidates[int(index)] for index in np.argsort(-score)[:take])

    output = {}
    for model, chosen in selected.items():
        positive = [row for row in chosen if row["onset"]]
        captured = {row["ep"] for row in positive}
        lead_by_episode: dict[tuple[int, int], int] = {}
        for row in positive:
            lead = int(row["ep"][1] - row["r"])
            lead_by_episode[row["ep"]] = max(lead_by_episode.get(row["ep"], 0), lead)
        counts = {subject: 0 for subject in subjects}
        for row in chosen:
            counts[row["i"]] = counts.get(row["i"], 0) + 1
        output[model] = {
            "selected_landmarks": len(chosen),
            "captured_episodes": len(captured),
            "positive_landmarks_selected": len(positive),
            "review_yield": len(positive) / len(chosen) if chosen else None,
            "review_yield_status": "evaluated" if chosen else "not_applicable_no_reviews",
            "mean_episode_lead_time_days": (
                float(np.mean(list(lead_by_episode.values()))) if lead_by_episode else None
            ),
            "lead_time_status": (
                "evaluated" if lead_by_episode else "not_applicable_no_captured_episodes"
            ),
            "attention_gini": _gini(list(counts.values())),
            "attention_gini_status": ("evaluated" if chosen else "not_applicable_no_reviews"),
        }
    return output


def resample_panel_rows(
    rows: list[dict], subjects: list[int], rng: np.random.Generator
) -> list[dict]:
    """Cluster-resample the full panel and give duplicate draws distinct ids."""
    by_subject: dict[int, list[dict]] = {subject: [] for subject in subjects}
    for row in rows:
        by_subject[row["i"]].append(row)
    sampled = rng.choice(subjects, size=len(subjects), replace=True)
    rebuilt: list[dict] = []
    for clone_id, source_id in enumerate(sampled):
        for original in by_subject[int(source_id)]:
            row = deepcopy(original)
            row["i"] = clone_id
            if row["ep"] is not None:
                row["ep"] = (clone_id, row["ep"][1])
            rebuilt.append(row)
    return rebuilt


def queue_bootstrap(
    rows: list[dict],
    subjects: list[int],
    budget: int | float,
    models: tuple[str, ...],
    comparisons: tuple[tuple[str, str], ...],
    n_boot: int,
    seed: int,
) -> dict[str, dict]:
    """Single-budget wrapper around the shared multiplicity-aware routine."""
    return queue_bootstrap_budgets(
        rows,
        subjects,
        (budget,),
        models,
        comparisons,
        n_boot,
        seed,
    )[str(budget)]


def queue_bootstrap_budgets(
    rows: list[dict],
    subjects: list[int],
    budgets: tuple[int | float, ...],
    models: tuple[str, ...],
    comparisons: tuple[tuple[str, str], ...],
    n_boot: int,
    seed: int,
) -> dict[str, dict]:
    """Shared paired bootstrap across budgets, rebuilding each sampled panel once."""
    points = {budget: _recalls(rows, budget, models, seed) for budget in budgets}
    values = {budget: {pair: [] for pair in comparisons} for budget in budgets}
    invalid = {budget: 0 for budget in budgets}
    rng = np.random.default_rng(seed)
    for _replicate in range(n_boot):
        boot_rows = resample_panel_rows(rows, subjects, rng)
        for budget in budgets:
            recall = _recalls(boot_rows, budget, models, seed)
            if not all(np.isfinite(recall[m]) for m in models):
                invalid[budget] += 1
                continue
            for pair in comparisons:
                values[budget][pair].append(recall[pair[0]] - recall[pair[1]])
    output = {}
    family_size = len(budgets) * len(comparisons)
    for budget in budgets:
        contrasts = {}
        for pair, draws in values[budget].items():
            key = f"{pair[0]}-{pair[1]}"
            delta = points[budget][pair[0]] - points[budget][pair[1]]
            if not draws:
                contrasts[key] = {"delta": delta, "verdict": "not_applicable", "n_valid": 0}
                continue
            lo, hi = np.percentile(draws, [2.5, 97.5])
            fw_lo, fw_hi = np.percentile(
                draws, [100 * 0.025 / family_size, 100 * (1.0 - 0.025 / family_size)]
            )
            deterministic = (
                np.std(draws, ddof=1) <= 1e-12 * max(1.0, abs(float(np.mean(draws))))
                if len(draws) > 1
                else True
            )
            contrasts[key] = {
                "delta": delta,
                "ci_lo": float(lo),
                "ci_hi": float(hi),
                "familywise_ci_lo": float(fw_lo),
                "familywise_ci_hi": float(fw_hi),
                "family_size": family_size,
                "nominal_verdict": "different" if lo > 0 or hi < 0 else "inconclusive",
                "verdict": (
                    "deterministic"
                    if deterministic
                    else "different"
                    if fw_lo > 0 or fw_hi < 0
                    else "inconclusive"
                ),
                "deterministic_relation": (
                    "identical"
                    if deterministic and abs(float(np.mean(draws))) <= 1e-12
                    else "constant_gap"
                    if deterministic
                    else None
                ),
                "n_valid": len(draws),
            }
        output[str(budget)] = {
            "episode_recall": points[budget],
            "contrasts": contrasts,
            "missed_episodes": {
                model: int(
                    round(
                        (1.0 - points[budget][model]) * len({x["ep"] for x in rows if x["onset"]})
                    )
                )
                for model in models
            },
            "burden": _burden(rows, budget),
            "decision_metrics": _decision_metrics(rows, subjects, budget, models, seed),
            "n_invalid": invalid[budget],
        }
    return output


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
    models = tuple(cfg["models"])
    comparisons = tuple(tuple(x) for x in cfg["comparisons"])
    subjects = sorted(trajectories)
    result = {
        "evaluation_type": "retrospective report-landmark replay",
        "outcome": "observed self-report threshold-crossing proxy",
        "eligibility": "report observed at decision time; no future-retention conditioning",
        "threshold": threshold,
        "horizon_days": horizon,
        "subjects": len(subjects),
        "candidate_rows": len(rows),
        "raw_episodes": len(raw_episodes),
        "eligible_episodes": len({x["ep"] for x in rows if x["onset"]}),
        "bootstrap": "full participant-panel resampling with daily queues rebuilt",
        "multiplicity": "Bonferroni simultaneous percentile intervals over all configured model-by-budget contrasts",
        "by_budget": {},
    }
    if result["eligible_episodes"] == 0:
        # No landmark has an observed outcome window: report not-applicable
        # instead of letting an empty denominator produce NaN downstream.
        result["status"] = "not_applicable_no_eligible_episodes"
        return result
    result["by_budget"] = queue_bootstrap_budgets(
        rows,
        subjects,
        tuple(cfg["budgets"]),
        models,
        comparisons,
        int(cfg["inference"]["n_boot"]),
        int(cfg["seed"]),
    )
    tie_salts = int(cfg["inference"].get("tie_salts", 0))
    result["tie_sensitivity"] = {}
    if tie_salts:
        for budget in cfg["budgets"]:
            observed = [_recalls(rows, budget, models, salt) for salt in range(tie_salts)]
            result["tie_sensitivity"][str(budget)] = {
                f"{left}-{right}": {
                    "min": float(min(x[left] - x[right] for x in observed)),
                    "max": float(max(x[left] - x[right] for x in observed)),
                    "n_salts": tie_salts,
                }
                for left, right in comparisons
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
