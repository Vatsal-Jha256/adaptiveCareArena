"""Evaluation metrics computed post-hoc from logged run history. Ground
truth (true_state, converted to true_need = -true_state so "higher is
worse" matches predicted_need's convention) is read here and only here
outside world/ -- policies and estimators never see it during the run.

Regret vs. oracle, allocation fairness (check-rate parity) and
information fairness (estimator-error parity conditional on check count)
are reported as two different quantities, never one alone, because they
are dissociable."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def compute_all_metrics(history: list[dict], subject_specs, budget: int) -> dict:
    """Compute regret, check rates, and allocation/information fairness from a run's history.

    Allocation parity and estimation-error parity are reported as separate
    quantities, because they are dissociable.
    """
    groups = sorted({spec.group for spec in subject_specs})
    group_of = {spec.subject_id: spec.group for spec in subject_specs}
    n_subjects = len(subject_specs)
    n_per_group = {g: sum(1 for s in subject_specs if s.group == g) for g in groups}

    regret = _regret_vs_oracle(history, budget)
    check_rates = _group_check_rate_over_time(history, group_of, n_per_group, groups)
    allocation_fairness = _allocation_fairness_summary(check_rates, groups)
    information_fairness = _information_fairness_by_group(history, group_of, groups)

    return {
        "n_subjects": n_subjects,
        "n_rounds": len(history),
        "n_per_group": n_per_group,
        "cumulative_regret_vs_oracle": regret["cumulative"],
        "regret_per_round": regret["per_round"],
        "group_check_rate_over_time": check_rates,
        "allocation_fairness": allocation_fairness,
        "information_fairness": information_fairness,
    }


def _regret_vs_oracle(history: list[dict], budget: int) -> dict:
    """Oracle is computed directly from logged ground truth after the run
    -- never a live policy, never influences the causal path."""
    per_round = []
    for rec in history:
        true_need = {sid: -s for sid, s in rec["true_state"].items()}
        oracle_ids = sorted(true_need, key=lambda s: true_need[s], reverse=True)[:budget]
        oracle_mass = sum(true_need[s] for s in oracle_ids)
        actual_mass = sum(true_need.get(s, 0.0) for s in rec["checked"])
        per_round.append(oracle_mass - actual_mass)
    return {"per_round": per_round, "cumulative": float(np.sum(per_round))}


def _group_check_rate_over_time(
    history: list[dict],
    group_of: dict[int, str],
    n_per_group: dict[str, int],
    groups: list[str],
    window: int = 20,
) -> dict:
    n_rounds = len(history)
    checks_in_round = {g: [] for g in groups}
    for rec in history:
        counts = {g: 0 for g in groups}
        for sid in rec["checked"]:
            counts[group_of[sid]] += 1
        for g in groups:
            checks_in_round[g].append(counts[g])

    series: dict[str, list[float]] = {}
    for g in groups:
        arr = np.array(checks_in_round[g], dtype=float)
        possible = n_per_group[g]
        rolled = []
        for t in range(n_rounds):
            lo = max(0, t - window + 1)
            rate = arr[lo : t + 1].sum() / (possible * (t - lo + 1))
            rolled.append(float(rate))
        series[g] = rolled
    return series


def _allocation_fairness_summary(check_rates: dict, groups: list[str]) -> dict:
    """Check-rate parity across severity-tercile groups."""
    final_rates = {g: (check_rates[g][-1] if check_rates[g] else None) for g in groups}
    valid = [v for v in final_rates.values() if v is not None]
    max_gap = float(max(valid) - min(valid)) if len(valid) >= 2 else 0.0
    return {"final_check_rate_by_group": final_rates, "max_pairwise_gap": max_gap}


def _information_fairness_by_group(
    history: list[dict],
    group_of: dict[int, str],
    groups: list[str],
    buckets: tuple[int, ...] = (0, 1, 5, 20),
) -> dict:
    """Parity of estimator error across groups, conditional on check
    count -- a policy can hit check-rate parity while one group's
    estimator remains systematically worse per check it receives."""
    hi_bounds = buckets[1:] + (10**9,)
    labels = [f"{lo}-{hi}" for lo, hi in zip(buckets, hi_bounds, strict=False)]

    def bucket_for(n: int) -> str:
        """Return the label of the check-count bucket containing ``n``."""
        for lo, hi, lab in zip(buckets, hi_bounds, labels, strict=False):
            if lo <= n < hi:
                return lab
        return labels[-1]

    check_counts: dict[int, int] = {}
    errors_by_group_bucket: dict[str, dict[str, list[float]]] = {
        g: {lab: [] for lab in labels} for g in groups
    }
    errors_by_group_overall: dict[str, list[float]] = {g: [] for g in groups}

    for rec in history:
        predicted = rec["predicted_need"]
        true_need = {sid: -s for sid, s in rec["true_state"].items()}
        for sid, p in predicted.items():
            g = group_of[sid]
            err = abs(p - true_need.get(sid, p))
            errors_by_group_overall[g].append(err)
            n = check_counts.get(sid, 0)
            errors_by_group_bucket[g][bucket_for(n)].append(err)
        for sid in rec["checked"]:
            check_counts[sid] = check_counts.get(sid, 0) + 1

    overall = {g: (float(np.mean(v)) if v else None) for g, v in errors_by_group_overall.items()}
    by_bucket = {
        g: {lab: (float(np.mean(v)) if v else None) for lab, v in bd.items()}
        for g, bd in errors_by_group_bucket.items()
    }

    valid_overall = [v for v in overall.values() if v is not None]
    max_gap_overall = (
        float(max(valid_overall) - min(valid_overall)) if len(valid_overall) >= 2 else 0.0
    )

    return {
        "mean_error_by_group": overall,
        "mean_error_by_group_and_checkcount_bucket": by_bucket,
        "max_pairwise_gap_overall": max_gap_overall,
    }


def save_history_and_metrics(run_dir: Path, history: list[dict], metrics: dict) -> None:
    """Write ``metrics.json`` and a JSON-safe copy of the history into ``run_dir``."""
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    serializable = [
        {
            "round": rec["round"],
            "checked": rec["checked"],
            "true_state": {str(k): v for k, v in rec["true_state"].items()},
            "predicted_need": {str(k): v for k, v in rec["predicted_need"].items()},
        }
        for rec in history
    ]
    with open(run_dir / "history.json", "w") as f:
        json.dump(serializable, f)
