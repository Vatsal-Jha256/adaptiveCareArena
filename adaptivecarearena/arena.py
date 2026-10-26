"""External-method evaluation for the report-landmark benchmark."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from adaptivecarearena.composed import sequential_calibrated_trajectories
from adaptivecarearena.interfaces import MeasurementPolicy, ReviewPolicy, StateEstimator
from runtime.manifest import load_config
from scripts.composition_benchmark import full_trajectories, participant_folds, score_diagnostics
from scripts.forward_benchmark import queue_bootstrap_budgets
from scripts.questionnaire_panel import load_questionnaire_panel
from scripts.report_landmarks import KALMAN_PARAMETERS, build_report_landmarks
from stage2_env.replay_experiment import load_panel


class _IdentityReviewPolicy:
    def reset(self) -> None:
        """No state to clear: the identity policy ranks purely by estimated need."""
        pass

    def priority(self, subject_id: int, prediction_time: int, estimated_need: float) -> float:
        """Return ``estimated_need`` unchanged, so review priority equals need."""
        return estimated_need


@dataclass(frozen=True)
class MethodArm:
    """One estimator-policy arm evaluated on the common task."""

    estimator: StateEstimator
    review_policy: ReviewPolicy | None = None


@dataclass(frozen=True)
class EvaluationResult:
    """Immutable aggregate output; no participant rows are exposed."""

    summary: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict copy of the aggregate summary."""
        return dict(self.summary)


class AdaptiveCareArena:
    """Evaluate external estimators under the audited forward-review task.

    Parameters are read from a committed benchmark config. Raw source data stay
    at ``data.local_root`` and are never copied into the result.
    """

    def __init__(self, config: str | Path | Mapping[str, Any]):
        self._config = load_config(str(config)) if isinstance(config, (str, Path)) else dict(config)
        self._validate_config()

    def _validate_config(self) -> None:
        required = {"data", "outcome", "inference", "seed"}
        missing = required.difference(self._config)
        if missing:
            raise ValueError(f"benchmark config is missing: {sorted(missing)}")

    def evaluate(
        self,
        estimators: Mapping[str, StateEstimator | MethodArm],
        *,
        comparisons: tuple[tuple[str, str], ...] | None = None,
    ) -> EvaluationResult:
        """Evaluate estimator arms with paired queue-rebuilding inference.

        Estimator names identify result columns and must be unique, non-reserved
        lowercase snake_case strings. Comparisons default to every non-first arm
        against the first arm, preserving a predeclared reference comparator.
        """
        if len(estimators) < 2:
            raise ValueError("evaluation requires at least two estimator arms")
        if "budgets" not in self._config:
            raise ValueError("forward evaluation config is missing: ['budgets']")
        names = tuple(estimators)
        arms = {
            name: value if isinstance(value, MethodArm) else MethodArm(value)
            for name, value in estimators.items()
        }
        for name, arm in arms.items():
            if not name or not name.isidentifier() or name.lower() != name:
                raise ValueError(f"invalid estimator name: {name!r}")
            if not isinstance(arm.estimator, StateEstimator):
                raise TypeError(f"estimator {name!r} does not implement StateEstimator")
            policy = arm.review_policy or _IdentityReviewPolicy()
            if not isinstance(policy, ReviewPolicy):
                raise TypeError(f"review policy for {name!r} does not implement ReviewPolicy")
        if comparisons is None:
            reference = names[0]
            comparisons = tuple((name, reference) for name in names[1:])
        unknown = {name for pair in comparisons for name in pair}.difference(names)
        if unknown:
            raise ValueError(f"comparisons reference unknown estimators: {sorted(unknown)}")

        cfg = self._config
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
            external_estimators={name: arm.estimator for name, arm in arms.items()},
        )
        for name, arm in arms.items():
            policy = arm.review_policy or _IdentityReviewPolicy()
            policy.reset()
            for row in rows:
                priority = float(policy.priority(row["i"], row["r"], row[name]))
                if not np.isfinite(priority):
                    raise ValueError(
                        f"review policy for {name!r} returned non-finite priority "
                        f"for subject={row['i']}, prediction_time={row['r']}"
                    )
                row[name] = priority
        subjects = sorted(trajectories)
        eligible_episodes = len({row["ep"] for row in rows if row["onset"]})
        output: dict[str, Any] = {
            "evaluation_type": "retrospective report-landmark replay",
            "outcome": "observed self-report threshold-crossing proxy",
            "scope": "not diagnostic, causal, treatment, or deployment validation",
            "dataset": cfg["data"]["source"],
            "threshold": threshold,
            "horizon_days": horizon,
            "subjects": len(subjects),
            "candidate_rows": len(rows),
            "raw_episodes": len(raw_episodes),
            "eligible_episodes": eligible_episodes,
            "estimators": list(names),
            "comparisons": [list(pair) for pair in comparisons],
        }
        if eligible_episodes == 0:
            output["status"] = "not_applicable_no_eligible_episodes"
            output["by_budget"] = {}
        else:
            output["status"] = "evaluated"
            output["by_budget"] = queue_bootstrap_budgets(
                rows,
                subjects,
                tuple(cfg["budgets"]),
                names,
                comparisons,
                int(cfg["inference"]["n_boot"]),
                int(cfg["seed"]),
            )
        return EvaluationResult(MappingProxyType(output))

    def evaluate_measurement_policies(
        self,
        policies: Mapping[str, MeasurementPolicy],
        *,
        measurement_budget: int,
        comparisons: tuple[tuple[str, str], ...] | None = None,
    ) -> EvaluationResult:
        """Compose external sequential item policies with review evaluation."""
        if measurement_budget <= 0:
            raise ValueError("measurement_budget must be positive")
        if not policies:
            raise ValueError("at least one measurement policy is required")
        cfg = self._config
        required = {"measurement_budgets", "review_budgets", "calibration"}
        missing = required.difference(cfg)
        if missing:
            raise ValueError(f"composition config is missing: {sorted(missing)}")
        dataset = cfg["data"]["source"]
        subject_rows, items, default_threshold = load_questionnaire_panel(
            dataset, align="calendar", local_root=cfg["data"].get("local_root")
        )
        if measurement_budget > len(items):
            raise ValueError("measurement_budget exceeds the instrument length")
        for name, policy in policies.items():
            if not name or not name.isidentifier() or name.lower() != name:
                raise ValueError(f"invalid measurement policy name: {name!r}")
            if not all(hasattr(policy, method) for method in ("reset", "choose")):
                raise TypeError(f"measurement policy {name!r} has an invalid interface")

        threshold = float(cfg["outcome"].get("threshold", default_threshold))
        horizon = int(cfg["outcome"]["horizon_days"])
        folds = participant_folds(
            sorted(subject_rows), int(cfg["calibration"]["n_folds"]), int(cfg["seed"])
        )
        truth = full_trajectories(subject_rows)
        base_rows, _ = build_report_landmarks(
            truth, threshold, horizon, None, truth=truth, risk="report", at_risk=False
        )
        keyed = {(row["i"], row["r"]): row for row in base_rows}
        for row in base_rows:
            row["full"] = row["ema3"]
        diagnostics = {"full": score_diagnostics(truth, truth)}
        policy_diagnostics = {}
        model_names = ["full"]
        for name, policy in policies.items():
            arm_truth, belief, reveal_diagnostics = sequential_calibrated_trajectories(
                subject_rows,
                items,
                policy,
                float(measurement_budget),
                folds,
                int(cfg["calibration"]["n_folds"]),
                float(cfg["calibration"]["ridge_alpha"]),
            )
            if arm_truth != truth:
                raise AssertionError("measurement policy changed the recorded truth trajectory")
            rows, _ = build_report_landmarks(
                belief, threshold, horizon, None, truth=truth, risk="report", at_risk=False
            )
            if {(row["i"], row["r"]) for row in rows} != set(keyed):
                raise AssertionError("measurement policy changed review eligibility")
            model_names.append(name)
            diagnostics[name] = score_diagnostics(truth, belief)
            policy_diagnostics[name] = reveal_diagnostics
            for row in rows:
                keyed[(row["i"], row["r"])][name] = row["ema3"]
        if comparisons is None:
            comparisons = tuple((name, "full") for name in policies)
        unknown = {name for pair in comparisons for name in pair}.difference(model_names)
        if unknown:
            raise ValueError(f"comparisons reference unknown arms: {sorted(unknown)}")
        result = {
            "evaluation_type": "sequential recorded-item replay composed with report-landmark review",
            "scope": "retrospective proxy evaluation; not intervention or deployment validation",
            "dataset": dataset,
            "items": list(items),
            "measurement_budget": measurement_budget,
            "subjects": len(subject_rows),
            "candidate_rows": len(base_rows),
            "eligible_episodes": len({row["ep"] for row in base_rows if row["onset"]}),
            "score_diagnostics": diagnostics,
            "policy_diagnostics": policy_diagnostics,
            "by_review_budget": queue_bootstrap_budgets(
                base_rows,
                sorted(truth),
                tuple(cfg["review_budgets"]),
                tuple(model_names),
                comparisons,
                int(cfg["inference"]["n_boot"]),
                int(cfg["seed"]),
            ),
        }
        return EvaluationResult(MappingProxyType(result))
