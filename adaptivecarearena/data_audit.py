"""Aggregate-only preflight for a locally obtained benchmark cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from runtime.manifest import load_config
from runtime.reproducibility import data_fingerprint
from scripts.report_landmarks import KALMAN_PARAMETERS, build_report_landmarks
from stage2_env.replay_experiment import load_panel


def audit_config(config: dict[str, Any]) -> dict[str, Any]:
    """Load the real adapter and return no participant-level information."""
    data_cfg = dict(config["data"], align="calendar", horizon=None)
    _, _, trajectories, _ = load_panel(data_cfg)
    report_counts = [sum(value is not None for value in series) for series in trajectories.values()]
    n_rounds = max((len(series) for series in trajectories.values()), default=0)
    threshold = float(config["outcome"]["threshold"])
    horizon = int(config["outcome"]["horizon_days"])
    rows, raw_episodes = build_report_landmarks(
        trajectories,
        threshold,
        horizon,
        KALMAN_PARAMETERS.get(config["data"]["source"]),
        risk="report",
        at_risk=True,
    )
    fingerprint = data_fingerprint(config["data"]["local_root"])
    return {
        "status": "passed_adapter_and_task_preflight",
        "scope": "aggregate audit only; no participant rows or identifiers emitted",
        "dataset": config["data"]["source"],
        "alignment": "calendar",
        "data_root_exists": fingerprint["exists"],
        "data_file_count": fingerprint["n_files"],
        "data_combined_sha256": fingerprint["combined_sha256"],
        "participants": len(trajectories),
        "calendar_rounds": n_rounds,
        "recorded_reports": sum(report_counts),
        "minimum_reports_per_participant": min(report_counts) if report_counts else None,
        "maximum_reports_per_participant": max(report_counts) if report_counts else None,
        "empty_trajectory_count": sum(count == 0 for count in report_counts),
        "candidate_landmarks": len(rows),
        "raw_proxy_episodes": len(raw_episodes),
        "eligible_proxy_episodes": len({row["ep"] for row in rows if row["onset"]}),
        "threshold": threshold,
        "horizon_days": horizon,
    }


def main() -> None:
    """Audit the cohort named by ``--config`` and print an aggregate-only report."""
    parser = argparse.ArgumentParser(
        description="Audit a locally obtained real cohort without exporting participant rows."
    )
    parser.add_argument("--config", required=True, help="committed YAML config")
    parser.add_argument("--output", type=Path, help="optional aggregate JSON destination")
    args = parser.parse_args()
    config = load_config(args.config)
    report = audit_config(config)
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
