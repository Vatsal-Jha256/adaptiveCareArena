"""Construct leakage-safe report-landmark rows for the benchmark runners.

This module contains task construction only. Queue allocation and inference live
in ``forward_benchmark.py`` so legacy exploratory runners cannot become an
implicit source of truth for current artifacts.
"""

from __future__ import annotations

import bisect
from collections.abc import Mapping
from typing import Protocol

import numpy as np

MIN_HISTORY = 6
KALMAN_PARAMETERS = {"crosscheck": (0.02, 0.25), "ces": None}


class ExternalStateEstimator(Protocol):
    """Decision-time-only interface used by the public benchmark API."""

    def reset(self) -> None: ...

    def observe(self, subject_id: int, observed_at: int, value: float) -> None: ...

    def score(self, subject_id: int, prediction_time: int) -> float: ...


def deterministic_jitter(subject: int, day: int, seed: int = 0) -> float:
    """Return a tiny stable value used only to resolve exact ranking ties."""
    value = (subject * 2654435761 + day * 40503 + seed * 97) & 0xFFFFFFFF
    value ^= value >> 15
    value = (value * 2246822519) & 0xFFFFFFFF
    value ^= value >> 13
    return (value % 1000003) / 1000003 * 1e-9


def build_report_landmarks(
    trajectories: dict,
    threshold: float,
    horizon: int,
    kalman: tuple[float, float] | None,
    truth: dict | None = None,
    risk: str = "report",
    at_risk: bool = True,
    external_estimators: Mapping[str, ExternalStateEstimator] | None = None,
) -> tuple[list[dict], set[tuple[int, int]]]:
    """Build decision-time rows and observed future threshold-crossing episodes.

    ``trajectories`` are values visible to the belief; ``truth`` supplies the
    recorded outcome stream when retrospective measurement masking is used.
    ``risk='report'`` is the operational task: landmark membership depends
    only on a report observed on the decision day.
    """
    if risk not in {"report", "observable", "active", "all"}:
        raise ValueError(f"unknown risk set: {risk}")
    truth = trajectories if truth is None else truth
    external_estimators = dict(external_estimators or {})
    reserved = {
        "i",
        "r",
        "last",
        "lastraw",
        "ema3",
        "ema6",
        "ema15",
        "kal",
        "z",
        "pool",
        "onset",
        "ep",
    }
    collisions = reserved.intersection(external_estimators)
    if collisions:
        raise ValueError(f"external estimator names are reserved: {sorted(collisions)}")
    for estimator in external_estimators.values():
        estimator.reset()
    n_days = max(len(values) for values in trajectories.values())
    reports = {
        subject: [(day, value) for day, value in enumerate(values) if value is not None]
        for subject, values in trajectories.items()
    }
    report_days = {subject: np.array([day for day, _ in rows]) for subject, rows in reports.items()}
    report_truth = {
        subject: np.array([truth[subject][day] for day, _ in rows])
        for subject, rows in reports.items()
    }
    reports_by_day: dict[int, list[float]] = {}
    for rows in reports.values():
        for day, value in rows:
            reports_by_day.setdefault(day, []).append(value)

    state = {
        subject: {
            "count": 0,
            "ema3": None,
            "ema6": None,
            "ema15": None,
            "last": None,
            "last_truth": None,
            "kalman_mean": None,
            "kalman_var": None,
            "history": [],
            "last_day": -1,
        }
        for subject in reports
    }
    raw_episodes: set[tuple[int, int]] = set()
    for subject, rows in reports.items():
        for index in range(MIN_HISTORY, len(rows)):
            if report_truth[subject][index - 1] > threshold >= report_truth[subject][index]:
                raw_episodes.add((subject, int(report_days[subject][index])))

    output: list[dict] = []
    pooled_history: list[float] = []
    for day in range(n_days):
        for subject in reports:
            current = state[subject]
            days = report_days[subject]
            report_index = np.searchsorted(days, day)
            report_today = report_index < len(days) and days[report_index] == day
            if report_today:
                value = trajectories[subject][day]
                for estimator in external_estimators.values():
                    estimator.observe(subject, day, float(value))
                need = -value
                for key, alpha in (("ema3", 0.3), ("ema6", 0.6), ("ema15", 0.15)):
                    current[key] = (
                        need if current[key] is None else alpha * need + (1 - alpha) * current[key]
                    )
                if kalman is not None:
                    process_var, observation_var = kalman
                    if current["kalman_mean"] is None:
                        current["kalman_mean"], current["kalman_var"] = need, 1.0
                    else:
                        predicted_var = current["kalman_var"] + process_var
                        gain = predicted_var / (predicted_var + observation_var)
                        current["kalman_mean"] += gain * (need - current["kalman_mean"])
                        current["kalman_var"] = (1 - gain) * predicted_var
                previous = list(current["history"])
                current["history"].append(need)
                current["z_reference"] = (
                    (np.mean(previous), np.std(previous) + 1e-6) if len(previous) >= 3 else None
                )
                current["last"] = value
                current["last_truth"] = truth[subject][day]
                current["last_day"] = day
                current["count"] += 1

            if current["count"] < MIN_HISTORY:
                continue
            window_start = np.searchsorted(days, day, side="right")
            window_end = np.searchsorted(days, day + horizon, side="right")
            if at_risk and current["last"] <= threshold:
                continue
            if risk == "report" and not report_today:
                continue
            if risk == "observable" and window_end <= window_start:
                continue
            if risk == "active" and day > days[-1]:
                continue

            future_values = report_truth[subject][window_start:window_end]
            severe_offsets = np.where(future_values <= threshold)[0]
            onset = int(current["last_truth"] > threshold and len(severe_offsets) > 0)
            episode = (subject, int(days[window_start + severe_offsets[0]])) if onset else None
            z_reference = current.get("z_reference")
            z_score = (-current["last"] - z_reference[0]) / z_reference[1] if z_reference else 0.0
            percentile = (
                bisect.bisect_left(pooled_history, -current["last"]) / len(pooled_history)
                if len(pooled_history) > 20
                else 0.5
            )
            row = {
                "i": subject,
                "r": day,
                "last": -current["last"],
                "lastraw": current["last"],
                "ema3": current["ema3"],
                "ema6": current["ema6"],
                "ema15": current["ema15"],
                "kal": current["kalman_mean"] if kalman is not None else current["ema3"],
                "z": z_score,
                "pool": percentile,
                "onset": onset,
                "ep": episode,
            }
            for name, estimator in external_estimators.items():
                score = float(estimator.score(subject, day))
                if not np.isfinite(score):
                    raise ValueError(
                        f"external estimator {name!r} returned non-finite score "
                        f"for subject={subject}, prediction_time={day}"
                    )
                row[name] = score
            output.append(row)
        for value in reports_by_day.get(day, []):
            bisect.insort(pooled_history, -value)
    return output, raw_episodes
