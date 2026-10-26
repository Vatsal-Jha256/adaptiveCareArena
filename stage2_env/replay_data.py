"""Builds ReplaySubjectWorld inputs from real CrossCheck.

Subjects are assigned to severity terciles by their own baseline period
only -- never by their full-study mean, which would leak the outcome the
environment is about into the group label used for fairness metrics."""

from __future__ import annotations

import numpy as np

from caredata.crosscheck import CrossCheckDataset
from stage2_env.core import SubjectSpec

BASELINE_FRACTION = 0.25  # of each subject's own EMA history, earliest first


def build_replay_inputs(
    local_root: str = "data/raw/crosscheck",
    min_ema: int = 20,
    max_subjects: int | None = None,
    align: str = "relative",
    horizon: int | None = 270,
) -> tuple[list[SubjectSpec], dict[str, object], dict[int, list[float | None]], dict[int, str]]:
    """Build the CrossCheck replay panel.

    Returns:
        (subject specs, group specs, per-subject trajectories with ``None``
        where no report exists, subject-to-group map). Subjects need at least
        ``min_ema`` reports; groups are baseline-severity terciles.
    """
    ds = CrossCheckDataset(local_root=local_root)
    df = ds.load_daily()
    df = df[df["ema_score"].notna()]

    counts = df.groupby("eureka_id")["ema_score"].size()
    eligible = sorted(counts[counts >= min_ema].index)
    if max_subjects:
        eligible = eligible[:max_subjects]

    sub = df[df["eureka_id"].isin(eligible)]

    # Alignment is a real modelling choice, not a detail:
    #
    #   "calendar" indexes every subject by the true shared calendar. It
    #   preserves who was actually enrolled at the same time -- but
    #   CrossCheck enrolments are staggered across a 904-day span, so the
    #   median day has exactly one subject with a recorded EMA and the
    #   allocation decision is degenerate (one candidate, budget >= 1).
    #
    #   "relative" (default) indexes each subject by days since their own
    #   first EMA, simulating a clinic panel where everyone is enrolled
    #   together. This is what makes the allocation problem non-trivial,
    #   and it is a simulation choice that must be stated: it does not
    #   claim these people were ever monitored concurrently. No EMA value
    #   is moved, duplicated or invented -- only the round index changes.
    if align not in ("relative", "calendar"):
        raise ValueError(f"unknown align {align!r}")
    if align == "calendar":
        lo = int(sub["day_ordinal"].min())
        n_rounds = int(sub["day_ordinal"].max()) - lo + 1
        offset_of = {u: lo for u in eligible}
    else:
        offset_of = {u: int(sub[sub["eureka_id"] == u]["day_ordinal"].min()) for u in eligible}
        n_rounds = (
            int(
                max(
                    sub[sub["eureka_id"] == u]["day_ordinal"].max() - offset_of[u] for u in eligible
                )
            )
            + 1
        )

    baselines: dict[str, float] = {}
    trajectories: dict[int, list[float | None]] = {}
    for i, uid in enumerate(eligible):
        g = sub[sub["eureka_id"] == uid].sort_values("day_ordinal")
        traj: list[float | None] = [None] * n_rounds
        for _, row in g.iterrows():
            traj[int(row["day_ordinal"]) - offset_of[uid]] = float(row["ema_score"])
        trajectories[i] = traj
        # Baseline = earliest quarter of this subject's own EMAs.
        k = max(1, int(round(len(g) * BASELINE_FRACTION)))
        baselines[uid] = float(g["ema_score"].iloc[:k].mean())

    # Horizon truncation. Study lengths vary, so late rounds are thin:
    # at 731 rounds the median day has 3 candidates and only 1 subject is
    # still enrolled, which is attrition rather than an allocation
    # problem. Measured profile (relative alignment, min_ema=20):
    #   270d -> 49/64 still enrolled, median 13 candidates/day, 4,335 EMAs
    #   365d -> 33/64,  547d -> 1/64
    # 270 keeps a real panel; set horizon=None to use the full span and
    # accept the attrition tail.
    if horizon is not None and horizon < n_rounds:
        n_rounds = horizon
        trajectories = {sid: t[:n_rounds] for sid, t in trajectories.items()}
        trajectories = {sid: t for sid, t in trajectories.items() if any(v is not None for v in t)}
        keep = set(trajectories)
        eligible = [u for i, u in enumerate(eligible) if i in keep]
        remap = {old: new for new, old in enumerate(sorted(keep))}
        trajectories = {remap[o]: trajectories[o] for o in sorted(keep)}

    vals = np.array([baselines[u] for u in eligible])
    t1, t2 = np.percentile(vals, [33.3, 66.7])

    def tercile(v):
        """Return the baseline-severity group for a subject's baseline value."""
        return "low_baseline" if v <= t1 else ("mid_baseline" if v <= t2 else "high_baseline")

    group_of = {i: tercile(baselines[u]) for i, u in enumerate(eligible)}
    specs = [
        SubjectSpec(
            subject_id=i, group=group_of[i], baseline_offset=float(baselines[u] - vals.mean())
        )
        for i, u in enumerate(eligible)
    ]
    # Group specs carry only what the metrics need (names); no generative
    # parameters, because nothing is generated here.
    group_specs = {g: {"name": g} for g in ("low_baseline", "mid_baseline", "high_baseline")}
    return specs, group_specs, trajectories, group_of
