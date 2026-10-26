"""Builds ReplaySubjectWorld inputs from real CES (College Experience
Study), mirroring `replay_data.py`'s CrossCheck adapter so the same
policies/estimator/metrics code runs against a second, independent real
cohort -- this is what turns "one heuristic on one schizophrenia cohort"
into a benchmark claim rather than a single case study.

Two structural differences from CrossCheck, both real and both handled
here rather than left implicit:

1. **Sign convention.** CrossCheck's `ema_score` is pos-minus-neg (higher
   = better mood); CES's PHQ-4 is a symptom-count sum (higher = worse).
   `estimators.py` and `deterioration_threshold` both assume "lower state
   = worse", so CES's state is stored as the negative of its recomputed
   PHQ-4 score. This is a relabelling, not a rescaling -- no value is
   altered beyond sign.
2. **Deterioration threshold.** CrossCheck's -5.0 was chosen against its
   own composite scale. CES's clinically meaningful threshold is PHQ-4's
   own published severe-distress cutoff, total score >= 9 (Kroenke et
   al. 2009, Psychosomatics) -- so state <= -9.0.

Subjects are assigned to severity terciles by their own baseline period
only, exactly as in `replay_data.py`, for the same leakage reason."""

from __future__ import annotations

import numpy as np

from caredata.ces import CESDataset
from stage2_env.core import SubjectSpec

BASELINE_FRACTION = 0.25  # of each subject's own PHQ-4 history, earliest first


def build_replay_inputs(
    local_root: str = "data/raw/ces",
    min_ema: int = 20,
    max_subjects: int | None = None,
    align: str = "relative",
    horizon: int | None = 270,
) -> tuple[list[SubjectSpec], dict[str, object], dict[int, list[float | None]], dict[int, str]]:
    """Build the CES replay panel; same return value as the CrossCheck builder.

    The PHQ-4 total is negated so lower is worse, matching CrossCheck.
    """
    ds = CESDataset(local_root=local_root)
    df = ds.load_ema()

    counts = df.groupby("uid").size()
    eligible = sorted(counts[counts >= min_ema].index)
    if max_subjects:
        eligible = eligible[:max_subjects]

    sub = df[df["uid"].isin(eligible)]

    if align not in ("relative", "calendar"):
        raise ValueError(f"unknown align {align!r}")
    if align == "calendar":
        lo = int(sub["day_ordinal"].min())
        n_rounds = int(sub["day_ordinal"].max()) - lo + 1
        offset_of = {u: lo for u in eligible}
    else:
        offset_of = {u: int(sub[sub["uid"] == u]["day_ordinal"].min()) for u in eligible}
        n_rounds = (
            int(max(sub[sub["uid"] == u]["day_ordinal"].max() - offset_of[u] for u in eligible)) + 1
        )

    baselines: dict[str, float] = {}
    trajectories: dict[int, list[float | None]] = {}
    for i, uid in enumerate(eligible):
        g = sub[sub["uid"] == uid].sort_values("day_ordinal")
        traj: list[float | None] = [None] * n_rounds
        for _, row in g.iterrows():
            idx = int(row["day_ordinal"]) - offset_of[uid]
            if 0 <= idx < n_rounds:
                traj[idx] = -float(row["phq4_recomputed"])  # sign flip, see module docstring
        trajectories[i] = traj
        k = max(1, int(round(len(g) * BASELINE_FRACTION)))
        baselines[uid] = float(-g["phq4_recomputed"].iloc[:k].mean())

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
    group_specs = {g: {"name": g} for g in ("low_baseline", "mid_baseline", "high_baseline")}
    return specs, group_specs, trajectories, group_of
