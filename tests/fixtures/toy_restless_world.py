"""The restless-subject world. This module holds ground truth
(SubjectState.true_state). Ground truth is read outside world/ only by
metrics/ (via the true_state_now dict pull_and_advance returns), never fed
to a policy or estimator -- SubjectFeature (what policies/estimators
actually receive) has no true_state field at all.

A restless-arm pattern adapted to CrossCheck's real composite EMA
score (`ema_pos_score - ema_neg_score`), a continuous mood/symptom
measure, not a probability. Two adaptations from the source pattern, both
grounded in real calibration (scripts/calibrate_from_crosscheck.py), not
guessed:

1. `unobserved_drift` here is a variance increment, not a directional
   push. A Bernoulli-risk restless-arm world would push an unpulled arm's risk
   upward (a specific harmful direction) every round it isn't pulled --
   real CrossCheck data doesn't support a directional claim: the measured
   relationship is `corr(gap_since_last_EMA, |change_in_EMA_score|) =
   +0.03` (weak but real, p=0.019) -- longer gaps see bigger swings, not
   swings in a consistent direction. Modeled here as extra noise variance
   on unchecked rounds, not an additive drift.
2. Checking a subject reveals `true_state` exactly (no separate
   observation-noise term) -- the AR(1) fit used to calibrate `noise_std`
   already pools process variation and self-report imprecision (they
   cannot be separated from observational EMA data alone), so introducing
   a second, invented noise term would be double-counting one estimated
   quantity as if it were two."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stage2_env.core import Observation, SubjectFeature, SubjectSpec, SubjectState


@dataclass(frozen=True)
class GroupSpec:
    """Static per-group dynamics, one group per calibrated baseline
    severity tercile (see scripts/calibrate_from_crosscheck.py) --
    `mu` is that tercile's real mean composite EMA score, not invented."""

    name: str
    weight: float
    mu: float
    reversion: float
    noise_std: float
    unobserved_extra_noise_std: float = 0.0
    subject_std: float = 0.0  # real per-subject baseline spread within the group


def build_subjects(
    groups: list[GroupSpec], n_subjects: int, rng: np.random.Generator
) -> list[SubjectSpec]:
    weights = np.array([g.weight for g in groups], dtype=float)
    weights = weights / weights.sum()
    assignment = rng.choice(len(groups), size=n_subjects, p=weights)
    specs = []
    for i, gi in enumerate(assignment):
        g = groups[gi]
        offset = float(rng.normal(0.0, g.subject_std)) if g.subject_std > 0 else 0.0
        specs.append(SubjectSpec(subject_id=i, group=g.name, baseline_offset=offset))
    return specs


class RestlessSubjectWorld:
    def __init__(
        self,
        subject_specs: list[SubjectSpec],
        group_specs: dict[str, GroupSpec],
        rng: np.random.Generator,
        state_clip: tuple[float, float] = (-25.0, 25.0),
    ):
        self.subject_specs = subject_specs
        self.group_specs = group_specs
        self.rng = rng
        self.groups = sorted(group_specs)
        self.state_clip = state_clip
        self.states: dict[int, SubjectState] = {
            spec.subject_id: SubjectState(
                subject_id=spec.subject_id,
                true_state=float(
                    group_specs[spec.group].mu
                    + spec.baseline_offset
                    + rng.normal(0.0, group_specs[spec.group].noise_std)
                ),
            )
            for spec in subject_specs
        }

    def _onehot(self, group: str) -> tuple[float, ...]:
        return tuple(1.0 if g == group else 0.0 for g in self.groups)

    def current_features(
        self, round_idx: int, n_rounds_total: int, check_counts: dict[int, int]
    ) -> dict[int, SubjectFeature]:
        out = {}
        for spec in self.subject_specs:
            st = self.states[spec.subject_id]
            out[spec.subject_id] = SubjectFeature(
                group_onehot=self._onehot(spec.group),
                rounds_since_check=st.rounds_since_check,
                last_observed_state=(
                    st.last_observed_state if st.last_observed_state is not None else 0.0
                ),
                round_idx_norm=round_idx / max(1, n_rounds_total),
                check_count_so_far=check_counts.get(spec.subject_id, 0),
            )
        return out

    def pull_and_advance(
        self,
        round_idx: int,
        checked_subject_ids: set[int],
        features_this_round: dict[int, SubjectFeature],
    ) -> tuple[list[Observation], dict[int, float]]:
        """Reveal outcomes for checked subjects against current true_state,
        then advance every subject's dynamics by one round (restless: all
        subjects evolve, checked or not). Returns (observations for
        checked subjects only, {subject_id: true_state before this
        round's update} for metrics/ only)."""
        observations: list[Observation] = []
        true_state_now = {sid: st.true_state for sid, st in self.states.items()}

        for spec in self.subject_specs:
            st = self.states[spec.subject_id]
            g = self.group_specs[spec.group]
            checked = spec.subject_id in checked_subject_ids

            if checked:
                observations.append(
                    Observation(
                        round_idx=round_idx,
                        subject_id=spec.subject_id,
                        group=spec.group,
                        feature=features_this_round[spec.subject_id],
                        observed_state=st.true_state,
                        true_state_at_check=st.true_state,
                    )
                )
                st.rounds_since_check = 0
                st.last_observed_round = round_idx
                st.last_observed_state = st.true_state
            else:
                st.rounds_since_check += 1

            target = g.mu + spec.baseline_offset
            extra_noise = g.unobserved_extra_noise_std if not checked else 0.0
            noise = self.rng.normal(0.0, g.noise_std + extra_noise)
            new_state = st.true_state + g.reversion * (target - st.true_state) + noise
            st.true_state = float(np.clip(new_state, *self.state_clip))

        return observations, true_state_now

    def true_state_snapshot(self) -> dict[int, float]:
        return {sid: st.true_state for sid, st in self.states.items()}
