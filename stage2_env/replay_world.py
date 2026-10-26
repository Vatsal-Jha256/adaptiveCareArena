"""A real-data restless world: replays actual CrossCheck trajectories
instead of simulating an Ornstein-Uhlenbeck process calibrated from them.

Why this exists. `world.py`'s `RestlessSubjectWorld` is synthetic -- real
statistics (per-group mean, reversion, noise from
scripts/calibrate_from_crosscheck.py) driving invented trajectories. That
validates the pipeline but cannot say anything about real acquisition
timing, because the dynamics are the ones we assumed. This class swaps
the generative model for recorded data and leaves every policy,
estimator and metric untouched: it implements the same
`current_features` / `pull_and_advance` / `true_state_snapshot`
interface, so the same policy and estimator code runs against either.

**What the decision is.** The EMA was already
administered by the original study on the days it exists. We do not
invent responses for days a participant was not asked -- that answer does
not exist and fabricating it would make every downstream number
meaningless. So the modelled scarcity is *attention*, not data
collection: on each day some subjects have a recorded EMA, a reviewer
with capacity B can look at B of them, and the rest go unread. That is
exactly the shape of the deployed public-health system this project takes
its framing from ("which anomalies should a limited number of experts
inspect?"), which is what the data can support.

Restlessness is therefore observed rather than assumed: a participant's state
really does move between the days you look at them, and you really only
learn where it went when you look. Nothing about that trajectory is
generated here.

**What this world does not model:** reviewing a subject cannot change
their future state, because in the recorded data nobody intervened based
on our policy. Any claim about intervention effects needs interventional
data; this is active surveillance, and metrics must be read that way."""

from __future__ import annotations

import numpy as np

from stage2_env.core import Observation, SubjectFeature, SubjectSpec


class ReplaySubjectWorld:
    """Restless world backed by recorded per-subject trajectories.

    `trajectories[subject_id][round_idx]` is the recorded state on that
    round, or None when no EMA exists that day. Rounds are dense (one per
    calendar day); availability is sparse, which is the real structure.
    """

    def __init__(
        self,
        subject_specs: list[SubjectSpec],
        group_specs: dict,
        trajectories: dict[int, list[float | None]],
        rng: np.random.Generator,
        carry_forward_state: bool = True,
    ):
        self.subject_specs = subject_specs
        self.group_specs = group_specs
        self.trajectories = trajectories
        self.rng = rng
        self.carry_forward_state = carry_forward_state
        self.n_rounds = max(len(v) for v in trajectories.values())
        self._groups = sorted(group_specs)

        self.rounds_since_check: dict[int, int] = {s.subject_id: 0 for s in subject_specs}
        self.last_observed_state: dict[int, float | None] = {
            s.subject_id: None for s in subject_specs
        }
        self.last_observed_round: dict[int, int | None] = {
            s.subject_id: None for s in subject_specs
        }
        # Last recorded value, carried forward so a "true state" exists on
        # days between EMAs. This is an assumption (persistence), and it is
        # the same assumption Stage 1 found nothing beats -- but it is an
        # assumption, so `available_only` metrics should be preferred when
        # a claim depends on it.
        self._carried: dict[int, float | None] = {s.subject_id: None for s in subject_specs}

    def _onehot(self, group: str) -> tuple[float, ...]:
        return tuple(1.0 if group == g else 0.0 for g in self._groups)

    def available(self, round_idx: int) -> set[int]:
        """Subjects with a real recorded EMA on this round. Only these can
        be meaningfully reviewed -- reviewing anyone else would reveal a
        value that was never collected."""
        out = set()
        for spec in self.subject_specs:
            traj = self.trajectories[spec.subject_id]
            if round_idx < len(traj) and traj[round_idx] is not None:
                out.add(spec.subject_id)
        return out

    def true_state(self, subject_id: int, round_idx: int) -> float | None:
        """Return the recorded state at ``round_idx``.

        If none was recorded, return the last recorded value when carry-forward is
        on, otherwise ``None``.
        """
        traj = self.trajectories[subject_id]
        v = traj[round_idx] if round_idx < len(traj) else None
        if v is not None:
            self._carried[subject_id] = float(v)
            return float(v)
        return self._carried[subject_id] if self.carry_forward_state else None

    def current_features(
        self, round_idx: int, n_rounds_total: int, check_counts: dict[int, int]
    ) -> dict[int, SubjectFeature]:
        """Build each subject's ``SubjectFeature`` for this round from what has been observed so far."""
        out = {}
        for spec in self.subject_specs:
            last = self.last_observed_state[spec.subject_id]
            out[spec.subject_id] = SubjectFeature(
                group_onehot=self._onehot(spec.group),
                rounds_since_check=self.rounds_since_check[spec.subject_id],
                last_observed_state=last if last is not None else 0.0,
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
        """Auto-ingest every real EMA that arrives this round, then advance
        the human-attention clock separately.

        CrossCheck's EMA is digital: the self-report reaches the study's
        own database whether or not a clinician later opens that chart.
        Gating the returned `Observation` (and therefore the estimator's
        belief) on `checked_subject_ids` would make "not reviewed" and "not
        observed" the same event, collapsing two different scarce
        resources (a human's attention vs. the model's own data feed)
        into one action. `mean_belief_staleness` would then reflect the
        review policy rather than CrossCheck's real EMA cadence.

        Here, every subject with a genuine recorded EMA this round is
        ingested into `last_observed_state` regardless of
        `checked_subject_ids` -- the belief the automated system holds is
        always as current as the real data allows. `rounds_since_check`
        is the separate, review-specific clock: it resets only when a
        subject is both available and actually chosen, and is what the
        attention-allocation metrics and the staleness-of-attention
        policies (`fair_floor`, `whittle_index`) use. See
        `tests/test_replay_world.py::test_belief_updates_without_review_but_attention_clock_does_not`
        for the regression test that would have caught the original
        conflation."""
        observations: list[Observation] = []
        true_state_now: dict[int, float] = {}
        available = self.available(round_idx)

        for spec in self.subject_specs:
            sid = spec.subject_id
            state = self.true_state(sid, round_idx)
            if state is not None:
                true_state_now[sid] = state

            # A subject with no EMA today has nothing to reveal to
            # anyone, reviewer or model alike.
            if sid in available and state is not None:
                observations.append(
                    Observation(
                        round_idx=round_idx,
                        subject_id=sid,
                        group=spec.group,
                        feature=features_this_round[sid],
                        observed_state=state,
                        true_state_at_check=state,
                    )
                )
                self.last_observed_round[sid] = round_idx
                self.last_observed_state[sid] = state

            # Human attention recency is independent of data recency: it
            # resets only on an actual, available review.
            if sid in checked_subject_ids and sid in available:
                self.rounds_since_check[sid] = 0
            else:
                self.rounds_since_check[sid] += 1

        return observations, true_state_now

    def true_state_snapshot(self) -> dict[int, float]:
        """Return the latest known true state per subject, omitting subjects with none yet."""
        out = {}
        for spec in self.subject_specs:
            v = self._carried[spec.subject_id]
            if v is not None:
                out[spec.subject_id] = v
        return out
