"""Per-subject exponential moving average of observed state -- the naive
baseline, for continuous (not Bernoulli) state.
Predicts need = -observed_state directly (higher need = worse predicted
mood), so policies never have to know the sign convention. Deliberately
naive: a never-rechecked subject's prediction never moves from its last
observed value (or the prior, if never checked at all), even though the
world's own dynamics keep moving their true state -- this staleness is
the mechanism under study, not a bug to fix here.
"""

from __future__ import annotations

from stage2_env.core import ESTIMATORS, Observation, SubjectFeature


@ESTIMATORS.register("naive_ema")
class NaiveEMAEstimator:
    """Per-subject exponential moving average of observed need.

    Need is the negated observed state, so higher need means worse predicted
    mood. A subject never observed gets ``prior_need``.
    """

    def __init__(self, alpha: float = 0.3, prior_need: float = 0.0):
        self.alpha = alpha
        self.prior_need = prior_need
        self._need_ema: dict[int, float] = {}

    def update(self, observations: list[Observation]) -> None:
        """Fold each observation's need into that subject's moving average."""
        for obs in observations:
            need = -obs.observed_state
            prev = self._need_ema.get(obs.subject_id, self.prior_need)
            self._need_ema[obs.subject_id] = self.alpha * need + (1 - self.alpha) * prev

    def predict(self, features: dict[int, SubjectFeature]) -> dict[int, float]:
        """Return each subject's current need estimate."""
        return {sid: self._need_ema.get(sid, self.prior_need) for sid in features}


@ESTIMATORS.register("kalman_need")
class KalmanNeedEstimator:
    """Per-subject local-level Kalman filter over observed need: state z_t evolves as a random walk (process
    noise `q`), observations are noisy (`r`). Unlike NaiveEMAEstimator's
    fixed-alpha blend, the gain adapts to elapsed time since the last
    observation -- every tracked subject's variance grows by `q` each
    round regardless of whether they were observed, so a subject who has
    gone unreviewed for many rounds gets a wider posterior and therefore
    a bigger correction the next time real data arrives, instead of the
    same fixed 0.3 blend every time. `q` and `r` are population-level
    hyperparameters shared across all subjects (partial pooling of the
    noise model, not of the state itself, which stays fully per-subject)
    -- the amount of hierarchy 64-210 subjects can support without
    overfitting the noise model, selected once via one-step-ahead
    prediction error, never via the deployment-time metrics this
    estimator is later judged on (see scripts/kalman_tune_noise.py)."""

    def __init__(
        self, q: float = 0.05, r: float = 1.0, prior_need: float = 0.0, prior_var: float = 1.0
    ):
        self.q = q
        self.r = r
        self.prior_need = prior_need
        self.prior_var = prior_var
        self._mean: dict[int, float] = {}
        self._var: dict[int, float] = {}

    def update(self, observations: list[Observation]) -> None:
        """Advance every tracked subject one round, then correct the observed ones.

        Variance grows by ``q`` for every tracked subject each round, and shrinks
        for those observed this round.
        """
        for sid in self._var:
            self._var[sid] += self.q
        for obs in observations:
            sid = obs.subject_id
            need = -obs.observed_state
            mean_pred = self._mean.get(sid, self.prior_need)
            var_pred = self._var.get(sid, self.prior_var + self.q)
            gain = var_pred / (var_pred + self.r)
            self._mean[sid] = mean_pred + gain * (need - mean_pred)
            self._var[sid] = (1.0 - gain) * var_pred

    def predict(self, features: dict[int, SubjectFeature]) -> dict[int, float]:
        """Return each subject's posterior mean need, or ``prior_need`` if never observed."""
        return {sid: self._mean.get(sid, self.prior_need) for sid in features}
