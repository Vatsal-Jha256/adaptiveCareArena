"""Selective prediction for the item-selection environment: knowing when
the evidence gathered so far is not enough to answer.

This is the third step of the local loop (acquire -> predict or abstain
-> escalate). Without it the environment can only ever answer "what is
the best guess", never "we should not be guessing", which is the part
that matters in a psychiatric monitoring context where a confidently
wrong estimate is worse than a declared unknown.

Why locally adaptive conformal rather than plain split conformal. Plain
split conformal gives every prediction the same interval width -- one
global quantile of the calibration residuals. That is a valid coverage
guarantee and completely useless for selective prediction, because if
every interval is the same width there is no basis on which to abstain
on one case rather than another. So residuals are normalised by a
learned difficulty estimate sigma_hat(x) before taking the quantile:

    score_i = |y_i - yhat_i| / (sigma_hat(x_i) + eps)
    qhat    = ceil((n+1)(1-alpha))/n  empirical quantile of the scores
    interval_i = yhat_i +- qhat * (sigma_hat(x_i) + eps)

Widths now vary with predicted difficulty, so "abstain on the widest
intervals" is a meaningful rule, and marginal coverage is preserved
under exchangeability. sigma_hat must not be fit on the same residuals it then normalises.
Doing that overfits the difficulty model to the calibration residuals,
which shrinks the conformal scores, which shrinks qhat, which produces
intervals that are too narrow at deployment. Measured, not theorised: an
earlier version of this class did exactly that and delivered 76%
empirical coverage against a nominal 90%. The calibration set is
therefore split in two -- sigma_hat is fit on the first half, and the
conformal scores are computed on the held-out second half, which
restores exchangeability between the calibration scores and a test
score. Neither half ever sees test data.

Everything here is fit on train/calibration only. Test labels are used
solely to evaluate, never to choose a threshold."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

EPS = 1e-6


def _usable_columns(X: np.ndarray) -> np.ndarray:
    """Column mask keeping only features with >= 2 distinct observed
    values. A policy that asks nothing leaves every item column entirely
    NaN, and sklearn's histogram binner raises on a column it cannot
    split ("window shape cannot be larger than input array shape") rather
    than ignoring it. Constant and all-NaN columns carry no information
    about difficulty anyway, so dropping them changes nothing except
    whether the fit succeeds."""
    keep = np.zeros(X.shape[1], dtype=bool)
    for j in range(X.shape[1]):
        col = X[:, j]
        finite = col[np.isfinite(col)]
        keep[j] = finite.size > 0 and np.unique(finite).size >= 2
    return keep


class LocallyAdaptiveConformal:
    """Split-conformal intervals whose width tracks predicted difficulty."""

    def __init__(self, alpha: float = 0.1, random_state: int = 0):
        self.alpha = alpha
        self.random_state = random_state
        self._sigma_model = None
        self._keep: np.ndarray | None = None
        self.qhat: float | None = None

    def calibrate(
        self, model, cal_events: list, cal_asked: list[set[str]], y_cal: np.ndarray
    ) -> None:
        """`cal_asked` must be the same acquisition the policy would make
        at deployment -- calibrating on full questionnaires and then
        deploying on short ones would understate the width of exactly the
        cases the policy actually produces."""
        y_cal = np.asarray(y_cal)
        pred = model.predict(cal_events, cal_asked)
        resid = np.abs(y_cal - pred)
        X = model._to_matrix(cal_events, asked=cal_asked)

        # Two disjoint halves: fit difficulty on A, score on B. See the
        # module docstring -- fitting and scoring on the same rows caused
        # measured 76% coverage at a nominal 90%.
        rng = np.random.default_rng(self.random_state)
        perm = rng.permutation(len(cal_events))
        half = len(perm) // 2
        fit_idx, score_idx = perm[:half], perm[half:]
        if len(fit_idx) == 0 or len(score_idx) == 0:
            raise ValueError("calibration set too small to split for locally adaptive conformal")

        self._keep = _usable_columns(X[fit_idx])
        if not self._keep.any():
            # Nothing varies -- difficulty is unpredictable, so fall back
            # to a constant-width (plain split conformal) interval rather
            # than failing. Widths will not vary, and
            # abstention has nothing to rank on in that case.
            self._sigma_model = None
            scores = resid[score_idx]
            n = len(scores)
            level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
            self.qhat = float(np.quantile(scores, level))
            return

        self._sigma_model = HistGradientBoostingRegressor(
            max_iter=120, max_depth=4, random_state=self.random_state
        )
        # Predict LOG residual, then exponentiate: residuals are
        # non-negative and heavy-tailed, and a plain regressor on the raw
        # scale happily predicts negative "difficulty", which would flip
        # the sign of the normaliser.
        self._sigma_model.fit(X[fit_idx][:, self._keep], np.log(resid[fit_idx] + EPS))

        sigma = np.exp(self._sigma_model.predict(X[score_idx][:, self._keep]))
        scores = resid[score_idx] / (sigma + EPS)
        n = len(scores)
        # Finite-sample corrected quantile level, not the naive (1-alpha).
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.qhat = float(np.quantile(scores, level))

    def interval_halfwidth(self, model, events: list, asked: list[set[str]]) -> np.ndarray:
        """Return the half-width of the prediction interval for each event.

        Raises ``RuntimeError`` if ``calibrate`` has not been called. Falls back
        to a constant width when no scale model was fitted.
        """
        if self.qhat is None:
            raise RuntimeError("calibrate() must be called before interval_halfwidth()")
        if self._sigma_model is None:
            # Constant-width fallback (see calibrate()).
            return np.full(len(events), self.qhat, dtype=float)
        X = model._to_matrix(events, asked=asked)
        sigma = np.exp(self._sigma_model.predict(X[:, self._keep]))
        return self.qhat * (sigma + EPS)


def risk_coverage_curve(
    y_true: np.ndarray,
    pred: np.ndarray,
    uncertainty: np.ndarray,
    coverages: tuple[float, ...] = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5),
) -> list[dict]:
    """Abstain on the most uncertain cases first; report MAE on what
    remains. A useful uncertainty signal makes MAE fall as coverage
    falls. If it does not, the model does not know what it does not
    know, and that is worth reporting rather than hiding."""
    y_true, pred, unc = map(np.asarray, (y_true, pred, uncertainty))
    order = np.argsort(unc)  # most certain first
    out = []
    for c in coverages:
        k = max(1, int(round(c * len(order))))
        keep = order[:k]
        out.append(
            {
                "coverage": float(k / len(order)),
                "mae": float(np.mean(np.abs(y_true[keep] - pred[keep]))),
                "n_retained": int(k),
            }
        )
    return out


def aurc(curve: list[dict]) -> float:
    """Area under the risk-coverage curve (trapezoid over coverage).
    Lower is better. Summarises the whole curve so two uncertainty
    signals can be compared with one number instead of eyeballing."""
    pts = sorted(curve, key=lambda d: d["coverage"])
    cov = np.array([p["coverage"] for p in pts])
    risk = np.array([p["mae"] for p in pts])
    return float(np.trapezoid(risk, cov) / (cov[-1] - cov[0])) if len(cov) > 1 else float(risk[0])
