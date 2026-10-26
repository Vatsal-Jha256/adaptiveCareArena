"""Evaluation metrics with the participant as the unit of statistical
inference -- Days within a subject are
correlated by construction (and CrossCheck's EMA cadence makes this
especially true), so bootstrap CIs resample subjects, never rows."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error between two equal-length arrays."""
    return float(np.mean(np.abs(y_true - y_pred)))


def participant_bootstrap_mae_ci(
    examples: list,
    y_pred: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = 1000,
    alpha: float = 0.05,
) -> dict:
    """MAE with a percentile bootstrap CI that resamples participants, not rows.

    Days within a participant are correlated, so resampling rows would give
    intervals that are too narrow.

    Returns:
        dict with ``mae``, ``ci_lo``, ``ci_hi``, ``n_participants``, ``n_examples``.
    """
    subjects = sorted({e.subject_id for e in examples})
    by_subject_errors: dict[str, list[float]] = {}
    for e, p in zip(examples, y_pred, strict=False):
        by_subject_errors.setdefault(e.subject_id, []).append(abs(e.target_y - p))

    point = float(np.mean([v for errs in by_subject_errors.values() for v in errs]))

    boot_means = []
    for _ in range(n_boot):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        pooled = [v for sid in sampled for v in by_subject_errors[sid]]
        if pooled:
            boot_means.append(float(np.mean(pooled)))
    boot_means_arr = np.array(boot_means)

    lo = float(np.quantile(boot_means_arr, alpha / 2)) if len(boot_means_arr) else float("nan")
    hi = float(np.quantile(boot_means_arr, 1 - alpha / 2)) if len(boot_means_arr) else float("nan")

    return {
        "mae": point,
        "ci_lo": lo,
        "ci_hi": hi,
        "n_participants": len(subjects),
        "n_examples": len(examples),
    }


def participant_bootstrap_paired_mae_delta(
    examples: list,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = 1000,
    alpha: float = 0.05,
) -> dict:
    """Paired participant bootstrap of `MAE(a) - MAE(b)`: both models are
    re-scored on the same resampled subjects each draw.

    This is the comparison that actually answers "does A beat B," and it
    is not interchangeable with checking whether two marginal CIs overlap.
    Non-overlapping marginals do imply a real difference, but overlapping
    marginals do not imply no difference -- most of each marginal CI's
    width is between-subject variance shared by both models, and pairing
    cancels it. This matters concretely here: on the event-grouped Stage 1
    split, persistence (2.2336, CI [1.7517, 2.7989]) and GBM (2.4095, CI
    [2.0768, 2.8048]) have heavily overlapping marginal CIs, which says
    almost nothing about whether the 0.176 MAE difference between them is
    real. Both models score the same subjects, so pairing is free.

    An unpaired bootstrap CI per model would not use this pairing; the
    delta below does.

    Negative `delta` means model A has lower error, i.e. A is better.
    `delta_excludes_zero` is the decision-relevant field."""
    subjects = sorted({e.subject_id for e in examples})
    idx_by_subject: dict[str, list[int]] = {}
    for i, e in enumerate(examples):
        idx_by_subject.setdefault(e.subject_id, []).append(i)

    y_true = np.array([e.target_y for e in examples])
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)
    point = mae(y_true, y_pred_a) - mae(y_true, y_pred_b)

    boot = []
    for _ in range(n_boot):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        idx = [i for sid in sampled for i in idx_by_subject[sid]]
        if not idx:
            continue
        d = mae(y_true[idx], y_pred_a[idx]) - mae(y_true[idx], y_pred_b[idx])
        if np.isfinite(d):
            boot.append(float(d))

    boot_arr = np.array(boot)
    lo = float(np.quantile(boot_arr, alpha / 2)) if len(boot_arr) else float("nan")
    hi = float(np.quantile(boot_arr, 1 - alpha / 2)) if len(boot_arr) else float("nan")

    return {
        "delta": point,
        "ci_lo": lo,
        "ci_hi": hi,
        "delta_excludes_zero": bool(len(boot_arr) > 0 and (lo > 0 or hi < 0)),
        "n_participants": len(subjects),
        "n_boot_valid": int(len(boot_arr)),
    }


def equivalence_test(
    values_a: list[float],
    values_b: list[float],
    margin: float,
    alpha: float = 0.05,
) -> dict:
    """Two one-sided tests (TOST) for practical equivalence of A and B.

    This exists because overlapping error bars were being used to argue
    that two policies are equivalent, and that is not what they show.
    Overlapping intervals mean "we failed to detect a difference," which
    is compatible with there being a large one we lacked power to see.
    Absence of evidence is not evidence of absence, and an absence-of-
    effect claim -- "the Whittle index buys nothing over greedy-need" --
    has to be tested as such.

    TOST inverts the burden: it asks whether the difference is
    demonstrably smaller than a margin fixed in advance. Equivalence is
    concluded only if the (1-2*alpha) confidence interval for the
    difference lies entirely inside [-margin, +margin].

    `margin` must be chosen on substantive grounds before looking at the
    result and recorded with the claim, because a margin picked
    afterwards can make any pair of numbers "equivalent". It is a
    required argument here for that reason -- there is no default.

    The observations are paired: element ``i`` in each arm must come from
    the same seed/split. Returns the paired difference, its 90% TOST CI at
    the default alpha, and one of four verdicts: ``equivalent``,
    ``different``, ``inconclusive``, or ``deterministic``. The last is kept
    separate because a zero empirical variance is not sampling evidence.
    """
    import math

    if margin <= 0:
        raise ValueError("margin must be positive and chosen on substantive grounds")
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if len(a) != len(b):
        raise ValueError("paired equivalence testing needs equal-length arms")
    if len(a) < 2:
        raise ValueError("equivalence testing needs at least 2 paired observations")

    paired = a - b
    diff = float(paired.mean())
    se = float(paired.std(ddof=1) / math.sqrt(len(paired)))
    if se <= 1e-12 * max(1.0, abs(diff)):
        return {
            "diff": diff,
            "ci_lo": diff,
            "ci_hi": diff,
            "margin": margin,
            "se": se,
            "verdict": "deterministic",
            "deterministic_relation": (
                "identical"
                if abs(diff) <= 1e-12
                else "inside_margin"
                if abs(diff) < margin
                else "outside_margin"
            ),
            "note": "zero variance in paired differences -- descriptive, not statistically powered",
        }

    # TOST at level alpha is equivalent to checking the (1-2*alpha) CI.
    # This remains a normal approximation; small-n callers should prefer
    # eval.stats.tost_equivalence, which uses a paired Student-t interval.
    z = NormalDist().inv_cdf(1.0 - alpha)
    lo, hi = diff - z * se, diff + z * se

    inside = (lo > -margin) and (hi < margin)
    excludes_zero = (lo > 0) or (hi < 0)
    verdict = "equivalent" if inside else ("different" if excludes_zero else "inconclusive")
    return {
        "diff": diff,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "margin": margin,
        "se": float(se),
        "verdict": verdict,
        "note": "normal approximation, small n -- approximate",
    }
