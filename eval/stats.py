"""Statistical machinery for comparing many policies across many cells.

A single sweep produces many cells and policies; at alpha=0.05 several
false positives are expected from noise alone, so comparisons need
correction.

What is here, and why each piece:

  bootstrap_paired_ci   A CI for the paired difference by resampling
                        seeds, instead of mean +- sd. The sd route
                        assumes normality and, worse, uses a sample sd
                        that is downward-biased at small n: an effect
                        read as 6.43 sd at 5 seeds was 1.90 sd at 20, with
                        the mean barely moving.
  holm_bonferroni       Family-wise error control. Use when any false
                        positive is costly (a headline claim).
  benjamini_hochberg    False-discovery-rate control. Use when screening
                        many cells and a known fraction of false leads
                        is acceptable.
  summarise_family      Runs a whole family of comparisons and reports
                        which survive correction, with the count of
                        naive vs corrected "significant" results.

Everything returns an explicit verdict that includes "inconclusive". A
summary with no such category resolves ambiguity into whichever outcome
the author already believed."""

from __future__ import annotations

import math
import statistics as st

import numpy as np


def bootstrap_paired_ci(
    per_seed_a: list[float],
    per_seed_b: list[float],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Percentile bootstrap CI for the mean paired difference.

    Resamples seeds (the independent replicate here), not observations
    within a run -- runs share a subject split, so rows inside one are
    not independent. Reports the bootstrap sd alongside the sample sd:
    when they disagree markedly the normal approximation is not safe and
    the CI should be preferred."""
    if len(per_seed_a) != len(per_seed_b):
        raise ValueError("paired comparison needs one value per arm per seed")
    if len(per_seed_a) < 3:
        raise ValueError("bootstrap needs at least 3 seeds to say anything")

    diffs = np.asarray(per_seed_a, dtype=float) - np.asarray(per_seed_b, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boots = np.array([rng.choice(diffs, size=n, replace=True).mean() for _ in range(n_boot)])

    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    # Two-sided bootstrap p: how often does a recentred draw exceed the
    # observed effect? Never reported as 0 -- with n_boot resamples the
    # smallest resolvable value is 1/n_boot, and printing 0 would claim
    # more precision than the procedure has.
    centred = boots - boots.mean()
    p = (np.abs(centred) >= abs(diffs.mean())).mean()
    p = max(p, 1.0 / n_boot)

    return {
        "mean_diff": float(diffs.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_value": float(p),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "sample_sd": float(st.stdev(diffs)) if n > 1 else 0.0,
        "bootstrap_sd": float(boots.std(ddof=1)),
        "n_seeds": n,
    }


def holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Family-wise error rate control. Uniformly more powerful than plain
    Bonferroni and just as valid, so there is no reason to use the
    latter."""
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    out: dict[str, dict] = {}
    rejected_so_far = True
    for i, (name, p) in enumerate(ordered):
        threshold = alpha / (m - i)
        # Holm stops at the first failure; everything after stays accepted.
        reject = rejected_so_far and p <= threshold
        rejected_so_far = reject
        out[name] = {"p": p, "threshold": threshold, "significant": reject, "rank": i + 1}
    return out


def benjamini_hochberg(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """False-discovery-rate control -- expected proportion of false
    positives among the rejected hypotheses. Appropriate for screening a
    grid of cells; not appropriate for a single headline claim, where a
    false positive is not tolerable at any rate."""
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    # Largest k with p_(k) <= k/m * alpha; everything up to it is rejected.
    max_k = 0
    for i, (_n, p) in enumerate(ordered, start=1):
        if p <= i / m * alpha:
            max_k = i
    return {
        name: {"p": p, "threshold": (i / m) * alpha, "significant": i <= max_k, "rank": i}
        for i, (name, p) in enumerate(ordered, start=1)
    }


def summarise_family(
    comparisons: dict[str, tuple[list[float], list[float]]],
    higher_is_better: bool,
    alpha: float = 0.05,
    method: str = "holm",
    n_boot: int = 10000,
) -> dict:
    """Run a family of paired comparisons with multiple-testing control.

    `comparisons[name] = (per_seed_a, per_seed_b)`. Reports both the
    naive and the corrected count of significant results, because the
    difference between them is the number of findings that were an
    artifact of testing many things -- and that difference belongs in
    the writeup, not just in the analyst's head."""
    if method not in ("holm", "bh"):
        raise ValueError("method must be 'holm' (family-wise) or 'bh' (FDR)")

    detail = {
        name: bootstrap_paired_ci(a, b, n_boot=n_boot, alpha=alpha)
        for name, (a, b) in comparisons.items()
    }
    corrected = (holm_bonferroni if method == "holm" else benjamini_hochberg)(
        {k: v["p_value"] for k, v in detail.items()}, alpha=alpha
    )

    for name, c in corrected.items():
        d = detail[name]
        d.update(corrected_significant=c["significant"], corrected_threshold=c["threshold"])
        if not c["significant"]:
            d["verdict"] = "inconclusive"
        else:
            a_better = (d["mean_diff"] > 0) if higher_is_better else (d["mean_diff"] < 0)
            d["verdict"] = "A better" if a_better else "B better"

    naive = sum(v["excludes_zero"] for v in detail.values())
    surv = sum(v["corrected_significant"] for v in detail.values())
    return {
        "method": method,
        "alpha": alpha,
        "n_comparisons": len(detail),
        "n_significant_naive": naive,
        "n_significant_corrected": surv,
        "n_lost_to_correction": naive - surv,
        "comparisons": detail,
    }


def tost_equivalence(
    per_seed_a: list[float],
    per_seed_b: list[float],
    margin: float,
    alpha: float = 0.05,
) -> dict:
    """Two One-Sided Tests: is the paired difference A-B small enough to
    call the two arms equivalent within +/- `margin`?

    This exists because a non-significant difference is not a null
    result. A wide CI that straddles zero says "we did not detect a
    difference", which is compatible with both "there is none" and "we
    lacked power". Reading the first from the second is the most
    common way a negative result gets overclaimed (5-seed sds moved an
    effect from 6.4 to 1.9 sigma with the mean barely changing).

    TOST inverts the burden: the null is "the difference is at least as
    large as `margin`", and rejecting it in both directions licenses the
    positive claim of equivalence. Concretely, equivalence is declared
    iff the (1-2*alpha) CI for the difference lies entirely inside
    (-margin, +margin).

    `margin` must be pre-specified from what would matter substantively,
    never read off the observed effect -- choosing it after seeing the
    data makes the test circular. The caller passes it in for exactly
    that reason; there is no default.

    Returns p_tost = max(p_lower, p_upper): equivalence at `alpha` iff
    p_tost < alpha. Outcomes are deliberately three-valued, because
    "neither different nor equivalent" (underpowered) is a real and
    common answer that a two-valued verdict would hide.
    """
    import statistics as st

    if margin <= 0:
        raise ValueError("margin must be positive and pre-specified")
    diffs = [a - b for a, b in zip(per_seed_a, per_seed_b, strict=False)]
    n = len(diffs)
    if n < 3:
        raise ValueError(f"TOST needs >=3 paired observations, got {n}")

    mean = st.mean(diffs)
    sd = st.stdev(diffs)
    se = sd / math.sqrt(n)

    # Zero-variance difference: the paired gap is the same on every seed,
    # so it is known exactly and no sampling argument is needed. This
    # branch is guarded explicitly because dividing by a ~1e-16 float
    # residue has manufactured a ~1e14 effect size. It is not an error case: a constant
    # difference is the most precise result possible, and each of its
    # three sub-cases has a definite answer rather than a degenerate one.
    if se <= 1e-12 * max(1.0, abs(mean)):
        identical = abs(mean) <= 1e-12
        within = abs(mean) <= margin
        if identical:
            outcome = (
                "IDENTICAL (zero variance, zero gap -- the arms are the same series, "
                "which is stronger than equivalence and usually means the comparison "
                "never actually ran)"
            )
        elif within:
            outcome = f"EQUIVALENT exactly (constant gap {mean:+.4g}, inside +/-{margin})"
        else:
            outcome = f"DIFFERENT exactly (constant gap {mean:+.4g}, outside +/-{margin})"
        return {
            "n": n,
            "mean_diff": mean,
            "sd": sd,
            "se": se,
            "margin": margin,
            "ci_lo": mean,
            "ci_hi": mean,
            "p_lower": 0.0 if within else 1.0,
            "p_upper": 0.0 if within else 1.0,
            "p_tost": 0.0 if within else 1.0,
            "p_difference": 1.0 if identical else 0.0,
            "equivalent": within,
            "different": not within,
            "outcome": outcome,
        }

    df = n - 1
    t_lower = (mean + margin) / se  # H0: diff <= -margin
    t_upper = (mean - margin) / se  # H0: diff >= +margin
    p_lower = 1.0 - _t_cdf(t_lower, df)
    p_upper = _t_cdf(t_upper, df)
    p_tost = max(p_lower, p_upper)

    # (1 - 2*alpha) CI, the interval TOST is equivalent to.
    t_crit = _t_ppf(1.0 - alpha, df)
    ci_lo, ci_hi = mean - t_crit * se, mean + t_crit * se
    equivalent = p_tost < alpha

    # The difference test, so the three-way outcome can be formed.
    p_diff = 2.0 * (1.0 - _t_cdf(abs(mean / se), df))
    different = p_diff < alpha

    if different and not equivalent:
        outcome = "DIFFERENT (and not equivalent within margin)"
    elif equivalent and not different:
        outcome = "EQUIVALENT within margin"
    elif equivalent and different:
        # Possible and not a contradiction: a small but reliably nonzero
        # difference. Statistically distinguishable, substantively
        # negligible -- which is usually the right reading.
        outcome = "trivially different: nonzero but smaller than the margin that would matter"
    else:
        outcome = "INCONCLUSIVE (underpowered -- neither different nor equivalent)"

    return {
        "n": n,
        "mean_diff": mean,
        "sd": sd,
        "se": se,
        "margin": margin,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p_tost": p_tost,
        "p_difference": p_diff,
        "equivalent": equivalent,
        "different": different,
        "outcome": outcome,
    }


def _t_cdf(t: float, df: int) -> float:
    """Student-t CDF via the regularised incomplete beta function. Hand-
    rolled to keep the package dependency-light (numpy/sklearn only, no
    scipy) -- checked against scipy.stats.t.cdf to 1e-10 in tests."""

    x = df / (df + t * t)
    p = 0.5 * _betainc(0.5 * df, 0.5, x)
    return p if t <= 0 else 1.0 - p


def _t_ppf(p: float, df: int) -> float:
    """Inverse Student-t CDF by bisection. Ample precision here: the
    result feeds a reported CI, not an inner loop."""
    lo, hi = -1e3, 1e3
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a,b), Lentz continued fraction
    (Numerical Recipes 6.4), with the standard symmetry reflection for
    x above the convergence-friendly region."""

    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _betainc(b, a, 1.0 - x)

    tiny = 1e-30
    f, c, d = 1.0, 1.0, 0.0
    for i in range(200):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2.0 * m - 1.0) * (a + 2.0 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2.0 * m) * (a + 2.0 * m + 1.0))
        d = 1.0 + num * d
        if abs(d) < tiny:
            d = tiny
        d = 1.0 / d
        c = 1.0 + num / c
        if abs(c) < tiny:
            c = tiny
        f *= c * d
        if abs(1.0 - c * d) < 1e-12:
            break
    return front * (f - 1.0) / a
