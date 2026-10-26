"""Computes stage2_env's world parameters directly from real CrossCheck
statistics (not invented) and writes them to a JSON file that
configs/dev/stage2_smoke.yaml's `world.groups` block is a direct
transcription of -- run this to see exactly where those numbers come
from, or to recompute them if the adapter changes.

Method:
- Population AR(1) fit on consecutive genuine EMA responses (gap <= 10
  days): ema_score[t] ~ mu + b*(ema_score[t-1] - mu). `b` is the
  persistence/reversion parameter; residual std is `noise_std`.
- unobserved_extra_noise_std: |change| grows with gap length
  (corr=+0.03, p=0.019, weak but real) -- the per-extra-day slope of that
  relationship, used as a variance increment (see world.py's docstring
  for why this is modeled as variance, not a directional push).
- Three groups = real baseline-severity terciles (per-subject mean
  ema_score, split at the 33rd/67th percentiles) -- `mu`/`subject_std`
  per group are the real tercile mean/within-tercile spread, not assumed
  equal or invented.

python -m scripts.calibrate_from_crosscheck
"""

from __future__ import annotations

import json

import numpy as np
from scipy import stats as st

from caredata.crosscheck import CrossCheckDataset


def calibrate() -> dict:
    ds = CrossCheckDataset()
    df = ds.load_daily()

    xs, ys, gaps = [], [], []
    for _sid, g in df.groupby("eureka_id"):
        g = g.sort_values("day_ordinal")
        ema = g[g["ema_score"].notna()][["day_ordinal", "ema_score"]].to_numpy()
        for i in range(1, len(ema)):
            gap = ema[i, 0] - ema[i - 1, 0]
            if gap <= 10:
                xs.append(ema[i - 1, 1])
                ys.append(ema[i, 1])
                gaps.append(gap)
    xs, ys, gaps = np.array(xs), np.array(ys), np.array(gaps)

    mu = float(np.mean(np.concatenate([xs, ys])))
    b, intercept = np.polyfit(xs - mu, ys - mu, 1)
    resid = (ys - mu) - (b * (xs - mu) + intercept)
    noise_std = float(np.std(resid))
    reversion = float(1 - b)

    abs_change = np.abs(ys - xs)
    corr, pval = st.pearsonr(gaps, abs_change)
    unobserved_extra_noise_std = float(np.polyfit(gaps, abs_change, 1)[0])  # slope only

    subj_means = df[df["ema_score"].notna()].groupby("eureka_id")["ema_score"].mean()
    cuts = np.percentile(subj_means, [33.3, 66.7])
    low = subj_means[subj_means <= cuts[0]]
    mid = subj_means[(subj_means > cuts[0]) & (subj_means <= cuts[1])]
    high = subj_means[subj_means > cuts[1]]

    groups = []
    for name, series in (("low_baseline", low), ("mid_baseline", mid), ("high_baseline", high)):
        groups.append(
            {
                "name": name,
                "weight": float(len(series)) / len(subj_means),
                "mu": float(series.mean()),
                "reversion": reversion,
                "noise_std": noise_std,
                "unobserved_extra_noise_std": unobserved_extra_noise_std,
                "subject_std": float(series.std()),
            }
        )

    calibration = {
        "source": "CrossCheck_Daily_Data.csv, real fit, 2026-09-02",
        "n_consecutive_pairs_used": int(len(xs)),
        "n_subjects_with_ema": int(len(subj_means)),
        "population_mu": mu,
        "ar1_persistence_b": float(b),
        "reversion": reversion,
        "noise_std": noise_std,
        "gap_vs_abschange_corr": float(corr),
        "gap_vs_abschange_pvalue": float(pval),
        "unobserved_extra_noise_std_per_day": unobserved_extra_noise_std,
        "groups": groups,
    }
    return calibration


def main() -> None:
    calibration = calibrate()
    out_path = "data/stage2_calibration.json"
    with open(out_path, "w") as f:
        json.dump(calibration, f, indent=2)
    print(f"wrote {out_path}")
    for g in calibration["groups"]:
        print(
            f"  {g['name']:>14}: weight={g['weight']:.3f} mu={g['mu']:.2f} "
            f"subject_std={g['subject_std']:.2f}"
        )
    print(
        f"  shared: reversion={calibration['reversion']:.4f} noise_std={calibration['noise_std']:.3f} "
        f"unobserved_extra_noise_std/day={calibration['unobserved_extra_noise_std_per_day']:.4f}"
    )


if __name__ == "__main__":
    main()
