"""Selects (q, r) for KalmanNeedEstimator (stage2_env/estimators.py) via
one-step-ahead prediction error on the real trajectories, never via the
deployment-time metrics (missed_deterioration_days, attention_lag, ...)
this estimator is later judged on. That separation matters: tuning noise
hyperparameters against the same metrics used to report the result would
let the world model "know" which policy wins before it is evaluated.

For every subject, walk their real EMA trajectory in chronological order.
Before each observed round, ask the filter to predict; only after
predicting, feed it the real value. Score is the MAE between predicted
and actual need over every subject-round with a real observation. This
uses every real self-report exactly as CrossCheckReview/CESReview do
(auto-ingested regardless of any review policy), so it never touches the
review-axis decision loop or its bootstrap outputs at all.

python -m scripts.kalman_tune_noise --dataset crosscheck
python -m scripts.kalman_tune_noise --dataset ces
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import yaml

from stage2_env.estimators import KalmanNeedEstimator
from stage2_env.replay_experiment import load_panel

Q_GRID = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8)
R_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)


def one_step_mae(traj: dict[int, list], q: float, r: float) -> float:
    est = KalmanNeedEstimator(q=q, r=r)
    errs = []
    n_rounds = max(len(v) for v in traj.values())
    for round_idx in range(n_rounds):
        # Predict for every subject who has a real value this round
        # (matches the deployed loop's restrict_to_available=True gate),
        # score against that value, then update -- predict-before-update,
        # identical order to replay_experiment.run_one_policy.
        due = [
            (sid, v[round_idx])
            for sid, v in traj.items()
            if round_idx < len(v) and v[round_idx] is not None
        ]
        if not due:
            continue
        for sid, val in due:
            need = -float(val)
            pred = est._mean.get(sid, est.prior_need)
            errs.append(abs(pred - need))
        from stage2_env.core import Observation, SubjectFeature

        obs = [
            Observation(
                round_idx=round_idx,
                subject_id=sid,
                group="",
                true_state_at_check=val,
                observed_state=val,
                feature=SubjectFeature(
                    group_onehot=(),
                    rounds_since_check=0,
                    last_observed_state=0.0,
                    round_idx_norm=0.0,
                    check_count_so_far=0,
                ),
            )
            for sid, val in due
        ]
        est.update(obs)
    return float(np.mean(errs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["crosscheck", "ces"], required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg_path = f"configs/dev/replay_{args.dataset}.yaml"
    base_cfg = yaml.safe_load(open(cfg_path))
    _, _, traj, _ = load_panel(base_cfg["data"])

    results = []
    for q, r in itertools.product(Q_GRID, R_GRID):
        mae = one_step_mae(traj, q, r)
        results.append({"q": q, "r": r, "one_step_mae": mae})
        print(f"q={q:<6} r={r:<5} one-step MAE={mae:.4f}")

    best = min(results, key=lambda d: d["one_step_mae"])
    print(f"\nbest for {args.dataset}: q={best['q']} r={best['r']} MAE={best['one_step_mae']:.4f}")

    out_path = args.out or f"outputs/kalman_tune_{args.dataset}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(
        json.dumps({"dataset": args.dataset, "grid": results, "best": best}, indent=2)
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
