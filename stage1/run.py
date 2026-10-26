"""Stage 1: can passive sensing forecast the next genuine EMA
response at all, beyond persistence? Naive ladder (population mean,
personal mean, last-known-EMA-carried-forward) vs. real models (ridge,
GBM) on passive-sensing lag features, under future-time and
leave-subject-out splits.

python -m stage1.run --local-root data/raw/crosscheck --out outputs/stage1_report.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from caredata.crosscheck import CrossCheckDataset
from eval.metrics import participant_bootstrap_mae_ci, participant_bootstrap_paired_mae_delta
from models.gbm import GBMModel
from models.naive import LastKnownEMAModel, PersonalMeanModel, PopulationMeanModel
from models.residual import ResidualGBMModel, ResidualRidgeModel
from models.statistical import RidgeRegressionModel
from splits.splits import (
    count_shared_target_events,
    future_time_split,
    kfold_subject_groups,
    leave_subject_out,
)

MODEL_LADDER = {
    "population_mean": PopulationMeanModel,
    "personal_mean": PersonalMeanModel,
    "last_known_ema": LastKnownEMAModel,
    "ridge": RidgeRegressionModel,
    "gbm": GBMModel,
    # Persistence-offset residual models: y_hat = last_known_ema + f(x).
    # These nest persistence (a zero correction reproduces it exactly), so
    # unlike ridge/gbm they never have to rediscover carry-forward before
    # they can model anything else. See models/residual.py.
    "residual_ridge": ResidualRidgeModel,
    "residual_gbm": ResidualGBMModel,
}


def run_stage1(
    local_root: str,
    min_history_days: int,
    train_fraction: float,
    n_folds: int,
    n_boot: int,
    seed: int,
    use_epoch_features: bool = False,
    use_hourly_features: bool = False,
    embargo: float = 0.0,
) -> dict:
    """Compare naive and learned forecasters of the next EMA score.

    Evaluates under both a future-time split and leave-subject-out folds, with
    participant-level bootstrap intervals. Returns a JSON-serializable report.
    """
    ds = CrossCheckDataset(local_root=local_root)
    rng = np.random.default_rng(seed)

    examples = ds.build_task_table(
        min_history_days=min_history_days,
        use_epoch_features=use_epoch_features,
        use_hourly_features=use_hourly_features,
    )
    ds.validate_temporal_contract(examples)

    train, test = future_time_split(examples, train_fraction, embargo=embargo)
    # Recorded in the report, not merely assumed: an independent review
    # measured 47 shared target events (4.26% of test rows) under the old
    # ungrouped split, so the report must carry evidence that the current
    # split has none.
    shared_events = count_shared_target_events(train, test)
    if shared_events:
        raise AssertionError(
            f"future_time_split produced {shared_events} target events shared between "
            "train and test -- this split cannot back a reported number"
        )
    future_time_results = {}
    future_time_preds = {}
    for name, cls in MODEL_LADDER.items():
        model = cls()
        model.fit(train)
        preds = model.predict(test)
        future_time_preds[name] = preds
        future_time_results[name] = participant_bootstrap_mae_ci(test, preds, rng, n_boot=n_boot)
        # For residual models, a tie with persistence is only interpretable
        # alongside how much correction was applied: near-zero means "found
        # no residual signal," which is a far stronger statement than
        # "our regressor lost."
        if hasattr(model, "learned_correction_magnitude"):
            future_time_results[name]["mean_abs_correction"] = model.learned_correction_magnitude(
                test
            )

    # Paired comparison against persistence. The marginal CIs above are
    # unpaired and overlap heavily, which says almost nothing about
    # whether a model really differs from persistence -- every model here
    # scores the same subjects, so pairing is free and strictly more
    # informative. Negative delta = the model beats persistence.
    baseline_name = "last_known_ema"
    paired_vs_persistence = {}
    if baseline_name in future_time_preds:
        for name, preds in future_time_preds.items():
            if name == baseline_name:
                continue
            paired_vs_persistence[name] = participant_bootstrap_paired_mae_delta(
                test, preds, future_time_preds[baseline_name], rng, n_boot=n_boot
            )

    subjects = sorted({e.subject_id for e in examples})
    folds = kfold_subject_groups(subjects, n_folds, rng)
    lso_maes: dict[str, list[float]] = {name: [] for name in MODEL_LADDER}
    for fold in folds:
        train_lso, test_lso = leave_subject_out(examples, fold)
        if not train_lso or not test_lso:
            continue
        for name, cls in MODEL_LADDER.items():
            model = cls()
            model.fit(train_lso)
            preds = model.predict(test_lso)
            r = participant_bootstrap_mae_ci(test_lso, preds, rng, n_boot=200)
            lso_maes[name].append(r["mae"])
    lso_results = {
        name: {"mae": float(np.mean(v)), "n_folds": len(v)} for name, v in lso_maes.items() if v
    }

    return {
        "n_examples": len(examples),
        "n_subjects": len(subjects),
        "future_time": future_time_results,
        "future_time_paired_vs_persistence": paired_vs_persistence,
        "shared_target_events": shared_events,
        "leave_subject_out": lso_results,
    }


def main() -> None:
    """Command-line entry point for the forecasting comparison."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-root", default="data/raw/crosscheck")
    parser.add_argument("--min-history-days", type=int, default=7)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-epoch-features", action="store_true")
    parser.add_argument("--use-hourly-features", action="store_true")
    parser.add_argument(
        "--embargo",
        type=float,
        default=0.0,
        help="extra time margin purged from train before the test period",
    )
    parser.add_argument("--out", default="outputs/stage1_report.json")
    args = parser.parse_args()

    report = run_stage1(
        args.local_root,
        args.min_history_days,
        args.train_fraction,
        args.n_folds,
        args.n_boot,
        args.seed,
        args.use_epoch_features,
        args.use_hourly_features,
        embargo=args.embargo,
    )
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"n_examples={report['n_examples']}  n_subjects={report['n_subjects']}")
    for name, m in report["future_time"].items():
        corr = (
            f"  |correction|={m['mean_abs_correction']:.4f}" if "mean_abs_correction" in m else ""
        )
        print(
            f"  future_time  {name:>16}: MAE={m['mae']:.4f} [{m['ci_lo']:.4f}, {m['ci_hi']:.4f}]  n_p={m['n_participants']}{corr}"
        )
    print(f"  shared target events across the split: {report['shared_target_events']} (must be 0)")
    if report.get("future_time_paired_vs_persistence"):
        print("  -- PAIRED vs persistence (negative delta = beats persistence) --")
        for name, m in report["future_time_paired_vs_persistence"].items():
            verdict = "SIGNIFICANT" if m["delta_excludes_zero"] else "not distinguishable"
            print(
                f"  future_time  {name:>16}: dMAE={m['delta']:+.4f} "
                f"[{m['ci_lo']:+.4f}, {m['ci_hi']:+.4f}]  {verdict}"
            )
    for name, m in report["leave_subject_out"].items():
        print(f"  leave_subj_out {name:>14}: MAE={m['mae']:.4f} (n_folds={m['n_folds']})")


if __name__ == "__main__":
    main()
