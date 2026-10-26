"""Direct test of the validation-set-overfitting hypothesis.

The observation to explain: a cheap fixed rule beats an expensive
val-selected subset in two separate environments --
  * emaselect/reconstruct: `greedy_voi` (a fixed rank order) beats
    `fixed_subset` (greedy forward selection under budget) in 2/12 cells
  * p4/ExtraSensory: `cheapest_fixed` (0.5673) beats `best_fixed_subset`
    (0.5547) at identical cost

Hypothesis, sharpened. "The cheap rule does not use validation" would be
wrong -- both are fitted on val. The difference is how much selection
happens there:

  greedy_voi         10 marginal scores, each item evaluated alone.
                     Independent estimates, no compounding.
  best_fixed_subset  up to 55 subset evaluations, each conditioned on
                     the picks before it. Sequential argmax over noisy
                     val estimates.

Repeated argmax over noisy estimates inflates the selected option's
apparent value, and the inflation does not transfer to test. That is
selection-induced optimism -- the winner's curse.
The same phenomenon contaminates the baseline here.

Prediction: the `greedy_voi` minus `fixed_subset` gap should shrink as
the validation set grows, because more val subjects means less noise for
the sequential search to fit. If the gap is flat in val size, the
explanation is wrong and the cross-environment pattern needs a different
one.

This matters beyond one number: if the sophisticated methods lose
because of small-N selection rather than because adaptivity is useless,
the whole cross-environment framing changes.

**Design and its confound, stated up front.** 74 subjects is not enough
to hold both train and test fixed while varying val. Train+val is held
at 52 so the test split is constant at 22 subjects and every condition
is evaluated on the same amount of data; train therefore shrinks as val
grows (46/6, 40/12, 28/24). Training size affects model quality, but it
affects every policy in a condition identically, and the quantity of
interest is a within-condition contrast (greedy_voi vs fixed_subset), so
a shared model change largely cancels. It does not cancel perfectly, and
a clean test needs a larger cohort.

python -m scripts.val_overfitting_test --seeds 5
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from emaselect.run import run_experiment

TRAIN_PLUS_VAL = 52  # leaves 22 test subjects in every condition
SPLITS = [(46, 6), (40, 12), (28, 24)]

BASE = {
    "experiment_id": "val_overfitting_test",
    "seed": 0,
    "target_mode": "reconstruct",  # where the effect was seen
    "dataset": {"local_root": "data/raw/crosscheck"},
    "budgets": [2.0, 4.0, 6.0, 8.0],
    "burden_profiles": ["uniform", "sensitivity_weighted", "inverted_sensitivity"],
    "selective_prediction": False,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="outputs/val_overfitting_test.json")
    ap.add_argument("--ckpt-dir", default="outputs/_val_overfit_ckpt")
    args = ap.parse_args()

    ckpt = Path(args.ckpt_dir)
    report = {}
    print(f"{'n_val':>6} {'n_train':>8} {'greedy_voi - fixed_subset gap':>32} {'verdict':>14}")
    for n_train, n_val in SPLITS:
        gaps: list[float] = []
        for seed in range(args.seeds):
            f = ckpt / f"val{n_val}_seed{seed}.json"
            if f.exists():
                res = json.loads(f.read_text())
            else:
                cfg = {**BASE, "seed": seed, "split": {"n_train": n_train, "n_val": n_val}}
                res = run_experiment(cfg)
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(json.dumps(res))
                print(f"  n_val={n_val} seed {seed} done", flush=True)
            for _prof, blk in res["by_burden_profile"].items():
                for _b, by in blk["by_budget"].items():
                    # frontier_gap is measured against the fixed frontier,
                    # so the greedy_voi-minus-fixed_subset difference is
                    # the direct contrast. Negative = the cheap rank order
                    # beats the expensive val-selected subset.
                    gaps.append(by["greedy_voi"]["mae"] - by["fixed_subset"]["mae"])
        m = st.mean(gaps)
        sd = st.stdev(gaps) if len(gaps) > 1 else 0.0
        verdict = (
            "cheap wins" if m + sd < 0 else "expensive wins" if m - sd > 0 else "no difference"
        )
        report[f"n_val={n_val}"] = {
            "n_train": n_train,
            "n_val": n_val,
            "mean_diff": m,
            "std": sd,
            "n_cells": len(gaps),
            "verdict": verdict,
        }
        print(f"{n_val:>6} {n_train:>8} {m:>+20.4f}±{sd:<10.4f} {verdict:>14}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    ms = [report[k]["mean_diff"] for k in report]
    print()
    print("HYPOTHESIS: if val-set overfitting explains it, the cheap rule's advantage")
    print("            should SHRINK (mean_diff rise toward 0) as n_val grows.")
    print(f"            observed: {ms[0]:+.4f} (n_val=6) -> {ms[1]:+.4f} (12) -> {ms[2]:+.4f} (24)")
    print(f"            trend {'SUPPORTS' if ms[2] > ms[0] else 'does NOT support'} the hypothesis")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
