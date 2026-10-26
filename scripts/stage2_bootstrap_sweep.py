"""Participant-level bootstrap for the Stage 2 policy sweep.

The earlier `stage2_policy_sweep.py` varied only the seed controlling
policy tie-breaking jitter on one fixed 64- or 210-subject panel. Most of
its comparisons involve no genuine randomness at all (`round_robin` has
none; `greedy_need` only breaks exact float ties, which are rare once the
estimator has seen any data), so its Holm-corrected p-values describe the
reproducibility of a near-deterministic function under negligible
perturbation, not uncertainty over which patients this cohort contains.
That is the wrong question for a claim meant to generalize beyond these
exact 64 (or 210) people.

Here, each replicate resamples subjects with replacement, stratified by
severity tercile (so every replicate keeps the same group sizes as the
real panel), rebuilds the panel from the real trajectories of the
resampled subjects, and runs every policy once against it. All policies
within a replicate share the same resampled panel, so policy-vs-baseline
comparisons stay paired; the uncertainty this produces is about
which patients are in the cohort, not about tie-breaking.

python -m scripts.stage2_bootstrap_sweep --dataset crosscheck --n-boot 1000
python -m scripts.stage2_bootstrap_sweep --dataset ces --n-boot 1000
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from stage2_env.core import POLICIES
from stage2_env.replay_experiment import load_panel, run_one_policy

BUDGETS = (3, 5, 10)
N_BOOT_DEFAULT = 2000
# Bonferroni-adjusted percentile CI within each budget's family of 5
# non-baseline policies (not pooled across all 3 budgets, which would
# need an infeasible number of replicates to resolve stable 0.17th/
# 99.83th percentiles): alpha=0.05/5=0.01, two-sided percentiles below.
# This is a deliberately modest correction, not a full Holm procedure --
# stated as a limitation, not hidden.
FAMILY_SIZE = 5
ALPHA = 0.05 / FAMILY_SIZE
PCTS = (100 * ALPHA / 2, 100 * (1 - ALPHA / 2))


def bootstrap_panel(specs, traj, group_of, rng: np.random.Generator):
    """Resample subjects with replacement, stratified by tercile group,
    at each group's original size. Returns a fresh (specs, trajectories)
    pair with new sequential ids; no trajectory values are altered, only
    which real subjects (with repeats allowed) populate this replicate's
    panel."""
    by_group: dict[str, list[int]] = defaultdict(list)
    for s in specs:
        by_group[s.group].append(s.subject_id)

    new_specs = []
    new_traj = {}
    new_id = 0
    for group, ids in sorted(by_group.items()):
        sampled = rng.choice(ids, size=len(ids), replace=True)
        for old_id in sampled:
            old_spec = next(s for s in specs if s.subject_id == old_id)
            new_specs.append(
                type(old_spec)(
                    subject_id=new_id,
                    group=group,
                    baseline_offset=old_spec.baseline_offset,
                )
            )
            new_traj[new_id] = traj[old_id]
            new_id += 1
    return new_specs, new_traj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["crosscheck", "ces"], required=True)
    ap.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT)
    ap.add_argument("--baseline", default="round_robin")
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="bootstrap RNG seed; parallel shards use different seeds and are merged by scripts/merge_bootstrap.py",
    )
    ap.add_argument(
        "--config",
        default=None,
        help="override the default configs/dev/replay_<dataset>.yaml, e.g. to "
        "swap in a different world model (same dataset, same metrics)",
    )
    args = ap.parse_args()

    cfg_path = args.config or f"configs/dev/replay_{args.dataset}.yaml"
    base_cfg = yaml.safe_load(open(cfg_path))
    specs, gspecs, traj, group_of = load_panel(base_cfg["data"])
    policies = sorted(POLICIES.names())
    baseline = args.baseline
    print(
        f"{args.dataset}: {len(specs)} subjects, {args.n_boot} bootstrap replicates x "
        f"{len(BUDGETS)} budgets x {len(policies)} policies"
    )

    # per_metric[metric][(budget, policy)] -> list of (policy_value - baseline_value), one per replicate
    diffs: dict[str, dict[tuple[int, str], list[float]]] = defaultdict(lambda: defaultdict(list))
    boot_rng = np.random.default_rng(args.seed)

    for rep in range(args.n_boot):
        rep_specs, rep_traj = bootstrap_panel(specs, traj, group_of, boot_rng)
        for budget in BUDGETS:
            out_by_policy = {}
            for pname in policies:
                out_by_policy[pname] = run_one_policy(
                    pname,
                    rep_specs,
                    gspecs,
                    rep_traj,
                    budget=budget,
                    estimator_name=base_cfg["estimator"]["name"],
                    estimator_params=base_cfg["estimator"].get("params", {}),
                    policy_params=base_cfg.get("policy_params", {}).get(pname, {}),
                    restrict=base_cfg.get("restrict_to_available", True),
                    deterioration_threshold=base_cfg.get("deterioration_threshold", -5.0),
                    seed=0,  # tie-break jitter fixed; the resample IS the randomness now
                )
            base_m = out_by_policy[baseline]
            for pname in policies:
                if pname == baseline:
                    continue
                m = out_by_policy[pname]
                for metric in (
                    "missed_deterioration_days",
                    "allocation_max_gap",
                    "mean_attention_lag_at_risk",
                ):
                    diffs[metric][(budget, pname)].append(m[metric] - base_m[metric])
        if (rep + 1) % max(1, args.n_boot // 10) == 0:
            print(f"  {rep + 1}/{args.n_boot} replicates done", flush=True)

    out: dict = {
        "dataset": args.dataset,
        "n_boot": args.n_boot,
        "budgets": list(BUDGETS),
        "baseline": baseline,
        "n_subjects": len(specs),
        "ci_percentiles": list(PCTS),
        "family_size": FAMILY_SIZE,
        "by_metric": {},
    }
    for metric, cells in diffs.items():
        cell_summ = {}
        for (budget, pname), vals in sorted(cells.items()):
            arr = np.array(vals)
            lo, hi = np.percentile(arr, PCTS)
            cell_summ[f"{pname} @ b={budget}"] = {
                "median_diff": float(np.median(arr)),
                "ci_lo": float(lo),
                "ci_hi": float(hi),
                "excludes_zero": bool(lo > 0 or hi < 0),
            }
        out["by_metric"][metric] = cell_summ
        print(
            f"\n{'=' * 78}\n{args.dataset} / {metric} -- vs {baseline}, "
            f"Bonferroni-adjusted {100 * (1 - ALPHA):.2f}% CI\n{'=' * 78}"
        )
        for k, v in cell_summ.items():
            flag = "  * excludes 0" if v["excludes_zero"] else ""
            print(
                f"  {k:>24} median={v['median_diff']:>+9.3f} "
                f"[{v['ci_lo']:>+9.3f}, {v['ci_hi']:>+9.3f}]{flag}"
            )

    out_path = args.out or f"outputs/stage2_bootstrap_{args.dataset}.json"
    raw = {m: {f"{p} @ b={b}": v for (b, p), v in cells.items()} for m, cells in diffs.items()}
    Path(out_path + ".raw.json").write_text(json.dumps(raw))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
