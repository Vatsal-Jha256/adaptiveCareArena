"""Compares every registered Stage 2 policy on real CrossCheck
trajectories, across budget levels, paired by seed and Holm-corrected.

`stage2_env/replay_world.py` replays the recorded trajectories (64
subjects x 270 rounds, 4,335 real EMA events), so the numbers describe
CrossCheck rather than a generator. The synthetic simulator lives in
tests/fixtures/ and only exercises the environment mechanics.

Every arm shares the seed, the trajectory draw and the estimator, so
comparisons are paired; the family spans policies x budgets, so p-values
are Holm-corrected; and anything non-significant goes through TOST so
"no difference" is never reported for an underpowered cell.

python scripts/stage2_policy_sweep.py --seeds 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from eval.stats import summarise_family, tost_equivalence
from stage2_env.core import POLICIES
from stage2_env.replay_experiment import run_experiment

BUDGETS = (3, 5, 10)

# Pre-specified per metric, from the observed scales on this cohort.
# allocation_max_gap is a per-subject check-rate difference between
# groups over 270 rounds, so 0.02 is ~5 checks per subject across the
# study -- below what would change a staffing decision.
MARGINS = {
    "allocation_max_gap": 0.02,
    "missed_deterioration_days": 5.0,
    "mean_belief_staleness": 0.25,
}
HIGHER_IS_BETTER = dict.fromkeys(MARGINS, False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dev/replay_crosscheck.yaml")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--baseline", default="round_robin")
    ap.add_argument("--out", default="outputs/stage2_policy_sweep.json")
    args = ap.parse_args()

    base_cfg = yaml.safe_load(open(args.config))
    policies = sorted(POLICIES.names())
    print(
        f"replay CrossCheck: {len(policies)} policies x {len(BUDGETS)} budgets x {args.seeds} seeds"
    )
    print(f"policies: {', '.join(policies)}")

    # per_seed[(budget, policy)][metric] -> list over seeds
    per_seed: dict[tuple[int, str], dict[str, list[float]]] = {}
    for budget in BUDGETS:
        for seed in range(args.seeds):
            cfg = dict(base_cfg)
            cfg["seed"] = seed
            cfg["budget"] = {"b_per_round": budget}
            cfg["policies"] = policies
            res = run_experiment(cfg)
            for pname, m in res["by_policy"].items():
                slot = per_seed.setdefault((budget, pname), {})
                for metric in MARGINS:
                    slot.setdefault(metric, []).append(float(m[metric]))
        print(f"  budget {budget}: {args.seeds} seeds done", flush=True)

    baseline = args.baseline
    if baseline not in policies:
        baseline = policies[0]
        print(f"\n(baseline {args.baseline!r} not registered; using {baseline!r})")

    out: dict = {
        "n_seeds": args.seeds,
        "budgets": list(BUDGETS),
        "baseline": baseline,
        "margins": MARGINS,
        "by_metric": {},
    }

    for metric, margin in MARGINS.items():
        comps, dead = {}, []
        for (budget, pname), slot in sorted(per_seed.items()):
            if pname == baseline:
                continue
            a = slot[metric]
            b = per_seed[(budget, baseline)][metric]
            key = f"{pname} @ b={budget}"
            if all(abs(x - y) <= 1e-12 for x, y in zip(a, b, strict=False)):
                dead.append(key)
            else:
                comps[key] = (a, b)
        if not comps:
            print(f"\n{metric}: no comparable cells ({len(dead)} identical to baseline every seed)")
            continue

        fam = summarise_family(
            comps, higher_is_better=HIGHER_IS_BETTER[metric], method="holm", n_boot=5000
        )
        fam["degenerate_cells"] = dead
        fam["tost"] = {k: tost_equivalence(*comps[k], margin) for k in comps}
        out["by_metric"][metric] = fam

        print(
            f"\n{'=' * 74}\n{metric} -- vs {baseline}, {len(comps)} cells, Holm "
            f"(margin +/-{margin})\n{'=' * 74}"
        )
        if dead:
            print(
                f"  identical to baseline every seed in {len(dead)}: {', '.join(dead)} -- excluded"
            )
        print(f"  naive {fam['n_significant_naive']} -> Holm {fam['n_significant_corrected']}")
        for k, v in sorted(fam["comparisons"].items(), key=lambda kv: kv[1]["mean_diff"]):
            t = fam["tost"][k]
            print(
                f"   {k:>30} {v['mean_diff']:>+9.4f} "
                f"[{v['ci_lo']:>+8.4f},{v['ci_hi']:>+8.4f}] p={v['p_value']:.4f} "
                f"{v['verdict']:>9} | {t['outcome'].split('(')[0].strip()}"
            )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
