"""Does the CrossCheck EMA-selection finding replicate on CES?

Two independent mental-health EMA cohorts, deliberately matched at the
adapter level (`caredata/ces.py` uses the same 10 PHQ-adjacent items,
the same 4 passive epochs, the same event construction as the CrossCheck
path) so that a difference between them is a difference between cohorts
and not between pipelines.

This is the strongest evidence a project like this can produce without
new data collection: the same question, the same code, two populations.
A finding that appears in one and not the other is a finding about that
one population -- which is worth knowing and is usually not what the
original write-up claimed.

Every contrast here is paired (arms share the seed, the split and the
fitted model), Holm-corrected across the whole family of cells, and --
where nothing significant turns up -- put through TOST so that
"underpowered" is never reported as "no difference".

python -m scripts.ces_replication
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from eval.stats import summarise_family, tost_equivalence

# Pre-specified before looking at the CES contrasts. PHQ-4 items are
# scored 0-3, so 0.10 MAE is ~3% of the 0-3 range: below the resolution
# at which a clinician would act differently on the prediction.
EQUIVALENCE_MARGIN = 0.10
MARGIN_SWEEP = (0.05, 0.10, 0.25)


def _leaves(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _leaves(v, f"{p}.{k}" if p else k)
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        yield p, float(o)


def load(pattern: str) -> list[dict]:
    fs = sorted(glob.glob(pattern), key=lambda f: int(f.split("seed")[-1].split(".")[0]))
    return [dict(_leaves(json.load(open(f)))) for f in fs]


def cells(per_seed: list[dict], a: str, b: str, metric: str) -> dict:
    """(profile, budget) cells where both policies were measured on every
    seed. Anything partial is dropped rather than compared on a subset --
    that would silently unpair the comparison."""
    out = {}
    for k in per_seed[0]:
        if not k.endswith(f".{a}.{metric}"):
            continue
        cell = k[len("by_burden_profile.") : -len(f".{a}.{metric}")]
        kb = f"by_burden_profile.{cell}.{b}.{metric}"
        if all(k in s and kb in s for s in per_seed):
            out[cell.replace(".by_budget.", " @ ")] = (
                [s[k] for s in per_seed],
                [s[kb] for s in per_seed],
            )
    return out


def _degenerate(a, b):
    return all(abs(x - y) <= 1e-12 for x, y in zip(a, b, strict=False))


def contrast(per_seed, a, b, metric, label, higher_is_better):
    comps = cells(per_seed, a, b, metric)
    dead = {k for k, v in comps.items() if _degenerate(*v)}
    comps = {k: v for k, v in comps.items() if k not in dead}
    if not comps:
        print(f"  {label}: no comparable cells")
        return {}
    fam = summarise_family(comps, higher_is_better=higher_is_better, method="holm", n_boot=5000)
    fam["degenerate_cells"] = sorted(dead)
    print(f"\n  {label}: {a} vs {b} on {metric}, {len(comps)} cells")
    if dead:
        print(
            f"    NO COMPARISON POSSIBLE in {len(dead)} ({', '.join(sorted(dead))}) "
            f"-- arms identical every seed, excluded"
        )
    print(f"    naive {fam['n_significant_naive']} -> Holm {fam['n_significant_corrected']}")
    for k, v in sorted(fam["comparisons"].items(), key=lambda kv: kv[1]["mean_diff"]):
        print(
            f"      {k:>36} {v['mean_diff']:>+8.4f} "
            f"[{v['ci_lo']:>+7.4f},{v['ci_hi']:>+7.4f}] p={v['p_value']:.4f}  {v['verdict']}"
        )

    # TOST on every cell, not only the non-significant ones. Two
    # distinct things need saying and significance alone says neither:
    #   * a non-significant cell may be equivalent OR merely underpowered
    #   * a significant cell may still be smaller than the margin that
    #     would matter -- reliably nonzero and substantively negligible.
    # Running TOST only on the nulls would hide the second case and
    # report a 0.03 MAE shift as a win on the same footing as a 0.73 one.
    eq = {k: tost_equivalence(*comps[k], EQUIVALENCE_MARGIN) for k in fam["comparisons"]}
    fam["tost"] = eq
    n_eq = sum(1 for r in eq.values() if r["equivalent"] and not r["different"])
    n_inc = sum(1 for r in eq.values() if "INCONCLUSIVE" in r["outcome"])
    n_triv = sum(1 for r in eq.values() if r["equivalent"] and r["different"])
    n_real = sum(1 for r in eq.values() if r["different"] and not r["equivalent"])
    print(
        f"    TOST all {len(eq)} cells (margin +/-{EQUIVALENCE_MARGIN}): "
        f"{n_real} materially different, {n_triv} trivially different "
        f"(real but inside margin), {n_eq} equivalent, {n_inc} underpowered"
    )
    if n_triv:
        triv = [k for k, r in eq.items() if r["equivalent"] and r["different"]]
        print(f"      trivially different: {', '.join(sorted(triv))}")
        print(
            f"      ^ statistically reliable, but below the {EQUIVALENCE_MARGIN} MAE "
            f"a clinician would act on"
        )
    for m in MARGIN_SWEEP:
        n = sum(1 for k in eq if tost_equivalence(*comps[k], m)["equivalent"])
        print(f"      margin +/-{m:<5} -> {n}/{len(eq)} within margin")
    return fam


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/ces_replication.json")
    args = ap.parse_args()

    out: dict = {"equivalence_margin": EQUIVALENCE_MARGIN, "by_target": {}}
    for mode in ("forecast", "reconstruct"):
        per = load(f"outputs/_ces_ckpt/{mode}_seed*.json")
        if not per:
            print(f"{mode}: no checkpoints")
            continue
        print(f"\n{'=' * 72}\nCES -- target={mode}, {len(per)} seeds\n{'=' * 72}")
        out["by_target"][mode] = {
            "n_seeds": len(per),
            # MAE: lower is better. frontier_gap: also an error gap here,
            # so lower is better -- opposite of p4's AUPRC convention.
            "ask_all_vs_fixed_mae": contrast(
                per, "ask_all", "fixed_subset", "mae", "raw MAE", higher_is_better=False
            ),
            "ask_all_vs_fixed_gap": contrast(
                per,
                "ask_all",
                "fixed_subset",
                "frontier_gap",
                "cost-matched (frontier_gap)",
                higher_is_better=False,
            ),
            "greedy_vs_fixed_gap": contrast(
                per,
                "greedy_voi",
                "fixed_subset",
                "frontier_gap",
                "cost-matched (frontier_gap)",
                higher_is_better=False,
            ),
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
