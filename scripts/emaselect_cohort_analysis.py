"""Item-selection (measurement-axis) analysis, generalized from
scripts/ces_replication.py to run identically against either cohort's
per-seed checkpoints, so EMASelect-v0's two-cohort claim is produced by
one script on two datasets rather than two separately-written analyses.

Every contrast is paired (arms share the seed, split and fitted model),
Holm-corrected across the family of (burden profile, budget) cells, and
-- where nothing significant turns up -- put through TOST so
"underpowered" is never reported as "no difference".

python -m scripts.emaselect_cohort_analysis --ckpt-glob "outputs/_emaselect_ckpt/forecast_seed*.json" --label crosscheck --margin 0.10 --out outputs/emaselect_crosscheck_analysis.json
python -m scripts.emaselect_cohort_analysis --ckpt-glob "outputs/_ces_ckpt/forecast_seed*.json" --label ces --margin 0.10 --out outputs/emaselect_ces_analysis.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from eval.stats import summarise_family, tost_equivalence

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


def contrast(per_seed, a, b, metric, label, higher_is_better, margin):
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
        print(f"    identical every seed in {len(dead)}: {', '.join(sorted(dead))} -- excluded")
    print(f"    naive {fam['n_significant_naive']} -> Holm {fam['n_significant_corrected']}")
    for k, v in sorted(fam["comparisons"].items(), key=lambda kv: kv[1]["mean_diff"]):
        print(
            f"      {k:>36} {v['mean_diff']:>+8.4f} "
            f"[{v['ci_lo']:>+7.4f},{v['ci_hi']:>+7.4f}] p={v['p_value']:.4f}  {v['verdict']}"
        )

    eq = {k: tost_equivalence(*comps[k], margin) for k in fam["comparisons"]}
    fam["tost"] = eq
    n_eq = sum(1 for r in eq.values() if r["equivalent"] and not r["different"])
    n_inc = sum(1 for r in eq.values() if "INCONCLUSIVE" in r["outcome"])
    n_triv = sum(1 for r in eq.values() if r["equivalent"] and r["different"])
    n_real = sum(1 for r in eq.values() if r["different"] and not r["equivalent"])
    print(
        f"    TOST all {len(eq)} cells (margin +/-{margin}): "
        f"{n_real} materially different, {n_triv} trivially different, "
        f"{n_eq} equivalent, {n_inc} underpowered"
    )
    for m in MARGIN_SWEEP:
        n = sum(1 for k in eq if tost_equivalence(*comps[k], m)["equivalent"])
        print(f"      margin +/-{m:<5} -> {n}/{len(eq)} within margin")
    return fam


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-glob", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    per = load(args.ckpt_glob)
    if not per:
        raise SystemExit(f"no checkpoints matched {args.ckpt_glob!r}")
    print(f"{'=' * 72}\n{args.label} -- {len(per)} seeds\n{'=' * 72}")

    out: dict = {
        "label": args.label,
        "n_seeds": len(per),
        "equivalence_margin": args.margin,
        "contrasts": {},
    }
    out["contrasts"]["ask_all_vs_fixed_mae"] = contrast(
        per, "ask_all", "fixed_subset", "mae", "raw MAE", higher_is_better=False, margin=args.margin
    )
    out["contrasts"]["ask_all_vs_fixed_gap"] = contrast(
        per,
        "ask_all",
        "fixed_subset",
        "frontier_gap",
        "cost-matched (ask_all)",
        higher_is_better=False,
        margin=args.margin,
    )
    out["contrasts"]["thompson_vs_fixed_gap"] = contrast(
        per,
        "thompson_items",
        "fixed_subset",
        "frontier_gap",
        "cost-matched (adaptive per-patient)",
        higher_is_better=False,
        margin=args.margin,
    )
    out["contrasts"]["greedy_vs_fixed_gap"] = contrast(
        per,
        "greedy_voi",
        "fixed_subset",
        "frontier_gap",
        "cost-matched (static greedy VOI)",
        higher_is_better=False,
        margin=args.margin,
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
