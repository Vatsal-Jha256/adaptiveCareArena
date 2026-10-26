"""Multi-seed check for the item-selection benchmark.

The seed controls the grouped-subject split, the model's random_state and
the bandit's replay order at once, and single-seed results can flip on replication. The reported quantity is
`frontier_gap` -- MAE minus the best fixed questionnaire buyable at the
same burden -- because raw MAE confounds "better policy" with "asked more
questions". Negative means adaptivity paid. A gap whose mean +- std
straddles zero is a null result and is reported as one.

python -m emaselect.multi_seed --seeds 5 --target-mode forecast
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import yaml

from emaselect.run import run_experiment


def main() -> None:
    """Run the measurement experiment over several seeds and write per-arm mean and sd."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dev/emaselect_smoke.yaml")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--target-mode", default=None)
    ap.add_argument("--out", default="outputs/emaselect_multi_seed.json")
    ap.add_argument("--ckpt-dir", default="outputs/_emaselect_ckpt")
    args = ap.parse_args()

    with open(args.config) as f:
        base = yaml.safe_load(f)
    if args.target_mode:
        base["target_mode"] = args.target_mode

    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else None
    acc: dict[tuple[str, str, str], dict[str, list[float]]] = {}
    rank_orders = []

    for seed in range(args.seeds):
        tag = f"{base.get('target_mode', 'forecast')}_seed{seed}"
        ckpt = (ckpt_dir / f"{tag}.json") if ckpt_dir else None
        if ckpt is not None and ckpt.exists():
            res = json.loads(ckpt.read_text())
            print(f"seed {seed} loaded from checkpoint", flush=True)
        else:
            cfg = dict(base)
            cfg["seed"] = seed
            res = run_experiment(cfg)
            if ckpt is not None:
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                ckpt.write_text(json.dumps(res))
            print(f"seed {seed} done", flush=True)
        rank_orders.append(res["item_rank_order"])
        for prof, block in res["by_burden_profile"].items():
            for budget, by_pol in block["by_budget"].items():
                for pol, m in by_pol.items():
                    slot = acc.setdefault((prof, budget, pol), {"mae": [], "burden": [], "gap": []})
                    slot["mae"].append(m["mae"])
                    slot["burden"].append(m["mean_burden"])
                    slot["gap"].append(m["frontier_gap"])

    def ms(xs):
        """Return (mean, sample sd); the sd is 0.0 for a single value."""
        return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)

    summary: dict = {}
    for (prof, budget, pol), slot in acc.items():
        mm, msd = ms(slot["mae"])
        bm, _ = ms(slot["burden"])
        gm, gsd = ms(slot["gap"])
        summary.setdefault(prof, {}).setdefault(budget, {})[pol] = {
            "mae_mean": mm,
            "mae_std": msd,
            "burden_mean": bm,
            "frontier_gap_mean": gm,
            "frontier_gap_std": gsd,
            "beats_fixed_at_matched_burden": bool(gm + gsd < 0.0),
        }

    # How stable is the item ranking across seeds? An unstable "most
    # informative question" is itself the finding -- it means the data
    # does not identify one.
    top1 = [r[0] for r in rank_orders]
    out_obj = {
        "seeds": args.seeds,
        "target_mode": base.get("target_mode", "forecast"),
        "item_rank_orders": rank_orders,
        "top_item_by_seed": top1,
        "top_item_stable": len(set(top1)) == 1,
        "by_burden_profile": summary,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_obj, indent=2))

    print(f"\n===== item-selection, {args.seeds} seeds, target={out_obj['target_mode']} =====")
    print(f"most informative item per seed: {top1}  (stable={out_obj['top_item_stable']})")
    for prof, by_b in summary.items():
        print(f"-- {prof} --")
        for budget, by_pol in by_b.items():
            print(f"  budget={budget}")
            for pol, m in by_pol.items():
                flag = "  <-- BEATS FIXED" if m["beats_fixed_at_matched_burden"] else ""
                print(
                    f"    {pol:>16}: MAE={m['mae_mean']:.4f}+-{m['mae_std']:.4f}  "
                    f"burden={m['burden_mean']:.2f}  gap={m['frontier_gap_mean']:+.4f}"
                    f"+-{m['frontier_gap_std']:.4f}{flag}"
                )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
