"""Measurement: which questions are worth a patient's burden at an EMA
that already happened?

The signal comes from the item responses themselves, not from a
passive-sensing forecast. Stage 1 found that persistence beat every
learned model, persistence needs today's EMA score, and today's score
costs all ten questions. So the question is how much of the
questionnaire is actually needed to forecast as well as full persistence
does.

python -m emaselect.run --config configs/dev/emaselect_smoke.yaml
"""

from __future__ import annotations

import argparse

import numpy as np

from emaselect.dataset import assert_no_item_leakage, build_ema_events
from emaselect.model import MaskAwareEMAModel
from emaselect.policies import (
    AskAllPolicy,
    AskNonePolicy,
    FixedSubsetPolicy,
    GreedyVOIPolicy,
    RandomMatchedPolicy,
    ThompsonItemPolicy,
)
from emaselect.uncertainty import LocallyAdaptiveConformal, aurc, risk_coverage_curve
from runtime.manifest import load_config, make_run_dir, write_manifest, write_results


def _mae(pred, y):
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(y))))


def _split(events, rng, n_train, n_val):
    subs = sorted({e.subject_id for e in events})
    perm = list(rng.permutation(subs))
    tr, va = set(perm[:n_train]), set(perm[n_train : n_train + n_val])
    te = set(perm[n_train + n_val :])

    def pick(subjects):
        return [e for e in events if e.subject_id in subjects]

    return pick(tr), pick(va), pick(te)


def _rank_items_by_marginal_value(model, val, y_val, items) -> list[str]:
    """Standalone value of each item alone (plus free context) on val."""
    scores = {}
    for it in items:
        scores[it] = _mae(model.predict(val, [{it}] * len(val)), y_val)
    return sorted(scores, key=lambda k: scores[k])  # lower MAE = more valuable


def _best_fixed_subset(model, val, y_val, cost_profile, budget, cache=None) -> set[str]:
    """Greedy forward selection under the burden budget, on validation."""
    chosen, spent, best = set(), 0.0, float("inf")
    remaining = set(cost_profile)
    while remaining:
        cands = {i: chosen | {i} for i in remaining if spent + cost_profile[i] <= budget}
        if not cands:
            break
        scored = {}
        for i, sub in cands.items():
            key = frozenset(sub)
            if cache is not None and key in cache:
                scored[i] = cache[key]
            else:
                scored[i] = _mae(model.predict(val, [sub] * len(val)), y_val)
                if cache is not None:
                    cache[key] = scored[i]
        pick = min(scored, key=lambda k: scored[k])
        if scored[pick] >= best:
            break
        best = scored[pick]
        chosen.add(pick)
        spent += cost_profile[pick]
        remaining.discard(pick)
    return chosen


def _static_frontier(model, val, y_val, test, y_test, cost_profile, n_points=16):
    """Best fixed short-form questionnaire at each budget, chosen on val
    and scored on test. Without this an adaptive policy that underspends
    gets compared against baselines that spend the full budget, which
    confounds "better policy" with "asked more questions"."""
    total = sum(cost_profile.values())
    cache, seen, pts = {}, set(), []
    for k in range(n_points + 1):
        b = total * k / n_points
        sub = _best_fixed_subset(model, val, y_val, cost_profile, b, cache)
        key = frozenset(sub)
        if key in seen:
            continue
        seen.add(key)
        cost = sum(cost_profile[i] for i in sub)
        pts.append((float(cost), _mae(model.predict(test, [sub] * len(test)), y_test)))
    return sorted(pts)


def _frontier_mae_at(frontier, cost) -> float:
    """Best (lowest) MAE buyable at or below `cost` from a fixed
    questionnaire -- running min, so this never credits the static
    baseline with a questionnaire it cannot afford."""
    best = float("nan")
    for c, m in frontier:
        if c <= cost + 1e-9:
            best = m if np.isnan(best) else min(best, m)
    return best


def _build_events(cfg: dict):
    """Dispatch on `dataset.name`. Returns (events, items, burden_profiles).

    Two instruments are supported and neither is privileged: CrossCheck's
    ten signed EMA items and CES's four PHQ-4 items. Everything
    downstream -- model, policies, frontier -- takes the item set as
    data, so adding a third instrument means adding an event builder and
    a burden profile, not touching any of the machinery."""
    name = cfg["dataset"].get("name", "crosscheck")
    if name == "crosscheck":
        from emaselect.items import BURDEN_PROFILES as profiles
        from emaselect.items import EMA_ITEMS as items

        events = build_ema_events(
            local_root=cfg["dataset"].get("local_root", "data/raw/crosscheck"),
            target_mode=cfg.get("target_mode", "forecast"),
        )
    elif name == "ces":
        from emaselect.ces_events import CES_BURDEN_PROFILES as profiles
        from emaselect.ces_events import PHQ4_ITEMS as items
        from emaselect.ces_events import build_ces_events

        events = build_ces_events(
            local_root=cfg["dataset"].get("local_root", "data/raw/ces"),
            target_mode=cfg.get("target_mode", "forecast"),
            min_ema=cfg["dataset"].get("min_ema", 20),
        )
    else:
        raise ValueError(f"unknown dataset {name!r}")
    return events, tuple(items), profiles


def run_experiment(cfg: dict) -> dict:
    """Run one measurement experiment from a config dict and return its metrics.

    Fits the mask-aware model on a training split, selects fixed subsets on a
    validation split under each burden profile and budget, and evaluates every
    policy on the held-out test split.
    """
    rng = np.random.default_rng(cfg["seed"])
    events, items, burden_profiles = _build_events(cfg)
    assert_no_item_leakage(events, items)

    train, val, test = _split(events, rng, cfg["split"]["n_train"], cfg["split"]["n_val"])
    model = MaskAwareEMAModel(items, random_state=cfg["seed"])
    model.fit(train)

    y_val = np.array([e.target_y for e in val])
    y_test = np.array([e.target_y for e in test])
    rank_order = _rank_items_by_marginal_value(model, val, y_val, items)

    results: dict = {
        "target_mode": cfg.get("target_mode", "forecast"),
        "n_train_subjects": len({e.subject_id for e in train}),
        "n_test_subjects": len({e.subject_id for e in test}),
        "n_test_events": len(test),
        "item_rank_order": rank_order,
        "by_burden_profile": {},
    }

    for pname in cfg.get("burden_profiles", list(burden_profiles)):
        cost_profile = burden_profiles[pname]
        frontier = _static_frontier(model, val, y_val, test, y_test, cost_profile)
        block = {"_static_frontier": frontier, "by_budget": {}}

        for budget in cfg["budgets"]:
            subset = _best_fixed_subset(model, val, y_val, cost_profile, budget)
            # Factories, not instances. ThompsonItemPolicy carries Beta
            # posterior state across run() calls, so the calibration pass
            # below must get a policy that has never seen test labels --
            # see the comment there.
            policy_factories = [
                AskAllPolicy,
                AskNonePolicy,
                lambda subset=subset: FixedSubsetPolicy(subset),
                RandomMatchedPolicy,
                lambda: GreedyVOIPolicy(rank_order),
                ThompsonItemPolicy,
            ]
            out = {}
            for make_policy in policy_factories:
                pol = make_policy()
                asked, cost = pol.run(test, model, budget, cost_profile, rng, y_true=y_test)
                pred = model.predict(test, asked)
                mae = _mae(pred, y_test)
                matched = _frontier_mae_at(frontier, cost)
                entry = {
                    "mae": mae,
                    "mean_burden": float(cost),
                    "frontier_matched_mae": float(matched),
                    # Negative = beats the best fixed questionnaire at the
                    # same burden, i.e. adaptivity actually paid.
                    "frontier_gap": float(mae - matched),
                    "mean_items_asked": float(np.mean([len(a) for a in asked])),
                }

                # Selective prediction. Conformal is calibrated on val
                # under the same kind of acquisition this policy makes, so
                # the intervals describe the questionnaires actually
                # deployed rather than full ones.
                #
                # A fresh policy instance, not `pol`: ThompsonItemPolicy
                # mutates its Beta posterior from the outcomes it is given,
                # and `pol` has just been run on test with y_test. Reusing
                # it here would let test labels shape which items are asked
                # on the calibration split, and therefore shape the
                # uncertainty estimates the test predictions are judged by.
                if cfg.get("selective_prediction", True):
                    cal_policy = make_policy()
                    val_asked, _ = cal_policy.run(
                        val,
                        model,
                        budget,
                        cost_profile,
                        np.random.default_rng(cfg["seed"]),
                        y_true=y_val,
                    )
                    conf = LocallyAdaptiveConformal(
                        alpha=cfg.get("alpha", 0.1), random_state=cfg["seed"]
                    )
                    conf.calibrate(model, val, val_asked, y_val)
                    hw = conf.interval_halfwidth(model, test, asked)
                    curve = risk_coverage_curve(y_test, pred, hw)
                    entry["mean_interval_halfwidth"] = float(np.mean(hw))
                    entry["empirical_coverage"] = float(np.mean(np.abs(y_test - pred) <= hw))
                    entry["risk_coverage"] = curve
                    entry["aurc"] = aurc(curve)
                out[pol.name] = entry
            block["by_budget"][str(budget)] = out
        results["by_burden_profile"][pname] = block

    return results


def main() -> None:
    """Command-line entry point: run the config at ``--config`` into ``--base-out``."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--base-out", default="outputs")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["_source_path"] = args.config
    run_dir = make_run_dir(cfg, base=args.base_out)
    write_manifest(run_dir, cfg)
    results = run_experiment(cfg)
    digest = write_results(run_dir, results)

    print(f"run: {run_dir}\nresults digest: {digest}")
    print(
        f"target_mode={results['target_mode']}  test: {results['n_test_subjects']} subjects, "
        f"{results['n_test_events']} events"
    )
    print(f"item value order (best first): {results['item_rank_order']}")
    for pname, block in results["by_burden_profile"].items():
        print(f"-- burden profile: {pname} --")
        print(
            "   fixed-questionnaire frontier: "
            + "  ".join(f"{c:.1f}->{m:.3f}" for c, m in block["_static_frontier"])
        )
        for budget, by_pol in block["by_budget"].items():
            print(f"  budget={budget}")
            for n, m in by_pol.items():
                extra = ""
                if "mean_interval_halfwidth" in m:
                    extra = (
                        f"  width={m['mean_interval_halfwidth']:.2f}"
                        f"  cov={m['empirical_coverage']:.2f}  aurc={m['aurc']:.3f}"
                    )
                print(
                    f"    {n:>16}: MAE={m['mae']:.4f}  burden={m['mean_burden']:.2f}  "
                    f"items={m['mean_items_asked']:.1f}  vs_fixed@burden={m['frontier_gap']:+.4f}{extra}"
                )


if __name__ == "__main__":
    main()
