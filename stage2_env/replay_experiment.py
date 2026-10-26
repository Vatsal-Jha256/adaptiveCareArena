"""Runs the registered policies against real CrossCheck trajectories.

The candidate set each round is today's available subjects -- those with
a real recorded EMA -- not the whole panel. This is a deliberate design
decision and it changes the results substantially:

Measured first: when policies were allowed
to choose from all 64 subjects, every one of them wasted 999-1259 of its
1350 review slots on subjects with no new assessment that day, because
only ~16 of 64 have an EMA on any given day. Under that setup `random`
and `round_robin` came out best on belief error -- not because spreading
attention is a good allocation strategy, but because spreading it happens
to land on available subjects more often. That is a confound that
measures availability-luck rather than allocation quality.

Restricting the candidate set models the real decision: a reviewer reads
some of today's new assessments. The alternative framing -- "which charts
do I open, including ones with nothing new?" -- is also legitimate and
arguably more realistic, but it makes the task partly about predicting
availability, which none of these policies were designed for. It is kept
available via `restrict_to_available=False` so the confound can be
demonstrated rather than merely asserted.

python -m stage2_env.replay_experiment --config configs/dev/replay_crosscheck.yaml
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from runtime.manifest import load_config, make_run_dir, write_manifest, write_results
from stage2_env import estimators as _estimators  # noqa: F401  (registry side effects)
from stage2_env import policies as _policies  # noqa: F401
from stage2_env.core import ESTIMATORS, POLICIES
from stage2_env.replay_data import build_replay_inputs as build_crosscheck_inputs
from stage2_env.replay_data_ces import build_replay_inputs as build_ces_inputs
from stage2_env.replay_world import ReplaySubjectWorld

BUILDERS = {"crosscheck": build_crosscheck_inputs, "ces": build_ces_inputs}


def load_panel(data_cfg: dict) -> tuple[list, dict, dict, dict]:
    """Builds (specs, group_specs, trajectories, group_of) once from the
    real files. Separated from `run_one_policy`/`run_experiment` so a
    bootstrap resampler can reuse the already-parsed pool across many
    resamples instead of re-reading the source CSVs every replicate."""
    source = data_cfg.get("source", "crosscheck")
    build_inputs = BUILDERS[source]
    return build_inputs(
        local_root=data_cfg.get("local_root", "data/raw/crosscheck"),
        min_ema=data_cfg.get("min_ema", 20),
        align=data_cfg.get("align", "relative"),
        horizon=data_cfg.get("horizon", 270),
    )


def run_one_policy(
    pname: str,
    specs: list,
    gspecs: dict,
    traj: dict,
    *,
    budget: int,
    estimator_name: str,
    estimator_params: dict,
    policy_params: dict,
    restrict: bool,
    deterioration_threshold: float,
    seed: int,
) -> dict:
    """Runs a single policy against a single (possibly resampled) panel.
    This is the reusable core: `run_experiment` calls it once per policy
    on the real panel; `scripts/stage2_bootstrap_sweep.py` calls it many
    times per policy on bootstrap-resampled panels built from the same
    parsed pool, without re-touching disk."""
    n_rounds = max(len(t) for t in traj.values())
    group_of = {s.subject_id: s.group for s in specs}
    rng = np.random.default_rng(seed)
    world = ReplaySubjectWorld(specs, gspecs, traj, rng)
    est = ESTIMATORS.get(estimator_name)(**estimator_params)
    pol = POLICIES.get(pname)(**policy_params)

    counts = {s.subject_id: 0 for s in specs}
    auto_ingested = wasted = 0
    attention_lag_at_risk: list[float] = []
    missed_deterioration = 0

    for r in range(n_rounds):
        feats_all = world.current_features(r, n_rounds, counts)
        avail = world.available(r)
        feats = {k: v for k, v in feats_all.items() if k in avail} if restrict else feats_all
        if not feats:
            world.pull_and_advance(r, set(), feats_all)
            continue

        need = est.predict(feats)
        chosen = pol.select(feats, need, budget, r, rng)
        wasted += len(chosen - avail)

        obs, truth = world.pull_and_advance(r, chosen, feats_all)
        est.update(obs)  # every real EMA this round, reviewed or not
        auto_ingested += len(obs)
        for sid in chosen & avail:
            counts[sid] += 1

        # Detection: a bad day (true state at/below the
        # clinically-set threshold) that no human reviewed today.
        # Attention lag: for that same set of bad days, how many rounds
        # had it already been since a human last looked at this subject,
        # Before today's decision (feats_all carries rounds_since_check
        # as of the start of this round). This is policy-comparable and
        # does not depend on the estimator's own belief, which is now
        # auto-ingested and therefore no longer a policy-dependent
        # quantity (see replay_world.py).
        for sid, tv in truth.items():
            if tv <= deterioration_threshold:
                attention_lag_at_risk.append(float(feats_all[sid].rounds_since_check))
                if sid not in chosen:
                    missed_deterioration += 1

    per_group = Counter()
    for sid, c in counts.items():
        per_group[group_of[sid]] += c
    rates = {
        g: per_group[g] / max(1, sum(1 for s in specs if s.group == g))
        for g in sorted({s.group for s in specs})
    }

    return {
        "reviewed_checks": sum(counts.values()),
        "auto_ingested_events": auto_ingested,
        "wasted_slots": wasted,
        "mean_attention_lag_at_risk": (
            float(np.mean(attention_lag_at_risk)) if attention_lag_at_risk else float("nan")
        ),
        "missed_deterioration_days": missed_deterioration,
        "checks_per_subject_by_group": rates,
        "allocation_max_gap": float(max(rates.values()) - min(rates.values())),
    }


def run_experiment(cfg: dict) -> dict:
    """Replay every registered policy and estimator pair on real trajectories.

    ``restrict_to_available`` (default True) limits each round's candidates to
    subjects with a recorded report that day.
    """
    specs, gspecs, traj, _ = load_panel(cfg.get("data", {}))
    n_rounds = max(len(t) for t in traj.values())
    budget = cfg["budget"]["b_per_round"]
    restrict = cfg.get("restrict_to_available", True)
    seed = cfg["seed"]

    results: dict = {
        "n_subjects": len(specs),
        "n_rounds": n_rounds,
        "budget": budget,
        "restrict_to_available": restrict,
        "total_real_ema_events": int(sum(1 for t in traj.values() for v in t if v is not None)),
        "by_policy": {},
    }

    for pname in cfg.get("policies", POLICIES.names()):
        results["by_policy"][pname] = run_one_policy(
            pname,
            specs,
            gspecs,
            traj,
            budget=budget,
            estimator_name=cfg["estimator"]["name"],
            estimator_params=cfg["estimator"].get("params", {}),
            policy_params=cfg.get("policy_params", {}).get(pname, {}),
            restrict=restrict,
            deterioration_threshold=cfg.get("deterioration_threshold", -5.0),
            seed=seed,
        )

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
    res = run_experiment(cfg)
    digest = write_results(run_dir, res)

    print(f"run: {run_dir}\nresults digest: {digest}")
    print(
        f"REAL CrossCheck replay: {res['n_subjects']} subjects, {res['n_rounds']} rounds, "
        f"{res['total_real_ema_events']} real EMA events, budget={res['budget']}/round, "
        f"restrict_to_available={res['restrict_to_available']}"
    )
    print(
        f"{'policy':>16} {'reviewed':>9} {'wasted':>7} {'attn_lag':>10} {'missed_det':>11} {'alloc_gap':>10}"
    )
    for p, m in res["by_policy"].items():
        print(
            f"{p:>16} {m['reviewed_checks']:>9} {m['wasted_slots']:>7} "
            f"{m['mean_attention_lag_at_risk']:>10.3f} {m['missed_deterioration_days']:>11} "
            f"{m['allocation_max_gap']:>10.2f}"
        )


if __name__ == "__main__":
    main()
