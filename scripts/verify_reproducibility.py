"""Executable reproducibility check for adaptiveCareArena's stage2_env.

Re-runs each registered experiment twice from the same config and seed and
compares a digest of the results. Skips when the real data are absent.

stage2_env threads one `rng` through world, policy and estimator across
every round, so an order-dependent iteration anywhere in that loop
changes the entire trajectory -- exactly what this catches.

python -m scripts.verify_reproducibility
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import yaml

from runtime.reproducibility import full_fingerprint, results_digest

EXPERIMENTS = [
    ("replay_crosscheck", "stage2_env.replay_experiment", "configs/dev/replay_crosscheck.yaml"),
    # emaselect threads one rng through a grouped-subject split, the
    # model's dropout augmentation, the conformal calibration split and
    # ThompsonItemPolicy's replay order -- four consumers of the same
    # generator, which is exactly the shape that produces
    # order-dependent non-determinism.
    ("emaselect", "emaselect.run", "configs/dev/emaselect_smoke.yaml"),
]


def _check_one(name: str, module_name: str, config_path: str) -> dict:
    import importlib

    if not Path(config_path).exists():
        return {"name": name, "status": "SKIP", "reason": f"config not found: {config_path}"}

    try:
        run_experiment = importlib.import_module(module_name).run_experiment
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        first = run_experiment(dict(cfg))
        second = run_experiment(dict(cfg))
    except FileNotFoundError as exc:
        # Restricted datasets (PhysioNet DUA) legitimately absent on a
        # given machine -- a skip, not a failure. Reported as skip so an
        # absent dataset can never be mistaken for a passing check.
        return {"name": name, "status": "SKIP", "reason": f"data unavailable: {exc}"}
    except Exception:
        return {"name": name, "status": "ERROR", "reason": traceback.format_exc(limit=3)}

    d1, d2 = results_digest(first), results_digest(second)
    return {
        "name": name,
        "status": "PASS" if d1 == d2 else "FAIL",
        "digest": d1,
        "digest_rerun": d2,
        "config": config_path,
    }


def _first_divergence(a, b, path="") -> str | None:
    """Locates where two results trees differ, so a fail points at the
    offending metric instead of just reporting two unequal hashes."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f"{path}.{k} present in only one run"
            found = _first_divergence(a[k], b[k], f"{path}.{k}")
            if found:
                return found
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path} length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b, strict=False)):
            found = _first_divergence(x, y, f"{path}[{i}]")
            if found:
                return found
        return None
    if isinstance(a, float) and isinstance(b, float):
        return None if f"{a:.12g}" == f"{b:.12g}" else f"{path}: {a!r} vs {b!r}"
    return None if a == b else f"{path}: {a!r} vs {b!r}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="only the two fastest experiments")
    parser.add_argument("--only", default=None, help="run a single experiment by name")
    parser.add_argument("--out", default="outputs/REPRODUCIBILITY_CHECK.json")
    args = parser.parse_args()

    experiments = EXPERIMENTS
    if args.only:
        experiments = [e for e in experiments if e[0] == args.only]
    elif args.quick:
        experiments = experiments[:2]

    print("=" * 68)
    print("PROVENANCE")
    print("=" * 68)
    fp = full_fingerprint()
    code = fp["code"]
    if not code["versioned"]:
        print("  code: NOT UNDER VERSION CONTROL -- results are not reproducible")
    else:
        print(
            f"  code: {code['commit'][:12]} on {code['branch']}"
            f"{'  *** DIRTY TREE ***' if code['dirty'] else ' (clean)'}"
        )
        if code["dirty"]:
            for f in code["dirty_files"][:10]:
                print(f"        modified: {f}")
    print(f"  python: {fp['env']['python']} ({fp['env']['implementation']})")
    for pkg, ver in fp["env"]["packages"].items():
        print(f"  {pkg}: {ver}")

    print()
    print("=" * 68)
    print("DETERMINISM (same config + seed, run twice in-process)")
    print("=" * 68)
    results = []
    for name, module_name, config_path in experiments:
        print(f"  {name} ... ", end="", flush=True)
        r = _check_one(name, module_name, config_path)
        results.append(r)
        if r["status"] == "PASS":
            print(f"PASS  digest={r['digest'][:16]}")
        elif r["status"] == "SKIP":
            print(f"SKIP  ({r['reason'].splitlines()[0][:80]})")
        elif r["status"] == "FAIL":
            print(f"FAIL  {r['digest'][:16]} != {r['digest_rerun'][:16]}")
        else:
            print("ERROR")
            print("      " + r["reason"].replace("\n", "\n      "))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"provenance": fp, "determinism": results}, f, indent=2)

    n_pass = sum(r["status"] == "PASS" for r in results)
    n_fail = sum(r["status"] in ("FAIL", "ERROR") for r in results)
    n_skip = sum(r["status"] == "SKIP" for r in results)
    print()
    print(f"{n_pass} pass, {n_fail} fail/error, {n_skip} skipped -> {out}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
