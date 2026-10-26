"""Run manifest + config loading for adaptiveCareArena.

Without a config hash, seed record and immutability guard, a re-run would
overwrite the previous result and two configs sharing an experiment_id
would be indistinguishable afterwards. Warns when the tree is dirty.

Contract: one experiment = one committed YAML config +
one seed, run dirs are immutable, and every run records the code, env
and data state that produced it."""

from __future__ import annotations

import hashlib
import json
import resource
from pathlib import Path

import yaml

from runtime.reproducibility import full_fingerprint, results_digest


def load_config(path: str) -> dict:
    """Load a YAML config; it must contain an ``experiment_id``."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "experiment_id" not in cfg:
        raise ValueError(f"config {path} missing required 'experiment_id'")
    return cfg


def config_hash(cfg: dict) -> str:
    """Return a 16-hex-digit hash of the config, ignoring keys that start with ``_``."""
    scrubbed = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return hashlib.sha256(json.dumps(scrubbed, sort_keys=True).encode()).hexdigest()[:16]


def make_run_dir(cfg: dict, base: str = "outputs") -> Path:
    """Create ``<base>/<experiment_id>_<config_hash>`` and return it.

    Raises:
        FileExistsError: if the directory exists. Outputs are immutable; change
            the config to start a new run.
    """
    run_dir = Path(base) / f"{cfg['experiment_id']}_{config_hash(cfg)}"
    if run_dir.exists():
        raise FileExistsError(
            f"run dir {run_dir} already exists -- outputs are immutable; "
            "change the config or experiment_id to start a new run"
        )
    run_dir.mkdir(parents=True)
    return run_dir


def write_manifest(run_dir: Path, cfg: dict, data_roots: dict | None = None) -> dict:
    """Manifest + frozen config + full provenance. Warns at run
    time on an unversioned or dirty tree: results produced from
    uncommitted edits are not reproducible, and that has to surface when
    it happens rather than at write-up time."""
    fp = full_fingerprint(data_roots=data_roots or {})
    manifest = {
        "experiment_id": cfg["experiment_id"],
        "config_path": str(cfg.get("_source_path", "")),
        "config_hash": config_hash(cfg),
        "seed": cfg.get("seed"),
        "provenance": fp,
    }
    with open(run_dir / "MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=2)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")}, f)

    code = fp["code"]
    if not code["versioned"]:
        print(
            f"WARNING [{run_dir.name}]: repo is not under version control -- "
            "this run's code state is unrecoverable"
        )
    elif code["dirty"]:
        print(
            f"WARNING [{run_dir.name}]: working tree DIRTY at commit {code['commit'][:8]} -- "
            f"{len(code['dirty_files'])} modified file(s); results not reproducible from that commit"
        )
    return manifest


def write_results(
    run_dir: Path,
    results: dict,
    filename: str = "metrics.json",
    *,
    wall_seconds: float | None = None,
) -> str:
    """Writes results and records their digest back into manifest.json, so
    a re-run can be checked for bit-identity rather than eyeballed."""
    with open(run_dir / filename, "w") as f:
        json.dump(results, f, indent=2)

    digest = results_digest(results)
    manifest_path = run_dir / "MANIFEST.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest.setdefault("results_digests", {})[filename] = digest
        if wall_seconds is not None:
            manifest["resources"] = {
                "wall_seconds": float(wall_seconds),
                "peak_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                "measurement_scope": "current runner process",
            }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
    return digest
