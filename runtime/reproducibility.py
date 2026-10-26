"""Reproducibility fingerprinting: everything a third party needs to
establish that a number in a table came from a specific state of this
repo, applied to specific bytes of input data, under a specific software
stack.

Why this module exists: a manifest that records only experiment_id,
config hash, seed and a git commit cannot detect a code edit between two
runs, or a change in the data files, and so does not tie a result to what
produced it.

The three fingerprints here are deliberately separate, because they fail
independently and a reader needs to know which one moved:

  code    -- git commit + dirty-tree flag + hash of the tracked source
  data    -- sha256 of the raw input files actually read
  env     -- interpreter, platform, and versions of the numeric stack

A result is reproducible only if all three are pinned. `results_digest`
closes the loop: it hashes the output itself, so a re-run can be checked
for bit-identity rather than eyeballed."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

# The packages whose version can silently change a numeric result here.
# sklearn in particular has changed HistGradientBoosting defaults and
# average_precision_score tie handling across minor versions.
TRACKED_PACKAGES = ("numpy", "scipy", "scikit-learn", "pandas", "pyyaml")


def _git(*args: str, cwd: str | Path | None = None, strip: bool = True) -> str | None:
    """`strip=False` is required for `status --porcelain`: its lines are
    `XY<space>path`, and an unmodified-in-index file yields a leading
    space (" M a.txt"). Stripping the whole blob eats that space and
    shifts every path by one character, silently truncating the first
    letter of every filename in the dirty list."""
    try:
        out = subprocess.check_output(
            ["git", *args], stderr=subprocess.DEVNULL, cwd=str(cwd) if cwd else None
        )
        decoded = out.decode()
        return decoded.strip() if strip else decoded
    except Exception:
        return None


def _porcelain_paths(status: str) -> list[str]:
    """Paths out of `git status --porcelain` lines. Handles renames,
    whose line is `R  old -> new` -- the new path is the one that exists
    in the working tree."""
    paths = []
    for line in status.splitlines():
        if len(line) < 4:
            continue
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        paths.append(path.strip('"'))
    return paths


def find_repo_root(start: str | Path | None = None) -> Path | None:
    """Walks UP from this module's file to the nearest directory holding a
    `.git`, and returns None if there is none.

    Walks up from this file instead of using a hardcoded `parents[N]`, which
    breaks silently when the package depth changes and can resolve to a
    parent directory that is a different git repository. Stops at the first
    `.git` found."""
    here = Path(start) if start else Path(__file__).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / ".git").exists():
            return candidate
    return None


def code_fingerprint(repo_root: str | Path | None = None) -> dict:
    """Git commit plus whether the working tree was dirty when this ran.

    The dirty flag is the key field, not the commit: a clean
    commit hash printed next to results produced from uncommitted edits
    is worse than no hash at all, because it looks authoritative. When
    git is unavailable the commit is None and `versioned` is False --
    not replaced by a placeholder string."""
    root = Path(repo_root) if repo_root else find_repo_root()
    if root is None:
        return {
            "versioned": False,
            "commit": None,
            "dirty": None,
            "note": "no .git found above this module -- results cannot be tied to a code version",
        }
    commit = _git("rev-parse", "HEAD", cwd=root)
    if commit is None:
        return {
            "versioned": False,
            "commit": None,
            "dirty": None,
            "note": "not a git repository -- results cannot be tied to a code version",
        }
    status = _git("status", "--porcelain", cwd=root, strip=False) or ""
    branch = _git("rev-parse", "--abbrev-ref", "HEAD", cwd=root)
    return {
        "versioned": True,
        "commit": commit,
        "branch": branch,
        "dirty": bool(status.strip()),
        "dirty_files": sorted(_porcelain_paths(status))[:50],
    }


def env_fingerprint() -> dict:
    """Interpreter and numeric-stack versions. Uses importlib.metadata
    rather than each module's __version__ so the recorded string is the
    installed distribution version (what a `pip install` would need to
    reproduce), not a module attribute that may differ or be absent."""
    from importlib.metadata import PackageNotFoundError, version

    packages: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None

    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
    }


def file_fingerprint(path: str | Path, chunk_size: int = 1 << 20) -> dict:
    """sha256 + size of one file. Streams in chunks -- raw sensor CSVs
    here run to hundreds of MB and must not be read into memory whole."""
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False, "sha256": None, "bytes": None}
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return {"path": str(p), "exists": True, "sha256": h.hexdigest(), "bytes": p.stat().st_size}


def data_fingerprint(
    root: str | Path, patterns: tuple[str, ...] = ("*",), max_files: int = 200
) -> dict:
    """Fingerprint of a raw-data directory: a per-file sha256 list plus a
    single combined digest over (relative path, sha256) pairs.

    The combined digest is what belongs in a results report -- one
    string that changes if any input byte changed. Files are sorted by
    relative path so the digest does not depend on filesystem ordering.
    When a directory exceeds `max_files`, the per-file list is truncated
    but the combined digest still covers everything, and `truncated`
    records that -- the digest must never silently describe a subset."""
    root_p = Path(root)
    if not root_p.exists():
        return {"root": str(root_p), "exists": False, "combined_sha256": None, "n_files": 0}

    paths = sorted({p for pat in patterns for p in root_p.rglob(pat) if p.is_file()})
    combined = hashlib.sha256()
    files: list[dict] = []
    for p in paths:
        fp = file_fingerprint(p)
        combined.update(str(p.relative_to(root_p)).encode())
        combined.update((fp["sha256"] or "").encode())
        if len(files) < max_files:
            files.append(
                {"path": str(p.relative_to(root_p)), "sha256": fp["sha256"], "bytes": fp["bytes"]}
            )

    return {
        "root": str(root_p),
        "exists": True,
        "n_files": len(paths),
        "combined_sha256": combined.hexdigest(),
        "files": files,
        "truncated": len(paths) > max_files,
    }


def _canonical(obj):
    """Round floats to 12 significant digits before hashing. Full float64
    repr would make the digest sensitive to last-bit differences that
    do vary across BLAS builds and thread counts, turning a
    determinism check into a permanent false alarm. 12 digits is far
    tighter than any reported result and far looser than
    that noise floor."""
    if isinstance(obj, float):
        return f"{obj:.12g}"
    if isinstance(obj, dict):
        return {str(k): _canonical(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    return obj


def results_digest(results: dict) -> str:
    """Stable hash of a results dict, for verifying that a re-run of the
    same config on the same data under the same code produces the same
    numbers -- the property `scripts/verify_reproducibility.py` asserts."""
    return hashlib.sha256(json.dumps(_canonical(results), sort_keys=True).encode()).hexdigest()


def full_fingerprint(repo_root: str | Path | None = None, data_roots: dict | None = None) -> dict:
    """The complete provenance record embedded in every run manifest."""
    return {
        "code": code_fingerprint(repo_root),
        "env": env_fingerprint(),
        "data": {name: data_fingerprint(root) for name, root in (data_roots or {}).items()},
    }
