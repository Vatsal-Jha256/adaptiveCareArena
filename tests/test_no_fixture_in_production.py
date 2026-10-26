"""Production code must not be able to reach a test fixture.

Guards against a known failure mode: a synthetic data generator lived beside the real dataset
adapters, registered itself by name, and was therefore selectable from
any experiment config. Code written against its vocabulary then ran
against real data and silently produced dead feature columns, while
every synthetic test stayed green.

The rule that prevents it is structural, not a matter of care: fixtures
live under tests/, nothing in a shipped package imports them, and no
config can name them.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PRODUCTION_DIRS = ("caredata", "emaselect", "stage2_env", "models", "eval", "runtime", "scripts")


def _production_files() -> list[Path]:
    out = []
    for d in PRODUCTION_DIRS:
        root = REPO / d
        if root.exists():
            out += [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]
    return out


def test_no_production_module_imports_a_test_fixture():
    offenders = []
    for path in _production_files():
        for node in ast.walk(ast.parse(path.read_text())):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            for m in mods:
                if m == "tests" or m.startswith("tests."):
                    offenders.append(f"{path.relative_to(REPO)} imports {m}")
    assert not offenders, "production code importing test fixtures:\n  " + "\n  ".join(offenders)


def test_no_config_targets_fixture_data():
    import yaml

    bad = []
    for path in REPO.rglob("configs/**/*.yaml"):
        if ".venv" in path.parts:
            continue
        try:
            cfg = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError:
            continue
        blob = str(cfg).lower()
        name = str((cfg.get("dataset") or {}).get("name", "")).lower()
        if "fixture" in name or "toy" in name or "synthetic" in name:
            bad.append(f"{path.relative_to(REPO)}: dataset.name={name!r}")
        elif "fixture" in blob and "toy" in blob:
            bad.append(f"{path.relative_to(REPO)}: references fixture data")
    assert not bad, (
        "experiment config targets fixture data -- its numbers are not "
        "results:\n  " + "\n  ".join(bad)
    )
