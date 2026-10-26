"""Enforces the repo's config/experiment naming convention.

A convention documented only in prose drifts the moment someone adds a
file in a hurry. These tests fail instead, which is the only
reason the convention survives contact with new work.

The rules:
  1. Config filenames and experiment_ids are lowercase snake_case.
  2. `experiment_id` == the config filename stem. A config and its id
     can never drift apart, and the id is derivable from the path
     without opening the file.
  3. No `_v1` / `_v2` version suffix. Run directories are already named
     `<experiment_id>_<config_hash>`, so the config hash IS the version
     -- a hand-written vN adds nothing, and goes stale the moment the
     config changes without anyone bumping it.
  4. No `real`/`full`/`test` qualifiers. Real data is the default,
     "full" is what an unqualified config means, and a config under
     `configs/test/` is already a test. Only genuine variants are
     qualified (`_smoke`, `tiny_`, `synthetic_`).
"""

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
CONFIGS = sorted(REPO.glob("configs/*/*.yaml"))

SNAKE_CASE = re.compile(r"^[a-z0-9]+(_[a-z0-9]+)*$")
VERSION_SUFFIX = re.compile(r"_v\d+$")
BANNED_TOKENS = ("real", "full", "test")


def test_there_are_configs_to_check():
    # Guards the guard: an empty glob would make every test below vacuous.
    assert CONFIGS, "no configs discovered -- the glob or layout changed"


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: str(p.relative_to(REPO)))
def test_config_filename_is_snake_case(path):
    assert SNAKE_CASE.match(path.stem), f"{path.name}: use lowercase snake_case"


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: str(p.relative_to(REPO)))
def test_experiment_id_matches_filename(path):
    cfg = yaml.safe_load(path.read_text())
    expected = path.stem
    assert cfg.get("experiment_id") == expected, (
        f"{path.name}: experiment_id is {cfg.get('experiment_id')!r}, expected {expected!r} "
        "-- the id must be derivable from the config path"
    )


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: str(p.relative_to(REPO)))
def test_no_version_suffix_anywhere(path):
    cfg = yaml.safe_load(path.read_text())
    assert not VERSION_SUFFIX.search(path.stem), (
        f"{path.name}: drop the _vN suffix -- run dirs are named "
        "<experiment_id>_<config_hash>, so the hash already versions the run"
    )
    assert not VERSION_SUFFIX.search(cfg.get("experiment_id", "")), (
        f"{path.name}: experiment_id carries a _vN suffix; the config hash is the version"
    )


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: str(p.relative_to(REPO)))
def test_no_redundant_qualifiers(path):
    tokens = path.stem.split("_")
    offenders = [t for t in tokens if t in BANNED_TOKENS]
    assert not offenders, (
        f"{path.name}: drop {offenders} -- real data is the default and an "
        "unqualified config already means the full run; qualify only genuine "
        "variants (_smoke, _medium, synthetic_)"
    )


def test_experiment_ids_are_unique_across_the_repo():
    seen: dict[str, Path] = {}
    for path in CONFIGS:
        eid = yaml.safe_load(path.read_text()).get("experiment_id")
        assert eid not in seen, f"duplicate experiment_id {eid!r}: {path} and {seen[eid]}"
        seen[eid] = path
