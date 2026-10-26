# Reproducibility

This page covers how to regenerate every result from the raw data, and what the
repository does so that a rerun can be trusted.

## 1. Set up

```bash
git clone https://github.com/Vatsal-Jha256/adaptiveCareArena.git
cd adaptiveCareArena
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,figures]"
```

Python 3.11 or newer. `uv.lock` records the dependency set used for development. Results depend on the numpy, scikit-learn and pandas
versions, so each run records them (see section 4).

Download the data as described in [DATA.md](DATA.md) and confirm it with the
audit commands there.

## 2. Run everything

```bash
scripts/reproduce_all.sh            # full settings
scripts/reproduce_all.sh --quick    # tiny settings; checks the pipeline end to end
scripts/reproduce_all.sh --from 3   # resume at stage 3
SAFE=1 scripts/reproduce_all.sh     # cap threads and memory (scripts/safe_run.sh)
```

`--quick` uses 20 bootstrap replicates and 3 seeds (the paired analysis needs at least 3). Its numbers are not the
reported ones; use it to confirm your environment and data work before starting
a long run. Everything is written to `outputs/` (gitignored).

| Stage | What it runs | Main outputs |
|---|---|---|
| 1. Data audit | EMA density, schema and leakage assertions, input fingerprints | `outputs/p0_audit_report.json`, `outputs/audit_*.json` |
| 2. Review axis | Participant bootstrap of every review policy against two cyclic baselines (`round_robin`, and `round_robin_shuffled` which ignores id order), Kalman noise selection, threshold sensitivity | `outputs/stage2_bootstrap_{crosscheck,ces}{,_rrshuf}.json`, `outputs/kalman_tune_*.json` |
| 3. Measurement axis | Item selection over seeds, paired and equivalence analysis, oracle headroom | `outputs/emaselect_{crosscheck,ces}_analysis.json`, `outputs/emaselect_oracle_*.json` |
| 4. Benchmark configs | Forward review, learned rung, measurement-to-review composition, discrimination diagnostics, for each config in `configs/benchmark/` | one run directory per config under `outputs/runs/<timestamp>/` (new for every invocation, because run directories are immutable) |
| 5. Tables and figures | LaTeX tables and PDF figures built from the JSON above | `outputs/tables/`, `outputs/figures/` |
| 6. Verification | Determinism check on real data, then the full test suite including real-data tests | `outputs/REPRODUCIBILITY_CHECK.json` |

Full settings are 2,000 bootstrap replicates per dataset and baseline, 20
item-selection seeds for CrossCheck and 35 for CES. The bootstrap is the slow
part. Its shards can run in parallel with different `--seed` values and be
combined with `scripts/merge_bootstrap.py`.

Each script also runs on its own; `python -m scripts.<name> --help` lists its
arguments.

## 3. Check the output

- Runners that write run directories (the benchmark configs and the replay and
  item-selection experiments) name them `<experiment_id>_<config_hash>` and
  treat an existing one as an error, so a rerun never silently overwrites an
  earlier result. Delete the directory, or change the config, to rerun. The
  bootstrap and analysis scripts write fixed file names such as
  `outputs/stage2_bootstrap_ces.json`, which a rerun replaces.
- `python -m scripts.verify_reproducibility` runs each registered experiment
  twice in one process from the same config and seed and compares a digest of
  the results. It reports `PASS`, `FAIL`, `SKIP` or `ERROR` per experiment. A
  skip (for example when the data are absent) is never counted as a pass.
- Floats are rounded to 12 significant digits before hashing. That is far
  tighter than any reported number and far looser than last-bit differences
  across BLAS builds, so the check does not raise permanent false alarms.

## 4. What a run records

One experiment is one committed YAML config plus one seed. Each run writes a
`MANIFEST.json` with three separate fingerprints, because they fail
independently (`runtime/reproducibility.py`):

- **code**: git commit and a dirty flag. The dirty flag matters more than the
  commit: a clean hash printed beside results produced from uncommitted edits
  looks authoritative and is not. Runs from a dirty tree print a warning.
- **environment**: Python version, platform, and installed versions of numpy,
  scipy, scikit-learn, pandas and pyyaml.
- **data**: SHA-256 of every input file plus one combined digest over
  (relative path, hash) pairs, which changes if any input byte changes.

To reproduce a specific number, open the run's `MANIFEST.json`, confirm
`provenance.code.dirty` is `false`, check out `provenance.code.commit`, install
the recorded versions, confirm the data digest against yours, and rerun the
config.

## 5. Tests

Model-level reproducibility is covered by `tests/test_model_reproducibility.py`
for every fitted model, online estimator and stochastic policy:

- two instances with the same seed, fitted on the same data, give bit-identical
  predictions;
- a different seed changes the output for components that use randomness, so a
  component that ignores its seed cannot pass;
- refitting or predicting twice does not change the answer;
- fitting does not mutate the training data.

The default `pytest` run uses synthetic fixtures and skips tests that need the
datasets. `pytest -m real_data` runs the adapter checks against the real files.
Synthetic fixtures show that the code path is deterministic; they are not
evidence about a cohort.

## 6. Running safely on a shared machine

`scripts/safe_run.sh -t 4 -m 6 -- <command>` pins every parallel backend (OpenMP,
OpenBLAS, MKL, numexpr) to a thread count, caps address space so a runaway job
dies instead of the machine, runs at low priority, and refuses to start when free
memory is already low. scikit-learn's histogram gradient boosting otherwise uses
every core, so a few "single" processes can starve a desktop.

## Known limits

- Exact agreement requires the same dependency versions. Different numpy,
  scikit-learn or pandas releases can move numbers in the last digits, and
  scikit-learn has changed `HistGradientBoosting` defaults across minor versions.
- The reported results were produced from a development tree. Rerunning from
  a clean checkout of this repository gives a clean provenance record; compare
  your outputs to the reported ones rather than expecting a byte match.

## Names in the write-up and in the code

Two review policies are displayed under different names in tables and figures
than in the registry and the output JSON files:

| Displayed as | Code and JSON name |
|---|---|
| `overdue_floor` | `fair_floor` |
| `need_recency` | `whittle_index` |

The mapping lives in `scripts/make_tables.py` and `scripts/make_figures.py`.
All other policy names are identical.
