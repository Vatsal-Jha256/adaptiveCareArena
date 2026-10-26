# AdaptiveCareArena

A replay benchmark that scores patient models by the resource-constrained
decisions their beliefs support (what to measure, which case a human
reviews), rather than by predictive fidelity alone. It runs on real
trajectories from two public cohorts, CrossCheck and the College Experience
Study (CES).

## Scope

This is retrospective analysis of de-identified datasets released by their
original research teams. It recruits no one and delivers nothing to
patients. Nothing here is a diagnostic, triage, or treatment tool, and no
result is evidence that a monitoring system works in deployment. The CrossCheck
EMA items include psychiatric symptom content; only aggregates are reported,
and re-identification of any participant is out of scope.

## Data

No data is included. CrossCheck and CES must be obtained from their owners
under their own terms; see [`docs/DATA.md`](docs/DATA.md) for
sources, expected file layout, and licensing notes. Place files under
`data/raw/`, which is gitignored. Do not commit or redistribute participant rows.

## Install

```bash
pip install -e ".[dev]"      # add ",figures" to regenerate figures
pytest                        # synthetic-fixture tests; real-data tests are deselected
pytest -m real_data           # requires the downloaded datasets
```

Python 3.11+.

## Use

Each experiment is one committed YAML config plus a seed; run directories are
named `<experiment_id>_<config_hash>` and are not overwritten.

```bash
adaptivecare-data-audit --config configs/benchmark/forward_ces.yaml
adaptivecare-evaluate   --config configs/benchmark/forward_ces.yaml
```

Evaluating your own state estimator:

```python
from adaptivecarearena import AdaptiveCareArena, ExponentialMovingAverage, Persistence

arena = AdaptiveCareArena("configs/benchmark/forward_ces.yaml")
result = arena.evaluate(
    {
        "persistence": Persistence(),
        "ema_03": ExponentialMovingAverage(alpha=0.3),
    }
)
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for how the pieces fit and
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for what is tested and how
runs are pinned.

## Layout

| Path | Contents |
|---|---|
| `adaptivecarearena/` | public API: arena, reference estimators, data audit |
| `caredata/` | dataset adapters (CrossCheck, CES) |
| `stage2_env/` | replay world, review policies, estimators, metrics |
| `emaselect/` | questionnaire item-selection (measurement) environment |
| `eval/` | participant-level bootstrap, paired comparison, equivalence tests |
| `scripts/` | analyses, figure and table generation |
| `configs/` | experiment configs |
| `tests/` | unit tests, including leakage and naming invariants |

## Evaluation conventions

- A feature's timestamp is at or before the prediction time, and a target is
  observed at or after prediction time plus horizon. EMA values are never
  forward-filled into the target.
- Splits are by participant and respect time. Random within-participant
  splits are diagnostic only.
- Policy comparisons that share a seed are paired. A non-significant
  difference is reported as inconclusive unless an equivalence test against a
  pre-specified margin supports equivalence.

## License

Code: MIT (see `LICENSE`). The datasets keep their own licenses. CES is
CC BY-NC-SA 4.0 and CrossCheck has no stated reuse license.
