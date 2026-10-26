# Architecture

AdaptiveCareArena separates three layers so that a better patient model and a
better decision rule can be evaluated independently:

```
observations  ->  belief (patient model)  ->  resource-constrained decision
 (recorded)        (estimator, M)               (policy P under budget R)
```

A retrospective replay never changes what is recorded. Actions choose which
recorded report is read or which case is reviewed; they do not change later
states, so nothing here estimates an intervention effect.

## Environments

| Environment | Decision | Code |
|---|---|---|
| Review | Which patients a human reviews each round, under a per-round budget | `stage2_env/` |
| Forward review | Same, scored on flagging an onset before it is reported | `scripts/forward_benchmark.py`, `adaptivecarearena/arena.py` |
| Measurement | Which questionnaire items to ask under a burden budget | `emaselect/`, `adaptivecarearena/measurement.py` |

## Packages

| Package | Role |
|---|---|
| `adaptivecarearena/` | Public API: `AdaptiveCareArena`, estimator and policy protocols, reference estimators, the recorded-questionnaire environment, a data audit CLI |
| `caredata/` | Dataset adapters. Each one builds examples that satisfy the leakage invariant |
| `task/` | `CareExample` and the leakage assertion |
| `splits/` | Participant-aware splits: leave-subject-out, k-fold by subject, future-time |
| `models/` | Supervised forecasters: naive baselines, ridge, gradient boosting, persistence-anchored residual models |
| `stage2_env/` | Replay world, estimators (EMA, Kalman), review policies, metrics |
| `emaselect/` | Mask-aware item-selection model, item policies, conformal intervals |
| `eval/` | Participant-level bootstrap, paired seed comparison, equivalence tests |
| `runtime/` | Run manifests, config hashing, code and data fingerprints |
| `scripts/` | Analyses, sweeps, and figure and table generation |
| `tests/` | Unit tests; `tests/fixtures/` holds synthetic generators and is not importable from shipped code |

## Plugging in your own method

An estimator implements `reset()`, `observe(subject_id, observed_at, value)`
and `score(subject_id, prediction_time)` (larger means greater need). A review
policy turns that score into a priority. The evaluator owns eligibility,
targets, queues and metrics, so a submitted method cannot read outcome labels.
See `adaptivecarearena/interfaces.py`.

## Invariants the tests enforce

- **Leakage.** Every feature timestamp is at or before the prediction time; a
  target is observed at or after prediction time plus horizon. EMA values are
  never forward-filled into the target (`task/`, `tests/test_splits_leakage.py`).
- **No fixture in production.** Synthetic generators live under `tests/` and
  cannot be named from a config (`tests/test_no_fixture_in_production.py`).
- **Naming.** `experiment_id` equals the config filename stem, and no
  `_v1`/`_v2` suffixes (`tests/test_naming_conventions.py`).
- **Reproducibility.** See [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
