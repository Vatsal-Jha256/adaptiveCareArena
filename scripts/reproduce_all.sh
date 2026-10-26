#!/usr/bin/env bash
# Regenerate every result, table and figure from the raw datasets.
#
#   scripts/reproduce_all.sh            full run (hours; see docs/REPRODUCE.md)
#   scripts/reproduce_all.sh --quick    tiny settings (20 replicates, 3 seeds) to check
#                                       that the pipeline runs; numbers are not the reported ones
#   scripts/reproduce_all.sh --from N   resume at stage N (1-6)
#
# Requires data/raw/crosscheck and data/raw/ces (see docs/DATA.md) and
# `pip install -e ".[dev,figures]"`. Outputs go to outputs/, which is
# gitignored. Set SAFE=1 to run each stage through scripts/safe_run.sh,
# which caps threads and memory.
set -euo pipefail
cd "$(dirname "$0")/.."

NBOOT=2000        # participant-bootstrap replicates per dataset and baseline
SEEDS_CC=20       # item-selection seeds, CrossCheck
SEEDS_CES=35      # item-selection seeds, CES
FROM=1
# Benchmark run directories are immutable, so each invocation gets a fresh one.
RUNS="outputs/runs/$(date -u +%Y%m%dT%H%M%SZ)"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick) NBOOT=20; SEEDS_CC=3; SEEDS_CES=3; shift ;;
    --from)  FROM="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

for d in crosscheck ces; do
  [[ -d "data/raw/$d" ]] || { echo "missing data/raw/$d -- see docs/DATA.md" >&2; exit 1; }
done

run() {
  echo "+ $*"
  if [[ "${SAFE:-0}" == "1" ]]; then scripts/safe_run.sh -- "$@"; else "$@"; fi
}
stage() { echo; echo "=== stage $1: $2"; [[ "$1" -ge "$FROM" ]]; }

mkdir -p outputs "$RUNS"

# 1. Data checks: schema, EMA density, leakage assertions, input fingerprints.
if stage 1 "data audit"; then
  run python -m p0_audit.run --local-root data/raw/crosscheck --out outputs/p0_audit_report.json
  for cfg in forward_crosscheck forward_ces; do
    run adaptivecare-data-audit --config "configs/benchmark/$cfg.yaml" \
      --output "outputs/audit_$cfg.json" > /dev/null
  done
fi

# 2. Review axis: participant bootstrap against two cyclic baselines,
#    threshold sensitivity, and the Kalman noise selection.
if stage 2 "review axis"; then
  for ds in crosscheck ces; do
    run python -m scripts.stage2_bootstrap_sweep --dataset "$ds" --n-boot "$NBOOT"
    run python -m scripts.stage2_bootstrap_sweep --dataset "$ds" --n-boot "$NBOOT" \
      --baseline round_robin_shuffled --out "outputs/stage2_bootstrap_${ds}_rrshuf.json"
    run python -m scripts.kalman_tune_noise --dataset "$ds"
  done
  run python -m scripts.stage2_threshold_sensitivity --dataset crosscheck --thresholds=-3,-5,-7,-9
  run python -m scripts.stage2_threshold_sensitivity --dataset ces --thresholds=-6,-9,-12
fi

# 3. Measurement axis: item selection over seeds, paired analysis, oracle headroom.
if stage 3 "measurement axis"; then
  run python -m emaselect.multi_seed --config configs/dev/emaselect_smoke.yaml \
    --seeds "$SEEDS_CC" --target-mode forecast --ckpt-dir outputs/_emaselect_ckpt \
    --out outputs/emaselect_multi_seed_crosscheck.json
  run python -m emaselect.multi_seed --config configs/dev/emaselect_ces.yaml \
    --seeds "$SEEDS_CES" --target-mode forecast --ckpt-dir outputs/_ces_ckpt \
    --out outputs/emaselect_multi_seed_ces.json
  run python -m scripts.emaselect_cohort_analysis --label crosscheck --margin 0.10 \
    --ckpt-glob "outputs/_emaselect_ckpt/forecast_seed*.json" \
    --out outputs/emaselect_crosscheck_analysis.json
  run python -m scripts.emaselect_cohort_analysis --label ces --margin 0.10 \
    --ckpt-glob "outputs/_ces_ckpt/forecast_seed*.json" \
    --out outputs/emaselect_ces_analysis.json
  run python -m scripts.emaselect_oracle_headroom --config configs/dev/emaselect_smoke.yaml \
    --dataset crosscheck
  run python -m scripts.emaselect_oracle_headroom --config configs/dev/emaselect_ces.yaml \
    --dataset ces
fi

# 4. Benchmark configs: forward review, learned rung, composition, discrimination.
if stage 4 "benchmark configs"; then
  for cfg in configs/benchmark/forward_*.yaml; do
    run python -m scripts.forward_benchmark --config "$cfg" --base-out "$RUNS"
  done
  for cfg in configs/benchmark/learned_forward_*.yaml; do
    run python -m scripts.learned_forward_benchmark --config "$cfg" --base-out "$RUNS"
  done
  for cfg in configs/benchmark/composition_*.yaml; do
    run python -m scripts.composition_benchmark --config "$cfg" --base-out "$RUNS"
  done
  for cfg in configs/benchmark/discrimination_*.yaml; do
    run python -m scripts.discrimination_benchmark --config "$cfg" --base-out "$RUNS"
  done
fi

# 5. Tables and figures from the JSON outputs above.
if stage 5 "tables and figures"; then
  run python -m scripts.make_tables
  run python -m scripts.emaselect_summary
  run python -m scripts.make_figures
fi

# 6. Determinism check on the real data, then the test suite.
if stage 6 "verification"; then
  run python -m scripts.verify_reproducibility
  run python -m pytest -m "real_data or not real_data" -p no:warnings
fi

echo; echo "done. Results are in outputs/ (tables in outputs/tables, figures in outputs/figures)."
