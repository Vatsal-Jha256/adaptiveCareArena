#!/usr/bin/env bash
# Install, test, and (if real data is present) run the data audit.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m pip install -e ".[dev]" -q
python -m pytest -q

if [ -f "data/raw/crosscheck/CrossCheck_Daily_Data.csv" ]; then
  mkdir -p outputs
  python -m p0_audit.run --local-root data/raw/crosscheck --out outputs/p0_audit_report.json
  echo "reproduce.sh: data audit complete; see outputs/p0_audit_report.json"
else
  echo "reproduce.sh: tests passed; real CrossCheck data not found at data/raw/crosscheck/ -- place it there to run Stage 0"
fi
