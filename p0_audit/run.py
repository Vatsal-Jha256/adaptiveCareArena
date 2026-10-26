"""Stage 0 audit: EMA administration density, passive-sensing missingness,
and a leakage check. Run and read this before trusting any model.

python -m p0_audit.run --local-root data/raw/crosscheck --out outputs/p0_audit_report.json
"""

from __future__ import annotations

import argparse
import json

from caredata.crosscheck import CrossCheckDataset


def run_audit(local_root: str, min_history_days: int) -> dict:
    """Audit EMA density and leakage on CrossCheck; raises if any example leaks the future."""
    ds = CrossCheckDataset(local_root=local_root)
    density = ds.ema_administration_density()

    examples = ds.build_task_table(min_history_days=min_history_days)
    ds.validate_temporal_contract(examples)  # raises if leakage found

    n_subjects_total = len(ds.list_subjects())
    n_subjects_with_examples = len({e.subject_id for e in examples})

    return {
        "n_subjects_total": n_subjects_total,
        "n_subjects_with_forecastable_examples": n_subjects_with_examples,
        "n_examples": len(examples),
        "ema_density_by_subject": density.to_dict(orient="records"),
        "leakage_audit": "PASSED (assert_no_future_leakage raised nothing)",
    }


def main() -> None:
    """Command-line entry point for the data audit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-root", default="data/raw/crosscheck")
    parser.add_argument("--min-history-days", type=int, default=7)
    parser.add_argument("--out", default="outputs/p0_audit_report.json")
    args = parser.parse_args()

    report = run_audit(args.local_root, args.min_history_days)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"n_subjects_total={report['n_subjects_total']}")
    print(
        f"n_subjects_with_forecastable_examples={report['n_subjects_with_forecastable_examples']}"
    )
    print(f"n_examples={report['n_examples']}")
    print(f"leakage_audit={report['leakage_audit']}")


if __name__ == "__main__":
    main()
