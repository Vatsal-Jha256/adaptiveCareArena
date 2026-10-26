"""Reference command-line evaluation through the public API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adaptivecarearena import AdaptiveCareArena, ExponentialMovingAverage, Persistence


def main() -> None:
    """Evaluate the reference estimators on one config and print aggregate JSON."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate reference estimators on a local retrospective cohort; "
            "this is not clinical or intervention validation."
        )
    )
    parser.add_argument("--config", required=True, help="committed YAML config")
    parser.add_argument(
        "--output",
        type=Path,
        help="optional aggregate JSON destination; participant rows are never written",
    )
    args = parser.parse_args()
    result = AdaptiveCareArena(args.config).evaluate(
        {
            "persistence": Persistence(),
            "ema_03": ExponentialMovingAverage(alpha=0.3),
        }
    )
    rendered = json.dumps(result.to_dict(), indent=2, sort_keys=True, allow_nan=False)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
