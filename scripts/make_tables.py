"""Generates LaTeX tables from the bootstrap sweep JSONs:
outputs/tables/results_summary.tex (budget=5 only) and
outputs/tables/results_full.tex (full sweep, both cohorts). Run after scripts/stage2_bootstrap_sweep.py has produced
outputs/stage2_bootstrap_{crosscheck,ces}.json for both datasets.

python -m scripts.make_tables
"""

from __future__ import annotations

import json
from pathlib import Path

TABLE_DIR = Path("outputs/tables")

METRIC_LABEL = {
    "missed_deterioration_days": r"$\Delta$ missed det.",
    "allocation_max_gap": r"$\Delta$ attn.\ disparity",
    "mean_attention_lag_at_risk": r"$\Delta$ attn.\ lag",
}
METRICS = list(METRIC_LABEL)
POLICIES = ["random", "fair_floor", "greedy_need", "ucb_explore", "whittle_index"]
DATASET_LABEL = {"crosscheck": "CrossCheck", "ces": "CES"}
# Display names only; JSON keys unchanged. See make_figures.py.
POLICY_DISPLAY = {"fair_floor": "overdue_floor", "whittle_index": "need_recency"}


def display(pol: str) -> str:
    return POLICY_DISPLAY.get(pol, pol)


def load(dataset: str) -> dict:
    return json.loads(Path(f"outputs/stage2_bootstrap_{dataset}.json").read_text())


def cell(d: dict, metric: str, policy: str, budget: int) -> dict:
    return d["by_metric"][metric][f"{policy} @ b={budget}"]


def fmt(v: dict, brief: bool = False) -> str:
    star = r"$^{*}$" if v["excludes_zero"] else ""
    if brief:
        return f"{v['median_diff']:+.2f}{star}"
    return f"{v['median_diff']:+.2f} [{v['ci_lo']:+.2f}, {v['ci_hi']:+.2f}]{star}"


def main() -> None:
    data = {ds: load(ds) for ds in ("crosscheck", "ces")}

    # --- Main-text table: budget=5, both datasets, brief cells ---
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Median bootstrap diff.\ vs.\ \textsc{round\_robin}, $B=5$ "
        r"(2{,}000 replicates). $^{*}$excludes 0 at 99\% CI. Full sweep: "
        r"Appendix~\ref{app:full}.}"
    )
    lines.append(r"\label{tab:main}")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{0.85}")
    lines.append(r"\begin{tabular}{llrrr}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & Policy & " + " & ".join(METRIC_LABEL[m] for m in METRICS) + r" \\")
    lines.append(r"\midrule")
    for ds in ("crosscheck", "ces"):
        for i, pol in enumerate(POLICIES):
            row = [DATASET_LABEL[ds] if i == 0 else "", display(pol).upper().replace("_", r"\_")]
            for m in METRICS:
                row.append(fmt(cell(data[ds], m, pol, 5), brief=True))
            lines.append(" & ".join(row) + r" \\")
        if ds == "crosscheck":
            lines.append(r"\addlinespace")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    (TABLE_DIR / "results_summary.tex").write_text("\n".join(lines) + "\n")

    # --- Appendix table: full sweep, both datasets, budgets, with CIs ---
    lines = []
    lines.append(
        r"Every (dataset, policy, budget) cell, all three metrics, "
        r"2{,}000 participant-resampling replicates. $^{*}$ marks a 99\% "
        r"per-budget-Bonferroni CI that excludes zero."
    )
    lines.append("")
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Full budget sweep, both cohorts: median bootstrap "
        r"difference vs.\ \textsc{round\_robin} [99\% CI].}"
    )
    lines.append(r"\label{tab:full}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{lllrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"Dataset & Policy & $B$ & " + " & ".join(METRIC_LABEL[m] for m in METRICS) + r" \\"
    )
    lines.append(r"\midrule")
    for ds in ("crosscheck", "ces"):
        for budget in (3, 5, 10):
            for pol in POLICIES:
                row = [
                    DATASET_LABEL[ds] if (budget == 3 and pol == POLICIES[0]) else "",
                    display(pol).replace("_", r"\_"),
                    str(budget),
                ]
                for m in METRICS:
                    row.append(fmt(cell(data[ds], m, pol, budget)))
                lines.append(" & ".join(row) + r" \\")
            lines.append(r"\addlinespace[1pt]")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")  # close resizebox
    lines.append(r"\end{table}")
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    (TABLE_DIR / "results_full.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote tables to {TABLE_DIR}")


if __name__ == "__main__":
    main()
