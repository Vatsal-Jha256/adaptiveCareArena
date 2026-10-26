"""Prints headline numbers for the EMASelect measurement analysis, plus a
compact per-contrast summary table (LaTeX) written to
outputs/tables/emaselect_summary.tex.

python -m scripts.emaselect_summary
"""

from __future__ import annotations

import json
from pathlib import Path

TABLE_DIR = Path("outputs/tables")

CONTRAST_LABEL = {
    "ask_all_vs_fixed_mae": "ask\\_all vs fixed\\_subset (raw MAE)",
    "ask_all_vs_fixed_gap": "ask\\_all vs fixed\\_subset (frontier\\_gap)",
    "thompson_vs_fixed_gap": "thompson\\_items vs fixed\\_subset (frontier\\_gap)",
    "greedy_vs_fixed_gap": "greedy\\_voi vs fixed\\_subset (frontier\\_gap)",
}


def summarize(fam: dict) -> dict:
    comps = fam.get("comparisons", {})
    tost = fam.get("tost", {})
    n = len(comps)
    n_holm = sum(1 for v in comps.values() if v.get("corrected_significant"))
    n_real = sum(1 for v in tost.values() if v["different"] and not v["equivalent"])
    n_triv = sum(1 for v in tost.values() if v["equivalent"] and v["different"])
    n_eq = sum(1 for v in tost.values() if v["equivalent"] and not v["different"])
    n_inc = sum(1 for v in tost.values() if "INCONCLUSIVE" in v["outcome"])
    n_dead = len(fam.get("degenerate_cells", []))
    return {
        "n_cells": n,
        "n_holm": n_holm,
        "n_real": n_real,
        "n_triv": n_triv,
        "n_eq": n_eq,
        "n_inc": n_inc,
        "n_dead": n_dead,
    }


def main() -> None:
    cc = json.loads(Path("outputs/emaselect_crosscheck_analysis.json").read_text())
    ces = json.loads(Path("outputs/emaselect_ces_analysis.json").read_text())

    print(f"CrossCheck seeds: {cc['n_seeds']}   CES seeds: {ces['n_seeds']}")
    print()
    for label, data in (("CrossCheck", cc), ("CES", ces)):
        print(f"=== {label} ===")
        for key, fam in data["contrasts"].items():
            s = summarize(fam)
            print(
                f"  {key:24s} cells={s['n_cells']:2d} holm_sig={s['n_holm']:2d} "
                f"dead={s['n_dead']:2d} | TOST real={s['n_real']} triv={s['n_triv']} "
                f"eq={s['n_eq']} inc={s['n_inc']}"
            )
        print()

    # Appendix table
    lines = [
        r"Holm: cells significant after Holm correction. TOST columns "
        r"(Diff./Trivial/Equiv./Inconcl.) partition all cells by outcome: "
        r"materially different, trivially different, equivalent, inconclusive.",
        "",
        r"\begin{table}[h]",
        r"\centering",
        r"\footnotesize",
        r"\caption{EMASelect per-contrast summary, both cohorts (margin $\pm0.10$ MAE).}",
        r"\label{tab:emaselect_full}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Dataset & Contrast & Cells & Holm & Diff. & Trivial & Equiv. & Inconcl. \\",
        r"\midrule",
    ]
    for label, data in (("CrossCheck", cc), ("CES", ces)):
        for i, (key, fam) in enumerate(data["contrasts"].items()):
            s = summarize(fam)
            row = [
                label if i == 0 else "",
                CONTRAST_LABEL[key],
                str(s["n_cells"]),
                str(s["n_holm"]),
                str(s["n_real"]),
                str(s["n_triv"]),
                str(s["n_eq"]),
                str(s["n_inc"]),
            ]
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\addlinespace")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")  # close resizebox
    lines.append(r"\end{table}")
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    (TABLE_DIR / "emaselect_summary.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {TABLE_DIR / 'emaselect_summary.tex'}")


if __name__ == "__main__":
    main()
