"""Generates the figures directly from the real result JSONs (no
hand-typed numbers). Run after stage2_bootstrap_sweep.py and
emaselect_cohort_analysis.py have produced their outputs.

python -m scripts.make_figures

Styling lives in scripts/figstyle.py (Origin-like boxed axes, black-edged
fills, fixed physical width so nothing is rescaled).
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

from scripts import figstyle as fs
from scripts.figstyle import BAND, CC, CES, GREY, INK

fs.apply()
OUT = Path("outputs/figures")
OUT.mkdir(parents=True, exist_ok=True)

POLICIES = ["random", "fair_floor", "greedy_need", "ucb_explore", "whittle_index"]
# Display names only; JSON keys come from stage2_env's registry.
POLICY_LABEL = {
    "random": "random",
    "fair_floor": "overdue_floor",
    "greedy_need": "greedy_need",
    "ucb_explore": "ucb_explore",
    "whittle_index": "need_recency",
}
METRICS = [
    ("missed_deterioration_days", r"$\Delta$ missed deterioration (days)"),
    ("allocation_max_gap", r"$\Delta$ attention disparity"),
    ("mean_attention_lag_at_risk", r"$\Delta$ attention lag (rounds)"),
]
PROFILE = {
    "uniform": "uniform",
    "sensitivity_weighted": "sens.-weighted",
    "inverted_sensitivity": "inverted",
}


def _save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def fig_review_forest(budget: int = 5, suffix: str = "_rrshuf") -> None:
    cc = json.loads(Path(f"outputs/stage2_bootstrap_crosscheck{suffix}.json").read_text())
    ces = json.loads(Path(f"outputs/stage2_bootstrap_ces{suffix}.json").read_text())

    fig, axes = plt.subplots(1, 3, figsize=(fs.TEXT_W, 2.2), sharey=True, layout="constrained")
    y_pos = {p: i for i, p in enumerate(POLICIES)}
    for k, (ax, (metric, label)) in enumerate(zip(axes, METRICS, strict=False)):
        for data, color, off, marker in ((cc, CC, -0.17, "o"), (ces, CES, 0.17, "s")):
            cells = data["by_metric"][metric]
            for pname in POLICIES:
                v = cells.get(f"{pname} @ b={budget}")
                if v is None:
                    continue
                y = y_pos[pname] + off
                ax.plot(
                    [v["ci_lo"], v["ci_hi"]],
                    [y, y],
                    color=color,
                    lw=2.0,
                    solid_capstyle="butt",
                    zorder=2,
                )
                for e in (v["ci_lo"], v["ci_hi"]):
                    ax.plot([e, e], [y - 0.07, y + 0.07], color=color, lw=1.2, zorder=2)
                ax.scatter(
                    [v["median_diff"]],
                    [y],
                    s=34,
                    marker=marker,
                    zorder=3,
                    facecolor=color if v["excludes_zero"] else "white",
                    edgecolor=INK,
                    linewidth=0.9,
                )
        ax.axvline(0, color=INK, lw=0.9, ls=(0, (4, 2)), zorder=1)
        ax.set_yticks(list(y_pos.values()))
        ax.set_yticklabels([POLICY_LABEL[p] for p in POLICIES])
        ax.set_ylim(len(POLICIES) - 0.4, -0.6)
        ax.set_xlabel(label)
        ax.grid(axis="x")
        ax.tick_params(left=(k == 0), labelleft=(k == 0))
        fs.panel_label(ax, "abc"[k])
    handles = [
        Line2D([], [], marker="o", color=CC, ls="", mec=INK, ms=6, label="CrossCheck (64 subj.)"),
        Line2D([], [], marker="s", color=CES, ls="", mec=INK, ms=6, label="CES (210 subj.)"),
        Line2D(
            [], [], marker="o", color="white", ls="", mec=INK, ms=6, label="open: 99% CI includes 0"
        ),
    ]
    fig.legend(
        handles=handles, loc="outside lower center", ncol=3, handletextpad=0.3, columnspacing=1.6
    )
    _save(fig, "fig_review_forest")


def fig_emaselect_margin() -> None:
    cc = json.loads(Path("outputs/emaselect_crosscheck_analysis.json").read_text())
    ces = json.loads(Path("outputs/emaselect_ces_analysis.json").read_text())
    margin = cc["equivalence_margin"]

    fig, axes = plt.subplots(1, 2, figsize=(fs.TEXT_W, 2.45), sharex=True, layout="constrained")
    for k, (ax, (title, data, color, marker)) in enumerate(
        zip(
            axes,
            (("CrossCheck (20 seeds)", cc, CC, "o"), ("CES (35 seeds)", ces, CES, "s")),
            strict=False,
        )
    ):
        comps = data["contrasts"]["thompson_vs_fixed_gap"]["comparisons"]
        cells = sorted(comps, key=lambda c: comps[c]["mean_diff"])
        ax.axvspan(-margin, margin, color=BAND, zorder=0)
        ax.axvline(0, color=INK, lw=0.9, ls=(0, (4, 2)), zorder=1)
        for i, c in enumerate(cells):
            v = comps[c]
            ax.plot(
                [v["ci_lo"], v["ci_hi"]],
                [i, i],
                color=color,
                lw=2.0,
                solid_capstyle="butt",
                zorder=2,
            )
            ax.scatter(
                [v["mean_diff"]],
                [i],
                s=34,
                marker=marker,
                zorder=3,
                facecolor=color if v["corrected_significant"] else "white",
                edgecolor=INK,
                linewidth=0.9,
            )
        labels = []
        for c in cells:
            prof, b = c.split(" @ ")
            labels.append(f"{PROFILE.get(prof, prof)}, k={int(float(b))}")
        ax.set_yticks(range(len(cells)))
        ax.set_yticklabels(labels, fontsize=6.6)
        ax.set_ylim(len(cells) - 0.4, -0.6)
        ax.set_xlim(-0.12, 0.12)
        ax.set_title(f"({'ab'[k]}) {title}", loc="left")
        ax.set_xlabel(r"$\Delta$ frontier gap (MAE)")
        ax.grid(axis="x")
    handles = [
        Patch(facecolor=BAND, edgecolor=INK, lw=0.6, label=f"$\\pm${margin:g} MAE margin"),
        Line2D([], [], marker="o", color=INK, ls="", mec=INK, ms=6, label="Holm-significant"),
        Line2D([], [], marker="o", color="white", ls="", mec=INK, ms=6, label="not significant"),
    ]
    fig.legend(
        handles=handles, loc="outside lower center", ncol=3, handletextpad=0.4, columnspacing=1.6
    )
    _save(fig, "fig_emaselect_margin")


def fig_oracle_headroom(profile: str = "uniform") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(fs.TEXT_W * 0.85, 2.1), layout="constrained")
    for k, (ax, (label, ds, color)) in enumerate(
        zip(axes, (("CrossCheck", "crosscheck", CC), ("CES", "ces", CES)), strict=False)
    ):
        data = json.loads(Path(f"outputs/emaselect_oracle_{ds}.json").read_text())
        by_b: dict[str, dict[str, list]] = {}
        for blk in data["by_seed"]:
            for b, vals in blk["by_burden_profile"][profile].items():
                slot = by_b.setdefault(b, {"fixed": [], "oracle": []})
                slot["fixed"].append(vals["fixed_subset_mae"])
                slot["oracle"].append(vals["oracle_mae"])
        budgets = sorted(by_b, key=float)
        x = [float(b) for b in budgets]
        for key, mk, ls, fc in (("fixed", "o", "-", color), ("oracle", "s", "--", "white")):
            m = [st.mean(by_b[b][key]) for b in budgets]
            sd = [st.stdev(by_b[b][key]) for b in budgets]
            ax.plot(x, m, marker=mk, ls=ls, color=color, mfc=fc, mec=INK, mew=0.9, ms=5.5, zorder=3)
            ax.fill_between(
                x,
                [a - s for a, s in zip(m, sd, strict=False)],
                [a + s for a, s in zip(m, sd, strict=False)],
                color=color,
                alpha=0.15,
                lw=0,
            )
        ax.set_xticks(x)
        ax.set_xlabel("item budget $k$")
        ax.set_title(f"({'ab'[k]}) {label}", loc="left")
        ax.grid(axis="y")
        if k == 0:
            ax.set_ylabel("raw MAE")
    handles = [
        Line2D([], [], marker="o", color=INK, ls="-", mfc=INK, ms=5.5, label="fixed subset"),
        Line2D(
            [],
            [],
            marker="s",
            color=INK,
            ls="--",
            mfc="white",
            ms=5.5,
            label="oracle (per-instance upper bound)",
        ),
    ]
    fig.legend(
        handles=handles, loc="outside lower center", ncol=2, handletextpad=0.4, columnspacing=1.6
    )
    _save(fig, "fig_oracle_headroom")


def fig_arena() -> None:
    """Decision-facing loop. Matches the caption: observations reach the
    model automatically (top arc), bypassing the review gate."""
    fig, ax = plt.subplots(figsize=(fs.TEXT_W, 1.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 44)
    ax.axis("off")

    def box(x, w, title, sub, fc, ec, tc=INK, ls="-", lw=1.4):
        y, h = 11, 15
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0,rounding_size=1.6",
                fc=fc,
                ec=ec,
                lw=lw,
                ls=ls,
                zorder=2,
            )
        )
        ax.text(
            x + w / 2,
            y + h * 0.66,
            title,
            ha="center",
            va="center",
            fontsize=5.8,
            fontweight="bold",
            color=tc,
            zorder=3,
        )
        ax.text(
            x + w / 2,
            y + h * 0.28,
            sub,
            ha="center",
            va="center",
            fontsize=6.0,
            color=tc if tc != INK else "#444444",
            zorder=3,
            linespacing=1.15,
        )

    def arrow(p, q, ls="-", color=INK, rad=0.0, lw=1.3):
        ax.add_patch(
            FancyArrowPatch(
                p,
                q,
                arrowstyle="-|>",
                mutation_scale=9,
                lw=lw,
                ls=ls,
                color=color,
                connectionstyle=f"arc3,rad={rad}",
                zorder=1,
            )
        )

    w, g = 16.4, 3.8
    xs = [0.8 + i * (w + g) for i in range(5)]
    box(xs[0], w, "Patient", "state", "#f2f2f2", GREY)
    box(xs[1], w, "Measurement", "which item\nto ask", "#dce9fa", CC)
    box(xs[2], w, "Patient model", "belief $b_t$", "#ffffff", INK, lw=1.8)
    box(xs[3], w, "Review", "which case a\nhuman inspects", "#dce9fa", CC)
    box(
        xs[4],
        w,
        "Intervention",
        "who receives\ncare",
        "#efefef",
        GREY,
        tc="#777777",
        ls=(0, (4, 2)),
    )
    ym = 11 + 7.5
    for a_, b_ in ((0, 1), (1, 2), (2, 3)):
        arrow((xs[a_] + w, ym), (xs[b_], ym))
    arrow((xs[3] + w, ym), (xs[4], ym), ls=(0, (4, 2)), color=GREY)
    ax.text(xs[1] + w + g / 2, ym + 3.0, "$o_t$", ha="center", fontsize=6.2)
    ax.text(xs[2] + w + g / 2, ym + 3.0, "$B$", ha="center", fontsize=6.2)
    arrow((xs[0] + w / 2, 26.5), (xs[2] + w / 2, 26.5), color=CES, rad=-0.3, lw=1.6)
    ax.text(
        xs[1] + w / 2 + 2,
        40.5,
        "recorded reports are ingested regardless of review",
        ha="center",
        fontsize=6.4,
        color="#b45a08",
        style="italic",
    )
    tag = dict(ha="center", va="center", fontsize=6.3, fontweight="bold", linespacing=1.15)
    ax.text(xs[1] + w / 2, 4.5, "EMASelect", color=CC, **tag)
    ax.text(xs[3] + w / 2, 4.5, "CrossCheckReview\nCESReview", color=CC, **tag)
    ax.text(xs[4] + w / 2, 4.5, "future work", color="#777777", **tag)
    _save(fig, "fig_arena")


if __name__ == "__main__":
    fig_arena()
    fig_review_forest()
    fig_emaselect_margin()
    fig_oracle_headroom()
