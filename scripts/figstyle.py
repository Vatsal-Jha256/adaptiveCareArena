"""Origin-inspired matplotlib style for the figures (matplotlib only).

Conventions: white background, closed 4-sided black frame at a single
linewidth, inward ticks on all sides, black-edged fills, sans-serif text
sized for a fixed 5.5 in figure width (no downscaling), embedded
Type-42 fonts, one colour per cohort across every figure.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CC = "#1f6fd1"  # CrossCheck
CES = "#f07d1a"  # CES
INK = "#111111"
GREY = "#8a8a8a"
BAND = "#e8e8e8"
FRAME_LW = 1.0
TEXT_W = 5.5  # figure width, inches


def apply() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8,
            "axes.titleweight": "bold",
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "mathtext.default": "regular",
            "axes.edgecolor": INK,
            "axes.linewidth": FRAME_LW,
            "axes.facecolor": "white",
            "axes.axisbelow": True,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.width": FRAME_LW,
            "ytick.major.width": FRAME_LW,
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "xtick.minor.visible": False,
            "axes.grid": False,
            "grid.color": "#dddddd",
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.6,
            "lines.markersize": 5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def panel_label(ax, letter: str) -> None:
    ax.text(
        0.0,
        1.03,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
