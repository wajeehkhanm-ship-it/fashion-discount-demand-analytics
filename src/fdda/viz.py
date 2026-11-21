"""Shared chart style and helpers for report figures (static matplotlib PNGs).

Colours follow a validated colour-blind-safe categorical order; categorical hues are
assigned in that fixed order (never cycled). One y-axis per chart, thin marks,
recessive grid, selective direct labels.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, PercentFormatter

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8984"
GRID = "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER_GRAY = "#b4b3ad"


def apply_style() -> None:
    """Set matplotlib defaults used by every project figure."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.edgecolor": GRID,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "text.color": TEXT,
            "legend.frameon": False,
            "lines.linewidth": 2,
        }
    )


def subtitle(ax: Axes, text: str) -> None:
    """Small secondary line under the title (source, unit, caveat)."""
    ax.annotate(
        text,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(0, 4),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=9,
        color=TEXT_SECONDARY,
    )


def pct_axis(ax: Axes, axis: str = "x", decimals: int = 0) -> None:
    """Format an axis holding fractions (0-1) as percentages."""
    fmt = PercentFormatter(xmax=1, decimals=decimals)
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)


def thousands_axis(ax: Axes, axis: str = "y") -> None:
    """Format an axis as 12k / 1.2M."""

    def _fmt(v: float, _pos: int) -> str:
        if abs(v) >= 1e6:
            return f"{v / 1e6:.1f}M"
        if abs(v) >= 1e3:
            return f"{v / 1e3:.0f}k"
        return f"{v:.0f}"

    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(FuncFormatter(_fmt))


def hbar(
    ax: Axes,
    labels: Sequence[str],
    series: dict[str, Sequence[float]],
    *,
    label_fmt: str = "{:.0%}",
    other_label: str | None = "Other",
) -> None:
    """Horizontal (grouped) bars, largest at top, with direct value labels.

    One series -> single colour; 2-4 series -> fixed categorical order plus legend.
    A row named `other_label` is drawn in neutral gray when there is one series.
    """
    n_series = len(series)
    y = np.arange(len(labels))
    height = 0.72 / n_series
    xmax = max(max(v) for v in series.values())
    for i, (name, values) in enumerate(series.items()):
        offset = (i - (n_series - 1) / 2) * height
        colors = [
            OTHER_GRAY if (n_series == 1 and lab == other_label) else SERIES[i] for lab in labels
        ]
        ax.barh(y + offset, values, height=height * 0.9, color=colors, label=name)
        for yi, v in zip(y + offset, values, strict=True):
            ax.text(
                v + xmax * 0.01,
                yi,
                label_fmt.format(v),
                va="center",
                fontsize=8,
                color=TEXT_SECONDARY,
            )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, xmax * 1.15)
    if n_series > 1:
        ax.legend(loc="lower right", ncols=n_series)


def vbar(
    ax: Axes,
    categories: Sequence[str],
    series: dict[str, Sequence[float]],
    *,
    label_fmt: str = "{:.0%}",
) -> None:
    """Vertical (grouped) bars for ordered categories such as discount bands, labelled."""
    n_series = len(series)
    x = np.arange(len(categories))
    width = 0.76 / n_series
    ymax = max(max(v) for v in series.values())
    for i, (name, values) in enumerate(series.items()):
        offset = (i - (n_series - 1) / 2) * width
        ax.bar(x + offset, values, width=width * 0.9, color=SERIES[i], label=name)
        for xi, v in zip(x + offset, values, strict=True):
            ax.text(
                xi,
                v + ymax * 0.015,
                label_fmt.format(v),
                ha="center",
                va="bottom",
                fontsize=8,
                color=TEXT_SECONDARY,
            )
    ax.set_xticks(x, categories)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, ymax * 1.15)
    if n_series > 1:
        ax.legend(loc="upper right")


def save_fig(fig: Figure, name: str, out_dir: Path) -> Path:
    """Save a figure as PNG under `out_dir` and return the path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path)
    return path


def new_fig(width: float = 8, height: float = 4.5) -> tuple[Figure, Axes]:
    """Create a styled single-axes figure."""
    apply_style()
    fig, ax = plt.subplots(figsize=(width, height), layout="constrained")
    return fig, ax
