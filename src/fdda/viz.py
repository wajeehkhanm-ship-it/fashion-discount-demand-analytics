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


DIVERGING_LOW = "#d03b3b"  # below reference (loss)
DIVERGING_MID = "#f0efec"  # neutral gray at the reference value
DIVERGING_HIGH = "#1c5cab"  # above reference (gain)


def index_lines(
    ax: Axes,
    categories: Sequence[str],
    series: dict[str, Sequence[float]],
    *,
    ref: float = 1.0,
    label_last: bool = True,
) -> None:
    """Lines with markers over ordered categories, a reference line, and end labels."""
    x = np.arange(len(categories))
    ax.axhline(ref, color=TEXT_MUTED, linewidth=1, linestyle="--", zorder=1)
    for i, (name, values) in enumerate(series.items()):
        vals = np.asarray(values, dtype=float)
        ax.plot(
            x,
            vals,
            color=SERIES[i],
            marker="o",
            markersize=6,
            label=name,
            zorder=3,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
        )
        if label_last:
            last = np.flatnonzero(~np.isnan(vals))
            if len(last):
                j = last[-1]
                ax.annotate(
                    f"{vals[j]:.2f}",
                    (x[j], vals[j]),
                    xytext=(8, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=8,
                    color=TEXT_SECONDARY,
                )
    ax.set_xticks(x, categories)
    ax.grid(axis="x", visible=False)
    if len(series) > 1:
        ax.legend(loc="best")


def diverging_heatmap(
    ax: Axes,
    data: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    *,
    center: float = 1.0,
    fmt: str = "{:.2f}",
    marks: Sequence[tuple[int, int]] = (),
) -> None:
    """Heatmap around `center` (red below, gray at, blue above) with cell values.

    `marks` are (row, col) cells outlined to highlight them (e.g. the payback ceiling).
    Missing values are left blank.
    """
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    cmap = LinearSegmentedColormap.from_list("div", [DIVERGING_LOW, DIVERGING_MID, DIVERGING_HIGH])
    finite = data[np.isfinite(data)]
    span = max(abs(finite.max() - center), abs(center - finite.min()), 1e-6)
    norm = TwoSlopeNorm(vcenter=center, vmin=center - span, vmax=center + span)
    ax.imshow(np.ma.masked_invalid(data), cmap=cmap, norm=norm, aspect="auto")
    for (r, c), v in np.ndenumerate(data):
        if np.isfinite(v):
            dark = abs(norm(v) - 0.5) > 0.32
            ax.text(
                c,
                r,
                fmt.format(v),
                ha="center",
                va="center",
                fontsize=8,
                color=SURFACE if dark else TEXT,
            )
    for r, c in marks:
        ax.add_patch(
            mpl.patches.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor=TEXT, linewidth=2)
        )
    ax.set_xticks(range(len(col_labels)), col_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def forest(
    ax: Axes,
    labels: Sequence[str],
    series: dict[str, tuple[Sequence[float], Sequence[float], Sequence[float]]],
    *,
    ref: float = 0.0,
) -> None:
    """Point estimates with confidence intervals; `series` maps name -> (est, low, high)."""
    y = np.arange(len(labels))
    n = len(series)
    step = 0.28 if n > 1 else 0
    ax.axvline(ref, color=TEXT_MUTED, linewidth=1, linestyle="--")
    for i, (name, (est, lo, hi)) in enumerate(series.items()):
        off = (i - (n - 1) / 2) * step
        est, lo, hi = (np.asarray(a, dtype=float) for a in (est, lo, hi))
        ax.errorbar(
            est,
            y + off,
            xerr=[est - lo, hi - est],
            fmt="o",
            color=SERIES[i],
            ecolor=SERIES[i],
            elinewidth=2,
            capsize=0,
            markersize=6,
            label=name,
            markeredgecolor=SURFACE,
            markeredgewidth=1.2,
        )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    if n > 1:
        ax.legend(loc="lower left")


SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def sequential_heatmap(
    ax: Axes,
    data: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    *,
    fmt: str = "{:.0%}",
    annotate: bool = True,
) -> None:
    """One-hue (blue) heatmap for magnitudes such as retention; missing cells are blank."""
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq", SEQUENTIAL_BLUE)
    finite = data[np.isfinite(data)]
    vmin, vmax = float(finite.min()), float(finite.max())
    ax.imshow(np.ma.masked_invalid(data), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    if annotate:
        for (r, c), v in np.ndenumerate(data):
            if np.isfinite(v):
                dark = (v - vmin) / max(vmax - vmin, 1e-9) > 0.55
                ax.text(
                    c,
                    r,
                    fmt.format(v),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=SURFACE if dark else TEXT,
                )
    ax.set_xticks(range(len(col_labels)), col_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def stacked_bars(
    ax: Axes,
    categories: Sequence[str],
    series: dict[str, Sequence[float]],
    *,
    other_label: str = "Other",
) -> None:
    """Vertical stacked bars; series in fixed colour order, `other_label` in neutral gray."""
    x = np.arange(len(categories))
    bottom = np.zeros(len(categories))
    i = 0
    for name, values in series.items():
        vals = np.asarray(values, dtype=float)
        color = OTHER_GRAY if name == other_label else SERIES[i]
        if name != other_label:
            i += 1
        ax.bar(
            x,
            vals,
            bottom=bottom,
            width=0.8,
            color=color,
            label=name,
            edgecolor=SURFACE,
            linewidth=0.8,
        )
        bottom += vals
    ax.set_xticks(x, categories)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))


def forecast_panel(
    ax: Axes,
    actual: tuple[Sequence, Sequence],
    forecast: tuple[Sequence, Sequence, Sequence, Sequence],
    *,
    baseline: tuple[Sequence, Sequence] | None = None,
    title: str = "",
    labels: tuple[str, str, str] = ("Actual", "Forecast", "Seasonal naive"),
) -> None:
    """Actual line, forecast line with its interval band, and an optional baseline line.

    `actual` = (dates, values); `forecast` = (dates, yhat, lower, upper).
    """
    ax.plot(actual[0], actual[1], color=TEXT_SECONDARY, linewidth=1.5, label=labels[0])
    ds, yhat, lo, hi = forecast
    ax.fill_between(ds, lo, hi, color=SERIES[0], alpha=0.18, linewidth=0, label="80% interval")
    ax.plot(ds, yhat, color=SERIES[0], linewidth=2, label=labels[1])
    if baseline is not None:
        ax.plot(
            baseline[0],
            baseline[1],
            color=SERIES[1],
            linewidth=1.5,
            linestyle="--",
            label=labels[2],
        )
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_ylim(bottom=0)
    ax.grid(axis="x", visible=False)


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
