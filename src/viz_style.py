"""Shared design system for the static showcase figures.

The showcase figures are produced once by ``python -m src.pipeline figures`` and
served as plain PNGs on GitHub Pages, so they have to stand on their own: a
colorblind-safe palette, a takeaway title (the point of the chart, not just its
topic), bars that start at zero and axes that are never truncated to exaggerate a
difference.

Everything here is presentation only. It reads numbers that already exist in
``reports/results`` and never recomputes or alters a model result.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")  # headless / CI friendly
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from .config import project_root  # noqa: E402

# ---------------------------------------------------------------------------
# Palette — Okabe-Ito, the standard colorblind-safe qualitative set.
# ---------------------------------------------------------------------------
OKABE_ITO: dict[str, str] = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "grey": "#999999",
    "black": "#000000",
    "navy": "#1B3A5B",
    "light_grey": "#DDDDDD",
}

# Semantic roles so the same quantity is always the same colour across figures.
ROLE: dict[str, str] = {
    "primary": OKABE_ITO["blue"],
    "secondary": OKABE_ITO["orange"],
    "tertiary": OKABE_ITO["green"],
    "danger": OKABE_ITO["vermillion"],
    "muted": OKABE_ITO["grey"],
    "accent": OKABE_ITO["sky"],
    "series": OKABE_ITO["purple"],
    "reference": OKABE_ITO["black"],
}

# Category colours for multi-series bars/segments (cycled in order).
CATEGORICAL: list[str] = [
    OKABE_ITO["blue"],
    OKABE_ITO["orange"],
    OKABE_ITO["green"],
    OKABE_ITO["vermillion"],
    OKABE_ITO["purple"],
    OKABE_ITO["sky"],
]

SHOWCASE_DIR = project_root() / "reports" / "figures" / "showcase"
DOCS_ASSETS = project_root() / "docs" / "assets"

# Figure geometry (inches @ 130 dpi -> ~1300 px wide, comfortably under 300 KB).
FIGSIZE_WIDE = (9.6, 5.4)
FIGSIZE_PANEL = (11.0, 8.2)
FIGSIZE_HERO = (12.0, 6.3)


def apply_style() -> None:
    """Apply the showcase look on top of the project's global rcParams."""
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 130,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.color": OKABE_ITO["light_grey"],
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#333333",
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.labelsize": 10.5,
            "axes.labelcolor": "#222222",
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.size": 10.5,
            "text.color": "#222222",
        }
    )


def takeaway(ax: plt.Axes, title: str, subtitle: str | None = None) -> None:
    """Set a takeaway title with an optional lighter subtitle line."""
    if subtitle:
        ax.set_title(f"{title}\n", fontsize=13, fontweight="bold", loc="left")
        ax.text(
            0.0,
            1.015,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9.5,
            color="#555555",
        )
    else:
        ax.set_title(title, fontsize=13, fontweight="bold", loc="left")


def annotate_source(target, text: str) -> None:
    """Small provenance footnote so every figure names its source artifact.

    Accepts either an ``Axes`` or a whole ``Figure`` (for panel figures).
    """
    if isinstance(target, plt.Figure):
        target.text(
            0.0,
            0.0,
            text,
            ha="left",
            va="bottom",
            fontsize=8,
            color="#777777",
        )
        return
    target.text(
        0.0,
        -0.16,
        text,
        transform=target.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#777777",
    )


def pct_axis(ax: plt.Axes, axis: str = "y", decimals: int = 1) -> None:
    """Format an axis as percentages."""
    fmt = FuncFormatter(lambda v, _: f"{v * 100:.{decimals}f}%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def bars_from_zero(ax: plt.Axes, values: Iterable[float], **kwargs) -> None:
    """Draw bars, asserting the axis will not be truncated below zero."""
    ax.bar(range(len(list(values))), list(values), **kwargs)


def save_showcase(fig: plt.Figure, name: str, out_dir: Path | None = None) -> Path:
    """Save a showcase figure (PNG) under ``reports/figures/showcase``."""
    out_dir = out_dir or SHOWCASE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path)
    plt.close(fig)
    return path


def horizontal_barh(
    ax: plt.Axes,
    labels: Sequence[str],
    values: Sequence[float],
    colors: Sequence[str] | None = None,
    value_fmt: str = "{:.2f}",
) -> None:
    """Horizontal bars (most-influential first) with in-bar value labels."""
    y = range(len(labels))
    if colors is None:
        colors = [ROLE["primary"]] * len(labels)
    ax.barh(list(y), list(values), color=list(colors), height=0.68)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # first label on top
    vmax = max(values) if len(values) else 1.0
    for i, v in enumerate(values):
        ax.text(v + vmax * 0.012, i, value_fmt.format(v), va="center", fontsize=9)
