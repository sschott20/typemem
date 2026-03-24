"""Shared plot styling for typemem benchmark figures."""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt

# Wong colorblind-safe palette
WONG_PALETTE = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "pink": "#CC79A7",
    "yellow": "#F0E442",
}

STRATEGY_COLORS = {
    "full_context": WONG_PALETTE["blue"],
    "monolithic_rag": WONG_PALETTE["orange"],
    "tiered_memory": WONG_PALETTE["green"],
    "rag_with_recency": WONG_PALETTE["pink"],
    "tiered_no_consol": WONG_PALETTE["yellow"],
}

STRATEGY_LINESTYLES = {
    "full_context": "solid",
    "monolithic_rag": "dashed",
    "tiered_memory": "dashdot",
    "rag_with_recency": "dotted",
    "tiered_no_consol": (0, (3, 1, 1, 1, 1, 1)),
}

STRATEGY_MARKERS = {
    "full_context": "o",
    "monolithic_rag": "s",
    "tiered_memory": "^",
    "rag_with_recency": "D",
    "tiered_no_consol": "v",
}

STRATEGY_LABELS = {
    "full_context": "Full Context",
    "monolithic_rag": "Monolithic RAG",
    "tiered_memory": "Tiered Memory",
    "rag_with_recency": "RAG + Recency",
    "tiered_no_consol": "Tiered (no consol.)",
}

FIG_WIDTH = 3.5
FIG_HEIGHT = 2.6


def apply_style() -> None:
    """Apply academic figure styling globally."""
    matplotlib.use("Agg")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.figsize": (FIG_WIDTH, FIG_HEIGHT),
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.grid.axis": "y",
    })


def save_fig(fig: plt.Figure, path: str) -> None:
    """Save figure as both PDF and PNG."""
    fig.savefig(path + ".pdf", bbox_inches="tight")
    fig.savefig(path + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def get_color(strategy: str) -> str:
    return STRATEGY_COLORS.get(strategy, "#333333")


def get_linestyle(strategy: str) -> str | tuple:
    return STRATEGY_LINESTYLES.get(strategy, "solid")


def get_marker(strategy: str) -> str:
    return STRATEGY_MARKERS.get(strategy, "x")


def get_label(strategy: str) -> str:
    return STRATEGY_LABELS.get(strategy, strategy)
