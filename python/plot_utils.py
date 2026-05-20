"""
Shared plotting utilities for all plot_*.py scripts.

Centralises model style constants and the apply_style() function so they are
defined once and imported by plot_powerspectra.py, plot_cells.py, and plot_snr.py.
"""

import matplotlib.pyplot as plt

MODEL_LABELS = {
    "lcdm": r"$\Lambda$CDM",
    "frhs": r"$f(R)$ HS",
    "ndgp": r"nDGP",
}
MODEL_LS = {
    "lcdm": "-",
    "frhs": "--",
    "ndgp": ":",
}
MODEL_COLORS = {
    "lcdm": "#1f77b4",
    "frhs": "#d62728",
    "ndgp": "#2ca02c",
}


def apply_style():
    """Apply a consistent publication-quality matplotlib style.

    All four spines are visible; ticks (major + minor) are drawn on all
    four sides, but value labels only on the default bottom/left.
    """
    try:
        import scienceplots
        plt.style.use(["science", "no-latex"])
    except ImportError:
        plt.rcParams.update({"font.family": "serif"})
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "lines.linewidth": 1.5,
        "font.size": 9,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "legend.frameon": True,
        # Four-sided frame + ticks (labels stay on bottom/left)
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.bottom": True,
        "axes.spines.left": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.top": True,
        "ytick.minor.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
    })
