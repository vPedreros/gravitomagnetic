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
    """Apply a consistent publication-quality matplotlib style."""
    try:
        import scienceplots
        plt.style.use(["science", "no-latex"])
    except ImportError:
        plt.rcParams.update({
            "font.family": "serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
        })
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "lines.linewidth": 1.5,
        "font.size": 9,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.frameon": False,
    })
