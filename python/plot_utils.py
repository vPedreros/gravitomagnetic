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

GOLDEN_RATIO = (5**0.5 + 1) / 2  # 1.618033988749895


def set_size(columns=1, subplots=(1, 1), aspect="square", ratio=None):
    """Calculate exact figure dimensions in inches for MNRAS.

    Parameters
    ----------
    columns : int
        1 for single-column width (240 pt = ~3.32 in), 2 for double-column width (504 pt = ~6.97 in).
    subplots : tuple (int, int)
        Number of subplot rows and columns: (rows, cols).
    aspect : str, optional
        - 'square' (default) : Each individual subplot panel is square (1:1 aspect ratio).
        - 'golden'           : Overall figure height follows the Golden Ratio (height = width / 1.618).
    ratio : float, optional
        Explicit total figure height-to-width ratio (fig_height = fig_width * ratio).
    """
    if columns == 1:
        fig_width = 240.0 / 72.27  # ~3.32088 inches
    else:
        fig_width = 504.0 / 72.27  # ~6.97385 inches

    rows, cols = subplots
    if ratio is not None:
        fig_height = fig_width * ratio
    elif aspect == "square":
        # Each panel is square (width == height)
        panel_width = fig_width / cols
        panel_height = panel_width  # 1:1 ratio
        fig_height = panel_height * rows
    elif aspect == "golden":
        # Overall figure height follows Golden Ratio
        fig_height = fig_width / GOLDEN_RATIO
    else:
        panel_width = fig_width / cols
        fig_height = panel_width * rows

    return (fig_width, fig_height)




def apply_style(columns=1):
    """Apply a consistent publication-quality matplotlib style for MNRAS.

    MNRAS specifications:
      - Single column: 3.32 in (240 pt)
      - Full text width: 6.97 in (504 pt)
      - Ticks inside on all 4 sides with major/minor ticks
      - Times-Roman serif fonts with 8-9pt text matching document font size
    """
    try:
        import scienceplots
        plt.style.use(["science", "no-latex"])
    except ImportError:
        pass

    fig_width, fig_height = set_size(columns=columns)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
        "mathtext.fontset": "stix",
        
        # MNRAS TYPOGRAPHY (Document main text is 9pt, captions 8pt)
        "font.size": 9.0,
        "axes.labelsize": 9.0,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "legend.title_fontsize": 8.5,

        # DYNAMIC FIGURE SIZE
        "figure.figsize": (fig_width, fig_height),
        "figure.subplot.hspace": 0.05,
        
        # LINES & GRID
        "lines.linewidth": 1.2,
        "axes.linewidth": 0.65,
        "axes.grid": False,
        
        # TICKS (MNRAS requires ticks on all 4 sides, pointing inward)
        "xtick.top": True,
        "xtick.bottom": True,
        "xtick.direction": "in",
        "xtick.major.size": 5,
        "xtick.minor.size": 3,
        "xtick.major.width": 0.5,
        "xtick.minor.width": 0.35,
        "xtick.minor.visible": True,
        
        "ytick.left": True,
        "ytick.right": True,
        "ytick.direction": "in",
        "ytick.major.size": 5,
        "ytick.minor.size": 3,
        "ytick.major.width": 0.5,
        "ytick.minor.width": 0.35,
        "ytick.minor.visible": True,

        # LEGEND STYLING
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",

        # RESOLUTION & SAVING
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "savefig.format": "pdf",
    })