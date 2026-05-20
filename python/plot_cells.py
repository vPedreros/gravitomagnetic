"""
Script to plot angular power spectra C_ell from output data.

Produces two figures:
  1. C_ell^{BxkSZ} and C_ell^Phi for all models at selected z_source values
  2. Ratio C_ell^X / C_ell^{X,LCDM} for the same quantities

Usage:
  python plot_cells.py --in-dir output --out-dir imgs
  python plot_cells.py --in-dir output --out-dir imgs --z-sources 0.5 1.0 1.5 2.0 --show
"""

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from plot_utils import MODEL_COLORS, MODEL_LABELS, MODEL_LS, apply_style

QUANTITIES = ("Phi", "B", "kSZ", "B_X_kSZ")

CELL_LABELS = {
    "Phi": r"$C_\ell^{\kappa\kappa}$",
    "B": r"$C_\ell^{BB}$",
    "kSZ": r"$C_\ell^{\mathrm{kSZ}\,\mathrm{kSZ}}$",
    "B_X_kSZ": r"$C_\ell^{B \times \mathrm{kSZ}}$",
}

CELL_TITLES = {
    "Phi": r"Lensing convergence $\kappa\kappa$",
    "B": r"Gravitomagnetic $BB$",
    "kSZ": r"kSZ auto-spectrum",
    "B_X_kSZ": r"$B \times \mathrm{kSZ}$ cross-spectrum",
}

CELL_RATIO_LABELS = {
    "Phi": r"$C_\ell^{\kappa\kappa} \,/\, C_\ell^{\kappa\kappa,\,\Lambda\mathrm{CDM}}$",
    "B": r"$C_\ell^{BB} \,/\, C_\ell^{BB,\,\Lambda\mathrm{CDM}}$",
    "kSZ": r"$C_\ell^{\mathrm{kSZ}\,\mathrm{kSZ}} \,/\, C_\ell^{\mathrm{kSZ}\,\mathrm{kSZ},\,\Lambda\mathrm{CDM}}$",
    "B_X_kSZ": r"$C_\ell^{B \times \mathrm{kSZ}} \,/\, C_\ell^{B \times \mathrm{kSZ},\,\Lambda\mathrm{CDM}}$",
}

CELL_RATIO_TITLES = {
    "Phi": r"Lensing $\kappa\kappa$ ratio",
    "B": r"$BB$ ratio",
    "kSZ": r"kSZ ratio",
    "B_X_kSZ": r"$B \times \mathrm{kSZ}$ ratio",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot angular power spectra C_ell.")
    parser.add_argument(
        "--in-dir",
        required=True,
        help="Base data directory (contains lcdm/, frhs/, ndgp/).",
    )
    parser.add_argument(
        "--out-dir", default="imgs", help="Directory for saved figures."
    )
    parser.add_argument(
        "--z-sources",
        nargs="+",
        type=float,
        default=[0.5, 1.0, 1.5, 2.0],
        help="Source redshifts to plot.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lcdm", "frhs", "ndgp"],
        choices=["lcdm", "frhs", "ndgp"],
    )
    parser.add_argument(
        "--quantities",
        nargs="+",
        default=list(QUANTITIES),
        choices=list(QUANTITIES),
        help="Which C_ell quantities to plot (one panel each).",
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def load_cells(base, model, z_sources):
    """Return dict: {z: {'ell': ..., 'Phi': ..., 'kSZ': ..., 'B': ..., 'B_X_kSZ': ...}}."""
    result = {}
    for z in z_sources:
        cell_path = base / model / "C_ells" / f"C_ells_XY_z={z}.npy"
        ell_path = base / model / "C_ells" / f"ell_grid_z={z}.npy"
        if not cell_path.exists():
            print(f"  Warning: missing {cell_path}")
            continue
        d = np.load(cell_path, allow_pickle=True).item()
        ell = np.load(ell_path)
        result[z] = {"ell": ell} | d
    return result


def z_colormap(z_values):
    norm = mcolors.Normalize(vmin=min(z_values), vmax=max(z_values))
    cmap = plt.cm.plasma_r
    return {z: cmap(norm(z)) for z in z_values}


def _make_axes(n_panels):
    """Layout for N panels: 1xN row (N<=4 covers our use case)."""
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4), squeeze=False)
    return fig, axes[0]


def _model_legend_handles(models):
    return [
        plt.Line2D([0], [0], ls=MODEL_LS[m], color="0.3", label=MODEL_LABELS[m])
        for m in models
    ]


def _z_legend_handles(z_sources, z_colors, data, ref_model):
    handles = []
    for z in z_sources:
        if z in data[ref_model]:
            handles.append(
                plt.Line2D([0], [0], ls="-", color=z_colors[z], label=rf"$z_s={z:.1f}$")
            )
    return handles


def plot_absolute(data, z_sources, models, quantities, out_dir, show):
    z_colors = z_colormap(z_sources)
    fig, axes = _make_axes(len(quantities))

    for ax, qty in zip(axes, quantities):
        for model in models:
            for z in z_sources:
                if z not in data[model]:
                    continue
                d = data[model][z]
                ax.loglog(
                    d["ell"],
                    d["ell"] * (d["ell"] + 1) * np.abs(d[qty]),
                    ls=MODEL_LS[model],
                    color=z_colors[z],
                    alpha=0.9,
                )

        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(CELL_LABELS[qty])
        ax.set_title(CELL_TITLES[qty])

    if models:
        axes[0].legend(
            handles=_model_legend_handles(models), loc="lower left", title="Model"
        )
    if len(axes) > 1 and models:
        axes[-1].legend(
            handles=_z_legend_handles(z_sources, z_colors, data, models[0]),
            loc="lower left", title="Source $z$",
        )

    plt.tight_layout()
    out = Path(out_dir) / "cells_absolute.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    if show:
        plt.show()
    plt.close(fig)


def plot_ratio(data, z_sources, models, quantities, out_dir, show):
    if "lcdm" not in models:
        print("Skipping ratio plot — lcdm not in selected models.")
        return
    other_models = [m for m in models if m != "lcdm"]
    if not other_models:
        return

    z_colors = z_colormap(z_sources)
    fig, axes = _make_axes(len(quantities))
    for ax in axes:
        ax.axhline(1.0, color="0.5", lw=0.8, ls="--")

    for ax, qty in zip(axes, quantities):
        for model in other_models:
            for z in z_sources:
                if z not in data[model] or z not in data["lcdm"]:
                    continue
                d = data[model][z]
                d_lcdm = data["lcdm"][z]
                lcdm_val = np.interp(d["ell"], d_lcdm["ell"], np.abs(d_lcdm[qty]))
                # _compute_C_ell returns 0 at ells outside the valid k-range,
                # which would produce inf/NaN here. NaN out those points so the
                # plot draws a gap instead of a meaningless spike.
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(lcdm_val > 0, np.abs(d[qty]) / lcdm_val, np.nan)
                ax.semilogx(
                    d["ell"], ratio, ls=MODEL_LS[model], color=z_colors[z], alpha=0.9
                )

        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(CELL_RATIO_LABELS[qty])
        ax.set_title(CELL_RATIO_TITLES[qty])

    axes[0].legend(
        handles=_model_legend_handles(other_models), loc="upper left", title="Model"
    )
    if len(axes) > 1:
        axes[-1].legend(
            handles=_z_legend_handles(z_sources, z_colors, data, other_models[0]),
            loc="upper left", title="Source $z$",
        )

    plt.tight_layout()
    out = Path(out_dir) / "cells_ratio.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    args = parse_args()
    apply_style()

    base = Path(args.in_dir).expanduser()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    z_sources = sorted(args.z_sources)

    print("Loading C_ell data...")
    data = {m: load_cells(base, m, z_sources) for m in args.models}

    plot_absolute(data, z_sources, args.models, args.quantities, out_dir, args.show)
    plot_ratio(data, z_sources, args.models, args.quantities, out_dir, args.show)


if __name__ == "__main__":
    main()
