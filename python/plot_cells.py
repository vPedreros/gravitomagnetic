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

CELL_LABELS = {
    "B_X_kSZ": r"$C_\ell^{B \times \mathrm{kSZ}}$",
    "Phi": r"$C_\ell^{\kappa\kappa}$",
}

CELL_TITLES = {
    "B_X_kSZ": r"$B \times \mathrm{kSZ}$ cross-spectrum",
    "Phi": r"Lensing convergence $\kappa\kappa$",
}

CELL_RATIO_LABELS = {
    "B_X_kSZ": r"$C_\ell^{B \times \mathrm{kSZ}} \,/\, C_\ell^{B \times \mathrm{kSZ},\,\Lambda\mathrm{CDM}}$",
    "Phi": r"$C_\ell^{\kappa\kappa} \,/\, C_\ell^{\kappa\kappa,\,\Lambda\mathrm{CDM}}$",
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


def plot_absolute(data, z_sources, models, out_dir, show):
    z_colors = z_colormap(z_sources)
    quantities = ["B_X_kSZ", "Phi"]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    model_handles, z_handles = [], []

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

        if qty == quantities[0]:
            for model in models:
                model_handles.append(
                    plt.Line2D(
                        [0],
                        [0],
                        ls=MODEL_LS[model],
                        color="0.3",
                        label=MODEL_LABELS[model],
                    )
                )
        if qty == quantities[-1] and models:
            first = models[0]
            for z in z_sources:
                if z in data[first]:
                    z_handles.append(
                        plt.Line2D(
                            [0], [0], ls="-", color=z_colors[z], label=rf"$z_s={z:.1f}$"
                        )
                    )

    axes[0].legend(handles=model_handles, loc="lower left", title="Model")
    axes[1].legend(handles=z_handles, loc="lower left", title="Source $z$")

    plt.tight_layout()
    out = Path(out_dir) / "cells_absolute.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    if show:
        plt.show()
    plt.close(fig)


def plot_ratio(data, z_sources, models, out_dir, show):
    if "lcdm" not in models:
        print("Skipping ratio plot — lcdm not in selected models.")
        return
    other_models = [m for m in models if m != "lcdm"]
    if not other_models:
        return

    z_colors = z_colormap(z_sources)
    quantities = ["B_X_kSZ", "Phi"]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax in axes:
        ax.axhline(1.0, color="0.5", lw=0.8, ls="--")

    model_handles, z_handles = [], []

    for ax, qty in zip(axes, quantities):
        for model in other_models:
            for z in z_sources:
                if z not in data[model] or z not in data["lcdm"]:
                    continue
                d = data[model][z]
                d_lcdm = data["lcdm"][z]
                ratio = np.abs(d[qty]) / np.interp(
                    d["ell"], d_lcdm["ell"], np.abs(d_lcdm[qty])
                )
                ax.semilogx(
                    d["ell"], ratio, ls=MODEL_LS[model], color=z_colors[z], alpha=0.9
                )

        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(CELL_RATIO_LABELS[qty])

        if qty == quantities[0]:
            for model in other_models:
                model_handles.append(
                    plt.Line2D(
                        [0],
                        [0],
                        ls=MODEL_LS[model],
                        color="0.3",
                        label=MODEL_LABELS[model],
                    )
                )
        if qty == quantities[-1] and other_models:
            first = other_models[0]
            for z in z_sources:
                if z in data[first]:
                    z_handles.append(
                        plt.Line2D(
                            [0], [0], ls="-", color=z_colors[z], label=rf"$z_s={z:.1f}$"
                        )
                    )

    axes[0].set_title(r"$B \times \mathrm{kSZ}$ ratio")
    axes[1].set_title(r"Lensing $\kappa\kappa$ ratio")
    axes[0].legend(handles=model_handles, loc="upper left", title="Model")
    axes[1].legend(handles=z_handles, loc="upper left", title="Source $z$")

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

    plot_absolute(data, z_sources, args.models, out_dir, args.show)
    plot_ratio(data, z_sources, args.models, out_dir, args.show)


if __name__ == "__main__":
    main()
