"""
Plot cumulative SNR vs source redshift for all survey/CMB-experiment combinations.

Produces two figures:
  1. Cumulative SNR(z_source) for each model, one panel per survey×experiment
  2. Per-ell SNR at a selected z_source for each model

Cumulative SNR = sqrt( sum_ell SNR_ell^2 )

Usage:
  python plot_snr.py --in-dir output --out-dir imgs
  python plot_snr.py --in-dir output --out-dir imgs --z-ref 1.5 --show
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pathlib import Path

from plot_utils import MODEL_LABELS, MODEL_LS, MODEL_COLORS, apply_style

SURVEYS = ["Euclid", "LSST"]
EXPERIMENTS = ["Planck", "SO"]
COMBOS = [f"{s}_{e}" for s in SURVEYS for e in EXPERIMENTS]

COMBO_LABELS = {
    "Euclid_Planck": "Euclid × Planck",
    "Euclid_SO": "Euclid × SO",
    "LSST_Planck": "LSST × Planck",
    "LSST_SO": "LSST × SO",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot SNR vs source redshift.")
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--out-dir", default="imgs")
    parser.add_argument("--models", nargs="+", default=["lcdm", "frhs", "ndgp"],
                        choices=["lcdm", "frhs", "ndgp"])
    parser.add_argument("--z-ref", type=float, default=1.5,
                        help="Reference z_source for the per-ell SNR plot.")
    parser.add_argument("--colorbar-experiments", nargs="+", default=["SO", "Planck"],
                        choices=["SO", "Planck"],
                        help="CMB experiments to plot in the colorbar (per-model) figure.")
    parser.add_argument("--only", nargs="+", default=["cumulative", "per-ell", "colorbar"],
                        choices=["cumulative", "per-ell", "colorbar"],
                        help="Which plot families to produce.")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def load_snr_data(base, model, combo):
    """Return (z_sources sorted, cumulative_SNR array)."""
    snr_dir = base / model / "SNRs" / combo
    z_vals, cumsnr = [], []
    for f in sorted(snr_dir.glob("SNR_z=*.npy")):
        z = float(f.stem.replace("SNR_z=", ""))
        snr_ell = np.load(f)
        z_vals.append(z)
        cumsnr.append(np.sqrt(np.sum(snr_ell**2)))
    order = np.argsort(z_vals)
    return np.array(z_vals)[order], np.array(cumsnr)[order]


def load_snr_per_ell(base, model, combo, z_ref):
    """Return (ell_grid, snr_ell) at the closest available z_source."""
    snr_dir = base / model / "SNRs" / combo
    files = sorted(snr_dir.glob("SNR_z=*.npy"))
    z_avail = [float(f.stem.replace("SNR_z=", "")) for f in files]
    idx = int(np.argmin(np.abs(np.array(z_avail) - z_ref)))
    snr_ell = np.load(files[idx])
    ell_path = base / model / "C_ells" / f"ell_grid_z={z_avail[idx]}.npy"
    ell = np.load(ell_path)
    return ell, snr_ell, z_avail[idx]


def plot_cumulative_snr(base, models, out_dir, show):
    """2×2 grid: one panel per survey×experiment, cumulative SNR vs z_source."""
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True)
    axes_flat = axes.flatten()

    handles = []
    for model in models:
        handles.append(
            plt.Line2D([0], [0], ls=MODEL_LS[model], color=MODEL_COLORS[model],
                       label=MODEL_LABELS[model])
        )

    for ax, combo in zip(axes_flat, COMBOS):
        for model in models:
            try:
                z_arr, cumsnr = load_snr_data(base, model, combo)
            except (FileNotFoundError, ValueError):
                continue
            ax.plot(z_arr, cumsnr,
                    ls=MODEL_LS[model], color=MODEL_COLORS[model], label=MODEL_LABELS[model])
        ax.set_title(COMBO_LABELS[combo])
        ax.set_ylabel(r"Cumulative SNR")

    for ax in axes[1]:
        ax.set_xlabel(r"Source redshift $z_s$")

    axes_flat[0].legend(handles=handles, loc="upper left")
    plt.suptitle(r"Cumulative SNR $= \sqrt{\sum_\ell \mathrm{SNR}_\ell^2}$", y=1.01)
    plt.tight_layout()
    out = Path(out_dir) / "snr_cumulative.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    if show:
        plt.show()
    plt.close(fig)


def plot_snr_per_ell(base, models, z_ref, out_dir, show):
    """2×2 grid: per-ell SNR at z_ref for each survey×experiment."""
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True)
    axes_flat = axes.flatten()

    handles = []
    for model in models:
        handles.append(
            plt.Line2D([0], [0], ls=MODEL_LS[model], color=MODEL_COLORS[model],
                       label=MODEL_LABELS[model])
        )

    z_used = None
    for ax, combo in zip(axes_flat, COMBOS):
        for model in models:
            try:
                ell, snr_ell, z_used = load_snr_per_ell(base, model, combo, z_ref)
            except (FileNotFoundError, ValueError):
                continue
            ax.semilogx(ell, snr_ell,
                        ls=MODEL_LS[model], color=MODEL_COLORS[model])
        ax.set_title(COMBO_LABELS[combo])
        ax.set_ylabel(r"SNR$_\ell$")

    for ax in axes[1]:
        ax.set_xlabel(r"$\ell$")

    axes_flat[0].legend(handles=handles, loc="upper left")
    ztitle = f"$z_s = {z_used:.1f}$" if z_used is not None else f"$z_s \\approx {z_ref}$"
    plt.suptitle(rf"Per-$\ell$ SNR at {ztitle}", y=1.01)
    plt.tight_layout()
    out = Path(out_dir) / "snr_per_ell.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    if show:
        plt.show()
    plt.close(fig)


def plot_snr_per_ell_colorbar(base, models, cmb_exp, out_dir, show):
    """1×N grid (one panel per model) of per-ell SNR vs ell, coloured by z_s.

    Both Euclid (solid) and LSST (dashed) are overlaid in each panel for the
    chosen CMB experiment.  A shared colorbar shows the source redshift.
    """
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True, squeeze=False)
    axes = axes[0]

    z_grid = np.round(np.arange(0.5, 3.05, 0.1), 1)
    cmap = cm.viridis
    norm = mcolors.Normalize(vmin=z_grid.min(), vmax=z_grid.max())

    for ax, model in zip(axes, models):
        for z in z_grid:
            color = cmap(norm(z))
            for survey, ls in [("Euclid", "-"), ("LSST", "--")]:
                combo = f"{survey}_{cmb_exp}"
                snr_path = base / model / "SNRs" / combo / f"SNR_z={z}.npy"
                ell_path = base / model / "C_ells" / f"ell_grid_z={z}.npy"
                if not snr_path.exists() or not ell_path.exists():
                    continue
                snr_ell = np.load(snr_path)
                ell = np.load(ell_path)
                ax.loglog(ell, snr_ell, ls=ls, color=color, alpha=0.8)

        ax.set_title(MODEL_LABELS[model])
        ax.set_xlabel(r"Multipole $\ell$")
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(True)
        ax.tick_params(which="both", top=True, right=True,
                       labeltop=False, labelright=False)

    axes[0].set_ylabel(r"SNR$_\ell$")

    survey_handles = [
        plt.Line2D([0], [0], ls="-", color="0.3", label="Euclid"),
        plt.Line2D([0], [0], ls="--", color="0.3", label="LSST"),
    ]
    axes[0].legend(handles=survey_handles, loc="upper left", title="Survey")

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist())
    cbar.set_label(r"Source redshift $z_s$")

    fig.suptitle(rf"SNR$_\ell$ — survey $\times$ {cmb_exp}", y=1.02)
    out = Path(out_dir) / f"SNR_colorbar_survey_X_{cmb_exp}.pdf"
    fig.savefig(out, bbox_inches="tight")
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

    if "cumulative" in args.only:
        plot_cumulative_snr(base, args.models, out_dir, args.show)
    if "per-ell" in args.only:
        plot_snr_per_ell(base, args.models, args.z_ref, out_dir, args.show)
    if "colorbar" in args.only:
        for exp in args.colorbar_experiments:
            plot_snr_per_ell_colorbar(base, args.models, exp, out_dir, args.show)


if __name__ == "__main__":
    main()
