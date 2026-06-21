"""
Plot matter and curl (momentum) power spectra from averaged simulation snapshots.

Produces two figures:
  1. P_m(k) and P_curl(k) at selected redshifts, all models overlaid
  2. Ratio P_m / P_m^LCDM and P_curl / P_curl^LCDM to highlight modified-gravity signatures

Usage:
  python plot_powerspectra.py --in-dir output --out-dir imgs
  python plot_powerspectra.py --in-dir output --out-dir imgs --redshifts 0 0.5 1 2 --show
"""

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from plot_utils import MODEL_LABELS, MODEL_LS, MODEL_COLORS, apply_style


def parse_args():
    parser = argparse.ArgumentParser(description="Plot matter and curl power spectra.")
    parser.add_argument(
        "--in-dir",
        required=True,
        help="Base data directory (contains lcdm/, frhs/, ndgp/).",
    )
    parser.add_argument(
        "--out-dir", default="imgs", help="Directory for saved figures."
    )
    parser.add_argument(
        "--redshifts",
        nargs="+",
        type=float,
        default=[0.0, 0.5, 1.0, 2.0],
        help="Target redshifts to plot (nearest available snapshot is used).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lcdm", "frhs", "ndgp"],
        choices=["lcdm", "frhs", "ndgp"],
        help="Models to include.",
    )
    parser.add_argument(
        "--show", action="store_true", help="Display figures interactively."
    )
    parser.add_argument(
        "--node", default="node_004", help="Number of the node"
    )
    return parser.parse_args()


def load_model_snapshots(base, model, node):
    """Return sorted list of dicts: [{z, k_m, Pk_m, k_curl, Pcurl}, ...]."""
    snaps = []
    pk_dir = base / model / node / "Pk_matter"
    pc_dir = base / model / node / "Pk_curl"
    for f in sorted(pk_dir.glob("*.npy")):
        dm = np.load(f, allow_pickle=True).item()
        dq = np.load(pc_dir / f.name, allow_pickle=True).item()
        snaps.append(
            {
                "z": float(dm["z"]),
                "k_m": dm["k"],
                "Pk_m": dm["Pk"],
                "k_curl": dq["k"],
                "Pcurl": dq["Pcurl"],
            }
        )
    return sorted(snaps, key=lambda s: s["z"])


def nearest_snapshot(snaps, z_target):
    """Return the snapshot closest to z_target."""
    return min(snaps, key=lambda s: abs(s["z"] - z_target))


def redshift_cmap(z_values):
    """Map a list of redshifts to colours from a perceptually-uniform colormap."""
    norm = mcolors.Normalize(vmin=min(z_values), vmax=max(z_values))
    cmap = plt.cm.plasma_r
    return {z: cmap(norm(z)) for z in z_values}


def plot_absolute(data, z_targets, models, out_dir, show):
    """Figure 1: P_m(k) and P_curl(k) for each model at each target redshift."""
    z_colors = redshift_cmap(z_targets)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=False)
    ax_m, ax_curl = axes

    # Track what goes in the legend
    model_handles, z_handles = [], []

    for model in models:
        snaps = data[model]
        for z_target in z_targets:
            snap = nearest_snapshot(snaps, z_target)
            z_used = snap["z"]

            color = z_colors[z_target]
            ls = MODEL_LS[model]

            h_m = ax_m.loglog(snap["k_m"], snap["Pk_m"], ls=ls, color=color, alpha=0.9)[
                0
            ]
            ax_curl.loglog(snap["k_curl"], snap["Pcurl"], ls=ls, color=color, alpha=0.9)

            # Collect proxy handles once per model and once per z_target
            if z_target == z_targets[0]:
                model_handles.append(
                    plt.Line2D([0], [0], ls=ls, color="0.3", label=MODEL_LABELS[model])
                )
        if model == models[0]:
            for z_target in z_targets:
                snap = nearest_snapshot(snaps, z_target)
                z_handles.append(
                    plt.Line2D(
                        [0],
                        [0],
                        ls="-",
                        color=z_colors[z_target],
                        label=rf"$z \approx {snap['z']:.2f}$",
                    )
                )

    ax_m.set_xlabel(r"$k$ (Mpc$^{-1}$)")
    ax_m.set_ylabel(r"$P_\mathrm{m}(k)$ (Mpc$^3$)")
    ax_m.set_title("Matter power spectrum")

    ax_curl.set_xlabel(r"$k$ (Mpc$^{-1}$)")
    ax_curl.set_ylabel(r"$P_\mathrm{curl}(k)$ (Mpc$^3$ km$^2$ s$^{-2}$)")
    ax_curl.set_title("Curl (momentum) power spectrum")

    # Combined legend: models (linestyle) + redshifts (colour)
    leg1 = ax_m.legend(handles=model_handles, loc="lower left", title="Model")
    ax_m.add_artist(leg1)
    ax_curl.legend(handles=z_handles, loc="lower left", title="Redshift")

    plt.tight_layout()
    out = Path(out_dir) / "powerspectra_absolute.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    if show:
        plt.show()
    plt.close(fig)


def plot_ratio(data, z_targets, models, out_dir, show):
    """Figure 2: P(k) / P_LCDM(k) for matter and curl at each target redshift."""
    if "lcdm" not in models:
        print("Skipping ratio plot — lcdm not in selected models.")
        return

    z_colors = redshift_cmap(z_targets)
    other_models = [m for m in models if m != "lcdm"]
    if not other_models:
        print("Skipping ratio plot — no non-LCDM models selected.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=False)
    ax_m, ax_curl = axes

    for m in axes:
        m.axhline(1.0, color="0.5", lw=0.8, ls="--")

    model_handles = []
    z_handles = []

    for model in other_models:
        snaps = data[model]
        lcdm_snaps = data["lcdm"]

        for z_target in z_targets:
            snap = nearest_snapshot(snaps, z_target)
            snap_lcdm = nearest_snapshot(lcdm_snaps, z_target)
            color = z_colors[z_target]
            ls = MODEL_LS[model]

            # Restrict to common k range to avoid np.interp clipping artifacts
            k_lo_m = max(snap["k_m"][0], snap_lcdm["k_m"][0])
            k_hi_m = min(snap["k_m"][-1], snap_lcdm["k_m"][-1])
            mask_m = (snap["k_m"] >= k_lo_m) & (snap["k_m"] <= k_hi_m)
            k_m = snap["k_m"][mask_m]
            ratio_m = snap["Pk_m"][mask_m] / np.interp(
                k_m, snap_lcdm["k_m"], snap_lcdm["Pk_m"]
            )

            k_lo_c = max(snap["k_curl"][0], snap_lcdm["k_curl"][0])
            k_hi_c = min(snap["k_curl"][-1], snap_lcdm["k_curl"][-1])
            mask_c = (snap["k_curl"] >= k_lo_c) & (snap["k_curl"] <= k_hi_c)
            k_c = snap["k_curl"][mask_c]
            ratio_c = snap["Pcurl"][mask_c] / np.interp(
                k_c, snap_lcdm["k_curl"], snap_lcdm["Pcurl"]
            )

            ax_m.semilogx(k_m, ratio_m, ls=ls, color=color, alpha=0.9)
            ax_curl.semilogx(k_c, ratio_c, ls=ls, color=color, alpha=0.9)

            if z_target == z_targets[0]:
                model_handles.append(
                    plt.Line2D([0], [0], ls=ls, color="0.3", label=MODEL_LABELS[model])
                )
        if model == other_models[0]:
            for z_target in z_targets:
                snap = nearest_snapshot(data["lcdm"], z_target)
                z_handles.append(
                    plt.Line2D(
                        [0],
                        [0],
                        ls="-",
                        color=z_colors[z_target],
                        label=rf"$z \approx {snap['z']:.2f}$",
                    )
                )

    ax_m.set_xlabel(r"$k$ (Mpc$^{-1}$)")
    ax_m.set_ylabel(r"$P_\mathrm{m}(k) \,/\, P_\mathrm{m}^\mathrm{\Lambda CDM}(k)$")
    ax_m.set_title("Matter power spectrum ratio")

    ax_curl.set_xlabel(r"$k$ (Mpc$^{-1}$)")
    ax_curl.set_ylabel(
        r"$P_\mathrm{curl}(k) \,/\, P_\mathrm{curl}^\mathrm{\Lambda CDM}(k)$"
    )
    ax_curl.set_title("Curl power spectrum ratio")

    leg1 = ax_m.legend(handles=model_handles, loc="upper left", title="Model")
    ax_m.add_artist(leg1)
    ax_curl.legend(handles=z_handles, loc="upper left", title="Redshift")

    plt.tight_layout()
    out = Path(out_dir) / "powerspectra_ratio.pdf"
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
    (out_dir).mkdir(parents=True, exist_ok=True)
    out_dir = out_dir

    print("Loading snapshots...")
    data = {m: load_model_snapshots(base, m, args.node) for m in args.models}

    z_targets = sorted(args.redshifts)

    plot_absolute(data, z_targets, args.models, out_dir, args.show)
    plot_ratio(data, z_targets, args.models, out_dir, args.show)


if __name__ == "__main__":
    main()
