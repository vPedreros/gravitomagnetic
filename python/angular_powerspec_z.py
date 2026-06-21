"""
The following script is to construct angular power spectra from fields.
The data should be in .npy format, containing the positions and velocities
of the dark matter particles. All of this is using the Pylians3 library.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import vp_utils as utils
from scipy.interpolate import RegularGridInterpolator, interp1d


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute angular power spectrum from a snapshot using Pylians3."
    )
    parser.add_argument(
        "--in-dir",
        type=str,
        required=True,
        help="Path containing .npy files and metadata.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="output",
        help="Output directory for .npy files.",
    )
    parser.add_argument(
        "--node", default="node_004", help="Number of the node"
    )
    parser.add_argument(
        "--z_source", type=float, required=True, help="Maximum redshift for integration"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    in_dir = Path(args.in_dir)
    in_dir = in_dir / args.node

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "C_ells").mkdir(parents=True, exist_ok=True)

    z_list = []
    Pkm_list = []
    Pk_curl_list = []
    km_list = []
    k_curl_list = []

    n_snaps = len(sorted((in_dir / "Pk_matter").glob("*.npy")))
    for i in range(n_snaps):
        data_m = np.load(in_dir / f"Pk_matter/{i:03d}.npy", allow_pickle=True).item()
        data_q = np.load(in_dir / f"Pk_curl/{i:03d}.npy", allow_pickle=True).item()

        km_list.append(data_m["k"])
        Pkm_list.append(data_m["Pk"])

        k_curl_list.append(data_q["k"])
        Pk_curl_list.append(data_q["Pcurl"])

        z_list.append(data_m["z"])

    # Sort redshift!
    z_order = np.argsort(z_list)
    z_grid = np.array(z_list)[z_order]
    Pkm_grid = np.array(Pkm_list)[z_order, :]
    Pk_curl_grid = np.array(Pk_curl_list)[z_order, :]

    for k in km_list[1:]:
        np.testing.assert_allclose(k, km_list[0])

    for k in k_curl_list[1:]:
        np.testing.assert_allclose(k, k_curl_list[0])

    k_m = km_list[0]
    k_curl = k_curl_list[0]

    logk_m = np.log(k_m)
    logk_c = np.log(k_curl)

    logPk_m_interp = RegularGridInterpolator(
        (z_grid, logk_m), np.log(Pkm_grid), bounds_error=False, fill_value=None
    )
    logPk_curl_interp = RegularGridInterpolator(
        (z_grid, logk_c), np.log(Pk_curl_grid), bounds_error=False, fill_value=None
    )

    def Pk_matter_interp(x):
        k, z = x
        k = np.asarray(k, float)
        z = np.asarray(z, float)
        pts = np.column_stack([z, np.log(k)])
        return np.exp(logPk_m_interp(pts))

    def Pk_q_interp(x):
        k, z = x
        k = np.asarray(k, float)
        z = np.asarray(z, float)
        pts = np.column_stack([z, np.log(k)])
        Pk_B = np.exp(logPk_curl_interp(pts))
        return Pk_B

    ell_grid = np.logspace(2, 4, 40)

    C_ells_XY = {}

    C_ells_XY["Phi"] = np.zeros_like(ell_grid, dtype=float)
    C_ells_XY["kSZ"] = np.zeros_like(ell_grid, dtype=float)
    C_ells_XY["B"] = np.zeros_like(ell_grid, dtype=float)
    C_ells_XY["B_X_kSZ"] = np.zeros_like(ell_grid, dtype=float)

    for idx, ell in enumerate(ell_grid):
        C_ells_XY["Phi"][idx] = utils.C_ell_XY(
            z_s=args.z_source,
            ell=ell,
            z_min=1e-5,
            kmin=k_m[0],
            kmax=k_m[-1],
            Pk=Pk_matter_interp,
            type_XY="Phi",
            Pk_evol=True,
        )

        C_ells_XY["kSZ"][idx] = utils.C_ell_XY(
            z_s=args.z_source,
            ell=ell,
            z_min=1e-5,
            kmin=k_curl[0],
            kmax=k_curl[-1],
            Pk=Pk_q_interp,
            type_XY="kSZ",
            Pk_evol=True,
        )

        C_ells_XY["B"][idx] = utils.C_ell_XY(
            z_s=args.z_source,
            ell=ell,
            z_min=1e-5,
            kmin=k_curl[0],
            kmax=k_curl[-1],
            Pk=Pk_q_interp,
            type_XY="B",
            Pk_evol=True,
        )

        C_ells_XY["B_X_kSZ"][idx] = utils.C_ell_XY(
            z_s=args.z_source,
            ell=ell,
            z_min=1e-5,
            kmin=k_curl[0],
            kmax=k_curl[-1],
            Pk=Pk_q_interp,
            type_XY="B_X_kSZ",
            Pk_evol=True,
        )

    np.save(out / f"C_ells/ell_grid_z={args.z_source}.npy", ell_grid)
    np.save(out / f"C_ells/C_ells_XY_z={args.z_source}.npy", C_ells_XY)


if __name__ == "__main__":
    main()

