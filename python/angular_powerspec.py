"""
The following script is to construct angular power spectra from fields.
The data should be in .npy format, containing the positions and velocities
of the dark matter particles. All of this is using the Pylians3 library.
"""

import argparse, json
import numpy as np

from pathlib import Path
from scipy.interpolate import RegularGridInterpolator, interp1d

import vp_utils as utils

parameters_sim = utils.parameters_sim

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute angular power spectrum from a snapshot using Pylians3."
    )
    parser.add_argument("--in-dir", required=True, help="Path containing .npy files and metadata.")
    parser.add_argument("--out-dir", default="outputs", help="Output directory for .npy files.")
    return parser.parse_args()


def _load_metadata(in_dir):
    meta_path = Path(in_dir) / "snapshot_metadata.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text())


def main():
    args = parse_args()
    in_dir = Path(args.in_dir)

    matter_powerspec_path = in_dir / "Pk_m.npy"
    if not matter_powerspec_path.exists():
        raise FileNotFoundError(f"Missing {matter_powerspec_path}")

    curl_powerspec_path = in_dir / "Pk_curl.npy"
    if not curl_powerspec_path.exists():
        raise FileNotFoundError(f"Missing {curl_powerspec_path}")

    k_m = np.load(in_dir / "k_m.npy")
    Pk_m = np.load(in_dir / "Pk_m.npy")
    k_curl = np.load(in_dir / "k_curl.npy")
    Pk_curl = np.load(in_dir / "Pk_curl.npy")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = _load_metadata(in_dir)
    z = meta.get("redshift")

    factor = 6*parameters_sim['H0']**2*parameters_sim['Omega_m']*(1+z) 

    Pk_B = Pk_curl*factor**2 / k_curl**6

    Pk_matter_int = interp1d(np.log(k_m), np.log(Pk_m), kind='cubic')
    Pk_B_int = interp1d(np.log(k_curl), np.log(Pk_B), kind='cubic')

    def Pk_matter_interp(k):
        return np.exp(Pk_matter_int(np.log(k)))

    def Pk_q_interp(k):
        return np.exp(Pk_B_int(np.log(k)))*k**4


    ell_grid = np.arange(int(1e2), int(1e4), step=49)

    C_ells_XY = {}

    C_ells_XY['Phi'] = np.zeros_like(ell_grid, dtype=float)
    C_ells_XY['kSZ'] = np.zeros_like(ell_grid, dtype=float)
    C_ells_XY['B'] = np.zeros_like(ell_grid, dtype=float)
    C_ells_XY['B_X_kSZ'] = np.zeros_like(ell_grid, dtype=float)

    for idx, ell in enumerate(ell_grid):
        C_ells_XY['Phi'][idx] = utils.C_ell_XY(
            z_s=z,
            ell=ell,
            z_min=1e-5,
            kmin=k_m[0],
            kmax=k_m[-1],
            Pk=Pk_matter_interp,
            type_XY='Phi',
            Pk_evol=False,
        )

        C_ells_XY['kSZ'][idx] = utils.C_ell_XY(
            z_s=z,
            ell=ell,
            z_min=1e-5,
            kmin=k_curl[0],
            kmax=k_curl[-1],
            Pk=Pk_q_interp,
            type_XY='kSZ',
            Pk_evol=False,
        )

        C_ells_XY['B'][idx] = utils.C_ell_XY(
            z_s=z,
            ell=ell,
            z_min=1e-5,
            kmin=k_curl[0],
            kmax=k_curl[-1],
            Pk=Pk_q_interp,
            type_XY='B',
            Pk_evol=False,
        )

        C_ells_XY['B_X_kSZ'][idx] = utils.C_ell_XY(
            z_s=z,
            ell=ell,
            z_min=1e-5,
            kmin=k_curl[0],
            kmax=k_curl[-1],
            Pk=Pk_q_interp,
            type_XY='B_X_kSZ',
            Pk_evol=False,
        )
    
    np.save(out / "ell_grid.npy", ell_grid)
    np.save(out / "C_ells_XY.npy", C_ells_XY)
    
if __name__ == "__main__":
    main()