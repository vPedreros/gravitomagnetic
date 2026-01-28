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
    
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    z_list = []
    Pkm_list = []
    PkB_list = []

    for i in range(27):
        snap_dir = in_dir / f"snap_{i:03d}"
        
        meta = _load_metadata(snap_dir)
        z = float(str(meta.get("redshift")).strip())
        factor = 6*parameters_sim['H0']**2*parameters_sim['Omega_m']*(1+z) 

        k_m = np.load(snap_dir / "k_m.npy")
        Pk_m = np.load(snap_dir / "Pk_m.npy")
        k_curl = np.load(snap_dir / "k_curl.npy")
        Pk_curl = np.load(snap_dir / "Pk_curl.npy")

        Pk_B = Pk_curl*factor**2 / k_curl**6

        z_list.append(z)
        Pkm_list.append(Pk_m)
        PkB_list.append(Pk_B)

    # Sort redshift!
    z_order = np.argsort(z_list)
    z_grid = np.array(z_list)[z_order]
    Pkm_grid = np.array(Pkm_list)[z_order, :]
    PkB_grid = np.array(PkB_list)[z_order, :]

    logk_m = np.log(k_m)
    logk_c = np.log(k_curl)

    logPk_m_interp = RegularGridInterpolator(
        (z_grid, logk_m), np.log(Pkm_grid),
        bounds_error=False, fill_value=None
    )
    logPk_B_interp = RegularGridInterpolator(
        (z_grid, logk_c), np.log(PkB_grid),
        bounds_error=False, fill_value=None
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
        Pk_B = np.exp(logPk_B_interp(pts))
        return Pk_B * k**4


    ell_grid = np.arange(int(1e2), int(1e4), step=49)

    C_ells_XY = {}

    C_ells_XY['Phi'] = np.zeros_like(ell_grid, dtype=float)
    C_ells_XY['kSZ'] = np.zeros_like(ell_grid, dtype=float)
    C_ells_XY['B'] = np.zeros_like(ell_grid, dtype=float)
    C_ells_XY['B_X_kSZ'] = np.zeros_like(ell_grid, dtype=float)

    for idx, ell in enumerate(ell_grid):
        C_ells_XY['Phi'][idx] = utils.C_ell_XY(
            z_s=3,
            ell=ell,
            z_min=1e-5,
            kmin=k_m[0],
            kmax=k_m[-1],
            Pk=Pk_matter_interp,
            type_XY='Phi',
            Pk_evol=True,
        )

        C_ells_XY['kSZ'][idx] = utils.C_ell_XY(
            z_s=3,
            ell=ell,
            z_min=1e-5,
            kmin=k_curl[0],
            kmax=k_curl[-1],
            Pk=Pk_q_interp,
            type_XY='kSZ',
            Pk_evol=True,
        )

        C_ells_XY['B'][idx] = utils.C_ell_XY(
            z_s=3,
            ell=ell,
            z_min=1e-5,
            kmin=k_curl[0],
            kmax=k_curl[-1],
            Pk=Pk_q_interp,
            type_XY='B',
            Pk_evol=True,
        )

        C_ells_XY['B_X_kSZ'][idx] = utils.C_ell_XY(
            z_s=3,
            ell=ell,
            z_min=1e-5,
            kmin=k_curl[0],
            kmax=k_curl[-1],
            Pk=Pk_q_interp,
            type_XY='B_X_kSZ',
            Pk_evol=True,
        )
    
    np.save(out / "ell_grid.npy", ell_grid)
    np.save(out / "C_ells_XY.npy", C_ells_XY)
    
if __name__ == "__main__":
    main()