"""
The following script is to construct power spectra from particle data.
The data should be in .npy format, containing the positions and velocities
of the dark matter particles. All of this is using the Pylians3 library.
"""

import argparse, read_snap, h5py, json
import numpy as np

from classy import Class
from pathlib import Path


from pylab import *
import MAS_library as MASL
import Pk_library as PKL

import vp_utils as utils


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute power spectrum from a snapshot using Pylians3."
    )
    parser.add_argument("--base-path", required=True, help="Path containing .npy files and metadata.")
    parser.add_argument("--mas", default="CIC", help="Mass assignment scheme.")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads for Pk computation.")
    parser.add_argument("--out-dir", default="outputs", help="Output directory for .npy files.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose Pylians output.")
    return parser.parse_args()


def _load_metadata(in_dir):
    meta_path = Path(in_dir) / "snapshot_metadata.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text())


def main():
    args = parse_args()
    data = np.load()
    meta = _load_metadata(args.base_path)

    box_size = meta.get("box_size")
    if not box_size:
        raise ValueError("BoxSize not available in metadata.")
    
    pos = data["pos"].astype(np.float32, copy=False)
    vel = data["vel"].astype(np.float32, copy=False)

    rho = np.zeros((args.ngrid, args.ngrid, args.ngrid), dtype=np.float32)
    MASL.MA(pos, rho, box_size, args.mas, verbose=args.verbose)
    delta = rho / np.mean(rho, dtype=np.float32) - 1.0

    pk = PKL.Pk(delta, box_size, 0, args.mas, args.threads, args.verbose)

    k_m = pk.k3D
    Pk_m = pk.Pk[:,0]

    rho_momentum = np.zeros((args.ngrid, args.ngrid, args.ngrid), dtype=np.float32)
    qx = np.zeros_like(rho_momentum)
    qy = np.zeros_like(rho_momentum)
    qz = np.zeros_like(rho_momentum)

    MASL.MA(pos, qx, box_size, args.mas, vel[:,0].astype(np.float32), True)
    qx /= np.mean(rho_momentum)

    MASL.MA(pos, qy, box_size, args.mas, vel[:,1].astype(np.float32), True)
    qy /= np.mean(rho_momentum)

    MASL.MA(pos, qz, box_size, args.mas, vel[:,2].astype(np.float32), True)
    qz /= np.mean(rho_momentum)

    k_curl, Pk_curl, _ = PKL.Pk_curl(qx, qy, qz, args.ngrid, args.mas, args.sthreads)


    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    np.save(out / "k_m.npy", k_m)
    np.save(out / "Pk_m.npy", Pk_m)

    np.save(out / "k_curl.npy", k_curl)
    np.save(out / "Pk_curl.npy", Pk_curl)
    
    print(f"Saved k and Pk to: {out}")

if __name__ == "__main__":
    main()