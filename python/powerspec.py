"""
The following script is to construct power spectra from fields.
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
    parser.add_argument("--in-dir", required=True, help="Path containing .npy files and metadata.")
    parser.add_argument("--ngrid", default=1024, help="Dimension of the grid.")
    parser.add_argument("--mas", default="CIC", help="Mass assignment scheme.")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads for Pk computation.")
    parser.add_argument("--out-dir", default="outputs", help="Output directory for .npy files.")
    parser.add_argument("--verbose", type=bool, help="Enable verbose Pylians output.")
    return parser.parse_args()


def _load_metadata(in_dir):
    meta_path = Path(in_dir) / "snapshot_metadata.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text())


def main():
    args = parse_args()
    in_dir = Path(args.in_dir)

    delta_path = in_dir / "delta.npy"
    if not delta_path.exists():
        raise FileNotFoundError(f"Missing {delta_path}")

    qx_path = in_dir / "momentum_x.npy"
    if not qx_path.exists():
        raise FileNotFoundError(f"Missing {qx_path}")

    qy_path = in_dir / "momentum_y.npy"
    if not qy_path.exists():
        raise FileNotFoundError(f"Missing {qy_path}")

    qz_path = in_dir / "momentum_z.npy"
    if not qz_path.exists():
        raise FileNotFoundError(f"Missing {qz_path}")
    
    meta = _load_metadata(in_dir)
    box_size = meta.get("box_size")

    print("Loading fields...")
    delta = np.load(delta_path).astype(np.float32, copy=False)
    qx = np.load(qx_path).astype(np.float32, copy=False)
    qy = np.load(qy_path).astype(np.float32, copy=False)
    qz = np.load(qz_path).astype(np.float32, copy=False)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    Pk = PKL.Pk(delta, args.ngrid, 0, args.mas, args.threads, args.verbose)
    k_m, Pk_m = Pk.k3D, Pk.Pk[:,0] #monopole

    k_curl, Pk_curl, _ = PKL.Pk_curl(qx, qy, qz, args.ngrid, args.mas, args.threads, cross_terms=False)
    
    np.save(out / "k_m.npy", k_m)
    np.save(out / "Pk_m.npy", Pk_m)
    np.save(out / "k_curl.npy", k_curl)
    np.save(out / "Pk_curl.npy", Pk_curl)

    print(f"Power spectra saved to: {out}")

if __name__ == "__main__":
    main()