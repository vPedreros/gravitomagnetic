"""
The following script is to construct fields from particle data.
The data should be in .npy format, containing the positions and velocities
of the dark matter particles. All of this is using the Pylians3 library.
"""

import argparse, json
import numpy as np

from pathlib import Path


from pylab import *
import MAS_library as MASL

import gc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute power spectrum from a snapshot using Pylians3."
    )
    parser.add_argument("--in-dir", required=True, help="Path containing .npy files and metadata.")
    parser.add_argument("--ngrid", type=int, default=1024, help="Dimension of the grid.")
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

    coords_path = in_dir / "Coordinates.npy"
    if not coords_path.exists():
        raise FileNotFoundError(f"Missing {coords_path}")

    vel_path = in_dir / "Velocities.npy"
    if not vel_path.exists():
        raise FileNotFoundError(f"Missing {vel_path}")
    
    meta = _load_metadata(in_dir)
    box_size = meta.get("box_size")

    print("Loading coordinates...")
    pos = np.load(coords_path).astype(np.float32)
    vel = np.load(vel_path).astype(np.float32, copy=False)
    
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rho = np.zeros((args.ngrid, args.ngrid, args.ngrid), dtype=np.float32)

    MASL.MA(pos, rho, box_size, args.mas, verbose=args.verbose)

    rho_mean = np.mean(rho, dtype=np.float32)
    delta = rho / rho_mean
    delta -= 1.0

    np.save(out / "delta.npy", delta)

    del delta, rho
    gc.collect()

    qx = np.zeros((args.ngrid, args.ngrid, args.ngrid), dtype=np.float32)
    qy = np.zeros((args.ngrid, args.ngrid, args.ngrid), dtype=np.float32)
    qz = np.zeros((args.ngrid, args.ngrid, args.ngrid), dtype=np.float32)

    MASL.MA(pos, qx, box_size, args.mas, vel[:,0].astype(np.float32), verbose=args.verbose)
    qx /= np.mean(rho_mean)

    np.save(out / "momentum_x.npy", qx)

    del qx
    gc.collect()

    MASL.MA(pos, qy, box_size, args.mas, vel[:,1].astype(np.float32), verbose=args.verbose)
    qy /= np.mean(rho_mean)

    np.save(out / "momentum_y.npy", qy)

    del qy
    gc.collect()

    MASL.MA(pos, qz, box_size, args.mas, vel[:,2].astype(np.float32), verbose=args.verbose)
    qz /= np.mean(rho_mean)

    np.save(out / "momentum_z.npy", qz)

    del qz
    gc.collect()

    print(f"Fields saved to: {out}")

if __name__ == "__main__":
    main()
