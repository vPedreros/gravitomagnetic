"""
Compute matter and curl power spectra from CIC grid fields using Pylians3.

Reads the density contrast (delta) and momentum fields (qx, qy, qz) written
by fields.py for a single snapshot, computes the 3-D power spectra with
Pylians3, and writes k and Pk arrays in physical (no-h) units.

Unit convention
---------------
Pylians3's PKL.Pk and PKL.Pk_curl return k in h/Mpc and P in (Mpc/h)^3
when box_size is supplied in Mpc/h (as stored in snapshot_metadata.json).

This script converts to physical units before saving using the relations:

    k  [Mpc^-1]   = k_Pylians [h/Mpc]       * h
    Pk [Mpc^3]    = Pk_Pylians [(Mpc/h)^3]   / h^3

The value of h is read from the cosmological parameters file via the
VP_PARAMS_FILE environment variable (see vp_utils.py).  It is critical
that VP_PARAMS_FILE points to the parameters-usedvalues file of the
model currently being processed.  Using the wrong h shifts the k grid
by a factor h_wrong/h_correct and suppresses Pk by (h_wrong/h_correct)^3.
If this mistake has already been made, the post-hoc correction in
averaging_powerspec.py (--h-orig / --h-correct) can recover the correct
units without reprocessing the raw fields.

Environment
-----------
VP_PARAMS_FILE : path to the Arepo parameters-usedvalues file for the
    model being processed.  Must be set before running this script.
    Example::

        export VP_PARAMS_FILE=output/frhs/parameters-usedvalues
        python powerspec.py --in-dir output/frhs/seed_2080/snap_000 \\
                            --out-dir output/frhs/seed_2080/snap_000
"""

import argparse
import json
from pathlib import Path

import numpy as np
import Pk_library as PKL
import vp_utils as utils

parameters_sim = utils.parameters_sim


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute power spectrum from a snapshot using Pylians3."
    )
    parser.add_argument(
        "--in-dir", required=True, help="Path containing .npy files and metadata."
    )
    parser.add_argument("--mas", default="CIC", help="Mass assignment scheme.")
    parser.add_argument(
        "--threads", type=int, default=1, help="Number of threads for Pk computation."
    )
    parser.add_argument(
        "--out-dir", default="outputs", help="Output directory for .npy files."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose Pylians output."
    )
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

    Pk = PKL.Pk(delta, box_size, 0, args.mas, args.threads, args.verbose)
    kh_m, Pkh_m = Pk.k3D, Pk.Pk[:, 0]  # monopole

    kh_curl, Pkh_curl, _ = PKL.Pk_curl(
        qx, qy, qz, box_size, args.mas, args.threads, cross_terms=False
    )

    # Trim to the reliable k range [kF, kN] in h/Mpc, then convert to Mpc^-1.
    # kF = fundamental mode (1/BoxSize), kN = Nyquist (N_grid/2 / BoxSize).
    # h comes from VP_PARAMS_FILE; using the wrong model's h here will
    # silently mis-scale all downstream k and Pk values.
    mask_m = (kh_m < parameters_sim["khN"]) & (kh_m > parameters_sim["khF"])
    mask_curl = (kh_curl < parameters_sim["khN"]) & (kh_curl > parameters_sim["khF"])
    h = parameters_sim["h"]

    k_m = kh_m[mask_m] * h  # h/Mpc  ->  Mpc^-1
    Pk_m = Pkh_m[mask_m] / h**3  # (Mpc/h)^3  ->  Mpc^3
    k_curl = kh_curl[mask_curl] * h
    Pk_curl = Pkh_curl[mask_curl] / h**3

    np.save(out / "k_m.npy", k_m)
    np.save(out / "Pk_m.npy", Pk_m)
    np.save(out / "k_curl.npy", k_curl)
    np.save(out / "Pk_curl.npy", Pk_curl)

    print(f"Power spectra saved to: {out}")


if __name__ == "__main__":
    main()
