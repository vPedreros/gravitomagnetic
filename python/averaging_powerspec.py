"""
Averages the matter and momemtum curl power spectra over two independent
simulation seeds to reduce the effect of cosmic variance.

Each model (lcdm, frhs, ndgp) is run from two different (oppossite) sets of initial
conditions (seeds) to reduce cosmic variance.  This script reads the
per-seed Pk files produced by powerspec.py and writes their mean into
the model-level Pk_matter/ and Pk_curl/ directories which are later used by
angular_powerspec_z.py.

Expected input layout (one entry per snapshot):

    <base-dir>/<model>/<seed1>/snap_NNN/Pk_m.npy
    <base-dir>/<model>/<seed1>/snap_NNN/Pk_curl.npy
    <base-dir>/<model>/<seed1>/snap_NNN/k_m.npy
    <base-dir>/<model>/<seed1>/snap_NNN/k_curl.npy
    <base-dir>/<model>/<seed1>/snap_NNN/snapshot_metadata.json
    (same structure for <seed2>)

Output (overwritten on each run):

    <base-dir>/<model>/Pk_matter/NNN.npy   -- dict {Pk, k, z}
    <base-dir>/<model>/Pk_curl/NNN.npy     -- dict {Pcurl, k, z}

Both k and Pk are stored in physical (no-h) units: k in Mpc^-1,
Pk_matter in Mpc^3, Pk_curl in Mpc^3 (km/s)^2.  The unit conversion
from Pylians h-units is performed by powerspec.py using the h value
read from the model's parameters-usedvalues file (via VP_PARAMS_FILE).

h-correction flag
-----------------
Optional: If powerspec.py was run with the wrong parameters file (wrong h), the
stored k and Pk values are off by factors of h_wrong/h_correct and
(h_wrong/h_correct)^3 respectively. Pass --h-orig and --h-correct to
re-scale during averaging without needing to reprocess the raw snapshots:

    k_out  = k_stored  * (h_correct / h_orig)
    Pk_out = Pk_stored / (h_correct / h_orig)^3

Typical usage (examples)
-------------
1. Normal run (seeds share the same cosmological node):

    python averaging_powerspec.py --models lcdm frhs ndgp

2. Re-averaging MG models from node-037 seeds with h correction:

    python averaging_powerspec.py \\
        --models frhs ndgp \\
        --seed1 seed_2080_node39 --seed2 seed_4257_node39 \\
        --h-orig 0.78052 --h-correct 0.673
"""

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Average power spectra across two simulation seeds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base-dir", default="output", help="Base output directory.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lcdm", "frhs", "ndgp"],
        choices=["lcdm", "frhs", "ndgp"],
    )
    parser.add_argument(
        "--seed1", default="seed_2080", help="Name of the first seed subdirectory."
    )
    parser.add_argument(
        "--seed2", default="seed_4257", help="Name of the second seed subdirectory."
    )
    parser.add_argument(
        "--h-orig",
        type=float,
        default=None,
        help=(
            "h value that was used when the per-seed Pk files were computed "
            "by powerspec.py.  Supply together with --h-correct to re-scale "
            "k and Pk to the correct units without reprocessing raw snapshots."
        ),
    )
    parser.add_argument(
        "--h-correct",
        type=float,
        default=None,
        help="True h for this cosmological node (used as target for the rescaling).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_path = Path(args.base_dir)

    for m in args.models:
        path_1 = base_path / m / args.seed1
        path_2 = base_path / m / args.seed2

        if not path_1.is_dir():
            raise ValueError("path_1 not a directory")

        if not path_2.is_dir():
            raise ValueError("path_2 not a directory")

        # Discover snapshot count from the seed directory instead of hardcoding.
        nsnaps = len(sorted(path_1.glob("snap_*")))

        # h-correction: only active when both flags are provided.
        # Needed when powerspec.py was run with the wrong parameters file so
        # that the stored k/Pk are in units of a different h than intended.
        apply_h_correction = args.h_orig is not None and args.h_correct is not None
        if apply_h_correction:
            h_ratio = args.h_correct / args.h_orig
            print(
                f"computing {m} ({args.seed1} + {args.seed2}), h correction {args.h_orig} → {args.h_correct}"
            )
        else:
            h_ratio = None
            print(f"computing {m} ({args.seed1} + {args.seed2})")

        out_pk = base_path / m / "Pk_matter"
        out_pcurl = base_path / m / "Pk_curl"

        out_pk.mkdir(parents=True, exist_ok=True)
        out_pcurl.mkdir(parents=True, exist_ok=True)

        for i in range(nsnaps):
            path_snap1 = path_1 / f"snap_{i:03d}"
            path_snap2 = path_2 / f"snap_{i:03d}"

            pkm_1 = np.load(path_snap1 / "Pk_m.npy")
            pkq_1 = np.load(path_snap1 / "Pk_curl.npy")
            km_1 = np.load(path_snap1 / "k_m.npy")
            kq_1 = np.load(path_snap1 / "k_curl.npy")

            pkm_2 = np.load(path_snap2 / "Pk_m.npy")
            pkq_2 = np.load(path_snap2 / "Pk_curl.npy")
            km_2 = np.load(path_snap2 / "k_m.npy")
            kq_2 = np.load(path_snap2 / "k_curl.npy")

            # Both seeds must share the same k grid (same box, same N_grid).
            np.testing.assert_allclose(km_1, km_2, rtol=1e-5)
            np.testing.assert_allclose(kq_1, kq_2, rtol=1e-5)

            Pk = np.mean([pkm_1, pkm_2], axis=0)
            Pcurl = np.mean([pkq_1, pkq_2], axis=0)

            if apply_h_correction:
                # Re-scale from h_orig units to h_correct units.
                # powerspec.py stores  k  = k_Pylians * h  and  P = P_Pylians / h^3.
                # If the wrong h was used we undo and redo:
                #   k_out = k_stored * (h_correct / h_orig)
                #   P_out = P_stored / (h_correct / h_orig)^3
                km_1 = km_1 * h_ratio
                kq_1 = kq_1 * h_ratio
                Pk = Pk / h_ratio**3
                Pcurl = Pcurl / h_ratio**3

            with open(path_snap1 / "snapshot_metadata.json") as f:
                meta = json.load(f)

            z = meta["redshift"]

            data_m = {"Pk": Pk, "k": km_1, "z": z}

            data_curl = {"Pcurl": Pcurl, "k": kq_1, "z": z}

            np.save(out_pcurl / f"{i:03d}.npy", data_curl)
            np.save(out_pk / f"{i:03d}.npy", data_m)


if __name__ == "__main__":
    main()
