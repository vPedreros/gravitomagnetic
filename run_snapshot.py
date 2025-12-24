import argparse
from pathlib import Path

import snapshot_fields as sf


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute fields and power spectra from a snapshot and save .npy outputs."
    )
    parser.add_argument("--base-path", required=True, help="Path containing snapdir_###.")
    parser.add_argument("--snap-num", type=int, required=True, help="Snapshot number.")
    parser.add_argument("--ngrid", type=int, required=True, help="Grid size per dimension.")
    parser.add_argument("--box-size", type=float, default=None, help="Box size in Mpc/h.")
    parser.add_argument("--mas", default="CIC", help="Mass assignment scheme.")
    parser.add_argument("--out-dir", required=True, help="Output directory for .npy files.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose Pylians output.")
    return parser.parse_args()


def main():
    args = parse_args()
    products = sf.snapshot_products(
        base_path=args.base_path,
        snap_num=args.snap_num,
        ngrid=args.ngrid,
        box_size=args.box_size,
        mas=args.mas,
        verbose=args.verbose,
    )
    sf.save_products_npy(products, Path(args.out_dir))


if __name__ == "__main__":
    main()
