import argparse
from pathlib import Path

import h5py
import numpy as np


def _resolve_snapdir(base_path, snap_num):
    base = Path(base_path)
    if base.is_dir() and base.name.startswith("snapdir_"):
        return base
    return base / f"snapdir_{snap_num:03d}"


def _list_snapshot_files(snapdir, snap_num, num_files=None):
    snapdir = Path(snapdir)
    if num_files is not None:
        files = [snapdir / f"snap_{snap_num:03d}.{i}.hdf5" for i in range(num_files)]
    else:
        files = sorted(snapdir.glob(f"snap_{snap_num:03d}.*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No snapshot files found in {snapdir} for snap {snap_num:03d}")
    missing = [fp for fp in files if not fp.exists()]
    if missing:
        raise FileNotFoundError(f"Missing snapshot files: {missing}")
    return files


def read_snapshot_particles(base_path, snap_num, part_type="PartType1"):
    snapdir = _resolve_snapdir(base_path, snap_num)

    # Read header from the first file to determine split count and box size.
    header_path = snapdir / f"snap_{snap_num:03d}.0.hdf5"
    if not header_path.exists():
        raise FileNotFoundError(f"Header file not found: {header_path}")

    with h5py.File(header_path, "r") as f0:
        header = dict(f0["Header"].attrs)
        num_files = int(header.get("NumFilesPerSnapshot", 0)) or None
        box_size = float(header.get("BoxSize", 0.0))
        redshift = header.get("Redshift", None)

    files = _list_snapshot_files(snapdir, snap_num, num_files=num_files)

    # First pass: count particles.
    total = 0
    for fp in files:
        with h5py.File(fp, "r") as f:
            if part_type not in f:
                raise KeyError(f"{part_type} not found in {fp}")
            total += f[part_type]["Coordinates"].shape[0]

    pos = np.empty((total, 3), dtype=np.float32)
    vel = np.empty((total, 3), dtype=np.float32)

    # Second pass: fill arrays, handling IntegerCoordinates if needed.
    offset = 0
    for fp in files:
        with h5py.File(fp, "r") as f:
            group = f[part_type]
            n = group["Coordinates"].shape[0]
            if "Coordinates" in group:
                coords = group["Coordinates"][:]
            elif "IntegerCoordinates" in group:
                if box_size <= 0:
                    raise ValueError("BoxSize is required to convert IntegerCoordinates")
                coords_int = group["IntegerCoordinates"][:].astype(np.float64)
                coords = coords_int / (2.0**32) * box_size
            else:
                raise KeyError(f"Coordinates not found in {fp}")
            vels = group["Velocities"][:]

            pos[offset:offset + n] = coords
            vel[offset:offset + n] = vels
            offset += n

    return {
        "pos": pos,
        "vel": vel,
        "box_size": box_size,
        "redshift": redshift,
        "header": header,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read snapshot particles and save Coordinates/Velocities arrays."
    )
    parser.add_argument("--base-path", required=True, help="Path containing snapdir_### or the snapdir itself.")
    parser.add_argument("--snap-num", type=int, required=True, help="Snapshot number (e.g. 0).")
    parser.add_argument("--part-type", default="PartType1", help="Particle group to read.")
    parser.add_argument("--out-dir", default="outputs", help="Output directory for .npy files.")
    return parser.parse_args()


def main():
    args = parse_args()
    data = read_snapshot_particles(args.base_path, args.snap_num, part_type=args.part_type)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "Coordinates.npy", data["pos"])
    np.save(out / "Velocities.npy", data["vel"])

    print(f"Total particles: {data['pos'].shape[0]}")
    if data["redshift"] is not None:
        print(f"Redshift z={data['redshift']}")
    if data["box_size"]:
        print(f"BoxSize={data['box_size']}")
    print(f"Saved to: {out}")


if __name__ == "__main__":
    main()
