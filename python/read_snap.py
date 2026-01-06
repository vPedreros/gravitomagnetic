"""
The following script is to read the data from a snapshot file.
It takes a path to the snapshot file as input, and outputs the
positions and velocities of the dark matter particles.
"""


import h5py, argparse, json
import numpy as np
from pathlib import Path


def find_path(base_path, snap_num):
    """
    Function to find the snapshot directory.
    """ 
    base = Path(base_path)
    if base.is_dir() and base.name.startswith("snapdir_"):
        return base
    return base / f"snapdir_{snap_num:03d}"


def _json_ready(value):
    if isinstance(value, (np.integer, np.int64)):
        return int(value)
    if isinstance(value, (np.floating, np.float64)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    return value


def list_snapshot_files(snapdir, snap_num, num_files=None):
    """
    Function to list snapshot files inside a directory
    """

    snapdir = Path(snapdir)
    if num_files is not None:
        files = [snapdir / f"snap_{snap_num:03d}.{i}.hdf5" for i in range(num_files)]
    else:
        files = sorted(snapdir.glob(f"snap_{snap_num:03d}.*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No snapshot files in {snapdir}, for snapshot {snap_num:03d}")
    missing = [m for m in files if not m.exists()]
    if missing:
        raise FileNotFoundError(f"Missing snapshot files: {missing}")
    return files



def read_snap(base_path, snap_num, part_type="PartType1"):
    """
    Function that reads the snapshot files and returns the positions and velocities of the particles.
    """
    snapdir = find_path(base_path, snap_num)

    header_path = snapdir / f"snap_{snap_num:03d}.0.hdf5"

    if not header_path.exists():
        raise FileNotFoundError(f"Header not found: {header_path}")
    
    with h5py.File(header_path, "r") as f0:
        header = dict(f0["Header"].attrs)
        num_files = int(header.get("NumFilesPerSnapshot", 0)) or None
        box_size = header.get("BoxSize", 0.)
        redshift = header.get("Redshift", None)

    files = list_snapshot_files(snapdir, snap_num, num_files=num_files)

    total_part = 0
    for fp in files:
        with h5py.File(fp, "r") as f:
            if part_type not in f:
                raise KeyError(f"{part_type} not found in {fp}")
            total_part += f[part_type]["Coordinates"].shape[0]

    pos = np.empty((total_part, 3), dtype=np.float32)
    vel = np.empty((total_part, 3), dtype=np.float32)

    offset = 0
    for fp in files:
        with h5py.File(fp, "r") as f:
            group = f[part_type]
            n = group["Coordinates"].shape[0]
            if "Coordinates" in group:
                coords = group["Coordinates"][:]
            elif "IntegerCoordinates" in group:
                coords_int = group["IntegerCoordinates"][:]
                coords = coords_int /(2.0**32) * box_size
            else:
                raise KeyError(f"Coordinates not found in {fp}")
            vels = group["Velocities"][:]

            pos[offset:offset+n] = coords
            vel[offset:offset+n] = vels
            offset +=n

    return {
        "path": base_path,
        "snap_num": snap_num,
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
    data = read_snap(args.base_path, args.snap_num, part_type=args.part_type)
    
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    header = data.get("header", {})
    meta = {
        "snap_num": args.snap_num,
        "part_type": args.part_type,
        "box_size": data.get("box_size"),
        "redshift": data.get("redshift"),
        "num_files": header.get("NumFilesPerSnapshot"),
        "num_particles": int(data["pos"].shape[0]),
    }

    meta = {k: _json_ready(v) for k, v in meta.items()}
    meta_path = out / "snapshot_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))

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
