"""
The following script is to read the data from a snapshot file.
It takes a path to the snapshot file as input, and outputs the
positions and velocities of the dark matter particles.
"""


import h5py, argparse, json
import numpy as np
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read snapshot particles and save Coordinates/Velocities arrays."
    )
    parser.add_argument("--base-path", required=True, help="Path containing snapdir_### or the snapdir itself.")
    parser.add_argument("--snap-num", type=int, required=True, help="Snapshot number (e.g. 0).")
    parser.add_argument("--part-type", default="PartType1", help="Particle group to read.")
    parser.add_argument("--out-dir", default="outputs", help="Output directory for .npy files.")
    return parser.parse_args()


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


def read_snap(base_path, snap_num, part_type="PartType1", N=8):
    """
    Function that reads the snapshot files and returns the
    positions and velocities of the particles.
    N is the number of divisions for a snapshot
    """
    snapdir = find_path(base_path, snap_num)

    print('Reading snapshot from %s'%snapdir)
    
    nparts = 0
    for i in range(N):
        with h5py.File(snapdir.name +"/snap_%03d.%i.hdf5"%(snap_num,i), "r") as f:
            nparts += f[part_type]['Coordinates'].shape[0]
            box_size = f['Header'].attrs['BoxSize']
            redshift = f['Header'].attrs['Redshift']

    print("Found DM particles:", nparts)

    # Allocate big arrays up front
    pos = np.empty((nparts, 3), dtype=np.float32)
    vel = np.empty((nparts, 3), dtype=np.float32)

    # Second pass: fill them
    offset = 0
    for i in range(N):
        with h5py.File(snapdir.name +"/snap_%03d.%i.hdf5"%(snap_num,i), "r") as f:
            coords = f[part_type]['IntegerCoordinates'][:] / pow(2,32) * box_size
            vels = f[part_type]['Velocities'][:]

            n = coords.shape[0]
            pos[offset:offset+n] = coords
            vel[offset:offset+n] = vels
            offset += n

    return {
        "path": base_path,
        "snap_num": snap_num,
        "pos": pos,
        "vel": vel,
        "box_size": box_size,
        "redshift": redshift,
    }


def main():
    args = parse_args()
    data = read_snap(args.base_path, args.snap_num, part_type=args.part_type)
    
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    meta = {
        "snap_num": args.snap_num,
        "part_type": args.part_type,
        "box_size": data.get("box_size"),
        "redshift": data.get("redshift"),
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
