import numpy as np
from pathlib import Path
import h5py


def _resolve_snapdir(base_path, snap_num):
    base = Path(base_path)
    if base.is_dir() and base.name.startswith("snapdir_"):
        return base
    return base / f"snapdir_{snap_num:03d}"


def _list_snapshot_files(snapdir, snap_num):
    snapdir = Path(snapdir)
    pattern = f"snap_{snap_num:03d}.*.hdf5"
    files = sorted(snapdir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No snapshot files found at {snapdir}/{pattern}")
    return files


def read_snapshot_particles(base_path, snap_num, part_type="PartType1", fields=("Coordinates", "Velocities")):
    """
    Read particle arrays from a Gadget-style HDF5 snapshot split into multiple files.

    Returns a dict with requested arrays plus header metadata.
    """
    snapdir = _resolve_snapdir(base_path, snap_num)
    files = _list_snapshot_files(snapdir, snap_num)

    # Inspect header from first file for metadata and fallback conversions.
    with h5py.File(files[0], "r") as f0:
        header = dict(f0["Header"].attrs)
        box_size = float(header.get("BoxSize", 0.0))

    # First pass: count particles.
    total = 0
    for fp in files:
        with h5py.File(fp, "r") as f:
            total += f[part_type][fields[0]].shape[0]

    out = {"header": header, "box_size": box_size}
    arrays = {name: np.empty((total, 3), dtype=np.float32) for name in fields}

    offset = 0
    for fp in files:
        with h5py.File(fp, "r") as f:
            n = f[part_type][fields[0]].shape[0]
            for name in fields:
                if name in f[part_type]:
                    arrays[name][offset:offset + n] = f[part_type][name][:]
                elif name == "Coordinates" and "IntegerCoordinates" in f[part_type]:
                    coords_int = f[part_type]["IntegerCoordinates"][:].astype(np.float64)
                    arrays[name][offset:offset + n] = (coords_int / (2.0**32) * box_size).astype(np.float32)
                else:
                    raise KeyError(f"{name} not found in {fp}")
            offset += n

    out.update({name.lower(): arr for name, arr in arrays.items()})
    return out


def _require_pylians():
    try:
        import MAS_library as MASL  # noqa: N812
        import Pk_library as PKL  # noqa: N812
    except ImportError as exc:
        raise ImportError("Pylians3 libraries (MAS_library, Pk_library) are required.") from exc
    return MASL, PKL


def density_field(pos, ngrid, box_size, mas="CIC", verbose=False):
    """
    Return (rho, delta) on a regular grid using Pylians MAS.
    """
    MASL, _ = _require_pylians()
    rho = np.zeros((ngrid, ngrid, ngrid), dtype=np.float32)
    MASL.MA(pos.astype(np.float32), rho, box_size, mas, verbose=verbose)
    delta = rho / np.mean(rho, dtype=np.float64) - 1.0
    return rho, delta


def momentum_field(pos, vel, ngrid, box_size, mas="CIC", rho=None, verbose=False):
    """
    Return momentum density q = (1 + delta) * v_avg using Pylians MAS weights.
    """
    MASL, _ = _require_pylians()
    if rho is None:
        rho, _ = density_field(pos, ngrid, box_size, mas=mas, verbose=verbose)

    qx = np.zeros_like(rho)
    qy = np.zeros_like(rho)
    qz = np.zeros_like(rho)

    MASL.MA(pos.astype(np.float32), qx, box_size, mas, vel[:, 0].astype(np.float32), verbose)
    MASL.MA(pos.astype(np.float32), qy, box_size, mas, vel[:, 1].astype(np.float32), verbose)
    MASL.MA(pos.astype(np.float32), qz, box_size, mas, vel[:, 2].astype(np.float32), verbose)

    mean_rho = np.mean(rho, dtype=np.float64)
    qx /= mean_rho
    qy /= mean_rho
    qz /= mean_rho
    return qx, qy, qz, rho


def velocity_field(pos, vel, ngrid, box_size, mas="CIC", rho=None, verbose=False, eps=1e-12):
    """
    Return the mass-weighted velocity field v_avg on a grid.
    """
    qx, qy, qz, rho = momentum_field(
        pos, vel, ngrid, box_size, mas=mas, rho=rho, verbose=verbose
    )
    mean_rho = np.mean(rho, dtype=np.float64)
    denom = np.maximum(rho, eps)
    vx = qx * mean_rho / denom
    vy = qy * mean_rho / denom
    vz = qz * mean_rho / denom
    return vx, vy, vz, rho


def matter_power_spectrum(delta, box_size, mas="CIC", threads=1, verbose=True):
    """
    Return k and Pk monopole for the matter power spectrum.
    """
    _, PKL = _require_pylians()
    pk = PKL.Pk(delta, box_size, 0, mas, threads, verbose)
    k = pk.k3D
    pk0 = pk.Pk[:, 0]
    return k, pk0, pk


def omega_power_spectrum(qx, qy, qz, ngrid, mas="CIC", threads=1, cross_terms=False):
    """
    Return k, Pk_omega and Nmodes for the curl power spectrum.
    """
    _, PKL = _require_pylians()
    k, pk_omega, nmodes = PKL.Pk_curl(
        qx, qy, qz, ngrid, MAS=mas, threads=threads, cross_terms=cross_terms
    )
    return k, pk_omega, nmodes


def snapshot_products(base_path, snap_num, ngrid, box_size=None, mas="CIC", verbose=False):
    """
    Convenience wrapper to return density, velocity, Pk_matter, and Pk_omega.
    """
    data = read_snapshot_particles(base_path, snap_num)
    pos = data["coordinates"]
    vel = data["velocities"]
    if box_size is None:
        box_size = data["box_size"]

    rho, delta = density_field(pos, ngrid, box_size, mas=mas, verbose=verbose)
    vx, vy, vz, _ = velocity_field(pos, vel, ngrid, box_size, mas=mas, rho=rho, verbose=verbose)
    k_m, pk_m, _ = matter_power_spectrum(delta, box_size, mas=mas, threads=1, verbose=verbose)
    qx, qy, qz, _ = momentum_field(pos, vel, ngrid, box_size, mas=mas, rho=rho, verbose=verbose)
    k_om, pk_om, nm = omega_power_spectrum(qx, qy, qz, ngrid, mas=mas, threads=1)

    return {
        "rho": rho,
        "delta": delta,
        "vx": vx,
        "vy": vy,
        "vz": vz,
        "k_m": k_m,
        "pk_m": pk_m,
        "k_omega": k_om,
        "pk_omega": pk_om,
        "nmodes_omega": nm,
    }


def save_products_npy(products, out_dir):
    """
    Save snapshot products to .npy files in out_dir.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "rho.npy", products["rho"])
    np.save(out / "delta.npy", products["delta"])
    np.save(out / "vx.npy", products["vx"])
    np.save(out / "vy.npy", products["vy"])
    np.save(out / "vz.npy", products["vz"])
    np.save(out / "k_m.npy", products["k_m"])
    np.save(out / "pk_m.npy", products["pk_m"])
    np.save(out / "k_omega.npy", products["k_omega"])
    np.save(out / "pk_omega.npy", products["pk_omega"])
    np.save(out / "nmodes_omega.npy", products["nmodes_omega"])
