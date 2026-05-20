"""
Integration / smoke tests for angular_powerspec_z.py.

Creates a minimal synthetic dataset in a tmp directory and runs main() to verify
the output structure — in particular that C_ells/ is created (previously missing)
and snapshot count is discovered dynamically rather than hardcoded.

C_ell_XY is mocked to a constant to keep the test fast.
"""

import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import vp_utils  # already loaded by conftest with mocked params
import angular_powerspec_z as apz


def _write_synthetic_snapshot(pk_matter_dir, pk_curl_dir, idx, z, n_k=10):
    k = np.logspace(-2, 0, n_k)
    np.save(pk_matter_dir / f"{idx:03d}.npy", {"k": k, "Pk": 1e-2 * k**-2, "z": z})
    np.save(pk_curl_dir / f"{idx:03d}.npy", {"k": k, "Pcurl": 1e-3 * k**-2, "z": z})


@pytest.fixture
def synthetic_input(tmp_path):
    in_dir = tmp_path / "input"
    pk_m = in_dir / "Pk_matter"
    pk_c = in_dir / "Pk_curl"
    pk_m.mkdir(parents=True)
    pk_c.mkdir(parents=True)
    for i, z in enumerate([0.5, 1.0, 1.5]):
        _write_synthetic_snapshot(pk_m, pk_c, i, z)
    return in_dir


def test_c_ells_dir_created(synthetic_input, tmp_path):
    """C_ells/ subdirectory must be created before np.save is called."""
    out_dir = tmp_path / "output"

    fake_args = MagicMock()
    fake_args.in_dir = str(synthetic_input)
    fake_args.out_dir = str(out_dir)
    fake_args.z_source = 1.5

    with patch("angular_powerspec_z.parse_args", return_value=fake_args), \
         patch("vp_utils.C_ell_XY", return_value=1e-10):
        apz.main()

    assert (out_dir / "C_ells").is_dir(), "C_ells/ subdirectory must be created"
    assert (out_dir / "C_ells" / "ell_grid_z=1.5.npy").exists()
    assert (out_dir / "C_ells" / "C_ells_XY_z=1.5.npy").exists()


def test_snapshot_count_dynamic(synthetic_input, tmp_path):
    """main() should load exactly 3 snapshots — not a hardcoded 25."""
    out_dir = tmp_path / "output2"

    fake_args = MagicMock()
    fake_args.in_dir = str(synthetic_input)
    fake_args.out_dir = str(out_dir)
    fake_args.z_source = 1.5

    load_calls = []
    _orig_load = np.load

    def tracking_load(path, **kwargs):
        p = Path(str(path))
        if "Pk_matter" in str(p) or "Pk_curl" in str(p):
            load_calls.append(p.name)
        return _orig_load(path, **kwargs)

    with patch("angular_powerspec_z.parse_args", return_value=fake_args), \
         patch("vp_utils.C_ell_XY", return_value=1e-10), \
         patch("numpy.load", side_effect=tracking_load):
        apz.main()

    # 3 snapshots × 2 files (Pk_matter + Pk_curl) = 6 loads
    assert len(load_calls) == 6, f"Expected 6 file loads for 3 snapshots, got {len(load_calls)}"
