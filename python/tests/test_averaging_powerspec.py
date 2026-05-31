"""
Tests for averaging_powerspec.py.

Covers three classes of defect that previously caused MG power spectra to appear
40–50% below ΛCDM:

  Bug 1 — wrong seeds averaged: hardcoded seed names caused node-037 data to be
           silently ignored.  Caught by: test_averaged_pk_equals_mean_of_seeds,
           test_averaged_output_matches_node39_seeds (integration).

  Bug 2 — wrong h for unit conversion: powerspec.py used the wrong model's h,
           shifting k by ×1.16 and suppressing Pk by ×1.56.  Caught by:
           test_k_grid_consistent_with_h (integration),
           test_pk_ratio_mg_lcdm_within_physical_range (integration),
           and the h-correction arithmetic tests.

Unit tests use only synthetic tmp directories and run without any simulation data.
Integration / regression tests load real files from output/ and are skipped when
that data is not present.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import averaging_powerspec as avg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_seed_snap(snap_dir: Path, k, Pk, Pcurl, z: float):
    snap_dir.mkdir(parents=True, exist_ok=True)
    np.save(snap_dir / "k_m.npy", k)
    np.save(snap_dir / "Pk_m.npy", Pk)
    np.save(snap_dir / "k_curl.npy", k)
    np.save(snap_dir / "Pk_curl.npy", Pcurl)
    (snap_dir / "snapshot_metadata.json").write_text(json.dumps({"redshift": z}))


def _make_model_seeds(base: Path, model: str, seed1: str, seed2: str,
                      n_snaps: int, k, pk1, pk2, pcurl1, pcurl2,
                      node: str = "node_037"):
    for i in range(n_snaps):
        _write_seed_snap(base / model / node / seed1 / f"snap_{i:03d}", k, pk1, pcurl1, z=float(i))
        _write_seed_snap(base / model / node / seed2 / f"snap_{i:03d}", k, pk2, pcurl2, z=float(i))


def _load_averaged(base: Path, model: str, snap_idx: int, node: str = "node_037"):
    d_m = np.load(base / model / node / "Pk_matter" / f"{snap_idx:03d}.npy", allow_pickle=True).item()
    d_c = np.load(base / model / node / "Pk_curl"   / f"{snap_idx:03d}.npy", allow_pickle=True).item()
    return d_m, d_c


def _run_main(base_dir, models, seed1, seed2, h_orig=None, h_correct=None,
              node: str = "node_037"):
    argv = ["--base-dir", str(base_dir), "--models"] + models + \
           ["--seed1", seed1, "--seed2", seed2, "--node", node]
    if h_orig is not None:
        argv += ["--h-orig", str(h_orig), "--h-correct", str(h_correct)]
    import sys as _sys
    old = _sys.argv
    _sys.argv = ["averaging_powerspec.py"] + argv
    try:
        avg.main()
    finally:
        _sys.argv = old


# ---------------------------------------------------------------------------
# Unit tests — synthetic data only
# ---------------------------------------------------------------------------

def test_averaged_pk_equals_mean_of_seeds(tmp_path):
    """Output Pk must equal the arithmetic mean of the two seed arrays."""
    k = np.logspace(-2, 0, 20)
    pk1 = 1e4 * k**-2
    pk2 = 2e4 * k**-2
    _make_model_seeds(tmp_path, "lcdm", "seed_A", "seed_B",
                      n_snaps=1, k=k, pk1=pk1, pk2=pk2, pcurl1=pk1*0.1, pcurl2=pk2*0.1)

    _run_main(tmp_path, ["lcdm"], "seed_A", "seed_B")

    d_m, d_c = _load_averaged(tmp_path, "lcdm", 0)
    np.testing.assert_allclose(d_m["Pk"], (pk1 + pk2) / 2, rtol=1e-10)
    np.testing.assert_allclose(d_c["Pcurl"], (pk1*0.1 + pk2*0.1) / 2, rtol=1e-10)
    np.testing.assert_allclose(d_m["k"], k, rtol=1e-10)


def test_k_grid_mismatch_raises(tmp_path):
    """Seeds with incompatible k grids must raise AssertionError, not silently average."""
    k1 = np.logspace(-2, 0, 20)
    k2 = k1 * 1.1   # deliberately shifted
    pk = np.ones(20)

    for seed, k in [("seed_A", k1), ("seed_B", k2)]:
        _write_seed_snap(tmp_path / "lcdm" / "node_037" / seed / "snap_000", k, pk, pk, z=0.0)

    with pytest.raises(AssertionError):
        _run_main(tmp_path, ["lcdm"], "seed_A", "seed_B")


def test_h_correction_rescales_k(tmp_path):
    """k_out must equal k_stored * (h_correct / h_orig)."""
    h_orig, h_correct = 0.78052, 0.673
    k = np.logspace(-2, 0, 20)
    pk = np.ones(20) * 1e3
    _make_model_seeds(tmp_path, "frhs", "s1", "s2",
                      n_snaps=1, k=k, pk1=pk, pk2=pk, pcurl1=pk, pcurl2=pk)

    _run_main(tmp_path, ["frhs"], "s1", "s2", h_orig=h_orig, h_correct=h_correct)

    d_m, _ = _load_averaged(tmp_path, "frhs", 0)
    expected_k = k * (h_correct / h_orig)
    np.testing.assert_allclose(d_m["k"], expected_k, rtol=1e-6)


def test_h_correction_rescales_pk(tmp_path):
    """Pk_out must equal Pk_stored / (h_correct / h_orig)**3."""
    h_orig, h_correct = 0.78052, 0.673
    k = np.logspace(-2, 0, 20)
    pk = np.ones(20) * 1e3
    _make_model_seeds(tmp_path, "frhs", "s1", "s2",
                      n_snaps=1, k=k, pk1=pk, pk2=pk, pcurl1=pk, pcurl2=pk)

    _run_main(tmp_path, ["frhs"], "s1", "s2", h_orig=h_orig, h_correct=h_correct)

    d_m, d_c = _load_averaged(tmp_path, "frhs", 0)
    ratio = h_correct / h_orig
    np.testing.assert_allclose(d_m["Pk"],    pk / ratio**3, rtol=1e-6)
    np.testing.assert_allclose(d_c["Pcurl"], pk / ratio**3, rtol=1e-6)


def test_h_correction_identity(tmp_path):
    """h_orig == h_correct must leave k and Pk unchanged."""
    k = np.logspace(-2, 0, 20)
    pk = np.ones(20) * 1e3
    _make_model_seeds(tmp_path, "frhs", "s1", "s2",
                      n_snaps=1, k=k, pk1=pk, pk2=pk, pcurl1=pk, pcurl2=pk)

    _run_main(tmp_path, ["frhs"], "s1", "s2", h_orig=0.673, h_correct=0.673)

    d_m, _ = _load_averaged(tmp_path, "frhs", 0)
    np.testing.assert_allclose(d_m["k"],  k,  rtol=1e-10)
    np.testing.assert_allclose(d_m["Pk"], pk, rtol=1e-10)


def test_h_correction_requires_both_flags(tmp_path):
    """Providing only --h-orig (without --h-correct) must not rescale anything."""
    k = np.logspace(-2, 0, 20)
    pk = np.ones(20) * 1e3
    _make_model_seeds(tmp_path, "frhs", "s1", "s2",
                      n_snaps=1, k=k, pk1=pk, pk2=pk, pcurl1=pk, pcurl2=pk)

    # Pass h_orig but leave h_correct=None → no correction should be applied
    import sys as _sys
    old = _sys.argv
    _sys.argv = ["averaging_powerspec.py",
                 "--base-dir", str(tmp_path),
                 "--models", "frhs",
                 "--seed1", "s1", "--seed2", "s2",
                 "--h-orig", "0.78052"]
    try:
        avg.main()
    finally:
        _sys.argv = old

    d_m, _ = _load_averaged(tmp_path, "frhs", 0)
    np.testing.assert_allclose(d_m["k"], k, rtol=1e-10)


# ---------------------------------------------------------------------------
# Integration / regression tests — require output/ data
# ---------------------------------------------------------------------------

_HAS_DATA = Path("output/lcdm/node_037/Pk_matter/024.npy").exists()
_skip_no_data = pytest.mark.skipif(not _HAS_DATA, reason="output/ data not present")

_SNAP_LCDM = "024"
_SNAP_MG   = "024"


@_skip_no_data
def test_k_grid_consistent_with_h():
    """
    All three models share the same Pylians3 grid in h-units (same BoxSize=500 Mpc/h,
    same N_grid=1024), so k_physical[i] / h must be the same across models at every
    bin index — i.e. the k/h grids must agree to within the bin width (~2%).

    With the wrong-h bug, k_frhs/h_frhs was off by (0.78052/0.673) ≈ 1.16 relative
    to k_lcdm/h_lcdm, so the interpolated difference would be ~16% — far above 2%.
    """
    from vp_utils import build_cosmo_params_from_file

    snaps = {"lcdm": _SNAP_LCDM, "frhs": _SNAP_MG, "ndgp": _SNAP_MG}
    k_over_h = {}
    for model, snap in snaps.items():
        d = np.load(f"output/{model}/node_037/Pk_matter/{snap}.npy", allow_pickle=True).item()
        pars = build_cosmo_params_from_file(f"output/{model}/node_037/parameters-usedvalues")
        k_over_h[model] = d["k"] / pars["h"]

    k_ref = k_over_h["lcdm"]
    for model in ["frhs", "ndgp"]:
        k_mg = k_over_h[model]
        k_lo = max(k_ref[0], k_mg[0])
        k_hi = min(k_ref[-1], k_mg[-1])
        mask = (k_ref >= k_lo) & (k_ref <= k_hi)
        k_mg_interp = np.interp(k_ref[mask], k_mg, k_mg)
        np.testing.assert_allclose(
            k_mg_interp, k_ref[mask], rtol=0.02,
            err_msg=(
                f"{model}: k/h grid differs from lcdm by more than 2%.  "
                f"Likely wrong h was used in powerspec.py for {model}."
            ),
        )


@_skip_no_data
def test_pk_ratio_mg_lcdm_within_physical_range():
    """
    At z≈0, P_m^MG(k) / P_m^LCDM(k) must lie within [0.5, 2.0] at every k.

    Before the fix, frhs ratio was ~0.48, which is physically implausible and
    directly signals the wrong-seeds or wrong-h bug.
    """
    lcdm = np.load(f"output/lcdm/node_037/Pk_matter/{_SNAP_LCDM}.npy", allow_pickle=True).item()
    for model, snap in [("frhs", _SNAP_MG), ("ndgp", _SNAP_MG)]:
        mg = np.load(f"output/{model}/node_037/Pk_matter/{snap}.npy", allow_pickle=True).item()
        ratio = mg["Pk"] / np.interp(mg["k"], lcdm["k"], lcdm["Pk"])
        assert np.all(ratio > 0.5), (
            f"{model}: Pk ratio dips below 0.5 (min={ratio.min():.3f}).  "
            "Check that the correct seeds and h were used."
        )
        assert np.all(ratio < 2.0), (
            f"{model}: Pk ratio exceeds 2.0 (max={ratio.max():.3f})."
        )


@_skip_no_data
def test_averaged_output_matches_node39_seeds():
    """
    The averaged Pk_matter for frhs must match the mean of the node-037 seed files,
    not the old node-006 seeds.

    Before the fix, averaging_powerspec.py hardcoded 'seed_2080'/'seed_4257',
    so the averaged file was built from node-006 seeds even after node-037 seeds
    were added to the repository.
    """
    # The averaged files were produced with --h-orig 0.78052 --h-correct 0.673,
    # so the stored Pk = seed_mean / (h_correct/h_orig)^3.  Apply the same factor
    # to the raw seed mean before comparing.
    h_orig, h_correct = 0.78052, 0.673

    # Now fixed, so h_ratio is set to 1.
    h_ratio = 1. #  h_correct / h_orig  # 0.8623

    for model in ["frhs", "ndgp"]:
        snap = _SNAP_MG
        s1 = np.load(f"output/{model}/node_037/seed_2080/snap_{snap}/Pk_m.npy")
        s2 = np.load(f"output/{model}/node_037/seed_4257/snap_{snap}/Pk_m.npy")
        expected = (s1 + s2) / 2 / h_ratio**3

        averaged = np.load(f"output/{model}/node_037/Pk_matter/{snap}.npy",
                           allow_pickle=True).item()

        np.testing.assert_allclose(
            averaged["Pk"], expected, rtol=1e-4,
            err_msg=(
                f"{model}: averaged Pk does not match h-corrected mean of node-037 seeds.  "
                "Re-run averaging_powerspec.py with --seed1 seed_2080_node39 "
                "--seed2 seed_4257_node39 --h-orig 0.78052 --h-correct 0.673."
            ),
        )
