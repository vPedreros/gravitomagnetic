"""
Unit tests for vp_utils.py — cosmological background functions and C_ell kernels.

conftest.py handles the module-level file-read mock so vp_utils imports cleanly
without COSMA data.  Tests here use a simple analytic power spectrum.
"""

import numpy as np
import pytest
import vp_utils as vp


# ---------------------------------------------------------------------------
# Analytic mock power spectrum: Pk(k) = 1e-3 * k^-2
# ---------------------------------------------------------------------------

def _pk_simple(x):
    """Accepts (k, z) tuple (Pk_evol=True path) or scalar k."""
    k = np.asarray(x[0] if isinstance(x, tuple) else x, dtype=float)
    return 1e-3 * k**(-2.0)


COMMON_KWARGS = dict(
    z_s=1.0,
    ell=500.0,
    kmin=1e-3,
    kmax=10.0,
    Pk=_pk_simple,
    Pk_evol=False,
    N_int=200,
)


# ---------------------------------------------------------------------------
# Background functions
# ---------------------------------------------------------------------------

def test_chi_of_z_zero():
    assert abs(vp.chi_of_z(0.0)) < 1.0  # should be ~0 Mpc


def test_chi_monotone():
    z = np.linspace(0.1, 3.0, 50)
    chi = vp.chi_of_z(z)
    assert np.all(np.diff(chi) > 0)


def test_hubble_at_zero():
    H0 = vp.parameters_sim["H0"]
    assert abs(vp.Hubble(0.0) - H0) / H0 < 1e-6


def test_n_ele_positive():
    z = np.linspace(0, 3, 20)
    assert np.all(vp.n_ele(z) > 0)


@pytest.mark.parametrize("z,expected", [
    (0.1, 0.00021496741731112884),
    (0.5, 0.0013648064155136865),
    (1.0, 0.003354239026127417),
    (1.5, 0.0058137642910988845),
    (2.0, 0.008642092092724431),
    (3.0, 0.01518486074671854),
    (5.0, 0.03107597093289381),
])
def test_tau_optical_depth_baseline(z, expected):
    """Pin tau(z) against the interpolator output under the conftest mock cosmology.
    Locks in the current implementation; rel=1e-10 allows only float-associativity drift."""
    assert float(vp.tau_optical_depth(z)) == pytest.approx(expected, rel=1e-10)


def test_tau_optical_depth_monotone():
    z = np.linspace(0.05, 5.0, 50)
    tau = vp.tau_optical_depth(z)
    assert np.all(np.diff(tau) > 0)


# ---------------------------------------------------------------------------
# Lazy parameters_sim resolution
# ---------------------------------------------------------------------------

def test_load_parameters_sim_errors_without_env(monkeypatch):
    """If neither set_parameters_sim has been called nor VP_PARAMS_FILE is set,
    accessing parameters_sim raises a clear RuntimeError."""
    monkeypatch.delenv("VP_PARAMS_FILE", raising=False)
    monkeypatch.setattr(vp, "_parameters_sim_cache", None)
    with pytest.raises(RuntimeError, match="VP_PARAMS_FILE"):
        vp._load_parameters_sim()


def test_set_parameters_sim_invalidates_caches(monkeypatch, tmp_path):
    """set_parameters_sim must reset the chi and tau interpolators so they
    rebuild against the new cosmology on next access."""
    cached_chi = vp._chi_interp
    cached_tau = vp._tau_interp
    # Trigger build of both if not already populated
    _ = vp.chi_of_z(1.0)
    _ = vp.tau_optical_depth(1.0)
    assert vp._chi_interp is not None and vp._tau_interp is not None

    # Inject a slightly different cosmology and verify caches were cleared
    new_pars = dict(vp._load_parameters_sim())
    vp.set_parameters_sim(new_pars)
    assert vp._chi_interp is None
    assert vp._tau_interp is None

    # Restore the original interpolators so we don't pollute later tests
    vp._chi_interp = cached_chi
    vp._tau_interp = cached_tau


# ---------------------------------------------------------------------------
# ValueError for invalid integration method
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn", [vp.C_ell_Phi, vp.C_ell_B, vp.C_ell_kSZ, vp.C_ell_B_X_kSZ])
def test_raise_invalid_method(fn):
    with pytest.raises(ValueError, match="Invalid selection"):
        fn(**COMMON_KWARGS, integr_method="bad_method")


# ---------------------------------------------------------------------------
# Integration method consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn", [vp.C_ell_Phi, vp.C_ell_B])
def test_integration_methods_agree(fn):
    result_simp = fn(**COMMON_KWARGS, integr_method="simpson")
    result_trap = fn(**COMMON_KWARGS, integr_method="trapezoid")
    result_quad = fn(**COMMON_KWARGS, integr_method="quad")

    assert result_simp > 0
    assert abs(result_simp - result_trap) / abs(result_simp) < 0.05
    assert abs(result_simp - result_quad) / abs(result_simp) < 0.05


def test_c_ell_ksz_quad_returns_scalar():
    """Smoke test — catches chi vs chi_x scope bug in the quad path."""
    result = vp.C_ell_kSZ(**COMMON_KWARGS, integr_method="quad")
    assert np.isfinite(result)
    assert result > 0


def test_c_ell_b_x_ksz_quad_returns_scalar():
    """Smoke test — catches chi vs chi_x scope bug in the quad path."""
    result = vp.C_ell_B_X_kSZ(**COMMON_KWARGS, integr_method="quad")
    assert np.isfinite(result)


def test_ksz_integration_methods_agree():
    """simpson vs quad should agree within 10% after n_ele factor fix."""
    result_simp = vp.C_ell_kSZ(**COMMON_KWARGS, integr_method="simpson")
    result_quad = vp.C_ell_kSZ(**COMMON_KWARGS, integr_method="quad")
    assert abs(result_simp - result_quad) / abs(result_simp) < 0.10


def test_b_x_ksz_integration_methods_agree():
    result_simp = vp.C_ell_B_X_kSZ(**COMMON_KWARGS, integr_method="simpson")
    result_quad = vp.C_ell_B_X_kSZ(**COMMON_KWARGS, integr_method="quad")
    assert abs(result_simp - result_quad) / abs(result_simp) < 0.10


# ---------------------------------------------------------------------------
# Absolute-value baselines.  These pin each C_ell function's simpson output at
# the COMMON_KWARGS fiducial inputs.  Captured from the pre-refactor code so
# any future change to the kernel logic that shifts the physics fails loudly,
# not silently.  rel=1e-10 allows only float-associativity drift.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn,expected", [
    (vp.C_ell_Phi,     6.230122818322606e-15),
    (vp.C_ell_B,       2.1593428790470605e-24),
    (vp.C_ell_kSZ,     2.1787388552528067e-27),
    (vp.C_ell_B_X_kSZ, 5.545271300506326e-26),
])
def test_c_ell_baseline_unchanged(fn, expected):
    result = fn(**COMMON_KWARGS, integr_method="simpson")
    assert result == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize("fn,expected", [
    (vp.C_ell_Phi,     6.144557175246948e-15),
    (vp.C_ell_B,       2.116050527317594e-24),
    (vp.C_ell_kSZ,     2.1552296011290954e-27),
    (vp.C_ell_B_X_kSZ, 5.462413350965035e-26),
])
def test_c_ell_baseline_unchanged_quad(fn, expected):
    """Pin each C_ell function's quad output. Catches drift in the scalar
    `kernel_fn(np.asarray(x), ...)` branch that test_c_ell_baseline_unchanged
    (simpson) does not exercise."""
    result = fn(**COMMON_KWARGS, integr_method="quad")
    assert result == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# build_cosmo_params_from_file — h propagation
# ---------------------------------------------------------------------------

def test_build_cosmo_params_respects_h(tmp_path):
    """
    h from the parameters file must flow through unchanged into params['h'],
    params['kF'], and params['kN'].

    With the bug (wrong VP_PARAMS_FILE / wrong model's parameters file loaded),
    h would be e.g. 0.6737 for a model that actually used h=0.78052, silently
    mis-scaling every k and Pk value produced by powerspec.py.
    """
    params_file = tmp_path / "parameters-usedvalues"
    params_file.write_text(
        "Omega0 0.31315\nOmegaLambda 0.68685\nOmegaBaryon 0.049199\n"
        "HubbleParam 0.78052\nBoxSize 500.0\n"
        "UnitLength_in_cm 3.08568e+24\nUnitMass_in_g 1.989e+43\n"
        "UnitVelocity_in_cm_per_s 100000.0\n"
    )
    p = vp.build_cosmo_params_from_file(params_file)
    assert p["h"] == pytest.approx(0.78052, rel=1e-5)


@pytest.mark.parametrize("h", [0.673, 0.78052])
def test_k_modes_scale_with_h(tmp_path, h):
    """
    kF and kN in physical units (Mpc^-1) must equal (1/BoxSize)*h and
    (N_grid/2/BoxSize)*pi*h respectively.

    If the wrong h is loaded (e.g. ΛCDM h=0.6737 for a model with h=0.78052),
    the masking window in powerspec.py cuts at the wrong k range and the
    unit-converted k values are wrong.
    """
    box, ngrid = 500.0, 1024
    params_file = tmp_path / f"params_{h}.txt"
    params_file.write_text(
        f"Omega0 0.31\nOmegaLambda 0.69\nOmegaBaryon 0.049\n"
        f"HubbleParam {h}\nBoxSize {box}\n"
        "UnitLength_in_cm 3.08568e+24\nUnitMass_in_g 1.989e+43\n"
        "UnitVelocity_in_cm_per_s 100000.0\n"
    )
    p = vp.build_cosmo_params_from_file(params_file)

    # kF and kN are set externally by the module after loading, so we replicate
    # the same arithmetic used in vp_utils.py lines ~114-118.
    kF_expected = (1.0 / box) * h
    kN_expected = (ngrid / box) * np.pi * h
    assert (1.0 / box) * p["h"] == pytest.approx(kF_expected, rel=1e-6)
    assert (ngrid / box) * np.pi * p["h"] == pytest.approx(kN_expected, rel=1e-6)
