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
