"""
Tests for python/vp_utils.py

Covers:
- read_params_file / build_cosmo_params_from_file
- Background cosmology functions (a_of_z, z_of_a, Hubble, chi_of_z)
- Electron number density n_ele
- C_ell_XY dispatcher (Phi, B, kSZ, B_X_kSZ)
"""
import warnings
import numpy as np
import pytest

import vp_utils as u


# ── fixtures / shared values ─────────────────────────────────────────────────

PARAMS = u.parameters_sim   # built from the real lcdm parameters-usedvalues file


# ── read_params_file ──────────────────────────────────────────────────────────

class TestReadParamsFile:
    def test_known_values(self):
        raw = u.parameters_sim["simfile_raw"]
        assert float(raw["Omega0"]) == pytest.approx(0.31315)
        assert float(raw["HubbleParam"]) == pytest.approx(0.6737)
        assert float(raw["BoxSize"]) == pytest.approx(500.0)
        assert float(raw["OmegaBaryon"]) == pytest.approx(0.049199)

    def test_types_cast(self):
        raw = u.parameters_sim["simfile_raw"]
        assert isinstance(raw["BoxSize"], (int, float))
        assert isinstance(raw["HubbleParam"], (int, float))


# ── build_cosmo_params_from_file ──────────────────────────────────────────────

class TestBuildCosmoParams:
    def test_omega_m(self):
        assert PARAMS["Omega_m"] == pytest.approx(0.31315)

    def test_h_and_H0(self):
        assert PARAMS["h"] == pytest.approx(0.6737)
        assert PARAMS["H0"] == pytest.approx(67.37, rel=1e-4)

    def test_derived_rho_c_positive(self):
        assert PARAMS["rho_c"] > 0

    def test_tau_H_positive(self):
        assert PARAMS["tau_H"] > 0

    def test_kSZ_bfac(self):
        assert PARAMS["kSZ_bfac"] == pytest.approx(PARAMS["tau_H"] * PARAMS["xe"])

    def test_omega_b(self):
        assert PARAMS["Omega_b"] == pytest.approx(0.049199)

    def test_ombh2(self):
        expected = PARAMS["Omega_b"] * PARAMS["h"] ** 2
        assert PARAMS["Ombh2"] == pytest.approx(expected)


# ── scale factor / redshift ───────────────────────────────────────────────────

class TestScaleFactor:
    def test_a_at_z0(self):
        assert u.a_of_z(0) == pytest.approx(1.0)

    def test_a_at_z1(self):
        assert u.a_of_z(1) == pytest.approx(0.5)

    def test_a_decreases_with_z(self):
        zs = np.linspace(0, 5, 20)
        a = np.array([u.a_of_z(z) for z in zs])
        assert np.all(np.diff(a) < 0)

    def test_z_at_a1(self):
        assert u.z_of_a(1.0) == pytest.approx(0.0)

    def test_z_at_a05(self):
        assert u.z_of_a(0.5) == pytest.approx(1.0)

    def test_roundtrip(self):
        for z in [0.0, 0.5, 1.0, 2.0, 5.0]:
            assert u.z_of_a(u.a_of_z(z)) == pytest.approx(z, rel=1e-6)


# ── Hubble parameter ──────────────────────────────────────────────────────────

class TestHubble:
    def test_H_at_z0_equals_H0(self):
        assert u.Hubble(0) == pytest.approx(PARAMS["H0"], rel=1e-6)

    def test_H_positive_range(self):
        zs = np.linspace(0, 10, 50)
        Hs = np.array([u.Hubble(z) for z in zs])
        assert np.all(Hs > 0)

    def test_H_increases_with_z(self):
        # Matter-dominated: H grows with z
        zs = np.linspace(0, 3, 20)
        Hs = np.array([u.Hubble(z) for z in zs])
        assert np.all(np.diff(Hs) > 0)


# ── comoving distance ─────────────────────────────────────────────────────────

class TestChiOfZ:
    def test_chi_z0_is_zero(self):
        assert u.chi_of_z(0) == pytest.approx(0.0, abs=1.0)  # within 1 Mpc

    def test_chi_positive(self):
        for z in [0.1, 0.5, 1.0, 2.0]:
            assert u.chi_of_z(z) > 0

    def test_chi_monotonically_increasing(self):
        zs = np.linspace(0.01, 5, 30)
        chis = u.chi_of_z(zs)
        assert np.all(np.diff(chis) > 0)

    def test_chi_z1_reasonable_Mpc(self):
        # chi(z=1) for a flat ΛCDM cosmology ~3000 Mpc (rough sanity check)
        chi1 = u.chi_of_z(1.0)
        assert 2000 < chi1 < 5000, f"chi(z=1) = {chi1:.1f} Mpc looks off"


# ── electron number density ───────────────────────────────────────────────────

class TestElectronDensity:
    def test_n_ele_positive(self):
        for z in [0.0, 0.5, 1.0]:
            assert u.n_ele(z) > 0

    def test_n_ele_scales_as_1pz3(self):
        # n ∝ (1+z)^3
        z0, z1 = 0.5, 1.0
        ratio = u.n_ele(z1) / u.n_ele(z0)
        expected = ((1 + z1) / (1 + z0)) ** 3
        assert ratio == pytest.approx(expected, rel=1e-4)


# ── C_ell_XY dispatcher ───────────────────────────────────────────────────────

def _flat_pk(k):
    """Flat (constant) power spectrum for quick integration tests."""
    return np.ones_like(np.asarray(k, float)) * 1e3


def _flat_pk_evol(x):
    k, z = x
    return np.ones_like(np.asarray(k, float)) * 1e3


class TestCellXY:
    """Smoke tests: results should be finite non-negative floats."""

    _kmin = PARAMS["kF"]
    _kmax = PARAMS["kN"]

    @pytest.mark.parametrize("type_XY", ["Phi", "B", "kSZ", "B_X_kSZ"])
    def test_returns_finite(self, type_XY):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            val = u.C_ell_XY(
                z_s=0.5,
                ell=200,
                z_min=1e-4,
                kmin=self._kmin,
                kmax=self._kmax,
                Pk=_flat_pk,
                type_XY=type_XY,
                Pk_evol=False,
            )
        assert np.isfinite(val), f"C_ell_{type_XY} returned {val}"

    @pytest.mark.parametrize("type_XY", ["Phi", "B", "kSZ", "B_X_kSZ"])
    def test_returns_nonneg(self, type_XY):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            val = u.C_ell_XY(
                z_s=0.5,
                ell=200,
                z_min=1e-4,
                kmin=self._kmin,
                kmax=self._kmax,
                Pk=_flat_pk,
                type_XY=type_XY,
                Pk_evol=False,
            )
        assert val >= 0

    def test_raises_on_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown type_XY"):
            u.C_ell_XY(
                z_s=0.5, ell=200, z_min=1e-4,
                kmin=self._kmin, kmax=self._kmax,
                Pk=_flat_pk, type_XY="NotAType",
            )

    def test_zero_when_no_valid_k_range(self):
        # Force ell/chi to be outside [kmin, kmax] so mask is empty
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            val = u.C_ell_XY(
                z_s=0.5, ell=1,
                z_min=1e-4,
                kmin=1e5, kmax=1e6,   # k range impossible to reach
                Pk=_flat_pk,
                type_XY="Phi",
                Pk_evol=False,
            )
        assert val == 0.0

    @pytest.mark.parametrize("type_XY", ["Phi", "B", "kSZ", "B_X_kSZ"])
    def test_pk_evol_returns_finite(self, type_XY):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            val = u.C_ell_XY(
                z_s=0.5,
                ell=200,
                z_min=1e-4,
                kmin=self._kmin,
                kmax=self._kmax,
                Pk=_flat_pk_evol,
                type_XY=type_XY,
                Pk_evol=True,
            )
        assert np.isfinite(val)
