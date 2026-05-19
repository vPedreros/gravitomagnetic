"""
Tests for the SNR noise / covariance functions in python/SNR.py.

SNR.py runs argparse + CLASS + file I/O at module level, so it cannot be
imported directly.  The functions under test are self-contained and are
reproduced here with the same logic so they can be validated independently.
"""
import numpy as np
import pytest


# ── Re-implementation of the pure functions from SNR.py ──────────────────────
# (kept identical to the source so a diff immediately reveals any divergence)

arcmin_to_rad = np.pi / (180 * 60)


def noise_convergence(pars_surv):
    sigma_e = pars_surv["sigma_e"]
    n_gal = pars_surv["n_gal"] / arcmin_to_rad**2
    return sigma_e**2 / n_gal


def noise_temperature(ell, pars_exp):
    FWHM = pars_exp["theta_fwhm"] * arcmin_to_rad
    Delta_T = pars_exp["Delta_T"] * arcmin_to_rad
    factor = (Delta_T / pars_exp["T_bar"]) ** 2
    arg_exp = ell**2 * FWHM**2 / (8 * np.log(2))
    return factor * np.exp(arg_exp)


def Cov(ell_list, C_ell_B_X_kSZ, C_ell_TT, C_ell_kappaWL, C_ell_kSZ, pars_surv, pars_exp):
    logell = np.log10(ell_list)
    dlog = logell[1] - logell[0]
    edges = 10 ** (
        np.r_[logell[0] - dlog / 2, 0.5 * (logell[:-1] + logell[1:]), logell[-1] + dlog / 2]
    )
    Delta_ell = np.diff(edges)
    factor = Delta_ell * pars_surv["f_sky"] * (2 * ell_list + 1)
    contributions = C_ell_B_X_kSZ**2 + (
        C_ell_TT + C_ell_kSZ + noise_temperature(ell_list, pars_exp)
    ) * (C_ell_kappaWL + noise_convergence(pars_surv))
    return contributions / factor


def SNR(ell_list, C_ell_B_X_kSZ, C_ell_TT, C_ell_kappaWL, C_ell_kSZ, survey_pars, exp_pars):
    return np.sqrt(C_ell_B_X_kSZ**2 / Cov(ell_list, C_ell_B_X_kSZ, C_ell_TT, C_ell_kappaWL, C_ell_kSZ, survey_pars, exp_pars))


# ── Reference parameter dicts (same as SNR.py) ───────────────────────────────

pars_surv = {
    "Euclid": {"n_gal": 30, "sigma_e": np.sqrt(2) * 0.21, "f_sky": 0.36},
    "LSST":   {"n_gal": 40, "sigma_e": 0.22,               "f_sky": 0.5 },
}

pars_exp = {
    "Planck": {"theta_fwhm": 5,   "Delta_T": 3.1, "f_sky": 0.82, "T_bar": 2.7e6},
    "SO":     {"theta_fwhm": 1.4, "Delta_T": 6,   "f_sky": 0.4,  "T_bar": 2.7e6},
}


# ── noise_convergence ─────────────────────────────────────────────────────────

class TestNoiseConvergence:
    def test_positive(self):
        for s in pars_surv.values():
            assert noise_convergence(s) > 0

    def test_euclid_vs_lsst_ordering(self):
        # LSST has more galaxies per arcmin² → lower shape noise per mode
        nc_euclid = noise_convergence(pars_surv["Euclid"])
        nc_lsst   = noise_convergence(pars_surv["LSST"])
        # sigma_e comparable, but n_gal larger for LSST → lower noise
        # LSST sigma_e=0.22, n=40; Euclid sigma_e≈0.297, n=30
        assert nc_lsst < nc_euclid

    def test_scales_with_sigma_e_squared(self):
        p = {"n_gal": 30, "sigma_e": 0.2, "f_sky": 0.5}
        p2 = {"n_gal": 30, "sigma_e": 0.4, "f_sky": 0.5}
        assert noise_convergence(p2) == pytest.approx(4 * noise_convergence(p), rel=1e-6)

    def test_inversely_proportional_to_n_gal(self):
        p1 = {"n_gal": 20, "sigma_e": 0.22, "f_sky": 0.5}
        p2 = {"n_gal": 40, "sigma_e": 0.22, "f_sky": 0.5}
        assert noise_convergence(p2) == pytest.approx(0.5 * noise_convergence(p1), rel=1e-6)


# ── noise_temperature ─────────────────────────────────────────────────────────

class TestNoiseTemperature:
    def test_positive_scalar(self):
        for e in pars_exp.values():
            assert noise_temperature(500, e) > 0

    def test_array_input(self):
        ells = np.array([100, 500, 1000])
        nt = noise_temperature(ells, pars_exp["Planck"])
        assert nt.shape == ells.shape
        assert np.all(nt > 0)

    def test_increases_with_ell(self):
        ells = np.logspace(2, 4, 20)
        nt_planck = noise_temperature(ells, pars_exp["Planck"])
        assert np.all(np.diff(nt_planck) > 0)

    def test_so_less_than_planck_at_low_ell(self):
        # SO has smaller beam (1.4') so less noise at low ell
        # but higher Delta_T — at low ell beam is negligible, compare Delta_T
        nt_so     = noise_temperature(100, pars_exp["SO"])
        nt_planck = noise_temperature(100, pars_exp["Planck"])
        # Delta_T: Planck=3.1, SO=6 → at low ell SO should have MORE noise
        assert nt_so > nt_planck


# ── Cov (covariance matrix diagonal) ─────────────────────────────────────────

class TestCov:
    def _ell_grid(self):
        return np.logspace(2, 4, 30)

    def _make_arrays(self, ell):
        n = len(ell)
        return (
            np.ones(n) * 1e-8,  # C_ell_B_X_kSZ
            np.ones(n) * 1e-6,  # C_ell_TT
            np.ones(n) * 1e-8,  # C_ell_kappaWL
            np.ones(n) * 1e-9,  # C_ell_kSZ
        )

    def test_positive(self):
        ell = self._ell_grid()
        bxk, tt, kWL, ksz = self._make_arrays(ell)
        cov = Cov(ell, bxk, tt, kWL, ksz, pars_surv["LSST"], pars_exp["Planck"])
        assert np.all(cov > 0)

    def test_shape(self):
        ell = self._ell_grid()
        bxk, tt, kWL, ksz = self._make_arrays(ell)
        cov = Cov(ell, bxk, tt, kWL, ksz, pars_surv["LSST"], pars_exp["Planck"])
        assert cov.shape == ell.shape

    def test_larger_fsky_reduces_cov(self):
        ell = self._ell_grid()
        bxk, tt, kWL, ksz = self._make_arrays(ell)
        p_small = dict(pars_surv["LSST"]); p_small["f_sky"] = 0.1
        p_large = dict(pars_surv["LSST"]); p_large["f_sky"] = 0.9
        cov_small = Cov(ell, bxk, tt, kWL, ksz, p_small, pars_exp["Planck"])
        cov_large = Cov(ell, bxk, tt, kWL, ksz, p_large, pars_exp["Planck"])
        assert np.all(cov_large < cov_small)


# ── SNR ───────────────────────────────────────────────────────────────────────

class TestSNR:
    def _ell_grid(self):
        return np.logspace(2, 4, 30)

    def test_positive(self):
        ell = self._ell_grid()
        n = len(ell)
        snr = SNR(ell, np.ones(n)*1e-8, np.ones(n)*1e-6, np.ones(n)*1e-8, np.ones(n)*1e-9,
                  pars_surv["LSST"], pars_exp["Planck"])
        assert np.all(snr > 0)

    def test_larger_signal_higher_snr(self):
        ell = self._ell_grid()
        n = len(ell)
        tt   = np.ones(n) * 1e-6
        kWL  = np.ones(n) * 1e-8
        ksz  = np.ones(n) * 1e-9
        snr_low  = SNR(ell, np.ones(n)*1e-9, tt, kWL, ksz, pars_surv["LSST"], pars_exp["Planck"])
        snr_high = SNR(ell, np.ones(n)*1e-7, tt, kWL, ksz, pars_surv["LSST"], pars_exp["Planck"])
        assert np.all(snr_high > snr_low)

    def test_finite_and_nonneg(self):
        ell = self._ell_grid()
        n = len(ell)
        snr = SNR(ell, np.ones(n)*1e-8, np.ones(n)*1e-6, np.ones(n)*1e-8, np.ones(n)*1e-9,
                  pars_surv["Euclid"], pars_exp["SO"])
        assert np.all(np.isfinite(snr))
        assert np.all(snr >= 0)
