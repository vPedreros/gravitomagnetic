"""
Unit tests for the noise and SNR functions in SNR.py.

SNR.py runs CLASS and reads CSV files at import time, so we extract the pure
functions (noise_convergence, noise_temperature, Cov, SNR) directly and test
them with synthetic parameters — no CLASS or data files needed.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Replicate the pure functions from SNR.py so tests work without CLASS / CSV
# ---------------------------------------------------------------------------

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
        np.r_[
            logell[0] - dlog / 2,
            0.5 * (logell[:-1] + logell[1:]),
            logell[-1] + dlog / 2,
        ]
    )
    Delta_ell = np.diff(edges)
    factor = Delta_ell * pars_surv["f_sky"] * (2 * ell_list + 1)
    contributions = C_ell_B_X_kSZ**2 + (
        C_ell_TT + C_ell_kSZ + noise_temperature(ell_list, pars_exp)
    ) * (C_ell_kappaWL + noise_convergence(pars_surv))
    return contributions / factor


def SNR_fn(ell_list, C_ell_B_X_kSZ, C_ell_TT, C_ell_kappaWL, C_ell_kSZ, pars_surv, pars_exp):
    return np.sqrt(
        C_ell_B_X_kSZ**2
        / Cov(ell_list, C_ell_B_X_kSZ, C_ell_TT, C_ell_kappaWL, C_ell_kSZ, pars_surv, pars_exp)
    )


# ---------------------------------------------------------------------------
# Reference parameter sets
# ---------------------------------------------------------------------------

PARS_LSST = {"n_gal": 40, "sigma_e": 0.22, "f_sky": 0.5}
PARS_EUCLID = {"n_gal": 30, "sigma_e": np.sqrt(2) * 0.21, "f_sky": 0.36}
PARS_PLANCK = {"theta_fwhm": 5, "Delta_T": 3.1, "f_sky": 0.82, "T_bar": 2.7e6}
PARS_SO = {"theta_fwhm": 1.4, "Delta_T": 6, "f_sky": 0.4, "T_bar": 2.7e6}

ELL = np.logspace(2, 4, 40)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pars", [PARS_LSST, PARS_EUCLID])
def test_noise_convergence_positive(pars):
    result = noise_convergence(pars)
    assert result > 0


@pytest.mark.parametrize("pars", [PARS_LSST, PARS_EUCLID])
def test_noise_convergence_order_of_magnitude(pars):
    result = noise_convergence(pars)
    # shape noise per mode should be ~1e-9 to 1e-6 sr
    assert 1e-11 < result < 1e-5


@pytest.mark.parametrize("pars_exp", [PARS_PLANCK, PARS_SO])
def test_noise_temperature_positive(pars_exp):
    result = noise_temperature(ELL, pars_exp)
    assert np.all(result > 0)


@pytest.mark.parametrize("pars_exp", [PARS_PLANCK, PARS_SO])
def test_noise_temperature_increases_with_ell(pars_exp):
    result = noise_temperature(ELL, pars_exp)
    assert np.all(np.diff(result) > 0), "Beam-deconvolution noise must increase with ell"


def test_cov_positive():
    C_sig = np.ones_like(ELL) * 1e-10
    C_TT = np.ones_like(ELL) * 1e-8
    C_kappa = np.ones_like(ELL) * 1e-9
    C_ksz = np.ones_like(ELL) * 1e-9
    result = Cov(ELL, C_sig, C_TT, C_kappa, C_ksz, PARS_LSST, PARS_PLANCK)
    assert np.all(result > 0)


def test_snr_positive_finite():
    C_sig = np.ones_like(ELL) * 1e-10
    C_TT = np.ones_like(ELL) * 1e-8
    C_kappa = np.ones_like(ELL) * 1e-9
    C_ksz = np.ones_like(ELL) * 1e-9
    result = SNR_fn(ELL, C_sig, C_TT, C_kappa, C_ksz, PARS_LSST, PARS_PLANCK)
    assert np.all(result > 0)
    assert np.all(np.isfinite(result))
