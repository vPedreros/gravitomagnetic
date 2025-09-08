"""
angular_cross_spectra.py

Self-contained utilities to compute angular cross spectra

    C_ell^{XY} = (1/2) \int d\chi \; \chi^{-2} K_X(\chi) K_Y(\chi) P_{q_\perp}(k=\ell/\chi, z(\chi))

with kernels

    K_{κ_B}(\chi) = (3/2) H0^2 Ω_m a(\chi)^{-1} \int_0^{\chi} d\chi' \frac{\chi' (\chi - \chi')}{\chi} \frac{d\chi'}{dz} p(z(\chi'))

    K_b(\chi) = (σ_T n̄_{e,0} / c) a(\chi)^{-2} e^{-\tau(\chi)}

The lensing kernel simplifies for a delta source distribution p(z) = δ_D(\chi' - \chi_s):

    K_{κ_B}(\chi) = 0                      for \chi < \chi_s
                   = (3/2) H0^2 Ω_m a(\chi)^{-1} [\chi_s (\chi - \chi_s)/\chi] * (d\chi/dz)|_{\chi_s}
                     = (3/2) H0^2 Ω_m a(\chi)^{-1} [\chi_s (\chi - \chi_s)/\chi] * c / H(z_s)

Assumptions & units
-------------------
- Flat ΛCDM (Ω_Λ = 1 - Ω_m). Extend as needed.
- Distances are in Mpc, H in km/s/Mpc, c in km/s. Then d\chi/dz = c/H has units of Mpc.
- K_b carries σ_T n̄_{e,0}/c; provide n̄_{e,0} in m^{-3} or cm^{-3} consistently with σ_T and c if you prefer SI/CGS.
  If you keep everything in cosmology-friendly mixed units, ensure P(k,z) is consistent with your choice.

You provide P(k, z) as a callable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple, Iterable

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d


# -----------------------------
# Cosmology helpers
# -----------------------------
@dataclass
class Cosmology:
    H0: float = 70.0                 # km/s/Mpc
    Omega_m: float = 0.3
    c: float = 299_792.458           # km/s

    @property
    def Omega_L(self) -> float:
        return 1.0 - self.Omega_m


def H_z(z: float | np.ndarray, cosmo: Cosmology) -> np.ndarray:
    """Hubble parameter H(z) in km/s/Mpc for flat ΛCDM."""
    z = np.asarray(z)
    return cosmo.H0 * np.sqrt(cosmo.Omega_m * (1.0 + z) ** 3 + cosmo.Omega_L)


def dchi_dz(z: float | np.ndarray, cosmo: Cosmology) -> np.ndarray:
    """dχ/dz = c / H(z), in Mpc."""
    return cosmo.c / H_z(z, cosmo)


def chi_of_z(z: float, cosmo: Cosmology) -> float:
    """Comoving distance χ(z) in Mpc via 1D quadrature."""
    integrand = lambda zp: dchi_dz(zp, cosmo)
    chi, _ = quad(integrand, 0.0, float(z), epsabs=0, epsrel=1e-6, limit=256)
    return chi


def build_chi_z_interpolators(z_max: float, cosmo: Cosmology, npts: int = 4096) -> Tuple[Callable[[float | np.ndarray], np.ndarray], Callable[[float | np.ndarray], np.ndarray]]:
    """Return (chi_of_z_interp, z_of_chi_interp) callables.

    The forward χ(z) is computed on a dense grid and cubic-spline interpolated; z(χ) is the inverse spline.
    """
    z_grid = np.linspace(0.0, z_max, npts)
    chi_grid = np.empty_like(z_grid)
    chi_val = 0.0
    # cumulative trapezoid using dχ/dz to cheaply build a smooth, monotonic χ(z)
    dz = z_grid[1] - z_grid[0]
    chi_grid[0] = 0.0
    for i in range(1, npts):
        z_mid = 0.5 * (z_grid[i] + z_grid[i - 1])
        chi_val += dchi_dz(z_mid, cosmo) * dz
        chi_grid[i] = chi_val

    chi_of_z_interp = interp1d(z_grid, chi_grid, kind="cubic", bounds_error=False, fill_value="extrapolate")
    z_of_chi_interp = interp1d(chi_grid, z_grid, kind="cubic", bounds_error=False, fill_value="extrapolate")
    return chi_of_z_interp, z_of_chi_interp


def a_of_chi(chi: float | np.ndarray, z_of_chi: Callable, /) -> np.ndarray:
    z = np.asarray(z_of_chi(chi))
    return 1.0 / (1.0 + z)


# -----------------------------
# Kernels
# -----------------------------

def K_kappaB_general(
    chi: float | np.ndarray,
    *,
    cosmo: Cosmology,
    z_of_chi: Callable[[float | np.ndarray], np.ndarray],
    p_of_z: Callable[[float | np.ndarray], np.ndarray],
    n_eval: int = 400,
) -> np.ndarray:
    """General gravitomagnetic convergence kernel for arbitrary source distribution p(z).

    Implements
        K_{κ_B}(χ) = (3/2) H0^2 Ω_m a(χ)^{-1} \int_0^{χ} dχ' [ χ'(χ-χ')/χ ] (dχ'/dz) p(z(χ'))

    Parameters
    ----------
    chi : scalar or array of comoving distances (Mpc)
    cosmo : Cosmology
    z_of_chi : callable mapping χ -> z
    p_of_z : normalized source redshift distribution p(z)
    n_eval : # of χ' samples (Simpson-like composite rule). Increase if you need higher accuracy.
    """
    chi = np.atleast_1d(chi)
    out = np.zeros_like(chi, dtype=float)

    for j, ch in enumerate(chi):
        if ch <= 0:
            out[j] = 0.0
            continue
        a = 1.0 / (1.0 + float(z_of_chi(ch)))

        # integrate over χ' from 0..χ
        xp = np.linspace(0.0, ch, n_eval)
        zp = z_of_chi(xp)
        weight = (xp * (ch - xp) / max(ch, 1e-300)) * dchi_dz(zp, cosmo) * p_of_z(zp)
        integral = np.trapz(weight, xp)

        out[j] = 1.5 * (cosmo.H0 ** 2) * cosmo.Omega_m * (a ** -1) * integral

    return out if out.shape != () else out.item()


def K_kappaB_delta(
    chi: float | np.ndarray,
    *,
    cosmo: Cosmology,
    chi_s: float,
    z_of_chi: Callable[[float | np.ndarray], np.ndarray],
) -> np.ndarray:
    """Delta-source gravitomagnetic kernel.

    For χ < χ_s, returns 0. Otherwise
        K_{κ_B}(χ) = (3/2) H0^2 Ω_m a(χ)^{-1} [χ_s (χ - χ_s)/χ] * (c / H(z_s)).
    """
    chi = np.atleast_1d(chi).astype(float)
    out = np.zeros_like(chi)

    z_s = float(z_of_chi(chi_s))
    Hs = H_z(z_s, cosmo)
    pref_J = cosmo.c / Hs  # this is (dχ/dz)|_{χ_s}

    mask = chi > chi_s
    if np.any(mask):
        a_ch = 1.0 / (1.0 + z_of_chi(chi[mask]))
        geom = (chi_s * (chi[mask] - chi_s) / chi[mask])
        out[mask] = 1.5 * (cosmo.H0 ** 2) * cosmo.Omega_m * (a_ch ** -1) * geom * pref_J

    return out if out.shape != () else out.item()


def K_b(
    chi: float | np.ndarray,
    *,
    cosmo: Cosmology,
    z_of_chi: Callable[[float | np.ndarray], np.ndarray],
    sigma_T: float,            # e.g. 6.6524587321e-25 cm^2 (if using CGS)
    n_e0: float,               # mean electron density today, consistent with sigma_T units
    tau_of_chi: Callable[[float | np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """kSZ kernel K_b(χ) = (σ_T n̄_{e,0} / c) a(χ)^{-2} e^{-τ(χ)}.

    Ensure your (σ_T, n_e0, c) are in a consistent system of units.
    """
    chi = np.atleast_1d(chi)
    a = a_of_chi(chi, z_of_chi)
    tau = np.zeros_like(chi) if tau_of_chi is None else np.asarray(tau_of_chi(chi))
    # Note: here c is the *same* c used elsewhere. If σ_T and n_e0 are CGS, convert c accordingly.
    return (sigma_T * n_e0 / cosmo.c) * (a ** -2.0) * np.exp(-tau)


# -----------------------------
# Angular cross spectrum
# -----------------------------

def C_ell_cross(
    ells: Iterable[float] | np.ndarray,
    *,
    chi_min: float,
    chi_max: float,
    KX: Callable[[float | np.ndarray], np.ndarray],
    KY: Callable[[float | np.ndarray], np.ndarray],
    Pkz: Callable[[float | np.ndarray, float | np.ndarray], np.ndarray],
    z_of_chi: Callable[[float | np.ndarray], np.ndarray],
    n_chi: int = 800,
) -> np.ndarray:
    """Compute C_ell^{XY} via LOS integral with a simple composite trapezoid.

        C_ell = (1/2) ∫ dχ χ^{-2} KX(χ) KY(χ) P(k=ℓ/χ, z(χ))

    Parameters
    ----------
    ells : array of multipoles
    chi_min, chi_max : integration bounds in Mpc
    KX, KY : kernel callables accepting χ-array
    Pkz : callable returning P(k, z)
    z_of_chi : χ -> z mapping
    n_chi : number of χ samples (increase if you need more precision)
    """
    ells = np.atleast_1d(ells).astype(float)

    # χ grid (avoid χ=0)
    chi = np.linspace(max(1e-6, chi_min), chi_max, n_chi)
    z = z_of_chi(chi)
    chi_inv2 = 1.0 / (chi ** 2)

    KX_vals = KX(chi)
    KY_vals = KY(chi)

    out = np.empty_like(ells)
    for i, ell in enumerate(ells):
        k = ell / chi  # Limber
        Pvals = Pkz(k, z)
        integrand = chi_inv2 * KX_vals * KY_vals * Pvals
        out[i] = 0.5 * np.trapz(integrand, chi)

    return out


# -----------------------------
# Example wiring (replace Pkz with your power spectrum)
# -----------------------------
if __name__ == "__main__":
    cosmo = Cosmology(H0=70.0, Omega_m=0.3)

    # Build mappings up to z_max capturing your χ-range of interest
    chi_of_z_interp, z_of_chi_interp = build_chi_z_interpolators(z_max=6.0, cosmo=cosmo)

    # Choose a source plane at z_s and compute χ_s
    z_s = 1.0
    chi_s = float(chi_of_z_interp(z_s))

    # Define kernels
    KX = lambda chi: K_kappaB_delta(chi, cosmo=cosmo, chi_s=chi_s, z_of_chi=z_of_chi_interp)

    # Example τ(χ) ≈ 0 (transparent) and placeholder numbers for σ_T, n_e0
    sigma_T = 6.6524587321e-25  # cm^2
    n_e0 = 2e-7                 # cm^{-3} (toy value; set from your baryon model)

    KY = lambda chi: K_b(chi, cosmo=cosmo, z_of_chi=z_of_chi_interp, sigma_T=sigma_T, n_e0=n_e0, tau_of_chi=None)

    # Dummy power spectrum for demonstration only (replace with your Pq⊥)
    def Pkz_demo(k, z):
        k = np.asarray(k)
        return (k + 1e-6) ** -1.5 * (1 + z) ** -0.5

    # ℓ-range and χ-limits
    ells = np.unique(np.geomspace(30, 3000, 32).astype(int))
    chi_min, chi_max = 1.0, float(chi_of_z_interp(5.0))

    C_ell = C_ell_cross(
        ells,
        chi_min=chi_min,
        chi_max=chi_max,
        KX=KX,
        KY=KY,
        Pkz=Pkz_demo,  # <-- plug your power spectrum here
        z_of_chi=z_of_chi_interp,
        n_chi=1200,
    )

    # Quick printout
    for L, C in zip(ells[:5], C_ell[:5]):
        print(f"ell={L:4d}  C_ell={C:.3e}")