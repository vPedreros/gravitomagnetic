import warnings
from pathlib import Path

import numpy as np
from astropy import constants as const
from astropy import units
from scipy.integrate import (
    cumulative_simpson,
    cumulative_trapezoid,
    quad,
    simpson,
    trapezoid,
)
from scipy.interpolate import interp1d

c_kms = const.c.to("km/s").value  # speed of light in km/s
Mpc_2_m = units.Mpc.to(units.m)

# ==================================================
# Read parameters used in the simulation from a file.
# ==================================================


def read_params_file(path):
    """
    Read a simple 'key value' parameters file and return dict.
    Lines starting with # or empty lines are ignored.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} not found")
    d = {}
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            key = parts[0]
            # join remainder in case value contains spaces (not expected but safe)
            val = " ".join(parts[1:])
            # try to cast to float or int if possible, else keep string
            try:
                if "." in val or "e" in val.lower():
                    d[key] = float(val)
                else:
                    d[key] = int(val)
            except Exception:
                d[key] = val
    return d


def build_cosmo_params_from_file(path, extra_defaults=None):
    """
    Read simulation params and return a cosmo_params dict with derived values.
    extra_defaults: optional dict to override/add defaults (e.g. SigmaT, Yp).
    """
    sim = read_params_file(path)

    Omega_m = float(sim.get("Omega0"))
    Omega_L = float(sim.get("OmegaLambda", 1.0 - Omega_m))
    Omega_b = float(sim.get("OmegaBaryon", sim.get("OmegaB")))

    h = sim.get("HubbleParam", sim.get("h", None))
    H0 = 100.0 * h

    BoxSize = float(sim.get("BoxSize"))  # Mpc/h

    # Unit conversions (optional but useful)
    UnitLength_in_cm = float(sim.get("UnitLength_in_cm", 3.08568e24))
    UnitMass_in_g = float(sim.get("UnitMass_in_g", 1.989e43))
    UnitVelocity_in_cm_per_s = float(sim.get("UnitVelocity_in_cm_per_s", 100000.0))

    # Physical constants / defaults (you can override via extra_defaults)
    defaults = {
        "c": c_kms,  # speed of light km/s
        "m_ele": 9.11e-31,  # kg
        "m_H": 1.67e-27,  # kg
        "m_He": 6.65e-27,  # kg
        "SigmaT": const.sigma_T.value,  # m^2
        "Ombh2": Omega_b * h**2,
        "xe": 1.0,
        "Yp": 0.25,
    }
    if extra_defaults:
        defaults.update(extra_defaults)

    # derived
    params = {}
    params.update(defaults)
    params["Omega_m"] = Omega_m
    params["Omega_Lambda"] = Omega_L
    params["Omega_b"] = Omega_b
    params["h"] = h
    params["H0"] = H0
    params["BoxSize"] = BoxSize
    params["UnitLength_in_cm"] = UnitLength_in_cm
    params["UnitMass_in_g"] = UnitMass_in_g
    params["UnitVelocity_in_cm_per_s"] = UnitVelocity_in_cm_per_s

    # critical density (kg/m^3).
    params["rho_c"] = 1.88e-26 * h**2

    # tau_H and kSZ prefactor example
    params["tau_H"] = 0.07 * (1.0 - params["Yp"]) * params["Ombh2"] / params["h"]
    params["kSZ_bfac"] = params["tau_H"] * params["xe"]

    # keep original sim dict in case you need other keys
    params["simfile_raw"] = sim

    return params


import os as _os

# ============================================================================
# Cosmological parameters
# ============================================================================
#
# `parameters_sim` is the dict of cosmological parameters for the model
# currently being processed.  It is resolved lazily on first access via
# a module-level __getattr__ — importing vp_utils no longer requires
# VP_PARAMS_FILE to be set; only *using* anything that needs cosmology does.
#
# Resolution order on first access:
#   1. If a caller has injected parameters via set_parameters_sim(...), use those.
#   2. Otherwise read the path in the VP_PARAMS_FILE environment variable.
#   3. If neither is set, raise RuntimeError — there is no silent fallback.
#
# IMPORTANT: using the wrong VP_PARAMS_FILE silently propagates incorrect h,
# Omega_m, etc.  Set it explicitly per script invocation:
#
#     export VP_PARAMS_FILE=output/frhs/parameters-usedvalues
#     python powerspec.py --in-dir output/frhs/seed_2080/snap_000 ...


_parameters_sim_cache = None


def _augment_params(params):
    """Add the derived constants the pipeline expects (w_de, kF/kN modes)."""
    params["w_de"] = -1
    params["khN"] = 1024 / 500 * np.pi
    params["kN"] = params["khN"] * params["h"]
    params["khF"] = 1 / 500
    params["kF"] = params["khF"] * params["h"]
    return params


def _load_parameters_sim():
    """Resolve and cache the cosmological parameters dict."""
    global _parameters_sim_cache
    if _parameters_sim_cache is not None:
        return _parameters_sim_cache
    path = _os.environ.get("VP_PARAMS_FILE")
    if not path:
        raise RuntimeError(
            "VP_PARAMS_FILE environment variable is not set. "
            "Point it at the parameters-usedvalues file of the model you are "
            "processing, e.g.\n"
            "  export VP_PARAMS_FILE=output/lcdm/parameters-usedvalues\n"
            "Or call vp_utils.set_parameters_sim(path_or_dict) explicitly."
        )
    _parameters_sim_cache = _augment_params(build_cosmo_params_from_file(Path(path)))
    return _parameters_sim_cache


def set_parameters_sim(params_or_path):
    """Override the cached cosmological parameters.

    Accepts either a fully-built params dict (typical in tests) or a path
    to an Arepo parameters-usedvalues file.  Subsequent accesses to
    `parameters_sim` and the chi/tau interpolators will rebuild against the
    new values.
    """
    global _parameters_sim_cache, _chi_interp, _tau_interp
    if isinstance(params_or_path, dict):
        _parameters_sim_cache = _augment_params(dict(params_or_path))
    else:
        _parameters_sim_cache = _augment_params(
            build_cosmo_params_from_file(Path(params_or_path))
        )
    # Invalidate dependent caches so they rebuild against the new cosmology
    _chi_interp = None
    _tau_interp = None


def __getattr__(name):
    if name == "parameters_sim":
        return _load_parameters_sim()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

######################
# Cosmo Background functions
######################


def a_of_z(z):
    """Scale factor as a function of redshift."""
    return 1 / (1 + z)


def z_of_a(a):
    """Redshift as a function of the scale factor."""
    return 1 / a - 1


def Hubble(z, pars=None):
    """Hubble parameter in km/s/Mpc for wCDM."""
    if pars is None:
        pars = _load_parameters_sim()
    return pars["H0"] * np.sqrt(
        pars["Omega_m"] * (1.0 + z) ** 3
        + (1.0 - pars["Omega_m"]) * (1.0 + z) ** (3 * (1 + pars["w_de"]))
    )


# chi(z) and tau(z) are precomputed onto z-grids and accessed via cubic
# interpolation.  Both depend on cosmology, so the interpolators are built
# lazily on first call (and invalidated by set_parameters_sim).
_chi_interp = None
_tau_interp = None


def _build_chi_interp():
    pars = _load_parameters_sim()
    z_table = np.linspace(0, 1080, 10000)
    chi_table = np.empty_like(z_table)
    integrand = lambda x: 1 / Hubble(x, pars=pars)
    for i, zz in enumerate(z_table):
        chi_table[i] = quad(integrand, 0, zz)[0] * pars["c"]
    return interp1d(z_table, chi_table, kind="cubic", fill_value="extrapolate")


def chi_of_z(z):
    """Comoving distance as a function of redshift χ(z), in Mpc."""
    global _chi_interp
    if _chi_interp is None:
        _chi_interp = _build_chi_interp()
    return _chi_interp(z)


# =============================================
# Functions for computing angular power spectra
# =============================================


def n_ele(z, pars=None):
    """Electron number density n_e = n_H + 2 n_He, in 1/m^3."""
    if pars is None:
        pars = _load_parameters_sim()
    common = (pars["Ombh2"] / pars["h"] ** 2) * pars["rho_c"] * (1 + z) ** 3
    n_H = (1 - pars["Yp"]) * common / pars["m_H"]
    n_He = pars["Yp"] * common / pars["m_He"]
    return n_H + 2 * n_He


# Optical depth tau(z) = int_0^z SigmaT * c * Mpc_2_m * n_e(x) / (1+x) / H(x) dx.
# Built lazily as a precomputed z-grid + cubic interp (same pattern as chi_of_z)
# to avoid running scipy.quad per call inside the kSZ / B x kSZ kernels.
_TAU_Z_MAX = 10.0
_TAU_NGRID = 4000


def _build_tau_interp():
    pars = _load_parameters_sim()
    z_grid = np.linspace(0, _TAU_Z_MAX, _TAU_NGRID)
    integrand = (
        lambda x: pars["SigmaT"] * pars["c"] * Mpc_2_m
        * n_ele(x, pars) / (1 + x) / Hubble(x, pars)
    )
    tau_table = np.empty_like(z_grid)
    for i, zz in enumerate(z_grid):
        tau_table[i] = quad(integrand, 0, zz)[0]
    return interp1d(z_grid, tau_table, kind="cubic", fill_value="extrapolate")


def tau_optical_depth(z):
    """Optical depth to Thomson scattering, tau(z), dimensionless."""
    global _tau_interp
    if _tau_interp is None:
        _tau_interp = _build_tau_interp()
    return _tau_interp(z)


# =====================
# Angular power spectra
# =====================


def _resolve_pars(pars):
    """Resolve a pars argument: None -> lazy load of the module-level cosmology."""
    return _load_parameters_sim() if pars is None else pars


def _compute_C_ell(
    z_s,
    ell,
    kmin,
    kmax,
    Pk,
    kernel_fn,
    prefactor=1.0,
    name="C_ell",
    z_min=1e-5,
    Pk_evol=True,
    pars=None,
    N_int=int(1e4),
    integr_method="simpson",
):
    """
    Generic Limber integration of  Pk * kernel_fn(z, chi, chi_s, ell, pars)  in z.

    This is a shared backend for C_ell_Phi, C_ell_B, C_ell_kSZ, C_ell_B_X_kSZ.

    The `quad` branch evaluates kernel_fn on a scalar z, so the same kernel
    drives every integration method i.e. there is no separate hand-written
    integrand to keep in sync.
    """
    pars = _resolve_pars(pars)

    z_grid = np.geomspace(z_min, z_s, N_int)
    chi_grid = chi_of_z(z_grid)
    chi_s = chi_of_z(z_s)
    mask = (ell / chi_grid < kmax) & (ell / chi_grid > kmin)
    z, chi = z_grid[mask], chi_grid[mask]
    if z.size == 0:
        warnings.warn(f"{name}: No valid z found for ell={ell}. Returning 0.")
        return 0.0

    Pk_val = Pk((ell / chi, z)) if Pk_evol else Pk(ell / chi)
    integrand_arr = Pk_val * kernel_fn(z, chi, chi_s, ell, pars)

    if integr_method == "simpson":
        result = simpson(integrand_arr, x=z)
    elif integr_method == "trapezoid":
        result = trapezoid(integrand_arr, x=z)
    elif integr_method == "cumsum":
        dz = np.diff(z)
        dz = np.append(dz, dz[-1])
        result = np.sum(integrand_arr * dz)
    elif integr_method == "cum_simpson":
        result = cumulative_simpson(integrand_arr, x=z)[-1]
    elif integr_method == "cum_trapezoid":
        result = cumulative_trapezoid(integrand_arr, x=z)[-1]
    elif integr_method == "quad":

        def scalar_integrand(x):
            chi_x = chi_of_z(x)
            pk_x = Pk((ell / chi_x, x)) if Pk_evol else Pk(ell / chi_x)
            return pk_x * kernel_fn(np.asarray(x), np.asarray(chi_x), chi_s, ell, pars)

        result = quad(scalar_integrand, z[0], z[-1], limit=400)[0]
    else:
        raise ValueError(
            "Invalid selection of integration method, please choose between "
            "quad, simpson, trapezoid, cumsum, cum_simpson, cum_trapezoid"
        )
    return result * prefactor


def C_ell_Phi(
    z_s,
    ell,
    kmin,
    kmax,
    Pk,
    z_min=1e-5,
    Pk_evol=True,
    pars=None,
    N_int=int(1e4),
    integr_method="simpson",
):
    """
    Angular power spectrum of the weak-lensing convergence C_ell^{kappa kappa}.

    Evaluates the Limber approximation:

        C_ell^{kappa kappa} = (3/2 H0^2 Omega_m / c^2)^2 * c
                              * int_{z_min}^{z_s} dz
                                  P_m(ell/chi(z), z)
                                  * [(chi(z)/chi_s - 1)]^2
                                  * (1+z)^2 / H(z)

    where chi = comoving distance [Mpc], H(z) in km/s/Mpc.
    The factor (chi/chi_s - 1)^2 is the lensing kernel W(z).

    Parameters
    ----------
    z_s : float
        Source redshift.
    ell : float
        Multipole moment (dimensionless).
    kmin, kmax : float
        k range [Mpc^-1] of the input power spectrum; restricts the integration
        to chi in [ell/kmax, ell/kmin].
    Pk : callable
        Matter power spectrum.
            If Pk_evol=True, called as Pk((k, z));
            if Pk_evol=False, called as Pk(k).
            Units: Mpc^3.
    Pk_evol : bool
        Whether Pk evolves with redshift (True for multi-snapshot interpolant).
    N_int : int
        Number of z points on the integration grid.
    integr_method : str
        One of 'simpson', 'trapezoid', 'cumsum', 'cum_simpson',
        'cum_trapezoid', 'quad'.

    Returns
    -------
    float
        C_ell^{kappa kappa} [dimensionless].
    """

    pars = _resolve_pars(pars)

    def kernel(z, chi, chi_s, ell, pars):
        return (chi / chi_s - 1) ** 2 * (1 + z) ** 2 / Hubble(z, pars)

    prefactor = (1.5 * pars["H0"] ** 2 * pars["Omega_m"] / pars["c"] ** 2) ** 2 * pars[
        "c"
    ]
    return _compute_C_ell(
        z_s,
        ell,
        kmin,
        kmax,
        Pk,
        kernel,
        prefactor,
        name="C_ell_Phi",
        z_min=z_min,
        Pk_evol=Pk_evol,
        pars=pars,
        N_int=N_int,
        integr_method=integr_method,
    )


def C_ell_B(
    z_s,
    ell,
    kmin,
    kmax,
    Pk,
    z_min=1e-5,
    Pk_evol=True,
    pars=None,
    N_int=int(1e4),
    integr_method="simpson",
):
    """
    Angular power spectrum of the gravitomagnetic (vector-mode) field C_ell^{BB}.

    Evaluates the Limber approximation:

        C_ell^{BB} = (3 H0^2 Omega_m / c^3)^2 * c
                     * int_{z_min}^{z_s} dz
                         P_q_perp(ell/chi(z), z)
                         * [(chi(z)/chi_s - 1)]^2
                         * (1+z)^2 / H(z)

    where P_q_perp(k) = (1/2) * P_q(k) / k^2 is the perpendicular momentum
    power spectrum.  P_q is the input curl power spectrum in (Mpc km/s)^3,
    so k is in Mpc^-1 and the 1/k^2 converts from P_q to P_q_perp.
    The factor 1/2 arises from the two transverse polarisation modes.

    The gravitomagnetic field B is related to the momentum density field q by
    the Poisson equation in vector form: B ~ (H0^2 Omega_m / c^2) * q / k^2.

    Parameters
    ----------
    z_s : float
        Source redshift.
    ell : float
        Multipole moment (dimensionless).
    kmin, kmax : float
        k range [Mpc^-1] of the input power spectrum.
    Pk : callable
        Curl (momentum) power spectrum P_q(k, z).  Units: Mpc^3 (km/s)^2.
    Pk_evol : bool
        Whether Pk evolves with redshift.
    N_int : int
        Number of z points on the integration grid.
    integr_method : str
        One of 'simpson', 'trapezoid', 'cumsum', 'cum_simpson',
        'cum_trapezoid', 'quad'.

    Returns
    -------
    float
        C_ell^{BB} [(km/s)^2 Mpc^2] (units inherited from P_q).
    """

    pars = _resolve_pars(pars)

    def kernel(z, chi, chi_s, ell, pars):
        lens = (chi / chi_s - 1) ** 2 * (1 + z) ** 2 / Hubble(z, pars)
        return 0.5 * lens / (ell / chi) ** 2  # 1/2 from P_q; 1/k^2 -> P_q_perp

    prefactor = (3 * pars["H0"] ** 2 * pars["Omega_m"] / pars["c"] ** 3) ** 2 * pars[
        "c"
    ]
    return _compute_C_ell(
        z_s,
        ell,
        kmin,
        kmax,
        Pk,
        kernel,
        prefactor,
        name="C_ell_B",
        z_min=z_min,
        Pk_evol=Pk_evol,
        pars=pars,
        N_int=N_int,
        integr_method=integr_method,
    )


def C_ell_kSZ(
    z_s,
    ell,
    kmin,
    kmax,
    Pk,
    z_min=1e-5,
    Pk_evol=True,
    pars=None,
    N_int=int(1e4),
    integr_method="simpson",
):
    """
    Angular power spectrum of the kinetic Sunyaev-Zel'dovich (kSZ) effect C_ell^{kSZ kSZ}.

    Evaluates the Limber approximation:

        C_ell^{kSZ} = int_{z_min}^{z_s} dz
                          P_q_perp(ell/chi(z), z)
                          * [sigma_T * n_e(z) / (1+z) * exp(-tau(z))]^2
                          / (c * H(z) * chi(z)^2)

    where sigma_T is the Thomson cross-section [m^2], n_e(z) is the free
    electron number density [m^-3] (n_e / (1+z) accounts for the
    comoving volume expansion), tau(z) is the optical depth to reionisation,
    and chi(z) is the comoving distance [Mpc].

    The kSZ signal is produced by the line-of-sight peculiar velocity of
    free electrons modulated by the Thomson scattering optical depth.
    P_q_perp = (1/2) P_q / k^2 is the perpendicular momentum power spectrum.

    Parameters
    ----------
    z_s : float
        Source redshift (integration upper limit).
    ell : float
        Multipole moment (dimensionless).
    kmin, kmax : float
        k range [Mpc^-1] of the input power spectrum.
    Pk : callable
        Curl (momentum) power spectrum P_q(k, z).  Units: Mpc^3 (km/s)^2.
    Pk_evol : bool
        Whether Pk evolves with redshift.
    N_int : int
        Number of z points on the integration grid.
    integr_method : str
        One of 'simpson', 'trapezoid', 'cumsum', 'cum_simpson',
        'cum_trapezoid', 'quad'.

    Returns
    -------
    float
        C_ell^{kSZ kSZ} [dimensionless, in units of (Delta T / T)^2 * sr].
    """

    pars = _resolve_pars(pars)

    def kernel(z, chi, chi_s, ell, pars):
        thomson = (
            pars["SigmaT"]
            * Mpc_2_m
            * n_ele(z, pars)
            / (1 + z)
            * np.exp(-tau_optical_depth(z))
        ) ** 2 / pars["c"]
        return 0.5 * thomson / Hubble(z, pars) / chi**2 / (ell / chi) ** 2

    return _compute_C_ell(
        z_s,
        ell,
        kmin,
        kmax,
        Pk,
        kernel,
        prefactor=1.0,
        name="C_ell_kSZ",
        z_min=z_min,
        Pk_evol=Pk_evol,
        pars=pars,
        N_int=N_int,
        integr_method=integr_method,
    )


def C_ell_B_X_kSZ(
    z_s,
    ell,
    kmin,
    kmax,
    Pk,
    z_min=1e-5,
    Pk_evol=True,
    pars=None,
    N_int=int(1e4),
    integr_method="simpson",
):
    """
    Cross angular power spectrum of the gravitomagnetic field and the kSZ effect,
    C_ell^{B x kSZ}.  This is the primary observable we compute for the
    cosmological gravitomagnetic signal, as we use this to isolate the B
    contribution present in the full GR lensing signal (which includes the standard
    scalar contribution from Phi which dominates the total signal).

    Evaluates the Limber approximation:

        C_ell^{B x kSZ} = 3 (H0^2 Omega_m / c^3) * sigma_T
                          * int_{z_min}^{z_s} dz
                              P_q_perp(ell/chi(z), z)
                              * n_e(z) * exp(-tau(z))
                              * (chi_s - chi) / (chi_s * chi)
                              / H(z)

    The cross-kernel combines the lensing kernel weight (chi_s - chi)/(chi_s * chi)
    from the B-field projection with the Thomson scattering weight n_e * exp(-tau)
    from the kSZ projection.  Because both signals trace the same momentum field
    q, their cross-spectrum is non-zero and provides a direct probe of the
    gravitomagnetic vector mode.

    Parameters
    ----------
    z_s : float
        Source redshift (sets the lensing weight to zero at z_s).
    ell : float
        Multipole moment (dimensionless).
    kmin, kmax : float
        k range [Mpc^-1] of the input power spectrum.
    Pk : callable
        Curl (momentum) power spectrum P_q(k, z).  Units: Mpc^3 (km/s)^2.
    Pk_evol : bool
        Whether Pk evolves with redshift.
    N_int : int
        Number of z points on the integration grid.
    integr_method : str
        One of 'simpson', 'trapezoid', 'cumsum', 'cum_simpson',
        'cum_trapezoid', 'quad'.

    Returns
    -------
    float
        C_ell^{B x kSZ} in mixed units [(km/s) * (Delta T / T) * Mpc * sr^{1/2}].
    """

    pars = _resolve_pars(pars)

    def kernel(z, chi, chi_s, ell, pars):
        weight = (
            n_ele(z, pars)
            * np.exp(-tau_optical_depth(z))
            * (chi_s - chi)
            / (chi_s * chi)
        )
        return 0.5 * weight / Hubble(z, pars) / (ell / chi) ** 2

    prefactor = (
        3
        * (pars["H0"] ** 2 / pars["c"] ** 3)
        * pars["Omega_m"]
        * pars["SigmaT"]
        * Mpc_2_m
    )
    return _compute_C_ell(
        z_s,
        ell,
        kmin,
        kmax,
        Pk,
        kernel,
        prefactor,
        name="C_ell_B_X_kSZ",
        z_min=z_min,
        Pk_evol=Pk_evol,
        pars=pars,
        N_int=N_int,
        integr_method=integr_method,
    )


def C_ell_XY(
    z_s,
    ell,
    kmin,
    kmax,
    Pk,
    type_XY,
    z_min=1e-5,
    Pk_evol=True,
    pars=None,
    N_int=int(1e4),
    integr_method="simpson",
):
    """
    Wrapper that dispatches to C_ell_Phi, C_ell_B, C_ell_kSZ, or C_ell_B_X_kSZ
    based on the `type_XY` string.
    """
    dispatch = {
        "Phi": C_ell_Phi,
        "B": C_ell_B,
        "kSZ": C_ell_kSZ,
        "B_X_kSZ": C_ell_B_X_kSZ,
    }
    try:
        fn = dispatch[type_XY]
    except KeyError:
        raise ValueError(
            f"Unknown type_XY: {type_XY}. Choose from 'Phi', 'B', 'kSZ', 'B_X_kSZ'."
        )
    return fn(z_s, ell, kmin, kmax, Pk, z_min, Pk_evol, pars, N_int, integr_method)


