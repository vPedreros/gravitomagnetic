import numpy as np
from pathlib import Path
from scipy.integrate import quad, simpson, trapezoid, cumulative_trapezoid, cumulative_simpson
from scipy.interpolate import interp1d
from astropy import constants as const
from astropy import units


c_kms = const.c.to('km/s').value  # speed of light in km/s
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
                if '.' in val or 'e' in val.lower():
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
    UnitMass_in_g = float(sim.get("UnitMass_in_g", 1.989e+43))
    UnitVelocity_in_cm_per_s = float(sim.get("UnitVelocity_in_cm_per_s", 100000.0))

    # Physical constants / defaults (you can override via extra_defaults)
    defaults = {
        "c": c_kms,          # speed of light km/s
        "m_ele": 9.11E-31, # kg
        "m_H": 1.67E-27,  # kg
        "m_He": 6.65E-27, # kg (approx)
        "SigmaT": const.sigma_T.value, # m^2
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

parameters_sim = build_cosmo_params_from_file("parameters-usedvalues")

parameters_sim['w_de'] = -1
parameters_sim['khN'] = 1024/500*np.pi
parameters_sim['kN'] = parameters_sim['khN']*parameters_sim['h']

parameters_sim['khF'] = 1/500
parameters_sim['kF'] = parameters_sim['khF']*parameters_sim['h']

######################
# Background functions
######################


def a_of_z(z):
    """
    Scale factor as a function of redshift.
    """
    return 1/(1+z)


def z_of_a(a):
    """
    Redshift as a function of the scale factor.
    """
    return 1/a - 1


def Hubble(z, pars=parameters_sim):
    """
    Hubble parameter in km/s/Mpc for wCDM.
    """
    # CLASS['const_z'].Omega_r()*(1.+z)**4 + 
    H = pars['H0'] * np.sqrt(pars['Omega_m']*(1.+z)**3+(1.-pars['Omega_m'])*(1. + z)**(3*(1+pars['w_de'])))
    return H


z_table = np.linspace(0, 1080, 10000)
chi_table = np.zeros_like(z_table)

pars = parameters_sim
integrand = lambda x: 1 / Hubble(x, pars=parameters_sim)

for i, zz in enumerate(z_table):
    chi_table[i] = quad(integrand, 0, zz)[0] * pars['c']

chi_interp = interp1d(z_table, chi_table, kind="cubic", fill_value="extrapolate")

def chi_of_z(z):
    """
    Comoving distance as a function in redshift χ(z), in Mpc/h.
    """
    return chi_interp(z)



# =============================================
# Functions for computing angular power spectra
# =============================================


def n_ele(z, pars=parameters_sim):
    """
    Electron number density n_e = n_H + 2*n_He, in 1/m^3
    """
    n_H  = (1-pars['Yp'])*((pars['Ombh2']/pars['h']**2)*pars['rho_c'])/pars['m_H'] *(1+z)**3
    n_He =    pars['Yp'] *((pars['Ombh2']/pars['h']**2)*pars['rho_c'])/pars['m_He']*(1+z)**3
    n_ele= n_H + 2*n_He
    return n_ele


def tau_optical_depth(z, pars=parameters_sim):
    """
    [tau] = dimensionless
    Integration in terms of dz=H*dr
    """
    tau_optical_depth_int = lambda x: pars['SigmaT']*pars['c']*Mpc_2_m*n_ele(x, pars)*(1+x)**2/Hubble(x, pars)
    return quad(tau_optical_depth_int, 0, z)[0]

tau_optical_depth = np.vectorize(tau_optical_depth)


# =====================
# Angular power spectra
# =====================


def C_ell_Phi(z_s, ell, kmin, kmax, Pk, z_min=1e-5, Pk_evol=False, pars=parameters_sim, N_int=int(1e4), integr_method='simpson'):
    """
    """
    z_grid = np.geomspace(z_min, z_s, N_int)
    chi_grid = chi_of_z(z_grid)
    chi_s = chi_of_z(z_s)
    # restrict to valid chi by k-range
    mask = (ell/chi_grid < kmax) & (ell/chi_grid > kmin)
    z = z_grid[mask]
    chi = chi_grid[mask]
    if z.size == 0:
        import warnings
        warnings.warn(f"C_ell_PhiPhi: No valid z found for ell={ell}. Returning 0.")
        return 0.0
    if Pk_evol:
        C_ell_int = Pk((ell/chi , z))
    else:
        C_ell_int = Pk(ell/chi)

    C_ell_int *= (chi/chi_s-1)**2
    C_ell_int *= (1 + z)**2 / Hubble(z, pars)

    # integrate in z
    if integr_method == 'simpson':
        C_ell = simpson(C_ell_int, x=z)
    elif integr_method == 'cumsum':
        dz = np.diff(z)
        dz = np.append(dz, dz[-1])  # pad last element for same length
        C_ell = np.sum(C_ell_int * dz)
    elif integr_method == 'quad':
        def integrand(x):
            chi_x = chi_of_z(x)
            chi_s = chi_of_z(z_s)
            val = Pk((ell/chi_x, x)) if Pk_evol else Pk(ell/chi_x)
            val *= (chi_x/chi_s-1)**2
            val *= (1 + x)**2 / Hubble(x, pars)
            return val
        C_ell = quad(integrand, z[0], z[-1], limit=400)[0]
    elif integr_method == 'trapezoid':
        C_ell = trapezoid(C_ell_int, x=z)
    elif integr_method == 'cum_simpson':
        C_ell = cumulative_simpson(C_ell_int, x=z)[-1]
    elif integr_method == 'cum_trapezoid':
        C_ell = cumulative_trapezoid(C_ell_int, x=z)[-1]
    else: 
        raise('Invalid selection of integration method, please choose between quad, simpson, cumsum or trapz')
    return C_ell * 9/4 * pars['H0']**4 * pars['Omega_m']**2 / pars['c']**3


def C_ell_B(z_s, ell, kmin, kmax, Pk, z_min=1e-5, Pk_evol=False, pars=parameters_sim, N_int=int(1e4), integr_method='simpson'):
    return (4/pars['c']**2) * C_ell_Phi(z_s, ell, kmin, kmax, Pk, z_min, Pk_evol, pars, N_int, integr_method)


def C_ell_kSZ(z_s, ell, kmin, kmax, Pk, z_min=1e-5, Pk_evol=False, pars=parameters_sim, N_int=int(1e4), integr_method='simpson'):
    """
    """
    z_grid = np.geomspace(z_min, z_s, N_int)
    chi_grid = chi_of_z(z_grid)
    # restrict to valid chi by k-range
    mask = (ell/chi_grid < kmax) & (ell/chi_grid > kmin)
    z = z_grid[mask]
    chi = chi_grid[mask]
    if z.size == 0:
        import warnings
        warnings.warn(f"C_ell_lensing: No valid z found for ell={ell}. Returning 0.")
        return 0.0

    C_ell_int = Pk((ell/chi, z)) if Pk_evol else Pk(ell/chi)
    C_ell_int *= (pars['SigmaT']*Mpc_2_m*n_ele(z)*(1+z)**2*np.exp(-tau_optical_depth(z)))**2/pars['c']
    C_ell_int /= Hubble(z, pars)
    C_ell_int /= chi**2

    # integrate in z
    if integr_method == 'simpson':
        C_ell = simpson(C_ell_int, x=z)
    elif integr_method == 'cumsum':
        dz = np.diff(z)
        dz = np.append(dz, dz[-1])  # pad last element for same length
        C_ell = np.sum(C_ell_int * dz)
    elif integr_method == 'quad':
        def integrand(x):
            chi_x = chi_of_z(x)
            val = Pk((ell/chi_x, x)) if Pk_evol else Pk(ell/chi_x)
            val *= (pars['SigmaT']*Mpc_2_m*n_ele(x)*(1+x)**2*np.exp(-2*tau_optical_depth(x)))**2/pars['c']
            val /= Hubble(x, pars)
            val /= chi_x**2
            return val
        C_ell = quad(integrand, z[0], z[-1], limit=400)[0]
    elif integr_method == 'trapezoid':
        C_ell = trapezoid(C_ell_int, x=z)
    elif integr_method == 'cum_simpson':
        C_ell = cumulative_simpson(C_ell_int, x=z)[-1]
    elif integr_method == 'cum_trapezoid':
        C_ell = cumulative_trapezoid(C_ell_int, x=z)[-1]
    else: 
        raise('Invalid selection of integration method, please choose between quad, simpson, cumsum or trapz')
    return C_ell


def C_ell_B_X_kSZ(z_s, ell, kmin, kmax, Pk, z_min=1e-5, Pk_evol=False, pars=parameters_sim, N_int=int(1e4), integr_method='simpson'):
    """
    """
    z_grid = np.geomspace(z_min, z_s, N_int)
    chi_grid = chi_of_z(z_grid)
    chi_s = chi_of_z(z_s)
    # restrict to valid chi by k-range
    mask = (ell/chi_grid < kmax) & (ell/chi_grid > kmin)
    z = z_grid[mask]
    chi = chi_grid[mask]
    if z.size == 0:
        import warnings
        warnings.warn(f"C_ell_lensing: No valid z found for ell={ell}. Returning 0.")
        return 0.0

    C_ell_int = Pk((ell/chi, z)) if Pk_evol else Pk(ell/chi)
    C_ell_int *= 3*(pars['H0']**2/pars['c']**3) * pars['Omega_m']*pars['SigmaT']*Mpc_2_m
    C_ell_int *= n_ele(z)*(1+z)**3 * np.exp(-tau_optical_depth(z)) * (chi_s - chi)/(chi_s*chi)
    C_ell_int /= Hubble(z, pars)

    # integrate in z
    if integr_method == 'simpson':
        C_ell = simpson(C_ell_int, x=z)
    elif integr_method == 'cumsum':
        dz = np.diff(z)
        dz = np.append(dz, dz[-1])  # pad last element for same length
        C_ell = np.sum(C_ell_int * dz)
    elif integr_method == 'quad':
        def integrand(x):
            chi_x = chi_of_z(x)
            val = Pk((ell/chi_x, x)) if Pk_evol else Pk(ell/chi_x)
            val *= 3*(pars['H0']**2/pars['c']**3) * pars['Omega_m']*pars['SigmaT']*Mpc_2_m
            val *= n_ele(x)*(1+x)**3 * np.exp(-tau_optical_depth(x)) * (chi_s - chi_x)/(chi_s*chi_x)
            val /= Hubble(x, pars)
            return val
        C_ell = quad(integrand, z[0], z[-1], limit=400)[0]
    elif integr_method == 'trapezoid':
        C_ell = trapezoid(C_ell_int, x=z)
    elif integr_method == 'cum_simpson':
        C_ell = cumulative_simpson(C_ell_int, x=z)[-1]
    elif integr_method == 'cum_trapezoid':
        C_ell = cumulative_trapezoid(C_ell_int, x=z)[-1]
    else: 
        raise('Invalid selection of integration method, please choose between quad, simpson, cumsum or trapz')
    return C_ell


def C_ell_XY(z_s, ell, kmin, kmax, Pk, type_XY, z_min=1e-5, Pk_evol=False, pars=parameters_sim, N_int=int(1e4), integr_method='simpson'):
    if type_XY == 'Phi':
        return C_ell_Phi(z_s, ell, kmin, kmax, Pk, z_min, Pk_evol, pars, N_int, integr_method)
    elif type_XY == 'B':
        return C_ell_B(z_s, ell, kmin, kmax, Pk, z_min, Pk_evol, pars, N_int, integr_method)
    elif type_XY == 'kSZ':
        return C_ell_kSZ(z_s, ell, kmin, kmax, Pk, z_min, Pk_evol, pars, N_int, integr_method)
    elif type_XY == 'B_X_kSZ':
        return C_ell_B_X_kSZ(z_s, ell, kmin, kmax, Pk, z_min, Pk_evol, pars, N_int, integr_method)
    else:
        raise ValueError(f"Unknown type_XY: {type_XY}. Choose from 'PhiPhi', 'BB', 'kSZ', 'B_X_kSZ'.")