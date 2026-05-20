import numpy as np
from pathlib import Path
import pandas as pd
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute angular power spectrum from a snapshot using Pylians3."
    )
    parser.add_argument("--in-dir", type=str, required=True, help="Path containing .npy files and metadata.")
    parser.add_argument("--out-dir", type=str, default="outputs", help="Output directory for .npy files.")
    parser.add_argument("--z_source", type=float, required=True, help="Maximum redshift for integration")
    parser.add_argument("--survey", type=str, required=True, help="Name of the survey (LSST or Euclid)")
    parser.add_argument("--cmb-exp", type=str, required=True, help="Name of the CMB experiment (Planck or SO)")
    return parser.parse_args()

args = parse_args()

arcmin_to_rad = np.pi / (180*60) # Unit conversion

base_path = Path(args.in_dir).expanduser()

models = ['lcdm', 'frhs', 'ndgp']
surveys = ['LSST', 'Euclid']
experiments = ['Planck', 'SO']

col_names_frhs = ['Omega_m', 'S8', 'h', '|f_R0|', 'sigma8', 'A_s', 'B0']
col_names_ndgp = ['Omega_m', 'S8', 'h', 'H0rc', 'sigma8', 'A_s', 'B0']

df_frhs = pd.read_csv(base_path / 'frhs/Nodes_Omm-S8-h-fR0-sigma8-As-B0_LHCrandommaximin_Seed1_Nodes50_Dim4_AddFidTrue_extended.dat', sep=r'\s+', skiprows=1, names=col_names_frhs, skipfooter=2, engine='python')
df_ndgp = pd.read_csv(base_path / 'ndgp/Nodes_Omm-S8-h-H0rc-sigma8-As-B0_LHCrandommaximin_Seed1_Nodes50_Dim4_AddFidTrue_extended_modified.dat', sep=r'\s+', skiprows=1, names=col_names_ndgp, skipfooter=2, engine='python')

df_frhs[['A_s_1', 'A_s_2']] = df_frhs['A_s'].str.split('/', expand=True).astype(float)
df_ndgp[['A_s_1', 'A_s_2']] = df_ndgp['A_s'].str.split('/', expand=True).astype(float)

df_frhs = df_frhs.drop(columns=['A_s'])
df_ndgp = df_ndgp.drop(columns=['A_s'])

Omega_b = 0.049199
n_s = 0.9652

# Find C_ell_TT from CLASS for the different models
## Cosmological parameters
params_base = {
    'output': 'tCl,sCl,lCl,mPk',
    'Omega_b': Omega_b,
    'n_s': n_s,
    'lensing': 'yes',
    'l_switch_limber': 50,
    'l_max_scalars': 10000
}

params_lcdm = params_base | {
    'h': df_frhs['h'][0],
    'Omega_cdm': df_frhs['Omega_m'][0] - Omega_b,
    'A_s': df_frhs['A_s_2'][0],
}

params_frhs = params_base | {
    'h': df_frhs['h'][6],
    'Omega_cdm': df_frhs['Omega_m'][6] - Omega_b,
    'A_s': df_frhs['A_s_2'][6],
}

params_ndgp = params_base | {
    'h': df_ndgp['h'][6],
    'Omega_cdm': df_ndgp['Omega_m'][6] - Omega_b,
    'A_s': df_ndgp['A_s_2'][6],
}


## CLASS run
CLASS = {}
C_ell_TT = {}
ells = {}

params_by_model = {
    'lcdm': params_lcdm,
    'frhs': params_frhs,
    'ndgp': params_ndgp,
}

for m in models:
    cache = base_path / m / "Cl_TT.npy"
    if cache.exists():
        print(f"[{m}] Loading cached Cl_TT from {cache}")
        C_ell_TT[m] = np.load(cache)
        ells[m] = np.arange(len(C_ell_TT[m]))
    else:
        from classy import Class
        CLASS[m] = Class()
        CLASS[m].set(params_by_model[m])
        CLASS[m].compute()
        c_ell = CLASS[m].lensed_cl(10000)
        ells[m] = c_ell['ell']
        C_ell_TT[m] = c_ell['tt']
        np.save(cache, C_ell_TT[m])


# Define the survey and experiment parmeters
pars_surv, pars_exp = {}, {}
pars_surv['Euclid'], pars_surv['LSST'] = {}, {}
pars_exp['Planck'], pars_exp['SO'] = {}, {}

pars_surv['Euclid']['n_gal'] = 30
pars_surv['Euclid']['sigma_e'] = np.sqrt(2) * 0.21  # Updated from 2302.04507 (old 0.22)
pars_surv['Euclid']['f_sky'] = 0.36

pars_surv['LSST']['n_gal'] = 40
pars_surv['LSST']['sigma_e'] = 0.22
pars_surv['LSST']['f_sky'] = 0.5

pars_exp['Planck']['theta_fwhm'] = 5
pars_exp['Planck']['Delta_T'] = 3.1
pars_exp['Planck']['f_sky'] = 0.82
pars_exp['Planck']['T_bar'] = 2.7E6

pars_exp['SO']['theta_fwhm'] = 1.4
pars_exp['SO']['Delta_T'] = 6
pars_exp['SO']['f_sky'] = 0.4
pars_exp['SO']['T_bar'] = 2.7E6


# Useful functions for SNR
def noise_convergence(pars_surv):
    sigma_e = pars_surv['sigma_e']
    n_gal = pars_surv['n_gal']  # In 1/arcmin^2
    n_gal /= arcmin_to_rad**2   # convert to 1/radians^2
    return sigma_e**2/n_gal


def noise_temperature(ell, pars_exp):
    FWHM = pars_exp['theta_fwhm'] * arcmin_to_rad
    Delta_T = pars_exp['Delta_T'] * arcmin_to_rad
    factor = (Delta_T/pars_exp['T_bar'])**2
    arg_exp = ell**2 * FWHM**2/(8*np.log(2))
    return factor * np.exp(arg_exp)


def Cov(ell_list, C_ell_B_X_kSZ, C_ell_TT, C_ell_kappaWL, C_ell_kSZ, pars_surv, pars_exp):
    logell = np.log10(ell_list)
    dlog = logell[1] - logell[0]
    edges = 10**(np.r_[logell[0] - dlog/2, 0.5*(logell[:-1] + logell[1:]), logell[-1] + dlog/2])
    Delta_ell = np.diff(edges)

    factor = Delta_ell * pars_surv['f_sky'] * (2*ell_list + 1)
    contributions = C_ell_B_X_kSZ**2 + (C_ell_TT + C_ell_kSZ + noise_temperature(ell_list, pars_exp))*(C_ell_kappaWL + noise_convergence(pars_surv))
    return contributions / factor


def SNR(ell_list, C_ell_B_X_kSZ, C_ell_TT, C_ell_kappaWL, C_ell_kSZ, survey, experiment):
    return np.sqrt(C_ell_B_X_kSZ**2 / Cov(ell_list, C_ell_B_X_kSZ, C_ell_TT, C_ell_kappaWL, C_ell_kSZ, pars_surv[survey], pars_exp[experiment]))


def main():
    out_base = Path(args.out_dir).expanduser()
    for m in models:
        path_C_ell = base_path / m

        C_ell_XY = np.load(path_C_ell / 'C_ells' / f"C_ells_XY_z={args.z_source}.npy", allow_pickle=True).item()
        ell_grid = np.load(path_C_ell / 'C_ells' / f"ell_grid_z={args.z_source}.npy")
        ell_idx = np.round(ell_grid).astype(int)

        signal_to_noise = SNR(ell_grid, C_ell_XY['B_X_kSZ'], C_ell_TT[m][ell_idx], C_ell_XY['Phi'], C_ell_XY['kSZ'], args.survey, args.cmb_exp)

        out = out_base / m / "SNRs" / f"{args.survey}_{args.cmb_exp}"
        out.mkdir(parents=True, exist_ok=True)

        np.save(out / f"SNR_z={args.z_source}.npy", signal_to_noise)
        
if __name__ == "__main__":
    main()
