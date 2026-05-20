# Gravitomagnetic lensing x kSZ Pipeline

[![tests](https://github.com/crisbh/gravitomagnetic/actions/workflows/tests.yml/badge.svg)](https://github.com/crisbh/gravitomagnetic/actions/workflows/tests.yml)

Pipeline computing the signal-to-noise ratio (SNR) for detecting the cosmological **gravitomagnetic field B** (vector-mode, frame-dragging component of the metric) via the cross-correlation of weak-lensing convergence ($\kappa$) and the kinetic Sunyaev-Zel'dovich (kSZ) effect on the CMB, for $\Lambda$CDM, Hu-Sawicki $f(R)$ and nDGP modified gravity.

The detection statistic is $C_\ell^{B \times \mathrm{kSZ}}$. The gravitomagnetic field is sourced by the transverse (curl) component of the momentum field $\mathbf{q} = (1+\delta)\mathbf{v}$, which also sources kSZ — hence the cross-correlation.

---

## Repository structure

```
python/                   Core pipeline scripts and utilities
  vp_utils.py             Cosmological functions and C_ell kernels (central library)
  read_snap.py            Step 1 — extract particles from Arepo HDF5
  fields.py               Step 2 — CIC density/momentum grid (Pylians3)
  powerspec.py            Step 3 — 3D matter and momentum-curl P(k)
  averaging_powerspec.py  Step 3b — average over seeds, optional h-correction
  angular_powerspec_z.py  Step 4 — multi-redshift C_ell via Limber
  SNR.py                  Step 5 — covariance and cumulative SNR (CLASS for C_ell^TT)
  plot_powerspectra.py    P_m(k), P_curl(k) vs redshift
  plot_cells.py           Any of C_ell^{kk, BB, kSZ, B x kSZ}
  plot_snr.py             Cumulative-vs-z, per-ell, cumulative-vs-ell, colorbar
  plot_utils.py           Shared matplotlib style
  tests/                  pytest suite (60 tests)
scripts/                  SLURM batch scripts for COSMA HPC
parameters/               LHS node files for f(R) and nDGP
output/, imgs/, reports/  Pipeline outputs, figures, LaTeX reports
legacy/                   Deprecated code, old audit suite, old COSMA outputs
Makefile                  `make test`, `make plots`, plot subtargets
```

---

## Dependencies

[Pylians3](https://github.com/franciscovillaescusa/Pylians3) (CIC + 3D P(k)), [classy](https://github.com/lesgourg/class_public) (CLASS Boltzmann wrapper), numpy, scipy, h5py, astropy, matplotlib, pytest. Install Pylians3 and classy from their repos; the rest via pip.

---

## Critical environment variable

`vp_utils.py` resolves cosmological parameters ($h$, $\Omega_m$, $L_\mathrm{box}$, ...) from the Arepo `parameters-usedvalues` file on first use, controlled by:

```bash
export VP_PARAMS_FILE=/path/to/output/<model>/parameters-usedvalues
```

**Must match the model being processed.** A wrong file silently mis-scales $k$ by $h_\mathrm{wrong}/h_\mathrm{correct}$ and $P(k)$ by the cube of that ratio.

Resolution is lazy: importing `vp_utils` no longer needs the variable, and an unset variable raises a clear `RuntimeError` only on first cosmology-dependent use. Tests and callers that want to bypass it inject directly:

```python
import vp_utils
vp_utils.set_parameters_sim("/path/to/parameters-usedvalues")  # or a dict
```

`set_parameters_sim` invalidates the $\chi(z)$ and $\tau(z)$ interpolators so they rebuild against the new cosmology on next access.

---

## Pipeline

Each model has two seeds (`seed_2080`, `seed_4257`); per-seed outputs live under `output/<model>/seed_<id>/snap_NNN/`. Averaged products land in `output/<model>/{Pk_matter,Pk_curl,C_ells,SNRs}/`.

```bash
export VP_PARAMS_FILE=output/lcdm/parameters-usedvalues

# 1. Extract particles (PartType1 = dark matter)
python3 python/read_snap.py --base-path /path/to/arepo --snap-num 0 \
    --out-dir output/lcdm/seed_2080/snap_000

# 2. CIC density and momentum grid (writes delta.npy, momentum_{x,y,z}.npy)
python3 python/fields.py --in-dir output/lcdm/seed_2080/snap_000 \
    --out-dir output/lcdm/seed_2080/snap_000 --ngrid 1024 --threads 8

# 3. 3D matter and curl P(k), in physical units (Mpc^-1, Mpc^3)
python3 python/powerspec.py --in-dir output/lcdm/seed_2080/snap_000 \
    --out-dir output/lcdm/seed_2080/snap_000 --threads 8

# 3b. Average over seeds (use --h-orig/--h-correct if powerspec.py was run with wrong h)
python3 python/averaging_powerspec.py --base-dir output \
    --models lcdm frhs ndgp --seed1 seed_2080 --seed2 seed_4257

# 4. Angular power spectra via Limber integral, for a given source redshift.
#    Outputs dict with keys Phi, B, kSZ, B_X_kSZ.
python3 python/angular_powerspec_z.py --in-dir output/lcdm \
    --out-dir output/lcdm --z_source 1.5

# 5. SNR for a (survey, CMB experiment) pair
python3 python/SNR.py --in-dir output --out-dir output \
    --z_source 1.5 --survey LSST --cmb-exp SO
```

Loops over $z_\mathrm{source} \in [0.5, 3.0]$ and survey/experiment combinations are in `scripts/run_cells_z.sh` and `scripts/run_snrs.sh`. Argument details are exposed via `--help` on each script.

---

## Plotting

`make plots` runs everything. Individual targets and the equivalent CLI calls:

```bash
make plot-powerspec               # P_m(k) and P_curl(k)
make plot-cells                   # all four C_ell quantities
make plot-cells-cross             # Phi + B_X_kSZ only
make plot-snr                     # cumulative-vs-z + per-ell + cumulative-ell + colorbar
make plot-snr-cumulative-ell      # cumulative SNR vs ell, at z = $(z_redshift)
make plot-snr-colorbar-so         # colorbar plot, SO experiment
make plot-snr-colorbar-planck     # colorbar plot, Planck experiment
```

Override variables on the command line:

```bash
make CELL_QUANTITIES="Phi B" plot-cells
make z_redshift=1.0 plot-snr-cumulative-ell
```

Direct CLI: `python3 python/plot_cells.py --quantities Phi B_X_kSZ --z-sources 0.5 1.0 1.5 2.0`, or `python3 python/plot_snr.py --only cumulative-ell --z-ref 2.0`. All plot scripts accept `--show` and `--models {lcdm frhs ndgp}`.

---

## Testing

```bash
make test                                                          # full suite
VP_PARAMS_FILE=output/lcdm/parameters-usedvalues pytest python/tests/ -v
```

60 tests across five modules: `test_vp_utils.py` (background functions, C_ell kernels, $h$ propagation, $\tau(z)$ interpolator, lazy `parameters_sim` resolution, float-level `simpson` + `quad` baselines for all four C_ells), `test_averaging_powerspec.py` (seed averaging, h-correction; integration tests skip when `output/lcdm/Pk_matter/026.npy` is absent), `test_angular_powerspec_z.py`, `test_fields.py`, `test_snr.py`. `conftest.py` injects a fiducial cosmology via `set_parameters_sim`, so unit tests run without COSMA data and without `VP_PARAMS_FILE`.

---

## Units

All pipeline outputs are in physical (no-$h$) units: $k$ in $\mathrm{Mpc}^{-1}$, $P_m$ in $\mathrm{Mpc}^3$, $P_\mathrm{curl}$ in $\mathrm{Mpc}^3\,(\mathrm{km/s})^2$, $\chi$ in $\mathrm{Mpc}$, $H(z)$ in $\mathrm{km/s/Mpc}$. Pylians3 returns $h$-units; `powerspec.py` does the conversion using $h$ from `VP_PARAMS_FILE`.

---

## COSMA (SLURM)

`scripts/run_lcdm.sh`, `run_fr.sh`, `run_ndgp.sh` cover Steps 1–3 per model; `run_cells_z.sh` runs Step 4 over $z \in [0.5, 3.0]$; `run_snrs.sh` runs Step 5 over all survey × experiment × redshift combinations. Update `BASE`, `OUTROOT`, and `VP_PARAMS_FILE` at the top of each before submitting.
