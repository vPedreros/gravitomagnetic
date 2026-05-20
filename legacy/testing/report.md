# Python scripts audit — `python/`

## Test results

| Suite | Tests | Status |
|---|---|---|
| `test_read_snap.py` | 13 | ✅ all pass |
| `test_snr_functions.py` | 14 | ✅ all pass |
| `test_vp_utils.py` | 38 | ✅ all pass |
| **Total** | **65** | **65 passed, 0 failed** |

Run with:
```bash
.venv/bin/python -m pytest testing/ -v
```

---

## Static analysis — issues found

### 1. `vp_utils.py` — `raise(string)` instead of `raise ValueError(string)`

**Lines 247 and 302** both contain:
```python
raise('Invalid selection of integration method, please choose between ...')
```
In Python 3, `raise` requires an exception instance or class. `raise('...')` raises a `TypeError` (not the intended error) and the message is lost. Should be:
```python
raise ValueError('Invalid selection of integration method ...')
```

### 2. `vp_utils.py` — heavy module-level execution on import

**Lines 105–161**: When `vp_utils` is imported, it immediately:
- reads `~/nerding/gravitomagnetic/output/lcdm/parameters-usedvalues` from a hardcoded path
- builds a 10 000-point comoving distance table via numerical integration

This works fine in the current environment, but it means:
- Any script that imports `vp_utils` will fail if the output directory doesn't exist (e.g. on a new machine or COSMA)
- Import time is non-trivial (numerical integration on every import)

**Recommendation**: wrap inside a `get_params()` / `build_chi_table()` function called lazily, or accept the path as an argument.

### 3. `SNR.py` — argparse + CLASS + file I/O at module level

**Lines 19–95**: `parse_args()` is called at the top of the module, and CLASS runs + CSV files are read before `main()` is ever called. This means:
- `SNR.py` **cannot be imported** as a library (running `import SNR` will immediately demand `--in-dir`, `--z_source`, `--survey`, `--cmb-exp` from sys.argv and crash).
- All these side effects should live inside `main()` or behind an `if __name__ == '__main__'` guard.

The `noise_convergence`, `noise_temperature`, `Cov`, and `SNR` functions are pure and correct — see `test_snr_functions.py` for validation.

### 4. `angular_powerspec_z.py` — missing `mkdir` for `C_ells/` subdirectory

**Lines 147–148**:
```python
np.save(out / f"C_ells/ell_grid_z={args.z_source}.npy", ell_grid)
np.save(out / f"C_ells/C_ells_XY_z={args.z_source}.npy", C_ells_XY)
```
`out.mkdir(parents=True, exist_ok=True)` is called earlier (line 29) for `out`, but the `C_ells/` sub-directory is never explicitly created. `np.save` will raise `FileNotFoundError` if it doesn't exist yet.

**Fix**: add `(out / "C_ells").mkdir(parents=True, exist_ok=True)` before the saves.

### 5. `fields.py` and `powerspec.py` — `from pylab import *`

Both files use wildcard imports from `pylab` (lines 13 in each). `pylab` merges matplotlib and numpy into the global namespace, which can silently shadow builtins (e.g. `max`, `min`, `sum`). Neither script actually uses any matplotlib plotting functions.

**Recommendation**: replace with `import numpy as np` (already used implicitly via pylab).

### 6. `averaging_powerspec.py` — hardcoded `range(25)` for non-lcdm models

**Line 38**: the loop iterates `range(25)`, but earlier (line 11) the code correctly uses `nsnaps = 27 if m == "lcdm" else 25`. The hardcoded `25` in `angular_powerspec_z.py` (line 38) would silently skip the last 2 snapshots of the lcdm run if that script is ever pointed at the lcdm model directly.

---

## Scripts not covered by unit tests (require real data / heavy dependencies)

| Script | Reason |
|---|---|
| `fields.py` | requires `MAS_library` (Pylians3) + real `.npy` particle data |
| `powerspec.py` | requires `Pk_library` (Pylians3) + field arrays |
| `read_snap.py` (main) | requires real HDF5 snapshot files |
| `angular_powerspec.py` | integration smoke-tested indirectly via `test_vp_utils.py::TestCellXY` |
| `angular_powerspec_z.py` | same — uses same `C_ell_XY` core |
| `averaging_powerspec.py` | pure numpy, but needs real power-spectrum `.npy` files |
| `SNR.py` (main) | requires CLASS cosmological run + saved C_ell files |
