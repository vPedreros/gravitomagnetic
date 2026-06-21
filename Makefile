PYTHON ?= python
IN_DIR ?= output
NODE ?= node_001
MODEL ?= lcdm
IMG_DIR ?= imgs/$(NODE)/
VP_PARAMS_FILE ?= output/$(MODEL)/$(NODE)/parameters-usedvalues
CELL_QUANTITIES ?= Phi B kSZ B_X_kSZ

.PHONY: test plots plot-powerspec plot-cells plot-cells-cross plot-snr plot-snr-cumulative-ell plot-snr-colorbar plot-snr-colorbar-so plot-snr-colorbar-planck

test:
	$(PYTHON) -m pytest python/tests/ -v

plots: plot-powerspec plot-cells plot-snr plot-snr-colorbar

plot-powerspec:
	VP_PARAMS_FILE=$(VP_PARAMS_FILE) $(PYTHON) python/plot_powerspectra.py --in-dir $(IN_DIR) --out-dir $(IMG_DIR) --node $(NODE)

plot-cells:
	VP_PARAMS_FILE=$(VP_PARAMS_FILE) $(PYTHON) python/plot_cells.py --in-dir $(IN_DIR) --out-dir $(IMG_DIR) --quantities $(CELL_QUANTITIES) --node $(NODE)

plot-cells-cross:
	VP_PARAMS_FILE=$(VP_PARAMS_FILE) $(PYTHON) python/plot_cells.py --in-dir $(IN_DIR) --out-dir $(IMG_DIR) --quantities Phi B_X_kSZ --node $(NODE)

plot-snr:
	VP_PARAMS_FILE=$(VP_PARAMS_FILE) $(PYTHON) python/plot_snr.py --in-dir $(IN_DIR) --out-dir $(IMG_DIR) --node $(NODE)

z_redshift = 2.0
plot-snr-cumulative-ell:
	VP_PARAMS_FILE=$(VP_PARAMS_FILE) $(PYTHON) python/plot_snr.py --in-dir $(IN_DIR) --out-dir $(IMG_DIR) --only cumulative-ell --z-ref $(z_redshift) --node $(NODE)

plot-snr-colorbar: plot-snr-colorbar-so plot-snr-colorbar-planck

plot-snr-colorbar-so:
	VP_PARAMS_FILE=$(VP_PARAMS_FILE) $(PYTHON) python/plot_snr.py --in-dir $(IN_DIR) --out-dir $(IMG_DIR) --only colorbar --colorbar-experiments SO --node $(NODE)

plot-snr-colorbar-planck:
	VP_PARAMS_FILE=$(VP_PARAMS_FILE) $(PYTHON) python/plot_snr.py --in-dir $(IN_DIR) --out-dir $(IMG_DIR) --only colorbar --colorbar-experiments Planck --node $(NODE)
