#!/bin/bash -l
source /Users/vpedreros/nerding/gravitomagnetic/.venv/bin/activate
export VP_PARAMS_FILE="output/lcdm/node_004/parameters-usedvalues"

for z in $(seq 0.5 0.1 3); do
    printf "\rProcessing redshift: %s..." "$z"
    python python/SNR.py --in-dir output/ --out-dir output/ --z_source "$z" --survey "LSST" --cmb-exp "SO"
    python python/SNR.py --in-dir output/ --out-dir output/ --z_source "$z" --survey "LSST" --cmb-exp "Planck"
    python python/SNR.py --in-dir output/ --out-dir output/ --z_source "$z" --survey "Euclid" --cmb-exp "SO"
    python python/SNR.py --in-dir output/ --out-dir output/ --z_source "$z" --survey "Euclid" --cmb-exp "Planck"
done

echo ""
echo "Done."