#!/bin/bash -l
source /Users/vpedreros/nerding/gravitomagnetic/.venv/bin/activate
export VP_PARAMS_FILE="output/lcdm/node_004/parameters-usedvalues"

for z in $(seq 0.5 0.1 3); do
    export VP_PARAMS_FILE="output/lcdm/node_004/parameters-usedvalues"
    python3 python/angular_powerspec_z.py --in-dir output/lcdm/ --out-dir output/lcdm/ --z_source "$z"
    
    export VP_PARAMS_FILE="output/frhs/node_004/parameters-usedvalues"
    python3 python/angular_powerspec_z.py --in-dir output/frhs/ --out-dir output/frhs/ --z_source "$z"

    export VP_PARAMS_FILE="output/ndgp/node_004/parameters-usedvalues"
    python3 python/angular_powerspec_z.py --in-dir output/ndgp/ --out-dir output/ndgp/ --z_source "$z"
done