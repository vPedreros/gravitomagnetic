#!/bin/bash -l
source /Users/vpedreros/nerding/gravitomagnetic/.venv/bin/activate

for NODE_N in $(seq 1 49); do 
    node_fmt=$(printf '%03d' "$NODE_N")
    NODE="node_$node_fmt"

    # export VP_PARAMS_FILE="output/lcdm/$NODE/parameters-usedvalues"

    # python python/averaging_powerspec.py --models lcdm frhs ndgp --node "$NODE"
    for z in $(seq 0.5 0.1 3); do
        # export VP_PARAMS_FILE="output/lcdm/$NODE/seed_2080/parameters-usedvalues"
        # python3 python/angular_powerspec_z.py --in-dir output/lcdm/ --out-dir output/lcdm/ --z_source "$z" --node "$NODE"

        # export VP_PARAMS_FILE="output/frhs/$NODE/seed_2080/parameters-usedvalues"
        # python3 python/angular_powerspec_z.py --in-dir output/frhs/ --out-dir output/frhs/ --z_source "$z" --node "$NODE"

        # export VP_PARAMS_FILE="output/ndgp/$NODE/seed_2080/parameters-usedvalues"
        # python3 python/angular_powerspec_z.py --in-dir output/ndgp/ --out-dir output/ndgp/ --z_source "$z" --node "$NODE"

        python python/SNR.py --in-dir output/ --out-dir output/ --z_source "$z" --survey "LSST" --cmb-exp "SO" --node "$NODE"
        python python/SNR.py --in-dir output/ --out-dir output/ --z_source "$z" --survey "LSST" --cmb-exp "Planck" --node "$NODE"
        python python/SNR.py --in-dir output/ --out-dir output/ --z_source "$z" --survey "Euclid" --cmb-exp "SO" --node "$NODE"
        python python/SNR.py --in-dir output/ --out-dir output/ --z_source "$z" --survey "Euclid" --cmb-exp "Planck" --node "$NODE"
    done
done