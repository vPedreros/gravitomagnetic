#!/bin/bash -l

#SBATCH -J f_004
#SBATCH --ntasks 16
#SBATCH -p cosma8
#SBATCH -A dp203
#SBATCH -t 8:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=vicente.pedreros@ug.uchile.cl

BASE="/cosma8/data/dp203/bl267/Data/MG_Arepo_runs/FORGE-BRIDGE/FORGE/Particle_Snapshots/L500_N1024_Seed_2080_Node_004_Omega_m_0.31592_S8_0.61685_h_0.68845_fR0_8.07056e-07_sigma8_0.60111/"
OUTROOT="/cosma8/data/dp203/dc-pedr3/gravitomagnetic/output/frhs/node_004/seed_2080"

for snap in $(seq 25 26); do
    out_dir="${OUTROOT}/snap_$(printf '%03d' "$snap")"

    vp_param="${BASE}/parameters-usedvalues"
    export VP_PARAMS_FILE="$vp_param"

    python3 gravitomagnetic/python/read_snap.py --base-path "$BASE" --out-dir "$out_dir" --snap-num $snap 
    python3 gravitomagnetic/python/fields.py --in-dir "$out_dir" --out-dir "$out_dir" --threads 16
    python3 gravitomagnetic/python/powerspec.py --in-dir "$out_dir" --out-dir "$out_dir" --threads 16

    rm -f "${out_dir}/Coordinates.npy"
    rm -f "${out_dir}/Velocities.npy"
    rm -f "${out_dir}/delta.npy"
    rm -f "${out_dir}"/momentum*.npy

done

BASE="/cosma8/data/dp203/bl267/Data/MG_Arepo_runs/FORGE-BRIDGE/FORGE/Particle_Snapshots/L500_N1024_Seed_4257_Node_004_Omega_m_0.31592_S8_0.61685_h_0.68845_fR0_8.07056e-07_sigma8_0.60111/"
OUTROOT="/cosma8/data/dp203/dc-pedr3/gravitomagnetic/output/frhs/node_004/seed_4257"

for snap in $(seq 25 26); do
    out_dir="${OUTROOT}/snap_$(printf '%03d' "$snap")"

    vp_param="${BASE}/parameters-usedvalues"
    export VP_PARAMS_FILE="$vp_param"

    python3 gravitomagnetic/python/read_snap.py --base-path "$BASE" --out-dir "$out_dir" --snap-num $snap 
    python3 gravitomagnetic/python/fields.py --in-dir "$out_dir" --out-dir "$out_dir" --threads 16
    python3 gravitomagnetic/python/powerspec.py --in-dir "$out_dir" --out-dir "$out_dir" --threads 16

    rm -f "${out_dir}/Coordinates.npy"
    rm -f "${out_dir}/Velocities.npy"
    rm -f "${out_dir}/delta.npy"
    rm -f "${out_dir}"/momentum*.npy

done