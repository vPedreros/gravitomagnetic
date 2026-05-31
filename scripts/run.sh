#!/bin/bash -l
#SBATCH -p cosma7 
#SBATCH -A dp203
#SBATCH -t 2:00:00
#SBATCH --ntasks 8

BASE="/cosma8/data/dp203/bl267/Data/MG_Arepo_runs/FORGE-BRIDGE/FORGE/Particle_Snapshots/L500_N1024_Seed_4257_Node_000_Omega_m_0.31315_S8_0.83954_h_0.6737_fR0_0.0_sigma8_0.82172/"
OUTROOT="/cosma7/data/dp203/dc-pedr3/gravitomagnetic/output/lcdm/seed_4257"

for snap in $(seq 8 26); do
    out_dir="${OUTROOT}/snap_$(printf '%03d' "$snap")"
  
    python3 gravitomagnetic/python/read_snap.py --base-path "$BASE" --out-dir "$out_dir" --snap-num $snap 
    python3 gravitomagnetic/python/fields.py --in-dir "$out_dir" --out-dir "$out_dir" --verbose True --threads 8
    python3 gravitomagnetic/python/powerspec.py --in-dir "$out_dir" --out-dir "$out_dir" --verbose True --threads 8

done

python3 gravitomagnetic/python/angular_powerspec_z.py --in-dir 'gravitomagnetic/output/lcdm/' --out-dir 'gravitomagnetic/output/lcdm'