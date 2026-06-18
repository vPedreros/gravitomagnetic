#!/bin/bash -l

#SBATCH -J copy_pars_files
#SBATCH --output=logs/lcdm_%A_%a.out
#SBATCH --error=logs/lcdm_%A_%a.err
#SBATCH -p cosma8-pauper
#SBATCH -A dp203
#SBATCH -t 0:05:00
#SBATCH --array=0-99

DATA_ROOT="/cosma8/data/dp203/bl267/Data/MG_Arepo_runs/FORGE-BRIDGE/LCDM/Particle_Snapshots"
OUT_BASE="/cosma8/data/dp203/dc-pedr3/gravitomagnetic/output/lcdm"

# Create an array of all target directories
DIRS=("$DATA_ROOT"/L500_N1024_Seed_*_Node_*)

# Get the specific directory for this array task
BASE="${DIRS[$SLURM_ARRAY_TASK_ID]}"

# Safety check: if the array index exceeds the number of directories, exit
if [ -z "$BASE" ]; then
    exit 0
fi

DIR_NAME=$(basename "$BASE")
SEED=$(echo "$DIR_NAME" | sed -n 's/.*Seed_\([0-9]*\).*/\1/p')
NODE=$(echo "$DIR_NAME" | sed -n 's/.*Node_\([0-9]*\).*/\1/p')

OUTROOT="${OUT_BASE}/node_${NODE}/seed_${SEED}"

mkdir -p "$OUTROOT"

cp "${BASE}/parameters-usedvalues" "$OUTROOT/"

echo "====================================================="
echo "Processing Node: $NODE | Seed: $SEED"