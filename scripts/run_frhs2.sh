#!/bin/bash -l

#SBATCH -J frhs
#SBATCH --output=logs/frhs_%A_%a.out
#SBATCH --error=logs/frhs_%A_%a.err
#SBATCH -p cosma8-pauper
#SBATCH -A dp203
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END
#SBATCH -t 6:00:00
#SBATCH --mail-user=vicente.pedreros@ug.uchile.cl
#SBATCH --array=90-94

DATA_ROOT="/cosma8/data/dp203/bl267/Data/MG_Arepo_runs/FORGE-BRIDGE/FORGE/Particle_Snapshots"
OUT_BASE="/cosma8/data/dp203/dc-pedr3/gravitomagnetic/output/frhs"

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

export VP_PARAMS_FILE="${BASE}/parameters-usedvalues"

echo "====================================================="
echo "Processing Node: $NODE | Seed: $SEED"
echo "SLURM Task ID: $SLURM_ARRAY_TASK_ID"
echo "Target Base: $BASE"
echo "====================================================="

LAST_SNAPDIR=$(ls -1d "${BASE}"/snapdir_* 2>/dev/null | sort -V | tail -n 1)

if [ -z "$LAST_SNAPDIR" ]; then
        echo "  -> Error: No snapdir_* folders found in $BASE. Exiting task."
        exit 0
    fi

LAST_SNAP_FMT=$(basename "$LAST_SNAPDIR" | sed 's/snapdir_//')

MAX_SNAP=$((10#$LAST_SNAP_FMT))
echo "  -> Dynamic Discovery: Found snapshots up to $MAX_SNAP"

for snap in $(seq 0 $MAX_SNAP); do
    # Format the snapshot number (e.g., 000, 024) to use throughout the loop
    snap_fmt=$(printf '%03d' "$snap")
    
    # ---------------------------------------------------------
    # SNAPSHOT EXISTENCE CHECK
    # Check if the first chunk of the Arepo snapshot exists
    # Format: snapdir_NNN/snap_NNN.0.hdf5
    # ---------------------------------------------------------
    if [ ! -f "${BASE}/snapdir_${snap_fmt}/snap_${snap_fmt}.0.hdf5" ]; then
        echo "  -> Source snapshot $snap_fmt not found (missing snapdir_${snap_fmt}). Skipping."
        continue
    fi

    # Create the output directory for this snapshot
    out_dir="${OUTROOT}/snap_${snap_fmt}"
    mkdir -p "$out_dir"

    echo "  -> Running snapshot $snap_fmt..."

    python3 gravitomagnetic/python/read_snap.py --base-path "$BASE" --out-dir "$out_dir" --snap-num "$snap" 
    python3 gravitomagnetic/python/fields.py --in-dir "$out_dir" --out-dir "$out_dir" --threads $SLURM_CPUS_PER_TASK
    python3 gravitomagnetic/python/powerspec.py --in-dir "$out_dir" --out-dir "$out_dir" --threads $SLURM_CPUS_PER_TASK

    rm -f "${out_dir}/Coordinates.npy" "${out_dir}/Velocities.npy" "${out_dir}/delta.npy" "${out_dir}"/momentum*.npy
done