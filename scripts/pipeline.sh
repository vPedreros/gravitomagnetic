#!/bin/bash

# --- 1. CONFIGURATION ---
BASE_DIR="output"
MODELS=("lcdm" "frhs" "ndgp")
SEED1="seed_2080"
SEED2="seed_4257"
Z_SOURCE="1.5"
SURVEY="LSST"
CMB_EXP="SO"

echo "====================================================="
echo "Starting Final Pipeline Stages (3b, 4, 5) Locally"
echo "====================================================="

# --- 2. GLOBAL ENVIRONMENT SETUP (For h unit in Step 3b) ---
# Grab the parameter file from the first model (lcdm) to establish the base h value
BASE_PARAM_FILE=$(find "${BASE_DIR}/${MODELS[0]}" -name "parameters-usedvalues" | head -n 1)

if [ -z "$BASE_PARAM_FILE" ]; then
    echo "ERROR: Could not find base parameters-usedvalues in ${BASE_DIR}/${MODELS[0]}. Exiting."
    exit 1
fi

export VP_PARAMS_FILE="$BASE_PARAM_FILE"
echo "[Setup] Exported global VP_PARAMS_FILE: $VP_PARAMS_FILE"
echo "-----------------------------------------------------"

# --- STEP 3b: Average over seeds ---
echo "[Step 3b] Averaging power spectra across seeds..."
python3 python/averaging_powerspec.py \
    --base-dir "$BASE_DIR" \
    --models "${MODELS[@]}" \
    --seed1 "$SEED1" \
    --seed2 "$SEED2"

echo "-----------------------------------------------------"

# --- STEP 4: Angular Power Spectra ---
echo "[Step 4] Computing Angular Power Spectra for all models..."
for MODEL in "${MODELS[@]}"; do
    echo "  -> Processing model: $MODEL"
    
    # Dynamically update the parameter file for the specific model being processed
    MODEL_PARAM_FILE=$(find "${BASE_DIR}/${MODEL}" -name "parameters-usedvalues" | head -n 1)
    
    if [ -n "$MODEL_PARAM_FILE" ]; then
        export VP_PARAMS_FILE="$MODEL_PARAM_FILE"
        echo "     (Updated VP_PARAMS_FILE: $VP_PARAMS_FILE)"
    else
        echo "     WARNING: No parameters-usedvalues found for $MODEL. Using previous."
    fi

    python3 python/angular_powerspec_z.py \
        --in-dir "${BASE_DIR}/${MODEL}" \
        --out-dir "${BASE_DIR}/${MODEL}" \
        --z_source "$Z_SOURCE"
done

echo "-----------------------------------------------------"

# --- STEP 5: SNR Calculation ---
echo "[Step 5] Calculating SNR for ($SURVEY, $CMB_EXP)..."
# We leave the VP_PARAMS_FILE as whatever it was last set to, or you can reset it to base if SNR.py needs it.
python3 python/SNR.py \
    --in-dir "$BASE_DIR" \
    --out-dir "$BASE_DIR" \
    --z_source "$Z_SOURCE" \
    --survey "$SURVEY" \
    --cmb-exp "$CMB_EXP"

echo "====================================================="
echo "Pipeline Complete!"