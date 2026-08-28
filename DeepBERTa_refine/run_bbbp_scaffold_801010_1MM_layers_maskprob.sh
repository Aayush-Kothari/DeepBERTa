#!/bin/bash

set -u

DATASET_NAME="bbbp"

MOLECULENET_ROOT="/home/makl1/ml_cheminformatics_group/sourish/data/moleculenet"
DATASET_TRAIN_PATH="${MOLECULENET_ROOT}/${DATASET_NAME}_scaffold_refine80.csv"
DATASET_VAL_PATH="${MOLECULENET_ROOT}/${DATASET_NAME}_scaffold_val10.csv"
DATASET_BENCHMARK_PATH="${MOLECULENET_ROOT}/${DATASET_NAME}_scaffold_bench10.csv"

BASE_MODEL_ROOT="/home/makl1/ml_cheminformatics_group/sourish/data/pubchem/CID-DEEPSMILES-TRAINING"
LOG_DIR="/home/makl1/ml_cheminformatics_group/sourish/DeepBERTa/DeepBERTa_refine/logs"
OUTPUT_ROOT="${MOLECULENET_ROOT}"

MODEL_DATE="${MODEL_DATE:-20260501}"

LAYERS=("4" "6" "9" "12")
MASK_PROBS=("0.10" "0.15" "0.20" "0.30" "0.50" "0.70")
SEEDS="1,2,3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/run_bbbp_pubchem_multiseed_80_10_10.py"

mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_ROOT"

for REQUIRED_FILE in \
    "$DATASET_TRAIN_PATH" \
    "$DATASET_VAL_PATH" \
    "$DATASET_BENCHMARK_PATH"
do
    if [[ ! -f "$REQUIRED_FILE" ]]; then
        echo "ERROR: Required dataset file does not exist: $REQUIRED_FILE" >&2
        exit 1
    fi
done

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "ERROR: Python script does not exist: $PYTHON_SCRIPT" >&2
    exit 1
fi

for LAYER in "${LAYERS[@]}"
do
    for MASK_PROB in "${MASK_PROBS[@]}"
    do
        MASK_TAG=$(echo "$MASK_PROB" | sed 's/0\.//')

        MODEL_DIR="${BASE_MODEL_ROOT}/GPU1MM_OUTPUT_Layers${LAYER}_MaskProb${MASK_PROB}_${MODEL_DATE}/default_run/final"

        RUN_DATE_LOG=$(date +%Y-%m-%d)
        RUN_DATE_TAG=$(date +%Y%m%d)

        LOGFILE="${LOG_DIR}/GPU1M_Layers${LAYER}_Mask${MASK_TAG}_REFINE_OUTPUT-${DATASET_NAME}-${RUN_DATE_LOG}.log"

        OUTPUTFOLDER="GPU1MM_Layers${LAYER}_Mask${MASK_TAG}_REFINE_OUTPUT_${DATASET_NAME}_${RUN_DATE_TAG}"
        FULL_OUTPUT_DIR="${OUTPUT_ROOT}/${OUTPUTFOLDER}"

        echo "============================================================"
        echo "Starting BBBP scaffold refine/validate/benchmark"
        echo "Layer:               ${LAYER}"
        echo "Mask probability:    ${MASK_PROB}"
        echo "Model directory:     ${MODEL_DIR}"
        echo "Train CSV:           ${DATASET_TRAIN_PATH}"
        echo "Validation CSV:      ${DATASET_VAL_PATH}"
        echo "Benchmark CSV:       ${DATASET_BENCHMARK_PATH}"
        echo "Output directory:    ${FULL_OUTPUT_DIR}"
        echo "Log file:            ${LOGFILE}"
        echo "Seeds:               ${SEEDS}"
        echo "============================================================"

        if [[ ! -d "$MODEL_DIR" ]]; then
            echo "WARNING: Model directory does not exist; skipping:"
            echo "  $MODEL_DIR"
            continue
        fi

        python "$PYTHON_SCRIPT" \
          --model_name_or_path "$MODEL_DIR" \
          --train_csv "$DATASET_TRAIN_PATH" \
          --val_csv "$DATASET_VAL_PATH" \
          --bench_csv "$DATASET_BENCHMARK_PATH" \
          --output_root_dir "$FULL_OUTPUT_DIR" \
          --seeds "$SEEDS" \
          > "$LOGFILE" 2>&1

        EXIT_CODE=$?

        if [[ $EXIT_CODE -ne 0 ]]; then
            echo "ERROR: BBBP run failed for layer=${LAYER}, mask=${MASK_PROB}."
            echo "Inspect log: $LOGFILE"
        else
            echo "Completed layer=${LAYER}, mask=${MASK_PROB}."
            echo "Results: $FULL_OUTPUT_DIR"
        fi
    done
done

echo "All available BBBP layer/masking configurations have been processed."
