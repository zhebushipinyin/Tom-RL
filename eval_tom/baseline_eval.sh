#!/bin/bash

set -x

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

TODAY=$(date +%Y%m%d)
# CKPTS_DIR="./checkpoints/GRPO_merge_tom/Qwen2.5-7B-Instruct-1M-5e-7-True-16/actor"
MODEL_PATHS=("Qwen/Qwen2.5-7B-Instruct-1M")

DATA_PATHS=(
    # "./data/cleaned_tom/ToM_test_HiExTi_hint.parquet"
    "./eval_tom/test_dataset/expert_tom_data.csv"
    # "./eval_tom/test_dataset/explore_tom_test_2662.parquet"
)

OUTPUT_DIR="./eval_tom/baseline_results"
if [ ! -d "${OUTPUT_DIR}" ]; then
    mkdir -p ${OUTPUT_DIR}
fi

LOG_DIR="./logs/${TODAY}"
if [ ! -d "${LOG_DIR}" ]; then
    mkdir -p ${LOG_DIR}
fi

for MODEL_PATH in ${MODEL_PATHS[@]}; do
    echo "Evaluating ${MODEL_PATH}"
        for DATA_PATH in ${DATA_PATHS[@]}; do
            BASENAME=$(basename ${DATA_PATH})
            DATA_NAME="${BASENAME%.*}"
            python3 eval_tom/qwen_series_eval.py \
                --model_path ${MODEL_PATH} \
                --data_path ${DATA_PATH} \
                --output_dir ${OUTPUT_DIR}
    done
done

