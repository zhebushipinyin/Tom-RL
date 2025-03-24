#!/bin/bash

set -x

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

TODAY=$(date +%Y%m%d)
# CKPTS_DIR="./checkpoints/GRPO_merge_tom/Qwen2.5-7B-Instruct-1M-5e-7-True-16/actor"
MODEL_PATHS=("Qwen/Qwen2.5-0.5B-Instruct" 
             "Qwen/Qwen2.5-1.5B-Instruct" 
             "Qwen/Qwen2.5-3B-Instruct" 
             "Qwen/Qwen2.5-7B-Instruct" 
             "Qwen/Qwen2.5-7B-Instruct-1M")

# hi_tom (0-1-2-3-4): 1000
# explore_tom: 600
# tom_i
# expert_tom
# explore_tom_test: 2662

DATA_PATHS=(
    # "./data/cleaned_tom/raw/Hi_ToM_cleaned.csv"
    "./data/cleaned_tom/ToM_test_HiExTi_hint.parquet"
    # "./eval_tom/test_dataset/expert_tom_data.csv"
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
    for DATA_PATH in ${DATA_PATHS[@]}; do
        echo "Evaluating ${MODEL_PATH} on ${DATA_PATH}"
        BASENAME=$(basename ${DATA_PATH})
        DATA_NAME="${BASENAME%.*}"
        python3 eval_tom/qwen_series_eval.py \
                --model_path ${MODEL_PATH} \
                --data_path ${DATA_PATH} \
                --output_dir ${OUTPUT_DIR}
    done
done

