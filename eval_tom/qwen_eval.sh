#!/bin/bash

set -x

CKPTS_DIR="./checkpoints/GRPO_merge_tom/Qwen2.5-7B-Instruct-1M-4e-7-True-16/actor"

DATA_PATHS=(
    # "data/cleaned_tom/raw/ToM_train_600.parquet"
    # "data/cleaned_tom/raw/hi_tom_3000.csv"
    # "data/cleaned_tom/raw/Hi_ToM_cleaned.csv"
    "./data/cleaned_tom/ToM_test_HiExTi_hint.parquet"
)

OUTPUT_DIR="./eval_tom/ckpts_results"

for CKPT_PATH in ${CKPTS_DIR}/*; do
    if [ -d "${CKPT_PATH}" ]; then
        echo "Evaluating ${CKPT_PATH}"
        # global_step_100
        STEP="${CKPT_PATH##*_}"
        for DATA_PATH in ${DATA_PATHS[@]}; do
            # python3 eval_tom/qwen_series_eval.py --model_path ${CKPT_PATH} --data_path ${DATA_PATH} --output_dir ${OUTPUT_DIR}/ckpt_${STEP}.csv
            python3 eval_tom/reasoning_model_eval.py --model_path ${CKPT_PATH} --data_path ${DATA_PATH} --output_dir ${OUTPUT_DIR}/ckpt_${STEP}.csv
        done
    fi
done

