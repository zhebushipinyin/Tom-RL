#!/bin/bash

set -x

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

TODAY=$(date +%Y%m%d)
# CKPTS_DIR="./checkpoints/GRPO_merge_tom/Qwen2.5-7B-Instruct-1M-5e-7-True-16/actor"
ALL_CKPTS_DIRS=(
    "./checkpoints/GRPO_merge_tom_20250319/Qwen2.5-0.5B-Instruct-5e-7-True-16/actor"
    "./checkpoints/GRPO_merge_tom_20250319/Qwen2.5-1.5B-Instruct-5e-7-True-16/actor"
    "./checkpoints/GRPO_merge_tom_20250319/Qwen2.5-3B-Instruct-5e-7-True-16/actor"
    "./checkpoints/GRPO_merge_tom_20250319/Qwen2.5-7B-Instruct-5e-7-True-16/actor"
    "./checkpoints/GRPO_merge_tom_20250319/Qwen2.5-7B-Instruct-1M-5e-7-True-16/actor"
)
DATA_PATHS=(
    "./data/cleaned_tom/raw/Hi_ToM_cleaned.parquet"
    "./data/cleaned_tom/ToM_test_HiExTi_hint.parquet"
    "./eval_tom/test_dataset/expert_tom_data.csv"
    "./eval_tom/test_dataset/explore_tom_test_2662.parquet"
)

OUTPUT_DIR="./eval_tom/ckpts_results"
if [ ! -d "${OUTPUT_DIR}" ]; then
    mkdir -p ${OUTPUT_DIR}
fi

LOG_DIR="./logs/${TODAY}"
if [ ! -d "${LOG_DIR}" ]; then
    mkdir -p ${LOG_DIR}
fi

for CKPTS_DIR in ${ALL_CKPTS_DIRS[@]}; do
    MODEL_ID=$(echo ${CKPTS_DIR} | awk -F'/' '{print $(NF-1)}' | sed -E 's|(Qwen2\.5-[0-9.]+B-Instruct)(-1M)?(-.*)?|\1\2|')
    echo "Processing model: ${MODEL_ID}"
    for CKPT_PATH in ${CKPTS_DIR}/*; do
        echo "Evaluating ${CKPT_PATH}"
        # global_step_100
        STEP="${CKPT_PATH##*_}"
        if [ ${STEP} -eq 700 ] || [ ${STEP} -eq 750 ] || [ ${STEP} -eq 800 ]; then
            for DATA_PATH in ${DATA_PATHS[@]}; do
                BASENAME=$(basename ${DATA_PATH})
                DATA_NAME="${BASENAME%.*}"
                # python3 eval_tom/qwen_series_eval.py --model_path ${CKPT_PATH} --data_path ${DATA_PATH} --output_dir ${OUTPUT_DIR}/ckpt_${STEP}.csv
                python3 eval_tom/reasoning_model_eval.py \
                    --model_path ${CKPT_PATH} \
                    --data_path ${DATA_PATH} \
                    --output_dir ${OUTPUT_DIR}/ckpt_${MODEL_ID}_${STEP}_${DATA_NAME}_${TODAY}.csv \
                    $@ 2>&1 | tee ${LOG_DIR}/eval_${MODEL_ID}_${STEP}_${DATA_NAME}.log
            done
        fi
    done
done

