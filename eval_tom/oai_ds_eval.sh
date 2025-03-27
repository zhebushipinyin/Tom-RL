#!/bin/bash

set -x

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

TODAY=$(date +%Y%m%d)
# CKPTS_DIR="./checkpoints/GRPO_merge_tom/Qwen2.5-7B-Instruct-1M-5e-7-True-16/actor"
MODEL_PATHS=(
        # 'gpt-4o-mini-2024-07-18'
        # 'gpt-4o-2024-08-06'
        # 'o1-mini-2024-09-12'
        # 'o3-mini-2025-01-31'
        'deepseek-chat'
)

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
        
        # Run with smaller batch size to avoid rate limits
        python3 eval_tom/oai_ds_series_eval.py \
                --model_path ${MODEL_PATH} \
                --data_path ${DATA_PATH} \
                --output_dir ${OUTPUT_DIR} \
                --batch_size 10 \
                --checkpoint_interval 5
        
        # Add a pause between datasets to avoid rate limits
        echo "Waiting 30 seconds before processing next dataset to avoid rate limits..."
        sleep 30
    done
    
    # Add a longer pause between models
    echo "Waiting 60 seconds before processing next model to avoid rate limits..."
    sleep 60
done

