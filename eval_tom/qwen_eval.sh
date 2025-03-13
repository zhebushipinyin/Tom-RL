#!/bin/bash

set -x

MODEL_PATHS=(
    # "Qwen/Qwen2.5-0.5B-Instruct"
    # "Qwen/Qwen2.5-1.5B-Instruct"
    # "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
)

DATA_PATHS=(
    "data/cleaned_tom/raw/ToM_train_600.parquet"
    "data/cleaned_tom/raw/hi_tom_3000.csv"
)

for MODEL_PATH in ${MODEL_PATHS[@]}; do
    for DATA_PATH in ${DATA_PATHS[@]}; do
        python3 eval_tom/qwen_series_eval.py --model_path ${MODEL_PATH} --data_path ${DATA_PATH}
    done
done

