#!/bin/bash

#SBATCH -J tom
#SBATCH --mem=512g
#SBATCH -p HGX,DGX                    
#SBATCH --nodes=1
#SBATCH --gres=gpu:4            
#SBATCH --cpus-per-task=64
#SBATCH -t 40:00:00
#SBATCH -e logs/slurm-%j.err
#SBATCH -o logs/slurm-%j.out
#SBATCH --qos=lv1           

source ~/anaconda3/etc/profile.d/conda.sh
conda activate logic

set -x

ray stop

export VLLM_ATTENTION_BACKEND=XFORMERS

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)


train_batch_size=8
enable_gradient_checkpointing=True

use_hints=(True False)
model_names=("Qwen/Qwen2.5-7B-Instruct-1M" "Qwen/Qwen2.5-7B-Instruct" "Qwen/Qwen2.5-3B-Instruct" "Qwen/Qwen2.5-1.5B-Instruct" "Qwen/Qwen2.5-0.5B-Instruct")
lrs=(3e-7 4e-7 5e-7)
num_epochs=2

# model_names=("Qwen/Qwen2.5-7B-Instruct")
# train_sources=("hi_tom")
# use_hints=(True)
# lrs=(3e-7)

for model_name in ${model_names[@]}
do
    # 提取 "Qwen/Qwen2.5-7B-Instruct-1M" 中的 7B
    model_size=$(echo $model_name | grep -o "[0-9.]\+B" | head -1)
    for lr in ${lrs[@]}
    do
        for use_hint in ${use_hints[@]}
        do
            if [ $use_hint == "True" ]; then
                data_train_files=$HOME/data/tom/hi_tom_train_2000_hint.parquet
                test_files=$HOME/data/tom/hi_tom_explore_tom_test_hint.parquet
            else
                data_train_files=$HOME/data/tom/hi_tom_train_2000.parquet
                test_files=$HOME/data/tom/hi_tom_explore_tom_test.parquet
            fi

            python3 -m verl.trainer.main_ppo \
                algorithm.adv_estimator=grpo \
                data.train_files=$data_train_files \
                data.val_files=$test_files \
                data.train_batch_size=$train_batch_size \
                data.val_batch_size=16 \
                data.max_prompt_length=1024 \
                data.max_response_length=2048 \
                actor_rollout_ref.model.path=$model_name \
                actor_rollout_ref.actor.optim.lr=$lr \
                actor_rollout_ref.model.use_remove_padding=True \
                actor_rollout_ref.actor.ppo_mini_batch_size=256 \
                actor_rollout_ref.actor.ppo_micro_batch_size=64 \
                actor_rollout_ref.actor.use_kl_loss=True \
                actor_rollout_ref.actor.kl_loss_coef=0.001 \
                actor_rollout_ref.actor.kl_loss_type=low_var_kl \
                actor_rollout_ref.model.enable_gradient_checkpointing=$enable_gradient_checkpointing \
                actor_rollout_ref.actor.fsdp_config.param_offload=True \
                actor_rollout_ref.actor.fsdp_config.grad_offload=True \
                actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
                actor_rollout_ref.rollout.log_prob_micro_batch_size=160 \
                actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
                actor_rollout_ref.rollout.name=vllm \
                actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
                actor_rollout_ref.rollout.n=16 \
                actor_rollout_ref.ref.log_prob_micro_batch_size=160 \
                actor_rollout_ref.ref.fsdp_config.param_offload=True \
                algorithm.kl_ctrl.kl_coef=0.001 \
                trainer.critic_warmup=0 \
                trainer.logger=['console','wandb'] \
                trainer.project_name='GRPO_tom_lambda' \
                trainer.experiment_name="$(basename $model_name)-$lr-$use_hint" \
                trainer.n_gpus_per_node=$NUM_GPUS \
                trainer.nnodes=1 \
                trainer.default_hdfs_dir=null \
                trainer.save_freq=50 \
                trainer.test_freq=10 \
                trainer.total_epochs=$num_epochs $@ 2>&1 | tee logs/tom_grpo_$(basename $model_name)_${lr}_${use_hint}.log
        done
    done
done

# trainer.default_local_dir=xxx \