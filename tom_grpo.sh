set -x

ray stop

export VLLM_ATTENTION_BACKEND=XFORMERS

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)


train_batch_size=8
enable_gradient_checkpointing=True

use_hints=(True False)
train_sources=("hi_tom" "explore_tom")
# model_names=("Qwen/Qwen2.5-7B-Instruct-1M" "Qwen/Qwen2.5-7B-Instruct" "Qwen/Qwen2.5-3B-Instruct" "Qwen/Qwen2.5-1.5B-Instruct" "Qwen/Qwen2.5-0.5B-Instruct")
model_names=("Qwen/Qwen2.5-0.5B-Instruct" "Qwen/Qwen2.5-1.5B-Instruct" "Qwen/Qwen2.5-3B-Instruct" "Qwen/Qwen2.5-7B-Instruct" "Qwen/Qwen2.5-7B-Instruct-1M")
lrs=(3e-7 4e-7 5e-7)

# model_names=("Qwen/Qwen2.5-7B-Instruct")
# train_sources=("hi_tom")
# use_hints=(True)
# lrs=(3e-7)
# num_epochs=4

for model_name in ${model_names[@]}
do
    for lr in ${lrs[@]}
    do
        for use_hint in ${use_hints[@]}
        do
            for train_source in ${train_sources[@]}
            do

                if [ $train_source == "explore_tom" ]; then
                    num_epochs=2
                    if [ $use_hint == "True" ]; then
                        data_train_files=$HOME/data/tom/explore_tom_600_both_hint.parquet
                        test_files=$HOME/data/tom/hi_tom_hint.parquet
                    else
                        data_train_files=$HOME/data/tom/explore_tom_600_both.parquet
                        test_files=$HOME/data/tom/hi_tom.parquet
                    fi
                else
                    num_epochs=4
                    if [ $use_hint == "True" ]; then
                        data_train_files=$HOME/data/tom/hi_tom_hint.parquet
                        test_files=$HOME/data/tom/explore_tom_600_both_hint.parquet
                    else
                        data_train_files=$HOME/data/tom/hi_tom.parquet
                        test_files=$HOME/data/tom/explore_tom_600_both.parquet
                    fi
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
                    trainer.experiment_name="$(basename $model_name)-$lr-$train_source-$use_hint" \
                    trainer.n_gpus_per_node=$NUM_GPUS \
                    trainer.nnodes=1 \
                    trainer.default_hdfs_dir=null \
                    trainer.save_freq=50 \
                    trainer.test_freq=10 \
                    trainer.total_epochs=$num_epochs $@ 2>&1 | tee tom_grpo_$(basename $model_name)_${lr}_${train_source}_${use_hint}.log
                sleep 600
            done
        done
    done
done

# trainer.default_local_dir=xxx \