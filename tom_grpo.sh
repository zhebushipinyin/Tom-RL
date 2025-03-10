set -x

ray stop

export VLLM_ATTENTION_BACKEND=XFORMERS

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

## MODEL_PATH=Qwen/Qwen2.5-7B-Instruct-1M
# TODO
# MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct
# MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct
# MODEL_PATH=Qwen/Qwen2.5-3B-Instruct
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
# MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

lr=3e-7
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/tom/explore_tom/explore_tom_train_600.parquet \
    data.val_files=$HOME/data/tom/explore_tom/hi_tom_test.parquet \
    data.train_batch_size=8 \
    data.val_batch_size=16 \
    data.max_prompt_length=768 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=$MODEL_PATH\
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
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
    trainer.project_name='GRPO_tom_new' \
    trainer.experiment_name="$(basename $MODEL_PATH)-$lr" \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=2 $@ 2>&1 | tee tom_grpo.log

# trainer.default_local_dir=xxx \