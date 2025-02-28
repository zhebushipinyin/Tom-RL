#!/bin/bash

#SBATCH -J tom_rl
#SBATCH --mem=200g
#SBATCH -p DGX                    # 作业提交的分区为 DGX
#SBATCH --nodes=1
#SBATCH --gres=gpu:2            # 单个节点使用 1 块 GPU 卡
#SBATCH --cpus-per-task=64
#SBATCH -t 40:00:00
#SBATCH -e logs/slurm-%j.err
#SBATCH -o logs/slurm-%j.out
#SBATCH --qos=lv0b               # 作业使用的 QoS 为 lv0b

source ~/anaconda3/etc/profile.d/conda.sh
conda activate logic

set -x

ray stop

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

########################################################
echo "job is starting on `hostname`"
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
echo "Nodes: $nodes"

nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo "Head Node IP: $head_node_ip"

if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${NUM_GPUS}" --block &
sleep 10
worker_num=$((SLURM_JOB_NUM_NODES - 1))
echo "Worker number: $worker_num"
########################################################

export VLLM_ATTENTION_BACKEND=XFORMERS
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct-1M

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/tom/explore_tom/train.parquet \
    data.val_files=$HOME/data/tom/explore_tom/test.parquet \
    data.train_batch_size=8 \
    data.val_batch_size=8 \
    data.max_prompt_length=768 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=$MODEL_PATH\
    actor_rollout_ref.actor.optim.lr=3e-7 \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=160 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='GRPO_tom' \
    trainer.experiment_name='Qwen-7B-IM' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=5 $@ 2>&1 | tee tom_grpo.log

# trainer.default_local_dir=xxx \