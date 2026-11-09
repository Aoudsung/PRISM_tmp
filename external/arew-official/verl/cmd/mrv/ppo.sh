#!/usr/bin/env bash
set -euo pipefail

DATA_NAME="MovieRec"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/${DATA_NAME}}"
BASE_MODEL="${BASE_MODEL:-${PROJECT_ROOT}/models/Qwen2.5-7B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/checkpoints}"
NUM_GPUS="${NUM_GPUS:-8}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-true}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/train_seen_10_un_10_attr_8_variant.parquet}"
VAL_FILE="${VAL_FILE:-${DATA_DIR}/val_seen_10_un_10_attr_8_variant.parquet}"

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"

experiments=(
  "baseline:false:2"
)

for exp_config in "${experiments[@]}"; do
  IFS=':' read -r experiment_name early_cut trunc_strength <<< "$exp_config"

  echo "Processing experiment: $experiment_name, early_cut: $early_cut, trunc_strength: $trunc_strength"

  project_name="${DATA_NAME}-Train"
  default_local_dir="${OUTPUT_ROOT}/${project_name}/${experiment_name}"

  mkdir -p "${default_local_dir}"

  python3 -m verl.trainer.main_ppo \
      max_turns=10 \
      data_name="${DATA_NAME}" \
      use_interactions=false \
      early_cut="${early_cut}" \
      +trunc_strength="${trunc_strength}" \
      \
      data.train_files="${TRAIN_FILE}" \
      data.val_files="${VAL_FILE}" \
      data.train_batch_size=512 \
      data.val_batch_size=512 \
      data.max_prompt_length=6656 \
      data.max_response_length=512 \
      data.max_start_length=2048 \
      data.max_obs_length=512 \
      \
      algorithm.adv_estimator=gae \
      \
      actor_rollout_ref.model.path="${BASE_MODEL}" \
      actor_rollout_ref.model.enable_gradient_checkpointing=true \
      actor_rollout_ref.model.use_remove_padding=true \
      actor_rollout_ref.actor.optim.lr=1e-6 \
      actor_rollout_ref.actor.use_kl_loss=false \
      actor_rollout_ref.actor.ppo_mini_batch_size=128 \
      actor_rollout_ref.actor.fsdp_config.param_offload=true \
      actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
      \
      actor_rollout_ref.rollout.name=vllm \
      actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
      actor_rollout_ref.rollout.n=1 \
      actor_rollout_ref.rollout.temperature=1.0 \
      \
      actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
      actor_rollout_ref.ref.fsdp_config.param_offload=true \
      \
      actor_rollout_ref.actor.use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
      actor_rollout_ref.ref.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
      actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
      critic.use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
      \
      critic.optim.lr=1e-5 \
      critic.model.use_remove_padding=true \
      critic.optim.lr_warmup_steps_ratio=0.015 \
      critic.model.path="${BASE_MODEL}" \
      critic.model.enable_gradient_checkpointing=true \
      critic.model.fsdp_config.param_offload=true \
      critic.model.fsdp_config.optimizer_offload=true \
      \
      trainer.logger='["console"]' \
      trainer.val_only=false \
      trainer.val_before_train=true \
      trainer.default_hdfs_dir=null \
      trainer.project_name="${project_name}" \
      trainer.experiment_name="${experiment_name}" \
      trainer.n_gpus_per_node="${NUM_GPUS}" \
      trainer.nnodes=1 \
      trainer.save_freq=100 \
      trainer.test_freq=6 \
      trainer.total_epochs=200 \
      trainer.total_training_steps=200 \
      trainer.default_local_dir="${default_local_dir}" \
      2>&1 | tee "${default_local_dir}.log"

  echo "Completed: $experiment_name"
done
