#!/usr/bin/env bash
set -euo pipefail

DATA_NAME="CircuitDecoding"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/${DATA_NAME}}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${PROJECT_ROOT}/checkpoints}"
NUM_GPUS="${NUM_GPUS:-${ARNOLD_WORKER_GPU:-8}}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-true}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/train__cand_10.parquet}"
VAL_FILE="${VAL_FILE:-${DATA_DIR}/val__cand_10.parquet}"

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"

# Checkpoint / step pairs to evaluate.
checkpoint_steps=(
  "baseline:147"
)

for pair in "${checkpoint_steps[@]}"; do
  IFS=':' read -r CHECKPOINT STEP <<< "$pair"

  echo "Processing checkpoint: $CHECKPOINT, step: $STEP"

  MERGED_MODEL_DIR="${CHECKPOINT_ROOT}/CircuitDecoding-Train/${CHECKPOINT}/actor-hf"
  python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${CHECKPOINT_ROOT}/CircuitDecoding-Train/${CHECKPOINT}/actor/best_step_${STEP}" \
    --target_dir "${MERGED_MODEL_DIR}"

  BASE_MODEL="${MERGED_MODEL_DIR}"

  project_name="${DATA_NAME}-Eval"
  experiment_name="${CHECKPOINT}_${STEP}"
  default_local_dir="${CHECKPOINT_ROOT}/${project_name}/${experiment_name}"
  mkdir -p "${default_local_dir}"

  python3 -m verl.trainer.main_ppo \
      max_turns=10 \
      data_name="${DATA_NAME}" \
      use_interactions=false \
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
      algorithm.adv_estimator=grpo \
      \
      actor_rollout_ref.model.path="${BASE_MODEL}" \
      actor_rollout_ref.model.enable_gradient_checkpointing=true \
      actor_rollout_ref.model.use_remove_padding=true \
      \
      actor_rollout_ref.actor.optim.lr=1e-6 \
      actor_rollout_ref.actor.use_kl_loss=false \
      actor_rollout_ref.actor.ppo_mini_batch_size=128 \
      actor_rollout_ref.actor.fsdp_config.param_offload=true \
      actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
      \
      actor_rollout_ref.rollout.name=vllm \
      actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
      actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
      actor_rollout_ref.ref.fsdp_config.param_offload=true \
      \
      actor_rollout_ref.actor.use_dynamic_bsz=${USE_DYNAMIC_BSZ} \
      actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${USE_DYNAMIC_BSZ} \
      actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${USE_DYNAMIC_BSZ} \
      \
      trainer.logger=[] \
      trainer.val_only=true \
      trainer.val_before_train=true \
      trainer.default_hdfs_dir=null \
      trainer.project_name="${project_name}" \
      trainer.experiment_name="${experiment_name}" \
      trainer.n_gpus_per_node="${NUM_GPUS}" \
      trainer.nnodes=1 \
      trainer.default_local_dir="${default_local_dir}" \
      2>&1 | tee "${default_local_dir}.log"

  echo "Completed: $CHECKPOINT-$STEP"
done
