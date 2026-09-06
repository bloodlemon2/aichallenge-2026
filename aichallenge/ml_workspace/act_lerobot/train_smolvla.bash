#!/usr/bin/env bash
set -euo pipefail

DATASET_REPO_ID="${DATASET_REPO_ID:-local/aichallenge_act}"
DATASET_ROOT="${DATASET_ROOT:-./dataset/aichallenge_act}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/train/smolvla_aichallenge}"
POLICY_REPO_ID="${POLICY_REPO_ID:-local/smolvla_aichallenge_policy}"
DEVICE="${DEVICE:-cuda}"
STEPS="${STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
CHUNK_SIZE="${CHUNK_SIZE:-20}"
N_ACTION_STEPS="${N_ACTION_STEPS:-20}"
NUM_WORKERS="${NUM_WORKERS:-2}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
LOG_FREQ="${LOG_FREQ:-100}"
POLICY_PATH="${POLICY_PATH:-lerobot/smolvla_base}"

COMMON_ARGS=(
  --dataset.repo_id="${DATASET_REPO_ID}"
  --dataset.root="${DATASET_ROOT}"
  --output_dir="${OUTPUT_DIR}"
  --job_name=smolvla_aichallenge
  --policy.device="${DEVICE}"
  --policy.repo_id="${POLICY_REPO_ID}"
  --policy.push_to_hub=false
  --wandb.enable="${WANDB_ENABLE}"
  --steps="${STEPS}"
  --batch_size="${BATCH_SIZE}"
  --num_workers="${NUM_WORKERS}"
  --save_freq="${SAVE_FREQ}"
  --log_freq="${LOG_FREQ}"
  --policy.chunk_size="${CHUNK_SIZE}"
  --policy.n_action_steps="${N_ACTION_STEPS}"
)

if [[ "${POLICY_PATH}" == "scratch" || "${POLICY_PATH}" == "none" ]]; then
  exec lerobot-train \
    "${COMMON_ARGS[@]}" \
    --policy.type=smolvla
fi

exec lerobot-train \
  "${COMMON_ARGS[@]}" \
  --policy.path="${POLICY_PATH}"
