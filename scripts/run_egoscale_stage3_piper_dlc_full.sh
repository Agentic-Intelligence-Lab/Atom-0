#!/usr/bin/env bash
set -euo pipefail

# Alibaba PAI DLC profile for the Stage 3 Piper-only checkpoint fine-tune.
# DLC starts one Python process per node. Do not wrap this JAX command in torchrun.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PYTHON_BIN="${PYTHON_BIN:-${REPO_DIR}/.venv/bin/python}"
export STAGE=stage3_robot
export ATOM_RLDS_ROOT="${ATOM_RLDS_ROOT:-/mnt/data/RLDS}"
export ASSETS_BASE_DIR="${ASSETS_BASE_DIR:-${REPO_DIR}/assets}"
export CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${REPO_DIR}/checkpoints}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/path/to/atom-workspace/cache/jax}"

# Stage 3 must initialize parameters from Stage 2 while creating a fresh optimizer
# and step counter. PARAMS_PATH must be the completed Stage 2 <step>/params path.
: "${PARAMS_PATH:?Set PARAMS_PATH to the Stage 2 20000/params directory}"

# Formal-training-1 recipe: Piper30 + Piper2, Unified80, one 8-GPU node.
export FSDP_DEVICES="${FSDP_DEVICES:-4}"
export BATCH_SIZE="${BATCH_SIZE:-512}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-96}"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-20000}"
export LOG_INTERVAL="${LOG_INTERVAL:-100}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
export NUM_VAL_BATCHES="${NUM_VAL_BATCHES:-10}"
export NUM_ACTION_MSE_BATCHES="${NUM_ACTION_MSE_BATCHES:-2}"
export SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-50000}"
export DATA_NUM_PARALLEL_READS="${DATA_NUM_PARALLEL_READS:-1}"
export DATA_NUM_PARALLEL_CALLS="${DATA_NUM_PARALLEL_CALLS:-2}"
export WANDB_ENABLED="${WANDB_ENABLED:-1}"
export RUN_ACTION_MSE="${RUN_ACTION_MSE:-1}"
export RESUME="${RESUME:-0}"
export OVERWRITE="${OVERWRITE:-0}"

if [[ "${CHECKPOINT_PARAMS_ONLY:-0}" == "1" ]]; then
  echo "DLC formal training requires full optimizer checkpoints; CHECKPOINT_PARAMS_ONLY must be 0." >&2
  exit 2
fi
export CHECKPOINT_PARAMS_ONLY=0

if [[ "${PREFLIGHT_ONLY:-0}" != "1" ]]; then
  : "${EXP_NAME:?Set a new EXP_NAME; do not reuse the Stage 2 experiment name}"
fi

if (( ${WORLD_SIZE:-1} > 1 )); then
  : "${RANK:?DLC must provide node-level RANK for multi-node training}"
  : "${MASTER_ADDR:?DLC must provide MASTER_ADDR for multi-node training}"
fi

echo "DLC Stage 3 Piper fine-tune: world_size=${WORLD_SIZE:-1} fsdp=${FSDP_DEVICES} global_batch=${BATCH_SIZE} steps=${NUM_TRAIN_STEPS}"
exec bash "${REPO_DIR}/scripts/run_egoscale_stage.sh" \
  --val-batch-size "${VAL_BATCH_SIZE}" \
  --val-flow-loss-mode fixed_seed \
  "$@"
