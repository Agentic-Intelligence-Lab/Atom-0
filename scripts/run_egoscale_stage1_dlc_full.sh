#!/usr/bin/env bash
set -euo pipefail

# Production Stage 1 profile for Alibaba PAI DLC.
# DLC must start exactly one process per node and provide WORLD_SIZE/RANK and
# MASTER_ADDR/MASTER_PORT for multi-node jobs. Do not wrap this script in torchrun.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PYTHON_BIN="${PYTHON_BIN:-${REPO_DIR}/.venv/bin/python}"
export STAGE=stage1_ego
export ATOM_RLDS_ROOT="${ATOM_RLDS_ROOT:-/mnt/workspace/RLDS}"
export ATOM_PI05_BASE_PARAMS="${ATOM_PI05_BASE_PARAMS:-/mnt/workspace/cache/openpi/openpi-assets/checkpoints/pi05_base/params}"
export ASSETS_BASE_DIR="${ASSETS_BASE_DIR:-${REPO_DIR}/assets}"
export CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-/path/to/atom-workspace/checkpoints}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/path/to/atom-workspace/cache/jax}"

# The production recipe is a global batch of 512 on 2 nodes x 8 GPUs. The
# training entrypoint verifies divisibility against the actual global device count.
export FSDP_DEVICES="${FSDP_DEVICES:-8}"
export BATCH_SIZE="${BATCH_SIZE:-512}"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-100000}"
export LOG_INTERVAL="${LOG_INTERVAL:-100}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
export NUM_VAL_BATCHES="${NUM_VAL_BATCHES:-10}"
export NUM_ACTION_MSE_BATCHES="${NUM_ACTION_MSE_BATCHES:-2}"
export SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-50000}"
export DATA_NUM_PARALLEL_READS="${DATA_NUM_PARALLEL_READS:--1}"
export DATA_NUM_PARALLEL_CALLS="${DATA_NUM_PARALLEL_CALLS:--1}"
export WANDB_ENABLED="${WANDB_ENABLED:-1}"
export RUN_ACTION_MSE="${RUN_ACTION_MSE:-1}"
export RESUME="${RESUME:-0}"
# Default to non-destructive initialization. A new experiment directory does not
# need --overwrite; resuming requires RESUME=1 and the same EXP_NAME.
export OVERWRITE="${OVERWRITE:-0}"

if [[ "${CHECKPOINT_PARAMS_ONLY:-0}" == "1" ]]; then
  echo "DLC full training requires complete optimizer checkpoints; CHECKPOINT_PARAMS_ONLY must be 0." >&2
  exit 2
fi
export CHECKPOINT_PARAMS_ONLY=0

if [[ "${PREFLIGHT_ONLY:-0}" != "1" ]]; then
  : "${EXP_NAME:?Set a unique EXP_NAME for a new run, or the existing name with RESUME=1}"
fi

if (( ${WORLD_SIZE:-1} > 1 )); then
  : "${RANK:?DLC must provide node-level RANK for multi-node training}"
  : "${MASTER_ADDR:?DLC must provide MASTER_ADDR for multi-node training}"
fi

echo "DLC Stage 1 profile: world_size=${WORLD_SIZE:-1} fsdp=${FSDP_DEVICES} global_batch=${BATCH_SIZE} steps=${NUM_TRAIN_STEPS}"
exec bash "${REPO_DIR}/scripts/run_egoscale_stage.sh" "$@"
