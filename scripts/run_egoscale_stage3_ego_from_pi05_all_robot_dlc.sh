#!/usr/bin/env bash
set -Eeuo pipefail

# Fresh full-parameter Stage 3 EgoVerse+RL2 fine-tune initialized from the
# Pi0.5-lineage aligned Stage 2 checkpoint at step 30000.
# PAI DLC must launch exactly one process on each of two 8-GPU nodes.

REPO_DIR=/path/to/Atom-0
RANK_ID="${RANK:-unknown}"
TRAIN_LOG="${STAGE3_DLC_LOG:-/path/to/atom-workspace/logs/stage3_ego_from_pi05_all_robot_30000_dlc_rank${RANK_ID}.log}"
mkdir -p "$(dirname "${TRAIN_LOG}")"
exec > >(tee -a "${TRAIN_LOG}") 2>&1
trap 'rc=$?; echo "FAILED rc=${rc} line=${LINENO} command=${BASH_COMMAND}"; exit "${rc}"' ERR

echo "===== Stage 3 Ego Pi0.5 lineage DLC $(date --iso-8601=seconds) ====="
cd "${REPO_DIR}"

: "${WORLD_SIZE:?DLC must provide WORLD_SIZE=2}"
: "${RANK:?DLC must provide node-level RANK=0 or RANK=1}"
: "${MASTER_ADDR:?DLC must provide MASTER_ADDR}"
if [[ "${WORLD_SIZE}" != "2" ]]; then
  echo "Expected WORLD_SIZE=2 nodes, got ${WORLD_SIZE}." >&2
  exit 2
fi
if [[ "${RANK}" != "0" && "${RANK}" != "1" ]]; then
  echo "Expected node-level RANK=0 or RANK=1, got ${RANK}." >&2
  exit 2
fi
LOCAL_GPU_COUNT="$(nvidia-smi -L | wc -l)"
if [[ "${LOCAL_GPU_COUNT}" != "8" ]]; then
  echo "Expected exactly 8 local GPUs on rank ${RANK}, got ${LOCAL_GPU_COUNT}." >&2
  exit 2
fi
nvidia-smi -L
export JAX_COORDINATOR_ADDRESS="${JAX_COORDINATOR_ADDRESS:-${MASTER_ADDR}:${MASTER_PORT:-29500}}"
echo "DLC topology: rank=${RANK}/${WORLD_SIZE} local_gpus=${LOCAL_GPU_COUNT} global_gpus=16 coordinator=${JAX_COORDINATOR_ADDRESS}"

export ATOM_RLDS_ROOT=/mnt/data/RLDS
export ATOM_EGOVERSE_RL2_ROOT=/mnt/data/RLDS/EgoVerse_rl2
export ASSETS_BASE_DIR="${REPO_DIR}/assets"
export CHECKPOINT_BASE_DIR="${REPO_DIR}/checkpoints"
export JAX_COMPILATION_CACHE_DIR=/path/to/atom-workspace/cache/jax
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export TF_FORCE_GPU_ALLOW_GROWTH=true
mkdir -p "${JAX_COMPILATION_CACHE_DIR}"

export PARAMS_PATH="${REPO_DIR}/checkpoints/egoscale_stage2_aligned/pi05_all_robot/30000/params"
test -f "${PARAMS_PATH}/manifest.ocdbt"
test -f "${PARAMS_PATH}/_METADATA"

if [[ ! -x .venv/bin/python ]]; then
  echo "Venv missing; rebuilding it in the DLC container."
  bash scripts/setup_aliyun_dsw_env.sh
fi
export PYTHON_BIN="${REPO_DIR}/.venv/bin/python"

export WANDB_ENTITY="${WANDB_ENTITY:-your-wandb-team}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENABLED=1
if [[ "${PREFLIGHT_ONLY:-0}" != "1" ]]; then
  : "${WANDB_API_KEY:?Inject WANDB_API_KEY through a DLC Secret environment variable}"
fi

export STAGE=stage3_ego
export EXP_NAME=stage3_ego_from_pi05_all_robot_30000_dlc_2x8_b512_100k_20260901_v1
export FSDP_DEVICES=8
export BATCH_SIZE=512
export NUM_TRAIN_STEPS=100000
export LOG_INTERVAL=100
export SAVE_INTERVAL=5000
export EVAL_INTERVAL=1000
export NUM_VAL_BATCHES=10
export NUM_ACTION_MSE_BATCHES=2
export RUN_ACTION_MSE=1
export SHUFFLE_BUFFER_SIZE=50000
export DATA_NUM_PARALLEL_READS=4
export DATA_NUM_PARALLEL_CALLS=8
export CHECKPOINT_PARAMS_ONLY=0
export RESUME=0
export OVERWRITE=0

EXPERIMENT_DIR="${CHECKPOINT_BASE_DIR}/egoscale_stage3_ego/${EXP_NAME}"
if [[ "${RANK}" == "0" && -e "${EXPERIMENT_DIR}" ]]; then
  echo "Refusing to reuse existing experiment directory: ${EXPERIMENT_DIR}" >&2
  exit 2
fi

echo "Fresh model params: ${PARAMS_PATH}"
echo "EgoVerse root: ${ATOM_RLDS_ROOT}/EgoVerse_full"
echo "EgoVerse RL2 root: ${ATOM_EGOVERSE_RL2_ROOT}"
echo "New experiment: ${EXPERIMENT_DIR}"

exec bash scripts/run_egoscale_stage.sh \
  --project-name Atom0 \
  --val-batch-size 96 \
  --val-flow-loss-mode fixed_seed \
  --seed 42
