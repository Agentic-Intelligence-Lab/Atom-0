#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"
source scripts/atom0_env.sh

CONFIG_NAME="${CONFIG_NAME:?Set a supported FastWAM CONFIG_NAME}"
EXP_NAME="${EXP_NAME:?Set EXP_NAME}"
MODE="${MODE:-train}"

# FastWAM reuses norm assets under assets/<assets_name>; default matches
# fastwam_cotrain_real_robot_ego_fix -> cotrain_real_robot_ego_fix.
ASSET_CONFIG_NAME="${ASSET_CONFIG_NAME:-cotrain_real_robot_ego_fix}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${REPO_DIR}/checkpoints}"
ASSETS_BASE_DIR="${ASSETS_BASE_DIR:-${REPO_DIR}/assets}"
DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${REPO_DIR}/checkpoints/fastwam}"
export DIFFSYNTH_MODEL_BASE_PATH

case "${CONFIG_NAME}" in
  fastwam_cotrain_real_robot_ego_fix)
    DEFAULT_STEPS=100000
    DEFAULT_WARMUP=1000
    DEFAULT_BATCH_SIZE=32
    DEFAULT_LOG_INTERVAL=50
    DEFAULT_SAVE_INTERVAL=2000
    DEFAULT_SHUFFLE_BUFFER=10000
    ;;
  fastwam_cotrain_real_robot_ego_fix_debug)
    DEFAULT_STEPS=2
    DEFAULT_WARMUP=0
    DEFAULT_BATCH_SIZE=2
    DEFAULT_LOG_INTERVAL=1
    DEFAULT_SAVE_INTERVAL=10
    DEFAULT_SHUFFLE_BUFFER=256
    ;;
  *)
    echo "Unsupported CONFIG_NAME=${CONFIG_NAME}" >&2
    exit 2
    ;;
esac

GLOBAL_DEVICE_COUNT=$((${WORLD_SIZE:-1} * ${NPROC_PER_NODE:-8}))
BATCH_SIZE="${BATCH_SIZE:-${DEFAULT_BATCH_SIZE}}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-${DEFAULT_STEPS}}"
LOG_INTERVAL="${LOG_INTERVAL:-${DEFAULT_LOG_INTERVAL}}"
SAVE_INTERVAL="${SAVE_INTERVAL:-${DEFAULT_SAVE_INTERVAL}}"
SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-${DEFAULT_SHUFFLE_BUFFER}}"
RANK_ID="${RANK:-0}"

if (( BATCH_SIZE <= 0 || BATCH_SIZE % GLOBAL_DEVICE_COUNT != 0 )); then
  echo "Invalid BATCH_SIZE=${BATCH_SIZE}: must be positive and divisible by ${GLOBAL_DEVICE_COUNT} devices" >&2
  exit 2
fi

test -d "${RLDS_DATA_DIR}"
test -d "${ASSETS_BASE_DIR}/${ASSET_CONFIG_NAME}"
mkdir -p "${DIFFSYNTH_MODEL_BASE_PATH}" "${CHECKPOINT_BASE_DIR}" "${LOG_DIR}"

if [[ "${MODE}" == "smoke" ]]; then
  NUM_TRAIN_STEPS="${SMOKE_STEPS:-20}"
  LOG_INTERVAL=1
  SAVE_INTERVAL=1000000
  WANDB_ENABLED=0
  SHUFFLE_BUFFER_SIZE="${SMOKE_SHUFFLE_BUFFER:-512}"
else
  WANDB_ENABLED="${WANDB_ENABLED:-1}"
fi

if [[ "${WANDB_ENABLED}" == "1" ]]; then
  : "${WANDB_API_KEY:?Set WANDB_API_KEY for production training}"
fi

PYTHON_BIN="${PYTHON_BIN:-${REPO_DIR}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing ${PYTHON_BIN}; create Atom-0 .venv on PFS before submitting Baige jobs." >&2
  exit 2
fi

args=(
  "${CONFIG_NAME}"
  "--exp-name=${EXP_NAME}"
  "--batch-size=${BATCH_SIZE}"
  "--num-train-steps=${NUM_TRAIN_STEPS}"
  "--log-interval=${LOG_INTERVAL}"
  "--save-interval=${SAVE_INTERVAL}"
  "--shuffle-buffer-size=${SHUFFLE_BUFFER_SIZE}"
  "--data-num-parallel-reads=${DATA_NUM_PARALLEL_READS:-1}"
  "--data-num-parallel-calls=${DATA_NUM_PARALLEL_CALLS:-2}"
  "--data.rlds-data-dir=${RLDS_DATA_DIR}"
  "--assets-base-dir=${ASSETS_BASE_DIR}"
  "--checkpoint-base-dir=${CHECKPOINT_BASE_DIR}"
  "--lr-schedule.warmup-steps=${WARMUP_STEPS:-${DEFAULT_WARMUP}}"
  "--lr-schedule.decay-steps=${DECAY_STEPS:-${NUM_TRAIN_STEPS}}"
)

if [[ "${WANDB_ENABLED}" == "1" ]]; then
  args+=("--wandb-enabled")
else
  args+=("--no-wandb-enabled")
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  args+=("--overwrite")
fi

exec > >(tee -a "${LOG_DIR}/baige_${CONFIG_NAME}_${EXP_NAME}_rank${RANK_ID}.log") 2>&1
echo "CONFIG_NAME=${CONFIG_NAME} EXP_NAME=${EXP_NAME} MODE=${MODE}"
echo "WORLD_SIZE=${WORLD_SIZE:-1} RANK=${RANK_ID} MASTER=${MASTER_ADDR:-127.0.0.1}:${MASTER_PORT:-29500}"
echo "BATCH_SIZE=${BATCH_SIZE} NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS} SHUFFLE_BUFFER_SIZE=${SHUFFLE_BUFFER_SIZE}"
echo "RLDS_DATA_DIR=${RLDS_DATA_DIR} ASSETS=${ASSETS_BASE_DIR}/${ASSET_CONFIG_NAME}"
echo "DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH}"

exec torchrun \
  --nproc_per_node="${NPROC_PER_NODE:-8}" \
  --nnodes="${WORLD_SIZE:-1}" \
  --node_rank="${RANK_ID}" \
  --master_addr="${MASTER_ADDR:-127.0.0.1}" \
  --master_port="${MASTER_PORT:-29500}" \
  scripts/train_fastwam.py "${args[@]}"
