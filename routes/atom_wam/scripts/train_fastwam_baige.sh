#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"
source scripts/atom0_env.sh

CONFIG_NAME="${CONFIG_NAME:?Set a supported FastWAM CONFIG_NAME}"
EXP_NAME="${EXP_NAME:?Set EXP_NAME}"
MODE="${MODE:-train}"

case "${CONFIG_NAME}" in
  wam-cross-robot)
    DEFAULT_STEPS=300000
    DEFAULT_WARMUP=15000   # 5% of DEFAULT_STEPS; train_fastwam enforces 5% anyway
    DEFAULT_BATCH_SIZE=160
    DEFAULT_LOG_INTERVAL=50
    DEFAULT_SAVE_INTERVAL=10000
    DEFAULT_EVAL_INTERVAL=0
    DEFAULT_VAL_BATCH_SIZE=4
    DEFAULT_NUM_VAL_BATCHES=1
    DEFAULT_NUM_ACTION_MSE_BATCHES=1
    DEFAULT_RUN_ACTION_MSE=0
    DEFAULT_VAL_FLOW_LOSS_MODE=fixed_seed
    DEFAULT_VAL_MAX_DATASETS=8
    DEFAULT_SHUFFLE_BUFFER=256
    DEFAULT_PEAK_LR="1e-4"
    DEFAULT_DECAY_LR="1e-6"
    DEFAULT_ASSET_CONFIG_NAME="atom_wam_paper50"
    DEFAULT_RLDS_PARTITION_BUILDERS=0
    ;;
  wam-cross-fix)
    DEFAULT_STEPS=300000
    DEFAULT_WARMUP=15000   # 5% of DEFAULT_STEPS; train_fastwam enforces 5% anyway
    DEFAULT_BATCH_SIZE=160
    DEFAULT_LOG_INTERVAL=50
    DEFAULT_SAVE_INTERVAL=10000
    DEFAULT_EVAL_INTERVAL=0
    DEFAULT_VAL_BATCH_SIZE=4
    DEFAULT_NUM_VAL_BATCHES=1
    DEFAULT_NUM_ACTION_MSE_BATCHES=1
    DEFAULT_RUN_ACTION_MSE=0
    DEFAULT_VAL_FLOW_LOSS_MODE=fixed_seed
    DEFAULT_VAL_MAX_DATASETS=8
    DEFAULT_SHUFFLE_BUFFER=256
    DEFAULT_PEAK_LR="1e-4"
    DEFAULT_DECAY_LR="1e-6"
    DEFAULT_ASSET_CONFIG_NAME="atom_wam_paper50"
    DEFAULT_RLDS_PARTITION_BUILDERS=1
    ;;
  wam-cross-piper-ft)
    DEFAULT_STEPS=20000
    DEFAULT_WARMUP=1000    # 5% of DEFAULT_STEPS; train_fastwam enforces 5% anyway
    DEFAULT_BATCH_SIZE=208
    DEFAULT_LOG_INTERVAL=50
    DEFAULT_SAVE_INTERVAL=5000
    DEFAULT_EVAL_INTERVAL=1000
    DEFAULT_VAL_BATCH_SIZE=4
    DEFAULT_NUM_VAL_BATCHES=10
    DEFAULT_NUM_ACTION_MSE_BATCHES=2
    DEFAULT_RUN_ACTION_MSE=1
    DEFAULT_VAL_FLOW_LOSS_MODE=fixed_seed
    DEFAULT_VAL_MAX_DATASETS=
    DEFAULT_SHUFFLE_BUFFER=256
    DEFAULT_PEAK_LR="1e-4"
    DEFAULT_DECAY_LR="1e-6"
    DEFAULT_ASSET_CONFIG_NAME="atom_wam_paper50"
    DEFAULT_RLDS_PARTITION_BUILDERS=0
    ;;
  *)
    echo "Unsupported CONFIG_NAME=${CONFIG_NAME} (expected wam-cross-robot, wam-cross-fix, or wam-cross-piper-ft)" >&2
    exit 2
    ;;
esac

ASSET_CONFIG_NAME="${ASSET_CONFIG_NAME:-${DEFAULT_ASSET_CONFIG_NAME}}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${REPO_DIR}/checkpoints}"
ASSETS_BASE_DIR="${ASSETS_BASE_DIR:-${REPO_DIR}/assets}"
DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${REPO_DIR}/checkpoints/fastwam}"
export DIFFSYNTH_MODEL_BASE_PATH
export DIFFSYNTH_DOWNLOAD_SOURCE="${DIFFSYNTH_DOWNLOAD_SOURCE:-huggingface}"
export HF_HOME="${HF_HOME:-${DIFFSYNTH_MODEL_BASE_PATH}/hf_cache}"

GLOBAL_DEVICE_COUNT=$((${WORLD_SIZE:-1} * ${NPROC_PER_NODE:-8}))
BATCH_SIZE="${BATCH_SIZE:-${DEFAULT_BATCH_SIZE}}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-${DEFAULT_STEPS}}"
LOG_INTERVAL="${LOG_INTERVAL:-${DEFAULT_LOG_INTERVAL}}"
SAVE_INTERVAL="${SAVE_INTERVAL:-${DEFAULT_SAVE_INTERVAL:-2000}}"
EVAL_INTERVAL="${EVAL_INTERVAL:-${DEFAULT_EVAL_INTERVAL:-0}}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-${DEFAULT_VAL_BATCH_SIZE:-8}}"
NUM_VAL_BATCHES="${NUM_VAL_BATCHES:-${DEFAULT_NUM_VAL_BATCHES:-2}}"
NUM_ACTION_MSE_BATCHES="${NUM_ACTION_MSE_BATCHES:-${DEFAULT_NUM_ACTION_MSE_BATCHES:-1}}"
RUN_ACTION_MSE="${RUN_ACTION_MSE:-${DEFAULT_RUN_ACTION_MSE:-1}}"
VAL_FLOW_LOSS_MODE="${VAL_FLOW_LOSS_MODE:-${DEFAULT_VAL_FLOW_LOSS_MODE:-fixed_seed}}"
VAL_MAX_DATASETS="${VAL_MAX_DATASETS:-${DEFAULT_VAL_MAX_DATASETS:-}}"
SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-${DEFAULT_SHUFFLE_BUFFER}}"
RANK_ID="${RANK:-0}"

if (( BATCH_SIZE <= 0 || BATCH_SIZE % GLOBAL_DEVICE_COUNT != 0 )); then
  echo "Invalid BATCH_SIZE=${BATCH_SIZE}: must be positive and divisible by ${GLOBAL_DEVICE_COUNT} devices" >&2
  exit 2
fi

if [[ ! -d "${RLDS_DATA_DIR}" ]]; then
  echo "RLDS_DATA_DIR not found: ${RLDS_DATA_DIR}" >&2
  echo "Set RLDS_DATA_DIR to the BOS mount (e.g. /path/to/rlds) before submitting." >&2
  exit 2
fi
if [[ ! -d "${ASSETS_BASE_DIR}/${ASSET_CONFIG_NAME}" ]]; then
  echo "Norm assets not found: ${ASSETS_BASE_DIR}/${ASSET_CONFIG_NAME}" >&2
  exit 2
fi
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

# Default warmup = 5% of steps when WARMUP_STEPS unset; train_fastwam.py uses config.lr_schedule.warmup_steps.
WARMUP_STEPS="${WARMUP_STEPS:-$(( (NUM_TRAIN_STEPS * 5 + 99) / 100 ))}"
DECAY_STEPS="${DECAY_STEPS:-${NUM_TRAIN_STEPS}}"

if [[ "${WANDB_ENABLED}" == "1" ]]; then
  : "${WANDB_API_KEY:?Set WANDB_API_KEY for production training}"
fi

PYTHON_BIN="${PYTHON_BIN:-${REPO_DIR}/.venv/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${REPO_DIR}/.venv/bin/torchrun}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing ${PYTHON_BIN}; create Atom-0 .venv on PFS before submitting Baige jobs." >&2
  exit 2
fi
if [[ ! -x "${TORCHRUN_BIN}" ]]; then
  echo "Missing ${TORCHRUN_BIN}; expected torchrun in Atom-0 .venv." >&2
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
  "--data-num-parallel-calls=${DATA_NUM_PARALLEL_CALLS:-1}"
  "--data.rlds-data-dir=${RLDS_DATA_DIR}"
  "--assets-base-dir=${ASSETS_BASE_DIR}"
  "--checkpoint-base-dir=${CHECKPOINT_BASE_DIR}"
  "--lr-schedule.warmup-steps=${WARMUP_STEPS:-${DEFAULT_WARMUP}}"
  "--lr-schedule.decay-steps=${DECAY_STEPS:-${NUM_TRAIN_STEPS}}"
)
if [[ -n "${PEAK_LR:-${DEFAULT_PEAK_LR:-}}" ]]; then
  args+=("--lr-schedule.peak-lr=${PEAK_LR:-${DEFAULT_PEAK_LR}}")
fi
if [[ -n "${DECAY_LR:-${DEFAULT_DECAY_LR:-}}" ]]; then
  args+=("--lr-schedule.decay-lr=${DECAY_LR:-${DEFAULT_DECAY_LR}}")
fi
if [[ "${EVAL_INTERVAL:-0}" -gt 0 ]]; then
  args+=(
    "--eval-interval=${EVAL_INTERVAL}"
    "--val-batch-size=${VAL_BATCH_SIZE}"
    "--num-val-batches=${NUM_VAL_BATCHES}"
    "--num-action-mse-batches=${NUM_ACTION_MSE_BATCHES}"
    "--val-flow-loss-mode=${VAL_FLOW_LOSS_MODE}"
  )
  if [[ -n "${VAL_MAX_DATASETS}" ]]; then
    args+=("--val-max-datasets=${VAL_MAX_DATASETS}")
  fi
else
  args+=("--eval-interval=0")
fi

if [[ "${WANDB_ENABLED}" == "1" ]]; then
  args+=("--wandb-enabled")
else
  args+=("--no-wandb-enabled")
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  args+=("--overwrite")
fi
if [[ "${RESUME:-0}" == "1" ]]; then
  args+=("--resume")
fi
# Fine-tune init: accept step dir / exp dir / model.safetensors.
# Prefer PYTORCH_WEIGHT_PATH; INIT_CHECKPOINT is an alias.
INIT_CKPT="${PYTORCH_WEIGHT_PATH:-${INIT_CHECKPOINT:-}}"
if [[ -n "${INIT_CKPT}" ]]; then
  if [[ "${INIT_CKPT}" != /* ]]; then
    INIT_CKPT="${REPO_DIR}/${INIT_CKPT}"
  fi
  args+=("--pytorch-weight-path=${INIT_CKPT}")
fi
if [[ "${RUN_ACTION_MSE}" == "1" ]]; then
  args+=("--run-action-mse")
else
  args+=("--no-run-action-mse")
fi
if [[ "${RLDS_PARTITION_BUILDERS:-${DEFAULT_RLDS_PARTITION_BUILDERS:-}}" == "0" ]]; then
  args+=("--no-rlds-partition-builders-by-rank")
elif [[ "${RLDS_PARTITION_BUILDERS:-${DEFAULT_RLDS_PARTITION_BUILDERS:-}}" == "1" ]]; then
  args+=("--rlds-partition-builders-by-rank")
fi
# MoT video→action attention (default on in FastWAMConfig). Set 0 to disable.
if [[ "${MOT_VIDEO_ATTENDS_TO_ACTION:-}" == "0" ]]; then
  args+=("--model.no-mot-video-attends-to-action")
elif [[ "${MOT_VIDEO_ATTENDS_TO_ACTION:-}" == "1" ]]; then
  args+=("--model.mot-video-attends-to-action")
fi
# Optional loss weight overrides (config defaults otherwise).
if [[ -n "${LAMBDA_ROBOT_VIDEO:-}" ]]; then
  args+=("--model.loss.lambda-robot-video=${LAMBDA_ROBOT_VIDEO}")
fi
if [[ -n "${LAMBDA_ROBOT_ACTION:-}" ]]; then
  args+=("--model.loss.lambda-robot-action=${LAMBDA_ROBOT_ACTION}")
fi
if [[ -n "${LAMBDA_EGO_VIDEO:-}" ]]; then
  args+=("--model.loss.lambda-ego-video=${LAMBDA_EGO_VIDEO}")
fi
if [[ -n "${LAMBDA_EGO_ACTION:-}" ]]; then
  args+=("--model.loss.lambda-ego-action=${LAMBDA_EGO_ACTION}")
fi

exec > >(tee -a "${LOG_DIR}/baige_${CONFIG_NAME}_${EXP_NAME}_rank${RANK_ID}.log") 2>&1
echo "CONFIG_NAME=${CONFIG_NAME} EXP_NAME=${EXP_NAME} MODE=${MODE}"
echo "WORLD_SIZE=${WORLD_SIZE:-1} RANK=${RANK_ID} MASTER=${MASTER_ADDR:-127.0.0.1}:${MASTER_PORT:-29500}"
echo "BATCH_SIZE=${BATCH_SIZE} NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS} SHUFFLE_BUFFER_SIZE=${SHUFFLE_BUFFER_SIZE}"
echo "EVAL_INTERVAL=${EVAL_INTERVAL} (0=disabled)"
echo "MOT_VIDEO_ATTENDS_TO_ACTION=${MOT_VIDEO_ATTENDS_TO_ACTION:-default}"
echo "LAMBDA_ROBOT_VIDEO=${LAMBDA_ROBOT_VIDEO:-default} LAMBDA_ROBOT_ACTION=${LAMBDA_ROBOT_ACTION:-default}"
echo "INIT_CKPT=${INIT_CKPT:-} RESUME=${RESUME:-0}"
echo "RLDS_DATA_DIR=${RLDS_DATA_DIR} ASSETS=${ASSETS_BASE_DIR}/${ASSET_CONFIG_NAME}"
echo "DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH}"
echo "DIFFSYNTH_DOWNLOAD_SOURCE=${DIFFSYNTH_DOWNLOAD_SOURCE} HF_HOME=${HF_HOME}"
echo "PYTHON_BIN=${PYTHON_BIN} TORCHRUN_BIN=${TORCHRUN_BIN}"

exec "${TORCHRUN_BIN}" \
  --nproc_per_node="${NPROC_PER_NODE:-8}" \
  --nnodes="${WORLD_SIZE:-1}" \
  --node_rank="${RANK_ID}" \
  --master_addr="${MASTER_ADDR:-127.0.0.1}" \
  --master_port="${MASTER_PORT:-29500}" \
  scripts/train_fastwam.py "${args[@]}"
