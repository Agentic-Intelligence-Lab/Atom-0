#!/usr/bin/env bash
# Baige / torchrun entry for Atom-0 HPT cotrain (mirrors train_fastwam_baige.sh).
#
# Required:
#   CONFIG_NAME  hpt_cotrain_real_only | hpt_cotrain_real_robot_ego_fix |
#                hpt_cotrain_piper_ft | hpt_cotrain_piper_ft_full | hpt_cotrain_hhz_robot_ft
#   EXP_NAME
#
# Optional:
#   MODE=train|smoke  WANDB_ENABLED=1  WANDB_API_KEY=...
#   PRETRAINED_TRUNK_PATH=...  (warm-start shared trunk from liruiw/HPT trunk.pth)
#   HEAD_MODE=action_only|action_world  (default: action_world)
#   ACTION_HEAD_TYPE=dit|cross_transformer|mlp|diffusion|transformer_decoder  (default: transformer_decoder)
#   PYTORCH_WEIGHT_PATH=...  (required for finetune configs; recommended for piper_ft_full)
#   BATCH_SIZE / NUM_TRAIN_STEPS / EVAL_INTERVAL / ...
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"
source scripts/atom0_env.sh

CONFIG_NAME="${CONFIG_NAME:?Set a supported HPT CONFIG_NAME}"
EXP_NAME="${EXP_NAME:?Set EXP_NAME}"
MODE="${MODE:-train}"

case "${CONFIG_NAME}" in
  hpt_cotrain_real_only)
    # piper30+piper2; trunk warm-start (PRETRAINED_TRUNK_PATH); cosine 100k.
    DEFAULT_STEPS=100000
    DEFAULT_WARMUP=5000
    DEFAULT_BATCH_SIZE=128
    DEFAULT_LOG_INTERVAL=50
    DEFAULT_SAVE_INTERVAL=10000
    DEFAULT_EVAL_INTERVAL=1000
    DEFAULT_VAL_BATCH_SIZE=32
    DEFAULT_NUM_VAL_BATCHES=10
    DEFAULT_SHUFFLE_BUFFER=10000
    DEFAULT_PEAK_LR="1e-4"
    DEFAULT_DECAY_LR="1e-5"
    DEFAULT_ASSET_CONFIG_NAME="cotrain_real_only"
    DEFAULT_RLDS_PARTITION_BUILDERS=0
    DEFAULT_TRAIN_MODE="pretrain"
    ;;
  hpt_cotrain_real_robot_ego_fix)
    # large real+robot+ego; piper-only in-train eval (loss + action MSE).
    DEFAULT_STEPS=300000
    DEFAULT_WARMUP=15000
    DEFAULT_BATCH_SIZE=128
    DEFAULT_LOG_INTERVAL=50
    DEFAULT_SAVE_INTERVAL=30000
    DEFAULT_EVAL_INTERVAL=5000
    DEFAULT_VAL_BATCH_SIZE=32
    DEFAULT_NUM_VAL_BATCHES=10
    DEFAULT_NUM_ACTION_MSE_BATCHES=2
    DEFAULT_SHUFFLE_BUFFER=256
    DEFAULT_PEAK_LR="1e-4"
    DEFAULT_DECAY_LR="1e-5"
    DEFAULT_ASSET_CONFIG_NAME="cotrain_real_robot_ego_fix"
    DEFAULT_RLDS_PARTITION_BUILDERS=0
    DEFAULT_TRAIN_MODE="pretrain"
    ;;
  hpt_cotrain_piper_ft)
    # piper30+piper2 FT from mixture ckpt; trunk + world_head frozen; stems + action head train.
    DEFAULT_STEPS=20000
    DEFAULT_WARMUP=1000
    DEFAULT_BATCH_SIZE=128
    DEFAULT_LOG_INTERVAL=50
    DEFAULT_SAVE_INTERVAL=5000
    DEFAULT_EVAL_INTERVAL=1000
    DEFAULT_VAL_BATCH_SIZE=32
    DEFAULT_NUM_VAL_BATCHES=10
    DEFAULT_NUM_ACTION_MSE_BATCHES=2
    DEFAULT_SHUFFLE_BUFFER=10000
    DEFAULT_PEAK_LR="1.5e-4"
    DEFAULT_DECAY_LR="1e-6"
    DEFAULT_ASSET_CONFIG_NAME="cotrain_real_only"
    DEFAULT_RLDS_PARTITION_BUILDERS=0
    DEFAULT_TRAIN_MODE="finetune"
    ;;
  hpt_cotrain_piper_ft_full)
    # piper30+piper2 full FT (stem + trunk + heads); init from PYTORCH_WEIGHT_PATH.
    DEFAULT_STEPS=20000
    DEFAULT_WARMUP=1000
    DEFAULT_BATCH_SIZE=128
    DEFAULT_LOG_INTERVAL=50
    DEFAULT_SAVE_INTERVAL=5000
    DEFAULT_EVAL_INTERVAL=1000
    DEFAULT_VAL_BATCH_SIZE=32
    DEFAULT_NUM_VAL_BATCHES=10
    DEFAULT_NUM_ACTION_MSE_BATCHES=2
    DEFAULT_SHUFFLE_BUFFER=10000
    DEFAULT_PEAK_LR="1e-4"
    DEFAULT_DECAY_LR="1e-5"
    DEFAULT_ASSET_CONFIG_NAME="cotrain_real_only"
    DEFAULT_RLDS_PARTITION_BUILDERS=0
    DEFAULT_TRAIN_MODE="pretrain"
    ;;
  hpt_cotrain_hhz_robot_ft)
    # AtomAligned_full_front_cam v3 aligned_hangzhou_robot_right; trunk + world_head frozen.
    DEFAULT_STEPS=20000
    DEFAULT_WARMUP=1000
    DEFAULT_BATCH_SIZE=128
    DEFAULT_LOG_INTERVAL=50
    DEFAULT_SAVE_INTERVAL=5000
    DEFAULT_EVAL_INTERVAL=1000
    DEFAULT_VAL_BATCH_SIZE=32
    DEFAULT_NUM_VAL_BATCHES=10
    DEFAULT_NUM_ACTION_MSE_BATCHES=2
    DEFAULT_SHUFFLE_BUFFER=4096
    DEFAULT_PEAK_LR="1.5e-4"
    DEFAULT_DECAY_LR="1e-6"
    DEFAULT_ASSET_CONFIG_NAME="cotrain_real_robot_ego_fix"
    DEFAULT_RLDS_PARTITION_BUILDERS=0
    DEFAULT_TRAIN_MODE="finetune"
    ;;
  hpt_cotrain_smoke)
    DEFAULT_STEPS=2
    DEFAULT_WARMUP=0
    DEFAULT_BATCH_SIZE=2
    DEFAULT_LOG_INTERVAL=1
    DEFAULT_SAVE_INTERVAL=10
    DEFAULT_EVAL_INTERVAL=0
    DEFAULT_VAL_BATCH_SIZE=2
    DEFAULT_NUM_VAL_BATCHES=1
    DEFAULT_SHUFFLE_BUFFER=64
    DEFAULT_PEAK_LR=""
    DEFAULT_DECAY_LR=""
    DEFAULT_ASSET_CONFIG_NAME="cotrain_real_only"
    DEFAULT_RLDS_PARTITION_BUILDERS=0
    DEFAULT_TRAIN_MODE="pretrain"
    ;;
  *)
    echo "Unsupported CONFIG_NAME=${CONFIG_NAME}" >&2
    echo "Supported: hpt_cotrain_real_only | hpt_cotrain_real_robot_ego_fix | hpt_cotrain_piper_ft | hpt_cotrain_piper_ft_full | hpt_cotrain_hhz_robot_ft | hpt_cotrain_smoke" >&2
    exit 2
    ;;
esac

ASSET_CONFIG_NAME="${ASSET_CONFIG_NAME:-${DEFAULT_ASSET_CONFIG_NAME}}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${REPO_DIR}/checkpoints}"
ASSETS_BASE_DIR="${ASSETS_BASE_DIR:-${REPO_DIR}/assets}"
# HuggingFace cache for DINOv2 / T5 (prefer PFS so Baige nodes share downloads).
# Pre-seed on a networked host via hf-mirror, then keep Baige pods offline.
export HF_HOME="${HF_HOME:-${REPO_DIR}/../cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
# Do not set TRANSFORMERS_CACHE to ${HF_HOME}/transformers — hf download uses hub/.
# An empty transformers/ dir shadows HUGGINGFACE_HUB_CACHE and breaks offline load.
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
# If online download is ever needed on the login node: HF_ENDPOINT=https://hf-mirror.com
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

GLOBAL_DEVICE_COUNT=$((${WORLD_SIZE:-1} * ${NPROC_PER_NODE:-8}))
BATCH_SIZE="${BATCH_SIZE:-${DEFAULT_BATCH_SIZE}}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-${DEFAULT_STEPS}}"
LOG_INTERVAL="${LOG_INTERVAL:-${DEFAULT_LOG_INTERVAL}}"
SAVE_INTERVAL="${SAVE_INTERVAL:-${DEFAULT_SAVE_INTERVAL:-2000}}"
EVAL_INTERVAL="${EVAL_INTERVAL:-${DEFAULT_EVAL_INTERVAL:-0}}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-${DEFAULT_VAL_BATCH_SIZE:-32}}"
NUM_VAL_BATCHES="${NUM_VAL_BATCHES:-${DEFAULT_NUM_VAL_BATCHES:-10}}"
NUM_ACTION_MSE_BATCHES="${NUM_ACTION_MSE_BATCHES:-${DEFAULT_NUM_ACTION_MSE_BATCHES:-2}}"
RUN_ACTION_MSE="${RUN_ACTION_MSE:-1}"
SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-${DEFAULT_SHUFFLE_BUFFER}}"
RANK_ID="${RANK:-0}"

if (( BATCH_SIZE <= 0 || BATCH_SIZE % GLOBAL_DEVICE_COUNT != 0 )); then
  echo "Invalid BATCH_SIZE=${BATCH_SIZE}: must be positive and divisible by ${GLOBAL_DEVICE_COUNT} devices" >&2
  exit 2
fi

if [[ ! -d "${RLDS_DATA_DIR}" ]]; then
  echo "RLDS_DATA_DIR not found: ${RLDS_DATA_DIR}" >&2
  echo "Set RLDS_DATA_DIR to the BOS mount (e.g. /mnt/bos/bo23lu) before submitting." >&2
  exit 2
fi
if [[ ! -d "${ASSETS_BASE_DIR}/${ASSET_CONFIG_NAME}" ]]; then
  echo "Norm assets not found: ${ASSETS_BASE_DIR}/${ASSET_CONFIG_NAME}" >&2
  exit 2
fi
mkdir -p "${CHECKPOINT_BASE_DIR}" "${LOG_DIR}" "${HF_HOME}"

if [[ "${MODE}" == "smoke" ]]; then
  NUM_TRAIN_STEPS="${SMOKE_STEPS:-20}"
  LOG_INTERVAL=1
  SAVE_INTERVAL=1000000
  EVAL_INTERVAL=0
  WANDB_ENABLED=0
  SHUFFLE_BUFFER_SIZE="${SMOKE_SHUFFLE_BUFFER:-512}"
else
  WANDB_ENABLED="${WANDB_ENABLED:-1}"
fi

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
  "--data-num-parallel-calls=${DATA_NUM_PARALLEL_CALLS:-2}"
  "--data.rlds-data-dir=${RLDS_DATA_DIR}"
  "--assets-base-dir=${ASSETS_BASE_DIR}"
  "--checkpoint-base-dir=${CHECKPOINT_BASE_DIR}"
  "--lr-schedule.warmup-steps=${WARMUP_STEPS:-${DEFAULT_WARMUP}}"
  "--lr-schedule.decay-steps=${DECAY_STEPS:-${NUM_TRAIN_STEPS}}"
  "--model.train-mode=${TRAIN_MODE:-${DEFAULT_TRAIN_MODE}}"
)
if [[ -n "${HEAD_MODE:-}" ]]; then
  args+=("--model.head-mode=${HEAD_MODE}")
fi
if [[ -n "${ACTION_HEAD_TYPE:-}" ]]; then
  args+=("--model.action-head-type=${ACTION_HEAD_TYPE}")
fi
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
    "--num-action-mse-batches=${NUM_ACTION_MSE_BATCHES:-2}"
  )
  if [[ "${RUN_ACTION_MSE:-1}" == "1" ]]; then
    args+=("--run-action-mse")
  else
    args+=("--no-run-action-mse")
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

# Optional warm-start / true finetune: model.safetensors / step dir / exp dir.
if [[ -n "${PRETRAINED_TRUNK_PATH:-}" ]]; then
  TRUNK_PATH="${PRETRAINED_TRUNK_PATH}"
  if [[ "${TRUNK_PATH}" != /* ]]; then
    TRUNK_PATH="${REPO_DIR}/${TRUNK_PATH}"
  fi
  args+=("--model.pretrained-trunk-path=${TRUNK_PATH}")
fi

INIT_CKPT="${PYTORCH_WEIGHT_PATH:-${INIT_CHECKPOINT:-}}"
if [[ -n "${INIT_CKPT}" ]]; then
  if [[ "${INIT_CKPT}" != /* ]]; then
    INIT_CKPT="${REPO_DIR}/${INIT_CKPT}"
  fi
  args+=("--pytorch-weight-path=${INIT_CKPT}")
fi
# If user asks for finetune without a checkpoint, refuse (frozen random trunk is useless).
if [[ "${TRAIN_MODE:-${DEFAULT_TRAIN_MODE}}" == "finetune" && -z "${INIT_CKPT}" && "${MODE}" != "smoke" ]]; then
  echo "ERROR: TRAIN_MODE=finetune requires PYTORCH_WEIGHT_PATH or INIT_CHECKPOINT." >&2
  echo "       Without weights, keep default TRAIN_MODE=pretrain (random init, train all)." >&2
  exit 2
fi

if [[ "${RLDS_PARTITION_BUILDERS:-${DEFAULT_RLDS_PARTITION_BUILDERS:-0}}" == "0" ]]; then
  args+=("--no-rlds-partition-builders-by-rank")
elif [[ "${RLDS_PARTITION_BUILDERS}" == "1" ]]; then
  args+=("--rlds-partition-builders-by-rank")
fi

exec > >(tee -a "${LOG_DIR}/baige_${CONFIG_NAME}_${EXP_NAME}_rank${RANK_ID}.log") 2>&1
echo "CONFIG_NAME=${CONFIG_NAME} EXP_NAME=${EXP_NAME} MODE=${MODE}"
echo "train_mode=${TRAIN_MODE:-${DEFAULT_TRAIN_MODE}}"
echo "WORLD_SIZE=${WORLD_SIZE:-1} RANK=${RANK_ID} MASTER=${MASTER_ADDR:-127.0.0.1}:${MASTER_PORT:-29500}"
echo "BATCH_SIZE=${BATCH_SIZE} NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS} SHUFFLE_BUFFER_SIZE=${SHUFFLE_BUFFER_SIZE}"
echo "EVAL_INTERVAL=${EVAL_INTERVAL} (0=disabled) WANDB_ENABLED=${WANDB_ENABLED}"
echo "PRETRAINED_TRUNK_PATH=${PRETRAINED_TRUNK_PATH:-}"
echo "HEAD_MODE=${HEAD_MODE:-} ACTION_HEAD_TYPE=${ACTION_HEAD_TYPE:-dit}"
echo "INIT_CKPT=${INIT_CKPT:-} RESUME=${RESUME:-0}"
echo "RLDS_DATA_DIR=${RLDS_DATA_DIR} ASSETS=${ASSETS_BASE_DIR}/${ASSET_CONFIG_NAME}"
echo "HF_HOME=${HF_HOME}"
echo "PYTHON_BIN=${PYTHON_BIN} TORCHRUN_BIN=${TORCHRUN_BIN}"

exec "${TORCHRUN_BIN}" \
  --nproc_per_node="${NPROC_PER_NODE:-8}" \
  --nnodes="${WORLD_SIZE:-1}" \
  --node_rank="${RANK_ID}" \
  --master_addr="${MASTER_ADDR:-127.0.0.1}" \
  --master_port="${MASTER_PORT:-29500}" \
  scripts/train_hpt.py "${args[@]}"
