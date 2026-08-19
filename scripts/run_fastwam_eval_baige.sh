#!/usr/bin/env bash
# Baige / local entry for scripts/run_fastwam_eval.py
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"
# shellcheck disable=SC1091
source scripts/atom0_env.sh

export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${REPO_DIR}/checkpoints/fastwam}"
export DIFFSYNTH_DOWNLOAD_SOURCE="${DIFFSYNTH_DOWNLOAD_SOURCE:-huggingface}"
export HF_HOME="${HF_HOME:-${DIFFSYNTH_MODEL_BASE_PATH}/hf_cache}"
export ASSETS_BASE_DIR="${ASSETS_BASE_DIR:-${REPO_DIR}/assets}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_DIR}/.venv/bin/python}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to a FastWAM step dir}"
CONFIG_NAME="${CONFIG_NAME:-wam-cross-piper-ft}"
DATASET="${DATASET:-piper30}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-576,512}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/tmp/fastwam_eval/$(basename "$(dirname "${CHECKPOINT}")")-$(basename "${CHECKPOINT}")-${DATASET}}"
DEVICE="${DEVICE:-cuda:0}"
NUM_VAL_BATCHES="${NUM_VAL_BATCHES:-20}"
NUM_ACTION_MSE_BATCHES="${NUM_ACTION_MSE_BATCHES:-5}"
RUN_ACTION_MSE="${RUN_ACTION_MSE:-1}"
VAL_FLOW_LOSS_MODE="${VAL_FLOW_LOSS_MODE:-fixed_seed}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"

MSE_FLAG=()
if [[ "${RUN_ACTION_MSE}" == "1" || "${RUN_ACTION_MSE}" == "true" ]]; then
  MSE_FLAG=(--run-action-mse)
else
  MSE_FLAG=(--no-run-action-mse)
fi

echo "CHECKPOINT=${CHECKPOINT}"
echo "CONFIG_NAME=${CONFIG_NAME} DATASET=${DATASET} IMAGE_RESOLUTION=${IMAGE_RESOLUTION}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

exec "${PYTHON_BIN}" scripts/run_fastwam_eval.py \
  --config-name "${CONFIG_NAME}" \
  --checkpoint "${CHECKPOINT}" \
  --dataset "${DATASET}" \
  --image-resolution "${IMAGE_RESOLUTION}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --assets-base-dir "${ASSETS_BASE_DIR}" \
  --num-val-batches "${NUM_VAL_BATCHES}" \
  --num-action-mse-batches "${NUM_ACTION_MSE_BATCHES}" \
  "${MSE_FLAG[@]}" \
  --val-flow-loss-mode "${VAL_FLOW_LOSS_MODE}" \
  --val-batch-size "${VAL_BATCH_SIZE}"
