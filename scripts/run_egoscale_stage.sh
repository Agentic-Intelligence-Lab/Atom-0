#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_DIR}/.venv/bin/python}"
STAGE="${STAGE:-stage1_ego}"
ATOM_RLDS_ROOT="${ATOM_RLDS_ROOT:-/mnt/data/RLDS}"
ASSETS_BASE_DIR="${ASSETS_BASE_DIR:-${REPO_DIR}/assets}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${REPO_DIR}/checkpoints}"
RESUME="${RESUME:-0}"
if [[ -z "${OVERWRITE+x}" ]]; then
  if [[ "${RESUME}" == "1" ]]; then
    OVERWRITE=0
  else
    OVERWRITE=1
  fi
fi

if [[ "${RESUME}" == "1" && "${OVERWRITE}" == "1" ]]; then
  echo "RESUME=1 and OVERWRITE=1 are mutually exclusive" >&2
  exit 2
fi

case "${STAGE}" in
  stage1_ego) CONFIG_NAME="egoscale_stage1_ego" ;;
  stage2_robot) CONFIG_NAME="egoscale_stage2_robot" ;;
  stage2_aligned) CONFIG_NAME="egoscale_stage2_aligned" ;;
  stage2_egomimic) CONFIG_NAME="egoscale_stage2_egomimic" ;;
  stage3_robot) CONFIG_NAME="egoscale_stage3_robot" ;;
  *) echo "Unknown STAGE=${STAGE}" >&2; exit 2 ;;
esac

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ "${STAGE}" == "stage1_ego" ]]; then
  : "${ATOM_PI05_BASE_PARAMS:?Set ATOM_PI05_BASE_PARAMS to the local pi05_base/params directory}"
else
  : "${PARAMS_PATH:?Set PARAMS_PATH to the previous-stage <step>/params directory}"
fi

export ATOM_RLDS_ROOT
export ATOM_PI05_BASE_PARAMS="${ATOM_PI05_BASE_PARAMS:-}"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/packages/openpi-client/src:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
export TF_FORCE_GPU_ALLOW_GROWTH=true

PREFLIGHT_ARGS=(
  --config-name "${CONFIG_NAME}"
  --assets-base-dir "${ASSETS_BASE_DIR}"
)
if [[ "${STAGE}" != "stage1_ego" ]]; then
  PREFLIGHT_ARGS+=(--params-path "${PARAMS_PATH}")
fi
"${PYTHON_BIN}" "${REPO_DIR}/scripts/check_egoscale_setup.py" "${PREFLIGHT_ARGS[@]}"

TRAIN_ARGS=(
  "${CONFIG_NAME}"
  --exp-name "${EXP_NAME:-${CONFIG_NAME}_smoke}"
  --assets-base-dir "${ASSETS_BASE_DIR}"
  --checkpoint-base-dir "${CHECKPOINT_BASE_DIR}"
  --fsdp-devices "${FSDP_DEVICES:-1}"
  --batch-size "${BATCH_SIZE:-2}"
  --num-train-steps "${NUM_TRAIN_STEPS:-20}"
  --log-interval "${LOG_INTERVAL:-1}"
  --save-interval "${SAVE_INTERVAL:-10}"
  --eval-interval "${EVAL_INTERVAL:-10}"
  --num-val-batches "${NUM_VAL_BATCHES:-1}"
  --num-action-mse-batches "${NUM_ACTION_MSE_BATCHES:-1}"
  --shuffle-buffer-size "${SHUFFLE_BUFFER_SIZE:-256}"
  --data-num-parallel-reads "${DATA_NUM_PARALLEL_READS:-1}"
  --data-num-parallel-calls "${DATA_NUM_PARALLEL_CALLS:-2}"
)
if [[ "${RESUME}" == "1" ]]; then
  TRAIN_ARGS+=(--resume)
elif [[ "${OVERWRITE}" == "1" ]]; then
  TRAIN_ARGS+=(--overwrite)
fi
if [[ "${STAGE}" != "stage1_ego" ]]; then
  TRAIN_ARGS+=(--weight-loader.params-path "${PARAMS_PATH}")
fi
if [[ "${WANDB_ENABLED:-0}" == "0" ]]; then
  TRAIN_ARGS+=(--no-wandb-enabled)
else
  : "${WANDB_API_KEY:?Set WANDB_API_KEY when WANDB_ENABLED=1}"
fi
if [[ "${CHECKPOINT_PARAMS_ONLY:-0}" == "1" ]]; then
  if [[ "${RESUME}" == "1" ]]; then
    echo "CHECKPOINT_PARAMS_ONLY=1 cannot be combined with RESUME=1" >&2
    exit 2
  fi
  TRAIN_ARGS+=(--checkpoint-params-only)
fi
if [[ "${RUN_ACTION_MSE:-0}" == "0" ]]; then
  TRAIN_ARGS+=(--no-run-action-mse --no-viz-action-traj --val-flow-loss-mode fixed_seed)
fi

cd "${REPO_DIR}"
echo "Launching ${CONFIG_NAME} on host $(hostname); do not wrap this JAX command in torchrun."
"${PYTHON_BIN}" -u scripts/train_cotrain.py "${TRAIN_ARGS[@]}" "$@"
