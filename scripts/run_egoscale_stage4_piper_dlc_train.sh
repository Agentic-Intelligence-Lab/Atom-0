#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_LOG="${STAGE4_DLC_LOG:-${REPO_DIR}/stage4_dlc_train.log}"
mkdir -p "$(dirname "${TRAIN_LOG}")"
exec > >(tee -a "${TRAIN_LOG}") 2>&1
trap 'rc=$?; echo "FAILED rc=${rc} line=${LINENO} command=${BASH_COMMAND}"; exit "${rc}"' ERR

echo "===== Stage 4 DLC formal train $(date --iso-8601=seconds) ====="
cd "${REPO_DIR}"
echo "commit=$(git rev-parse HEAD)"
nvidia-smi -L

if [[ ! -x .venv/bin/python ]]; then
  echo "Venv missing; rebuilding it in the DLC container."
  bash scripts/setup_aliyun_dsw_env.sh
fi

.venv/bin/python - <<'PY'
import jax

devices = jax.devices()
print("JAX devices:", devices, flush=True)
if len(devices) != 8:
    raise RuntimeError(f"Expected 8 GPUs, got {len(devices)}: {devices}")
PY

export ATOM_RLDS_ROOT="${ATOM_RLDS_ROOT:-/mnt/data/RLDS}"
export CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${REPO_DIR}/checkpoints}"

# Explicit model-parameter entry for starting a new Stage 4 run. The legacy
# PARAMS_PATH variable remains accepted for compatibility, while
# MODEL_PARAMS_PATH takes precedence when both are provided.
DEFAULT_MODEL_PARAMS_PATH="${REPO_DIR}/checkpoints/egoscale_stage3_real_robot_fix/stage3_real_robot_fix_from_stage2_20000_b200_3x8_b1536_20260812_v1/97727/params"
export MODEL_PARAMS_PATH="${MODEL_PARAMS_PATH:-${PARAMS_PATH:-${DEFAULT_MODEL_PARAMS_PATH}}}"
export PARAMS_PATH="${MODEL_PARAMS_PATH}"

test -f "${PARAMS_PATH}/manifest.ocdbt"
test -f "${PARAMS_PATH}/_METADATA"

: "${WANDB_API_KEY:?Inject WANDB_API_KEY through a DLC Secret environment variable}"
export WANDB_ENTITY="${WANDB_ENTITY:-your-wandb-team}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENABLED=1

export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-40000}"
MODEL_CHECKPOINT_DIR="${MODEL_PARAMS_PATH%/params}"
MODEL_CHECKPOINT_STEP="${MODEL_CHECKPOINT_DIR##*/}"
export EXP_NAME="${EXP_NAME:-stage4_piper_finetune_from_stage3_${MODEL_CHECKPOINT_STEP}_dlc_1x8_b512_40k_20260818_v1}"
export FSDP_DEVICES=4
export BATCH_SIZE=512
export VAL_BATCH_SIZE=96
export SAVE_INTERVAL=5000
export EVAL_INTERVAL=1000
export NUM_VAL_BATCHES=10
export NUM_ACTION_MSE_BATCHES=2
export RUN_ACTION_MSE=1
export SHUFFLE_BUFFER_SIZE=50000
export DATA_NUM_PARALLEL_READS=1
export DATA_NUM_PARALLEL_CALLS=2
export CHECKPOINT_PARAMS_ONLY=0
export RESUME="${RESUME:-0}"
export OVERWRITE="${OVERWRITE:-0}"

EXPERIMENT_DIR="${CHECKPOINT_BASE_DIR}/egoscale_stage4_piper_finetune/${EXP_NAME}"
if [[ "${RESUME}" == "1" ]]; then
  if [[ "${OVERWRITE}" == "1" ]]; then
    echo "RESUME=1 and OVERWRITE=1 are mutually exclusive." >&2
    exit 2
  fi
  if [[ ! -d "${EXPERIMENT_DIR}" ]]; then
    echo "Cannot resume missing Stage 4 experiment directory: ${EXPERIMENT_DIR}" >&2
    exit 2
  fi
  if [[ ! -s "${EXPERIMENT_DIR}/wandb_id.txt" ]]; then
    echo "Cannot resume without W&B run id: ${EXPERIMENT_DIR}/wandb_id.txt" >&2
    exit 2
  fi
  shopt -s nullglob
  RESUME_MANIFESTS=("${EXPERIMENT_DIR}"/[0-9]*/train_state/manifest.ocdbt)
  shopt -u nullglob
  if (( ${#RESUME_MANIFESTS[@]} == 0 )); then
    echo "Cannot resume: no full train_state checkpoint found under ${EXPERIMENT_DIR}" >&2
    exit 2
  fi
  LATEST_RESUME_STEP=-1
  for manifest in "${RESUME_MANIFESTS[@]}"; do
    checkpoint_dir="${manifest%/train_state/manifest.ocdbt}"
    checkpoint_step="${checkpoint_dir##*/}"
    if [[ "${checkpoint_step}" =~ ^[0-9]+$ ]] && (( checkpoint_step > LATEST_RESUME_STEP )); then
      LATEST_RESUME_STEP="${checkpoint_step}"
    fi
  done
  COMPLETED_TRAIN_STEPS=$((LATEST_RESUME_STEP + 1))
  if [[ ! "${NUM_TRAIN_STEPS}" =~ ^[0-9]+$ ]] || (( NUM_TRAIN_STEPS <= COMPLETED_TRAIN_STEPS )); then
    echo "NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS} must exceed completed steps ${COMPLETED_TRAIN_STEPS}." >&2
    exit 2
  fi
  echo "Resuming ${EXP_NAME} from checkpoint ${LATEST_RESUME_STEP} to total step ${NUM_TRAIN_STEPS}."
elif [[ -e "${EXPERIMENT_DIR}" ]]; then
  echo "Refusing to reuse existing Stage 4 experiment directory: ${EXPERIMENT_DIR}" >&2
  exit 2
fi

echo "Launching ${EXP_NAME}; resume=${RESUME} total_steps=${NUM_TRAIN_STEPS} initial_params=${PARAMS_PATH}"
exec bash scripts/run_egoscale_stage4_piper_dlc_full.sh
