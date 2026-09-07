#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIAG_LOG="${STAGE3_DLC_LOG:-${REPO_DIR}/stage3_dlc_smoke.log}"
exec > >(tee -a "${DIAG_LOG}") 2>&1
trap 'rc=$?; echo "FAILED rc=${rc} line=${LINENO} command=${BASH_COMMAND}"; exit "${rc}"' ERR

echo "===== Stage 3 DLC smoke $(date --iso-8601=seconds) ====="
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
export PARAMS_PATH="${PARAMS_PATH:-${REPO_DIR}/checkpoints/egoscale_stage2_aligned/stage2_aligned_relative_se3_baige_1x8_20260810_v1/20000/params}"
export CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${REPO_DIR}/checkpoints}"

test -f "${PARAMS_PATH}/manifest.ocdbt"
test -f "${PARAMS_PATH}/_METADATA"

: "${WANDB_API_KEY:?Inject WANDB_API_KEY through a DLC Secret environment variable}"
export WANDB_ENTITY="${WANDB_ENTITY:-your-wandb-team}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENABLED=1

export EXP_NAME="${EXP_NAME:-stage3_piper_from_stage2_20000_dlc_1x8_b512_smoke100_20260812_v4}"
export FSDP_DEVICES=4
export BATCH_SIZE=512
export VAL_BATCH_SIZE=96
export NUM_TRAIN_STEPS=100
export SAVE_INTERVAL=99
export EVAL_INTERVAL=50
export NUM_VAL_BATCHES=1
export NUM_ACTION_MSE_BATCHES=1
export RUN_ACTION_MSE=0
export SHUFFLE_BUFFER_SIZE=1024
export DATA_NUM_PARALLEL_READS=1
export DATA_NUM_PARALLEL_CALLS=2
export CHECKPOINT_PARAMS_ONLY=0
export RESUME=0
export OVERWRITE=0

echo "Launching ${EXP_NAME}"
exec bash scripts/run_egoscale_stage3_piper_dlc_full.sh
