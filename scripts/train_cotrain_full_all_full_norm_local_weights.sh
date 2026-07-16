#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_ROOT="${ATOM0_STATE_ROOT:-$(dirname "${REPO_DIR}")}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${STATE_ROOT}/cache/openpi}"
PARAMS_PATH="${PARAMS_PATH:-${OPENPI_DATA_HOME}/openpi-assets/checkpoints/pi05_base/params}"
RLDS_DATA_DIR="${RLDS_DATA_DIR:-${STATE_ROOT}/data/RLDS}"
LOG_DIR="${LOG_DIR:-${REPO_DIR}}"
RANK_ID="${RANK:-0}"

if [[ ! -f "${PARAMS_PATH}/_CHECKPOINT_METADATA" || ! -f "${PARAMS_PATH}/manifest.ocdbt" ]]; then
  echo "Missing local pi05 params checkpoint at: ${PARAMS_PATH}" >&2
  exit 1
fi

: "${WANDB_API_KEY:?Please export WANDB_API_KEY before running this script.}"

cd "${REPO_DIR}"

export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/packages/openpi-client/src:${PYTHONPATH:-}"
export OPENPI_DATA_HOME
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

if [[ -z "${JAX_COORDINATOR_ADDRESS:-}" && -n "${MASTER_ADDR:-}" ]]; then
  export JAX_COORDINATOR_ADDRESS="${MASTER_ADDR}:29500"
fi

.venv/bin/python -u scripts/train_cotrain.py cotrain_full_all_full_norm \
  --exp_name="${EXP_NAME:-cotrain_full_all_full_norm_16gpus_local_weights}" \
  --fsdp_devices "${FSDP_DEVICES:-8}" \
  --batch-size "${BATCH_SIZE:-512}" \
  --num-train-steps "${NUM_TRAIN_STEPS:-3000000}" \
  --data-num-parallel-reads "${DATA_NUM_PARALLEL_READS:-1}" \
  --data-num-parallel-calls "${DATA_NUM_PARALLEL_CALLS:-2}" \
  --data.rlds-data-dir "${RLDS_DATA_DIR}" \
  --weight-loader.params-path "${PARAMS_PATH}" \
  --overwrite \
  2>&1 | tee "${LOG_DIR}/dlc_run_16gpu_local_weights_${RANK_ID}.log"
