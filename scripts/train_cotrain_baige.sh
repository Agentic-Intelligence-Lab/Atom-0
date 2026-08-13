#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"
source scripts/atom0_env.sh

CONFIG_NAME="${CONFIG_NAME:?Set a supported co-training CONFIG_NAME}"
EXP_NAME="${EXP_NAME:?Set EXP_NAME}"
MODE="${MODE:-train}"
# Keep 64 samples/GPU by default.  WORLD_SIZE is the number of Baige nodes and
# NPROC_PER_NODE is 8 for the B200 jobs submitted by atom0_train_job.py.
BATCH_SIZE="${BATCH_SIZE:-$((64 * ${WORLD_SIZE:-1} * ${NPROC_PER_NODE:-8}))}"

case "${CONFIG_NAME}" in
  cotrain_real_only|cotrain_real_only_legacy32)
    # One aggregate pass over the current piper30+piper2 norm metadata frames.
    TRAIN_SAMPLES="${TRAIN_SAMPLES:-2757208}"
    DEFAULT_STEPS=$(((TRAIN_SAMPLES + BATCH_SIZE - 1) / BATCH_SIZE))
    DEFAULT_WARMUP=200
    DEFAULT_EVAL_INTERVAL=1000
    DEFAULT_SAVE_INTERVAL=2000
    DEFAULT_VAL_BATCH_SIZE=96
    DEFAULT_VAL_BATCHES=10
    DEFAULT_ACTION_MSE=1
    ;;
  cotrain_piper30_legacy32_aliyun_replay|cotrain_real_only_legacy32_aliyun_recipe|cotrain_real_only_unified80_aliyun_recipe|cotrain_real_only_sa_filtered)
    # Aliyun-recipe comparisons are explicitly run for 20k optimizer updates.
    # Sample counts are informative only; these comparison variants intentionally use
    # the same optimizer-step horizon and global batch.
    if [[ "${CONFIG_NAME}" == "cotrain_piper30_legacy32_aliyun_replay" ]]; then
      TRAIN_SAMPLES="${TRAIN_SAMPLES:-2067680}"
    elif [[ "${CONFIG_NAME}" == "cotrain_real_only_sa_filtered" ]]; then
      TRAIN_SAMPLES="${TRAIN_SAMPLES:-2640072}"
    else
      TRAIN_SAMPLES="${TRAIN_SAMPLES:-2757208}"
    fi
    DEFAULT_STEPS=20000
    DEFAULT_WARMUP=1000
    DEFAULT_EVAL_INTERVAL=1000
    DEFAULT_SAVE_INTERVAL=5000
    DEFAULT_VAL_BATCH_SIZE=96
    DEFAULT_VAL_BATCHES=10
    DEFAULT_ACTION_MSE=1
    ;;
  cotrain_real_robot|cotrain_real_robot_fix)
    # One aggregate pass over norm metadata frames. The audited fix mixture removes
    # Leju s54, Agilex fps50 and Agilex s26 (34 datasets, 149,953,766 frames).
    if [[ "${CONFIG_NAME}" == "cotrain_real_robot_fix" ]]; then
      TRAIN_SAMPLES="${TRAIN_SAMPLES:-149953766}"
    else
      TRAIN_SAMPLES="${TRAIN_SAMPLES:-164970758}"
    fi
    DEFAULT_STEPS=$(((TRAIN_SAMPLES + BATCH_SIZE - 1) / BATCH_SIZE))
    DEFAULT_WARMUP=5000
    DEFAULT_EVAL_INTERVAL=5000
    DEFAULT_SAVE_INTERVAL=10000
    DEFAULT_VAL_BATCH_SIZE=96
    DEFAULT_VAL_BATCHES=5
    DEFAULT_ACTION_MSE=0
    ;;
  cotrain_full_all_sa_filtered)
    # One aggregate pass over the 36 active State-Action-filtered datasets.
    TRAIN_SAMPLES="${TRAIN_SAMPLES:-243049603}"
    DEFAULT_STEPS=$(((TRAIN_SAMPLES + BATCH_SIZE - 1) / BATCH_SIZE))
    DEFAULT_WARMUP=5000
    DEFAULT_EVAL_INTERVAL=5000
    DEFAULT_SAVE_INTERVAL=10000
    DEFAULT_VAL_BATCH_SIZE=96
    DEFAULT_VAL_BATCHES=5
    DEFAULT_ACTION_MSE=0
    ;;
  *)
    echo "Unsupported CONFIG_NAME=${CONFIG_NAME}" >&2
    exit 2
    ;;
esac

FSDP_DEVICES="${FSDP_DEVICES:-4}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-${DEFAULT_STEPS}}"
WARMUP_STEPS="${WARMUP_STEPS:-${DEFAULT_WARMUP}}"
DECAY_STEPS="${DECAY_STEPS:-${NUM_TRAIN_STEPS}}"
EVAL_INTERVAL="${EVAL_INTERVAL:-${DEFAULT_EVAL_INTERVAL}}"
SAVE_INTERVAL="${SAVE_INTERVAL:-${DEFAULT_SAVE_INTERVAL}}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-${DEFAULT_VAL_BATCH_SIZE}}"
NUM_VAL_BATCHES="${NUM_VAL_BATCHES:-${DEFAULT_VAL_BATCHES}}"
RUN_ACTION_MSE="${RUN_ACTION_MSE:-${DEFAULT_ACTION_MSE}}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${REPO_DIR}/checkpoints}"
ASSETS_BASE_DIR="${ASSETS_BASE_DIR:-${REPO_DIR}/assets}"
ASSET_CONFIG_NAME="${CONFIG_NAME}"
if [[ "${CONFIG_NAME}" == "cotrain_real_only_legacy32" ||
      "${CONFIG_NAME}" == "cotrain_piper30_legacy32_aliyun_replay" ||
      "${CONFIG_NAME}" == "cotrain_real_only_legacy32_aliyun_recipe" ||
      "${CONFIG_NAME}" == "cotrain_real_only_unified80_aliyun_recipe" ]]; then
  # Comparison configs all reuse the audited production-1 Piper norm assets.
  # Legacy32 variants additionally project those stats back into native 14D order.
  ASSET_CONFIG_NAME="cotrain_real_only"
fi
if [[ "${CONFIG_NAME}" == "cotrain_real_only_sa_filtered" ]]; then
  ASSET_CONFIG_NAME="cotrain_full_all_sa_filtered"
fi
RANK_ID="${RANK:-0}"

if [[ "${CONFIG_NAME}" == "cotrain_full_all_sa_filtered" ||
      "${CONFIG_NAME}" == "cotrain_real_only_sa_filtered" ]]; then
  export SA_FILTERED_RLDS_DATA_DIR="${SA_FILTERED_RLDS_DATA_DIR:-/data/wudi/RLDS_SA_Filtered}"
  TRAIN_RLDS_DATA_DIR="${SA_FILTERED_RLDS_DATA_DIR}"
else
  TRAIN_RLDS_DATA_DIR="${RLDS_DATA_DIR}"
fi

# PARAMS_PATH controls model-weight initialization for a fresh EXP_NAME. Accept:
#   /data/models/paligemma/pt_224.npz
#   /data/models/openpi
#   checkpoints/<config>/<exp>/<step>
#   checkpoints/<config>/<exp>/<step>/params
# A PaliGemma NPZ initializes only the vision/language backbone; the action stack
# remains random. An Orbax path initializes all shape-compatible model weights.
# train_state/optimizer/step are deliberately ignored in both cases.
REQUESTED_PARAMS_PATH="${PARAMS_PATH}"
if [[ -f "${REQUESTED_PARAMS_PATH}" && "${REQUESTED_PARAMS_PATH}" == *.npz ]]; then
  PARAMS_PATH="${REQUESTED_PARAMS_PATH}"
  PARAMS_LAYOUT="paligemma-vlm-npz"
elif [[ -f "${REQUESTED_PARAMS_PATH}/_CHECKPOINT_METADATA" &&
      -f "${REQUESTED_PARAMS_PATH}/params/manifest.ocdbt" ]]; then
  PARAMS_PATH="${REQUESTED_PARAMS_PATH}/params"
  PARAMS_LAYOUT="training-step"
elif [[ -f "${REQUESTED_PARAMS_PATH}/manifest.ocdbt" &&
        -f "${REQUESTED_PARAMS_PATH}/_CHECKPOINT_METADATA" ]]; then
  PARAMS_PATH="${REQUESTED_PARAMS_PATH}"
  PARAMS_LAYOUT="released-params"
elif [[ -f "${REQUESTED_PARAMS_PATH}/manifest.ocdbt" &&
        -f "${REQUESTED_PARAMS_PATH}/../_CHECKPOINT_METADATA" ]]; then
  PARAMS_PATH="${REQUESTED_PARAMS_PATH}"
  PARAMS_LAYOUT="training-params"
else
  echo "Invalid PARAMS_PATH=${REQUESTED_PARAMS_PATH}" >&2
  echo "Expected a PaliGemma .npz, released params directory, training step, or its params/ child." >&2
  exit 2
fi
PARAMS_PATH="$(readlink -f -- "${PARAMS_PATH}")"
export PARAMS_PATH

if [[ "${MODE}" == "smoke" ]]; then
  NUM_TRAIN_STEPS="${SMOKE_STEPS:-20}"
  WARMUP_STEPS="${SMOKE_WARMUP_STEPS:-2}"
  DECAY_STEPS="${NUM_TRAIN_STEPS}"
  EVAL_INTERVAL=1000000
  SAVE_INTERVAL=1000000
  NUM_VAL_BATCHES=1
  RUN_ACTION_MSE=0
  LOG_INTERVAL=1
  WANDB_ENABLED=0
else
  WANDB_ENABLED="${WANDB_ENABLED:-1}"
fi

if (( WARMUP_STEPS < 0 || DECAY_STEPS <= WARMUP_STEPS )); then
  echo "Invalid LR schedule: require 0 <= WARMUP_STEPS < DECAY_STEPS, got ${WARMUP_STEPS} and ${DECAY_STEPS}" >&2
  exit 2
fi

GLOBAL_DEVICE_COUNT=$((${WORLD_SIZE:-1} * ${NPROC_PER_NODE:-8}))
if (( VAL_BATCH_SIZE <= 0 || VAL_BATCH_SIZE % GLOBAL_DEVICE_COUNT != 0 )); then
  echo "Invalid VAL_BATCH_SIZE=${VAL_BATCH_SIZE}: must be positive and divisible by ${GLOBAL_DEVICE_COUNT} devices" >&2
  exit 2
fi

if [[ "${PARAMS_LAYOUT}" == "paligemma-vlm-npz" ]]; then
  test -f "${PARAMS_PATH}"
else
  test -f "${PARAMS_PATH}/manifest.ocdbt"
fi
test -d "${TRAIN_RLDS_DATA_DIR}"
test -d "${ASSETS_BASE_DIR}/${ASSET_CONFIG_NAME}"

if [[ "${WANDB_ENABLED}" == "1" ]]; then
  : "${WANDB_API_KEY:?Set WANDB_API_KEY for production training}"
fi

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
export JAX_COORDINATOR_ADDRESS="${JAX_COORDINATOR_ADDRESS:-${MASTER_ADDR:-127.0.0.1}:${MASTER_PORT:-29500}}"

args=(
  "${CONFIG_NAME}"
  "--exp-name=${EXP_NAME}"
  "--fsdp-devices=${FSDP_DEVICES}"
  "--batch-size=${BATCH_SIZE}"
  "--num-train-steps=${NUM_TRAIN_STEPS}"
  "--lr-schedule.warmup-steps=${WARMUP_STEPS}"
  "--lr-schedule.decay-steps=${DECAY_STEPS}"
  "--eval-interval=${EVAL_INTERVAL}"
  "--val-batch-size=${VAL_BATCH_SIZE}"
  "--save-interval=${SAVE_INTERVAL}"
  "--log-interval=${LOG_INTERVAL}"
  "--num-val-batches=${NUM_VAL_BATCHES}"
  "--data-num-parallel-reads=${DATA_NUM_PARALLEL_READS:-1}"
  "--data-num-parallel-calls=${DATA_NUM_PARALLEL_CALLS:-2}"
  "--data.rlds-data-dir=${TRAIN_RLDS_DATA_DIR}"
  "--assets-base-dir=${ASSETS_BASE_DIR}"
  "--checkpoint-base-dir=${CHECKPOINT_BASE_DIR}"
  "--weight-loader.params-path=${PARAMS_PATH}"
)

if [[ "${WANDB_ENABLED}" == "1" ]]; then
  args+=("--wandb-enabled")
else
  args+=("--no-wandb-enabled")
fi
if [[ "${RUN_ACTION_MSE}" == "1" ]]; then
  args+=("--run-action-mse" "--viz-action-traj")
else
  args+=("--no-run-action-mse" "--no-viz-action-traj")
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  args+=("--overwrite")
fi

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/baige_${CONFIG_NAME}_${EXP_NAME}_rank${RANK_ID}.log") 2>&1
echo "CONFIG_NAME=${CONFIG_NAME} EXP_NAME=${EXP_NAME} MODE=${MODE}"
echo "WORLD_SIZE=${WORLD_SIZE:-1} RANK=${RANK_ID} MASTER=${JAX_COORDINATOR_ADDRESS}"
echo "INIT_PARAMS_PATH=${PARAMS_PATH} PARAMS_LAYOUT=${PARAMS_LAYOUT} (model weights only; optimizer/step reset for fresh EXP_NAME)"
if [[ "${PARAMS_LAYOUT}" == "paligemma-vlm-npz" ]]; then
  echo "INIT_POLICY=PaliGemma vision+language loaded; action expert, timestep MLP, and action projections random"
else
  echo "INIT_POLICY=all shape-compatible checkpoint model weights loaded"
fi
echo "FSDP_DEVICES=${FSDP_DEVICES} BATCH_SIZE=${BATCH_SIZE} VAL_BATCH_SIZE=${VAL_BATCH_SIZE} NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS}"

if [[ "${CONFIG_NAME}" == "cotrain_full_all_sa_filtered" ||
      "${CONFIG_NAME}" == "cotrain_real_only_sa_filtered" ]]; then
  .venv/bin/python scripts/preflight_cotrain_baige.py \
    "${CONFIG_NAME}" \
    --assets-base "${ASSETS_BASE_DIR}" \
    --params-path "${PARAMS_PATH}"
fi

exec .venv/bin/python -u scripts/train_cotrain.py "${args[@]}"
