#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"
source scripts/atom0_env.sh

CONFIG_NAME="${CONFIG_NAME:?Set CONFIG_NAME to cotrain_real_only, cotrain_real_robot_fix, cotrain_real_robot_ego_fix, or cotrain_real_only_unified80_aliyun_recipe}"
EXP_NAME="${EXP_NAME:?Set EXP_NAME}"
MODE="${MODE:-train}"
# Keep 64 samples/GPU by default.  WORLD_SIZE is the number of Baige nodes and
# NPROC_PER_NODE is 8 for the B200 jobs submitted by atom0_train_job.py.
BATCH_SIZE="${BATCH_SIZE:-$((64 * ${WORLD_SIZE:-1} * ${NPROC_PER_NODE:-8}))}"

case "${CONFIG_NAME}" in
  cotrain_real_only)
    # One aggregate pass over the current piper30+piper2 norm metadata frames.
    TRAIN_SAMPLES="${TRAIN_SAMPLES:-2913191}"
    DEFAULT_STEPS=$(((TRAIN_SAMPLES + BATCH_SIZE - 1) / BATCH_SIZE))
    DEFAULT_WARMUP=200
    DEFAULT_EVAL_INTERVAL=1000
    DEFAULT_SAVE_INTERVAL=2000
    DEFAULT_VAL_BATCH_SIZE=96
    DEFAULT_VAL_BATCHES=10
    DEFAULT_ACTION_MSE=1
    ;;
  cotrain_real_only_unified80_aliyun_recipe)
  TRAIN_SAMPLES="${TRAIN_SAMPLES:-2913191}"
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
    # Leju s54, Agilex fps50 and Agilex s26 (34 datasets, 150,109,749 frames).
    if [[ "${CONFIG_NAME}" == "cotrain_real_robot_fix" ]]; then
      TRAIN_SAMPLES="${TRAIN_SAMPLES:-150109749}"
    else
      TRAIN_SAMPLES="${TRAIN_SAMPLES:-165126741}"
    fi
    DEFAULT_STEPS=$(((TRAIN_SAMPLES + BATCH_SIZE - 1) / BATCH_SIZE))
    DEFAULT_WARMUP=5000
    DEFAULT_EVAL_INTERVAL=5000
    DEFAULT_SAVE_INTERVAL=10000
    DEFAULT_VAL_BATCH_SIZE=96
    DEFAULT_VAL_BATCHES=5
    DEFAULT_ACTION_MSE=0
    ;;
  cotrain_real_robot_ego_fix)
    # Baige 1-epoch: real_robot_fix + 5 EgoVerse (≈ colleague cotrain_full_all_full_norm)
    TRAIN_SAMPLES="${TRAIN_SAMPLES:-266674679}"
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
RANK_ID="${RANK:-0}"
ASSET_CONFIG_NAME="${CONFIG_NAME}"
if [[ "${CONFIG_NAME}" == "cotrain_real_only_unified80_aliyun_recipe" ]]; then
  ASSET_CONFIG_NAME="cotrain_real_only"
fi

# # PARAMS_PATH: 支持 released params、训练 step 目录、或 step/params
# REQUESTED_PARAMS_PATH="${PARAMS_PATH}"
# if [[ -f "${REQUESTED_PARAMS_PATH}/_CHECKPOINT_METADATA" &&
#       -f "${REQUESTED_PARAMS_PATH}/params/manifest.ocdbt" ]]; then
#   PARAMS_PATH="${REQUESTED_PARAMS_PATH}/params"
#   PARAMS_LAYOUT="training-step"
# elif [[ -f "${REQUESTED_PARAMS_PATH}/manifest.ocdbt" &&
#         -f "${REQUESTED_PARAMS_PATH}/_CHECKPOINT_METADATA" ]]; then
#   PARAMS_PATH="${REQUESTED_PARAMS_PATH}"
#   PARAMS_LAYOUT="released-params"
# elif [[ -f "${REQUESTED_PARAMS_PATH}/manifest.ocdbt" &&
#         -f "${REQUESTED_PARAMS_PATH}/../_CHECKPOINT_METADATA" ]]; then
#   PARAMS_PATH="${REQUESTED_PARAMS_PATH}"
#   PARAMS_LAYOUT="training-params"
# else
#   echo "Invalid PARAMS_PATH=${REQUESTED_PARAMS_PATH}" >&2
#   echo "Expected a released params directory, a training step directory, or its params/ child." >&2
#   exit 2
# fi
# PARAMS_PATH="$(readlink -f -- "${PARAMS_PATH}")"
# export PARAMS_PATH

# PARAMS_PATH controls model-weight initialization for a fresh EXP_NAME. Accept:
#   /data/models/paligemma/pt_224.npz
#   /data/models/openpi
#   checkpoints/<config>/<exp>/<step>
#   checkpoints/<config>/<exp>/<step>/params
# A PaliGemma NPZ initializes only the vision/language backbone; the action stack
# remains random. An Orbax path initializes all shape-compatible model weights.
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

# test -f "${PARAMS_PATH}/manifest.ocdbt"
if [[ "${PARAMS_LAYOUT}" == "paligemma-vlm-npz" ]]; then
  test -f "${PARAMS_PATH}"
else
  test -f "${PARAMS_PATH}/manifest.ocdbt"
fi
test -d "${RLDS_DATA_DIR}"
test -d "${ASSETS_BASE_DIR}/${ASSET_CONFIG_NAME}"

if [[ "${WANDB_ENABLED}" == "1" ]]; then
  : "${WANDB_API_KEY:?Set WANDB_API_KEY for production training}"
fi

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
export JAX_COORDINATOR_ADDRESS="${JAX_COORDINATOR_ADDRESS:-${MASTER_ADDR:-127.0.0.1}:${MASTER_PORT:-29500}}"

# ---- Ablation mixture (read by CotrainDataConfig.create) ----
# Round1: MIX_MODE=fixed ALIGN_BY_DURATION=1  扫 P_OTHER∈{0.6,0.8,0.9}
# Round2: ALIGN_BY_DURATION=0  扫 LAB_IN_ALIGN∈{0.1,0.3,0.5}，P_OTHER 用 Round1 选定值
# Round3: 配比锁死，只改 --model.ot-alpha（或下面 OT_ALPHA）
export MIX_MODE="${MIX_MODE:-fixed}"
export P_OTHER="${P_OTHER:-0.8}"
export LAB_IN_ALIGN="${LAB_IN_ALIGN:-0.3}"
export O_EV_HUMAN="${O_EV_HUMAN:-0.02}"
export ALIGN_BY_DURATION="${ALIGN_BY_DURATION:-1}"
export OT_ALPHA="${OT_ALPHA:-0.7}"
export OT_ENABLED="${OT_ENABLED:-True}"

args=(
  "${CONFIG_NAME}"
  "--exp-name=${EXP_NAME}"
  "--fsdp-devices=${FSDP_DEVICES}"
  "--batch-size=${BATCH_SIZE}"
  "--val-batch-size=${VAL_BATCH_SIZE}"
  "--num-train-steps=${NUM_TRAIN_STEPS}"
  "--lr-schedule.warmup-steps=${WARMUP_STEPS}"
  "--lr-schedule.decay-steps=${DECAY_STEPS}"
  "--lr-schedule.peak-lr=${PEAK_LR:-1e-6}"
  "--lr-schedule.decay-lr=${DECAY_LR:-1e-7}"
  "--eval-interval=${EVAL_INTERVAL}"
  "--save-interval=${SAVE_INTERVAL}"
  "--log-interval=${LOG_INTERVAL}"
  "--num-val-batches=${NUM_VAL_BATCHES}"
  "--data-num-parallel-reads=${DATA_NUM_PARALLEL_READS:-1}"
  "--data-num-parallel-calls=${DATA_NUM_PARALLEL_CALLS:-2}"
  "--data.rlds-data-dir=${RLDS_DATA_DIR}"
  "--assets-base-dir=${ASSETS_BASE_DIR}"
  "--checkpoint-base-dir=${CHECKPOINT_BASE_DIR}"
  "--weight-loader.params-path=${PARAMS_PATH}"
  "--model.ot-alpha=${OT_ALPHA}"
)

if [[ "${WANDB_ENABLED}" == "1" ]]; then
  args+=("--wandb-enabled")
else
  args+=("--no-wandb-enabled")
fi
case "${OT_ENABLED,,}" in
  0|false|no|off) args+=("--model.no-ot-enabled") ;;
  *)              args+=("--model.ot-enabled") ;;
esac
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
echo "MIX_MODE=${MIX_MODE} P_OTHER=${P_OTHER} LAB_IN_ALIGN=${LAB_IN_ALIGN} O_EV_HUMAN=${O_EV_HUMAN} ALIGN_BY_DURATION=${ALIGN_BY_DURATION}"
echo "WORLD_SIZE=${WORLD_SIZE:-1} RANK=${RANK_ID} MASTER=${JAX_COORDINATOR_ADDRESS}"
echo "FSDP_DEVICES=${FSDP_DEVICES} BATCH_SIZE=${BATCH_SIZE} VAL_BATCH_SIZE=${VAL_BATCH_SIZE} NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS}"
echo "INIT_PARAMS_PATH=${PARAMS_PATH} PARAMS_LAYOUT=${PARAMS_LAYOUT} (model weights only; optimizer/step reset for fresh EXP_NAME)"
if [[ "${PARAMS_LAYOUT}" == "paligemma-vlm-npz" ]]; then
  echo "INIT_POLICY=PaliGemma vision+language loaded; action expert, timestep MLP, and action projections random"
else
  echo "INIT_POLICY=all shape-compatible checkpoint model weights loaded"
fi

exec .venv/bin/python -u scripts/train_cotrain.py "${args[@]}"
